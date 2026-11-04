"""A learned, input-dependent score function ("score MLP") for unnormalized dot-product attention.

The output is ``o_t = sum_(i<=t) f(x_t, s_ti) v_i``, never divided by a row sum, with
``s_ti = q_t . k_i / sqrt(d_K)`` after QK normalization and RoPE. Per layer and head, J hidden
units read the attention input ``x_t`` (after the block's pre-attention RMSNorm, as the
query-coefficient generators do):

    a_tj = alpha_j + x_t . A_j        b_tj = beta_j + x_t . B_j

    kernel:        f(x_t, s) = sum_j a_tj SiLU(w_j s + b_tj)
    exp_residual:  f(x_t, s) = exp(s) (1 + sum_j a_tj SiLU(w_j s + b_tj))

``A`` and ``B`` ([hidden, J] per head) start at zero, so ``f`` starts query-independent; the
amplitudes ``alpha``, offsets ``beta`` and slopes ``w`` ([J] per head) start at recipe constants.
Projections and kernel arithmetic are FP32; weights may be signed and exp is never row-shifted.
``exp_residual_initialization`` starts ``exp_residual`` exactly at exp, and
``fit_exp_initialization`` fits the ``kernel`` constants to exp on [-4, 4].
"""

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np

FORMS = ("kernel", "exp_residual")
ACTIVATIONS = ("silu",)
# Model-owned arrays: A and alpha give the unit amplitudes a, B and beta the offsets b, w the slopes.
AMPLITUDE_KERNEL, AMPLITUDE_BIAS = "score_mlp_amplitude_kernel", "score_mlp_amplitude_bias"
OFFSET_KERNEL, OFFSET_BIAS, SLOPE = "score_mlp_offset_kernel", "score_mlp_offset_bias", "score_mlp_slope"
PARAMETERS = (AMPLITUDE_KERNEL, AMPLITUDE_BIAS, OFFSET_KERNEL, OFFSET_BIAS, SLOPE)
KERNEL_NAMES = {"kernel": "score_mlp_unnormalized", "exp_residual": "exp_score_mlp_residual_unnormalized"}

# The exp fit: relative squared error on FIT_INTERVAL, with nonnegative amplitudes, slopes in
# FIT_SLOPES, and every unit's transition -beta/w (where w s + beta = 0) in FIT_TRANSITIONS.
FIT_INTERVAL = (-4.0, 4.0)
FIT_POINTS = 1601
FIT_SLOPES = (0.25, 4.0)
FIT_TRANSITIONS = (-8.0, 4.0)
ERROR_POINTS = 16001
# Each key chunk evaluates every unit on K / chunks keys, so a [B,H,Q,K/chunks,J] intermediate
# holds about this many [B,H,Q,K] score tiles whatever J is. Larger values trade memory for fewer
# loop iterations (and change only the FP32 order of the gradient sums).
UNITS_PER_KEY_CHUNK = 1


def _validate_width(width) -> int:
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("The score-MLP width must be a positive integer")
    return width


def exp_residual_initialization(width: int) -> dict[str, tuple[float, ...]]:
    """alpha = 0 (f starts exactly at exp), w = 1, beta evenly spaced on [-3, 3] (0 for one unit).

    With w = 1 unit j switches on at s = -beta_j, so the transitions cover the typical
    score range. At alpha = 0 only alpha and A receive gradient on the first update.
    """
    width = _validate_width(width)
    offsets = (0.0,) if width == 1 else tuple(float(value) for value in np.linspace(-3.0, 3.0, width))
    return {"amplitude": (0.0,) * width, "slope": (1.0,) * width, "offset": offsets}


@dataclass(frozen=True)
class ExpFit:
    """Shared constants with sum_j alpha_j SiLU(w_j s + beta_j) ~ exp(s), ordered by transition, then slope."""

    amplitude: tuple[float, ...]
    slope: tuple[float, ...]
    offset: tuple[float, ...]
    objective: float
    """Trapezoid integral over FIT_INTERVAL of (f(s)/exp(s) - 1)^2 on FIT_POINTS points."""
    max_relative_error: float
    """max |f(s)/exp(s) - 1| on ERROR_POINTS points of FIT_INTERVAL."""


def _sigmoid(values):
    return 0.5 * (1.0 + np.tanh(0.5 * values))


def _nnls(matrix, target, tolerance=1e-12):
    """Lawson-Hanson nonnegative least squares for the small dense amplitude problem."""
    count = matrix.shape[1]
    solution = np.zeros(count)
    passive = np.zeros(count, dtype=bool)
    for _ in range(3 * count + 3):
        gradient = np.where(passive, -np.inf, matrix.T @ (target - matrix @ solution))
        if passive.all() or gradient.max() <= tolerance:
            break
        passive[np.argmax(gradient)] = True
        while True:
            trial = np.zeros(count)
            trial[passive] = np.linalg.lstsq(matrix[:, passive], target, rcond=None)[0]
            if np.all(trial[passive] > 0):
                solution = trial
                break
            blocking = passive & (trial <= 0)
            step = np.min(solution[blocking] / (solution[blocking] - trial[blocking]))
            solution = solution + step * (trial - solution)
            passive &= solution > tolerance
            solution[~passive] = 0.0
    return solution


def _silu_sum(points, amplitude, slope, transition):
    pre = slope[None] * (points[:, None] - transition[None])
    return (pre * _sigmoid(pre)) @ amplitude


def fit_exp_initialization(width: int, *, restarts: int = 16, seed: int = 0, iterations: int = 3000) -> ExpFit:
    """Fit the ``kernel`` form's shared constants to exp by relative squared error on [-4, 4].

    Without constraints the problem has no minimizer: SiLU(z) ~ z e^z as z -> -inf, so units moved
    into their left tails reproduce exp with an error tending to zero while the amplitudes diverge
    with opposite signs (and the kernel is an exp in disguise). The fit therefore keeps
    alpha_j >= 0, w_j in FIT_SLOPES, and each transition t_j = -beta_j / w_j in FIT_TRANSITIONS:
    every unit switches on at or before the interval's right end, so the fitted kernel increases
    beyond it. Variable projection (alpha by nonnegative least squares for given w, t) with a
    projected Levenberg-Marquardt step and Kaufman's Jacobian, from one spread start (w = 1,
    transitions evenly spaced) and ``restarts - 1`` starts drawn from ``numpy.random.default_rng(seed)``,
    keeps the lowest objective. Deterministic for fixed arguments on a given numerical library
    (other BLAS kernels move the parameters by about 1e-5 relative). Units are returned by
    transition, then slope, so their order does not depend on the solver's internal unit order.
    """
    width = _validate_width(width)
    if isinstance(restarts, bool) or not isinstance(restarts, int) or restarts < 1:
        raise ValueError("The exp fit needs at least one start")
    low, high = FIT_INTERVAL
    points = np.linspace(low, high, FIT_POINTS)
    weights = np.full(FIT_POINTS, (high - low) / (FIT_POINTS - 1))
    weights[[0, -1]] *= 0.5
    root = np.sqrt(weights)
    scale = root * np.exp(-points)        # residual r = scale * f - root, so sum r^2 is the objective
    lower = np.concatenate([np.full(width, FIT_SLOPES[0]), np.full(width, FIT_TRANSITIONS[0])])
    upper = np.concatenate([np.full(width, FIT_SLOPES[1]), np.full(width, FIT_TRANSITIONS[1])])

    def evaluate(theta, jacobian=True):
        slope, transition = theta[:width], theta[width:]
        distance = points[:, None] - transition[None]
        pre = slope[None] * distance
        gate = _sigmoid(pre)
        design = scale[:, None] * pre * gate
        amplitude = _nnls(design, root)
        residual = design @ amplitude - root
        if not jacobian:
            return residual, amplitude
        derivative = scale[:, None] * gate * (1.0 + pre * (1.0 - gate))
        columns = np.concatenate([derivative * distance * amplitude, -derivative * slope * amplitude], axis=1)
        active = amplitude > 0
        if active.any():
            basis, _ = np.linalg.qr(design[:, active])
            columns = columns - basis @ (basis.T @ columns)
        return residual, amplitude, columns

    rng = np.random.default_rng(seed)
    best = None
    for start in range(restarts):
        if start == 0:
            slope = np.ones(width)
            transition = np.linspace(*FIT_TRANSITIONS, width + 2)[1:-1]
        else:
            slope = np.exp(rng.uniform(np.log(FIT_SLOPES[0]), np.log(FIT_SLOPES[1]), width))
            transition = np.sort(rng.uniform(*FIT_TRANSITIONS, width))
        theta = np.concatenate([slope, transition])
        residual, amplitude, columns = evaluate(theta)
        objective, damping = residual @ residual, 1e-2
        for _ in range(iterations):
            gradient = columns.T @ residual
            free = ~(((theta <= lower) & (gradient > 0)) | ((theta >= upper) & (gradient < 0)))
            curvature = columns[:, free].T @ columns[:, free]
            diagonal = np.diag(np.diag(curvature) + 1e-30)
            improved = False
            for _ in range(50):
                step = np.zeros_like(theta)
                try:
                    step[free] = np.linalg.solve(curvature + damping * diagonal, -gradient[free])
                except np.linalg.LinAlgError:
                    damping *= 4.0
                    continue
                candidate = np.clip(theta + step, lower, upper)
                trial_residual, _ = evaluate(candidate, jacobian=False)
                trial = trial_residual @ trial_residual
                if np.isfinite(trial) and trial < objective:
                    improved = True
                    break
                damping *= 4.0
            if not improved:
                break
            gain = (objective - trial) / objective
            theta, objective = candidate, trial
            residual, amplitude, columns = evaluate(theta)
            damping = max(damping / 4.0, 1e-16)
            if gain < 1e-13:
                break
        if best is None or objective < best[0]:
            best = (objective, theta.copy(), amplitude.copy())
    objective, theta, amplitude = best
    slope, transition = theta[:width], theta[width:]
    # Units held at a transition bound tie exactly; their slopes order them, whatever the solver's unit order.
    order = np.lexsort((slope, transition))
    amplitude, slope, transition = amplitude[order], slope[order], transition[order]
    fine = np.linspace(low, high, ERROR_POINTS)
    error = np.max(np.abs(_silu_sum(fine, amplitude, slope, transition) / np.exp(fine) - 1.0))
    return ExpFit(amplitude=tuple(float(value) for value in amplitude), slope=tuple(float(value) for value in slope),
                  offset=tuple(float(-w * t) for w, t in zip(slope, transition)),
                  objective=float(objective), max_relative_error=float(error))


# fit_exp_initialization(width) with its defaults, pinned so recipes copy identical frozen constants.
EXP_FITS = {
    4: ExpFit(
        amplitude=(0.24944715164677966, 3.4329674577333193, 26.118431089090134, 57.170507755491165),
        slope=(4.0, 0.5762735840768994, 0.5851374403404328, 0.9626833856704156),
        offset=(32.0, 1.6162684147176132, -1.0002536017877026, -3.8507335426816622),
        objective=0.0004769143149365164, max_relative_error=0.034655098841087195),
    8: ExpFit(
        amplitude=(0.21362532106199963, 0.2584342997204461, 1.3814369279804064, 4.774756710707291,
                   14.012385683197275, 68.4636106926777, 10.553476440221052, 1.2381941152837954),
        slope=(4.0, 1.0977649319133609, 0.7256759885928358, 0.6273367560330935, 0.6512751016700491,
               0.7995070609263598, 1.7363730396150405, 4.0),
        offset=(32.0, 5.415365208960629, 2.42904424673612, 0.6429831689026846, -0.91263929192096,
                -3.1980282437054393, -6.945492158460162, -16.0),
        objective=5.042488745579497e-09, max_relative_error=0.00030460056886949527),
}


def kernel_initialization(width: int) -> dict[str, tuple[float, ...]]:
    """The ``kernel`` form's constants: the pinned fit for J = 4 and 8, otherwise a fresh fit."""
    fit = EXP_FITS.get(_validate_width(width)) or fit_exp_initialization(width)
    return {"amplitude": fit.amplitude, "slope": fit.slope, "offset": fit.offset}


def initial_constants(width: int, form: str) -> dict[str, tuple[float, ...]]:
    """Recipe constants for ``attention_score_mlp_{amplitude,slope,offset}_initial``."""
    if form not in FORMS:
        raise ValueError(f"Unknown score-MLP form: {form}")
    return exp_residual_initialization(width) if form == "exp_residual" else kernel_initialization(width)


def parameter_specs(*, hidden_size: int, width: int, amplitude_initial, slope_initial, offset_initial,
                    kernel_weight_decay: bool = False):
    """A and B [hidden, J] per layer/head at zero; alpha, w, beta [J] per layer/head at the given constants.

    No random draws, so the backbone's initialization is unchanged. Only A and B may decay
    (``kernel_weight_decay``); alpha, beta, and w never do.
    """
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    width = _validate_width(width)
    values = {}
    for name, constants in (("amplitude", amplitude_initial), ("slope", slope_initial), ("offset", offset_initial)):
        if (not isinstance(constants, (tuple, list)) or len(constants) != width
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                       for value in constants)):
            raise ValueError(f"The score-MLP {name} constants must be {width} finite numbers")
        values[name] = tuple(float(value) for value in constants)
    if not isinstance(kernel_weight_decay, bool):
        raise ValueError("Score-MLP projection weight decay must be a boolean")

    def kernel(name):
        return AttentionParameterSpec(name=name, argument=name, initial=(0.0,) * (hidden_size * width),
                                      initial_shape=(hidden_size, width), sharing="per_head",
                                      weight_decay=kernel_weight_decay)

    def vector(name, initial):
        return AttentionParameterSpec(name=name, argument=name, initial=initial, sharing="per_head",
                                      weight_decay=False)

    return (kernel(AMPLITUDE_KERNEL), vector(AMPLITUDE_BIAS, values["amplitude"]),
            kernel(OFFSET_KERNEL), vector(OFFSET_BIAS, values["offset"]), vector(SLOPE, values["slope"]))


def project_units(x, kernel, bias):
    """FP32 [batch, time, heads, J] bias + x kernel, with the query-coefficient projection's arithmetic."""
    from scaffolds.transformer.attention.input_dependent import project_coefficients

    if kernel.ndim != 3 or bias.shape != (kernel.shape[0], kernel.shape[-1]):
        raise ValueError("Score-MLP projections require kernel [heads, hidden, J] and bias [heads, J]")
    return project_coefficients(x, kernel, bias, activation="identity")


def _key_chunks(scores, width, *others):
    """ceil(J / UNITS_PER_KEY_CHUNK) chunks of the key axis (at most K), padded with zeros."""
    keys = scores.shape[-1]
    chunks = min(-(-width // UNITS_PER_KEY_CHUNK), keys)
    size = -(-keys // chunks)
    padding = chunks * size - keys
    if padding:
        widths = ((0, 0), (0, 0), (0, 0), (0, padding))
        scores, others = jnp.pad(scores, widths), tuple(jnp.pad(array, widths) for array in others)
    return chunks, size, scores, others


def _unit_sum(scores, amplitude, offset, slope):
    """sum_j a_j SiLU(w_j s + b_j), every unit evaluated per key chunk inside a loop.

    Unrolling the units over whole tiles let XLA keep one tile per unit alive inside the
    rematerialized blocked scan; the loop bounds the live intermediates for any J.
    """
    keys, width = scores.shape[-1], amplitude.shape[-1]
    chunks, size, scores, _ = _key_chunks(scores, width)
    a, b, w = amplitude[..., None, :], offset[..., None, :], slope[None, :, None, None, :]

    def chunk(index, total):
        start = index * size
        pre = w * jax.lax.dynamic_slice_in_dim(scores, start, size, axis=3)[..., None] + b
        return jax.lax.dynamic_update_slice_in_dim(
            total, jnp.sum(a * (pre * jax.nn.sigmoid(pre)), axis=-1), start, axis=3)

    return jax.lax.fori_loop(0, chunks, chunk, jnp.zeros_like(scores))[..., :keys]


@jax.custom_vjp
def unit_sum(scores, amplitude, offset, slope):
    """FP32 sum_j a_j SiLU(w_j s + b_j) for scores [B,H,Q,K], a and b [B,H,Q,J] per query, w [H,J].

    The custom derivative keeps only these inputs. Forward and backward loop over min(J, K)
    key chunks (one unit per chunk, ``UNITS_PER_KEY_CHUNK``) and evaluate every unit inside a
    chunk, so their [B,H,Q,K/J,J] intermediates hold about one score tile whatever J is, and
    no chunk's intermediates survive the next.
    """
    return _unit_sum(scores, amplitude, offset, slope)


def _unit_sum_forward(scores, amplitude, offset, slope):
    return _unit_sum(scores, amplitude, offset, slope), (scores, amplitude, offset, slope)


def _unit_sum_backward(residuals, cotangent):
    scores, amplitude, offset, slope = residuals
    keys, width = scores.shape[-1], amplitude.shape[-1]
    # Padded keys have zero cotangent and contribute nothing.
    chunks, size, scores, (cotangent,) = _key_chunks(scores, width, cotangent)
    a, b, w = amplitude[..., None, :], offset[..., None, :], slope[None, :, None, None, :]

    def chunk(index, carry):
        dscores, damplitude, doffset, dslope = carry
        start = index * size
        s = jax.lax.dynamic_slice_in_dim(scores, start, size, axis=3)[..., None]
        g = jax.lax.dynamic_slice_in_dim(cotangent, start, size, axis=3)[..., None]
        pre = w * s + b
        gate = jax.nn.sigmoid(pre)
        dpre = g * a * (gate * (1.0 + pre * (1.0 - gate)))   # dL/d(w s + b), with SiLU' = gate (1 + z (1 - gate))
        dscores = jax.lax.dynamic_update_slice_in_dim(dscores, jnp.sum(dpre * w, axis=-1), start, axis=3)
        return (dscores, damplitude + jnp.sum(g * (pre * gate), axis=3), doffset + jnp.sum(dpre, axis=3),
                dslope + jnp.sum(dpre * s, axis=3))

    initial = (jnp.zeros_like(scores), jnp.zeros_like(amplitude), jnp.zeros_like(offset), jnp.zeros_like(amplitude))
    dscores, damplitude, doffset, dslope = jax.lax.fori_loop(0, chunks, chunk, initial)
    return dscores[..., :keys], damplitude, doffset, jnp.sum(dslope, axis=(0, 2))


unit_sum.defvjp(_unit_sum_forward, _unit_sum_backward)


def score_kernel(scores, amplitude, offset, slope, form):
    """f on a tile: the unit sum, or exp(s) (1 + unit sum); scores are FP32 and unshifted."""
    if form not in FORMS:
        raise ValueError(f"Unknown score-MLP form: {form}")
    units = unit_sum(scores, amplitude, offset, slope)
    return units if form == "kernel" else jnp.exp(scores) * (1.0 + units)


def unit_statistics(amplitude, offset):
    """Actual unit amplitudes a and offsets b on the diagnostic sample, per head and unit ([H, J])."""
    statistics = {}
    for name, values in (("amplitude", amplitude), ("offset", offset)):
        values = values.astype(jnp.float32)
        statistics.update({
            f"score_mlp_{name}_mean": jnp.mean(values, axis=(0, 1)),
            f"score_mlp_{name}_std": jnp.std(values, axis=(0, 1)),
            f"score_mlp_{name}_min": jnp.min(values, axis=(0, 1)),
            f"score_mlp_{name}_max": jnp.max(values, axis=(0, 1)),
        })
    return statistics


def score_mlp_attention(q, k, v, amplitude, offset, slope, *, form="kernel", log_decay=None, backend="blocked",
                        return_diagnostics=False):
    """sum_(i<=t) f(x_t, s_ti) v_i on the blocked reference, from post-RoPE Q/K/V [B,T,H,C].

    ``amplitude`` and ``offset`` are the projected a and b [B,T,H,J]; ``slope`` is w [H,J].
    ``log_decay`` is an optional cumulative FP32 forget-gate sum L [B,T,H]: each weight is then
    multiplied by exp(L_t - L_i), as in the query-coefficient blocked branch.
    """
    from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention

    if backend != "blocked":
        raise ValueError("Score-MLP attention runs on the blocked reference backend only")
    result = blocked_dot_product_attention(
        q, k, v, attention_normalization=False, score_mlp_amplitude=amplitude, score_mlp_offset=offset,
        score_mlp_slope=slope, score_mlp_form=form, query_log_decay=log_decay, return_diagnostics=return_diagnostics)
    if not return_diagnostics:
        return result
    output, diagnostics = result
    diagnostics.update(jax.lax.stop_gradient(unit_statistics(amplitude, offset)))
    return output, diagnostics
