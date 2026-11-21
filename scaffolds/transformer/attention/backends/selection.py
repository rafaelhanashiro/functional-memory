"""Resolve an execution choice before tracing, then keep it fixed for the run."""
from __future__ import annotations


def _runtime_platform():
    import jax

    return jax.default_backend()


def _cuda_compute_capabilities():
    """Exact NVIDIA capabilities of the local training devices, or None if unknown."""
    import jax

    devices = jax.local_devices(backend="gpu")
    capabilities = []
    for device in devices:
        capability = getattr(device, "compute_capability", None)
        if callable(capability):
            capability = capability()
        if capability is None:
            return None
        capabilities.append(str(capability))
    return tuple(capabilities) if capabilities else None


def _cuda_compute_capability():
    """Lowest NVIDIA compute-capability major among local training devices."""
    capabilities = _cuda_compute_capabilities()
    return min(int(capability.split(".")[0]) for capability in capabilities) if capabilities else None


def resolve_attention_backend(model_config, *, dtype="float32", platform=None, sequence_length=None) -> str:
    """Return a concrete backend without changing scientific configuration.

    ``platform`` is a deterministic platform override for selection tests. Actual
    runs additionally inspect accelerator compatibility before choosing Triton.
    Explicit choices never fall back when they are unsupported.
    """
    request = model_config.attention_backend
    if request not in ("auto", "blocked", "fused", "mosaic", "manual", "xla", "cudnn"):
        raise ValueError(f"Unknown attention backend: {request}")
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("Attention requires float32 or bfloat16 computation")
    score_bias = getattr(model_config, "attention_score_bias", "none") != "none"
    if score_bias and request not in ("mosaic", "blocked"):
        # Nothing may resolve them to a kernel that ignores the biases.
        raise ValueError("Per-degree score biases run on an explicit mosaic (H100) or blocked backend; "
                         f"{request!r} never selects them")
    from scaffolds.transformer.attention.pieces import ScoreFunction
    from scaffolds.transformer.configs.base import _exponential_kernel

    score_function = ScoreFunction(getattr(model_config, "attention_score_transform", "identity"),
                                   getattr(model_config, "attention_score_prefactor", "none"))
    exponential_functions = (not score_function.identity
                             and getattr(model_config, "attention_mechanism", None) == "dot_product"
                             and _exponential_kernel(model_config))
    # A polynomial's transform or prefactor, w = f(s) p(phi(s)): the Mosaic transformed kernels or the blocked backend.
    transformed = score_function.transform != "identity" or (score_function.prefactor != "none"
                                                             and not exponential_functions)
    if exponential_functions:
        # The exponential's score functions (attention/pieces.py): an explicit blocked backend, or the fused
        # kernel, whose declaration ModelConfig has checked; nothing selects them automatically.
        if request not in ("blocked", "fused"):
            raise ValueError(f"Score functions of the exponential run on an explicit blocked or fused backend; "
                             f"{request!r} never selects them")
    elif transformed and request not in ("blocked", "mosaic"):
        # Nor may a transformed or prefactored polynomial reach a kernel that evaluates the plain polynomial.
        raise ValueError(f"A score transform or polynomial prefactor runs on an explicit blocked or mosaic backend; "
                         f"{request!r} never selects it")
    actual_platform = _runtime_platform() if platform is None else platform
    gpu = actual_platform in ("gpu", "cuda")
    if getattr(model_config, "attention_coefficient_exp_gate", False):
        if request == "blocked":
            return request
        if request == "mosaic":
            from scaffolds.transformer.attention.backends.mosaic.api import (
                EVEN_EXP_COMPUTE_CAPABILITIES, VALUE_WIDTH, WIDTHS,
            )

            capabilities = EVEN_EXP_COMPUTE_CAPABILITIES[:1] if platform is not None else _cuda_compute_capabilities()
            length = model_config.max_seq_len if sequence_length is None else sequence_length
            if (gpu and dtype == "bfloat16" and capabilities
                    and all(item in EVEN_EXP_COMPUTE_CAPABILITIES for item in capabilities)
                    and model_config.qk_head_dim in WIDTHS and model_config.value_head_dim == VALUE_WIDTH
                    and length % 256 == 0):
                return request
            raise ValueError("Gated even powers plus exp on Mosaic require BF16 training on a qualified GPU "
                             f"(compute capability {EVEN_EXP_COMPUTE_CAPABILITIES}), Q/K width in {WIDTHS}, value "
                             f"width {VALUE_WIDTH} and a sequence length divisible by 256")
        capability = (8 if platform is not None else _cuda_compute_capability()) if gpu else None
        if (request == "fused" and capability is not None and capability >= 8
                and max(model_config.qk_head_dim, model_config.value_head_dim) <= 128):
            return request
        raise ValueError("Gated even powers plus exp require the blocked reference, the mosaic kernels or fused "
                         "CUDA kernels with sm80+ and head widths at most 128")
    if getattr(model_config, "positional_encoding", "rope") == "path":
        from scaffolds.transformer.attention.backends.fla_path_attention import (
            BLACKWELL_HEAD_DIMS, SEQUENCE_MULTIPLE,
        )

        if request not in ("auto", "blocked", "fused"):
            raise ValueError("PaTH attention requires backend='auto', 'blocked' or 'fused'")
        length = model_config.max_seq_len if sequence_length is None else sequence_length
        capability = (9 if platform is not None else _cuda_compute_capability()) if gpu else None
        # FLA's PaTH kernels are qualified here on Hopper (H100) and Blackwell (B300).
        # FLA's dq kernel fails to launch at width 128 on Blackwell.
        fits_device = capability == 9 or model_config.qk_head_dim in BLACKWELL_HEAD_DIMS
        supports = (capability is not None and capability >= 9 and length % SEQUENCE_MULTIPLE == 0
                    and dtype == "bfloat16" and model_config.path_fused_supported() and fits_device)
        if request == "auto":
            return "fused" if supports else "blocked"
        if request == "fused" and not supports:
            raise ValueError(f"Fused PaTH requires an sm90+ GPU, sequence length divisible by {SEQUENCE_MULTIPLE}, "
                             "train.dtype=bfloat16, and equal Q/K and value widths in 16, 32, 64, 128 "
                             "(up to 64 on Blackwell)")
        return request
    if model_config.attention_mechanism == "gated_deltanet":
        if request not in ("auto", "blocked", "fused"):
            raise ValueError("Gated DeltaNet requires backend='auto', 'blocked' or 'fused'")
        length = model_config.max_seq_len if sequence_length is None else sequence_length
        capability = (9 if platform is not None else _cuda_compute_capability()) if gpu else None
        polynomial = model_config.gdn_polynomial_degree > 1
        fits_width = not polynomial or max(model_config.qk_head_dim, model_config.value_head_dim) <= 64
        # FLA's kernels are qualified here on Hopper (H100) and Blackwell (B300).
        supports = capability is not None and capability >= 9
        if request == "auto":
            return "fused" if length % 64 == 0 and supports and fits_width else "blocked"
        if request == "fused" and not fits_width:
            raise ValueError("Fused polynomial GDN requires key and value head dimensions at most 64")
        if request == "fused" and (length % 64 or not supports):
            raise ValueError("Fused Gated DeltaNet requires sequence length divisible by 64 and an sm90+ GPU")
        return request
    if model_config.attention_mechanism == "gla":
        if request not in ("auto", "blocked", "fused"):
            raise ValueError("GLA requires backend='auto', 'blocked' or 'fused'")
        length = model_config.max_seq_len if sequence_length is None else sequence_length
        fits_gla = (length % 64 == 0 and model_config.gla_chunk_size == 64
                    and model_config.gla_inner_chunk_size == 16)
        capability = (8 if platform is not None else _cuda_compute_capability()) if gpu else None
        supports_gla = capability is not None and capability >= 8
        if request == "auto":
            return "fused" if fits_gla and supports_gla else "blocked"
        if request == "fused":
            if not fits_gla:
                raise ValueError("Fused GLA requires sequence length divisible by 64 and chunk sizes 64/16")
            if not supports_gla:
                raise ValueError("Fused GLA requires an Ampere-or-newer NVIDIA GPU")
        return request
    if getattr(model_config, "attention_score_mlp_width", None) is not None:
        if request not in ("auto", "blocked", "fused"):
            raise ValueError("A score MLP requires backend='auto', 'blocked' or 'fused'")
        # The fused Triton kernel has no forget gate: a gated score MLP always stays blocked.
        gated = getattr(model_config, "attention_forget_gate", False)
        fits = max(model_config.qk_head_dim, model_config.value_head_dim) <= 128
        capability = (8 if platform is not None else _cuda_compute_capability()) if gpu else None
        supports = capability is not None and capability >= 8
        if request == "auto":
            return "fused" if supports and fits and not gated else "blocked"
        if request == "fused":
            if gated:
                raise ValueError("A gated score MLP runs on the blocked backend: the fused kernel has no forget gate")
            if not fits:
                raise ValueError("Fused attention supports Q/K and value head dimensions up to 128")
            if not supports:
                raise ValueError("Fused score-MLP attention requires an Ampere-or-newer NVIDIA GPU")
        return request
    if request == "mosaic" or (request == "auto" and gpu and model_config.mosaic_supported()):
        # Mosaic GPU kernels exist for Hopper (sm90) and Blackwell (sm10x); the recipe-level
        # mechanism, width, and degree checks live in ModelConfig.mosaic_supported. Automatic
        # selection prefers them and otherwise falls back to the fused Triton kernels.
        if not gpu:
            raise ValueError("Mosaic attention requires a GPU")
        length = model_config.max_seq_len if sequence_length is None else sequence_length
        capability = 9 if platform is not None else _cuda_compute_capability()
        multiple = 128 if capability == 9 else 256
        transformed_device_supported = False
        if transformed:
            from scaffolds.transformer.attention.backends.mosaic.api import transformed_compute_capability_supported

            capabilities = ("9.0",) if platform is not None else _cuda_compute_capabilities()
            transformed_device_supported = bool(capabilities) and all(
                transformed_compute_capability_supported(item) for item in capabilities)
        problems = [message for failed, message in (
            (dtype != "bfloat16", "Mosaic attention requires train.dtype=bfloat16"),
            (capability not in (9, 10), f"Mosaic attention requires an H100 (sm90) or B300 (sm10x) GPU, got sm{capability}x"),
            (score_bias and capability != 9,
             f"Mosaic score-bias attention is qualified on H100 (sm90) only, got sm{capability}x"),
            # B200 and B300 share kernels, but only the former is admitted on Blackwell.
            (transformed and not transformed_device_supported,
             "Mosaic transformed query-polynomial attention admits H100 (sm90) and B200 (sm100) only"),
            (length % multiple, f"Mosaic attention on sm{capability}x requires sequence length divisible by {multiple}"),
        ) if failed]
        if not problems:
            return "mosaic"
        if request == "mosaic":
            raise ValueError(problems[0])
    polynomial = model_config.attention_degree is not None
    unnormalized = not model_config.attention_normalization
    custom_kernel = polynomial or unnormalized
    width = max(model_config.qk_head_dim, model_config.value_head_dim)
    fits_fused = width <= 128
    capability = (8 if platform is not None else _cuda_compute_capability()) if gpu else None
    supports_triton = capability is not None and capability >= 8
    if request == "auto":
        if not gpu:
            return "blocked"
        if (unnormalized and model_config.attention_coefficient_conditioning == "static"
                and model_config.attention_coefficient_parameterization == "sum_of_squares"):
            # Static SoS expands its factors through the qualified Mosaic query kernel;
            # no static-factor Triton entry point exists for the fallback allocation.
            return "blocked"
        if not custom_kernel:
            return "cudnn" if dtype == "bfloat16" and fits_fused and supports_triton else "xla"
        return "fused" if fits_fused and supports_triton else "blocked"
    if custom_kernel and request in ("manual", "xla", "cudnn"):
        raise ValueError(f"Backend {request!r} supports exact softmax only")
    if request == "fused" and custom_kernel:
        if not fits_fused:
            raise ValueError("Fused attention supports Q/K and value head dimensions up to 128")
        if not supports_triton:
            raise ValueError("Fused custom attention requires an Ampere-or-newer NVIDIA GPU")
    if request == "cudnn" or (request == "fused" and not custom_kernel):
        if dtype != "bfloat16":
            raise ValueError("Fused exact-softmax attention uses cuDNN and requires train.dtype=bfloat16")
        if not gpu:
            raise ValueError("cuDNN attention requires a GPU")
        if not supports_triton:
            raise ValueError("cuDNN attention requires an Ampere-or-newer NVIDIA GPU")
        max_width = 256 if capability >= 9 else 128
        if width > max_width:
            raise ValueError(f"cuDNN attention supports padded head dimensions up to {max_width} on this GPU")
    return request


def resolve_config_backend(config, *, platform=None) -> str:
    """Resolve in place and retain the user's request outside serialized configs."""
    request = config.model.attention_backend
    resolved = resolve_attention_backend(config.model, dtype=config.train.dtype, platform=platform,
                                         sequence_length=config.data.sequence_length - 1)
    if not hasattr(config, "_attention_backend_request"):
        config._attention_backend_request = request
    config.model.attention_backend = resolved
    return request
