"""Exact all-layer spectra of native FP32 Gated DeltaNet recurrent states.

The model keeps every recurrent and short-convolution cache. Only the recurrent
[head, value, key] matrices are measured. Set BLAS threads to one when using
multiple SVD workers; workers parallelize independent layers, not the recurrence.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import torch

from scaffolds.state_rank.spectrum import CACHE_RTOL, RANK_TOL, spectral_metrics, threshold_rank


METRICS = ("participation_ratio", "threshold_rank", "energy_rank90", "energy_rank99", "log_trace")


def _validate_layers(model, input_ids, layers):
    if model.training or input_ids.ndim != 2 or input_ids.shape[0] != 1 or not input_ids.shape[1]:
        raise ValueError("Require an eval model and one nonempty [1, tokens] input.")
    try:
        layers = tuple(layers)
    except TypeError as error:
        raise ValueError("Layers must be a nonempty iterable of integer indices.") from error
    if not layers or any(
        not isinstance(layer, (int, np.integer)) or isinstance(layer, (bool, np.bool_))
        or not 0 <= layer < len(model.layers) for layer in layers
    ):
        raise ValueError("Layers must be nonempty integer indices inside the model range.")
    if len(set(layers)) != len(layers):
        raise ValueError("Layer indices must be unique.")
    if any(layer.attn.mode != "fused_recurrent" for layer in model.layers):
        raise ValueError("Every model layer must use fused_recurrent attention.")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Native state analysis requires FP32 model parameters.")
    return tuple(int(layer) for layer in layers)


def _snapshot_states(cache, layers):
    tensors = []
    for layer in layers:
        state = cache[layer]["recurrent_state"]
        if state is None or state.dtype != torch.float32:
            raise RuntimeError(f"Layer {layer} must have a native FP32 recurrent state.")
        if state.ndim != 4 or state.shape[0] != 1 or min(state.shape[1:]) < 1:
            raise RuntimeError(f"Invalid recurrent state shape at layer {layer}: {tuple(state.shape)}")
        tensors.append(state[0])
    if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
        raise ValueError("Measured layers must have matching recurrent state shapes.")
    # Stacking copies even CPU states, so later in-place cache updates cannot
    # mutate the snapshot. One transfer also avoids a CUDA synchronization/layer.
    states = torch.stack(tensors).to(device="cpu", dtype=torch.float64).numpy()
    if not np.isfinite(states).all():
        raise FloatingPointError("Nonfinite Gated DeltaNet recurrent state.")
    return states


def iter_gdn_states_layers(model, input_ids, layers):
    """Yield independent FP64 [layer, head, value, key] snapshots once/token.

    One model forward updates all layers for each token. The complete native
    cache is passed unchanged into the next call, including convolution buffers.
    """
    layers = _validate_layers(model, input_ids, layers)
    device = next(model.parameters()).device
    cache = None
    with torch.inference_mode():
        for token in input_ids.to(device).split(1, dim=1):
            outputs = model(input_ids=token, past_key_values=cache, use_cache=True, return_dict=True)
            cache = outputs.past_key_values
            yield _snapshot_states(cache, layers)


def _singular_values(state):
    return np.linalg.svd(state, compute_uv=False)


def compute_gdn_spectra_layers(
    model, input_ids, layers, spectrum_prefixes, *, full_spectrum_layers=(), svd_workers=4,
):
    """Return compact spectral arrays and independent prefix-cache validation.

    Scalar metrics have axes [layer, token, head]; retained normalized spectra
    have axes [layer, selected prefix, head, mode]. Cache-check errors have axes
    [layer, selected prefix, head]. Coordinate vectors describe each axis.

    ``validation['full_spectra']`` optionally maps requested layer indices to
    normalized spectra [token, head, mode] for a caller's historical baseline
    comparison. These arrays are transient; remove them before JSON serialization.
    All other validation fields are JSON serializable. Raw states are not returned.
    """
    layers = _validate_layers(model, input_ids, layers)
    length = input_ids.shape[1]
    spectrum_prefixes = tuple(spectrum_prefixes)
    if (not spectrum_prefixes or any(
        not isinstance(t, (int, np.integer)) or isinstance(t, (bool, np.bool_))
        or not 1 <= t <= length for t in spectrum_prefixes
    ) or tuple(sorted(set(spectrum_prefixes))) != spectrum_prefixes):
        raise ValueError("Spectrum prefixes must be nonempty, sorted, distinct in-range integers.")
    full_spectrum_layers = tuple(full_spectrum_layers)
    if (len(set(full_spectrum_layers)) != len(full_spectrum_layers)
            or any(layer not in layers for layer in full_spectrum_layers)):
        raise ValueError("Full-spectrum layers must be unique measured layers.")
    if not isinstance(svd_workers, int) or isinstance(svd_workers, bool) or svd_workers < 1:
        raise ValueError("svd_workers must be a positive integer.")
    prefix_indices = {int(prefix): index for index, prefix in enumerate(spectrum_prefixes)}
    workers = min(svd_workers, len(layers))
    arrays, full_spectra = {}, {}
    device = next(model.parameters()).device
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)
    with pool as executor:
        for token_index, states in enumerate(iter_gdn_states_layers(model, input_ids, layers)):
            prefix = token_index + 1
            singular = np.stack(list(executor.map(_singular_values, states)) if executor else [
                _singular_values(state) for state in states
            ])
            eigenvalues = singular**2
            metrics = spectral_metrics(eigenvalues, np.zeros(singular.shape[:-1]))
            metrics["threshold_rank"] = threshold_rank(singular, RANK_TOL)
            if token_index == 0:
                layer_count, heads, width = singular.shape
                arrays = {
                    name: np.empty((layer_count, length, heads), dtype=metrics[name].dtype)
                    for name in METRICS
                }
                arrays["normalized_eigenvalues"] = np.empty(
                    (layer_count, len(spectrum_prefixes), heads, width), dtype=np.float64,
                )
                arrays["prefix_check_relative_error"] = np.empty(
                    (layer_count, len(spectrum_prefixes), heads), dtype=np.float64,
                )
                full_spectra = {
                    layer: np.empty((length, heads, width), dtype=np.float64)
                    for layer in full_spectrum_layers
                }
            for name in METRICS:
                arrays[name][:, token_index] = metrics[name]
            for layer, retained in full_spectra.items():
                retained[token_index] = metrics["normalized_eigenvalues"][layers.index(layer)]
            if prefix in prefix_indices:
                selected = prefix_indices[prefix]
                arrays["normalized_eigenvalues"][:, selected] = metrics["normalized_eigenvalues"]
                # A fresh cache makes this independent of tokenwise evolution.
                # The generator retains its own full cache while this pass runs.
                with torch.inference_mode():
                    outputs = model(input_ids=input_ids[:, :prefix].to(device),
                                    use_cache=True, return_dict=True)
                    actual = _snapshot_states(outputs.past_key_values, layers)
                errors = np.linalg.norm(actual - states, axis=(-2, -1)) / np.maximum(
                    np.linalg.norm(states, axis=(-2, -1)), np.finfo(np.float64).tiny,
                )
                arrays["prefix_check_relative_error"][:, selected] = errors
                if np.any(errors > CACHE_RTOL):
                    location = np.unravel_index(np.argmax(errors), errors.shape)
                    raise FloatingPointError(
                        f"Prefix {prefix} cache mismatch at layer {layers[location[0]]}, "
                        f"head {location[1]}: {errors[location]:.3g}."
                    )
                del actual, outputs
            if prefix % 64 == 0:
                print(f"  analyzed all {len(layers)} layers at prefix {prefix}", flush=True)
    arrays.update(
        layers=np.asarray(layers, dtype=np.int64),
        head_indices=np.arange(heads), prefixes=np.arange(1, length + 1),
        spectrum_prefixes=np.asarray(spectrum_prefixes, dtype=np.int64),
    )
    errors = arrays["prefix_check_relative_error"]
    return arrays, {
        "maximum_prefix_relative_error": float(errors.max()),
        "maximum_prefix_relative_error_by_layer": [float(value) for value in errors.max(axis=(1, 2))],
        "checked_prefixes": [int(t) for t in spectrum_prefixes],
        "cache_relative_tolerance": CACHE_RTOL,
        "svd_workers": workers,
        "svd_method": "numpy.linalg.svd(compute_uv=False), float64, direct recurrent matrix",
        "full_spectra": full_spectra,
    }
