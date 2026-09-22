"""Resolve an execution choice before tracing, then keep it fixed for the run."""
from __future__ import annotations


def _runtime_platform():
    import jax

    return jax.default_backend()


def _cuda_compute_capability():
    """Lowest NVIDIA compute capability among local training devices."""
    import jax

    devices = jax.local_devices(backend="gpu")
    majors = []
    for device in devices:
        capability = getattr(device, "compute_capability", None)
        if callable(capability):
            capability = capability()
        if capability is None:
            return None
        majors.append(int(str(capability).split(".")[0]))
    return min(majors) if majors else None


def resolve_attention_backend(model_config, *, dtype="float32", platform=None) -> str:
    """Return a concrete backend without changing scientific configuration.

    ``platform`` is a deterministic platform override for selection tests. Actual
    runs additionally inspect accelerator compatibility before choosing Triton.
    Explicit choices never fall back when they are unsupported.
    """
    request = model_config.attention_backend
    if request not in ("auto", "blocked", "fused", "manual", "xla", "cudnn"):
        raise ValueError(f"Unknown attention backend: {request}")
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("Attention requires float32 or bfloat16 computation")
    actual_platform = _runtime_platform() if platform is None else platform
    gpu = actual_platform in ("gpu", "cuda")
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
    resolved = resolve_attention_backend(config.model, dtype=config.train.dtype, platform=platform)
    if not hasattr(config, "_attention_backend_request"):
        config._attention_backend_request = request
    config.model.attention_backend = resolved
    return request
