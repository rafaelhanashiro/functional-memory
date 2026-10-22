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


def resolve_attention_backend(model_config, *, dtype="float32", platform=None, sequence_length=None) -> str:
    """Return a concrete backend without changing scientific configuration.

    ``platform`` is a deterministic platform override for selection tests. Actual
    runs additionally inspect accelerator compatibility before choosing Triton.
    Explicit choices never fall back when they are unsupported.
    """
    request = model_config.attention_backend
    if request not in ("auto", "blocked", "fused", "mosaic", "triton_scalar", "manual", "xla", "cudnn"):
        raise ValueError(f"Unknown attention backend: {request}")
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("Attention requires float32 or bfloat16 computation")
    if model_config.attention_basis is not None:
        if request == "triton_scalar":
            actual = _runtime_platform() if platform is None else platform
            capability = 9 if platform is not None else (_cuda_compute_capability() if actual in ("gpu", "cuda") else None)
            if actual not in ("gpu", "cuda") or capability != 9:
                raise ValueError("Scalar Triton attention currently requires H100/sm90")
            if model_config.qk_head_dim not in (16, 32, 64, 128) or model_config.value_head_dim not in (16, 32, 64, 128):
                raise ValueError("Scalar Triton head widths must be 16/32/64/128")
            return request
        if request not in ("auto", "blocked"):
            raise ValueError("Scalar-basis attention requires the blocked backend")
        return "blocked"
    actual_platform = _runtime_platform() if platform is None else platform
    gpu = actual_platform in ("gpu", "cuda")
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
    if request == "mosaic" or (request == "auto" and gpu and model_config.mosaic_supported()):
        # Mosaic GPU kernels exist for Hopper (sm90) and Blackwell (sm10x); the recipe-level
        # mechanism, width, and degree checks live in ModelConfig.mosaic_supported. Automatic
        # selection prefers them and otherwise falls back to the fused Triton kernels.
        if not gpu:
            raise ValueError("Mosaic attention requires a GPU")
        length = model_config.max_seq_len if sequence_length is None else sequence_length
        capability = 9 if platform is not None else _cuda_compute_capability()
        multiple = 128 if capability == 9 else 256
        problems = [message for failed, message in (
            (dtype != "bfloat16", "Mosaic attention requires train.dtype=bfloat16"),
            (capability not in (9, 10), f"Mosaic attention requires an H100 (sm90) or B300 (sm10x) GPU, got sm{capability}x"),
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
