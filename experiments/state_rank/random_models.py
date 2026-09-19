"""Reproducible random Llama controls and immutable saved diagnostic inputs."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModel, PretrainedConfig


_HASH_CHUNK_BYTES = 1024 * 1024


def build_random_llama(config: PretrainedConfig, seed: int) -> torch.nn.Module:
    """Initialize an entire Llama using Hugging Face's reference initializer.

    Initialization, including all Linear/Embedding draws, takes place on the CPU
    in FP32. The returned model is in eval mode. No pretrained weights are loaded,
    no layers/heads are reused, and caller configuration, CPU RNG state, default
    dtype, and default device are preserved. CPU-only seeding also leaves CUDA
    generators untouched, even when the caller has already initialized CUDA.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be a nonnegative integer smaller than 2**64")
    if getattr(config, "model_type", None) != "llama":
        raise ValueError("build_random_llama requires a Llama configuration")

    random_config = deepcopy(config)
    random_config._attn_implementation = "eager"
    random_config.dtype = torch.float32
    previous_dtype = torch.get_default_dtype()
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.set_default_dtype(torch.float32)
            torch.random.default_generator.manual_seed(seed)
            model = AutoModel.from_config(
                random_config, dtype=torch.float32, attn_implementation="eager",
            )
    finally:
        torch.set_default_dtype(previous_dtype)

    if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
           for parameter in model.parameters()):
        raise RuntimeError("Random model initialization must produce CPU FP32 parameters")
    return model.eval()



def build_random_gated_deltanet(config: PretrainedConfig, seed: int) -> torch.nn.Module:
    """Build the pinned FLA Gated DeltaNet reference initialization on CPU.

    FLA's normal ``post_init`` path initializes Linear, Conv1d, and Embedding
    weights with ``config.initializer_range`` and initializes normalization
    scales to one. Its special decay/time-step parameters keep their reference
    distributions: exp(A_log) is uniform on [0, 16), and softplus(dt_bias) is
    log-uniform on [0.001, 0.1]. Do not overwrite these with generic normal draws
    or apply an additional initialization pass. Historical unused D parameters
    are absent from the pinned implementation.

    Parameters are drawn directly in FP32 on CPU, under an isolated CPU RNG.
    The caller's configuration and defaults are preserved; the returned eval
    model selects native fused_recurrent inference (which requires a GPU).
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be a nonnegative integer smaller than 2**64")
    if getattr(config, "model_type", None) != "gated_deltanet":
        raise ValueError("build_random_gated_deltanet requires a GatedDeltaNet configuration")

    random_config = deepcopy(config)
    random_config.attn_mode = "fused_recurrent"
    random_config.dtype = torch.float32
    previous_dtype = torch.get_default_dtype()
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.set_default_dtype(torch.float32)
            # Register the installed, revision-pinned FLA class with AutoModel.
            # Keep this lazy so Llama-only callers do not import GPU kernel code.
            import fla.models.gated_deltanet  # noqa: F401

            torch.random.default_generator.manual_seed(seed)
            model = AutoModel.from_config(random_config, dtype=torch.float32)
    finally:
        torch.set_default_dtype(previous_dtype)

    if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
           for parameter in model.parameters()):
        raise RuntimeError("Random model initialization must produce CPU FP32 parameters")
    return model.eval()


def parameter_fingerprint(model: torch.nn.Module) -> str:
    """SHA256 of parameter registration order, names, shapes, dtypes, and bytes.

    Values are transferred and hashed in bounded chunks, rather than converting
    an entire model or a large embedding table to one Python byte string. CPU
    contiguous parameters, as produced by ``build_random_llama``, need no tensor
    copy. The digest excludes non-parameter buffers and model configuration.
    """
    digest = hashlib.sha256(b"functional-memory-parameter-fingerprint-v1\0")
    for index, (name, parameter) in enumerate(model.named_parameters(remove_duplicate=False)):
        descriptor = json.dumps(
            [index, name, list(parameter.shape), str(parameter.dtype)],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update((parameter.numel() * parameter.element_size()).to_bytes(8, "little"))
        values = parameter.detach().reshape(-1)
        elements_per_chunk = max(1, _HASH_CHUNK_BYTES // parameter.element_size())
        for start in range(0, values.numel(), elements_per_chunk):
            chunk = values[start:start + elements_per_chunk].to(device="cpu").contiguous()
            digest.update(memoryview(chunk.view(torch.uint8).numpy()))
    return digest.hexdigest()


def load_saved_inputs(
    path: str | Path,
    expected_tokens: int,
    vocab_size: int,
    bos_token_id: int,
    expected_count: int | None = 4,
) -> list[dict]:
    """Read and validate the exact saved token IDs without retokenizing text.

    Records are returned unchanged, including titles, text, article indices, and
    body hashes. ``expected_count=None`` accepts any nonempty list of records.
    """
    for name, value, minimum in (
        ("expected_tokens", expected_tokens, 2),
        ("vocab_size", vocab_size, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if (isinstance(bos_token_id, bool) or not isinstance(bos_token_id, int)
            or not 0 <= bos_token_id < vocab_size):
        raise ValueError("bos_token_id must be an integer within the vocabulary")
    if expected_count is not None and (
        isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1
    ):
        raise ValueError("expected_count must be a positive integer or None")

    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Saved inputs must be a nonempty list of records")
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} saved inputs, found {len(records)}")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Input {index} must be a record mapping")
        token_ids = record.get("token_ids")
        if not isinstance(token_ids, list) or len(token_ids) != expected_tokens:
            raise ValueError(f"Input {index} must contain exactly {expected_tokens} token_ids")
        if any(isinstance(token, bool) or not isinstance(token, int) for token in token_ids):
            raise ValueError(f"Input {index} token_ids must be a flat list of integers")
        if any(not 0 <= token < vocab_size for token in token_ids):
            raise ValueError(f"Input {index} token_ids must be within [0, {vocab_size})")
        if token_ids[0] != bos_token_id:
            raise ValueError(f"Input {index} must start with BOS token {bos_token_id}")
    return records
