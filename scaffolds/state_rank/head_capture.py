"""Frozen Llama head capture and reproducible, document-disjoint token corpora.

Capture uses the model's own projections, rotary embeddings and pre-output-
projection activations. It does not alter its Q/K normalization or pretrained
weights. Full-model execution belongs in an allocated compute job.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import zipfile

import numpy as np
import torch

from .spectrum import (
    DATASET_CONFIG, DATASET_ID, DATASET_REVISION, MODEL_ID, MODEL_REVISION,
    iter_article_bodies,
)


DEFAULT_HEADS = tuple((layer, head) for layer in (2, 8, 14) for head in (0, 16))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _token_digest(ids: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def select_document_windows(
    tokenizer,
    rows_by_split: Mapping[str, Iterable[str]],
    counts: Mapping[str, int],
    *,
    tokens: int,
    seed: int,
    long_test_windows: int = 0,
    long_tokens: int = 2048,
) -> tuple[dict[str, np.ndarray], dict[str, list[dict]]]:
    """Select one fixed window per article from the official dataset splits.

    Articles shared between splits by normalized title or exact body are
    discarded from *every* split before selection. Selection order is a hash of
    seed, official split and body, independent of dictionary iteration order.
    Long evaluation windows may reuse test articles, never training articles.
    Every window starts at the article body with BOS and has no padding.
    """
    if tokens < 2 or tokenizer.bos_token_id is None:
        raise ValueError("Require at least two tokens and a tokenizer BOS token.")
    if set(counts) != {"train", "validation", "test"}:
        raise ValueError("counts must specify train, validation and test.")
    if any(not isinstance(count, int) or count < 1 for count in counts.values()):
        raise ValueError("Each split must request a positive integer window count.")
    if long_test_windows < 0 or (long_test_windows and long_tokens < tokens):
        raise ValueError("Long test windows must be nonnegative and at least tokens long.")

    articles = {}
    identity_splits: dict[tuple[str, str], set[str]] = {}
    for split in ("train", "validation", "test"):
        articles[split] = []
        for index, article in enumerate(iter_article_bodies(rows_by_split[split])):
            body_hash = hashlib.sha256(article["text"].encode()).hexdigest()
            title_identity = " ".join(article["title"].casefold().split())
            entry = {**article, "article_index": index, "body_sha256": body_hash,
                     "title_identity": title_identity}
            articles[split].append(entry)
            for identity in (("title", title_identity), ("body", body_hash)):
                identity_splits.setdefault(identity, set()).add(split)

    arrays, selected = {}, {}
    for split in ("train", "validation", "test"):
        ordered = sorted(articles[split], key=lambda a: _json_digest([seed, split, a["body_sha256"]]))
        seen_titles, seen_bodies = set(), set()
        eligible = []
        for article in ordered:
            title, body = article["title_identity"], article["body_sha256"]
            if (len(identity_splits["title", title]) != 1
                    or len(identity_splits["body", body]) != 1
                    or title in seen_titles or body in seen_bodies):
                continue
            seen_titles.add(title)
            seen_bodies.add(body)
            eligible.append(article)

        requests = [(split, counts[split], tokens)]
        if split == "test" and long_test_windows:
            requests.append(("test_long", long_test_windows, long_tokens))
        for name, count, length in requests:
            windows, descriptions = [], []
            for article in eligible:
                ids = tokenizer.encode(article["text"], add_special_tokens=False,
                                       truncation=True, max_length=length - 1)
                if len(ids) < length - 1:
                    continue
                ids = [tokenizer.bos_token_id, *ids]
                windows.append(ids)
                descriptions.append({
                    "source_split": split,
                    "article_index": article["article_index"],
                    "title": article["title"],
                    "body_sha256": article["body_sha256"],
                    "token_ids_sha256": _token_digest(ids),
                    "tokens": length,
                    "body_token_offset": 0,
                })
                if len(windows) == count:
                    break
            if len(windows) != count:
                raise ValueError(f"{name}: requested {count} articles with {length - 1} text tokens; "
                                 f"found {len(windows)} after document-disjoint filtering.")
            arrays[name] = np.asarray(windows, dtype=np.int64)
            selected[name] = descriptions
    return arrays, selected


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """NPZ without wall-clock ZIP metadata, for byte-identical regeneration."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in sorted(arrays.items()):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue())


def prepare_wikitext_corpus(
    destination: str | Path,
    *,
    train_windows: int = 200,
    validation_windows: int = 32,
    test_windows: int = 32,
    tokens: int = 512,
    long_test_windows: int = 8,
    long_tokens: int = 2048,
    seed: int = 20261015,
    local_files_only: bool = True,
) -> dict:
    """Write pinned token data outside the repository; return its provenance.

    Existing files must match the requested corpus exactly. Downloading is opt-in;
    normal experiment jobs should only load and verify the prepared archive.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                              local_files_only=local_files_only)
    rows, sources = {}, {}
    for split in ("train", "validation", "test"):
        filename = f"{DATASET_CONFIG}/{split}-00000-of-00001.parquet"
        path = Path(hf_hub_download(DATASET_ID, filename, revision=DATASET_REVISION,
                                   repo_type="dataset", local_files_only=local_files_only))
        rows[split] = pq.read_table(path)["text"].to_pylist()
        sources[split] = {"filename": filename, "path": str(path),
                          "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    counts = {"train": train_windows, "validation": validation_windows, "test": test_windows}
    arrays, selected = select_document_windows(
        tokenizer, rows, counts, tokens=tokens, seed=seed,
        long_test_windows=long_test_windows, long_tokens=long_tokens,
    )
    recipe = {"dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
              "dataset_config": DATASET_CONFIG, "tokenizer_id": MODEL_ID,
              "tokenizer_revision": MODEL_REVISION, "seed": seed, "tokens": tokens,
              "counts": counts, "long_test_windows": long_test_windows,
              "long_tokens": long_tokens, "selection": "hash_ordered_document_prefix_v1",
              "bos_token_id": tokenizer.bos_token_id}
    corpus_identity = _json_digest({"recipe": recipe, "selected": selected,
                                   "source_sha256": {s: v["sha256"] for s, v in sources.items()}})
    manifest_path, tokens_path = destination / "manifest.json", destination / "tokens.npz"
    if manifest_path.exists() or tokens_path.exists():
        if not manifest_path.exists() or not tokens_path.exists():
            raise FileExistsError("Partially written corpus exists; choose a fresh destination.")
        old = json.loads(manifest_path.read_text())
        if old["corpus_identity"] != corpus_identity or old["tokens_sha256"] != sha256_file(tokens_path):
            raise FileExistsError("Existing corpus does not match this recipe or its recorded checksum.")
        return old
    _write_deterministic_npz(tokens_path, arrays)
    manifest = {"schema_version": 1, "recipe": recipe, "sources": sources,
                "selected": selected, "corpus_identity": corpus_identity,
                "tokens_path": str(tokens_path), "tokens_sha256": sha256_file(tokens_path),
                "shapes": {name: list(array.shape) for name, array in arrays.items()}}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_token_corpus(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, torch.Tensor]:
    """Load prepared token IDs, checking both archive and per-window hashes."""
    path = Path(path)
    manifest_path = path / "manifest.json" if path.is_dir() else path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    tokens_path = manifest_path.with_name("tokens.npz")
    checksum = sha256_file(tokens_path)
    if checksum != manifest["tokens_sha256"] or (expected_sha256 and checksum != expected_sha256):
        raise ValueError("Token corpus checksum does not match its frozen identity.")
    with np.load(tokens_path, allow_pickle=False) as archive:
        if set(archive.files) != set(manifest["selected"]):
            raise ValueError("Token corpus split names disagree with its manifest.")
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, array in arrays.items():
        if array.dtype != np.int64 or list(array.shape) != manifest["shapes"][name]:
            raise ValueError(f"Invalid token shape or dtype for {name}.")
        if any(_token_digest(ids) != entry["token_ids_sha256"]
               for ids, entry in zip(array, manifest["selected"][name], strict=True)):
            raise ValueError(f"Token IDs for {name} disagree with selected document windows.")
    return {name: torch.from_numpy(array) for name, array in arrays.items()}


def cached_model_manifest(*, local_files_only: bool = True) -> dict:
    """Hash the immutable pretrained weights, config and tokenizer actually used."""
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                     local_files_only=local_files_only,
                                     allow_patterns=["*.json", "*.safetensors"]))
    files = sorted(path for path in snapshot.iterdir()
                   if path.suffix in {".json", ".safetensors"} and path.is_file())
    if not any(path.suffix == ".safetensors" for path in files):
        raise FileNotFoundError("Pinned Llama safetensors weights are not cached.")
    return {"model_id": MODEL_ID, "revision": MODEL_REVISION, "snapshot": str(snapshot),
            "files": {path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                      for path in files}}


def load_llama_for_capture(*, device: str | torch.device = "cuda", local_files_only: bool = True):
    """Load pinned pretrained Llama in FP32/eager mode, frozen and in eval mode."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                      torch_dtype=torch.float32, attn_implementation="eager",
                                      local_files_only=local_files_only)
    return model.to(device).eval().requires_grad_(False)


@dataclass(frozen=True)
class HeadCapture:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    teacher: torch.Tensor
    scale: float
    kv_head: int
    native_max_abs_error: float


def causal_softmax_output(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, scale: float) -> torch.Tensor:
    """FP32 causal teacher outputs for equal-length, unpadded [B,T,D] inputs."""
    if (q.ndim != 3 or k.shape != q.shape or v.ndim != 3
            or v.shape[:2] != q.shape[:2] or min(q.shape) < 1):
        raise ValueError("Require matching nonempty Q/K and value [batch,tokens,width] inputs.")
    scores = (q.float() @ k.float().transpose(-2, -1)) * scale
    causal = torch.ones(q.shape[1], q.shape[1], device=q.device, dtype=torch.bool).tril()
    weights = scores.masked_fill(~causal, -torch.inf).softmax(dim=-1)
    return weights @ v.float()


@torch.no_grad()
def capture_llama_heads(
    model,
    input_ids: torch.Tensor,
    heads: Sequence[tuple[int, int]] = DEFAULT_HEADS,
) -> dict[tuple[int, int], HeadCapture]:
    """Capture selected post-RoPE Q/K, GQA-matched V and native head outputs.

    Exactly one full forward is performed. Outputs are CPU FP32 [batch,tokens,d].
    Independent softmax reconstruction must match the model's native pre-o_proj
    activations. Padding, KV caching, training mode and mixed precision are not
    supported: they would make the intended teacher ambiguous.
    """
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    backbone = model.model if hasattr(model, "model") else model
    config = backbone.config
    if getattr(config, "model_type", None) != "llama" or model.training:
        raise ValueError("Capture requires an eval-mode Llama model.")
    if input_ids.ndim != 2 or min(input_ids.shape) < 1 or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be a nonempty unpadded LongTensor [batch,tokens].")
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("Capture requires FP32 pretrained weights.")
    if getattr(config, "_attn_implementation", None) != "eager":
        raise ValueError("Capture requires eager attention for native-output qualification.")
    heads = tuple(heads)
    if not heads or len(set(heads)) != len(heads):
        raise ValueError("Request at least one head, with no duplicates.")
    query_heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    if query_heads % kv_heads:
        raise ValueError("Query heads must be divisible by KV heads.")
    for layer, head in heads:
        if not 0 <= layer < config.num_hidden_layers or not 0 <= head < query_heads:
            raise ValueError(f"Invalid requested (layer, query_head): {(layer, head)}")
    layers = sorted({layer for layer, _ in heads})
    captured = {layer: {} for layer in layers}
    handles = []

    def projection_hook(layer, name):
        def hook(_module, _args, output):
            captured[layer][name] = output.detach()
        return hook

    def attention_pre_hook(layer):
        def hook(_module, args, kwargs):
            embeddings = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            if embeddings is None:
                raise ValueError("Llama attention did not receive explicit rotary embeddings.")
            captured[layer]["rope"] = tuple(t.detach() for t in embeddings)
        return hook

    def output_pre_hook(layer):
        def hook(_module, args):
            captured[layer]["teacher"] = args[0].detach()
        return hook

    try:
        for layer in layers:
            attention = backbone.layers[layer].self_attn
            handles.append(attention.register_forward_pre_hook(attention_pre_hook(layer), with_kwargs=True))
            for name in ("q", "k", "v"):
                handles.append(getattr(attention, f"{name}_proj").register_forward_hook(projection_hook(layer, name)))
            handles.append(attention.o_proj.register_forward_pre_hook(output_pre_hook(layer)))
        backbone(input_ids=input_ids.to(next(model.parameters()).device), use_cache=False,
                 return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    batch, length = input_ids.shape
    result = {}
    for layer in layers:
        attention = backbone.layers[layer].self_attn
        width = attention.head_dim
        entry = captured[layer]
        q = entry["q"].reshape(batch, length, query_heads, width).transpose(1, 2)
        k = entry["k"].reshape(batch, length, kv_heads, width).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *entry["rope"])
        values = entry["v"].reshape(batch, length, kv_heads, width)
        native = entry["teacher"].reshape(batch, length, query_heads, width)
        for _, head in (spec for spec in heads if spec[0] == layer):
            kv_head = head // (query_heads // kv_heads)
            qh, kh, vh = q[:, head], k[:, kv_head], values[:, :, kv_head]
            teacher = native[:, :, head]
            reconstructed = causal_softmax_output(qh, kh, vh, scale=attention.scaling)
            error = float((teacher - reconstructed).abs().max())
            if not torch.allclose(teacher, reconstructed, rtol=2e-5, atol=2e-6):
                raise AssertionError(f"Native teacher mismatch at layer {layer}, head {head}: max error {error}")
            result[layer, head] = HeadCapture(
                *(tensor.float().cpu().contiguous() for tensor in (qh, kh, vh, teacher)),
                scale=float(attention.scaling), kv_head=kv_head, native_max_abs_error=error,
            )
    return {spec: result[spec] for spec in heads}
