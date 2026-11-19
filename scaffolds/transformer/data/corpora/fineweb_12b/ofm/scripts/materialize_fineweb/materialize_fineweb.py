from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

from tqdm import tqdm

from data.packed import (
    PackedTokenPacker,
    ResumablePackedSequenceWriter,
    load_packed_manifest,
    manifest_fingerprint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize FineWeb text as indexed fixed-length token shards."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--tokenizer-name", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--sequence-length", type=int, default=2049)
    parser.add_argument("--train-sequences", type=int, required=True)
    parser.add_argument("--eval-sequences", type=int, required=True)
    parser.add_argument("--shard-sequences", type=int, default=65_536)
    parser.add_argument("--tokenizer-batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.005)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--max-epochs", type=int, default=4)
    return parser.parse_args()


def is_validation_document(text: str, fraction: float) -> bool:
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    value = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    return value < int(fraction * (1 << 64))


def resolve_revisions(args: argparse.Namespace) -> tuple[str, str]:
    from huggingface_hub import HfApi

    api = HfApi()
    dataset_revision = api.dataset_info(
        args.dataset_name,
        revision=args.dataset_revision,
    ).sha
    tokenizer_revision = api.model_info(
        args.tokenizer_name,
        revision=args.tokenizer_revision,
    ).sha
    if not dataset_revision or not tokenizer_revision:
        raise RuntimeError("Could not resolve immutable dataset/tokenizer revisions")
    return dataset_revision, tokenizer_revision


def add_texts(tokenizer, texts: list[str], packer: PackedTokenPacker) -> None:
    if not texts or packer.done:
        return
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        truncation=False,
    )["input_ids"]
    for token_ids in encoded:
        packer.add_document(token_ids, tokenizer.eos_token_id)
        if packer.done:
            break


def make_writer(
    output_dir: Path,
    *,
    split: str,
    num_sequences: int,
    args: argparse.Namespace,
    metadata: dict,
) -> ResumablePackedSequenceWriter:
    split_metadata = dict(metadata)
    split_metadata["split"] = split
    return ResumablePackedSequenceWriter(
        output_dir / split,
        sequence_length=args.sequence_length,
        num_sequences=num_sequences,
        shard_sequences=args.shard_sequences,
        dtype="uint16",
        metadata=split_metadata,
    )


def load_completed_corpus(
    output_dir: Path,
    args: argparse.Namespace,
) -> dict | None:
    corpus_path = output_dir / "corpus.json"
    if not corpus_path.exists():
        return None

    summary = json.loads(corpus_path.read_text())
    if summary.get("format") != "ofm-packed-corpus" or summary.get("version") != 1:
        raise ValueError(f"Unsupported corpus summary: {corpus_path}")

    split_specs = (
        ("train", args.train_sequences),
        ("eval", args.eval_sequences),
    )
    for split, expected_sequences in split_specs:
        relative_manifest = summary.get(f"{split}_manifest")
        if not isinstance(relative_manifest, str):
            raise ValueError(f"Missing {split} manifest in {corpus_path}")
        manifest_path = (output_dir / relative_manifest).resolve()
        if not manifest_path.is_relative_to(output_dir):
            raise ValueError(f"{split} manifest escapes corpus directory")
        loaded_path, manifest = load_packed_manifest(manifest_path)
        if loaded_path != manifest_path:
            raise ValueError(f"Unexpected {split} manifest path")
        if manifest["sequence_length"] != args.sequence_length:
            raise ValueError(f"Completed {split} sequence length does not match request")
        if manifest["num_sequences"] != expected_sequences:
            raise ValueError(f"Completed {split} sequence count does not match request")

        metadata = manifest["metadata"]
        expected_metadata = {
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "tokenizer_name": args.tokenizer_name,
            "partition": {
                "algorithm": "sha256-text",
                "validation_fraction": args.validation_fraction,
            },
            "shuffle_seed": args.shuffle_seed,
            "shuffle_buffer_size": args.shuffle_buffer_size,
            "split": split,
        }
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"Completed {split} metadata {key} does not match request"
                )

        expected_fingerprint = summary.get(f"{split}_manifest_sha256")
        if manifest_fingerprint(manifest_path) != expected_fingerprint:
            raise ValueError(f"Completed {split} manifest fingerprint does not match")

    return summary


def materialize(args: argparse.Namespace) -> dict:
    if args.tokenizer_batch_size < 1:
        raise ValueError("tokenizer_batch_size must be positive")
    if args.max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    if args.shuffle_buffer_size < 1:
        raise ValueError("shuffle_buffer_size must be positive")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = load_completed_corpus(output_dir, args)
    if completed is not None:
        return completed
    dataset_revision, tokenizer_revision = resolve_revisions(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        revision=tokenizer_revision,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token")
    if len(tokenizer) > 65_536:
        raise ValueError("Tokenizer vocabulary does not fit in uint16")

    metadata = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "dataset_revision": dataset_revision,
        "tokenizer_name": args.tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "eos_token_id": tokenizer.eos_token_id,
        "partition": {
            "algorithm": "sha256-text",
            "validation_fraction": args.validation_fraction,
        },
        "shuffle_seed": args.shuffle_seed,
        "shuffle_buffer_size": args.shuffle_buffer_size,
    }
    train_writer = make_writer(
        output_dir,
        split="train",
        num_sequences=args.train_sequences,
        args=args,
        metadata=metadata,
    )
    eval_writer = make_writer(
        output_dir,
        split="eval",
        num_sequences=args.eval_sequences,
        args=args,
        metadata=metadata,
    )
    train_packer = PackedTokenPacker(train_writer)
    eval_packer = PackedTokenPacker(eval_writer)

    total_sequences = args.train_sequences + args.eval_sequences
    progress = tqdm(
        total=total_sequences,
        initial=train_writer.num_written + eval_writer.num_written,
        unit="seq",
        desc="materialize",
    )
    epochs_scanned = 0
    for epoch in range(args.max_epochs):
        if train_packer.done and eval_packer.done:
            break
        if epoch > 0 and not eval_packer.done:
            raise RuntimeError(
                "The first source epoch did not contain enough held-out data; increase "
                "validation_fraction instead of duplicating evaluation documents"
            )

        before_epoch = train_writer.num_written + eval_writer.num_written
        dataset = load_dataset(
            args.dataset_name,
            name=args.dataset_config,
            split=args.dataset_split,
            revision=dataset_revision,
            streaming=True,
        )
        dataset = dataset.shuffle(
            seed=args.shuffle_seed + epoch,
            buffer_size=args.shuffle_buffer_size,
        )
        train_texts: list[str] = []
        eval_texts: list[str] = []

        def flush() -> None:
            before = train_writer.num_written + eval_writer.num_written
            add_texts(tokenizer, train_texts, train_packer)
            add_texts(tokenizer, eval_texts, eval_packer)
            train_texts.clear()
            eval_texts.clear()
            progress.update(train_writer.num_written + eval_writer.num_written - before)

        row_iterator = iter(dataset)
        try:
            for row in row_iterator:
                text = row.get(args.text_field)
                if not text:
                    continue
                if is_validation_document(text, args.validation_fraction):
                    if not eval_packer.done:
                        eval_texts.append(text)
                elif not train_packer.done:
                    train_texts.append(text)

                if len(train_texts) + len(eval_texts) >= args.tokenizer_batch_size:
                    flush()
                if train_packer.done and eval_packer.done:
                    break
        finally:
            close = getattr(row_iterator, "close", None)
            if close is not None:
                close()
            del row_iterator
            del dataset
            gc.collect()

        flush()
        epochs_scanned = epoch + 1
        if (
            not (train_packer.done and eval_packer.done)
            and train_writer.num_written + eval_writer.num_written == before_epoch
        ):
            raise RuntimeError(f"Source epoch {epoch} produced no complete sequences")
    progress.close()

    if not train_packer.done or not eval_packer.done:
        raise RuntimeError(
            f"Materialization stopped after {epochs_scanned} epochs with "
            f"train={train_writer.num_written}/{args.train_sequences} and "
            f"eval={eval_writer.num_written}/{args.eval_sequences} sequences"
        )

    train_writer.metadata["source_epochs_scanned"] = epochs_scanned
    eval_writer.metadata["source_epochs_scanned"] = epochs_scanned
    train_manifest = train_writer.finish()
    eval_manifest = eval_writer.finish()
    summary = {
        "format": "ofm-packed-corpus",
        "version": 1,
        "train_manifest": str(train_manifest.relative_to(output_dir)),
        "train_manifest_sha256": manifest_fingerprint(train_manifest),
        "eval_manifest": str(eval_manifest.relative_to(output_dir)),
        "eval_manifest_sha256": manifest_fingerprint(eval_manifest),
        "epochs_scanned": epochs_scanned,
    }
    corpus_path = output_dir / "corpus.json"
    temp_corpus_path = corpus_path.with_suffix(".json.tmp")
    temp_corpus_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temp_corpus_path, corpus_path)
    return summary


def main() -> None:
    summary = materialize(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
    # The streaming dependency stack leaves a native worker attached after an
    # early dataset exit. All artifacts are flushed above; bypass its broken
    # interpreter-finalization hook without masking Python exceptions.
    os._exit(0)
