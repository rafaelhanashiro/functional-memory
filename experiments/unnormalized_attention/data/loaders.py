"""Indexed immutable packed data; synthetic data exists only for inexpensive checks."""
from collections.abc import Iterator
from dataclasses import replace

import numpy as np

from experiments.unnormalized_attention.configs import DataConfig
from experiments.unnormalized_attention.data.packed import LoadedPackedManifest, PackedBatchIterator


def synthetic_batches(config: DataConfig, start_sequence: int = 0) -> Iterator[np.ndarray]:
    # Seed each sequence independently so cursor seeking never changes its tokens.
    cursor = start_sequence
    while True:
        batch = [
            np.random.default_rng(np.random.SeedSequence([config.shuffle_seed, i])).integers(
                config.synthetic_vocab_size, size=config.sequence_length, dtype=np.int32
            )
            for i in range(cursor, cursor + config.batch_size)
        ]
        yield np.stack(batch)
        cursor += config.batch_size


def make_train_iterator(
    config: DataConfig,
    *,
    start_sequence: int = 0,
    validated_manifest: LoadedPackedManifest | None = None,
) -> Iterator[np.ndarray]:
    if config.source == "synthetic":
        return synthetic_batches(config, start_sequence)
    if config.source != "packed":
        raise ValueError("Only immutable packed data and synthetic checks are supported")
    if not config.train_manifest:
        raise ValueError("data.train_manifest is required for packed data")
    return iter(PackedBatchIterator(
        config.train_manifest,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        start_sequence=start_sequence,
        validated_manifest=validated_manifest,
    ))


def make_eval_iterator(
    config: DataConfig,
    *,
    validated_manifest: LoadedPackedManifest | None = None,
) -> Iterator[np.ndarray]:
    if config.source == "synthetic":
        return synthetic_batches(replace(config, shuffle_seed=config.shuffle_seed + 1))
    if not config.eval_manifest:
        raise ValueError("data.eval_manifest is required for packed data")
    return iter(PackedBatchIterator(
        config.eval_manifest,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        validated_manifest=validated_manifest,
    ))
