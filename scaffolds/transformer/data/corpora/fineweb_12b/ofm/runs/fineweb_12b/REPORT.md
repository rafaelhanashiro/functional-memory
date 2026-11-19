# FineWeb-Edu 12B Packed Corpus

## Status

Complete. The corpus and common eval split were materialized and fully
checksum-validated.

## Question

Build one deterministic packed corpus that can supply all new 50M, 100M, and
238M runs, including the requested 11.5B-token training budget, and remove the
obsolete packed corpora.

## Setup

- Source: FineWeb-Edu `sample-100BT`, revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
- Tokenizer: `NousResearch/Llama-2-7b-hf`, revision
  `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`.
- Sequence shape: 2,049 stored token IDs, providing 2,048 next-token targets.
- Train target: 5,859,375 sequences, exactly 12,000,000,000 prediction tokens
  (12,005,859,375 stored token IDs).
- Eval target: 4,888 held-out sequences, or 10,010,624 prediction tokens.
- Partition: deterministic SHA-256 text split at 1%; shuffle seed 0 and buffer
  size 10,000.
- Execution branch/worktree: `data-12b` / `worktrees/data_12b`.

## Files Touched

- `runs/fineweb_12b/REPORT.md`
- `scripts/fineweb_12b/materialize.sbatch`
- `runs/fineweb_12b/data/`: 90 checksummed `uint16` train shards, one eval
  shard, manifests, resumable progress, corpus summary, and job logs.

The following exact obsolete packed-data paths were removed; reports, configs,
model checkpoints, and shared caches were not touched:

```text
worktrees/additive_hybrids_1b/runs/fineweb_1b_chinchilla/data
worktrees/functional_tape_softmax_50m/runs/fineweb_1b_chinchilla/data
worktrees/learned_feature_tape_readnorm_1b/runs/fineweb_1b_chinchilla/data
worktrees/materialized_data/runs/fineweb_1b_chinchilla/data
worktrees/tape_state_context_analysis/runs/fineweb_1b_chinchilla/data
worktrees/wide_state_gla_1b/runs/fineweb_1b_chinchilla/data
worktrees/baselines_238m_chinchilla/runs/fineweb_5b_chinchilla/data
worktrees/gla_two_level_chunked_kernel/runs/fineweb_5b_chinchilla/data
worktrees/transformer_238m_chinchilla_rerun/runs/fineweb_5b_chinchilla/data
worktrees/materialized_data/runs/materialized_fineweb/smoke-data
worktrees/materialized_data/runs/materialized_fineweb/smoke-data-http
worktrees/materialized_data/runs/materialized_fineweb/smoke-data-serial-tokenizer
worktrees/materialized_data/runs/materialized_fineweb/smoke-data-closed-stream
worktrees/materialized_data/runs/materialized_fineweb/smoke-data-final-exit
```

## Slurm Jobs

- `20811682`: production materialization on `node1900` in
  `mit_preemptable`; completed `0:0` in 1:37:42 with 34,996,344 KiB peak RSS.
- `20814071`: independent full-corpus validation on `node1603` in
  `mit_quicktest`; completed `0:0` in 1:44 with 8,384,116 KiB peak RSS.

## Results

- Train manifest SHA-256:
  `9de775ed836ab7c61857207335a981c09630caa3226d2dc0ee0b799cdcb2f4f4`.
- Eval manifest SHA-256:
  `0e8615538d9dccad17956d8d9839394d05a56f8ec210dc67716db829a543634f`.
- Train: 5,859,375 sequences, 12,005,859,375 stored IDs,
  12,000,000,000 prediction tokens, 90 contiguous shards.
- Eval: 4,888 sequences, 10,015,512 stored IDs, 10,010,624 prediction tokens,
  one shard with SHA-256
  `3e19cea9c1cfe9ca02e78436ed3defb319354ffd3e0c2cade0ce2924f502928a`.
- The source was scanned for exactly one epoch, so no training documents were
  introduced by source replay.
- The independent validation recomputed every manifest-listed shard checksum,
  checked shapes, dtypes, sequence continuity and totals, verified immutable
  source/tokenizer metadata and corpus fingerprints, and found no missing,
  extra, or temporary files. Regular files total 24,035,486,562 bytes.

Before deletion, the old inventory contained five 2,021,837,619-byte 1B data
roots, three 4.78B roots of 9,581,813,714--9,583,298,047 bytes, one
20,041,139-byte eval-only root, and five 183,993--184,382-byte smokes. The old
eval manifest and shard identities were respectively
`40e17c20aff7df1264d53ce3394a58af54178470dcd3a4cd290172345e8e6889`
and `00c5fd6a1d41c0927967948a5c1e46a1e83cc7a49442c1864f92f0eb36cf7b77`.
They differed from the new matched `sample-100BT` eval, so the new eval is the
single common eval for all reruns and the temporary legacy copy was removed.
A final project-wide worktree search found no old FineWeb 1B/5B data directory,
packed manifest, or large shard outside the new corpus.

## Interpretation

The 12B target is exact at this sequence length and leaves 500M prediction
tokens (4.35%) beyond the largest planned 11.5B run without requiring a less
memorable corpus size.

## Caveats

The production data are intentionally untracked large artifacts. They remain
under the stable `data_12b` worktree until transferred to the new
`scale_suite_2p5x` execution worktree; future experiment configs must reference
the transferred train and eval manifests together.

## Next Steps

Move the verified `runs/fineweb_12b/data/` directory into the stable
`scale_suite_2p5x` worktree after no job is active or requeueable, then launch
all reruns against its train manifest and the single common eval manifest.
