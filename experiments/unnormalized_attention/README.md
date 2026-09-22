# Transformer attention without the softmax denominator

This experiment trains the canonical OFM 250M Transformer from scratch with
causal attention `Y_i = sum_{j<=i} exp(Q_i K_j^T / sqrt(d_k)) V_j`.
The attention weights are not divided by their sum. Exponents use the original
post-RoPE scores: subtracting a row maximum without restoring its exponential
factor would change this experiment. There is no score clipping, prefix-length
division, new Q/K normalization, or added output normalization.

## Matched recipe

`configs/base.yaml` ports OFM's `transformer_250m.yaml` at source
`32ea743b7b3aed1d631116dfaa3c4b85101baf32`: **253,654,016 parameters**, 14 layers,
width 1024, 16 MHA heads, dK=dV=64, SwiGLU width 3008, 32,000-token untied
embeddings, 2,048 prediction positions, RoPE base 10,000, RMSNorm epsilon 1e-6,
and zero dropout. Existing block/final RMSNorm and vocabulary softmax remain
part of the baseline architecture.

Seed 0, packed FineWeb data, tokenizer, sequence order, AdamW recipe, and
learning-rate schedule match the historical control. The global batch is 256,
global microbatch 64, and accumulation count four. There are 21,935 updates,
**11,500,257,280 prediction tokens**, 2,000 warmup updates, and peak/minimum
learning rates 3e-4/3e-5. Evaluation uses the first 1,792 held-out sequences
every 1,000 updates and at completion. Parameters and optimizer state are FP32;
model compute is BF16. The packed corpus is an external immutable input under
`/home/rafah/projects/ofm/datasets/fineweb_12b`; runtime model/training code is
entirely in this repository.

The historical softmax model trained on 2 H200 GPUs and achieved terminal and
best validation PPL 13.3496. Its evidence is summarized in
`../../results/unnormalized_attention/baseline.json`. This is a single-seed
comparison; the normalized control uses cuDNN while this experiment uses a
Pallas/Triton kernel, so the execution arithmetic is not bitwise matched.
`model.attention_normalization: true` selects normalized attention for checks
or a separately configured future control; the authorized production run is
the unnormalized variant only.

## Implementation and acceptance

The scaffold is derived from this repository's retained polynomial-attention
training infrastructure. A blocked JAX implementation supplies a reference and
per-layer scalar diagnostics. The production kernel tiles queries and keys,
stores no sequence-square attention array, accumulates in FP32, and recomputes
exponentials in its custom reverse-mode derivative. The derivative is
`dS = exp(S) * (dY V^T)` without the softmax row-coupling term. FP32 weights are
split into three BF16 components for mixed products with BF16 V, avoiding a
single BF16 rounding of the exponential weights.

Tests compare outputs and Q/K/V gradients to a direct reference, exercise causal
boundaries, verify that zero scores produce prefix sums, and check sensitivity
to additive score shifts and visible overflow. Preflight verifies exact initial
FP32 parameter identity with the independent OFM port in `key_dimension`, runs
CPU and GPU acceptance tests, then runs 20 full-context/full-microbatch updates
and validation followed by a checkpoint resume to update 21. The pilot preserves
the full learning-rate horizon. Finite loss and pre-clipping gradient norm are
checked every training update; nonfinite values stop with recorded diagnostics.
Row sums are measured for inspection, never used as an unnormalized denominator.

## Slurm execution

For a fresh run on another cluster, see [the Kennel handoff](KENNEL.md) and
`run_allocated.sh`, which accept external data/config/output paths and leave
partition/account/resource selection to that cluster's submission script.

Develop and run in a dedicated branch and checkout. Both environment bootstrap
and execution use Slurm. From that checkout, substituting its absolute path:

```bash
project_root=/home/rafah/projects/functional_memory
execution_root="$PWD"
source_commit="$(git rev-parse HEAD)"
mkdir -p "$project_root/results/logs/unnormalized_attention/bootstrap"
sbatch --output="$project_root/results/logs/unnormalized_attention/bootstrap/slurm-%j.out" \
  --error="$project_root/results/logs/unnormalized_attention/bootstrap/slurm-%j.out" \
  experiments/unnormalized_attention/bootstrap.sbatch "$execution_root" "$project_root"
```

After bootstrap completes, submit `launch.sbatch preflight "$execution_root"
"$project_root" "$source_commit" 2 H100` with a log under
`results/logs/unnormalized_attention/preflight/slurm-%j.out`. Submit the same
launcher with `train` only with an `afterok:<preflight_job>` dependency. The
launcher defaults to MIT's `mit_preemptable`, 2 H100 GPUs, a 48-hour limit, and
requeue. H100s were selected for availability; the historical control used H200s.
Training verifies the matching successful preflight again at startup.
Never change this checkout or its environment while any job can run or requeue.

Preflight records are keyed by source, resolved configuration, dependency lock,
dataset manifest hashes, and GPU allocation. Passed checks survive retries;
production restores optimizer state and the exact packed-data position.
Shared caches and the isolated locked environment are stored under the main
repository's `.cache/` and `.envs/`, outside the execution checkout.

## Results and retention

Live metrics, configuration, data identity, status, and best/latest checkpoints
live in `results/unnormalized_attention/250m-exp-seed0/` in the execution
checkout. Keep both best and latest while training is active or requeueable.
After terminal evaluation and review, retain metrics and a provenance manifest
under `results/unnormalized_attention/`, exactly one validation-best checkpoint
under `experiments/unnormalized_attention/checkpoints/250m-exp-seed0/`, and one
successful Slurm log matching provenance. Terminal metrics survive independently
of checkpoint selection. Large artifacts remain ignored by Git. Safely retain
these outputs before merging the experiment, deleting its branch, and removing
the temporary checkout, as required by `AGENTS.md`.
