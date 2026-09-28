# Hermite and Taylor attention with QK normalization

Does the fixed standard-normal Hermite approximation of `exp(s)` improve on
Taylor truncation at the same degree when the score geometry and training recipe
match? This cohort uses **QK normalization for every run**, `d_K=d_V=64`, degrees
1–4, seed 0, and both normalized and unnormalized attention.

**Home cluster: Kennel.** The artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`. Each permanent run ID in
[runs.csv](runs.csv) owns its frozen recipe, execution record, measurements,
retained checkpoint, and `training.log` under `runs/`. Completed runs retain their
validation-best checkpoint; normalized degree 2 retains its step-3500 checkpoint to
replay its first nonfinite update, and the other failures retain none.
[sweep.yaml](sweep.yaml) lists membership and the exact existing Taylor controls.
Findings are in [results/attention_scaling](../../results/attention_scaling/README.md).

The cohort was submitted on 2026-09-27 as jobs **203102–203113**, from launch
commit `bd31f8ad7a8a964b286387ad7ad7723ea405b286`, after the compiled H100/B300
checks (jobs 203015/203016) passed. Runs use the `standard` allowance one at a
time on `akita`, or `long`. Individual job IDs, partitions, node pins, ordering
dependencies, and exact submission commands are stored in the run records; live
scheduler status and recorded training status are distinct.

## Method and comparison

Scores use post-RoPE Q and K with per-head, per-layer learned QK scales:
`s_ij = exp(gamma) cos(q_i, k_j)`, initialized with `exp(gamma)=sqrt(64)=8`.
This is independent of normalization, which divides the weighted value sum
by the causal sum of polynomial weights.

The eight Hermite runs use
`p_m(s) = exp(1/2) sum_(r=0)^m He_r(s)/r!`, the probabilists' Hermite expansion
under standard-normal weighting. Coefficients stay fixed; the QK scales are
learned. The factor `exp(1/2)` is retained for unnormalized attention and cancels
mathematically under normalization. Normalized degree 1 is consequently the
same mathematical kernel as normalized degree-1 Taylor and provides a check of
the comparison, subject to differences in floating-point evaluation.

Normalized Hermite degree 2 uses the factored evaluation
`p_2(s) = exp(1/2) (s+1)^2 / 2`, avoiding the known loss of precision near its
root when the expanded polynomial is evaluated. Other Hermite variants use the
expanded coefficients stored in their recipes. All use the fused attention
backends. Exact-zero/nonfinite normalizers and nonfinite training updates remain
explicit numerical failures; there is no denominator floor or coefficient repair.

Four new fixed unnormalized Taylor runs (`p_m(s)=sum_(r=0)^m s^r/r!`) supply the
missing QK-normalized controls. The normalized controls already exist:

| Degree | Study | Run ID | Recorded outcome |
| ---: | --- | --- | --- |
| 1 | `odd_taylor_attention` | `raa90edf82385` | Numerical failure |
| 2 | `key_dimension_taylor` | `r93c9d0814322` | Completed |
| 3 | `odd_taylor_attention` | `r8b46f1ad9453` | Numerical failure |
| 4 | `key_dimension_taylor` | `ra9b651caa405` | Completed |

All new models have 199,906,464 parameters, including 160 QK scales: 10 layers,
width 1024, 16 full MHA heads, SwiGLU width 3008, untied 32,000-token embeddings,
context 2048, RoPE base 10000, no dropout, and BF16 compute. The common FineWeb
data order, global batch 256, microbatch 64 with four accumulation steps, AdamW
peak/minimum rates `3e-4`/`3e-5`, 2,000 warmup updates, clipping at 1, and
19,074-update budget match the normalized Taylor controls. Weight decay is 0.1
with `exclude_embeddings_norms_biases_v1`. Validation uses seven batches every
1,000 updates and at completion. Terminal validation and validation-best
checkpoint selection are distinct.

The previous `hermite_attention` and `hermite_factored_m2` executions with
`qk_norm: false` are **legacy raw-score runs**, preserved in their original
worktrees until integration. They are excluded from this matched QK-normalized
comparison. New executions have new IDs and do not resume their checkpoints.

## Qualification and launch

Each allocation runs CPU and compiled GPU checks, verifies initialization,
executes a 20-update full-recipe pilot, and verifies an exact fresh-process
resume before production. A pilot numerical failure is recorded separately from
a production failure. Passing a short pilot does not establish long-run stability.
Hardware and resolved backend are recorded separately from the scientific recipe.
Jobs request four H100 or B300 GPUs on the partition chosen at submission;
qualification covers the actual allocated hardware before any production training.

From the worktree root, prepare and test the frozen source before submitting:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
JAX_PLATFORMS=cpu PYTHONPATH=. OMP_NUM_THREADS=1 \
  scaffolds/transformer/.venv/bin/python -m pytest -q \
  -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m 'not gpu' \
  experiments/hermite_attention/tests
```

The submission for each allocated run uses the following, with the partition
options (`--partition`, and for `standard` a 24-hour `--time`) recorded in its
run record:

```bash
RUN=<permanent-run-id>
LAUNCH_COMMIT=$(git rev-parse HEAD)
sbatch --parsable --gres=gpu:4 --job-name="hermite-$RUN" \
  --output="experiments/hermite_attention/runs/$RUN/training.log" \
  scaffolds/transformer/launchers/kennel.sbatch \
  experiments.hermite_attention.acceptance \
  "experiments/hermite_attention/runs/$RUN/config.yaml" "$LAUNCH_COMMIT" 4 B300,H100
```

Record the returned job ID and exact command in the run record. Requeues retain
the run ID and append attempts; a fresh execution requires a new ID. Do not
resubmit a run while its job can still execute. Every start checks the frozen
execution set against its launch commit. W&B is the live monitoring mirror;
canonical measurements remain under each run.

Regenerate the study index with:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records index \
  experiments/hermite_attention
```
