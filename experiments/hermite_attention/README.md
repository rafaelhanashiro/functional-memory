# Fixed Hermite attention

Does a fixed Hermite approximation improve on fixed Taylor coefficients at the
same degree, and does removing row normalization avoid failures caused by signed
kernel sums near zero? This experiment trains **eight 200M models**: degrees
**m = 1, 2, 3, 4**, each with and without row normalization, seed 0.

**Home cluster: Kennel.** While the experiment runs, its artifact root is
`/data/cl/u/rafah/home/projects/functional-memory/worktrees/hermite_attention`.
All scientific records, checkpoints, and logs stay in that worktree. Packed FineWeb
is under `/data/cl/u/rafah/home/datasets/fineweb_12b`. The
[provenance manifest](provenance/manifest.json) records the source commit, frozen
execution set, configurations, acceptance, allocations, and artifact checksums.

## Method

Use the probabilists' Hermite projection of the exponential under a standard
normal weighting, with fixed coefficients throughout training:

```text
p_m(s) = exp(1/2) sum_(r=0)^m He_r(s) / r!,   s_ij = q_i · k_j / sqrt(d_K)
normalized:   y_i = sum_(j<=i) p_m(s_ij) v_j / sum_(j<=i) p_m(s_ij)
unnormalized: y_i = sum_(j<=i) p_m(s_ij) v_j
```

The recipes store the expanded monomial coefficients below, each multiplied by
**exp(1/2)**. This factor cancels mathematically in normalized attention. It is
retained in unnormalized attention because it belongs to the Hermite projection
of `exp(s)`. No empirical score fitting or coefficient learning is used.

| Degree | Coefficients before multiplication by exp(1/2), constant first |
| ---: | --- |
| 1 | `[1, 1]` |
| 2 | `[1/2, 1, 1/2]` |
| 3 | `[1/2, 1/2, 1/2, 1/6]` |
| 4 | `[5/8, 1/2, 1/4, 1/6, 1/24]` |

Scores are raw post-RoPE dot products with `qk_norm: false`. Degrees 1, 3, and 4
permit negative kernels. Degree 2 is nonnegative but has a zero at `s=-1`.
Normalized attention uses the signed row sum directly, with no epsilon, clipping,
absolute value, score shift, or positivity repair. An exact zero or nonfinite
denominator is a numerical failure. Unnormalized attention has no denominator or
length scaling. Nonfinite loss, gradient norm, or updated parameters stop a run;
the stopping step, last valid measurements, failure diagnostics, and matching log
are retained as scientific evidence, never labeled as completed training.

Each model has **199,906,304 trainable parameters**: 10 layers, width 1024, 16 full
MHA heads, `d_K=d_V=64`, SwiGLU width 3008, untied 32,000-token embeddings, context
2048, RoPE base 10000, no dropout, and BF16 compute. Recipes use seed 0 and the
same packed-data order, global batch 256, global microbatch 64, four accumulation
steps, AdamW peak/minimum rates `3e-4`/`3e-5`, 2,000 warmup updates, and **19,074
updates / 10,000,269,312 prediction tokens**. Selective weight decay is explicitly
`exclude_embeddings_norms_biases_v1`, rate 0.1; global gradient clipping is 1.
Evaluation uses seven global batches (1,792 sequences) every 1,000 updates and at
completion. Terminal held-out perplexity is the primary endpoint; validation-best
metrics and checkpoint selection are recorded separately.

## Results

<!-- BEGIN GENERATED RESULTS -->
| Variant | Degree | Status | Last update | Terminal PPL | Failure update |
| --- | ---: | --- | ---: | ---: | ---: |
| normalized | 1 | numerical-failure | 173 |  | 173 |
| normalized | 2 | acceptance-numerical-failure |  |  | 1 |
| normalized | 3 | incomplete | 0 |  |  |
| normalized | 4 | numerical-failure | 53 |  | 53 |
| unnormalized | 1 | incomplete | 0 |  |  |
| unnormalized | 2 | incomplete | 0 |  |  |
| unnormalized | 3 | submitted |  |  |  |
| unnormalized | 4 | incomplete | 0 |  |  |
<!-- END GENERATED RESULTS -->

The generated table uses only the authoritative run files for scientific metrics.
`submitted` means a recorded Slurm submission exists and production has not yet
emitted metadata; it does not assert the job's current scheduler state.
`incomplete` means production has emitted metadata without a completion or
numerical-failure record; it does not assert that a scheduler job is still running.
An `acceptance-numerical-failure` is a failure of the short qualification pilot,
reported separately from production and linked in provenance.

## Implementation and acceptance

The shared Transformer scaffold accepts `attention_coefficients: fixed` with an
explicit `attention_fixed_coefficients` vector. It reuses the fused normalized
polynomial and fused unnormalized kernels; the coefficients add no trainable
parameters. The recipe pins `attention_backend: fused` and acceptance requires it.

Every allocated hardware/runtime combination must pass the declared CPU and GPU
tests, identical backbone initialization against its exact-softmax counterpart,
two full-recipe training updates with evaluation, and a fresh-process third update
with exact checkpoint-state, next-batch, and RNG auditing. A short pilot avoids
using a long stability preselection for an experiment whose endpoint can be early
numerical failure. An invalid pilot still fails acceptance. Qualification uses the
allocated GPU type and count and is repeated on a changed allocation identity.
GPU tests compare Hermite forwards and gradients with independent/reference
computations, including signed and zero-normalizer cases.

## Reproduction and monitoring

The experiment uses branch `codex/hermite-attention` in its dedicated Git worktree.
Run these commands from that worktree root before launching, and commit the frozen
execution set before recording `LAUNCH_COMMIT`:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
JAX_PLATFORMS=cpu PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m pytest \
  -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m 'not gpu' \
  scaffolds/transformer/tests/test_fixed_polynomial.py \
  scaffolds/transformer/tests/test_fixed_polynomial_config.py \
  experiments/hermite_attention/adapter/tests
LAUNCH_COMMIT=$(git rev-parse HEAD)
PYTHONPATH=. scaffolds/transformer/.venv/bin/python \
  -m experiments.hermite_attention.adapter.analysis.report \
  --refresh-provenance --launch-commit "$LAUNCH_COMMIT"
for CONFIG in experiments/hermite_attention/configs/*.yaml; do
  RUN=$(basename "$CONFIG" .yaml)
  mkdir -p "experiments/hermite_attention/logs/$RUN"
  sbatch --gres=gpu:2 --exclude=collie,samoyed,doodle,terrier,mastiff --job-name="$RUN" \
    --output="experiments/hermite_attention/logs/$RUN/slurm-%j.out" \
    scaffolds/transformer/launchers/kennel.sbatch \
    experiments.hermite_attention.adapter.execution.acceptance \
    "$CONFIG" "$LAUNCH_COMMIT" 2 B300,H100
done
```

The shared launcher uses Kennel's low-priority `sweep` partition, append-mode logs,
requeue, and a 48-hour limit. Two B300s or H100s are requested per run; global batch
and accumulation remain fixed. The actual GPU type, count, node, runtime settings,
partition, and every Slurm attempt are recorded. See [CLUSTERS.md](../../CLUSTERS.md).
Initial jobs 202696 and 202697 failed CUDA initialization on `mastiff` with
`CUDA_ERROR_ECC_UNCORRECTABLE`, before any model test or training. Their replacements
are 202704 and 202705; all eight submissions exclude `mastiff`. These hardware
failures are recorded in provenance and are not Hermite numerical outcomes.
Do not relaunch a concluded run into the same directory. Later reproductions use a
new worktree and record with rewritten output locations.

Training mirrors metrics to the private W&B project
`rafah-massachusetts-institute-of-technology/functional-memory`, group
`hermite_attention`; the run name identifies its degree and normalization. W&B
files stay in the run's ignored `logs/` directory. `runs/<run>/metrics.jsonl`
remains canonical. To consolidate execution records and regenerate the table:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python \
  -m experiments.hermite_attention.adapter.analysis.report --refresh-provenance
```

Omit `--refresh-provenance` to regenerate the CSV and README table solely from
retained records, without accessing datasets or local logs. Add
`--hash-checkpoints` during final retention to hash every selected checkpoint file.

If a full-recipe qualification pilot stops numerically, retain its scientific
evidence only after the corresponding Slurm job has concluded:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python \
  -m experiments.hermite_attention.adapter.analysis.report \
  --retain-qualification-failure "$RUN" --failure-job-id "$JOB_ID"
```

This checks Slurm accounting and refuses a live or requeueable job. It writes one
`runs/<run>/qualification_failure.json` containing the actual pilot configuration,
failure diagnostics, last available finite measurements, source commit, and hashes
of the evidence and matching log. It does not copy per-attempt preflight files.
The outcome stays `acceptance-numerical-failure`, with no production terminal
perplexity: qualification uses the production model, data, batch, and optimizer,
but a shortened evaluation interval and one held-out batch. Missing measurements
stay absent. The matching failure log remains retained.

## Artifacts and retention

| Artifact | Location in `experiments/hermite_attention/` |
| --- | --- |
| Exact scientific recipes | `configs/` |
| Authoritative metrics, metadata, terminal or failure evidence | `runs/<run>/` |
| Derived results | `summary.csv` and the generated README block |
| Authoritative execution provenance | `provenance/manifest.json` |
| Checkpoints, ignored | `checkpoints/<run>/<step>/` |
| Slurm log, ignored | `logs/<run>/slurm-<job_id>.out` |
| Disposable acceptance and attempt outputs, ignored | `logs/<run>/acceptance/`, `logs/<run>/submissions/` |

While training, retain the latest resumable and validation-best checkpoints. After
completion, retain the validation-best checkpoint, terminal metrics independently,
and one matching successful log; concluded numerical failures retain one matching
failure log and no diagnostic checkpoint unless justified here and in provenance.
Acceptance and allocation summaries belong in the manifest. Keep the frozen
execution set unchanged while any job can run or requeue.

After all jobs conclude and results are reviewed, move retained artifacts to the
primary checkout's experiment directory, verify checksums, record final paths,
reconcile and merge the experiment branch, and only then remove the worktree and
branch. The primary checkout's Overleaf project is read-only and is not copied
into this worktree.

## Comparisons and limitations

The fixed Taylor records provide the natural degree-m comparisons:
[normalized even degrees](../key_dimension_taylor/README.md),
[normalized odd degrees](../odd_taylor_attention/README.md), and
[unnormalized degrees 1–4](../unnormalized_taylor_attention/README.md).
Their geometry, raw score definition, data, and training schedule match this sweep,
but those legacy runs used decay on all parameters. This sweep uses the current
selective decay policy. The ongoing `normalized_taylor_rerun` uses selective decay
and QK-norm cosine scores, so it also is not a matched raw-score control. A claim
that an empirical difference is caused only by the coefficients requires a matched
fixed Taylor control under selective decay and `qk_norm: false`.

There is one seed per variant. GPU types and numerical trajectories may differ;
matching initial weights and data order does not ensure bitwise-identical updates
or identical failure steps. Diagnostics sample a held-out sequence and do not
characterize every training row. No result yet establishes improved perplexity or
stability. At degree 1, normalized Hermite and Taylor are mathematically identical
because the only coefficient difference is the canceled global factor; numerical
differences can still arise from floating-point evaluation and kernel selection.
