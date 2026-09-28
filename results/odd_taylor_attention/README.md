# Fixed odd Taylor attention

[Selected runs and source hashes](selection.yaml) · [Run index](../../experiments/attention_scaling/runs.csv) · [Execution guide](../../experiments/attention_scaling/README.md)

Do signed fixed odd Taylor polynomials train under the repository's 200M recipe?
Under the scaffold's current defaults (per-head QK-norm cosine scores and
selective weight decay), this concluded record holds **two scientific numerical
failures**: degree 1 at update **31** and degree 3 at update **419**. Neither
reached the first validation at update 1,000 or produced a checkpoint, and no
10B-token endpoint is assigned to either run.

**Home cluster: Kennel.** The artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`; the
[run records](../../experiments/attention_scaling/runs/) are authoritative for
each run's source, recipe, acceptance, attempts, and artifacts.

## Results

<!-- BEGIN GENERATED RESULTS -->
Fixed degree-one and degree-three Taylor models use the dK × Taylor-degree record's 200M-parameter, 10B-token, seed-0 recipe at d_K = 64 under the scaffold's current defaults (QK-norm and selective weight decay). Endpoints use committed terminal validation, independently of the retained validation-best checkpoint.

| Kernel | Parameters | Status | Terminal NLL | Terminal PPL | Failure step |
|---|---:|---|---:|---:|---:|
| Fixed Taylor m=2 | 199,906,464 | complete | 2.804009 | 16.5107 |  |
| Exact softmax | 199,906,304 | complete | 2.685498 | 14.6655 |  |
| Fixed Taylor m=1 | 199,906,464 | numerical-failure | — | — | 31 |
| Fixed Taylor m=3 | 199,906,464 | numerical-failure | — | — | 419 |

Fixed coefficients: p₁(s)=1+s; p₃(s)=1+s+s²/2+s³/6. Scores are QK-norm cosine scores s = exp(γ)·cos(q, k), with γ learned per layer and head from log 8, so |s| ≤ exp(γ); weights use the signed row sum. Negative weights and negative nonzero denominators are allowed; zero/nonfinite denominators and nonfinite computations fail without epsilon, clipping, or absolute-value normalization.

Each run follows the full 19,074-update schedule from initialization under the shared launcher, which checkpoints every 500 updates and requeues after preemption. One seed per variant does not estimate seed variation.

The stability plots use the fixed first held-out diagnostic sequence. Both odd-degree runs failed before their first validation at update 1,000, so only the controls have validation and diagnostic curves.

Both odd-degree runs failed on four B300s on akita, after earlier attempts were preempted during acceptance. Control throughput reflects its recorded allocations and is not a matched hardware benchmark. Hardware histories are in the run records.

The controls are retained in the [dK × Taylor-degree record](../key_dimension_taylor/README.md); their source commits and artifact checksums are in their run records:

- Fixed Taylor m=2: `a74e3863fdf1e1dd965252cd435712dec10c9a29`; fixed Taylor m=2 with QK-norm and selective weight decay.
- Exact softmax: `381f1fdf6caf2925a984e3932e58b6e70bfbc6b1`; softmax with selective weight decay and no QK-norm.

Recorded failures:

- Fixed Taylor m=1: FloatingPointError: Nonfinite training loss or optimizer update at step 31: nan
- Fixed Taylor m=3: FloatingPointError: Nonfinite training loss or optimizer update at step 419: nan

[Metrics](summary.csv) · [Terminal metrics](terminal_metrics.pdf) · [Learning curves](learning_curves.pdf) · [Stability](stability.pdf) · [Throughput](throughput.pdf)
<!-- END GENERATED RESULTS -->

## Signed attention over bounded scores

The fixed polynomials are `p1(s) = 1 + s` and `p3(s) = 1 + s + s^2/2 + s^3/6`,
with no learned coefficients. The scores are QK-norm cosine scores
`s = exp(gamma) · cos(q, k)`, with `gamma` learned per layer and head from
`log 8`, so every score lies in `[-exp(gamma), exp(gamma)]`. Both models have
199,906,464 parameters: the 199,906,304-parameter backbone plus 160 QK-norm scales.
Odd polynomials take negative values: `p1(s) < 0` for `s < -1` and `p3(s) < 0`
below its real root near `-1.596`, both inside the reachable score range from
initialization. Normalization uses the signed final row sum. Negative weights and
negative nonzero denominators are allowed; zero or nonfinite denominators and
nonfinite computations fail explicitly, with no denominator epsilon, clipping, or
absolute-value normalization. The fused signed kernel computes the final row sums
first and then recomputes tiles to accumulate the normalized values.

## Outcome

- **Degree 1** stopped at update 31 on a NaN forward loss, gradient norm, and
  parameter update. Its training NLL was still 10.874 at update 30, close to its
  initial value.
- **Degree 3** stopped at update 419 in the same way, with training NLL 7.471 at
  update 410.

The failure records were written after the attempted update, so their attention
diagnostics are nonfinite everywhere and do not isolate the mechanism, and no
validation preceded either failure. The signed row sum is the likely cause: odd
polynomials are negative for low scores, so a row's signed contributions can cancel
to zero. Bounding the scores with QK-norm does not prevent this.

## Model, data and optimization

The backbone has 10 decoder layers, hidden width 1024, 16 full-MHA heads,
query/key and value head widths 64, SwiGLU width 3008, untied 32,000-token
embeddings, RoPE base 10,000, RMSNorm epsilon 1e-6, and no dropout. Computation is
BF16 with FP32 parameters, polynomial arithmetic, and optimizer state. Every run
starts from seed 0 and the same packed FineWeb order, targeting 19,074 updates
(**10,000,269,312 prediction tokens**) at global batch 256 with microbatch 64 and
four accumulation steps. AdamW uses learning rates `3e-4` to `3e-5`, 2,000 warmup
updates then cosine decay, betas (0.9, 0.95), weight decay 0.1 on every parameter
except the token embedding, RMSNorm scales, biases, and QK-norm scales, and global
clipping at 1. Evaluation would use the first 1,792 held-out sequences every 1,000
updates; checkpoints are written every 500 updates.

## Execution and provenance

Both runs were launched on 2026-09-26 from the normalized Taylor rerun's worktree
(source `a74e386`, which also produced the `d_K = 64` even degrees in the
[dK × Taylor-degree record](../key_dimension_taylor/README.md)) with the shared
Kennel launcher. The launcher checked the frozen execution set and ran acceptance
in each allocation: CPU and distributed CPU tests, the fused, signed odd, and
QK-norm GPU tests, initialization identity, a 20-update pilot, and an exact resume
audit. Several early attempts were preempted during acceptance by debug-partition
jobs. Each run then passed acceptance on four B300s on `akita` and trained from
initialization until its failure (jobs 202614 and 202615). The run records record
every attempt, partition change, and acceptance summary; it also records each
run's executed recipe at the launch commit, since this record's recipes differ
only in output paths and the named source commit.

| Artifact | Location in `experiments/attention_scaling/` |
| --- | --- |
| Exact recipes | `experiments/attention_scaling/runs/<run_id>/config.yaml` |
| Committed metadata and numerical-failure records | `runs/<run_id>/` |
| Figures and summary | `results/odd_taylor_attention/` |
| Canonical manifest | `experiments/attention_scaling/sweep.yaml` |
| One failure log per run (ignored) | `runs/<run_id>/training.log` |

No checkpoint is retained, because neither run reached a validation. The failure
logs are indispensable evidence of the concluded outcomes and are not described
as successful training logs.

## Reproduction

Regenerate the summary, figures, and results block from the committed runs; no
GPU, dataset, or checkpoint is needed:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m results.odd_taylor_attention.plot
```

The plotter verifies retained hashes and semantic identities against the pinned selection and each run’s recipe
and rewrites only the marked results block.

Use the [execution guide](../../experiments/attention_scaling/README.md) to create and submit a fresh run.

Source, seed, and input order do not guarantee bitwise-identical GPU trajectories
or identical failure steps.

## Limitations

One seed per degree, on one hardware type. Both runs failed before any
validation, so there are no diagnostics at a valid state close to the failure. The
controls in the table differ from these runs in degree (m=2) or mechanism
(softmax, which uses selective decay without QK-norm).
