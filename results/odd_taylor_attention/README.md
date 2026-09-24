# Fixed odd Taylor attention

Do signed fixed odd Taylor polynomials train under the same recipe as the
positive quadratic and softmax controls? This concluded collection records
**two scientific numerical failures**: degree 1 at update **9**, degree 3 at
update **17**. Neither reached the first production validation or produced a
production checkpoint. No 10B-token endpoint is assigned to either run.

<!-- BEGIN GENERATED RESULTS -->
Source: `2e0e88d9cc955b57cd07cdd40a743d2c49bafd4c`. Sweep identity: `babbaf53f23a45d5b2205a593d34f7cd395a5b9713bd624881d17ffb30a5ac1e`.

Fixed degree-one and degree-three Taylor models match the historical 200M-parameter, 10B-token seed-0 recipe. Endpoints use committed terminal validation, independently of the retained validation-best checkpoint.

| Kernel | Parameters | Status | Terminal NLL | Terminal PPL | Failure step |
|---|---:|---|---:|---:|---:|
| Fixed Taylor m=2 | 199,906,304 | complete | 2.790001 | 16.2810 |  |
| Exact softmax | 199,906,304 | complete | 2.687451 | 14.6942 |  |
| Fixed Taylor m=1 | 199,906,304 | numerical-failure | — | — | 9 |
| Fixed Taylor m=3 | 199,906,304 | numerical-failure | — | — | 17 |

Fixed coefficients: p₁(s)=1+s; p₃(s)=1+s+s²/2+s³/6. Scores are unshifted after RoPE and weights use the signed row sum. Negative weights and negative nonzero denominators are allowed; zero/nonfinite denominators and nonfinite computations fail without epsilon, clipping, or absolute-value normalization.

Each model uses the full 19,074-update learning-rate schedule throughout a 5,000-update stability stage and subsequent continuation. Continuation requires finite committed validation and exact checkpoint, next-batch, and RNG restoration. One seed per variant does not estimate seed variation.

The stability plots use the fixed first held-out diagnostic sequence. Negative-weight fractions are computed after signed normalization; minimum absolute denominators and cancellation ratios expose near-zero signed sums.

Odd-degree runs use four H100s on dogo. Historical control throughput reflects its recorded allocations, including hardware migration, and is not a matched hardware benchmark. Original hardware histories remain in control provenance.

Historical controls retain original source revisions and artifact SHA256 checksums in the canonical provenance; the archived launch record preserves the original references:

- Fixed Taylor m=2: `1a8836a75836671cdf83a3ab7763425a6a88cac7`; original sweep `6bae4c76b615f8f1385b608d750948799b9e00c0f2166073009a9efc70cc814a`.
- Exact softmax: `1a8836a75836671cdf83a3ab7763425a6a88cac7`; original sweep `6bae4c76b615f8f1385b608d750948799b9e00c0f2166073009a9efc70cc814a`.

Recorded failures:

- Fixed Taylor m=1: FloatingPointError: Nonfinite training loss or signed-Taylor update at step 9: nan
- Fixed Taylor m=3: FloatingPointError: Nonfinite training loss or signed-Taylor update at step 17: nan

[Metrics](summary.csv) · [Terminal metrics](figures/terminal_metrics.pdf) · [Learning curves](figures/learning_curves.pdf) · [Stability](figures/stability.pdf) · [Throughput](figures/throughput.pdf)
<!-- END GENERATED RESULTS -->

## Signed attention and staged protocol

The fixed polynomials are `p1(s)=1+s` and `p3(s)=1+s+s^2/2+s^3/6`, with no
learned coefficient parameters. Both retain the 199,906,304-parameter backbone.
Scores stay unshifted after RoPE, and normalization uses the signed final row
sum. Negative normalized weights and negative nonzero denominators are allowed;
zero/nonfinite denominators and nonfinite computations fail explicitly. There
is no denominator epsilon, clipping, or absolute-value normalization. The
production implementation first computes final row sums, then recomputes tiles
to accumulate normalized weighted values. Zero/negative intermediate tile sums
are valid and are not mistaken for an invalid final denominator.

Production planned a 5,000-update stability stage using the full unchanged
19,074-update learning-rate schedule. Finite committed step-5000 validation and
exact restoration of parameters, optimizer, next input and RNG would authorize
continuation in the same allocation. The gate imposes no additional loss threshold
or scientific hyperparameter. Both variants failed before this stage could
finish, and neither continuation was run. The production metadata counts the
attempted failing update; finite updates completed were 8 and 16.

## Model, data and optimization

The fixed backbone has 10 decoder layers, hidden width 1024, 16 full-MHA heads,
query/key and value head widths 64, SwiGLU width 3008, untied 32,000-token
embeddings, RoPE base 10,000, RMSNorm epsilon 1e-6 and no dropout. Computation
is BF16 with FP32 parameters, polynomial arithmetic and optimizer state.

Every variant starts from the same backbone initialization and seed 0, reads
the same packed FineWeb order, and targets 19,074 optimizer updates:
**10,000,269,312 prediction tokens**. Global batch 256, global microbatch 64,
four accumulation steps and 2,048 prediction positions remain fixed. AdamW
uses learning rates 3e-4 to 3e-5, 2,000-update warmup then cosine decay, betas
(0.9,0.95), weight decay 0.1 on backbone parameters, and global gradient
clipping at 1 after accumulation and device averaging. The terminal endpoint
is held-out mean token NLL and its exponential perplexity. Evaluation uses
the same first 1,792 held-out sequences (3,670,016 prediction tokens) every
1,000 updates and at completion. Checkpoints were saved every 500 updates
and at validation. Failed runs have no imputed terminal endpoint.

Training reads `/data/cl/u/rafah/home/datasets/fineweb_12b` directly on NFS.
The train/eval manifests, tokenizer identity and shard checksums are recorded
in canonical provenance. No node-local staging was required. The original
fixed quadratic and softmax controls remain the independently retained
degree sweep, now retained in the [dK × Taylor-degree record](../key_dimension_taylor/README.md); they are referenced,
not copied or relabeled. In particular, softmax's recorded H100/B300 history
means control throughput is not a matched-hardware comparison.

## Preflight, production evidence and limits

Preflight **200224** passed separate ordinary CPU, two-device CPU and fused GPU
gates, initialization identity, full-context/microbatch two-update pilots and
fresh-process continuation to step 3. Each gate rejected skipped tests; resume
checks compared exact live/durable/restored state and actual next batch/RNG.
Runtime estimates allowed evaluation/checkpointing, three startup allowances
and 10% headroom. Shared infrastructure/correctness failures blocked production;
degree-specific numerical failures did not suppress the other degree.

Earlier full validation recorded **324 CPU**, **7 distributed**, and **18 GPU**
passes; all seven cases skipped without multiple CPU devices passed separately.
GPU validation **200214** included FP64 Q/K/V-gradient references and production
shape. The final preflight routing fix passed **46 scaffold tests**, without
changing attention/training arithmetic. These tests qualify the implementation
and restart checks, not stability throughout optimization.

Original production failures occurred in jobs **200230** and **200231**, after
48 and 51 seconds of allocation. Their numerical-failure records are the direct
evidence; wrapper exit errors are not the mechanism. Step-2 held-out pilot
diagnostics found maximum cancellation ratios **16,681.3** (degree 1) and
**693,707.1** (degree 3), with minimum absolute row sums **0.0060674** and
**0.00160217**. These sampled diagnostics came from one held-out sequence, not
the failing production batch. The structured failure analysis (in the manifest's `retired_records`)
and the pilot metric records in `runs/<run>/pilot-metrics.jsonl` preserve this distinction.

Negative-weight fractions are measured after signed normalization. The sampled
held-out diagnostics and one seed per degree do not establish a failure-time
distribution, and comparisons to historical controls are not matched throughput
benchmarks. No completed or failed production checkpoint exists to retain.

## Diagnostic replay of the failure mechanism

The following forensic check was performed on 2026-09-20 from the recorded
source, with temporary instrumentation. It was a diagnostic replay, not a
replacement production result or a change to the scientific recipe.

### Finding

Both diagnostic replays encountered the same concrete failure mechanism: signed polynomial contributions nearly cancelled, and the production FP32 arithmetic produced an exactly zero final row denominator. The existing zero-denominator guard then emitted NaNs. Q/K/V and all earlier traced operations were finite. Independent FP64 and exact rational calculations on the captured BF16 inputs show that both mathematical denominators were small but nonzero.

Higher precision avoids the immediate zero-denominator failure on these captured rows, but their normalized outputs remain enormous. Precision alone therefore does not remove the underlying sensitivity. This diagnostic did not test a stabilization method or change epsilon, clipping, normalization, scores, coefficients, optimizer, or learning-rate schedule.

### Observed events

All layer, head, token, microbatch, device, and sequence indices below are zero-based.

| Measurement | Fixed Taylor m=1 | Fixed Taylor m=3 |
|---|---:|---:|
| Original production failure update | 9 | 17 |
| Diagnostic replay failure update | 15 | 135 |
| Last finite replay update | 14 | 134 |
| First invalid traced operation | layer 6 attention denominator | layer 9 attention denominator |
| Head / query index | 8 / 153 | 4 / 1 |
| Causal keys in offending row | 154 | 2 |
| Accumulation index / original device index | 2 / 1 | 1 / 3 |
| Sequence within original per-device microbatch | 3 | 15 |
| Production FP32 final denominator | 0 | 0 |
| FP64 denominator | −7.145572453737259e−6 | −3.252949943854944e−7 |
| Exact-rational denominator, decimal | −7.145572453737259e−6 | −3.252949946045591e−7 |
| Sum of absolute polynomial contributions | 103.6574661 | 0.2743532465 |
| Cancellation ratio, exact-rational inputs | 14,506,530.69 | 843,398.30 |
| Largest absolute normalized output component, exact ratio | 6,327,706.91 | 889,521.29 |
| Nonfinite input Q/K/V entries in captured row | 0 | 0 |
| Last finite global gradient norm, before clipping | 1,304,696 | 49,419.10 |
| Last finite training NLL | 10.886187 | 9.513035 |
| Learning rate scheduled at failing update | 2.100016e−6 | 2.010001e−5 |

Each captured attention invocation contains 524,288 rows (16 sequences × 2,048 queries × 16 heads). Exactly one row had a zero raw denominator and nonfinite production output; no row had a nonfinite raw denominator. The temporary denominator probe matched the original saved denominator bit-for-bit on all 524,287 valid rows. The offending row's only diagnostic flag was **zero final denominator**; score, polynomial, and intermediate accumulated-denominator nonfinite flags were absent.

A separate original-model forward screen found exactly one invalid per-device microbatch out of the 16 in each failing update. The detailed instrumented forward also reproduced an invalid loss in that microbatch. These locations describe the diagnostic replays, not the unsaved historical failure states.

### Exact arithmetic verification

For m=1, the 154 polynomial values comprise 72 positive and 82 negative values, with none individually zero. Their sums are approximately +51.82872947724536 and −51.828736622817814. The exact denominator is:

`Z = -15345 / 2147483648`.

Output channel 50 has the exact nonzero numerator `12428628812247 / 274877906944`, approximately 45.21508822, giving output −6,327,706.914. This is cancellation across many keys, not a singleton 0/0 with a removable singularity.

For m=3, the two exact-input scores are approximately −1.781761884689331 and −1.3747681975364685. Their polynomial values are approximately −0.13717678589806498 and +0.13717646060307037. The exact denominator is:

`Z = -9216973077394991 / 28334198897217871282176`.

Output channel 45 has the exact nonzero numerator `1049432802371421105667909 / 3626777458843887524118528`, approximately 0.2893568228, giving output −889,521.2886. This is also cancellation between distinct keys. The independent NumPy FP64 denominator differs from the exact rational value by only 2.19e−16.

Exact verification converted the stored finite BF16 Q/K/V values to rational numbers, evaluated dot products divided by 8 (head width 64), then evaluated the polynomial and its normalization without floating-point rounding. The production zero cannot be attributed to a true zero of these exact-input row sums. The check does not isolate how much rounding came from score computation, polynomial evaluation, or tiled summation.

For additional scale, the local attention Jacobian with respect to the row's scores satisfies `dY/ds_j = p'(s_j) (V_j - Y) / Z`. Its largest absolute entry is approximately 8.86e11 for m=1 and 2.20e12 for m=3; operator norms are approximately 3.64e13 and 8.37e12. These local sensitivities have different units from the full model's parameter-gradient norm and should not be directly compared numerically.

### Method and limits

- Used a separate temporary checkout and branch based on frozen source `2e0e88d9cc955b57cd07cdd40a743d2c49bafd4c`. No tracked source files changed. Original resolved configurations were used with only the output directory redirected to scratch.
- Replayed original `create_state`, packed-data iterator, and compiled four-device training step. The model, initialization recipe, seed 0, BF16 compute, unshifted scores, 256-sequence global batch, 64-sequence global microbatch, four accumulation steps, AdamW, global clipping at 1, and full 19,074-update schedule were preserved. Each device still processed 16 sequences of 2,048 prediction tokens.
- Before each donated training update, copied model parameters to host. On the first invalid checked loss, saved those preceding finite parameters and the exact failing batch. The normal training computation was unmodified; failed parameters were not used for localization.
- Screened all 16 original microbatch/device pieces with the original model. A separate instrumented forward recorded operation finiteness and captured post-RoPE, cast Q/K/V. A temporary Pallas first-pass probe recovered raw FP32 denominators before the original kernel's NaN sentinel. Selected causal rows were recomputed independently in NumPy FP64 and exact rational arithmetic.
- The probe passed CPU interpreter and real-GPU checks for both degrees on constructed exact-zero denominators, including agreement with original valid denominators. Actual captured inputs, not reprojected high-precision parameters, were used for the precision comparison.
- The initial cubic replay remained finite for 32 updates. A fresh cubic replay with a 256-update ceiling failed at 135. Replays encountering invalid updates stopped at their first failure; the initial cubic replay reached its 32-update ceiling without failing. No terminal held-out metrics or production checkpoints were generated by this check.
- The original failures at 9 and 17 were **not reconstructed bit-for-bit**. The first cubic diagnostic and its extended rerun also followed different numerical trajectories. Fresh GPU compilations/execution and instrumented forwards can differ in rounding; this check does not isolate the source of trajectory differences. Source, seed, and data order matching do not establish bitwise replay. These results establish the failure mechanism in new replays, not the exact offending historical rows or a failure-time distribution.
- Instrumentation can alter compiler fusion. Both the ordinary forward screen and the instrumented forward were invalid on the selected shard, and the probed original fused kernel reproduced the invalid row. The report does not claim every intermediate is bitwise identical to the original joint forward/backward graph.
- No stabilization or higher-precision full-training experiment was run. A finite FP64 row output is not evidence that an FP64 model would train stably. Gradient clipping occurs after the forward computation and cannot prevent this zero-denominator event.


Successful diagnostic jobs **200497** (7m05s, degree 1 and initial degree-3
screen) and **200498** (2m19s, extended degree 3) are retained at
`logs/odd_taylor_attention/diagnostic/m1/slurm-200497.out` and
`logs/odd_taylor_attention/diagnostic/m3/slurm-200498.out`. Their success
means invalid training updates were captured and explained, not that training
completed. Replay provenance, folded into the manifest's `retired_records`, records exact
source/input hashes, retained log checksums, and hashes of removed scratch
evidence. The former diagnostic log paths in that historical JSON are preserved;
canonical log locations are in the current manifest.

The diagnostic checkout/branch, temporary scripts, copied parameters, batches,
row arrays, JSON scratch and compilation caches were removed after the original
analysis. They are not claimed as retained artifacts or reproducible installed
diagnostic code. This README consolidates the previous temporary narrative;
the precise measurements, limitations and provenance remain, along with its
successful logs. Reproducing those exact captured rows is not possible from
the retained files alone; the historical production experiment can be rerun.

## Provenance, artifact locations and retention

The home cluster is **Kennel** and the durable artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`. The authoritative [manifest](provenance/manifest.json) records
source `2e0e88d9cc955b57cd07cdd40a743d2c49bafd4c`, seeds, exact configuration and metric hashes,
data identity, runtime hardware, selected checkpoints and successful training/validation logs and indispensable numerical-failure logs.
The original launch record is summarized in the manifest's `retired_records`,
with its SHA256 and a commit containing the exact file, including its historical
control hashes and old absolute paths. Exact executed recipes and their historical dependency lock also keep
their original bytes; their old workdirs are historical, not current output
instructions. The current shared environment uses the scaffold's own lock.

| Artifact | Repository-relative location |
| --- | --- |
| Exact recipes and historical lock | `results/odd_taylor_attention/configs/` |
| Committed metrics, metadata and failure/completion evidence | `results/odd_taylor_attention/runs/<run>/` |
| Figures | `results/odd_taylor_attention/figures/` |
| Canonical manifest: pre-launch acceptance, execution and folded diagnostics | `results/odd_taylor_attention/provenance/manifest.json` |
| Retained checkpoint, when one exists | `checkpoints/odd_taylor_attention/<run>/<step>/` |
| Matching Slurm logs | `logs/odd_taylor_attention/.../slurm-<job_id>.out` |

A Git clone shares recipes, scientific evidence and shared code. It does not
include checkpoint tensors, datasets, environments or Slurm logs, which remain
ignored local artifacts on Kennel. Scientific failure logs are indispensable
evidence of the concluded outcomes; they are deliberately retained and are
not described as successful training logs. Failed development attempts and
redundant reports, caches and intermediate checkpoints are removed after their
replacement validation and retained artifact inventory have been verified.

## Reproduction and plotting

Consult [CLUSTERS.md](../../CLUSTERS.md) before selecting an allocation or
moving artifacts. This collection used four H100 SXM GPUs on `dogo`, 32 CPUs
and 256 GB memory, with no transport override recorded. The historical
`standard` profile allowed two hours for preflight and eight hours per training
job. Current availability and policy must be checked; a layout migration does
not qualify a new runtime or GPU combination.

To regenerate the report from a repository clone, without datasets or GPUs:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.envs/transformer"
uv sync --project scaffolds/transformer --locked --extra cuda
uv run --project scaffolds/transformer --locked --extra cuda python -m \
  scaffolds.transformer.studies.odd_taylor_attention.plot \
  --manifest results/odd_taylor_attention/provenance/manifest.json
```

The plotter verifies retained hashes and semantic identities, resolves paths
from the manifest's repository, and updates only the marked generated-results
section of this README. Checkpoint files and old execution checkouts are not
required. Completed controls retain their own source commits and terminal
metrics; incomplete validation histories are plotted without a terminal bar.

For an exact replay of the historical implementation, create a fresh branch and
separate checkout at the recorded source, keep the retained root outside it,
and install that source's historical locked environment. From the main checkout:

```bash
git worktree add -b codex/reproduce-odd_taylor_attention /path/to/fresh-checkout 2e0e88d9cc955b57cd07cdd40a743d2c49bafd4c
```

Inside the fresh checkout, with the recorded FineWeb corpus available and a
four-H100 Slurm allocation, choose one immutable recipe from this collection.
Set `REPLAY_CONFIG` to its absolute path in the retained checkout. Set
`REPLAY_OUTPUT` and `REPLAY_CHECKPOINTS` to fresh durable paths outside the
execution checkout. The guards prevent overwriting either directory:

```bash
export UV_PROJECT_ENVIRONMENT=/path/to/fresh-historical-environment
uv sync --project experiments/polynomial_attention --locked --extra cuda
test ! -e "$REPLAY_OUTPUT" && test ! -e "$REPLAY_CHECKPOINTS" && \
  mkdir -p "$REPLAY_OUTPUT" "$REPLAY_CHECKPOINTS" && \
  ln -s "$REPLAY_CHECKPOINTS" "$REPLAY_OUTPUT/checkpoints" && \
  uv run --project experiments/polynomial_attention --locked --extra cuda python -m \
    experiments.odd_taylor_attention.train "$REPLAY_CONFIG" --workdir "$REPLAY_OUTPUT" --no-resume
```

Those `experiments.*` commands exist at the recorded historical source. Current
code lives exclusively under `scaffolds/transformer`; for a new experiment
using current code, copy the recipe and explicitly override its historical
source and artifact paths in that copy (keep all other recipe sections):

```yaml
train:
  source_commit: auto
  workdir: /durable/repository/results/FRESH_EXPERIMENT/runs/FRESH_RUN
  checkpoint_dir: /durable/repository/checkpoints/FRESH_EXPERIMENT/FRESH_RUN
```

`auto` records the current frozen checkout commit; never leave the historical
source hash in a new-code reproduction. Use fresh output directories, then
invoke `python -m scaffolds.transformer.train CONFIG.yaml` inside the validated
allocation. The original retained YAML remains byte-identical.
Study launchers under `scaffolds.transformer.studies.odd_taylor_attention` provide
preflight and production gating. Run `--help`/`--dry-run` first and freeze the
tested source, environment and manifests while jobs can requeue. During an
active new run keep both resumable latest and validation-best checkpoints.

Historical source, seed and input order do not guarantee bitwise independent
GPU trajectories or identical numerical-failure steps. Original runtime
settings and hardware belong in execution provenance, separate from scientific
hyperparameters. No jobs are launched by this consolidation.
