# Learned quadratic attention

Can learning quadratic attention coefficients improve the fixed Taylor kernel,
and does an explicitly positive parameterization avoid signed-normalization
failures? This concluded collection has **two completed positive runs and two
scientific numerical failures**. The failures are findings of the requested
recipe, not unfinished jobs waiting to resume.

<!-- BEGIN GENERATED RESULTS -->
Source: `2358ccf2f194f4ed39fecb5bfbae45f42a72fdfb`. Sweep identity: `833c751215987cf19038b2164012038e4720f089894c43846c40814579ca5d48`.

Four learned variants match the degree sweep's 200M-parameter, 10B-token seed-0 recipe. Endpoints use committed terminal validation, independently of the retained validation-best checkpoint.

| Kernel | Parameters | Status | Terminal NLL | Terminal PPL |
|---|---:|---|---:|---:|
| Fixed Taylor quadratic | 199,906,464 | complete | 2.804009 | 16.5107 |
| Exact softmax | 199,906,304 | complete | 2.685498 | 14.6655 |
| Positive · global | 199,906,306 | complete | 2.784334 | 16.1890 |
| Positive · per head/layer | 199,906,624 | complete | 2.785781 | 16.2125 |
| Unrestricted · global | 199,906,306 | numerical-failure | — | — |
| Unrestricted · per head/layer | 199,906,624 | numerical-failure | — | — |

Positive: p(s)=0.5[1+(as+b)²], initialized at a=b=1. Unrestricted: p(s)=1+bs+cs², initialized at b=1,c=0.5. Coefficients are FP32 without weight decay; global sharing adds 2 parameters and per-head/per-layer sharing adds 320.

Coefficient magnitudes must be interpreted together with score scales: Q/K rescaling can absorb some coefficient movement. The constant scale of a normalized polynomial is unidentifiable. Positive and unrestricted families use different optimization coordinates, so equal learning rates do not isolate positivity alone. One seed per variant does not estimate seed variation.

The stability plots use the fixed first held-out diagnostic sequence. Negative-weight fractions are computed after signed normalization; minimum absolute denominators and cancellation ratios expose near-zero signed sums.

Learned runs use four H100s on dogo. Control throughput reflects each control's recorded allocations, including any hardware migration; it is not a matched hardware benchmark. Hardware histories are preserved in the control provenance.

The controls are retained in the [dK × Taylor-degree record](../key_dimension_taylor/README.md); their source commits and artifact checksums are in the provenance manifest:

- Fixed Taylor quadratic: `a74e3863fdf1e1dd965252cd435712dec10c9a29`; fixed Taylor m=2 under the current defaults (QK-norm and selective weight decay), source a74e386.
- Exact softmax: `381f1fdf6caf2925a984e3932e58b6e70bfbc6b1`; softmax baseline retrained 2026-09-25 under the selective weight-decay policy.

Recorded failures:

- Unrestricted · global: FloatingPointError: Nonfinite training loss or learned-quadratic update at step 2324: nan
- Unrestricted · per head/layer: FloatingPointError: Nonfinite training loss or learned-quadratic update at step 4187: nan

[Metrics](summary.csv) · [Terminal metrics](figures/terminal_metrics.pdf) · [Learning curves](figures/learning_curves.pdf) · [Coefficients](figures/coefficient_trajectories.pdf) · [Stability](figures/stability.pdf) · [Throughput](figures/throughput.pdf)
<!-- END GENERATED RESULTS -->

## Learned kernel and findings

| Family | Polynomial | Initial parameters | Sharing | Added parameters |
| --- | --- | --- | --- | ---: |
| Positive | `0.5 * (1 + (a*s+b)^2)` | `a=b=1` | Global | 2 |
| Positive | `0.5 * (1 + (a*s+b)^2)` | `a=b=1` | Per head and layer | 320 |
| Unrestricted | `1+b*s+c*s^2` | `b=1,c=0.5` | Global | 2 |
| Unrestricted | `1+b*s+c*s^2` | `b=1,c=0.5` | Per head and layer | 320 |

All initialize to the original fixed Taylor quadratic. Coefficients are FP32,
use the backbone learning-rate schedule and global clipping, and receive **no
weight decay**. The global models have 199,906,306 parameters; head/layer
models have 199,906,624. Scores remain unshifted after RoPE. Unrestricted
kernels permit negative normalized weights and negative nonzero denominators.
Zero/nonfinite denominators or nonfinite computations fail explicitly, without
clipping, epsilon, or absolute-value normalization.

Positive/global and positive/per-head complete with perplexities **16.1890345**
and **16.2124774**. Against the fixed Taylor quadratic trained in the same setup,
the positive family improved slightly; that control has since been retrained under
the scaffold's current defaults (QK-norm and selective weight decay) and replaced,
so the table's fixed-quadratic value (16.5107) is not a matched comparison. Exact
softmax, the selective-decay baseline without QK-norm, reads **14.6655**. These
single-seed observations do not establish a robust ordering. Unrestricted
global and per-head runs terminate at updates **2324** and **4187**. Their last
committed validations are at updates 2000 and 4000; these are partial-budget
observations, not terminal 10B-token results.

At completion, positive/global has `a=2.8825173378`, `b=0.9651518464`.
For head/layer sharing, `a` has mean 1.99759, standard deviation 0.67644 and
range 0.84989–3.71649; `b` has mean 1.05346, standard deviation 0.15129 and
range 0.46925–1.31322. The [per-head figure](figures/per_head_coefficients.pdf), regenerated from the terminal
metrics record, supplements raw/effective
coefficient histories. Head indices apply only within this trained model.
Raw coefficients must be interpreted with score scales: Q/K rescaling can
absorb some coefficient changes, and overall polynomial scale cancels in
normalized attention. Diagnostic records do not contain per-head score moments.
Positive and unrestricted families also use different optimization coordinates,
so matching learning rates does not isolate positivity alone.

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
in canonical provenance. No node-local staging was required. The fixed
quadratic and softmax controls are retained in the
[dK × Taylor-degree record](../key_dimension_taylor/README.md) and referenced here, not
copied. Both were retrained after these runs: softmax under selective weight decay,
and the fixed quadratic under QK-norm and selective weight decay. Their recorded
hardware histories also differ, so control throughput is not a matched-hardware
comparison.

## Validation, resume evidence and failure retention

Production preflight **199894** passed CPU reference/gradient/optimizer checks,
fused GPU parity, full-backbone initialization identity against the fixed control,
coefficient initial values/counts, full-context/full-microbatch pilots and exact
state/input/RNG restart checks for all four variants. Runtime estimates included
evaluation/checkpoint overhead and 10% headroom against eight hours. Numerical
or runtime pilot failure would reject only that variant; infrastructure or
shared-correctness failure would block the whole sweep.

An earlier pilot compared independently generated GPU trajectories and failed
before the eventual successful preflight. Its step-2 states already differed
before restart: 260/285 leaves, maximum parameter difference 1.527369e-7, with
finite NLLs 10.8814556599 and 10.8814499378. Step 3 differed in 276 leaves,
maximum parameter difference 3.455207e-7. Coefficients were bitwise equal while
optimizer moments differed. The nondeterministic primitive was not conclusively
identified. The diagnosis, folded into the
[provenance manifest](provenance/manifest.json) (`execution_evidence`), preserves those measurements. The replacement gate checks exact live state, durable checkpoint,
fresh-process restored parameters/optimizer/update count, and next batch/RNG;
CPU tests separately require uninterrupted/resumed trajectory identity. It
uses no floating-point tolerance and does not change production arithmetic.

Supporting reports retain 240 CPU passes, five separately passed distributed
checks, 46 revised resume-audit passes and 27 final scaffold passes, with source
hashes and the successful preflight evidence. The five distributed cases skipped
in a single-device run were exercised in their separate two-device gate.

The two completed runs each keep their validation-best checkpoint at **19074**.
For each concluded unrestricted failure, retain only its best finite checkpoint
at **2000** as a documented diagnostic/reuse exception. Per-head step 4000 was
the latest resumable state but is not best; it can be removed only after the
selected step-2000 inventory is verified at its durable destination. No resumed
training is planned for these concluded failures. Preserve their failure JSON,
partial metrics and matching failure logs because these are scientific evidence.

Successful production allocations were **2h50m20s** and **2h50m24s**; failed
unrestricted allocations lasted **21m07s** and **37m28s**. These include their
recorded Slurm wall times and are not resident-input kernel benchmarks.

## Provenance, artifact locations and retention

The home cluster is **Kennel**. The authoritative [manifest](provenance/manifest.json)
records source `2358ccf2f194f4ed39fecb5bfbae45f42a72fdfb`, seeds, exact configuration and
metric hashes, data identity, runtime hardware, pre-launch acceptance, selected
checkpoints, and the successful and numerical-failure logs. The executed code and
dependency lock are at that source commit.

| Artifact | Location in `experiments/learned_quadratic_attention/` |
| --- | --- |
| Exact recipes | `configs/` |
| Committed metrics, metadata and failure/completion evidence | `runs/<run>/` |
| Figures and summary | `figures/`, `summary.csv` |
| Canonical manifest | `provenance/manifest.json` |
| Retained checkpoint, when one exists (ignored) | `checkpoints/<run>/<step>/` |
| Matching Slurm logs (ignored) | `logs/<run>/slurm-<job_id>.out`, `logs/preflight/` |

A Git clone shares recipes, scientific evidence and shared code. It does not
include checkpoint tensors, datasets, environments or Slurm logs, which remain
ignored local artifacts on Kennel. Scientific failure logs are indispensable
evidence of the concluded outcomes; they are deliberately retained and are
not described as successful training logs.

## Reproduction and plotting

Regenerate the summary, figures, and results block from the committed runs; no GPU,
dataset, or checkpoint is needed:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m experiments.learned_quadratic_attention.adapter.analysis.plot
```

The plotter verifies retained hashes and semantic identities against the manifest
and updates only the marked generated-results section of this README. Completed
controls keep their own source commits and terminal metrics; incomplete validation
histories are plotted without a terminal bar.

This collection used four H100 SXM GPUs on `dogo`.

A rerun with the current scaffold is a new experiment with its own worktree, branch,
and record (see [AGENTS.md](../../AGENTS.md)); never relaunch into this record's `runs/`.
Copy the recipes from `configs/` with outputs under `experiments/<new>/` and
`train.source_commit: auto`, give the new adapter an acceptance spec like
[this one's](adapter/execution/acceptance.py), then submit each variant from the worktree root:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
mkdir -p "experiments/<new>/logs/$RUN"
sbatch --job-name="$RUN" --output="experiments/<new>/logs/$RUN/slurm-%j.out" \
  scaffolds/transformer/launchers/kennel.sbatch experiments.<new>.adapter.execution.acceptance \
  "experiments/<new>/configs/$RUN.yaml" "$(git rev-parse HEAD)" 4 H100
```

The launcher runs the shared acceptance stages (tests, initialization, pilot, and
resume audit) before training and reuses passed stages on requeue. Source, seed,
and input order do not guarantee bitwise-identical GPU trajectories or identical
numerical-failure steps.
