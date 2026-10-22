# Attention scaling

How do attention family, key dimension, polynomial degree, normalization, and
coefficient learning affect language-model quality and cost? This study contains
all Transformer attention executions: softmax, exponential, Taylor, Hermite,
learned polynomial attention, GLA, and Gated DeltaNet.

[Combined comparisons](../../results/attention_scaling/README.md) ·
[Run index](runs.csv) · [Study membership](sweep.yaml) ·
[Shared implementation](../../scaffolds/transformer/README.md)

## Frozen pretrained-head polynomial sparsity

This exploratory experiment asks whether fixed or query-dependent subsets of
polynomial monomials reproduce frozen pretrained softmax heads at a small active
feature budget. Its home cluster is **Kennel**. The
[exploratory collection](../../results/exploratory/sparse_polynomial_heads/README.md)
owns the explicit comparison, findings, and figures. It enters no canonical
attention selection.

The teacher is pinned Llama-3.2-1B, with its native Q/K/V projections, RoPE, score
scaling and pretrained weights preserved. This is calibration of a pretrained
model, not allocation or pretraining of a new Transformer; QK normalization is
therefore not inserted into the teacher. Six query heads (0 and 16 in layers 2,
8, 14) are fixed before fitting. WikiText-2 documents use their official train,
validation and test splits, with duplicate titles/bodies removed across splits.
One 512-token window is selected per article: 200 training, 32 validation and 32
test windows, plus eight 2,048-token test windows. Model, tokenizer, dataset
revision, source files and token windows are pinned and hashed.

Ten executions compare the full 47,905-feature degree-three bank with fixed,
random and query-dependent supports of 128, 512 or 2,048 features, including the
constant. All variants learn positive monomial weights. Sparse variants start
from the dense run's validation-best weights and receive equal fitting budgets.
The six heads fit independently within each run; checkpoint selection uses their
aggregate held-out validation error. The full polynomial is a reference, not a
lower bound for the query-dependent function class. Key writes remain dense.

Use a dedicated worktree and its own locked PyTorch environment:

```bash
uv sync --project scaffolds/state_rank --locked --link-mode hardlink
PYTHONPATH=. scaffolds/state_rank/.venv/bin/python -m experiments.attention_scaling.launch_sparse_heads \
  --prepare --corpus /data/cl/u/rafah/home/datasets/sparse_polynomial_head_probe/wikitext2_512_seed20261015
```

Preparation allocates permanent run IDs and prints them in execution order.
Commit the new recipes, sweep membership, execution modules and scaffold code
before submitting the printed IDs:

```bash
PYTHONPATH=. scaffolds/state_rank/.venv/bin/python -m experiments.attention_scaling.launch_sparse_heads \
  --submit r79ad5ebadada re2a34e737173 rfaccbff16fbd rb6b8eb9aa58e rc8d38fead82c r57cb14fbf264 ra64f3b7b5e86 r34a8abd83902 rfb737d019b58 r57edb320f33d \
  --partition debug --hours 2 --node dogo
```

The launcher uses one H100, appends per-run logs, verifies the frozen execution
set on every start/requeue, and runs the configurations serially. Checkpoints
retain the latest resumable fit and the validation best. A time-limit warning
saves progress before requeue. The dense run owns the ignored teacher activation
archives; sparse runs reference those archives rather than copying them. Compact
metrics and execution records stay in each run. At completion retain only each
validation-best checkpoint and move retained artifacts into Kennel's primary
checkout before integration, following the repository lifecycle.

## Fixed RBF basis counts

This sweep trains **unnormalized** scalar RBF attention with **3, 5, 9, or 20**
basis functions and **per-head RMSNorm** after the attention sum. QK normalization
is enabled independently. Its home cluster is **Kennel**, and its open comparison
is tracked in [the RBF exploratory topic](../../results/exploratory/rbf_basis/README.md).

| RBF count | Run ID | Status |
| --- | --- | --- |
| 3 | `re530e0627233` | submitted: job `223903` |
| 5 | `r1b62a76f9b55` | acceptance: job `223902` |
| 9 | `r46c3479f9681` | running: job `223901` |
| 20 | `rc28d7b48dc72` | running: job `223900` |

For every count, centers are evenly spaced on `[-6, 6]`, widths are one, and
`rbf(c, 1)(s) = exp(-0.5 * (s-c)**2)`. The first three center grids are nested;
the 20-function grid is a separate uniform grid on the same interval.
Coefficients are fitted once to `exp(s)` under `N(0, 1)` with 128 quadrature
nodes and checked with 256 nodes, then frozen in `model.attention_basis`.
The fitted coefficients, centers, and widths are shared across all layers and
heads. The Transformer parameters and per-layer/head QK scale train normally.
All fits pass the quadrature convergence diagnostic and a float32 evaluation
check. The Gaussian is a fitting measure, not an enforced score distribution.

The shared recipe is `scaffolds/transformer/configs/base.yaml`: ten layers,
hidden width 1,024, 16 heads, `d_K = d_V = 64`, 2,048 prediction tokens per
sequence, global batch 256, seed 0, and 19,074 updates (10,000,269,312 tokens),
with 400 warmup updates. These runs explicitly use `transformer_reference`
initialization and the native `triton_scalar` backend. Each run uses **one H100**,
layer rematerialization, and a microbatch of 16 sequences, accumulated to the
global optimization batch. The selected backend fuses head RMSNorm and uses
the regular-RBF recurrence for the 5-, 9-, and 20-function fits. Checkpoints are saved every 500 updates. Baselines are
existing executions to be identified explicitly; this sweep allocates only RBF runs.

The full frozen configurations are authoritative. To reconstruct a fit for a
count `n`, use:

```python
import numpy as np
from scaffolds.transformer.attention.basis import fit_exp, rbf
fit = fit_exp([rbf(float(c), 1.0) for c in np.linspace(-6, 6, n)],
              mean=0.0, std=1.0, quadrature_order=128, check_order=256)
```

Use `worktrees/rbf-triton-training` on `codex/rbf-triton-training`. Sync its
locked H100 environment with `uv sync --project scaffolds/transformer --locked
--extra cuda --link-mode hardlink`. The FineWeb train/evaluation manifests under
`/data/cl/u/rafah/home/datasets/fineweb_12b/` match the study's recorded hashes;
all 91 referenced shards are readable. The launcher validates dataset identity
inside each job. The [scaffold qualification](../../scaffolds/transformer/validation/README.md#native-triton-scalar-attention)
records operator correctness, profiles and complete optimizer-update benchmarks.

Submit each run to **standard**, requesting one H100. Clear inherited
`LD_LIBRARY_PATH` so system CUDA libraries do not override the locked wheels.
The launcher gates production on CPU and GPU checks (including native Triton),
initialization, a full-recipe 20-update real-data pilot, and exact fresh-process
checkpoint resumption. Pilot timing estimates total training time; runs can span
multiple 24-hour allocations. The job requeues ten minutes before each limit
and resumes its latest checkpoint. Queue waits and repeated work since that
checkpoint add to the estimate.

All four frozen recipes use online W&B tracking in
`rafah-massachusetts-institute-of-technology/functional-memory`, grouped under
`attention_scaling`, with one identity retained across checkpoint resumptions.
Training metrics are logged every ten updates, validation every 1,000 updates,
and checkpoints every 500 updates. Acceptance pilots disable W&B; production
starts its own tracking identity. Canonical measurements remain in each run.

```bash
export EJKERNEL_TRITON_CACHE_COMPILES=1
export EJKERNEL_TRITON_CACHE_DIR="$PWD/.cache/transformer/triton"
for RUN in rc28d7b48dc72 r46c3479f9681 r1b62a76f9b55 re530e0627233; do
  env -u LD_LIBRARY_PATH TRANSFORMER_ALLOCATION_HOURS=24 TRANSFORMER_WANDB_MODE=online \
    sbatch --partition=standard --time=24:00:00 --gres=gpu:h100:1 \
    --cpus-per-task=8 --mem=128G --job-name="$RUN" \
    --output="experiments/attention_scaling/runs/$RUN/training.log" \
    scaffolds/transformer/launchers/kennel.sbatch \
    experiments.attention_scaling.acceptance \
    "experiments/attention_scaling/runs/$RUN/config.yaml" "$(git rev-parse HEAD)" 1 H100
done
```

Keep the frozen execution set unchanged while any job can run or requeue.

## Runs and comparisons

Each `runs/<run_id>/` holds one complete recipe and execution. Settings are fields
in `config.yaml` and columns in `runs.csv`: attention family, `d_K`, degree,
normalization, output scaling, QK normalization, coefficient learning, input conditioning, and
restrictions, sharing, seed, training budget, and outcome. Varying these settings adds runs here;
it does not create another study directory. Use permanent IDs for lookups:
descriptive labels need not be unique.

The run's `record.json` owns its source commit, executed dependency lock, home
cluster, current artifact locations and hashes, backend, and execution attempts.
Frozen recipes and raw trainer outputs retain the exact bytes used or produced
by that execution, including their original execution paths. Current artifact
locations come from the record. New executions allocate fresh IDs; historical
recipes are never edited to inherit current defaults.

The heatmap and degree plot under `results/attention_scaling/` select exact run IDs
and pin their scientific inputs. Each comparison states its controls and settings
in its selection YAML; study membership alone never selects a figure's inputs.
Existing plain-score
runs, numerical failures, and allocated but unlaunched runs remain distinguishable
in the index. Terminal metrics and validation-best checkpoint metrics are distinct.

Taylor uses `sum_r s^r/r!`. The fixed standard-normal Hermite approximation is
`exp(1/2) sum_r He_r(s)/r!`, expressed by the coefficients in each recipe;
its normalized degree-two factored form avoids cancellation near its root.
Learned unrestricted polynomials train their coefficients; sum-of-squares recipes
train factors that guarantee a nonnegative polynomial; rational recipes train a
numerator and a separate denominator polynomial of its own degree, optionally
nonnegative and with its constant fixed at 1 or learned (`attention_denominator_*`), optionally starting at linear
attention (`attention_coefficient_initialization`) and with a penalty on high-order
coefficients in the training loss (`optim.coefficient_penalty`, with or without the
denominator). QK normalization changes
the scores independently of normalization of the attention weights. Unnormalized
attention may rescale each head's output before the output projection, by `1/t` or
by a per-head RMSNorm (`attention_output_scaling`). GLA uses its
own gated recurrence, and Gated DeltaNet its gated delta rule (FLA's kernels).
New runs use each mechanism's reference initialization (`model.initialization`); runs
whose recipes omit the field used the scaffold's Flax defaults. The shared scaffold defines each mechanism
and its kernels.

Query-conditioned degree-one runs predict a signed constant and linear coefficient
from the current attention input, per layer and head
(`model.attention_coefficient_conditioning: query`). The
[exploratory comparison](../../results/exploratory/input_dependent_coefficients/README.md)
uses input-independent learned coefficients as its matched control.

## Cluster and artifacts

Runs keep their original home cluster: **Kennel or ORCD**, recorded per run in
`runs.csv` and `record.json`. Kennel artifacts live under
`/data/cl/u/rafah/home/projects/functional-memory/experiments/attention_scaling/runs/`.
ORCD artifacts live under
`/orcd/pool/006/rafah/functional_memory/artifacts/experiments/attention_scaling/runs/`,
and their records state the absolute paths. ORCD ran every Flax-default GLA execution,
the Flax-default QK-norm softmax runs at `d_K = 8, 16, 32, 128`, the
reference-initialization normalized fixed Taylor `m = 2` and `4` at every `d_K`, and the
reference-initialization unnormalized learned Taylor `m = 1–4` at `d_K = 64`.
Artifacts on the other cluster are unavailable in each checkout.
Git carries the compact measurements and recipes, not checkpoints or logs.

Completed training retains the validation-best checkpoint and one successful log.
A scientific failure retains its matching log and diagnostics; a diagnostic
checkpoint is retained only when its record justifies it. There are no checkpoints
for planned executions. See [CLUSTERS.md](../../CLUSTERS.md) for access, resource
limits, and cluster-specific launch settings.

## Launch a fresh execution

Use a separate worktree for a new sweep. Sync that worktree's own locked environment:

```bash
git worktree add -b codex/attention-sweep worktrees/attention-sweep main
cd worktrees/attention-sweep
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
# Kennel B300 jobs use the CUDA 13 environment from the same lock (CLUSTERS.md).
UV_PROJECT_ENVIRONMENT=$PWD/scaffolds/transformer/.venv-cuda13 \
  uv sync --project scaffolds/transformer --locked --extra cuda13 --link-mode hardlink
```

Prepare a complete recipe with the desired settings and dataset paths for the home
cluster. New dot-product runs use `model.qk_norm: true`; GLA and degree zero do not.
New runs use their mechanism's reference initialization, which `records create` fills in
when the recipe leaves `model.initialization` unset.
Untied embeddings, the scaffold's selective weight decay, and 400 warmup updates
(`optim.warmup_steps`) are canonical. New unnormalized runs set
`model.attention_output_scaling: head_rms_norm`. Normalized fixed Taylor runs, and tanh
query-conditioned and static-coefficient unnormalized runs of degree up to 8, use the Mosaic
GPU kernels (`model.attention_backend: mosaic`). The
following allocates a fresh execution from an existing QK-normalized Taylor recipe that
predates them:

```bash
sed 's/attention_backend: fused/attention_backend: mosaic/' \
  experiments/attention_scaling/runs/r044b3356f215/config.yaml > /tmp/taylor-example.yaml
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records create \
  --study experiments/attention_scaling --config /tmp/taylor-example.yaml --label taylor-example
```

The CLI resolves current defaults, assigns new output paths and tracking identity,
and prints the new run directory. For different settings, provide an edited recipe
outside the registered runs. An explicitly requested plain-score experiment also
requires `--allow-plain-qk`. Allocate every run before committing the sweep; do not
change the frozen source, environment, membership, or recipes while any job can run.

Set `RUN` to the allocated ID. From the worktree root, commit and submit on Kennel:

```bash
RUN=<new-run-id>
git add experiments/attention_scaling
git commit -m "Prepare attention scaling sweep"
sbatch --job-name="$RUN" \
  --output="experiments/attention_scaling/runs/$RUN/training.log" \
  scaffolds/transformer/launchers/kennel.sbatch \
  experiments.attention_scaling.acceptance \
  "experiments/attention_scaling/runs/$RUN/config.yaml" "$(git rev-parse HEAD)" 4 B300,H100
```

On ORCD, use the ORCD worktree/storage policy in `CLUSTERS.md`, its dataset paths,
and the same module and run layout with `launchers/orcd.sbatch` and `2 H200`.
The launcher runs configuration-appropriate correctness tests, a full-recipe pilot,
and an exact fresh-process resume check before production. Record the returned
job ID and launch commit; requeues retain that identity. Qualification and backend
selection follow the recipe rather than a cohort-specific launcher.

Refresh the derived run index after collecting outputs:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records index \
  experiments/attention_scaling
```

## Input-dependent coefficient comparison

The pair is `r9cf9589b9855` (query-dependent) and `r03bfa6fd7b2e`
(input-independent learned control). Both use output RMSNorm and 400 warmup steps.
Both completed all 19,074 updates; their findings remain in the linked exploratory
comparison. Their executed source is `1e78579354087d3d3f62fdda865d90913068e960`.
The launch commands, from the experiment worktree on Kennel after committing all
inputs and code, were:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 sbatch \
  --nice=0 --partition=standard --time=1-00:00:00 --gres=gpu:h100:2 \
  --cpus-per-task=16 --mem=128G --job-name=r9cf9589b9855 \
  --output=experiments/attention_scaling/runs/r9cf9589b9855/training.log \
  scaffolds/transformer/launchers/kennel.sbatch \
  experiments.attention_scaling.acceptance \
  experiments/attention_scaling/runs/r9cf9589b9855/config.yaml "$(git rev-parse HEAD)" 2 H100

env -u XLA_PYTHON_CLIENT_MEM_FRACTION sbatch \
  --nice=0 --partition=standard --time=1-00:00:00 --gres=gpu:h100:2 \
  --cpus-per-task=16 --mem=128G --job-name=r03bfa6fd7b2e \
  --output=experiments/attention_scaling/runs/r03bfa6fd7b2e/training.log \
  scaffolds/transformer/launchers/kennel.sbatch \
  experiments.attention_scaling.acceptance \
  experiments/attention_scaling/runs/r03bfa6fd7b2e/config.yaml "$(git rev-parse HEAD)" 2 H100
```

The query-dependent run uses a GPU memory fraction of 0.90; the control uses the
runtime default. These are execution settings, separate from their scientific recipes.

### Tanh coefficients, degrees 1–4

Runs `r80300a4cd5fa` (`m = 1`), `rbfe1750cc24c` (`m = 2`), and `r0bc57959ef6c`
(`m = 3`) use `c_r(x_t) = b_r + a_r tanh(w_r^T x_t)` independently per layer and
head. The input is the activation after the block input RMSNorm. The coefficient
projection has no bias inside tanh; `b_r` starts at `1/r!`, `a_r` at one, and
`w_r` at zero. All coefficient parameters are FP32 and excluded from weight decay.
The recipes set `attention_coefficient_conditioning: query` and
`attention_coefficient_activation: tanh`, with unnormalized attention, per-head
output RMSNorm, QK normalization, `transformer_reference` initialization, and
400 warmup updates. They retain the 200M model, seed 0, and 19,074-update/10.0B-token
recipe. Each frozen run config contains the complete settings.

The tanh mechanisms use direct fused query-dependent polynomial attention.
Source `254cab946e637cd6cd5c004b6976351b1dee3651` ran as job `210243` on four B300
GPUs (`akita`, `standard`), job `210244` on four H100 GPUs (`dogo`, `long`), and
job `210245` on four B300 GPUs (`akita`, `long`), respectively. All have nice 0.
All three passed full acceptance, including full-recipe pilots and exact
fresh-process checkpoint resumption, and completed all 19,074 training updates.
The launch commands from `worktrees/input-dependent-tanh` on Kennel are:

```bash
sbatch --nice=0 --partition=standard --nodelist=akita --gres=gpu:b300:4 \
  --time=1-00:00:00 --cpus-per-task=32 --mem=256G --job-name=r80300a4cd5fa \
  --output=experiments/attention_scaling/runs/r80300a4cd5fa/training.log \
  scaffolds/transformer/launchers/kennel.sbatch experiments.attention_scaling.acceptance \
  experiments/attention_scaling/runs/r80300a4cd5fa/config.yaml 254cab946e637cd6cd5c004b6976351b1dee3651 4 B300

sbatch --nice=0 --partition=long --nodelist=dogo --gres=gpu:h100:4 \
  --time=1-00:00:00 --cpus-per-task=32 --mem=256G --job-name=rbfe1750cc24c \
  --output=experiments/attention_scaling/runs/rbfe1750cc24c/training.log \
  scaffolds/transformer/launchers/kennel.sbatch experiments.attention_scaling.acceptance \
  experiments/attention_scaling/runs/rbfe1750cc24c/config.yaml 254cab946e637cd6cd5c004b6976351b1dee3651 4 H100

sbatch --nice=0 --partition=long --nodelist=akita --gres=gpu:b300:4 \
  --time=1-00:00:00 --cpus-per-task=32 --mem=256G --job-name=r0bc57959ef6c \
  --output=experiments/attention_scaling/runs/r0bc57959ef6c/training.log \
  scaffolds/transformer/launchers/kennel.sbatch experiments.attention_scaling.acceptance \
  experiments/attention_scaling/runs/r0bc57959ef6c/config.yaml 254cab946e637cd6cd5c004b6976351b1dee3651 4 B300
```

Run `rfee72428a45e` extends this comparison to `m = 4`, retaining the complete
degree-three recipe apart from degree and run identity. Job `213697` completed
all 19,074 training updates on four B300 GPUs on `akita`, partition `standard`,
at nice 0. Its preselected static degree-four control is `rc2b555b905dd`.
All acceptance stages passed: 359 CPU checks,
nine distributed CPU checks, 52 GPU checks, canonical initialization, a
20-update full-recipe pilot, and exact checkpoint resumption. The pilot recorded
79 finite telemetry statistics over every update. Production started from step
zero with a fresh W&B identity, and all 79 statistics remained finite across
the completed production run.
The launch command from `worktrees/input-dependent-tanh-m4` was:

```bash
sbatch --nice=0 --partition=standard --nodelist=akita --gres=gpu:b300:4 \
  --time=1-00:00:00 --cpus-per-task=32 --mem=256G --job-name=rfee72428a45e \
  --output=experiments/attention_scaling/runs/rfee72428a45e/training.log \
  scaffolds/transformer/launchers/kennel.sbatch experiments.attention_scaling.acceptance \
  experiments/attention_scaling/runs/rfee72428a45e/config.yaml ec96b6cbaf04dfefd059fafc601559652204fcc6 4 B300,H100
```

The shared acceptance checks must pass on each allocation before production starts.
For a fresh execution, allocate a new run ID and launch from the current source
commit. The four-GPU recipe with `model.remat: false` automatically enables
[standard training telemetry](../../scaffolds/transformer/README.md): every-update
activation and parameter-group statistics, W&B samples at the logging cadence,
and validation-window summaries and gradient-spike diagnostics in `metrics.jsonl`.
Question, input selections, outcomes, and the open scientific decision live in the
[comparison README](../../results/exploratory/input_dependent_coefficients/README.md).

To update a figure, change its explicit run selection, run
`python -m scaffolds.records refresh results/<analysis>` in the scaffold environment,
and use that result collection's regeneration command.

### Tanh gate initialization: wider projection draw and inner bias

Eight single-GPU runs test, one change at a time, why the zero-amplitude tanh runs
(`r75b81509b10b`, `r8c9b076f9635`, `rb306d49a5abe`, `rf1160aa95e46`) barely use their
input dependence. All start at `a_r = 0` and `b_r = 1/r!`, use the Mosaic kernels
(`attention_backend: mosaic`), one B300 or H100 GPU, and global microbatch 32
(gradient accumulation eight; microbatch 64 does not fit one H100 on this path), and
otherwise keep the zero-amplitude recipe.

| Change | m = 1 | m = 2 | m = 3 | m = 4 |
| --- | --- | --- | --- | --- |
| `attention_coefficient_projection_initial_std: 0.03125`: `W ~ N(0, 1/hidden_size)`, pre-activations of order one | `r5be9a97f06e2` | `rec520e9303ac` | `r215d3d91dd02` | `rd4f66b1959d2` |
| `attention_coefficient_tanh_bias: true`: learned `beta_r` inside the tanh from zero, `W ~ N(0, 0.002^2)` | `rbd4eb6ccc16f` | `r6bba734352f0` | `rde534cbb87a1` | `r59490b6e0818` |

Launch commit `629cab71a2a3d490f078488918b8861a7cfce874` from `worktrees/tanh_gate_init` on
Kennel. Degrees 1 and 4 run on `standard` (jobs 219940–219943, 24 h, `akita` excluded so
they cannot preempt this project's `long` runs there); degrees 2 and 3 run on `long`
(jobs 219944–219947) with the launcher's default time limit:

```bash
RUN=<run-id>
sbatch --partition=standard --time=24:00:00 --exclude=collie,samoyed,doodle,terrier,akita \
  --gres=gpu:1 --cpus-per-task=16 --mem=128G --job-name="$RUN" \
  --output="experiments/attention_scaling/runs/$RUN/training.log" \
  scaffolds/transformer/launchers/kennel.sbatch experiments.attention_scaling.acceptance \
  "experiments/attention_scaling/runs/$RUN/config.yaml" 629cab71a2a3d490f078488918b8861a7cfce874 1 B300,H100
sbatch --partition=long --gres=gpu:1 --cpus-per-task=16 --mem=128G --job-name="$RUN" \
  --output="experiments/attention_scaling/runs/$RUN/training.log" \
  scaffolds/transformer/launchers/kennel.sbatch experiments.attention_scaling.acceptance \
  "experiments/attention_scaling/runs/$RUN/config.yaml" 629cab71a2a3d490f078488918b8861a7cfce874 1 B300,H100
```

### Tanh with saturating projection draws, degrees 2 and 4

Four single-GPU runs ask whether a gate that starts saturated keeps more input
dependence. They follow the zero-amplitude `N(0, 1/hidden_size)` runs above
(`r5be9a97f06e2` …) apart from the projection's draw: with `x_t` after RMSNorm the
pre-activation `w_r^T x_t` starts with standard deviation about 3 or 10, so the tanh is
mostly or fully saturated and `w_r` receives little gradient once `a_r` grows. The
start is still the static Taylor kernel (`a_r = 0`).

| `attention_coefficient_projection_initial_std` | m = 2 | m = 4 |
| --- | --- | --- |
| 0.09375 (`3/sqrt(d)`) | `rac317a500871` | `r4cee946a1a4b` |
| 0.3125 (`10/sqrt(d)`) | `r49ac1e58fba3` | `r362b2df6edbd` |

Launch commit `eec53e3e1539eca7d20207daea659003711d82d4` from `worktrees/tanh_large_draw`
on Kennel, one H100 each (degree 4 on `standard` on `dogo`, 24 h; degree 2 on `long` on
`mastiff`):

```bash
RUN=<run-id>; PARTITION=<standard|long>
sbatch --partition="$PARTITION" --time=1-00:00:00 --exclude=collie,samoyed,doodle,terrier \
  --gres=gpu:1 --cpus-per-task=16 --mem=128G --job-name="$RUN" \
  --output="experiments/attention_scaling/runs/$RUN/training.log" \
  scaffolds/transformer/launchers/kennel.sbatch experiments.attention_scaling.acceptance \
  "experiments/attention_scaling/runs/$RUN/config.yaml" eec53e3e1539eca7d20207daea659003711d82d4 1 B300,H100
```

### Query-key tanh coefficients, degrees 1–4

Four single-GPU runs test whether the writer's vote is what query-dependent coefficients
lack: `attention_coefficient_conditioning: query_key` multiplies each key's degree-`r`
term by `f_r(x_i) = 1 + a^k_r tanh(w_k_r^T x_i)` beside `c_r(x_t) = b_r + a_r tanh(w_r^T x_t)`,
with no recurrence, gate, or decay. Both amplitudes start at zero and both projections are
drawn from `N(0, 1/hidden_size)`, so training starts from the static Taylor kernel; the
rest is the gate-initialization recipe above (Mosaic kernels, one B300 or H100, global
microbatch 32, head RMSNorm, QK normalization, warmup 400).

| m = 1 | m = 2 | m = 3 | m = 4 |
| --- | --- | --- | --- |
| `rba839e6386b7` (blocked reference) | `r0aab2b77017f` | `r4a7c601b4b1f` | `r4e3f4097a60e` |

Degrees 2–4 run the Mosaic query kernel once per order with the key factor folded into the
values, so a step costs about 1.6× (m = 2) to 1.9× (m = 4) the query-only step. Each pass
rounds its output to BF16 before the passes are summed; at degree one that rounding reaches
the head RMSNorm and corrupts the gradient (the step-one gradient norm differs from the
blocked reference on H100 and B300 alike), so degree one runs on the blocked reference at
global microbatch 64, like the zero-amplitude reference runs. At degrees 2 and 4 the
step-one gradients of the blocked, per-degree, and single-pass paths agree to four digits.

Degrees 2–4 use launch commit `f13be89f1eacf37775e92f1ffc2507be17c96810` from
`worktrees/tanh_query_key` on Kennel, `mastiff` excluded: degrees 2 and 3 ran on `akita`
(`standard`), and degree 4 started on `long` on `dogo`, stalled at update 858 (the launcher
ended it after 20 minutes without log output), and resumed from its update-500 checkpoint on
`akita` under `standard`. Degree one uses launch commit
`624ef89621febc0f203cf7dc6463586598d3ec3c` from `worktrees/tanh_query_key_m1` on `akita`
(`standard`). With `RUN` and `COMMIT` set:

```bash
sbatch --partition=standard --time=24:00:00 --nodelist=akita --gres=gpu:1 --cpus-per-task=16 --mem=128G \
  --job-name="$RUN" --output="experiments/attention_scaling/runs/$RUN/training.log" \
  scaffolds/transformer/launchers/kennel.sbatch experiments.attention_scaling.acceptance \
  "experiments/attention_scaling/runs/$RUN/config.yaml" "$COMMIT" 1 B300
```

Outcomes are in the [input-dependent coefficient topic](../../results/exploratory/input_dependent_coefficients/README.md).

### Tanh starts on the blocked reference, degrees 1–4

Eight single-B300 runs on the plain-JAX reference path (`attention_backend: blocked`, no
fused kernel) check two things about the tanh coefficients: whether the fused kernel
explains the original start's deficit, and whether a zero-amplitude start removes it.
Both sets keep the degree-matched tanh recipe above (head RMSNorm, QK normalization,
`transformer_reference`, 400 warmup updates, global microbatch 64) apart from the
backend and the start.

| Start | m = 1 | m = 2 | m = 3 | m = 4 |
| --- | --- | --- | --- | --- |
| Original, `a_r = 1`, `W = 0`: the kernel check against `r80300a4cd5fa`, `rbfe1750cc24c`, `r0bc57959ef6c`, `rfee72428a45e` | `r6fdf6d484fd2` | `r722fc3dd263e` | `radd66df6371a` | `rc836509ca119` |
| Zero amplitude, `a_r = 0`, `W ~ N(0, 0.002^2)` (`attention_coefficient_amplitude_initial: 0`, `attention_coefficient_projection_initial_std: 0.002`) | `r75b81509b10b` | `r8c9b076f9635` | `rb306d49a5abe` | `rf1160aa95e46` |

All run on one B300 on `akita` at nice 0: zero-amplitude starts on `standard` (24 h),
original starts on `long`, where `standard` jobs preempt them and they resume from their
checkpoints. Six use launch commit `bb81667a6f510e165dab39f92367233c41d4ed8b` from
`worktrees/tanh_reference_check` (jobs 219218, 219220, 219221 and 219223–219225) on the
CUDA 12 environment. `r8c9b076f9635` (job 219521, commit
`67f535f85eca5776b63735f8de3440c9158bc88a`) and `r6fdf6d484fd2` (job 219555, commit
`8851a838fdb95d02c9fb900d4eb0194c777318b7`) use the B300 CUDA 13 environment
([validation record](../../scaffolds/transformer/validation/README.md#b300-cuda-13-environment))
from their own worktrees. With `RUN` set to a run ID and `COMMIT` to its launch commit:

```bash
sbatch --partition=standard --time=1-00:00:00 --nodelist=akita --gres=gpu:b300:1 \
  --cpus-per-task=16 --mem=128G --job-name="$RUN" \
  --output="experiments/attention_scaling/runs/$RUN/training.log" \
  scaffolds/transformer/launchers/kennel.sbatch experiments.attention_scaling.acceptance \
  "experiments/attention_scaling/runs/$RUN/config.yaml" "$COMMIT" 1 B300
sbatch --partition=long --nodelist=akita --gres=gpu:b300:1 --cpus-per-task=16 --mem=128G --job-name="$RUN" \
  --output="experiments/attention_scaling/runs/$RUN/training.log" \
  scaffolds/transformer/launchers/kennel.sbatch experiments.attention_scaling.acceptance \
  "experiments/attention_scaling/runs/$RUN/config.yaml" "$COMMIT" 1 B300
```

Outcomes and the comparison with the fused runs and the static controls live in the
[input-dependent coefficient topic](../../results/exploratory/input_dependent_coefficients/README.md).

### Tanh coefficients at degree 8

Three runs extend the zero-amplitude tanh comparison to m = 8, where the
static polynomial has more freedom: query-only (`rccf19c31669d`), query–key
(`r66db587096c4`), and the static learned control (`raa043e978a3d`, no static unnormalized
m = 8 run existed). They keep the 1/d-draw recipes (`r5be9a97f06e2`, `r4e3f4097a60e`) and
the static m = 4 recipe (`rc2b555b905dd`) apart from degree, at global microbatch 64. The
Mosaic query kernel covers degrees 1–4, so query-only and static use the Triton fused
kernels; their step-one gradients agree with the blocked reference within 2%. Query–key uses
the one-pass Triton kernel (`backends/fused_query_key_polynomial.py`), which forms the pair
coefficients inside each tile; its step-one gradients agree with the blocked reference within
1.4%, and it takes 99 ms per layer forward and backward on an H100 PCIe at batch 32 against
182 ms for the blocked reference, so the query–key run `r66db587096c4` uses four GPUs on `long`.
Query-only runs on `standard`, the control on `long`:

```bash
RUN=<run-id>; PARTITION=<standard|long>; GPUS=<2|4>; COMMIT=<launch commit in the run's record.json>
sbatch --partition="$PARTITION" --time=1-00:00:00 --exclude=collie,samoyed,doodle,terrier \
  --gres=gpu:$GPUS --cpus-per-task=32 --mem=256G --job-name="$RUN" \
  --output="experiments/attention_scaling/runs/$RUN/training.log" \
  scaffolds/transformer/launchers/kennel.sbatch experiments.attention_scaling.acceptance \
  "experiments/attention_scaling/runs/$RUN/config.yaml" "$COMMIT" "$GPUS" B300,H100
```

## Independent GDN erase gate

Run `rf7341031e3d5` asks whether GDN benefits from separate scalar erase and write
projections. It uses `model.gdn_erase_mode: independent`, with the erase projection
initialized from the write projection as a separate parameter. All original weights
match tied GDN at initialization. It preserves the canonical GDN block and reference
initialization, 400 warmup updates, standard training telemetry, and the 19,074-update
FineWeb recipe. The preselected control is `rf640fc13f3a8`; comparison findings and
current status belong in the [exploratory result](../../results/exploratory/gdn_polynomial/README.md).
Its home cluster is Kennel. Production job `217974` completed all 19,074 updates
on 2026-10-12 at 18:32 EDT, on `standard` at nice 0 with four H100 SXM GPUs on
`dogo`. Terminal and validation-best checkpoint both correspond to update 19,074.
[W&B run](https://wandb.ai/rafah-massachusetts-institute-of-technology/functional-memory/runs/3113f4ce156c46c78fcc7c87efb60273):
`200m-dk064-gdn-independent-erase-warmup400-seed0`.
Production and passed runtime qualification `217971` use
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, with the memory fraction unset. The frozen
global batch remains 256, global microbatch 64, and gradient accumulation four;
four devices divide each microbatch into 16 sequences per GPU.
Acceptance passed the complete test suite, a 20-update full-recipe pilot,
finite validation, checkpoint saving, and exact fresh-process resumption from
update 20 to 21. Production training, validation, and checkpointing took 3.52 hours.
The [qualification record](../../scaffolds/transformer/validation/runs/gdn-independent-precision-h100/sxm-four-gpu.json)
pins the acceptance evidence. The canonical 400-update warmup and gated output
RMSNorm were active throughout training. Completed findings, diagnostic summaries,
and the limitations of the historical-control comparison are recorded only in the
linked exploratory result.
Source `d3d93bc6ccaa8a76fc4b29124c80988ae1593405`
retains FP32 chunk intermediates and TF32x3 products for tied and independent gates.
The independently optimized erase and write projections have separate device
buffers while preserving equal canonical initial values and the original RNG draws.
The [scaffold validation record](../../scaffolds/transformer/validation/README.md#fp32-chunk-precision-repair)
contains the accuracy and runtime checks.
Qualification and production commands, from `worktrees/gdn-independent-erase`:

```bash
env -u XLA_PYTHON_CLIENT_MEM_FRACTION sbatch --parsable --nice=0 --partition=standard --nodelist=dogo --gres=gpu:h100:4 \
  --time=01:00:00 --cpus-per-task=32 --mem=256G \
  --export=ALL,XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --job-name=gdn-runtime-four-gpu \
  --output=scaffolds/transformer/validation/logs/gdn-runtime-%j.log \
  scaffolds/transformer/validation/qualify_gdn_runtime.sbatch \
  experiments/attention_scaling/runs/rf7341031e3d5/config.yaml \
  d3d93bc6ccaa8a76fc4b29124c80988ae1593405 H100 24 4

env -u XLA_PYTHON_CLIENT_MEM_FRACTION sbatch --parsable --nice=0 --partition=standard --nodelist=dogo --gres=gpu:h100:4 \
  --time=1-00:00:00 --cpus-per-task=32 --mem=256G \
  --export=ALL,XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --dependency=afterok:217971 --kill-on-invalid-dep=yes \
  --job-name=rf7341031e3d5 \
  --output=experiments/attention_scaling/runs/rf7341031e3d5/training.log \
  scaffolds/transformer/launchers/kennel.sbatch \
  experiments.attention_scaling.acceptance \
  experiments/attention_scaling/runs/rf7341031e3d5/config.yaml \
  d3d93bc6ccaa8a76fc4b29124c80988ae1593405 4 H100
```

Production verifies acceptance with the same source, recipe, hardware, and runtime
settings before training; matching passed stages are reused. Hardware qualification
and kernel benchmarks are documented in the linked scaffold validation record.
