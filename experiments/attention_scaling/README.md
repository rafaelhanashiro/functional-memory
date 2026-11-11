# Attention scaling

How do attention family, key dimension, polynomial degree, normalization, and
coefficient learning affect language-model quality and cost? This study contains
the established 10B-token Transformer attention executions: softmax, exponential, Taylor, Hermite,
learned polynomial attention, GLA, and Gated DeltaNet.

[Combined comparisons](../../results/attention_scaling/README.md) ·
[Run index](runs.csv) · [Study membership](sweep.yaml) ·
[Shared implementation](../../scaffolds/transformer/README.md)

## Fixed-input coefficient oracle

Two exploratory coefficient-only fits on **Kennel** use the retained degree-eight
query-only checkpoint `r50be104c2a9e`, step 19,074. `rec7a01f3282c` fits one sequence;
`rfc37cfdf49d0` fits a full batch of 256, with separate coefficients for every
example, layer, head, query and order. All pretrained parameters stay frozen. The
[exploratory collection](../../results/exploratory/coefficient_oracle/README.md)
documents the question, fixed-input selection, convergence rule and interpretation.
These runs enter no canonical comparison.

Both fits are complete. Their compact measurements and provenance live with their
run IDs; retained checkpoints and logs remain on Kennel. Fresh executions use new
run IDs in an isolated worktree, with its CUDA 12 and CUDA 13 environments synced
from its own lock as described in [CLUSTERS.md](../../CLUSTERS.md).
The allocation command for this pair was:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m experiments.attention_scaling.launch_coefficient_oracle prepare \
  --parent /data/cl/u/rafah/home/projects/functional-memory/experiments/attention_scaling/runs/r50be104c2a9e
```

After committing the frozen inputs, the two executions were submitted with:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m experiments.attention_scaling.launch_coefficient_oracle submit \
  rec7a01f3282c rfc37cfdf49d0 --partition standard,long
```

Each job requests one B300 or H100, microbatches the fixed inputs, checks the frozen
execution set on every start/requeue, qualifies checkpoint capture and replay before
fitting, and saves resumable progress on a time-limit signal. The launcher selects
CUDA 13 on B300 and CUDA 12 on H100. Fit-only metrics are kept distinct from held-out
validation metrics; the retained best coefficients are needed to reproduce the fit.

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
denominator); outcomes are in the
[rational polynomial comparison](../../results/rational_polynomial/README.md). QK normalization changes
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
[input-dependent coefficient comparison](../../results/input_dependent_coefficients/README.md)
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
query-conditioned and static-coefficient unnormalized runs of degree up to 12 (9 at
`d_K = 128`), use the Mosaic GPU kernels (`model.attention_backend: mosaic`). The
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
Both completed all 19,074 updates; their findings are in the linked input-dependent
coefficient comparison. Their executed source is `1e78579354087d3d3f62fdda865d90913068e960`.
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
[comparison README](../../results/input_dependent_coefficients/README.md).

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

Outcomes are in the [input-dependent coefficient comparison](../../results/input_dependent_coefficients/README.md).

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
[input-dependent coefficient comparison](../../results/input_dependent_coefficients/README.md).

### Tanh coefficients at degree 8

Three runs extend the zero-amplitude tanh comparison to m = 8, where the
static polynomial has more freedom: query-only (`rccf19c31669d`), query–key
(`r66db587096c4`), and the static learned control (`raa043e978a3d`, no static unnormalized
m = 8 run existed). They keep the 1/d-draw recipes (`r5be9a97f06e2`, `r4e3f4097a60e`) and
the static m = 4 recipe (`rc2b555b905dd`) apart from degree, at global microbatch 64.
Query-only and static use the Triton fused kernels; their step-one gradients agree with the blocked reference within 2%. Query–key uses
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

`ra39c41604704` asks whether the query-only deficit at m = 8 comes from Adam moving every
order's amplitude at the same rate. It repeats `rccf19c31669d` with
`model.attention_coefficient_amplitude_scale: factorial`, `c_r = b_r + (a_r/r!) tanh(w_r^T x_t)`,
so order r's modulation moves at lr/r!, and with the Mosaic query kernel, which now covers
degree 8. It runs on one GPU on `standard` (global microbatch 32) with the command above.

### Degree 12

Two runs extend this to m = 12, each from the query-only recipe `rccf19c31669d` apart from
the listed settings, at global microbatch 32 on one B300: static learned coefficients
(`r3012d19137cd`) and query-only with `attention_coefficient_amplitude_scale: factorial`
(`r0c9b8b75ec17`). Both use the Mosaic kernels, which cover degree 12 at `d_K = 64`, and
launched with the command above, `PARTITION=standard` and `GPUS=1`.

### Factorially scaled coefficients

Learned coefficients raise gradient clipping from 7% of updates (exp with head RMSNorm) to
26% (static m = 8) and 57% (static m = 12): each coefficient's gradient carries `s^r`.
Five runs divide the stored input-independent coefficient by `r!`
(`model.attention_coefficient_scale: factorial`), each from the query-only recipe
`rccf19c31669d` apart from the listed settings, on the Mosaic kernels at global microbatch 32
on one GPU: static m = 12 (`r6c2b0b793143`), query-only m = 12 with both the bias and the
amplitude scaled (`r5b013b1a762b`), static m = 8 (`rc9d269a084ce`), query-only m = 8 with both
scaled (`r50be104c2a9e`), and fixed unnormalized Taylor with head RMSNorm at m = 12
(`ra2e7161fdf4a`) and m = 8 (`r4836c6eba58f`), the start of the scaled runs with nothing learned. They launch with the
command above, `PARTITION=standard,long` and `GPUS=1`, ahead of the query–key runs.

Query–key with both scales (`attention_coefficient_scale` and
`attention_coefficient_amplitude_scale: factorial`; the key factor is already relative to
`c_r`) runs at m = 8 (`r0b7141a987dc`) and m = 12 (`r95c5dea57a89`) on the one-pass Triton
kernel with its H100 backward tiles (`--allow-triton-attention`), one H100 each on
`standard`; B300 keeps the slower original geometry.

### PaTH position encoding, softmax and tanh m = 2, 4

Three single-B300 runs replace RoPE with PaTH (`positional_encoding: path`, see the
[scaffold](../../scaffolds/transformer/README.md#path-position-encoding)): scores
transported through token Householder products, with the authors' gate design. The
query-conditioned runs keep the zero-amplitude, `N(0, 1/hidden_size)`-draw tanh recipe
(`r5be9a97f06e2` with its degree changed; head RMSNorm); the control keeps the canonical
softmax recipe (`rdf00da0866a8`). All use QK normalization, 400 warmup updates, and global
microbatch 32. Softmax trained throughout on the chunked JAX PaTH implementation
(`attention_backend: blocked`, about 6.0 s per update on one B300). The m = 2 and m = 4
runs trained on it up to updates 7,500 and 6,500 and then on FLA's fused kernels
(`fused`, the polynomial fork); each record's `attention_backend_segments` gives the
updates per implementation.

| Kernel | Run | RoPE counterpart | Updates on `blocked` / `fused` |
| --- | --- | --- | --- |
| Softmax | `rf1ece7e314b3` | `rdf00da0866a8` | all / none |
| Tanh query-conditioned, m = 2 | `rb901a10c4777` | `rec520e9303ac` | 1-7,500 / 7,501-19,074 |
| Tanh query-conditioned, m = 4 | `r4de2b1429c8a` | `rd4f66b1959d2` | 1-6,500 / 6,501-19,074 |

Findings are in the [exploratory result](../../results/exploratory/path_position_encoding/README.md).

The launch command, on Kennel from a worktree at the launch commit recorded in each run's
`record.json`:

```bash
RUN=<run-id>
sbatch --partition=long --gres=gpu:b300:1 --cpus-per-task=16 --mem=128G --job-name="$RUN" \
  --output="experiments/attention_scaling/runs/$RUN/training.log" \
  scaffolds/transformer/launchers/kennel.sbatch experiments.attention_scaling.acceptance \
  "experiments/attention_scaling/runs/$RUN/config.yaml" "$(git rev-parse HEAD)" 1 B300
```

## Independent GDN erase gate

Run `rf7341031e3d5` asks whether GDN benefits from separate scalar erase and write
projections. It uses `model.gdn_erase_mode: independent`, with the erase projection
initialized from the write projection as a separate parameter. All original weights
match tied GDN at initialization. It preserves the canonical GDN block and reference
initialization, 400 warmup updates, standard training telemetry, and the 19,074-update
FineWeb recipe. The preselected control is `rf640fc13f3a8`; comparison findings and
current status belong in the [polynomial GDN comparison](../../results/gdn_polynomial/README.md).
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
linked polynomial GDN comparison.
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

## Polynomial Gated DeltaNet

These runs add polynomial memory (`model.gdn_polynomial_degree`), narrower keys, or a tanh
linear write (`model.gdn_write_activation: tanh`) to independent-erase GDN; the
[polynomial GDN comparison](../../results/gdn_polynomial/README.md) defines the model and
holds the findings. Home cluster: Kennel. Each record's `submission` holds the exact
launch command, source commit, and working directory; the completed runs' retained
checkpoints and logs are in their run directories here.

| Run | Variant | d_K | Status |
| --- | --- | ---: | --- |
| `rd40af7956426` | quadratic | 64 | complete |
| `ra79e66bd9121` | cubic | 16 | complete |
| `ra98612db06d4` | cubic | 8 | complete |
| `rae2329afb5bd`, `r5c9701488368` | independent erase, `m = 1` | 16, 8 | cancelled before training |
| `rf885f946fc9b`, `r2fb020d67c0d` | quadratic | 16, 8 | cancelled before training |
| `rf65d38ec0552` | tanh linear write | 64 | complete |
