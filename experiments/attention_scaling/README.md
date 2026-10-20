# Attention scaling

How do attention family, key dimension, polynomial degree, normalization, and
coefficient learning affect language-model quality and cost? This study contains
all Transformer attention executions: softmax, exponential, Taylor, Hermite,
learned polynomial attention, GLA, and Gated DeltaNet.

[Combined comparisons](../../results/attention_scaling/README.md) ·
[Run index](runs.csv) · [Study membership](sweep.yaml) ·
[Shared implementation](../../scaffolds/transformer/README.md)

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
```

Prepare a complete recipe with the desired settings and dataset paths for the home
cluster. New dot-product runs use `model.qk_norm: true`; GLA and degree zero do not.
New runs use their mechanism's reference initialization, which `records create` fills in
when the recipe leaves `model.initialization` unset.
Untied embeddings, the scaffold's selective weight decay, and 400 warmup updates
(`optim.warmup_steps`) are canonical. The following allocates a fresh execution from an
existing QK-normalized Taylor recipe:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records create \
  --study experiments/attention_scaling \
  --config experiments/attention_scaling/runs/r044b3356f215/config.yaml \
  --label taylor-example
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

## Independent GDN erase gate

Run `rf7341031e3d5` asks whether GDN benefits from separate scalar erase and write
projections. It uses `model.gdn_erase_mode: independent`, with the erase projection
initialized from the write projection as a separate parameter. All original weights
match tied GDN at initialization. It preserves the canonical GDN block and reference
initialization, 400 warmup updates, standard training telemetry, and the 19,074-update
FineWeb recipe. The preselected control is `rf640fc13f3a8`; comparison findings and
current status belong in the [exploratory result](../../results/exploratory/gdn_polynomial/README.md#independent-erase-and-write-gates).
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

## Quadratic GDN query geometry

Run `rd40af7956426` tests `model.gdn_polynomial_degree: 2` with independent erase
gates. The original key-space map `P_t = I - lambda_t k_t k_t^T` composes with a
quadratic stored function; the added constant and quadratic tanh write projections
start at zero, preserving the reference-initialized linear GDN function. The run
keeps the 400-step warmup, gated output RMSNorm, and the complete training recipe of
the explicit linear control `rf7341031e3d5`. Its 211,335,104 parameters include
327,680 new write-projection weights. Methods, selection, and findings belong in
[the exploratory comparison](../../results/exploratory/gdn_polynomial/README.md).

Home cluster: Kennel. Qualification `218426` and dependent production `218427`
use source `46b776c9fc22e7ab5636e28a41f24105dcc508fd` on `mastiff`.
Develop and execute in `worktrees/gdn-quadratic`, with its
own locked CUDA environment. Full acceptance checks the production batch on four
H100 GPUs: effective batch 256, global microbatch 64, and 16 sequences per GPU.
The qualification job verifies GPU correctness, a full-recipe pilot, and exact
resumption, and rejects a projected training time above 22 hours. Production waits
for successful qualification and reuses matching acceptance evidence. Both jobs
use nice 0 on `standard`; the scheduler enforces the user's four-GPU cap.

From the experiment worktree, after committing the frozen inputs:

```bash
LAUNCH_COMMIT=$(git rev-parse HEAD)
QUALIFICATION_JOB=$(env -u XLA_PYTHON_CLIENT_MEM_FRACTION sbatch --parsable \
  --nice=0 --partition=standard --nodelist=mastiff --gres=gpu:h100:4 \
  --time=01:00:00 --cpus-per-task=32 --mem=256G \
  --export=ALL,XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --job-name=gdn-quadratic-qualification \
  --output=scaffolds/transformer/validation/logs/gdn-runtime-%j.log \
  scaffolds/transformer/validation/qualify_gdn_runtime.sbatch \
  experiments/attention_scaling/runs/rd40af7956426/config.yaml \
  "$LAUNCH_COMMIT" H100 24 4)

env -u XLA_PYTHON_CLIENT_MEM_FRACTION sbatch --parsable \
  --nice=0 --partition=standard --nodelist=mastiff --gres=gpu:h100:4 \
  --time=1-00:00:00 --cpus-per-task=32 --mem=256G \
  --export=ALL,XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --dependency="afterok:$QUALIFICATION_JOB" --kill-on-invalid-dep=yes \
  --job-name=rd40af7956426 \
  --output=experiments/attention_scaling/runs/rd40af7956426/training.log \
  scaffolds/transformer/launchers/kennel.sbatch \
  experiments.attention_scaling.acceptance \
  experiments/attention_scaling/runs/rd40af7956426/config.yaml \
  "$LAUNCH_COMMIT" 4 H100
```


## Cubic GDN query geometry

Run `r83c3394eb730` sets `model.gdn_polynomial_degree: 3` and keeps the complete
canonical recipe of quadratic run `rd40af7956426`: independent erase gates in
original key space, reference initialization, warmup 400, gated output RMSNorm,
Q/K L2 normalization, seed zero, and 19,074 updates on packed FineWeb. A separate
zero-initialized `tanh(W_3 x)` adds the cubic write. Its 211,498,944 parameters
include 163,840 more than quadratic; initial outputs and existing parameters match.
Methods and explicit comparisons remain in
[the exploratory topic](../../results/exploratory/gdn_polynomial/README.md).

Home cluster: Kennel. Develop and execute in `worktrees/gdn-cubic`, with its own
locked CUDA 13 environment. The run resumes checkpoint 7,000 using four B300 GPUs
on `long`, at nice 0. Its immutable recipe, run ID, and W&B identity are retained.
Global microbatch 64 and effective batch 256 are unchanged; local microbatch is 16.
The optimized runtime uses `bf16_values`, two warps, and one-CTA register-allocation
guidance. Transport geometry and accumulation remain FP32. These settings are
execution provenance, separate from the scientific recipe.

Acceptance in the production allocation checks the accurate backend against the
strict output/gradient oracles, the selected BF16 backend at its declared
tolerances, initialization, 20 full-recipe updates with validation, and exact
fresh-process checkpoint resume. Production starts only after these checks pass.
The source-change opt-in names the prior executed commit; all scientific settings
must still match, and the trainer updates resolved source provenance only after
restoring the full checkpoint and validating its step and data cursor.

From the experiment worktree, after committing the frozen inputs:

```bash
UV_PROJECT_ENVIRONMENT=$PWD/scaffolds/transformer/.venv-cuda13 \
  uv sync --project scaffolds/transformer --locked --extra cuda13 --link-mode hardlink
LAUNCH_COMMIT=$(git rev-parse HEAD)
env -u XLA_PYTHON_CLIENT_MEM_FRACTION sbatch --parsable \
  --nice=0 --partition=long --nodelist=akita --gres=gpu:b300:4 \
  --exclude=collie,dogo,mastiff,doodle,samoyed,terrier \
  --time=3-00:00:00 --cpus-per-task=32 --mem=256G \
  --export=ALL,XLA_PYTHON_CLIENT_PREALLOCATE=false,TRANSFORMER_HOST_STAGING=0,POLY_GDN_VALUE_PRECISION=bf16_values,POLY_GDN_NUM_WARPS=2,POLY_GDN_MIN_CTAS_PER_SM=1,TRANSFORMER_RESUME_SOURCE_COMMIT=6945d1b8c5f3e01755b2ef72f4f8474f7e529fc4 \
  --job-name=r83c3394eb730 \
  --output=experiments/attention_scaling/runs/r83c3394eb730/training.log \
  scaffolds/transformer/launchers/kennel.sbatch \
  experiments.attention_scaling.acceptance \
  experiments/attention_scaling/runs/r83c3394eb730/config.yaml \
  "$LAUNCH_COMMIT" 4 B300
```
