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
Untied embeddings and the scaffold's selective weight decay are canonical. The
following allocates a fresh execution from an existing QK-normalized Taylor recipe:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records create \
  --study experiments/attention_scaling \
  --config experiments/attention_scaling/runs/r93c9d0814322/config.yaml \
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

### Tanh coefficients, degrees 1–3

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

The new tanh mechanisms use direct fused query-dependent polynomial attention.
Source `254cab946e637cd6cd5c004b6976351b1dee3651` runs as job `210243` on four B300
GPUs (`akita`, `standard`), job `210244` on four H100 GPUs (`dogo`, `long`), and
job `210245` on four B300 GPUs (`akita`, `long`), respectively. All have nice 0.
All three passed full acceptance, including full-recipe pilots and exact
fresh-process checkpoint resumption, and are running production training.
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

The shared acceptance checks must pass on each allocation before production starts.
Question, input selections, outcomes, and the open scientific decision live in the
[comparison README](../../results/exploratory/input_dependent_coefficients/README.md).

To update a figure, change its explicit run selection, run
`python -m scaffolds.records refresh results/<analysis>` in the scaffold environment,
and use that result collection's regeneration command.
