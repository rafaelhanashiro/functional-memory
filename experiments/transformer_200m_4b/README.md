# 200M Transformer setup

How do attention variants perform under the 200M-parameter recipe, and how do their
comparisons change with the training-token budget? The budget is a per-run setting:
the reference is 1× Chinchilla (about 4B tokens), and a run at another budget scales its
schedule with the data: updates, warmup and validation cadence. The folder keeps its original `transformer_200m_4b` name.
This self-contained study owns its [recipe](recipe.yaml), [run index](runs.csv),
[membership](sweep.yaml), and [results](results/README.md), including exploratory
diagnostics. Reusable model, attention, data, optimizer, and runtime code live in
the [Transformer scaffold](../../scaffolds/transformer/README.md).

## Reference recipe

![200M reference recipe with learned per-head QK score scaling](recipe.png)

The pretraining reference has 15 layers, width 896, 14 query/KV heads of width 64,
SwiGLU width 2304, context 2048, vocabulary 32,000, untied embeddings, and RoPE
theta 10,000, with no biases or dropout. Normalized softmax has 198,440,146 parameters.
The canonical QK setup is RMSNorm with a learned positive score scale for every
layer and head (`model.qk_norm_head_scale: true`). Q and K retain independent
per-layer, 64-channel RMSNorm gain vectors shared across heads. After normalization
and these gains, Q is multiplied by `alpha = exp(gamma)` before RoPE; K receives
no additional scalar. Scores retain division by sqrt(64), so each head's scale
multiplies the score once. The 210 head log scales start at zero, making every
initial multiplier one; the channel gains also start at one. Both are shared
across sequences and query positions and excluded from weight decay.
Normalization statistics, master parameters, and optimizer state use FP32;
computation uses BF16.

New pretraining runs use this setup. Historical run configurations and checkpoint
resumptions retain their recorded normalization; changing the canonical recipe
does not change those experiments. The architecture and optimizer below otherwise
retain the established recipe, including its previously selected learning rate.

Linear and embedding weights use a Gaussian with underlying standard deviation
0.02, truncated to [-0.04, 0.04] without variance compensation. Attention output
and FFN down projections additionally scale by 1/sqrt(30). AdamW uses betas
(0.9, 0.95), epsilon 1e-8, global gradient clipping at 1, and weight decay 0.1,
excluding embeddings, normalization gains, biases, and attention coefficients.
The untied output head decays.

The reference uses shuffled, EOS-packed FineWeb-Edu data tokenized with the
Llama-2 tokenizer, seed 42, and global batch 256. Its token budget is 1× Chinchilla:
7,570 updates (3,968,860,160 prediction tokens). Peak LR is 0.001, with 455 warmup
updates and cosine decay to 0.0001 over the run. Validation uses 9,961,472 tokens every 100 updates through
1,500, then every 500 and at the final update. Individual diagnostic recipes
specify their own fitting objective, trainable parameters, and stopping rule.

### Token budgets

A run at another budget is the data-scaled reference: it copies its reference recipe
and multiplies `optim.total_steps` (and so the cosine horizon), `optim.warmup_steps`
(6.01% of the run), and the validation cadence (`train.eval_early_every`,
`train.eval_early_until`, `train.eval_every`) by the budget's factor, rounding warmup
to whole updates. Peak and floor learning rate, weight decay, batch, data order, seed,
and checkpoint interval stay the same.
With the same shuffle seed, a longer run first sees exactly the tokens of a
shorter one. The 12.0B-token training set covers up to about 22,900 updates
without repeating data. `runs.csv` records each run's `training_steps`; compare
runs at the same budget, or the same variant across budgets.

| Budget | Updates | Warmup | Validation | Prediction tokens |
| --- | ---: | ---: | --- | ---: |
| 1× Chinchilla (reference) | 7,570 | 455 | every 100 through 1,500, then every 500 | 3,968,860,160 |
| 2.5× Chinchilla | 18,925 | 1,138 | every 250 through 3,750, then every 1,250 | 9,922,150,400 |

The 2.5× runs repeat five 1× runs that use the canonical QK setup at the 2.5× budget.
All five are complete; the [token_budget topic](results/exploratory/token_budget/README.md)
compares them with their 1x partners. Each ran on one GPU:

| 2.5× run | Label | 1× partner |
| --- | --- | --- |
| `r8ccda981be2b` | `softmax-2.5x` | none with the canonical QK setup |
| `rfe12e5c36e94` | `exp-2.5x` | `r6a76e7fbde45` (`exp-head-scale`) |
| `r4f1682c1b54d` | `fixed-m4-2.5x` | `r816ab0fbe119` (`fixed-m4-head-scale`) |
| `raa81f92b6aaf` | `static-m4-2.5x` | `r4339a245820e` (`static-m4-head-scale`) |
| `reb9b5be4ff1b` | `query-m4-2.5x` | `r2b22f2ff6461` (`query-m4-head-scale`) |

They exclude the H100 PCIe node `mastiff`, whose throughput projects past 24 hours,
and the B200 node `shiba`. Each requests 24 hours on `standard,long`, except exp,
which requests 48 hours on `long` (its pilot projects about 23 hours):

```bash
sbatch --parsable --partition=standard,long --gres=gpu:1 --cpus-per-task=16 --mem=128G \
  --time=24:00:00 --exclude=mastiff,collie,samoyed,doodle,terrier,shiba --job-name=<run_id> \
  --output=experiments/transformer_200m_4b/runs/<run_id>/training.log \
  --error=experiments/transformer_200m_4b/runs/<run_id>/training.log \
  scaffolds/transformer/launchers/kennel.sbatch experiments.transformer_200m_4b.acceptance \
  experiments/transformer_200m_4b/runs/<run_id>/config.yaml "$(git rev-parse HEAD)" 1 B300,H100
```

Each record's `submission` holds its exact command and partition.

Regenerate the reference image from the current recipe with:

```bash
scaffolds/transformer/.venv/bin/python experiments/transformer_200m_4b/plot_recipe.py
```

## Exploratory diagnostics

The [coefficient fitting topic](results/exploratory/coefficient_oracle/README.md)
compares the fitting capacity of free query coefficients and the trained tanh
query generator on one sequence, and coefficients shared across a full batch,
each with its pretrained backbone frozen.
Run `r3c0f21ab7dce` is the completed free-coefficient fit from the static checkpoint;
`rd7bf9bc37cb6` is the completed fit of the original query generator from the query-only checkpoint.
`r0963b3e6610b` fits one position-dependent coefficient table shared across the
parent recipe's entire 256-sequence batch. Static fits `r22854de31b96` (one
sequence) and `r5b28697413fa` (full batch) additionally share coefficients across
query positions. Their fitting metrics enter no canonical pretraining comparison.

Allocate the query-generator fit in its own worktree (once), commit the frozen
inputs, then submit one B300 or H100 job:

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_coefficient_fit prepare \
  --parent /data/cl/u/rafah/home/projects/functional-memory/worktrees/transformer-200m-4b/experiments/transformer_200m_4b/runs/r98d5895c02d0
# Commit the allocated recipe and implementation before submitting.
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_coefficient_fit submit \
  rd7bf9bc37cb6 --partition debug,standard,long --time-limit 02:00:00
```

Allocate the shared-coefficient batch fit with the same parent in its own worktree:

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_coefficient_fit prepare \
  --parent /data/cl/u/rafah/home/projects/functional-memory/worktrees/transformer-200m-4b/experiments/transformer_200m_4b/runs/r98d5895c02d0 \
  --parameterization shared_batch_query_coefficients
# Commit frozen inputs before submitting; resumptions retain this run ID.
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_coefficient_fit submit \
  r0963b3e6610b --partition debug,standard,long --time-limit 02:00:00
```

Allocate the independent per-sequence batch fit in its own worktree. Run
`r7505c6d49711` uses the same query-only parent and 256 fitting sequences, with
separate coefficient tables for every sequence. Its completed fit automatically
produces the CPU coefficient-shape analysis in the existing exploratory topic.

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_coefficient_fit prepare \
  --parent /data/cl/u/rafah/home/projects/functional-memory/worktrees/transformer-200m-4b/experiments/transformer_200m_4b/runs/r98d5895c02d0 \
  --parameterization free_per_example_query_coefficients --sequence-count 256
# Commit frozen inputs before submitting; resumptions retain this run ID.
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_coefficient_fit submit \
  r7505c6d49711 --partition long --gpu-type B300 --time-limit 06:00:00
```

Allocate both static fits in one isolated worktree before committing or launching:

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_coefficient_fit prepare \
  --parent /data/cl/u/rafah/home/projects/functional-memory/worktrees/transformer-200m-4b/experiments/transformer_200m_4b/runs/r98d5895c02d0 \
  --parameterization static_coefficients --sequence-count 1
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_coefficient_fit prepare \
  --parent /data/cl/u/rafah/home/projects/functional-memory/worktrees/transformer-200m-4b/experiments/transformer_200m_4b/runs/r98d5895c02d0 \
  --parameterization static_coefficients --sequence-count 256
# Commit both recipes and implementation before either independent submission.
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_coefficient_fit submit \
  r22854de31b96 --partition debug,standard,long --time-limit 02:00:00
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_coefficient_fit submit \
  r5b28697413fa --partition debug,long --time-limit 02:00:00
```

The shared fitting runtime qualifies the frozen-backbone update and checkpoint
resume before production. The launcher checks frozen inputs at each start and
requeue and selects the locked CUDA environment for its GPU.

Degree-four query-generator fit `r1f2f749ff099` repeats the same single-sequence
diagnostic from its trained degree-four parent. It trains only the coefficient
generator, keeping the backbone frozen. Its settings and findings belong to the
same [coefficient fitting topic](results/exploratory/coefficient_oracle/README.md).
In its isolated worktree:

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_coefficient_fit prepare \
  --parent /data/cl/u/rafah/home/projects/functional-memory/worktrees/transformer_200m_4b_m4/experiments/transformer_200m_4b/runs/r49b7943aee1b
# Already allocated; use this ID for resumptions and commit frozen inputs before launch.
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_coefficient_fit submit \
  r1f2f749ff099 --partition standard --time-limit 02:00:00
```

## Matched Taylor continuations

The [delayed coefficient-learning topic](results/exploratory/taylor_static_continuation/README.md)
compares 500 updates with fixed Taylor against learning static coefficients from
the same pretrained Taylor checkpoint. It preserves the backbone optimizer and
next training-data position, with a fresh Adam clock for newly introduced
coefficients. Both continuations use the shared trainer and acceptance procedure.

Allocate the pair once in its isolated worktree, then commit all frozen inputs
before either independent submission:

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_continuation prepare \
  --parent /data/cl/u/rafah/home/projects/functional-memory/worktrees/transformer-200m-4b/experiments/transformer_200m_4b/runs/r019ca96d6533
# The allocated IDs below identify this pair; do not allocate it again for a requeue.
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_continuation submit \
  r28fb40a1d0e9 --partition standard,long --time-limit 06:00:00
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_continuation submit \
  r0332cff0b0c0 --partition standard,long --time-limit 06:00:00
```

The two recorded continuations are complete; findings and the open research
decision are in the topic. Fresh reproductions receive new run IDs. The launcher
defaults to six hours, including the shared acceptance procedure's two-hour reserve.
Each job requests one B300 or H100 GPU. The launcher's frozen-source check and
shared runtime qualification run on every allocation; the scientific global batch
stays fixed through gradient accumulation. See the topic for comparison settings,
findings, and regeneration.

The [query continuation topic](results/exploratory/taylor_query_continuation/README.md)
compares degree-four query-dependent coefficients with a matched fixed-Taylor
continuation. Both train all backbone parameters for the same 500 additional
updates. The query generator starts with the study's zero-amplitude/random-projection
initialization, reproducing Taylor exactly. Prepare in its own worktree, then
commit frozen inputs before independent submission:

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_continuation prepare \
  --conditioning query \
  --parent /data/cl/u/rafah/home/projects/functional-memory/worktrees/transformer_200m_4b_m4/experiments/transformer_200m_4b/runs/r79da929de109
# Already allocated below; reuse these IDs for resumptions.
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_continuation submit \
  rf311afb9339e --conditioning query --partition standard --time-limit 06:00:00
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_continuation submit \
  r6186c797b4db --conditioning query --partition standard --time-limit 06:00:00
```

The [per-token query scale topic](results/exploratory/token_query_scale_continuation/README.md)
compares a bounded per-token score scale on Q, `exp(gamma_h + a_h cos(w_h, x_t))`, with a
matched fixed-Taylor continuation of the canonical head-scaled parent. Both train all
backbone parameters for 2,000 additional updates; the gate starts closed (`a_h = 0`) and
its parameters use their own constant learning rate of 0.001. Prepare in its own
worktree, then commit frozen inputs before independent submission:

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_continuation prepare \
  --conditioning token_scale \
  --parent /data/cl/u/rafah/home/projects/functional-memory/experiments/transformer_200m_4b/runs/r816ab0fbe119
# Already allocated below; reuse these IDs for resumptions.
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_continuation submit \
  r82203a85ef8e --conditioning token_scale --partition standard --time-limit 10:00:00
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_continuation submit \
  ra988273933f7 --conditioning token_scale --partition standard --time-limit 20:00:00
```

The acceptance pilot projects the allocation from its slowest steady update, so a
single checkpoint-save stall during the pilot can project far beyond the real cost
(about 3.2 s per update for both branches on an H100 PCIe); the longer limit only
clears that projection.

## Teacher-activation coefficient distillation

The open [coefficient-oracle topic](results/exploratory/coefficient_oracle/README.md#dense-distillation-from-the-free-teacher)
documents the scientific protocol and its two batches: the fitting batch, and one
fresh batch that scores the selected generator once. `rec777591a61b` distills the degree-four B256 free teacher into the
existing dense generator with a frozen Taylor backbone. Home cluster: **Kennel**.
Its source checkpoints remain in the primary checkout; all new outputs belong
to the run in its isolated worktree until completion.

From the experiment worktree, after syncing both locked CUDA environments:

```bash
# Already allocated as rec777591a61b; prepare is only for a fresh execution.
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_coefficient_distillation prepare \
  --teacher /data/cl/u/rafah/home/projects/functional-memory/experiments/transformer_200m_4b/runs/rda6029c3286f
# Commit the frozen execution set before submission.
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.launch_coefficient_distillation submit rec777591a61b
```

The launcher uses one available H100 or B300 on `standard`, with no scheduling
dependency, and runs full-size qualification before fitting. `--gpu-type h100`
or `--gpu-type b300` pins a GPU type if required. Requeues restore the latest
generator, Adam state, fixed data identities, and the lowest-regression-error
candidate. Completed runs retain that selected generator and refer to the unchanged
parent checkpoint for their frozen backbone.

## Coefficient forms and forget gate

Three degree-four runs copy `query-m4-head-scale` (`r2b22f2ff6461`) and change one part
of its query attention; see the [coefficient forms](results/exploratory/coefficient_forms/README.md)
and [forget gate](results/exploratory/forget_gate/README.md) topics.

| Run | Label | Change |
| --- | --- | --- |
| `r1d578d107115` | pure-tanh-m4 | `attention_coefficient_activation: pure_tanh`, unscaled, no `b` or `a` |
| `r917f3650296e` | affine-sigmoid-m4 | `attention_coefficient_activation: affine_sigmoid` |
| `r111843b4c7cc` | query-m4-forget-gate | `attention_forget_gate: true` on `attention_backend: blocked` |

Pure tanh and affine sigmoid launch from `worktrees/coefficient-forms-4b`, and the forget
gate from `worktrees/forget-gate-4b` (branch `claude/forget-gate-4b`), whose diagnostics
summarize the gate's per-head statistics. After committing the inputs, submit each run on
one B300 or H100, excluding the B200 node `shiba` (and, for the slower blocked forget gate,
the H100 PCIe node `mastiff`):

```bash
sbatch --parsable --partition=<standard|long> --gres=gpu:1 --cpus-per-task=16 --mem=128G \
  --time=24:00:00 --exclude=collie,samoyed,doodle,terrier,shiba --job-name=<run_id> \
  --output=experiments/transformer_200m_4b/runs/<run_id>/training.log \
  --error=experiments/transformer_200m_4b/runs/<run_id>/training.log \
  scaffolds/transformer/launchers/kennel.sbatch experiments.transformer_200m_4b.acceptance \
  experiments/transformer_200m_4b/runs/<run_id>/config.yaml "$(git rev-parse HEAD)" 1 B300,H100
```

Each record's `submission` holds its exact command and partition.

## Execution and records

Home cluster: **Kennel**. Data manifests live under
`/data/cl/u/rafah/home/datasets/fineweb_12b/`; common hashes are in `sweep.yaml`.
Retained checkpoints and logs live under this study in the primary Kennel checkout,
at the locations recorded in each run. The frozen recipe and execution record
identify the source commit, dependency lock, source checkpoint, and submitted
command. Reproducing a completed execution uses that recorded source; a fresh
execution receives a new ID and its own frozen recipe.

Develop and execute new runs in an isolated worktree with its own locked
environments. Consult [CLUSTERS.md](../../CLUSTERS.md) for allocation, storage,
and CUDA environments, and [AGENTS.md](../../AGENTS.md) for lifecycle rules.

From the repository/worktree root:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records index experiments/transformer_200m_4b
scaffolds/transformer/.venv/bin/python -m scaffolds.records validate experiments/transformer_200m_4b r3c0f21ab7dce
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_200m_4b/results/exploratory/coefficient_oracle
```

Refresh selection pins only after an intentional update to their scientific inputs.
