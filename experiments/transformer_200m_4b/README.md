# 200M Transformer setup

How do attention variants perform under the 200M-parameter recipe, and how do their
comparisons change with the training-token budget? The budget is a per-run setting:
the reference is 1× Chinchilla (about 4B tokens), and runs at other budgets change only
`optim.total_steps`. The folder keeps its original `transformer_200m_4b` name.
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

A run at another budget copies its reference recipe and changes only
`optim.total_steps` (and so the cosine horizon); warmup (455 updates), peak and
floor learning rate, data order, seed, and validation cadence stay the same.
With the same shuffle seed, a longer run first sees exactly the tokens of a
shorter one. The 12.0B-token training set covers up to about 22,900 updates
without repeating data. `runs.csv` records each run's `training_steps`; compare
runs at the same budget, or the same variant across budgets.

| Budget | Updates | Prediction tokens |
| --- | ---: | ---: |
| 1× Chinchilla (reference) | 7,570 | 3,968,860,160 |
| 2.5× Chinchilla | 18,925 | 9,922,150,400 |

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

## Teacher-activation coefficient distillation

The open [coefficient-oracle topic](results/exploratory/coefficient_oracle/README.md#dense-distillation-from-the-free-teacher)
documents the scientific protocol, fitting metrics, and independent validation/test
selection. `rec777591a61b` distills the degree-four B256 free teacher into the
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
generator, Adam state, fixed data identities, and validation/capacity candidates.
Completed runs retain the validation-best generator and refer to the unchanged
parent checkpoint for their frozen backbone.

## Query–key polynomial pretraining

Three full-budget pretraining runs use unnormalized learned polynomial attention
whose order-r coefficient depends on both the query and the key,
`c_r(x_t) [1 + a_key_r tanh(W_key_r x_i)]` with `c_r(x_t) = [b_r + a_r tanh(W_r x_t)] / r!`.
Offsets start at one and amplitudes at zero, so each run starts at fixed Taylor;
coefficient projections start from N(0, 1/896). Learned parameters are independent
per layer and head, and the attention output uses per-head RMSNorm.

| Degree | Run |
| --- | --- |
| 4 | `r097fcda6a3af` |
| 8 | `rf0a5652acaa6` |
| 12 | `ree2eb2e4287e` |

Their recipe precedes the per-head QK score scale: QK RMSNorm uses the per-layer
gains only. Home cluster: **ORCD**, with ORCD's byte-identical FineWeb manifests.
Each ran on one H100 80GB HBM3 with the fused query–key kernel at source commit
`fc0903e` (branch `claude/transformer-200m-4b-query-key-orcd`). Retained
checkpoints and logs are under ORCD's artifact root,
`/orcd/pool/006/rafah/functional_memory/artifacts/`, at the paths recorded in each
run. From a worktree at the source commit, each run was submitted as below; the
recorded commands also pass a trailing `96`, which this launcher ignores.

```bash
for run in r097fcda6a3af rf0a5652acaa6 ree2eb2e4287e; do
  sbatch --parsable --job-name="$run" --output="experiments/transformer_200m_4b/runs/$run/training.log" \
    --gres=gpu:h100:1 --exclude=node3009,node3209,node1708,node1709,node4508,node2801,node3409 \
    scaffolds/transformer/launchers/orcd.sbatch experiments.transformer_200m_4b.acceptance \
    "experiments/transformer_200m_4b/runs/$run/config.yaml" "$(git rev-parse HEAD)" 1 H100
done
```

### With the per-head QK score scale

Three more runs repeat these recipes with the canonical QK setup: the same
per-layer gains plus a learned positive score scale per layer and head
(`model.qk_norm_head_scale: true`). Nothing else in the recipe changes; the fused
query–key kernel is kept (allocated with `--allow-triton-attention`) so that each
run pairs with its predecessor above. The degree-four fixed, static, and query
runs with the same QK setup are `r816ab0fbe119`, `r4339a245820e`, and
`r2b22f2ff6461`; no degree-8 or degree-12 baseline uses it.

| Degree | Run | Without the head scale |
| --- | --- | --- |
| 4 | `r501f5cca0d4c` | `r097fcda6a3af` |
| 8 | `r536dee0929d5` | `rf0a5652acaa6` |
| 12 | `r2459b843ca81` | `ree2eb2e4287e` |

Home cluster: **ORCD**, one H100 80GB HBM3 each, from branch
`claude/transformer-200m-4b-query-key-head-scale-orcd`. Its scaffold is newer than
`fc0903e`: the attention kernels, launcher, and lock are unchanged, and the query
coefficient readout evaluates the same FP32 expression. GPU acceptance runs the
fused query–key head-scale backend tests at degrees 4, 8, and 12 before training.
The node exclusions are those of the runs above: the throttled `node3009` and
`node3209`, the H100 NVL nodes `node1708`, `node1709`, and `node4508`, `node2801`
(failed CUDA initialization), and `node3409` (half the expected update rate).
The pilot projects every remaining validation from its single compile-inclusive
evaluation, which exceeds the default 48-hour horizon on a cold compilation cache,
so these runs qualify against a 96-hour horizon through the runtime override:

```bash
for run in r501f5cca0d4c r536dee0929d5 r2459b843ca81; do
  TRANSFORMER_ALLOCATION_SECONDS=345600 sbatch --parsable --job-name="$run" \
    --output="experiments/transformer_200m_4b/runs/$run/training.log" \
    --gres=gpu:h100:1 --exclude=node3009,node3209,node1708,node1709,node4508,node2801,node3409 \
    scaffolds/transformer/launchers/orcd.sbatch experiments.transformer_200m_4b.acceptance \
    "experiments/transformer_200m_4b/runs/$run/config.yaml" "$(git rev-parse HEAD)" 1 H100
done
```

## Execution and records

Home cluster: **Kennel**, except the query–key pretraining runs above (ORCD). Data manifests live under
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
