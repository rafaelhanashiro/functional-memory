# 200M Transformer / 4B-token setup

This self-contained study supports deliberate attention experiments at a smaller
training budget. Its runs and [results](results/README.md) live here; reusable model,
attention, data, optimization, and runtime code live in `scaffolds/transformer/`.
The earlier attention studies and their result selections retain their own recipes.

## Recipe

`recipe.yaml` defines the reference: 15 layers, width 896, 14 query/KV heads of width
64, SwiGLU width 2304, context 2048, 32,000 vocabulary, untied embeddings, RoPE theta
10,000, no biases/dropout, and 198,439,936 parameters for normalized softmax. Q and K
use independent per-layer 64-channel RMSNorm gains, shared across heads and applied
before RoPE; scores retain division by sqrt(64). All normalization statistics are
FP32. Computation uses BF16; master parameters and optimizer state are FP32.

Linear and embedding weights use a Gaussian with underlying standard deviation
0.02, truly truncated to [-0.04, 0.04], without variance compensation (actual standard
deviation about 0.0175925). Attention output and FFN down projections then scale by
1/sqrt(30). Norm gains start at one. AdamW uses betas (0.9, 0.95), epsilon 1e-8,
global gradient clipping at 1, and decay 0.1, excluding embeddings, normalization
gains, biases, and attention coefficients. The untied LM head decays.

All runs use the existing shuffled, EOS-packed FineWeb-Edu/Llama-2-tokenized data,
seed 42, global batch 256, and 7,570 updates (3,968,860,160 prediction tokens).
Warmup is 455 updates, followed by cosine decay to 10% of peak LR. Validation uses
19 batches = 9,961,472 held-out prediction tokens, every 100 steps through step
1,500 and every 500 afterwards, plus the final step. This cadence is frozen before
launch, including for any pilot resumption. Checkpoint interval is 500 updates.

## LR selection and requested comparison

**The canonical peak LR is 0.001 and the minimum LR is 0.0001.** New runs in this
setup inherit both from `recipe.yaml`; change them only for an explicitly requested
LR experiment. The winning softmax pilot is `ra2b851b0dd7e`, selected by the lowest
held-out NLL at step 1,500 on the full 7,570-step schedule. This is an early screening
choice, not an established full-budget optimum.

The discarded pilots at peaks 0.0005 and 0.002 (each with a 10% floor) reached
step-1,500 validation PPL 22.0744 and 22.0731, versus 21.8184 at the selected 0.001;
all used seed 42 and the same recipe. Their outcomes are retained only in this note.

The canonical LR and its 10% floor apply to these thirteen full-budget variants:

| Variant | Degrees | Backend |
| --- | --- | --- |
| Unnormalized exponential + head RMSNorm | exact exponential | fused |
| Fixed Taylor + head RMSNorm | 4, 8, 12 | Mosaic |
| Learned static + head RMSNorm | 4, 8, 12 | Mosaic |
| Learned query + head RMSNorm | 4, 8, 12 | Mosaic |
| Learned query–key + head RMSNorm | 4, 8, 12 | fused |

For order r, static coefficients are `c_r = c_tilde_r / r!`. Query coefficients
are `c_r(x_t) = [b_r + a_r tanh(W_r x_t)] / r!`; **both b and a** are scaled.
Query–key uses `c_r(x_t) [1 + a_key_r tanh(W_key_r x_i)]`. The dimensionless key
factor does not introduce a second factorial. Learned parameters are independent
per layer/head. Initialize stored `c_tilde` or `b` at one, amplitudes at zero,
and coefficient projections independently from N(0, 1/896); there is no tanh bias.
Thus every polynomial starts at its fixed Taylor counterpart. Projection
initialization follows this coefficient parameterization, not the backbone's
truncated Gaussian. All comparison variants use per-head attention-output RMSNorm.

## Execution on Kennel

Home cluster: Kennel, except the query–key runs (ORCD, below). Kennel data manifests live under
`/data/cl/u/rafah/home/datasets/fineweb_12b/`; hashes are in `sweep.yaml` and execution
evidence. Retained artifacts belong under this study in the primary checkout on
Kennel. Develop/execute in the study's own worktree. Sync both locked environments
as specified in the repository's `CLUSTERS.md`; B300 uses CUDA 13 / cuBLAS >= 13.2.

Allocate new runs from the canonical recipe in a fresh experiment worktree:

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m scaffolds.records create \
  --study experiments/transformer_200m_4b \
  --config experiments/transformer_200m_4b/recipe.yaml --label my-new-run
```

Set the requested attention variant before launch, preserving the inherited LR and
floor. Commit the complete run recipe before using the shared launcher. Rerunning
an LR candidate uses this same recipe with an explicit peak/floor override and a
fresh run ID; an ordinary new run does not rerun LR selection.

New jobs request one GPU, with global microbatch 16 and gradient accumulation to
the fixed global batch of 256. Schedule independent jobs in `standard` and the
available `long` allowance before using `sweep` for overflow, after checking the
live limits and allocations described in [CLUSTERS.md](../../CLUSTERS.md). The shared
launcher performs allocation-specific acceptance before training: CPU/accelerator
tests, initialization identity, a short full-model pilot, and exact fresh-process
checkpoint resumption. New checks cover actual 14-head, 2,048-token attention
forward/backward against the blocked reference, including learned gains and every
coefficient order. Acceptance failure prevents production.

The comparison recipes are frozen with the selected LR. Independent training
runs have no inter-run scheduling dependencies; Slurm schedules them according to
available resources and cluster policy. Each run retains its own frozen configuration
and source commit; resumed jobs preserve those inputs.
After the selected comparison jobs exit, a dependent CPU report job refreshes the
explicit selections and regenerates the figures and summary, including failures
and incomplete outcomes.

Run records contain job IDs, runtime/acceptance evidence, and artifact provenance;
`runs.csv` is derived. The Slurm coordinator log is `selection.log` (ignored).

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records index experiments/transformer_200m_4b
scaffolds/transformer/.venv/bin/python -m scaffolds.records validate experiments/transformer_200m_4b
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_200m_4b/results
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.results.plot
```

## Query–key execution

Query–key degrees 4, 8 and 12 are `r097fcda6a3af`, `rf0a5652acaa6` and `ree2eb2e4287e`,
with home cluster ORCD. Their recipes read ORCD's FineWeb manifests, byte-identical to
Kennel's; model, coefficients, seed, global batch and LR schedule match the other
comparisons. Each runs on one H100 80GB HBM3, where the fused kernel selects the
qualified 32-by-32 backward at all three degrees
([tiles](../../scaffolds/transformer/validation/README.md#query-key-polynomial-backward-tiles),
[B300 production evidence for degrees 8 and 12](../../scaffolds/transformer/validation/README.md#b300-query-key-production-qualification)).
Jobs exclude `node3009`/`node3209` (throttling GPUs) and the H100 NVL nodes
`node1708`/`node1709`, requeue every 12 hours and resume from their latest checkpoint;
every allocation repeats acceptance. Submit from the ORCD worktree root after
committing the inputs, and record each job in its run's `submission`:

```bash
for run in r097fcda6a3af rf0a5652acaa6 ree2eb2e4287e; do
  sbatch --parsable --job-name="$run" --output="experiments/transformer_200m_4b/runs/$run/training.log" \
    --gres=gpu:h100:1 --exclude=node3009,node3209,node1708,node1709 \
    scaffolds/transformer/launchers/orcd.sbatch experiments.transformer_200m_4b.acceptance \
    "experiments/transformer_200m_4b/runs/$run/config.yaml" "$(git rev-parse HEAD)" 1 H100
done
```

## Dense query coefficient sweep

Seven exploratory runs compare dense query coefficients, whose K tanh features are mixed
into every polynomial coefficient, with the original query generator and fixed Taylor; see
the [topic](results/exploratory/dense_query_coefficients/README.md) for runs, method and
results. Each copies the canonical `query-m4` or `fixed-m4` recipe and changes only the
degree and, for dense runs, `attention_coefficient_readout: dense` with
`attention_coefficient_feature_count: K`. Each requests one GPU (B300 or H100) and runs
from this worktree root after committing its inputs; scheduling priority 1 (dense, K = m+1)
before 2 (dense, K = 9) before 3 (degree-two controls) is set with `--nice`:

```bash
sbatch --parsable --partition=standard --nice=0 --gres=gpu:1 --cpus-per-task=8 --mem=96G \
  --time=1-00:00:00 --job-name=<run_id> --output=experiments/transformer_200m_4b/runs/<run_id>/training.log \
  scaffolds/transformer/launchers/kennel.sbatch experiments.transformer_200m_4b.acceptance \
  experiments/transformer_200m_4b/runs/<run_id>/config.yaml "$(git rev-parse HEAD)" 1 B300,H100
```

Each record's `submission` holds its exact command, partition and priority.

### Dense coefficients with per-head QK scaling

Two additional full-budget runs in the same exploratory topic test degree four and
eight with nine tanh features each. They retain the original QK RMSNorm and shared
per-layer channel gains, add a learned positive score scale per layer/head, and
multiply the dense coefficient mixing contribution by one-third. The new fields
are `model.qk_norm_head_scale: true` and
`model.attention_coefficient_readout_scale: 0.3333333333333333`.

The head log scales start at zero (score multipliers one); the dense mixing matrix
starts at zero, and the Taylor bias is unchanged. The two runs therefore preserve
the original initial function, training data, seed, learning-rate schedule, batch,
and total budget. They test both requested changes together; separate effects
cannot be identified from this pair. The comparison and findings stay in the
[dense coefficient topic](results/exploratory/dense_query_coefficients/README.md).

The allocated IDs are `r04a5b2a3e8dd` (m=4) and `rfccffdd239dd` (m=8), both
with K=9. From `worktrees/dense-qk-scale-4b`, after syncing both locked
environments and committing the frozen inputs, submit each independently:

```bash
for run in r04a5b2a3e8dd rfccffdd239dd; do
  sbatch --parsable --partition=standard --gres=gpu:1 --cpus-per-task=16 --mem=128G \
    --time=24:00:00 --job-name="$run" \
    --output="experiments/transformer_200m_4b/runs/$run/training.log" \
    --error="experiments/transformer_200m_4b/runs/$run/training.log" \
    scaffolds/transformer/launchers/kennel.sbatch experiments.transformer_200m_4b.acceptance \
    "experiments/transformer_200m_4b/runs/$run/config.yaml" "$(git rev-parse HEAD)" 1 B300,H100
done
```

The shared launcher qualifies the new head-scale and dense-readout gradients,
reference initialization, full-size pilot, and checkpoint resumption on the
allocated GPU before production. Keep these run IDs for resumptions.

### Controls with matching per-head QK scaling

Two fresh full-budget runs provide controls for the current degree-four dense run
`r04a5b2a3e8dd`: `r816ab0fbe119` (`fixed-m4-head-scale`) uses fixed degree-four
Taylor coefficients, and `r6a76e7fbde45` (`exp-head-scale`) uses the exact
unnormalized exponential. Both retain per-head attention-output RMSNorm, the
existing QK RMSNorm channel gains, and `model.qk_norm_head_scale: true`. They use
`transformer_truncated_reference`, seed 42, global batch 256, and the same
7,570-update learning-rate schedule. The fixed Taylor control uses Mosaic; the
exponential reference uses the fused backend.

The fixed Taylor comparison isolates the coefficient-generator choice under the
same QK setup. The current dense run retains its nine features and one-third dense
mixing contribution; the controls have fixed coefficients and no dense readout.
The exponential run supplies a reference with the same QK and output scaling.
Findings and explicit selections stay in the existing
[dense coefficient topic](results/exploratory/dense_query_coefficients/README.md).

From `worktrees/qk-head-controls-4b`, after syncing both locked environments and
committing the frozen inputs, submit each independently:

```bash
for run in r6a76e7fbde45 r816ab0fbe119; do
  sbatch --parsable --partition=standard --gres=gpu:1 --cpus-per-task=16 --mem=128G \
    --time=24:00:00 --job-name="$run" \
    --output="experiments/transformer_200m_4b/runs/$run/training.log" \
    --error="experiments/transformer_200m_4b/runs/$run/training.log" \
    scaffolds/transformer/launchers/kennel.sbatch experiments.transformer_200m_4b.acceptance \
    "experiments/transformer_200m_4b/runs/$run/config.yaml" "$(git rev-parse HEAD)" 1 B300,H100
done
```

Each allocation must pass the shared launcher's acceptance checks, full-model
pilot, and checkpoint-resumption checks before production starts from update zero.
