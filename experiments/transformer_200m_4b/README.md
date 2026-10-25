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

Three normalized-softmax pilots test peaks 0.0005, 0.001, and 0.002, with respective
floors 0.00005, 0.0001, and 0.0002. Each stops at update 1,500 while retaining the
full 7,570-update schedule and resumable state. Their combined budget is
2,359,296,000 prediction tokens. The canonical LR is **pending** until
`results/lr_selection.yaml` contains a non-null `canonical_learning_rate`.
The placeholder in `recipe.yaml` is not a selection result.

The selection rule is the lowest finite held-out NLL at step 1,500, breaking an
exact tie toward the lower peak. Numerical failures remain visible and are
ineligible; incomplete pilots and operational failures stop automatic selection.
This is an early screening decision, not evidence of the best LR after 4B tokens.
The selected LR and its 10% floor apply unchanged to these nine full-budget runs:

| Variant | Degrees | Backend |
| --- | --- | --- |
| Unnormalized exponential + head RMSNorm | exact exponential | fused |
| Fixed Taylor + head RMSNorm | 8, 12 | Mosaic |
| Learned static + head RMSNorm | 8, 12 | Mosaic |
| Learned query + head RMSNorm | 8, 12 | Mosaic |
| Learned query–key + head RMSNorm | 8, 12 | fused |

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

Home cluster: Kennel. Data manifests live under
`/data/cl/u/rafah/home/datasets/fineweb_12b/`; hashes are in `sweep.yaml` and execution
evidence. Retained artifacts belong under this study in the primary checkout on
Kennel. Develop/execute in the study's own worktree. Sync both locked environments
as specified in the repository's `CLUSTERS.md`; B300 uses CUDA 13 / cuBLAS >= 13.2.

From the study worktree root:

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.pipeline prepare-pilots
# Review and commit all scaffold/execution changes and the allocated recipes.
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.pipeline submit-pilots
```

Each job requests two H100/B300 GPUs on the low-priority `sweep` partition, with
microbatch 16 and gradient accumulation to the fixed global batch. The shared
launcher performs allocation-specific acceptance before training: CPU/accelerator
tests, initialization identity, a short full-model pilot, and exact fresh-process
checkpoint resumption. New checks cover actual 14-head, 2,048-token attention
forward/backward against the blocked reference, including learned gains and every
coefficient order. Acceptance failure prevents production.

The submitted CPU coordinator depends on all three pilot jobs exiting. Only then
does it select the LR, update the future recipe, allocate all nine comparison runs,
commit their frozen execution inputs, and submit two lanes (at most four new GPUs
concurrently). It never rewrites a launched run's configuration. Pilot checkpoint
resumption remains tied to the pilot's original source commit. Existing jobs in
other worktrees are untouched. An operational failure requires diagnosis before
rerunning `pipeline advance --submit`; it never silently chooses a partial pilot.
After all nine comparison jobs exit, a dependent CPU report job refreshes the
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
