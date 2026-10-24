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

The canonical LR and its 10% floor apply to these nine full-budget variants:

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
the fixed global batch of 256. Each comparison is scheduled independently on the
low-priority `sweep` partition. The shared
launcher performs allocation-specific acceptance before training: CPU/accelerator
tests, initialization identity, a short full-model pilot, and exact fresh-process
checkpoint resumption. New checks cover actual 14-head, 2,048-token attention
forward/backward against the blocked reference, including learned gains and every
coefficient order. Acceptance failure prevents production.

The nine comparison recipes are frozen with the selected LR. Independent training
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

Query–key degrees 8 and 12 are `r833fbb6eedcb` and `rfd48f7bbe9f4`. Each requests
one B300, eight CPUs, 96 GB host RAM, and a seven-day allocation. These are independent
jobs. The launcher qualifies runtime against the allocated Slurm time limit before
training; the model, coefficient definitions, seed, global batch, and canonical
0.001/0.0001 LR schedule are unchanged. Their run records contain the exact commands.
The B300 32-by-32 backward selector is qualified for both frozen recipes; see the
[scaffold qualification](../../scaffolds/transformer/validation/README.md#b300-query-key-production-qualification)
for independent gradient checks, full-update timings, and checkpoint/resume evidence.
The execution source is pinned by `validation/query-key-b300-4b-20261023`; each
allocation repeats production acceptance before training.

After committing the inputs, submit these two selected runs from this worktree:

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python - <<'PYTHON'
from experiments.transformer_200m_4b.pipeline import STUDY, _git, _submit, _submit_report
runs = [STUDY / "runs" / rid for rid in ("r833fbb6eedcb", "rfd48f7bbe9f4")]
jobs = [_submit(run, _git("rev-parse", "HEAD")) for run in runs]
_submit_report(runs, jobs)
PYTHON
```
