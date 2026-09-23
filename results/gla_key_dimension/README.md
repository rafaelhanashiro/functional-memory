# Gated linear attention key-width sweep

This planned experiment measures terminal held-out perplexity for a gated linear
attention (GLA) state-space baseline at **dK={8,16,32,64,128}**, matching the five
rows of the [Taylor heatmap](../key_dimension_taylor/README.md). The implementation
and CPU checks are complete; no production runs or GPU qualifications have been
completed. The execution source revision remains pending
until the implementation is tested, committed, and frozen.

## Scientific comparison

Each layer maintains one `dK × 64` matrix per head. For head-local vectors,
`S_t = diag(alpha_t) S_(t-1) + k_t v_t^T` and `o_t = q_t^T S_t`.
The decay is `alpha_t = exp(log_sigmoid(W2 W1 x_t + b) / 16)`, with gate
rank 16. As in OFM, Q/K use RoPE; per-head RMSNorm is applied to the readout,
which is multiplied by a learned SiLU output gate before output projection.
State starts at zero for every packed record; training does not carry it between
records. There is no softmax row normalization or additional `1/sqrt(dK)` scaling.

The GLA module and its fused kernels are adapted into this repository from OFM;
the runtime does not import the external OFM checkout. Implementation ancestry
and retained upstream notices are recorded with the shared GLA backend. The
reference recurrence and fused implementation use the same mathematics.
Chunk size 64 and inner chunk size 16 preserve the upstream implementation.
The local changes pad narrow kernel tiles to at least 16 and update the bridge's
manual-sharding detection for JAX 0.10. Original source commits, hashes, and
adaptations are retained in the shared backend's `gla_sources.json`.
OFM's mixed-precision details are also retained: RoPE promotes Q/K to FP32,
and the fused read enters head normalization in FP32; the blocked reference
casts its read to the configured computation dtype first. Production therefore
pins the fused path and does not assume bitwise BF16 parity between backends.

All runs retain the heatmap backbone: 10 layers, hidden width 1024, 16 heads,
value width 64, SwiGLU intermediate width 3008, vocabulary 32,000, untied
embeddings, RMSNorm epsilon `1e-6`, RoPE base 10,000, no dropout, and OFM/Flax
initialization. BF16 computation uses FP32 trainable parameters and optimizer
state. The shared model, data pipeline, optimizer, trainer, evaluation, and
checkpoint code live under `scaffolds/transformer/`.

This is a **200M-reference backbone comparison**, not a parameter-matched one.
Increasing dK increases both the recurrent state and trainable projections. GLA
also adds decay gates, an output gate, and head normalization compared with
softmax/Taylor attention; the comparison cannot isolate state size alone.

| dK | Trainable parameters | Recurrent state elements per layer |
| ---: | ---: | ---: |
| 8 | 192,237,824 | 8,192 |
| 16 | 194,881,024 | 16,384 |
| 32 | 200,167,424 | 32,768 |
| 64 | 210,740,224 | 65,536 |
| 128 | 231,885,824 | 131,072 |

## Data, optimization, and endpoint

Recipes in [configs/](configs/) preserve the heatmap's FineWeb-Edu `sample-100BT`
packed corpus and Llama-2 tokenizer. Each 2,049-token record supplies 2,048
next-token predictions. Seed and shuffle seed are both zero. The exact train/eval
manifest hashes are in [provenance/manifest.json](provenance/manifest.json).
The same seed fixes data order; different module shapes and added gates do not
imply identical parameter draws to the historical Transformer runs.

Each run uses global batch 256, global microbatch 64, four accumulation steps,
and 19,074 optimizer updates: **10,000,269,312 prediction tokens**. AdamW has
peak/minimum learning rates `3e-4`/`3e-5`, 2,000 warmup steps, cosine decay,
betas `(0.9,0.95)`, weight decay `0.1`, and global gradient clipping at `1.0`
after accumulation/device averaging. No learning-rate or gate sweep is included.

Evaluation every 1,000 updates and at completion uses the first seven fixed
held-out global batches: 1,792 records and 3,670,016 prediction tokens. The
primary endpoint is exponential mean token NLL at the common terminal training
budget. Validation-best metrics remain separate. Historical width-matched
softmax endpoints are pinned in the manifest, including dK64 PPL
`14.689767074685026`; the different polynomial-study softmax result is not the
reference. Taylor results are linked by their canonical manifest checksum.
One seed cannot establish statistical significance.

## Home cluster and artifacts

The proposed home cluster is **Kennel**, with durable artifact root
`/data/cl/u/rafah/home/projects/functional-memory`. Dataset manifests are under
`/data/cl/u/rafah/home/datasets/fineweb_12b`. These are planned locations, not
claims of verified current access or existing GLA artifacts. The launcher uses
four H100s on dogo by default; B300s on akita require their own qualification.
Consult [CLUSTERS.md](../../CLUSTERS.md) and live scheduler limits before launch.

| Artifact | Location relative to the artifact root |
| --- | --- |
| Single scientific record | `results/gla_key_dimension/README.md` |
| Exact recipes | `results/gla_key_dimension/configs/<run>.yaml` |
| Authoritative provenance | `results/gla_key_dimension/provenance/manifest.json` |
| Submission evidence | `results/gla_key_dimension/provenance/execution/` |
| Per-run metrics and metadata | `results/gla_key_dimension/runs/<run>/` |
| GPU acceptance evidence | `results/gla_key_dimension/validation/<identity>/<run>/` |
| Checkpoints | `checkpoints/gla_key_dimension/<run>/<step>/` |
| Matching scheduler logs | `results/logs/gla_key_dimension/<run>/slurm-<job>.out` |

The execution worktree and dedicated environment remain frozen while jobs can
run or requeue. Runtime transfer policy is selected by the shared Kennel launcher
helper; the model never chooses cluster policy. Recipes request `auto`; the
experiment's qualification and production adapter explicitly require `fused` and
record this resolution in run metadata. The launcher disables XLA command buffers
for the upstream custom-call bridge and records this setting with qualification.

## Prepare, qualify, and launch

CPU development evidence is retained in [validation/cpu.json](validation/cpu.json)
and [validation/ofm_parity.json](validation/ofm_parity.json). The final focused
suite passed 50 tests with two CPU devices, including the sequential recurrence
oracle, all GLA parameter gradients, BF16/rematerialization, checkpoint resume,
bridge tracing, shared diagnostics, and launch guards. Seven frozen-source tests
also passed for historical softmax and unnormalized attention. The earlier broad
shared CPU suite passed 251 tests (12 skipped). These checks do not qualify a GPU
kernel or production model allocation.

The optional source-extraction audit compares initialized parameters, outputs,
and all parameter gradients exactly with the recorded OFM Git revision in FP32
and BF16, each with and without RoPE. It reads OFM without modifying it:

```bash
JAX_PLATFORMS=cpu .envs/gla/bin/python \
  -m scaffolds.transformer.studies.gla_key_dimension.verify_ofm \
  --ofm-root /path/to/ofm
```

Develop in a dedicated `git worktree add` checkout. Do not copy `overleaf/` into
it. After reviewing and committing the implementation, run from that frozen
checkout on Kennel. The artifact root must not be inside the execution checkout,
and its GLA experiment manifest must not already exist: freezing refuses to
overwrite another experiment record. Keep the environment fixed after installation.

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.envs/gla"
export UV_CACHE_DIR="$PWD/.cache/uv"
uv sync --project scaffolds/transformer --locked --extra cuda
export GLA_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
export GLA_ARTIFACT_ROOT=/data/cl/u/rafah/home/projects/functional-memory
export GLA_MANIFEST="$GLA_ARTIFACT_ROOT/results/gla_key_dimension/provenance/manifest.json"

JAX_PLATFORMS=cpu "$GLA_PYTHON" -m pytest \
  -c scaffolds/transformer/pyproject.toml \
  scaffolds/transformer/tests scaffolds/transformer/studies/gla_key_dimension/tests \
  --import-mode=importlib -m 'not gpu'

"$GLA_PYTHON" -m scaffolds.transformer.studies.gla_key_dimension.sweep freeze \
  --execution-root "$PWD" --artifact-root "$GLA_ARTIFACT_ROOT"

# Inspect scheduler commands, then submit qualification for all five widths.
"$GLA_PYTHON" -m scaffolds.transformer.studies.gla_key_dimension.sweep submit \
  --manifest "$GLA_MANIFEST" --mode preflight --dry-run
"$GLA_PYTHON" -m scaffolds.transformer.studies.gla_key_dimension.sweep submit \
  --manifest "$GLA_MANIFEST" --mode preflight

# Run only after the five qualification jobs have passed.
"$GLA_PYTHON" -m scaffolds.transformer.studies.gla_key_dimension.sweep submit \
  --manifest "$GLA_MANIFEST" --mode train --dry-run
"$GLA_PYTHON" -m scaffolds.transformer.studies.gla_key_dimension.sweep submit \
  --manifest "$GLA_MANIFEST" --mode train
```

Add `--run-name 200m-dk008-gla-seed0` to operate on one coordinate. Add
`--node akita --gpu b300` to choose B300 allocations before qualification.
One frozen manifest supports one qualified runtime per coordinate. Changing
hardware after qualification requires an explicitly prepared new frozen identity;
existing H100 evidence cannot authorize B300 execution or be overwritten.
Source, recipes, data manifests, dependency lock,
hardware, transfer policy, and kernel environment are checked before training.
Actual submission records and run metadata carry job IDs and runtime settings;
none are populated by the planning record.

Each qualification runs GLA GPU output/gradient tests, 32 updates of the full
production model/context/microbatch, one global validation batch, a checkpoint,
and a fresh-process restore to step 33. The LR horizon remains 19,074; only pilot
evaluation/checkpoint cadence and output paths differ. Independent state, data
cursor, and RNG hashes must match across restore. The pilot does not estimate
final perplexity. It requires its own source/runtime acceptance even if the
same hardware previously ran Taylor attention.

Qualification checkpoints are disposable and live under the artifact root's
`.cache/gla_key_dimension/`; remove them after retaining the acceptance evidence.
Qualification jobs do not requeue: failed or preempted attempts preserve their
evidence for inspection and must be resolved before explicitly preparing a retry.
All selected GPU tests must pass without skips. A per-run lock prevents concurrent
production workers from writing the same metrics or checkpoints. Production resumes its latest checkpoint on
Slurm requeue, retaining the original recipe and data cursor. Keep latest and
validation-best checkpoints during training; after completion/review retain only
validation best and the independent terminal metrics. Record numerical failures
as failures with their stopping step, last valid metrics, diagnostics, and one
matching log. Keep one successful log per retained validation and training run.

No findings or figures exist yet. After the sweep, summarize width-matched
terminal NLL/PPL and deltas to softmax/Taylor, retain the canonical evidence, then
review/merge the experiment and retire its temporary worktree and branch once
durable artifacts are safe.
