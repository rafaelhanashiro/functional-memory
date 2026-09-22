# Query/key dimension and perplexity

Eight completed decoder-only Transformer runs vary per-head query/key width
**2, 4, 8, 16, 32, 64, 128, 256**, keeping the backbone and value width fixed.
The primary endpoint is terminal held-out perplexity at the same training-token
budget. All runs use seed 0 and the same data order, so this collection does
not estimate variation across independent seeds. Width 256 extends the
original seven-run sweep with the same recipe and data.

<!-- BEGIN GENERATED RESULTS -->

8/8 runs complete at 10,000,269,312 prediction tokens per run.

| dK | Parameters | Terminal PPL |
| --- | ---: | ---: |
| 2 | 179,590,144 | 22.3489 |
| 4 | 180,245,504 | 18.8928 |
| 8 | 181,556,224 | 16.7053 |
| 16 | 184,177,664 | 15.5764 |
| 32 | 189,420,544 | 14.9853 |
| 64 | 199,906,304 | 14.6898 |
| 128 | 220,877,824 | 14.3776 |
| 256 | 262,820,864 | 14.1736 |

[Machine-readable results](summary.csv) · [Terminal PPL](figures/ppl_vs_dk.pdf) · [Learning curves](figures/learningcurves.pdf)

<!-- END GENERATED RESULTS -->

## Model and training recipe

The backbone has 10 layers, residual width 1024, 16 full-MHA heads, value width
64, SwiGLU intermediate width 3008, and untied 32,000-token embeddings. It uses
RMSNorm (epsilon `1e-6`), RoPE (base 10,000), no dropout, and OFM's Flax NNX
initialization. BF16 computation uses FP32 parameters and optimizer state.

This is a **200M-reference family**, not a parameter-matched comparison. The
formula is `178934784 + 327680 * qk_head_dim`. Value/output projection sizes,
depth, MLP width, and head count remain fixed. Standard RoPE frequencies are
defined at each model's true query/key width. Q/K/V are zero-padded only for
the fused attention call, after RoPE; attention scaling uses the true width
and the output is sliced back to 64 value channels. Increasing query/key
width also increases parameter count, so the perplexity differences do not
isolate width independently of model size.

Each run trains for 19,074 optimizer updates with a global batch of 256
sequences and 2048 prediction positions, or **10,000,269,312 prediction
tokens**. A global microbatch of 64 supplies four accumulation steps. AdamW
uses peak/minimum learning rates `3e-4`/`3e-5`, a 2,000-step linear warmup and
cosine decay, betas `(0.9, 0.95)`, weight decay `0.1` on all parameters, and
global gradient clipping at `1.0` after accumulation/device averaging.

Evaluation uses the same first 1,792 held-out packed sequences every 1,000
updates and at completion: **3,670,016 prediction tokens**. Perplexity is the
exponential of mean token negative log likelihood, not an average of sequence
perplexities. Terminal results are compared at the common token budget;
validation-best values are recorded separately. Checkpoints are saved every
500 steps and at validation. During training, the latest checkpoint supports
exact resume and the validation-best checkpoint is preserved. Retained runs
keep only the validation-best checkpoint. For all eight runs, this is also
the terminal checkpoint at step **19074**. Terminal metrics remain in the
results independently of checkpoint selection.

## Source and data provenance

The historical implementation is adapted from OFM commit
`8977f529e83efc688daf96cdcfd365ebbce2afa8`, especially its canonical
`configs/experiments/transformer_250m.yaml`. The current implementation is consolidated under `scaffolds/transformer/`;
replay the recorded source revision when reproducing the original runs.

The source corpus is read-only at
`/home/rafah/projects/ofm/datasets/fineweb_12b/`. Its train/eval manifests
record source revisions, tokenizer revisions, shard checksums, and the
document-level train/eval partition. The existing 12B prediction-token corpus
is sufficient for this sweep. Training consumes packed 2049-token records as
2048 inputs and shifted next-token targets. Dataset identity and the exact
data cursor are checked when resuming.

The architecture is also informed by [Hyperloop Transformers](https://arxiv.org/pdf/2604.21254).
This experiment follows OFM's recipe and held-out subset; its perplexities
are not direct reproductions of that paper's reported results.


## Artifact layout and cluster ownership

This README is the experiment's complete narrative documentation. The
[canonical provenance](provenance/manifest.json) records the exact source,
seed, resolved configuration and checksum, terminal metrics, validation-best
checkpoint, data identity, and successful Slurm log for every run.

| Artifact | Location relative to the repository |
| --- | --- |
| Exact historical recipes and dependency lock | `results/key_dimension/configs/` |
| Per-run metrics and completion evidence | `results/key_dimension/runs/<run>/` |
| Figures | `results/key_dimension/figures/` |
| Derived summary CSV | `results/key_dimension/summary.csv` |
| Canonical provenance | `results/key_dimension/provenance/manifest.json` |
| Current shared Transformer implementation | `scaffolds/transformer/` |
| Study-specific sweep, preflight and plot tools | `scaffolds/transformer/studies/key_dimension/` |
| Canonical checkpoint layout | `checkpoints/key_dimension/<run>/<step>/` |
| Successful retained training log | `results/logs/key_dimension/<run>/slurm-<job_id>.out` |

The home cluster is **ORCD**; its recorded artifact root is
`/home/rafah/projects/functional_memory`. Relative checkpoint and log locations are
resolved against that artifact root, not whichever clone is reading this
report. Git shares code, recipes and scientific results; it does not transfer
ignored checkpoints, datasets, environments or Slurm logs.

The ORCD checkpoint and log inventory was recorded when the runs were retained;
it has not been reverified from Kennel. In particular, the historical checkpoints
were retained under `experiments/key_dimension/checkpoints/<run>/19074/` on
ORCD. Provenance keeps that recorded `selected_checkpoint.path` and records
`checkpoints/key_dimension/<run>/19074/` separately as the
`canonical_destination`, with migration pending. No remote filesystem move is
implied, and those ignored artifacts are absent from this Kennel clone.

The YAML files are preserved **byte for byte**. Their old absolute data and
work-directory paths describe historical execution; moving a recipe does not
make those paths portable. `configs/uv.lock` is the historical lock matching
the retained dependency hash, while `scaffolds/transformer/uv.lock` governs
new runs of the current shared scaffold. Source-hash keys inside historical
benchmark and launch records likewise refer to files at their recorded commits.

During training, keep the latest resumable checkpoint and the validation-best
checkpoint. Only after completion and review apply the validation-best-only
retention policy. Terminal metrics remain the scientific endpoint whether or
not the selected checkpoint is terminal. Keep one successful Slurm log per
retained run, benchmark and validation; remove failed or superseded logs once
debugging is resolved. Consult [AGENTS.md](../../AGENTS.md) and
[CLUSTERS.md](../../CLUSTERS.md) before a launch or migration.

## Environment, validation and figure regeneration

Run these commands from the repository root. The shared JAX environment is
separate from the root PyTorch environment because their CUDA-library pins
differ. Historical replay instead installs the lock at the recorded source.

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.envs/shared_transformer"
export UV_CACHE_DIR="$PWD/.cache/uv"
uv sync --project scaffolds/transformer --locked --extra cuda
JAX_PLATFORMS=cpu uv run --project scaffolds/transformer --locked --extra cuda python -m pytest \
  -c scaffolds/transformer/pyproject.toml scaffolds/transformer/studies/key_dimension/tests -m 'not gpu'
uv run --project scaffolds/transformer --locked --extra cuda python -m \
  scaffolds.transformer.studies.key_dimension.plot --manifest results/key_dimension/provenance/manifest.json
```

Plots use committed terminal validation at the common token budget and keep
best-checkpoint metrics separate. They read canonical repository-relative
artifact paths, not the vanished original execution checkouts or config
workdirs. Recipe, metric and historical-lock checksums are verified. No dataset,
GPU, or physical checkpoint is required to regenerate retained figures.

CPU tests cover recipe/initialization equality, independent attention output
and gradient references, packed data, metrics transactions, and checkpoint
resume. GPU acceptance must additionally check the intended attention backend,
full context and microbatch, training/evaluation pilots, and checkpoint resume
on the production allocation. CPU success alone does not establish GPU support.

## Reproducing and running

For an **exact historical replay**, create a dedicated branch and separate
checkout at the run's `source_commit`, install that checkout's original locked
environment, and use its historical training entry point
`experiments.key_dimension.train`. Copy `configs/200m-dk256-seed0.yaml` outside the frozen checkout and
provide the content-identical packed corpus. Run inside an allocation with
two H200 GPUs; the training CLI runs in the allocation and does not submit Slurm.
Override the historical `train.workdir` with a fresh result directory, and
keep checkpoint storage outside the temporary execution checkout. Old source
revisions predate `checkpoint_dir`; for those revisions, create a
`<fresh-workdir>/checkpoints` symlink to a fresh durable checkpoint directory
before starting. Do not rewrite the retained recipe or reuse its old outputs.

For a **new run of the shared scaffold**, use a fresh dedicated branch and
working checkout, copy the desired recipe into the new result directory,
record the new source commit and home cluster, and set these fields in the
copied configuration:

```yaml
train:
  source_commit: auto
  workdir: results/key_dimension_reproduction/runs/200m-dk256-seed0
  checkpoint_dir: checkpoints/key_dimension_reproduction/200m-dk256-seed0
```

Keep the complete original model, optimizer and data sections; this excerpt
only shows path/source overrides. Verify the dataset checksums, choose an
allocation on the recorded home cluster, and freeze the tested checkout before
launch. From that checkout, inside the allocation:

```bash
uv run --project scaffolds/transformer --locked --extra cuda python -m \
  scaffolds.transformer.train /path/to/fresh-resolved-config.yaml --no-resume
```

To resume, retain the same source, configuration and paths and omit
`--no-resume`. The trainer restores optimizer state and the packed-data cursor.
The family sweep launcher is `scaffolds.transformer.studies.key_dimension.sweep`; use its `--help` and
dry run to inspect the current scheduler profile before submission. New result
paths and run names must be fresh; changing only the manifest filename does
not authorize overwriting retained runs. A new clone supplies no ignored
artifacts. Before migrating a run, transfer and verify its checkpoints and
supporting artifacts, make matching data/environment available, update its
home cluster and locations, and avoid concurrent copies of the same run.

After review, retain the scientific results, exact recipes, best checkpoint and
successful logs in the canonical layout, merge to `main`, and remove temporary
checkouts and merged branches after their indispensable artifacts are safe.

## Historical launch and per-run records

The original seven runs used source
`5e871d2659373272e26668e10911922c7201678a`; width 256 used
`3d330a2f29e067b4df017ac08d12b9b6d3154b94`. Their historical launch records
[`sweep.json`](sweep.json) and [`sweep-dk256.json`](sweep-dk256.json) remain
byte-identical. The extension pins the original manifest SHA-256
`8080d58cddad8879687c804427c8746fc1133aec133f6d657d3be8ad20b96566`.
Archived absolute execution and configuration paths are historical; the
[canonical manifest](provenance/manifest.json) locates retained results now.
The original sweep identity was
`0ceca430e0bff3c418fe7a21cd99107059d4d5d229d85e8582621d658795fa1e`;
the extension identity was
`a237d91a0c287ec1a182aa17114a6294983be05d9dfbb2882076a3ba44df9883`.

All eight runs completed 19,074 updates and selected that same step as their
validation best. Per-run JSON retains the exact terminal/best NLL, perplexity,
source, seed, completion, and successful log. The compact table below replaces
the separate per-run narrative reports without removing their measurements.

| Run | Terminal / best NLL | Best step | Successful Slurm job |
| --- | ---: | ---: | --- |
| [200m-dk002-seed0](runs/200m-dk002-seed0/metadata.json) | 3.106774909 | 19074 | `22808234` |
| [200m-dk004-seed0](runs/200m-dk004-seed0/metadata.json) | 2.938780878 | 19074 | `22808236` |
| [200m-dk008-seed0](runs/200m-dk008-seed0/metadata.json) | 2.815727804 | 19074 | `22808237` |
| [200m-dk016-seed0](runs/200m-dk016-seed0/metadata.json) | 2.745754208 | 19074 | `22808238` |
| [200m-dk032-seed0](runs/200m-dk032-seed0/metadata.json) | 2.707067149 | 19074 | `22808239` |
| [200m-dk064-seed0](runs/200m-dk064-seed0/metadata.json) | 2.687151134 | 19074 | `22808243` |
| [200m-dk128-seed0](runs/200m-dk128-seed0/metadata.json) | 2.665670948 | 19074 | `22808244` |
| [200m-dk256-seed0](runs/200m-dk256-seed0/metadata.json) | 2.651382114 | 19074 | `22951341` |
