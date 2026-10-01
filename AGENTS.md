# Repository guidance

Whenever the user mentions the "write-up," always use `overleaf/iclr/iclr2027_conference.tex` as the reference. This is the main research write-up: notes using the ICLR LaTeX format, not an ICLR submission. Clone the Overleaf project directly into `overleaf/`; `iclr/` is an existing directory inside that project, not an extra clone destination.

Within the Overleaf project, `notes/` contains standalone literature-review and tensor-product notes, `notation.tex` provides shared mathematical notation, `notes.cls` provides the standalone notes format, and `media/` contains figures. `archive/main.tex` is an older draft; use the main write-up above for current definitions. Preserve this upstream layout.

Keep the local Overleaf clone only under `overleaf/` in the primary repository checkout, excluded from Git by the root `.gitignore`. Do not copy or clone it into experiment worktrees. Create worktrees with `git worktree add`; any filesystem-based checkout copy must explicitly exclude `overleaf/`. When working in an experiment worktree, read the write-up from the primary checkout's path.

Treat the Overleaf project as read-only unless the user explicitly instructs you to edit it. Reading or referencing the write-up, running experiments, and updating repository documentation do not authorize changes to any Overleaf files. Do not edit, generate, rename, move, or delete files inside `overleaf/`, or commit or push changes to its separate repository, without that explicit instruction.

Consult [CLUSTERS.md](CLUSTERS.md) for cluster access, storage, scheduler limits, and operational commands before launching or migrating experiments.

## Repository layout

```text
results/<analysis>/
  README.md                 findings, figure interpretation, regeneration command
  selection.yaml            explicit run IDs, metrics, and comparison settings
  *.pdf, *.png, summary.csv  derived figures and tables, committed
  plot.py, ...               analysis code; tests/ when needed
results/exploratory/
  README.md                 open exploratory topics and the date each opened
  <topic>/                  README.md, optional selection.yaml and figures
experiments/<study>/
  README.md                 scientific question, methods, launch instructions
  sweep.yaml                common recipe and study provenance; run membership
  runs.csv                  generated index of run settings, status, and outcomes
  runs/<run_id>/
    config.yaml             complete input recipe, frozen before launch
    record.json             authoritative execution and artifact provenance
    metrics.jsonl           canonical measurements (or mechanism-specific data)
    checkpoints/            ignored, on the run's home cluster
    training.log            ignored, retained execution log
  acceptance.py, ...        optional experiment-specific execution code
scaffolds/<capability>/      shared implementation, tests, and environment
```

- All shared-Transformer attention experiments belong to
  `experiments/attention_scaling/`. Attention family, normalization, QK normalization,
  degree, width, coefficient parameterization, sharing, and seed distinguish runs
  within this study; do not create a new study directory for a sweep or mechanism.
  `experiments/state_rank/` is a separate scientific question. Comparisons and their
  explicit primary/control run selections belong under `results/`.
- A study groups a scientific question. Varying several axes (for example key width,
  polynomial degree, normalization, coefficient parameterization, sharing, and seed)
  produces runs within a study. Each run ID identifies one execution of a complete
  configuration. Fresh executions receive new IDs; checkpoint resumptions and Slurm
  requeues retain their ID and append execution attempts. IDs are permanent and unique
  within the study; use collision-checked short IDs rather than encoding settings in paths.
- The run index exposes attention family, `d_K`, degree, normalization, QK
  normalization, fixed/learned coefficients, restrictions, sharing, seed, training
  recipe, status, and outcome where applicable. Missing historical facts stay unknown.
  Settings live in structured configurations and records, not directory names.
- Attention divided by its causal row sum is "normalized" and attention without that
  division is "unnormalized" in figures, legends, READMEs, and write-ups; never write
  "row-normalized" or "row normalization".
- `results/<analysis>/` groups figures around a comparison. Several analyses may
  reference the same run, and an analysis may combine several studies. Reference run
  IDs and checked source artifacts; never copy canonical measurements or checkpoints
  into a results directory. Selection must be explicit; do not silently choose the
  newest or best run at a coordinate. Numerical failures and incomplete coordinates
  remain visible. Terminal and validation-best metrics must be distinguished.
- Keep the maintained attention comparisons in `results/attention_scaling/`: the
  combined heatmap and degree plot, their explicit selections, and regeneration code.
  Do not create a result directory for each mechanism or sweep. Learning curves and
  coefficient diagnostics remain available in run measurements; add a maintained
  diagnostic figure only when it serves a requested comparison. State-rank results
  belong to `results/state_rank/`.
- Exploratory experiments, whose place in the canonical results is not yet decided,
  report under `results/exploratory/<topic>/`, one directory per question with flat
  files: a README with the question, runs, findings, the open decision, and the date
  opened; an optional selection YAML of explicit run IDs with roles, pinned by
  `scaffolds.records refresh`; and optional figures or tables that regenerate from
  it. Tests are not required. Their runs live in their study like any other run, with
  the usual lifecycle, records, and retained artifacts, and enter no canonical
  selection. `results/exploratory/README.md` lists the open topics; mention them in
  status updates. A topic ends in one of two ways: promote it (move its selection,
  findings, and code into the canonical collection, add tests, and delete the topic),
  or discard it (delete the topic and its runs with their artifacts). Exploratory
  topics are exempt from the one-collection rule and from tests, not from the
  provenance rules.
- Keep figures and analysis files directly in the result directory. Keep each run's
  recipe, provenance, raw output, retained checkpoint, and log together. Do not create
  parallel experiment-level `configs/`, `provenance/`, `logs/`, `checkpoints/`, or
  `figures/` trees. Create optional files/directories only when needed.
- Experiment execution modules live directly in their study; custom analysis modules
  live directly in their result collection. Add code only when the shared scaffold
  cannot express the experiment or analysis. Tests may use a `tests/` directory.
- `scaffolds/<capability>/` holds reusable code only: model, attention, data, optimizer,
  training, evaluation, checkpointing, launch and acceptance helpers, and shared
  analysis. `scaffolds/records/` owns reusable run identity, provenance, and indexing.
  Production imports go from experiments/results to scaffolds, never the reverse and
  never from one experiment's execution code into another's.
- A scaffold's correctness checks, hardware qualification, and throughput benchmarks
  belong in its own `validation/` record. They describe software/runtime behavior and
  are distinct from the scientific studies and comparisons under experiments/results.
- The root `README.md` is the results and study index. `CLUSTERS.md` documents cluster
  operations. Scaffold READMEs document reusable software APIs only.

## Records and provenance

- Each study has one README for its question, method, home cluster, and exact launch
  commands. Each result collection has one README for findings, limitations, and
  regeneration. Link between them rather than duplicating narrative. Generated
  sections may be rewritten, preserving authored text.
- `runs/<run_id>/config.yaml` is the immutable input recipe. `sweep.yaml` describes
  study defaults, membership, and common provenance; changing future defaults must
  not change any launched run's recipe. Actual runtime resolution (including backend,
  source commit, and absolute paths) is recorded separately in the run's execution
  record. Preserve the executed source and dependency lock by commit.
- `runs/<run_id>/record.json` is the authoritative execution record: configuration
  identity/hash, seed, source commit, frozen execution hash, home cluster, artifact
  locations and hashes, dataset hashes, GPU type/count/node, partition, Slurm attempts,
  runtime settings, resolved backend, acceptance outcomes, W&B ID, and selected
  checkpoint. Shared study facts may be referenced from `sweep.yaml`; per-run facts
  belong to the run. Raw trainer outputs such as `metadata.json`, `config.json`, and
  `completion.json` may be retained as evidence and are referenced by the record.
- Measurements live only under the run. `runs.csv`, result summaries, tables, and
  figures are derived and must regenerate byte-identically. Result selections name
  the exact input runs and analysis settings and pin each run's scientific inputs
  (recipe, metrics, completion, status metadata, validation-best and numerical-failure
  records), never its provenance. A result update changes its selection, never
  overwrites an earlier run: point the role at the new run ID, run
  `python -m scaffolds.records refresh results/<analysis>`, and regenerate.
- Commit configurations, provenance, and compact metrics, not datasets or large
  arrays. Numerical archives (`metrics.npz`, `spectra.npz`) stay beside run metadata,
  ignored by Git; record their SHA256. Keep datasets outside the repository.
- A concluded numerical failure is a scientific result: preserve its stopping step,
  last valid metrics, failure diagnostics, and matching log. Do not call it completed
  training. Retain a diagnostic checkpoint only when justified by the record.
- Records state current facts only; Git is the history. `sweep.yaml` holds study-wide
  facts (home cluster, artifact root, dataset and lock hashes, shared settings the
  analysis reads) and `run_ids`; everything else about a run lives in its record. Do
  not record migration, fold-in, or relocation notes, superseded attempts or runs,
  pre-migration hashes, copied measurements, per-attempt evidence dumps, or narrative
  about how the repository reached its current state, in records, study files, or
  READMEs. When a run is replaced, delete it and its artifacts rather than annotating
  it. Current paths resolve directly; source commits preserve historical code.
  Existing descriptive run names may be retained as labels.

## Artifacts

- Checkpoints live at `experiments/<study>/runs/<run_id>/checkpoints/<step>/`, and
  the retained Slurm output at `experiments/<study>/runs/<run_id>/training.log`, on
  the run's home cluster, ignored by Git. Record job IDs/attempts in `record.json`.
- During training keep the latest resumable checkpoint and validation best. After
  completion retain only the validation-best checkpoint, which need not be terminal;
  preserve terminal metrics independently. Analyses without training retain no
  checkpoints unless indispensable for reproduction.
- Retain one successful log, or a matching scientific-failure log, for each run.
  Launchers append so requeues preserve earlier attempts. Remove superseded debugging
  logs when resolved. Temporary acceptance output and W&B files stay ignored within
  the run, and are summarized in its record.
- Root `.gitignore` uses shared patterns for run checkpoints, logs, temporary runtime
  output, numerical archives, environments, caches, and scaffold validation artifacts.
  Do not use `.git/info/exclude` or machine-local exclusions.

## Scaffold and runtime separation

- New dot-product Transformer experiments must enable QK normalization unless the
  user explicitly requests plain QK scores. Set `model.qk_norm: true` in the frozen
  recipe; do not inherit `false` when copying a legacy run. GLA and degree-zero
  attention do not use QK normalization. Preserve historical recipes and resume
  settings unchanged; a QK-normalized rerun receives a fresh run ID. Fresh CLI
  allocation of an explicitly requested plain-score run requires `--allow-plain-qk`.
- Experiments sharing a model family use one shared model, attention, data pipeline,
  optimizer, trainer, evaluation, and checkpoint implementation. Keep cluster names,
  GPU-model workarounds, and transport policy out of model definitions, optimizer
  logic, and scientific recipes.
- Generic device placement, replication, sharding, and gradient synchronization belong
  in the shared training runtime (`training/distributed.py`). Host staging is a generic
  transfer option there; the trainer calls these helpers rather than implementing
  hardware branches itself.
- Launchers (Slurm scripts or Python submission scripts) select cluster resources and
  execution settings such as host staging, `NCCL_P2P_DISABLE`, or `CUDA_HOME`, then
  invoke the shared trainer. Document cluster policy in `CLUSTERS.md` and the runtime
  interface in the scaffold README.
- Keep device-capability checks and implementation selection in attention backends and
  the runtime. Hardware-specific kernels are allowed behind the shared attention
  interface; they must preserve the requested mathematics and explicit numerical-failure
  behavior. Qualify every new mechanism, kernel, or hardware/runtime combination before
  production.
- Record selected runtime settings, resolved backends, and hardware in provenance,
  separately from scientific hyperparameters.

## Environments

- Each worktree syncs its own environment from its own lock with
  `uv sync --locked --link-mode hardlink` (for example `uv sync --project
  scaffolds/transformer --locked --extra cuda`). Hardlinks share package files with the
  uv cache, so an additional environment costs almost no disk. Never copy an
  environment directory, and never modify installed packages in place: hardlinked files
  are shared by every environment.
- Launchers use the worktree's own environment, so dependency changes on `main` cannot
  reach a running experiment.

## Weights & Biases

- Training runs mirror their metrics to the private W&B project
  `rafah-massachusetts-institute-of-technology/functional-memory`, with the experiment
  as the W&B group, the run name as the W&B name, mechanism and hardware tags, the full
  recipe as config, metric prefixes `train/`, `val/`, and `attention/<mechanism>/`, and
  terminal metrics in the run summary. W&B's local files stay ignored within the run directory.
- W&B is a live monitoring mirror. `runs/<run>/metrics.jsonl`, the committed figures,
  and derived summaries remain the recorded results. A saved W&B view is
  analysis code in the corresponding results collection. Credentials come from the user's `wandb login`
  (`~/.netrc`); never request or print an API key.

## Experiment home cluster

- For a new experiment, the cluster from which the user requests it is its home
  cluster, unless the user explicitly specifies another cluster. Determine the current
  cluster from the session's execution environment. Do not inherit the home cluster
  from related experiments or comparison baselines, and do not ask the user to choose a
  cluster when the request's location already establishes it.
- Before continuing an existing experiment, check its recorded home cluster in
  its run/study records and README. If the request comes from a different cluster, tell the
  user which cluster they are on and which cluster owns the experiment, and flag the
  mismatch before launching or resuming. Migration requires an explicit user
  instruction.
- Each run has one home cluster, recorded in its `record.json` and `runs.csv`; one study
  may hold runs from several clusters. A run's checkpoints, logs, and resumptions stay
  on its home cluster. At completion they move out of the worktree into that cluster's
  artifact root (listed in `CLUSTERS.md`), and the record states their absolute paths
  there. Never copy them to another cluster except in an explicit migration.
- A checkout on another cluster has only the committed files, which is expected:
  `scaffolds.records validate` reports those artifacts as unavailable, not as errors.
  Figures, indices, and tests use only committed files and must pass on every cluster;
  code and tests must not require a cluster-specific path (dataset, artifact) elsewhere.
- Close out on the home cluster, push, and pull on the other clusters.
- When migrating an experiment, transfer and verify its checkpoints and supporting
  artifacts, provide the matching data and environment, update its recorded locations,
  and never run the same run concurrently on two clusters.

## Experiment lifecycle

- Develop and run each experiment entirely in its own worktree and branch. Its recipes,
  runs, logs, and checkpoints live in the worktree's `experiments/<experiment>/` while
  it runs, so changes to `main` never affect it. A sweep shares one worktree.
- While any of its jobs can run or requeue, the **frozen execution set** must not change:
  scaffold code (excluding its validation record), the dependency lock (and therefore
  the environment), `sweep.yaml`, `runs/*/config.yaml`, experiment execution modules,
  and training data (checked through dataset-manifest checksums). Run records, metrics,
  logs, and checkpoints are mutable outputs. READMEs, generated indices, and results
  analysis/figures may be edited while jobs run.
- Launchers record the launch commit and, at every start and requeue, compare only the
  frozen execution set with it, refusing to run on any difference. Run configurations
  are inputs even though they live under `runs/`; mutable run outputs are excluded.
  Each run record contains the frozen set's hash.
- Never remove a worktree while any of its jobs can still run or requeue.
- At completion:
  1. Move (not copy) the retained artifacts into the home cluster's artifact root:
     each run's metrics, its validation-best checkpoint, and one successful log.
  2. Verify their checksums.
  3. Record the final locations in the run records and regenerate the study index.
  4. Merge the branch into `main`, reconciling any scaffold changes `main` made in the
     meantime.
  5. Only then remove the worktree and delete the branch. `git worktree remove`
     silently deletes ignored files, so confirm nothing retained remains inside it.
- Before merging, remove temporary files, intermediate checkpoints, caches, redundant
  outputs, and duplicated documentation. An older active worktree keeps its frozen
  execution layout until its jobs finish; migrate retained artifacts into this layout
  when integrating it, without editing live execution paths.
