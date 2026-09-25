# Repository guidance

Whenever the user mentions the "write-up," always use `overleaf/iclr/iclr2027_conference.tex` as the reference. This is the main research write-up: notes using the ICLR LaTeX format, not an ICLR submission. Clone the Overleaf project directly into `overleaf/`; `iclr/` is an existing directory inside that project, not an extra clone destination.

Within the Overleaf project, `notes/` contains standalone literature-review and tensor-product notes, `notation.tex` provides shared mathematical notation, `notes.cls` provides the standalone notes format, and `media/` contains figures. `archive/main.tex` is an older draft; use the main write-up above for current definitions. Preserve this upstream layout.

Keep the local Overleaf clone only under `overleaf/` in the primary repository checkout, excluded from Git by the root `.gitignore`. Do not copy or clone it into experiment worktrees. Create worktrees with `git worktree add`; any filesystem-based checkout copy must explicitly exclude `overleaf/`. When working in an experiment worktree, read the write-up from the primary checkout's path.

Treat the Overleaf project as read-only unless the user explicitly instructs you to edit it. Reading or referencing the write-up, running experiments, and updating repository documentation do not authorize changes to any Overleaf files. Do not edit, generate, rename, move, or delete files inside `overleaf/`, or commit or push changes to its separate repository, without that explicit instruction.

Consult [CLUSTERS.md](CLUSTERS.md) for cluster access, storage, scheduler limits, and operational commands before launching or migrating experiments.

## Repository layout

```text
scaffolds/<capability>/              reusable code and its tests
  validation/                        the scaffold's own qualification record
experiments/<experiment>/
  README.md                          the experiment's complete documentation
  configs/                           exact scientific recipes
  runs/<run>/                        run metrics and metadata (authoritative results)
  provenance/manifest.json           authoritative execution record
  summary.csv, figures/              optional results derived from runs/
  adapter/                           experiment code: execution and analysis modules, tests
  logs/, checkpoints/                artifacts, ignored by Git
```

- `scaffolds/<capability>/` holds reusable code only: model, attention, data, trainer,
  evaluation, checkpointing, shared launch and acceptance helpers, reusable sweep and
  analysis helpers, and the scaffold's own tests. Nothing in a scaffold is named after
  or specific to one experiment. `scaffolds/transformer/` owns the language-model
  stack; `scaffolds/state_rank/` owns state-spectrum analysis.
- A scaffold may keep one validation record about itself at
  `scaffolds/<capability>/validation/`: correctness checks, kernel and throughput
  benchmarks, and hardware or runtime qualification. It has the shape of an experiment
  record (README, `configs/`, `runs/<run>/`, `provenance/manifest.json`, ignored
  `logs/`) and follows the same provenance and artifact rules. Its code (benchmarks,
  qualification harnesses) is ordinary scaffold code.
- `experiments/<experiment>/` holds everything for one experiment or sweep: its
  committed record (README, recipes, runs, provenance, and optional derived summary and
  figures), its committed `adapter/` package, and its ignored `logs/` and
  `checkpoints/`. An experiment record contains no other directories.
- Imports go one way: experiment code imports scaffold code, never the reverse, and
  never another experiment's code. Anything two experiments need belongs in a scaffold.
- The adapter separates **execution modules** (launchers, acceptance configuration,
  anything a running job executes) from **analysis modules** (plotting, summaries, W&B
  view definitions) and keeps its own tests. Add an adapter module only when
  configuration of the shared scaffold cannot express the experiment; acceptance
  logic itself lives in the scaffold's shared acceptance module, and the adapter only
  configures it.
- The root `README.md` is the experiment index. `CLUSTERS.md` documents cluster
  operations. Scaffold READMEs document reusable software APIs only.

## Records and provenance

- Each experiment has a **single canonical `README.md`** containing its question,
  method, home cluster, artifact locations, exact reproduction commands, findings, and
  limitations. Do not create competing READMEs, per-run reports, or nested narrative
  documents. Analysis code may rewrite a marked generated section but must preserve
  authored text.
- Recipes under `configs/` state current paths. The bytes that actually executed are
  preserved at the source commit recorded in the manifest
  (`git show <commit>:<path>`); do not keep path-relocation maps, old-location entries,
  or historical-path narratives.
- Scientific results (training and validation metrics, terminal and best evaluations)
  live only in `runs/<run>/`. `summary.csv`, README tables, and figures are derived from
  those files by the adapter's analysis code and must regenerate byte-identically.
- `provenance/manifest.json` is the single authoritative record of how the experiment
  and each run were executed: source commit, resolved configuration and hash, seed, home
  cluster, artifact root, data manifest hashes, the hash of the frozen execution set,
  hardware (GPU type and count, node), partition, Slurm job IDs and attempts (including
  preemptions and requeues), runtime settings (e.g. host staging, `NCCL_P2P_DISABLE`),
  resolved attention backend, a summary of pre-launch acceptance (tests passed,
  initialization hash, pilot and resume outcomes), W&B run id, selected checkpoint, and
  log path, each artifact with its SHA256. The trainer's `runs/<run>/metadata.json` is
  raw output that the manifest summarizes; do not copy per-attempt Slurm or preflight
  files into the record.
- Commit results, not data. Numerical archives, raw arrays, and datasets (for example
  `metrics.npz` or `spectra.npz`) stay beside their committed run metadata, ignored by
  Git; the manifest records their paths and SHA256. Keep datasets outside the
  repository.
- Hardware qualification and kernel or throughput benchmarks describe the shared
  scaffold and runtime, not an experiment's question; record them in the scaffold's
  validation record. An experiment whose acceptance qualifies a new mechanism or
  hardware type summarizes that acceptance in its own manifest and adds the result to
  the validation record.
- A concluded numerical failure is a scientific result: record its stopping step, last
  valid metrics, failure diagnostics, and one matching failure log. Do not label it as a
  completed training run. Retain a diagnostic checkpoint only when justified in
  provenance and the README.

## Artifacts

- Checkpoints live at `experiments/<experiment>/checkpoints/<run>/<step>/` and logs at
  `experiments/<experiment>/logs/<run>/slurm-<job_id>.out`, on the experiment's home
  cluster, ignored by Git.
- During training keep the latest resumable checkpoint and the validation best. After
  completion retain only the validation-best checkpoint per run, which need not be the
  final one, and preserve terminal metrics independently. Experiments without
  pretraining retain no checkpoints unless indispensable for reproduction.
- Retain one successful Slurm log per run, benchmark, or validation, matching its
  metadata; remove superseded and failed-attempt logs once debugging is resolved.
  Launchers open logs in append mode so a requeued job keeps its earlier attempts.
- The root `.gitignore` excludes `experiments/*/logs/`, `experiments/*/checkpoints/`,
  `scaffolds/*/validation/logs/`,
  numerical archives, environments, caches, and runtime checkpoint symlinks using shared
  patterns. Do not use `.git/info/exclude` or machine-local exclusions.

## Scaffold and runtime separation

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
  terminal metrics in the run summary. W&B's local files go under the experiment's
  `logs/`.
- W&B is a live monitoring mirror. `runs/<run>/metrics.jsonl`, the committed figures,
  and `summary.csv` remain the canonical results. An experiment's saved W&B view is
  analysis code in its adapter. Credentials come from the user's `wandb login`
  (`~/.netrc`); never request or print an API key.

## Experiment home cluster

- For a new experiment, the cluster from which the user requests it is its home
  cluster, unless the user explicitly specifies another cluster. Determine the current
  cluster from the session's execution environment. Do not inherit the home cluster
  from related experiments or comparison baselines, and do not ask the user to choose a
  cluster when the request's location already establishes it.
- Before continuing an existing experiment, check its recorded home cluster in
  provenance and its README. If the request comes from a different cluster, tell the
  user which cluster they are on and which cluster owns the experiment, and flag the
  mismatch before launching or resuming. Migration requires an explicit user
  instruction.
- Keep each experiment's runs, resumptions, checkpoints, and logs on its home cluster.
  Record the home cluster and artifact root in provenance and the README. A clone shares
  committed code, recipes, and results, not ignored artifacts; report unavailable or
  unverified artifacts accurately.
- When migrating an experiment, transfer and verify its checkpoints and supporting
  artifacts, provide the matching data and environment, update its recorded locations,
  and never run the same run concurrently on two clusters.

## Experiment lifecycle

- Develop and run each experiment entirely in its own worktree and branch. Its recipes,
  runs, logs, and checkpoints live in the worktree's `experiments/<experiment>/` while
  it runs, so changes to `main` never affect it. A sweep shares one worktree.
- While any of its jobs can run or requeue, the **frozen execution set** must not change:
  scaffold code (excluding the scaffold's validation record), the dependency lock (and
  so the environment), the experiment's `configs/`, the adapter's execution modules, and
  the training data (checked through dataset-manifest checksums). The README, figures, `summary.csv`, and the adapter's
  analysis modules may be edited and committed at any time.
- Launchers record the launch commit and, at every start and requeue, compare only the
  frozen execution set with it (ignoring `runs/`, `logs/`, and `checkpoints/`), refusing
  to run on any difference. The manifest records the frozen set's hash.
- Never remove a worktree while any of its jobs can still run or requeue.
- At completion:
  1. Move (not copy) the retained artifacts into the primary checkout's
     `experiments/<experiment>/`: each run's metrics, its validation-best checkpoint,
     and one successful log.
  2. Verify their checksums.
  3. Record the final locations in the manifest.
  4. Merge the branch into `main`, reconciling any scaffold changes `main` made in the
     meantime.
  5. Only then remove the worktree and delete the branch. `git worktree remove`
     silently deletes ignored files, so confirm nothing retained remains inside it.
- Before merging, remove temporary files, intermediate checkpoints, caches, redundant
  outputs, and duplicated documentation.
