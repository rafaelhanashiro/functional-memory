# Repository guidance

Whenever the user mentions the "write-up," always use `overleaf/iclr/iclr2027_conference.tex` as the reference. This is the main research write-up: notes using the ICLR LaTeX format, not an ICLR submission. Clone the Overleaf project directly into `overleaf/`; `iclr/` is an existing directory inside that project, not an extra clone destination.

Within the Overleaf project, `notes/` contains standalone literature-review and tensor-product notes, `notation.tex` provides shared mathematical notation, `notes.cls` provides the standalone notes format, and `media/` contains figures. `archive/main.tex` is an older draft; use the main write-up above for current definitions. Preserve this upstream layout.

Keep the local Overleaf clone only under `overleaf/` in the primary repository checkout, excluded from Git by the root `.gitignore`. Do not copy or clone it into experiment worktrees. Create worktrees with `git worktree add`; any filesystem-based checkout copy must explicitly exclude `overleaf/`. When working in an experiment worktree, read the write-up from the primary checkout's path.

Treat the Overleaf project as read-only unless the user explicitly instructs you to edit it. Reading or referencing the write-up, running experiments, and updating repository documentation do not authorize changes to any Overleaf files. Do not edit, generate, rename, move, or delete files inside `overleaf/`, or commit or push changes to its separate repository, without that explicit instruction.

Consult [CLUSTERS.md](CLUSTERS.md) for cluster access, storage, scheduler limits, and operational commands before launching or migrating experiments.

## Canonical scaffold and runtime separation

- Experiments sharing a model family must use one shared model, attention, data pipeline, optimizer, trainer, evaluation, and checkpoint implementation. Keep cluster names, GPU-model workarounds, and transport policy out of model definitions, optimizer logic, and scientific recipes.
- Generic device placement, replication, sharding, and gradient synchronization belong in the shared training runtime, such as `training/distributed.py`. Host staging is a generic transfer option implemented there; it must not require a separate scaffold for each GPU model. The trainer calls these helpers rather than implementing hardware branches itself.
- Launcher scripts (Slurm `.sbatch`/shell scripts or Python submission scripts), separate from model and runtime modules, select cluster resources and execution settings such as host staging and `NCCL_P2P_DISABLE`, then invoke the same trainer. Document cluster policy in `CLUSTERS.md` and the generic runtime interface in the scaffold README.
- Keep device-capability checks and implementation selection in attention backends and the runtime. Hardware-specific kernels are allowed behind the shared attention interface; they must preserve the requested mathematics and explicit numerical-failure behavior.
- Record selected runtime settings, resolved backends, and hardware in execution provenance separately from scientific hyperparameters. Preserve historical provenance and frozen execution checkouts. A layout refactor alone does not qualify a hardware/runtime combination; validate changes before using them for production.

## Canonical repository layout

- Organize reusable code by capability under `scaffolds/`: `scaffolds/transformer/` owns model, attention, data, training, evaluation, checkpointing, and reusable sweep/analysis machinery; `scaffolds/state_rank/` owns state-spectrum analysis. Experiments sharing a scaffold must use that implementation rather than copy it. Small scientific sweep/analysis adapters may live under a scaffold's `studies/`; add one only when configuration alone cannot express the experiment.
- Each experiment or sweep has one record under `results/<experiment>/`. Its **single canonical `README.md`** contains the question, method, home cluster, artifact locations, exact reproduction commands, findings, and limitations. Keep this narrative in one place; do not create competing scaffold experiment READMEs, per-run reports, or nested narrative READMEs. Scaffold READMEs document reusable software APIs only. Plotting may update a marked generated-results section but must preserve authored documentation.
- Store exact scientific recipes under `results/<experiment>/configs/`, run metrics and metadata under `runs/<run>/`, and figures under `figures/`. Configurations retained from an executed source revision remain byte-identical; document historical paths and overrides rather than silently rewriting them.
- An experiment record `results/<experiment>/` contains only: `README.md`,
  `configs/`, `runs/<run>/`, `provenance/manifest.json`, and optionally a derived
  `summary.csv` and `figures/`. Do not add other subdirectories such as
  `validation/`, `evidence/`, `performance/`, or `analysis/`.
- Scientific results (training and validation metrics, terminal and best
  evaluations) live only in `runs/<run>/`. Summaries, README tables, and figures
  are derived from those files and must be regenerable from them.
- `provenance/manifest.json` is the single authoritative record of how the
  experiment and each run were executed: source commit, resolved configuration
  and hash, seed, home cluster, artifact root, data manifest hashes, hardware
  (GPU type and count, node), partition, Slurm job IDs and attempts (including
  preemptions and requeues), runtime settings (e.g. host staging,
  `NCCL_P2P_DISABLE`), resolved attention backend, a summary of pre-launch
  acceptance (tests passed, initialization hash, pilot and resume outcomes),
  selected checkpoint, and log path. The trainer's `runs/<run>/metadata.json`
  is raw output that the manifest summarizes; do not copy per-attempt Slurm or
  preflight files into the record.
- Commit results, not data. Numerical archives, raw arrays, and datasets (for example
  `metrics.npz` or `spectra.npz`) stay at the experiment's home-cluster artifact root
  beside their committed run metadata, ignored by Git; the manifest records their
  paths and SHA256. Metrics logs, summaries, figures, and provenance are committed.
- Hardware qualification and kernel or throughput benchmarks describe the shared
  scaffold and runtime, not an experiment's question; retain them in
  `results/shared_transformer_validation/`.
- Keep the canonical provenance manifest at `results/<experiment>/provenance/manifest.json`. Record source commits, resolved configurations and hashes, seeds, home cluster, artifact root, and selected checkpoints. Provenance is the authoritative machine-readable record; summarize its important locations in the experiment README.
- Store checkpoints separately at `checkpoints/<experiment>/<run>/<step>/` on the experiment's home cluster. For completed pretraining retain only one validation-best checkpoint per run, which need not be the latest or final checkpoint. Preserve terminal metrics independently. Other experiments retain no checkpoints or intermediate data unless indispensable for reproduction or use.
- A concluded numerical failure is a scientific result: record its stopping step, last valid metrics, failure diagnostics, and one matching failure log. Do not label it as a completed training run. Retain a diagnostic checkpoint only when justified in provenance and the README.
- Retain one successful Slurm log per retained run, benchmark, or validation under `logs/<experiment>/.../slurm-<job_id>.out` on the experiment's home cluster, beside `checkpoints/` and outside `results/`, matching metadata. Remove superseded and failed-run logs after debugging is resolved.
- The root `README.md` is the experiment index; `CLUSTERS.md` documents cluster operations. Keep datasets outside the repository. Environments, temporary caches, execution worktrees, checkpoints, and logs are local artifacts, not Git contents.

## Experiment home cluster

- For a new experiment, the cluster from which the user requests it is its home cluster, unless the user explicitly specifies another cluster. Determine the current cluster from the session's execution environment. Do not inherit the home cluster from related experiments, comparison baselines, or their historical configurations, and do not ask the user to choose a cluster when the request's location already establishes it.
- Before continuing an existing experiment, check its recorded home cluster in provenance and its canonical README. If the request comes from a different cluster, tell the user which cluster they are on and which cluster owns the experiment, and flag the mismatch as a likely request-location mistake before launching or resuming. Do not silently change clusters or interpret the request as permission to migrate; migration requires an explicit user instruction.
- Keep each experiment's sweep runs, resumptions, checkpoints, and retained logs on its home cluster by default; different experiments may have different home clusters.
- Record the home cluster and artifact root in provenance and the README. A clone shares committed code, configurations, and scientific results; it does not provide ignored artifacts. Report unavailable or unverified external checkpoints accurately instead of claiming they were moved or checked.
- When migrating an experiment, transfer and verify required checkpoints and supporting artifacts, provide the matching data and environment, and update its recorded locations before continuing. Avoid running the same run concurrently on both clusters.
- Git ignoring checkpoints only excludes them from version control. During training retain the latest resumable checkpoint and validation best as needed; apply best-only cleanup after completion.

## Experiment lifecycle and retention

- Develop and run each new experiment on a dedicated branch in a separate working checkout. A sweep may share one branch. Keep unfinished or unreviewed experiment records separate from `main`; shared scaffold maintenance can be integrated independently without changing frozen execution checkouts.
- Freeze the execution source, environment, manifests, and referenced artifacts while jobs can run or requeue. A layout migration must preserve historical source commits and checksummed references. Do not rewrite frozen manifests or scientific controls to make a refactor pass.
- Before merging a retained experiment, remove temporary files, intermediate checkpoints, caches, redundant outputs, and duplicated documentation. Preserve the canonical README, exact recipes, provenance, scientific metrics/figures, selected checkpoints, and matching logs.
- Maintain artifact exclusions in the committed root `.gitignore`, using shared patterns. Do not use `.git/info/exclude` or machine-local global exclusions for repository artifacts. Exclude checkpoint and log directories and any runtime checkpoint symlinks.
- After cleanup and review, merge the retained experiment into `main`. Confirm indispensable artifacts live safely outside its temporary checkout, then remove the checkout and delete the merged branch. No Git tag is required.

## Historical artifact migration

- Preserve exact historical configurations, manifests, and source commits in the
  canonical experiment record and Git history. Historical absolute paths describe
  the original execution; current locations belong in `provenance/manifest.json`.
- Retire old-path compatibility aliases, duplicate root manifests, transitional
  ignore rules, and superseded execution checkouts once all their consumers are
  finished or explicitly migrated. Reproduce historical implementations by
  exporting their recorded Git commits instead of retaining extra source trees.
- Checkpoints and logs on an inaccessible home cluster retain their recorded existing
  location until a verified physical migration. Record the canonical destination
  and migration status without claiming it already exists.
