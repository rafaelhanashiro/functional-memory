# Repository guidance

Whenever the user mentions the "write-up," always use `overleaf/iclr/iclr2027_conference.tex` as the reference. This is the main write-up.

Keep one successful Slurm log per retained experiment at `results/logs/<experiment>/slurm-<job_id>.out`, matching its metadata. Remove superseded logs and failed-run logs once debugging is resolved.

## Experiment home cluster

- Assign each experiment one home cluster before launching it. Keep its sweep runs, resumptions, checkpoints, and retained Slurm logs on that cluster by default; different experiments may have different home clusters.
- Record the home cluster and artifact root in the experiment's provenance, along with checkpoint and log paths. A clone on another cluster shares committed code, configurations, and scientific results; it does not provide the checkpoints, datasets, environments, or other ignored artifacts.
- When migrating an experiment, transfer and verify the required checkpoints and supporting artifacts, make the matching data and environment available, and update its home cluster and artifact locations before continuing. Avoid running the same run concurrently on both clusters.
- Git ignoring a checkpoint only excludes it from version control. During training, retain the latest resumable checkpoint and the validation-best checkpoint as needed; apply the validation-best-only retention rule after the experiment is complete.

## Experiment lifecycle and retention

- Develop and run each new experiment on a dedicated branch in a separate working checkout. A sweep can share one branch, with separate directories for its individual runs. Keep the experiment separate from `main` until it is complete and its results have been reviewed.
- Before merging a retained experiment, clean up temporary files, intermediate checkpoints, caches, and redundant outputs. Keep `main` and the retained experiment directories minimal.
- Store the scaffold code and configuration needed to rerun the experiment under `experiments/<experiment>/`. For pretraining, retain only one validation-best checkpoint per run under `experiments/<experiment>/checkpoints/<run>/`; "best" does not mean latest or final. Other experiments should retain no checkpoints or intermediate data by default. Keep additional data only when needed to reproduce or use the experiment.
- Store scientific results, including metrics, figures, and reports, under `results/<experiment>/`, separate from the scaffold and checkpoints. Preserve terminal metrics even when the retained best checkpoint is from an earlier step. Keep a small provenance manifest recording resolved configurations, seeds, source commits, and the selected checkpoints. Retain Slurm logs as specified above.
- Do not add large artifacts such as checkpoints, datasets, or caches to Git. Keep them on disk or shared storage and ensure their paths are ignored by Git.
- Maintain artifact exclusions in the committed root `.gitignore`, using shared patterns for all experiment families. Do not use `.git/info/exclude` or machine-local global exclusions for repository artifacts. New experiments must inherit the same rules, including exclusions for checkpoint directories and checkpoint symlinks in result directories.
- Once cleanup is complete, merge the retained experiment into `main`. Before removing the temporary checkout, ensure that its best checkpoints, results, and any indispensable data are safely retained outside that checkout. Then remove the experiment checkout and delete the merged branch. No Git tag is required.
