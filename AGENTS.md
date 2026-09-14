# Repository guidance

Whenever the user mentions the "write-up," always use `overleaf/iclr/iclr2027_conference.tex` as the reference. This is the main write-up.

Keep one successful Slurm log per retained experiment at `results/logs/<experiment>/slurm-<job_id>.out`, matching its metadata. Remove superseded logs and failed-run logs once debugging is resolved.
