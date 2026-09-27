# Run records

Shared bookkeeping separates immutable run inputs from mutable execution evidence.
A study has `sweep.yaml` with common metadata and an ordered `run_ids` list. Each
`runs/<run_id>/` owns its complete `config.yaml`, authoritative `record.json`, and
raw measurements. `runs.csv` is derived from those files.

## Allocate, inspect, and validate

Use the environment of the scaffold executing the study:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records create \
  --study experiments/<study> --config <recipe.yaml> --label <description>
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records index experiments/<study>
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records validate experiments/<study>
```

`create` prints the allocated directory. It generates a collision-checked ID,
registers the run, sets its output paths, and hashes the recipe. For Transformer
recipes the CLI resolves scaffold defaults into a complete config and resets the
source-commit request to `auto`; execution resolves that request separately.
Changing scientific settings requires a fresh allocation. A resume reuses the same
ID and recipe; it does not call `create`.

Commit prepared recipes and study membership before production launch. A study’s
frozen execution set includes all of its registered input recipes. Allocate the
whole sweep before any jobs can run or requeue.

## Python interface

- `create_run(study, fully_resolved_config, label=None)` allocates a new execution.
  The API requires a complete scientific recipe from its caller and does not infer
  model defaults. Transformer `train.workdir` and `train.checkpoint_dir`, or a
  state-analysis `output_directory`, point into the allocated run.
- `load_run(run)` reads one record and checks its directory identity.
- `load_study(study)` combines common study facts and per-run records in memory.
  It does not write a second aggregate provenance manifest.
- `run_directory(study, id_or_unique_label)` resolves a run. Saved result selections
  use IDs; labels are only a convenience for interactive lookup.
- `update_record(run, changes)` atomically deep-merges execution facts while
  rejecting changes to run identity or the recipe descriptor. Lists are replaced.
- `capture_artifacts(run, checkpoint_directory=None, include_log=False)` hashes
  available outputs and returns provenance updates. Pass the selected checkpoint's
  step directory after pruning. Launchers capture the final log after the worker
  exits through `python -m scaffolds.launch.finalize --run <run> --exit-code <code>`.
- `validate_record(run, verify_hashes=True)` checks identity, recipe integrity, and
  available artifact hashes. Missing external artifacts are reported as unavailable;
  missing or modified input recipes are errors. Source-commit-qualified historical
  artifacts are not compared with current checkout files.
- `regenerate_index(study)` derives a deterministic CSV from recipes, records, and
  raw metrics. Missing historical settings remain empty; current defaults are not
  retroactively assigned to old results.

The input recipe and raw runtime `config.json` serve different purposes: the former
is frozen before launch, while the latter records resolved backend, source, and
runtime paths. Raw metadata and metrics remain beside the record. Numerical arrays,
checkpoints, temporary acceptance outputs, and logs are ignored by Git and retained
on their home cluster.

Run the bookkeeping tests with `python -m pytest scaffolds/records/tests` from a
scaffold environment that provides PyYAML and pytest.
