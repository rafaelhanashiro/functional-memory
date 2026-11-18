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
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records validate experiments/<study> [<run> ...]
```

`create` prints the allocated directory. It generates a collision-checked ID,
registers the run, sets its output paths, and hashes the recipe. For Transformer
recipes the CLI resolves scaffold defaults into a complete config and resets the
source-commit request to `auto`; execution resolves that request separately. The CLI
also assigns the target study as the W&B group, uses the new label or ID as its
name, and clears the copied local W&B directory.
`validate` checks every run, or only the named IDs or labels. It requires the record's
`status` to agree with the run's outcome files: `completion.json` only for `complete`, and
`numerical_failure_step_*.json` only for a numerical failure (exactly one). It re-hashes each
run's recipe and raw outputs, but checks retained checkpoints only against their
recorded file names and sizes, because re-hashing reads every checkpoint byte (tens of
gigabytes per study on a shared filesystem). Add `--checkpoint-hashes` to re-hash
checkpoint contents, and name only the runs whose checkpoints were copied, for example
after moving completed runs into the artifact root or migrating them between clusters.

Changing scientific settings requires a fresh allocation. A resume reuses the same
ID and recipe; it does not call `create`.

New dot-product Transformer runs use QK normalization unless the user explicitly
requests plain QK scores. An omitted `model.qk_norm` resolves to `true`. A copied
legacy recipe containing `qk_norm: false` is rejected by `create` unless
`--allow-plain-qk` is supplied, so reusing an old recipe cannot silently disable the
canonical default. For the usual rerun, change the input recipe to
`model.qk_norm: true` before allocating. The flag leaves an explicitly requested
plain-score recipe unchanged. GLA and degree-zero attention are exempt because they
do not use QK normalization. Historical records and resumptions are unaffected.

Likewise, a recipe that a Mosaic GPU kernel implements (normalized fixed Taylor, or tanh
query-conditioned or static-coefficient unnormalized polynomial attention of degree 1-8) is
rejected with `model.attention_backend: fused` unless
`--allow-triton-attention` is supplied; set `mosaic` (or `auto`) before allocating.

A dot-product recipe with RoPE or no positional encoding on the `blocked` or `auto` backend
trains on the blocked backend's causal strips: an omitted
`model.attention_blocked_implementation` resolves to `strips`, and an explicit `scan` is
rejected unless `--allow-scan-blocked-attention` is supplied. Recipes on other backends keep
the inert `scan`.

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
- `capture_artifacts(run, checkpoint_directory=None, include_log=False,
  diagnostic_checkpoints=None)` hashes available outputs and returns provenance updates.
  Pass the selected checkpoint's step directory after pruning. `diagnostic_checkpoints`
  lists other retained checkpoint directories, each a mapping with its `path`, a nonempty
  `justification`, and optional descriptive fields; they become `diagnostic_checkpoints`
  directory descriptors (an empty list records that none are kept, and `None` leaves the
  record's entry unchanged). Launchers capture the final log after the worker
  exits through `python -m scaffolds.launch.finalize --run <run> --exit-code <code>`.
- `validate_record(run, verify_hashes=True, checkpoint_hashes=True)` checks identity,
  recipe integrity, and available artifact hashes, and requires every
  `diagnostic_checkpoints` entry to be a directory descriptor with a justification.
  With `checkpoint_hashes=False`, checkpoint trees are compared by file names and sizes
  and listed as `size_checked`. Missing external artifacts are reported as unavailable;
  missing or modified input recipes are errors.
- `regenerate_index(study)` derives a deterministic CSV from recipes, records, and
  raw metrics, including attention family, coefficient conditioning, home cluster,
  and training budget. Coefficient conditioning distinguishes static learned
  coefficients from coefficients predicted from the incoming query activation.
  `coefficient_orders` lists explicitly selected polynomial powers as a JSON array;
  it stays empty when the recipe does not restrict the powers.
  `coefficient_zero_constant` records structural removal of direct constants or
  both SoS factor constants; it stays unknown for recipes that predate the field.
  Missing historical settings remain empty; current defaults are not
  retroactively assigned to old results.

## Result selections

A result's selection files (`results/<analysis>/*.yaml`, or
`results/exploratory/<topic>/*.yaml`) pin every plotted run to the
SHA256 of its scientific inputs: `config.yaml`, `metrics.jsonl`, `completion.json`,
`metadata.json`, `best_validation.json`, and any `numerical_failure_step_*.json`.
Provenance and bookkeeping (`record.json`, `launch.json`, `retention.json`, W&B and
data-identity files, raw `config.json`) are not pinned, so recording attempts,
artifacts, or retention never invalidates a figure. A run of a state analysis instead
pins its recipe hash and the inputs its record captured.

Comparison-specific settings and primary/control run choices belong in result
selections; they do not belong to study-wide defaults.

After pointing a role at a new `run_id`, editing a grid's cells, or correcting a
measurement, re-pin and then regenerate:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh results/<analysis>
```

`refresh` rewrites the input pins, lists each change, and preserves the roles in a
`runs` list. Renaming a study folder (`git mv experiments/<old> experiments/<new>`) is
followed by

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records rename-study experiments/<new> --from <old> --to <new>
```

which points every run record's retained-file locations (repository-relative, and
absolute under this checkout), every selection under `results/` and the study's
`results/`, and `sweep.yaml` at the new folder. Frozen recipes keep their recorded
paths, as do execution history (submission commands, attempts, resolved runtime
configurations) and locations on other roots; after moving a cluster's artifact tree,
`--artifact-root <root>` moves those locations too. A branch that predates the rename
merges with `git -c merge.directoryRenames=true merge <branch>`, after which the same
command handles the runs it added. Grids pin exactly their `cells`; array analyses pin exactly their
`collections` and comparison baseline/condition roles. A figure whose inputs changed
fails verification until its selection is refreshed. In Python,
`selections.run_sources(root, study, run_id)`
returns one run's pins and `selections.verify_pinned_runs(root, runs)` checks them.

The input recipe and raw runtime `config.json` serve different purposes: the former
is frozen before launch, while the latter records resolved backend, source, and
runtime paths. Raw metadata and metrics remain beside the record. Numerical arrays,
checkpoints, temporary acceptance outputs, and logs are ignored by Git and retained
on their home cluster.

Run the bookkeeping tests with `python -m pytest scaffolds/records/tests` from a
scaffold environment that provides PyYAML and pytest.
