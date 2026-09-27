# State-spectrum analysis

This scaffold measures tokenwise attention-state spectra, validates recurrence
and pretrained reproduction, and renders comparisons. It owns model-specific
collection and reusable analysis; studies select recipes and launch resources,
while result collections select runs and retain figures.

Sync an isolated environment in each worktree:

```bash
uv sync --project scaffolds/state_rank --locked --link-mode hardlink
```

`spectrum.py` collects a single pretrained layer. `compare_random.py` and
`compare_deltanet.py` prepare and execute pretrained-versus-random conditions.
Their `prepare_manifest` accepts a complete recipe mapping or a JSON recipe path;
`run_condition` validates its execution identity, input checksums, and prior
completed output before executing. Canonical manifests contain `run_directories`
and `run_config_sha256` mappings: each condition writes directly into its allocated
run directory and updates its authoritative record. A manifest without those
mappings retains the standalone workspace interface for historical reproduction.

`plot_comparison.py`, `unify_analysis.py`, and the spectrum plotting helpers derive
figures and summary tables from retained arrays. Temporary comparison workspaces
may reference canonical run directories without copying numerical archives.
Scientific rank definitions and tolerance checks stay in the collection and
analysis modules; launchers own cluster resources and runtime settings.

Run the CPU tests with:

```bash
PYTHONPATH=. scaffolds/state_rank/.venv/bin/python -m pytest \
  -c scaffolds/state_rank/pyproject.toml scaffolds/state_rank/tests
```

Tests requiring GPU inference use the configured CUDA device and qualified pinned
model cache. A skipped GPU test is not hardware qualification.
