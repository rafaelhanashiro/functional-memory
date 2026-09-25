# Functional memory experiments

Reusable code is organized by capability under `scaffolds/`. Each experiment under `experiments/` has one README containing its question, method, reproduction instructions, home cluster, findings, and artifact locations.

| Experiment | Canonical documentation | Scaffold |
| --- | --- | --- |
| State rank | [State-rank analysis](experiments/state_rank/README.md) | [State-spectrum analysis](scaffolds/state_rank/) |
| Learned quadratic attention | [Coefficient-learning results and numerical failures](experiments/learned_quadratic_attention/README.md) | [Transformer](scaffolds/transformer/README.md) |
| Odd Taylor attention | [Signed-kernel numerical failures](experiments/odd_taylor_attention/README.md) | [Transformer](scaffolds/transformer/README.md) |
| Query/key dimension × Taylor degree | [Width, degree, and joint sweeps](experiments/key_dimension_taylor/README.md) | [Transformer](scaffolds/transformer/README.md) |
| GLA key dimension | [Gated linear attention across d_K](experiments/gla_key_dimension/README.md) | [Transformer](scaffolds/transformer/README.md) |
| Unnormalized attention | [Unnormalized exponential attention at 200M and 250M](experiments/unnormalized_attention/README.md) | [Transformer](scaffolds/transformer/README.md) |
| Unnormalized Taylor attention | [Degrees 1–4 without row normalization](experiments/unnormalized_taylor_attention/README.md) | [Transformer](scaffolds/transformer/README.md) |

```text
scaffolds/<capability>/                   reusable code, tests, and environment definition
scaffolds/<capability>/validation/        the scaffold's qualification and benchmark record
experiments/<experiment>/README.md        the experiment's complete documentation
experiments/<experiment>/configs/         exact scientific recipes
experiments/<experiment>/runs/<run>/      run metrics and metadata
experiments/<experiment>/provenance/      authoritative manifest and execution provenance
experiments/<experiment>/figures/         figures (and an optional summary.csv)
experiments/<experiment>/adapter/         the experiment's launch, acceptance, and analysis code
experiments/<experiment>/logs/            retained Slurm logs, ignored by Git
experiments/<experiment>/checkpoints/     retained checkpoints, ignored by Git
```

See [AGENTS.md](AGENTS.md) for canonical structure and retention rules and [CLUSTERS.md](CLUSTERS.md) for cluster operations. Pulling this repository does not transfer checkpoints, datasets, logs, caches, or environments. Source commits in provenance preserve the exact executed implementations.

Each scaffold defines its environment: `scaffolds/transformer/pyproject.toml` (JAX) and `scaffolds/state_rank/pyproject.toml` (PyTorch). Every worktree syncs its own with `uv sync --project scaffolds/<capability> --locked --link-mode hardlink` (add `--extra cuda` for the Transformer); test commands are in the scaffold READMEs.

The scaffolds' own qualification records: [Transformer correctness, kernel and throughput benchmarks, and hardware qualification](scaffolds/transformer/validation/README.md). Each run's manifest records its source commit, where the exact executed code lives.

The main research notes are `overleaf/iclr/iclr2027_conference.tex`. The Overleaf project is cloned directly into the local, Git-ignored `overleaf/` directory; its own `iclr/` subdirectory holds the main notes and template. These are research notes using the ICLR LaTeX format, not an ICLR submission. Supporting notes live in `overleaf/notes/`, figures in `overleaf/media/`, and older drafts in `overleaf/archive/`; `notation.tex` and `notes.cls` are shared project files.
