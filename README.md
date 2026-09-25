# Functional memory experiments

Reusable code is organized by capability under `scaffolds/`. Each retained experiment has one README containing its method, reproduction instructions, home cluster, findings, and artifact locations.

| Experiment | Canonical documentation | Scaffold |
| --- | --- | --- |
| State rank | [State-rank analysis](results/state_rank/README.md) | [State-spectrum analysis](scaffolds/state_rank/) |
| Learned quadratic attention | [Coefficient-learning results and numerical failures](results/learned_quadratic_attention/README.md) | [Transformer](scaffolds/transformer/README.md) |
| Odd Taylor attention | [Signed-kernel numerical failures](results/odd_taylor_attention/README.md) | [Transformer](scaffolds/transformer/README.md) |
| Query/key dimension × Taylor degree | [Width, degree, and joint sweeps](results/key_dimension_taylor/README.md) | [Transformer](scaffolds/transformer/README.md) |
| GLA key dimension | [Gated linear attention across d_K](results/gla_key_dimension/README.md) | [Transformer](scaffolds/transformer/README.md) |
| Shared scaffold validation | [Correctness, kernel and throughput benchmarks, and hardware qualification](results/shared_transformer_validation/README.md) | [Transformer](scaffolds/transformer/README.md) |
| Unnormalized attention | [Unnormalized exponential attention at 200M and 250M](results/unnormalized_attention/README.md) | [Transformer](scaffolds/transformer/README.md) |
| Unnormalized Taylor attention | [Degrees 1–4 without row normalization](results/unnormalized_taylor_attention/README.md) | [Transformer](scaffolds/transformer/README.md) |

```text
scaffolds/<capability>/                 reusable code, tests, and environment definition
results/<experiment>/README.md         the experiment's complete documentation
results/<experiment>/configs/          exact scientific recipes
results/<experiment>/provenance/        authoritative manifest and execution provenance
results/<experiment>/runs/<run>/        metrics and metadata
results/<experiment>/figures/           figures
checkpoints/<experiment>/<run>/         home-cluster artifacts, ignored by Git
logs/<experiment>/                      retained Slurm logs, ignored by Git
```

See [AGENTS.md](AGENTS.md) for canonical structure and retention rules and [CLUSTERS.md](CLUSTERS.md) for cluster operations. Pulling this repository does not transfer checkpoints, datasets, logs, caches, or environments. Source commits in provenance preserve the exact executed implementations.

The root PyTorch environment serves state-rank analysis; `scaffolds/transformer/pyproject.toml` and its lock define the separate JAX environment. Run the state-rank CPU checks with `uv run --frozen python -m pytest`; Transformer checks are documented in the shared scaffold README.

Completed collections, including concluded numerical-failure studies, are retained here with their original source history. Their temporary branches and execution checkouts are retired after artifact verification. Historical source commits can be exported from Git when reproducing an original execution.

The main research notes are `overleaf/iclr/iclr2027_conference.tex`. The Overleaf project is cloned directly into the local, Git-ignored `overleaf/` directory; its own `iclr/` subdirectory holds the main notes and template. These are research notes using the ICLR LaTeX format, not an ICLR submission. Supporting notes live in `overleaf/notes/`, figures in `overleaf/media/`, and older drafts in `overleaf/archive/`; `notation.tex` and `notes.cls` are shared project files.
