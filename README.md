# Functional memory experiments

Reusable code is organized by capability under `scaffolds/`. Each retained experiment has one README containing its method, reproduction instructions, home cluster, findings, and artifact locations.

| Experiment | Canonical documentation | Scaffold |
| --- | --- | --- |
| State rank | [State-rank analysis](results/state_rank/README.md) | [State-spectrum analysis](scaffolds/state_rank/) |
| Key dimension | [Query/key-width sweep](results/key_dimension/README.md) | [Transformer](scaffolds/transformer/README.md) |
| Polynomial attention | [Taylor-degree sweep](results/polynomial_attention/README.md) | [Transformer](scaffolds/transformer/README.md) |
| Learned quadratic attention | [Coefficient-learning results and numerical failures](results/learned_quadratic_attention/README.md) | [Transformer](scaffolds/transformer/README.md) |
| Odd Taylor attention | [Signed-kernel numerical failures](results/odd_taylor_attention/README.md) | [Transformer](scaffolds/transformer/README.md) |
| Key dimension × Taylor degree | [Joint sweep and hardware validation](results/key_dimension_taylor/README.md) | [Transformer](scaffolds/transformer/README.md) |
| Shared scaffold validation | [Correctness and sampled performance](results/shared_transformer_validation/README.md) | [Transformer](scaffolds/transformer/README.md) |

```text
scaffolds/<capability>/                 reusable code, tests, and environment definition
results/<experiment>/README.md         the experiment's complete documentation
results/<experiment>/configs/          exact scientific recipes
results/<experiment>/provenance/        authoritative manifest and execution provenance
results/<experiment>/runs/<run>/        metrics and metadata
results/<experiment>/figures/           figures
checkpoints/<experiment>/<run>/         home-cluster artifacts, ignored by Git
results/logs/<experiment>/              retained logs, ignored by Git
```

See [AGENTS.md](AGENTS.md) for canonical structure and retention rules and [CLUSTERS.md](CLUSTERS.md) for cluster operations. Pulling this repository does not transfer checkpoints, datasets, logs, caches, or environments. Source commits in provenance preserve the exact executed implementations.

The root PyTorch environment serves state-rank analysis; `scaffolds/transformer/pyproject.toml` and its lock define the separate JAX environment. Run the state-rank CPU checks with `uv run --frozen python -m pytest`; Transformer checks are documented in the shared scaffold README.

Completed collections, including concluded numerical-failure studies, are retained here with their original source history. Their temporary branches and execution checkouts are retired after artifact verification. The unfinished unnormalized-attention experiment remains on `codex/unnormalized-attention-250m` in `worktrees/unnormalized_attention/`; it uses the canonical shared scaffold. Historical source commits can be exported from Git when reproducing an original execution.

The main write-up is `overleaf/iclr/iclr2027_conference.tex`.
