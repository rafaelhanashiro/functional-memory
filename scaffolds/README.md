# Reusable scientific software

Code is organized by capability, shared by experiments that need the same machinery.

| Capability | Interface | Environment |
| --- | --- | --- |
| [Transformer](transformer/README.md) | `python -m scaffolds.transformer.train CONFIG.yaml` | `transformer/pyproject.toml` and its locked JAX environment |
| [State spectrum](state_rank/) | `python -m scaffolds.state_rank.spectrum`, the comparison drivers, and `unify_analysis` | `state_rank/pyproject.toml` and its locked PyTorch environment |
| [Launch helpers](launch/) | `python3 scaffolds/launch/frozen.py check` | Standard library only |

An experiment's question, exact recipes, reproduction commands, home cluster,
findings, and artifact locations belong in its single
`experiments/<experiment>/README.md`. The experiment's `adapter/` configures these
capabilities (launch, acceptance, and analysis); it imports scaffold code, and no
scaffold imports experiment code. Nothing in a scaffold is named after one
experiment. A scaffold may keep one record about itself, `validation/`, for its
correctness checks, benchmarks, and hardware qualification.

Each worktree syncs its own environment from a scaffold's lock with
`uv sync --project scaffolds/<capability> --locked --link-mode hardlink`.
See [repository guidance](../AGENTS.md) for the layout and experiment lifecycle.
