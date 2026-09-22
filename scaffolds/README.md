# Reusable scientific software

Code is organized by capability, shared by experiments that need the same machinery.

| Capability | Interface | Environment |
| --- | --- | --- |
| [Transformer](transformer/README.md) | `python -m scaffolds.transformer.train CONFIG.yaml` | `transformer/pyproject.toml` and its locked JAX environment |
| [State spectrum](state_rank/) | `python -m scaffolds.state_rank.spectrum` and `unify_analysis` | Root `pyproject.toml` and its locked PyTorch environment |

An experiment's question, exact recipes, reproduction commands, home cluster,
findings, and artifact locations belong in its single
`results/<experiment>/README.md`. Scientific configurations and results stay in
that experiment's record; checkpoints live separately under
`checkpoints/<experiment>/<run>/` and are ignored by Git.

A capability may expose small `studies/` adapters when a scientific sweep requires
extra orchestration or analysis. Adapters use the common implementation; they are
not copies of the model, data pipeline, or training loop. Scaffold documentation
explains reusable APIs rather than repeating experiment narratives.

Develop and test changes in separate checkouts. Keep active experiment execution
checkouts and their dependency environments frozen. Retained provenance records
the original source commits, including implementations predating this layout.
See [repository guidance](../AGENTS.md) for retention and migration rules.
