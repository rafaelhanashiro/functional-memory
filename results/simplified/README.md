# Simplified figures

Easy-to-read versions of canonical result figures, for sending to people outside the
project. Each subdirectory simplifies the result collection of the same name and reads
its selections; findings, limitations, and full figures stay in that collection.

| Figures | Source |
| --- | --- |
| [attention_scaling/](attention_scaling/) | [results/attention_scaling](../attention_scaling/README.md) |

Regenerate after the source collection changes, in the primary checkout on Kennel:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m results.simplified.attention_scaling.plot
```
