# Transformer++ diagnostics

Closed records of how trained Transformer++ models behave, kept to explain results in the
study's topics. Each diagnostic has its own run in the study, a selection, and figures that
regenerate from it; none enters a canonical comparison.

| Diagnostic | Question | Opened |
| --- | --- | --- |
| [extrapolation_scores](extrapolation_scores/README.md) | Why does degree-four polynomial attention break past its 2,048-token training length while exp does not? | 2026-11-01 |
| [extrapolation_ablation](extrapolation_ablation/README.md) | Why does query attention break past 2,048 tokens: keys beyond the training distance, their RoPE phases, or their number, and does the picture change on single long documents? | 2026-11-02 |
