# Transformer++ diagnostics

Closed records of how trained Transformer++ models behave, kept to explain results in the
study's topics. Each diagnostic pins the runs whose measurements it uses and has figures
that regenerate from them; none enters a canonical comparison. Diagnostics of existing
training histories need no additional training or checkpoint execution.

| Diagnostic | Question | Opened |
| --- | --- | --- |
| [extrapolation_scores](extrapolation_scores/README.md) | Why does degree-four polynomial attention break past its 2,048-token training length while exp does not? | 2026-11-01 |
| [extrapolation_ablation](extrapolation_ablation/README.md) | Why does query attention break past 2,048 tokens: keys beyond the training distance, their RoPE phases, or their number, and does the picture change on single long documents? | 2026-11-02 |
| [sos_training_dynamics](sos_training_dynamics/README.md) | Are the completed sum-of-squares query-polynomial runs stable, and how do their effective coefficients evolve? | 2026-11-13 |
| [query_parameter_dynamics](query_parameter_dynamics/README.md) | How do one middle-layer head's a, b, and W norms evolve in regular query polynomials at degrees 8 and 12, and do its coefficients vary across queries? | 2026-11-13 |
