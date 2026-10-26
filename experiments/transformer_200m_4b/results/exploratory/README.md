# Exploratory 200M / 4B diagnostics

Open questions within the [200M / 4B study](../../README.md). Each topic owns its
findings and explicit selection; its runs remain under the study's `runs/`.
These diagnostics enter no canonical pretraining comparison. A topic closes by
promotion into a maintained collection or by discarding its results and runs.

| Topic | Question | Opened |
| --- | --- | --- |
| [coefficient_oracle](coefficient_oracle/README.md) | How do query, static, and free coefficients fit fixed 4B inputs, and what do their fits reveal about coefficient parameterization and conditioning? | 2026-10-23 |
| [taylor_static_continuation](taylor_static_continuation/README.md) | Does learning static coefficients after Taylor pretraining improve matched continuation? | 2026-10-24 |
| [taylor_query_continuation](taylor_query_continuation/README.md) | Does learning query-dependent coefficients after degree-four Taylor pretraining improve matched continuation? | 2026-10-24 |
