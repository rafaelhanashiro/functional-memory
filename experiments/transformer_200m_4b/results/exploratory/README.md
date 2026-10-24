# Exploratory 200M / 4B diagnostics

Open questions within the [200M / 4B study](../../README.md). Each topic owns its
findings and explicit selection; its runs remain under the study's `runs/`.
These diagnostics enter no canonical pretraining comparison. A topic closes by
promotion into a maintained collection or by discarding its results and runs.

| Topic | Question | Opened |
| --- | --- | --- |
| [coefficient_oracle](coefficient_oracle/README.md) | How does sharing coefficients across sequences and query positions affect fitting with a frozen 4B backbone? | 2026-10-23 |
| [taylor_static_continuation](taylor_static_continuation/README.md) | Does learning static coefficients after Taylor pretraining improve matched continuation? | 2026-10-24 |
