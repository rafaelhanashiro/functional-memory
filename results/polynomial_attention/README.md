# Taylor attention sweep

Source: `1a8836a75836671cdf83a3ab7763425a6a88cac7`. Identity: `6bae4c76b615f8f1385b608d750948799b9e00c0f2166073009a9efc70cc814a`.

1/6 runs complete. Endpoints use committed terminal validation at the common token budget; all other runs are excluded.

The degree-zero control is causal prefix averaging; its Q/K projections receive no attention-derived learning signal.

| Kernel | Parameters | Status | Terminal PPL |
|---|---:|---|---:|
| m=0 | 199,906,304 | complete | 47.0653 |
| m=2 | 199,906,304 | queued | — |
| m=4 | 199,906,304 | queued | — |
| m=8 | 199,906,304 | queued | — |
| m=16 | 199,906,304 | queued | — |
| Exact softmax | 199,906,304 | queued | — |

Retained results from earlier source revisions:

- m=0: `2484f00e9a32f1571b330fff5788d9a86175235f`.

[Machine-readable results](summary.csv) · [Terminal PPL](ppl_vs_degree.pdf) · [Learning curves](learningcurves.pdf)
