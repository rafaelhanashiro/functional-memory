# Taylor attention sweep

Source: `2484f00e9a32f1571b330fff5788d9a86175235f`. Identity: `24a4338e07a3863550e8968cd102c5bcfdc1fd905d04c126276e5b2c5690e123`.

0/6 runs complete. Endpoints use committed terminal validation at the common token budget; all other runs are excluded.

The degree-zero control is causal prefix averaging; its Q/K projections receive no attention-derived learning signal.

| Kernel | Parameters | Status | Terminal PPL |
|---|---:|---|---:|
| m=0 | 199,906,304 | pending | — |
| m=2 | 199,906,304 | pending | — |
| m=4 | 199,906,304 | pending | — |
| m=8 | 199,906,304 | pending | — |
| m=16 | 199,906,304 | pending | — |
| Exact softmax | 199,906,304 | pending | — |

[Machine-readable results](summary.csv) · [Terminal PPL](ppl_vs_degree.pdf) · [Learning curves](learningcurves.pdf)
