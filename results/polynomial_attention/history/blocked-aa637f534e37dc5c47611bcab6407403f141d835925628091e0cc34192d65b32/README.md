# Taylor attention sweep

Source: `2484f00e9a32f1571b330fff5788d9a86175235f`. Identity: `aa637f534e37dc5c47611bcab6407403f141d835925628091e0cc34192d65b32`.

1/6 runs complete. Endpoints use committed terminal validation at the common token budget; all other runs are excluded.

The degree-zero control is causal prefix averaging; its Q/K projections receive no attention-derived learning signal.

| Kernel | Parameters | Status | Terminal PPL |
|---|---:|---|---:|
| m=0 | 199,906,304 | complete | 47.0653 |
| m=2 | 199,906,304 | pending | — |
| m=4 | 199,906,304 | pending | — |
| m=8 | 199,906,304 | pending | — |
| m=16 | 199,906,304 | pending | — |
| Exact softmax | 199,906,304 | pending | — |

[Machine-readable results](summary.csv) · [Terminal PPL](ppl_vs_degree.pdf) · [Learning curves](learningcurves.pdf)
