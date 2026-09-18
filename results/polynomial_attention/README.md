# Taylor attention sweep

[Provenance](provenance.json) is the retained record of per-run source commits, seeds, resolved configurations, data identity, and selected checkpoints.

6/6 runs complete. Endpoints use committed terminal validation at the common token budget; all other runs are excluded.

The degree-zero control is causal prefix averaging; its Q/K projections receive no attention-derived learning signal.

| Kernel | Parameters | Status | Terminal PPL |
|---|---:|---|---:|
| m=0 | 199,906,304 | complete | 47.0653 |
| m=2 | 199,906,304 | complete | 16.2810 |
| m=4 | 199,906,304 | complete | 14.9836 |
| m=8 | 199,906,304 | complete | 14.6781 |
| m=16 | 199,906,304 | complete | 14.7062 |
| Exact softmax | 199,906,304 | complete | 14.6942 |

Only the validation-best checkpoint is retained for each run under `experiments/polynomial_attention/checkpoints/<run>/` (on shared storage, ignored by Git). Terminal metrics above remain the scientific endpoint even when the best checkpoint is earlier.

| Run | Seed | Source commit | Best step | Best PPL |
|---|---:|---|---:|---:|
| 200m-m000-seed0 | 0 | `2484f00e9a32f1571b330fff5788d9a86175235f` | 19000 | 47.0326 |
| 200m-m002-seed0 | 0 | `1a8836a75836671cdf83a3ab7763425a6a88cac7` | 19074 | 16.2810 |
| 200m-m004-seed0 | 0 | `1a8836a75836671cdf83a3ab7763425a6a88cac7` | 19074 | 14.9836 |
| 200m-m008-seed0 | 0 | `1a8836a75836671cdf83a3ab7763425a6a88cac7` | 19074 | 14.6781 |
| 200m-m016-seed0 | 0 | `1a8836a75836671cdf83a3ab7763425a6a88cac7` | 19074 | 14.7062 |
| 200m-softmax-seed0 | 0 | `1a8836a75836671cdf83a3ab7763425a6a88cac7` | 19074 | 14.6942 |

Resolved configurations and reproduction instructions are in the [experiment scaffold](../../experiments/polynomial_attention/README.md). [Performance measurements](performance/README.md) document the execution backends and hardware.

[Machine-readable results](summary.csv) · [Terminal PPL](ppl_vs_degree.pdf) · [Learning curves](learningcurves.pdf)
