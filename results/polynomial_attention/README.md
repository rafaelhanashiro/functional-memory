# Taylor attention sweep

Source: `cf363ea7e84f084a2e9eaae60369db44e37b0832`. Identity: `9cc83d42ad434d61d61a5f47fa756a7a14fb8a2976afa85fcd6b40e708d85128`.

0/6 runs complete. Endpoints use committed terminal validation at the common token budget; all other runs are excluded.

Engaging preflight job `22846336` was cancelled before starting on 2026-09-17 UTC, at the user's request to move this sweep to Kennel. No production jobs were submitted on Engaging. The pending rows below are planned runs, not active Slurm jobs.

These records retain the original Engaging paths, source identity, and job ID. For Kennel, create a fresh sweep manifest and run configurations after adapting the cluster settings, environment, and dataset paths; do not reuse this submission manifest.

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
