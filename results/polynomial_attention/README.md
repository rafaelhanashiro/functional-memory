# Taylor attention sweep

Source: `1a8836a75836671cdf83a3ab7763425a6a88cac7`. Identity: `6bae4c76b615f8f1385b608d750948799b9e00c0f2166073009a9efc70cc814a`.

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

Retained results from earlier source revisions:

- m=0: `2484f00e9a32f1571b330fff5788d9a86175235f`.

[Machine-readable results](summary.csv) · [Terminal PPL](ppl_vs_degree.pdf) · [Learning curves](learningcurves.pdf)

Final cleanup is recorded in [cleanup.json](cleanup.json): each retained training run, benchmark, and validation has one successful Slurm log matching its metadata. The degree-zero preflight is retained separately in [preflight-m000-retained](preflight-m000-retained/metadata.json), the fused preflight in [preflight](preflight/metadata.json), and the transferred-data validation in [data_validation](data_validation/metadata.json). Historical manifests keep their original bytes; relocated log paths are recorded in the cleanup manifest.
