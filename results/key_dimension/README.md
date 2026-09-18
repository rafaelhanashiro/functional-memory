# Key-dimension sweep

[Retention provenance](provenance.json) is the authority for resolved configuration and retained checkpoint locations. Historical source manifests below are archived launch records.

Archived launch manifests:

- Current sweep: source `3d330a2f29e067b4df017ac08d12b9b6d3154b94`; identity `a237d91a0c287ec1a182aa17114a6294983be05d9dfbb2882076a3ba44df9883`.
- Reference `/home/rafah/projects/functional_memory/results/key_dimension/sweep.json` (SHA-256 `8080d58cddad8879687c804427c8746fc1133aec133f6d657d3be8ad20b96566`): source `5e871d2659373272e26668e10911922c7201678a`; identity `0ceca430e0bff3c418fe7a21cd99107059d4d5d229d85e8582621d658795fa1e`.

8/8 runs complete. Endpoints use committed terminal validation at the common token budget; pending and incomplete runs are excluded.

| dK | Parameters | Status | Terminal PPL |
|---:|---:|---|---:|
| 2 | 179,590,144 | complete | 22.3489 |
| 4 | 180,245,504 | complete | 18.8928 |
| 8 | 181,556,224 | complete | 16.7053 |
| 16 | 184,177,664 | complete | 15.5764 |
| 32 | 189,420,544 | complete | 14.9853 |
| 64 | 199,906,304 | complete | 14.6898 |
| 128 | 220,877,824 | complete | 14.3776 |
| 256 | 262,820,864 | complete | 14.1736 |

[Machine-readable results](summary.csv) · [Terminal PPL](ppl_vs_dk.pdf) · [Learning curves](learningcurves.pdf)
