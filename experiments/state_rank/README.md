# State rank

Measure the tokenwise spectrum of an attention state before output processing.
Llama uses the implicit unnormalized softmax numerator state; Gated DeltaNet
uses its native recurrent state. Threshold rank counts singular values greater
than `1e-4` times the largest. Participation ratio uses squared singular values.

Run from the repository root using its existing PyTorch environment:

```bash
uv run --frozen experiments/state_rank/spectrum.py --layer 8 --tokens 512
uv run --frozen experiments/state_rank/spectrum.py --model gated-deltanet --layer 12
uv run --frozen python -m pytest
```

For Slurm, synchronize the environment first and use `.venv/bin/python` with
the same script arguments. Keep the successful combined log at
`results/logs/<experiment>/slurm-<job_id>.out`.

Retained results remain in their original root `results/` directories. Moving
the source did not alter their metadata, data, or figures. To regenerate plots
intentionally, pass `--replot <result-directory>`; this rewrites result metadata
and should not be used merely to verify a source-code relocation.
