# Spline attention

Opened 2026-11-08.

**Question.** Can static or query-dependent polynomial splines replace the global
polynomial attention function? The proposed comparison has no FoX gate. Both
variants would use the same fixed knots and continue their boundary polynomial
pieces outside the knot interval. Input-dependent knots are deferred.

## Score calibration

Run `rb1a3a991da6c` reads the regular degree-four static model `r4339a245820e`
and query + W decay model `rca97cc9db8f2`, each at its seed-42 initialization and
retained validation-best checkpoint (step 7,500). Both use the 15-layer width-896
Transformer++ recipe, 2,048-token contexts, RoPE, QK RMSNorm with learned head
scaling, and unnormalized attention followed by head RMSNorm. There is no FoX gate.
Checkpoints and configurations remain unchanged; the diagnostic performs no
optimizer updates and creates no checkpoints.
Initialization is reconstructed from each source recipe under the diagnostic's
recorded scaffold; it is not an archived step-zero checkpoint or a proven
bitwise replay of historical initialization.

All snapshots use the same 16 training rows, starting at row zero with stride
100,003. These rows lie within the sources' consumed training stream. The run
captures the actual operands supplied to attention after QK normalization, head
scaling, and RoPE, and recomputes their dot products in FP32 with highest matmul
precision. It retains per-layer/head histograms, sampled score quantiles, per-query
extrema, and score-count and absolute-polynomial-weight coverage for candidate
symmetric ranges. It samples 128 query positions uniformly without replacement per sequence,
using every causal key for those positions, plus 4,096 uniformly sampled causal
pairs per sequence for quantiles. The immutable diagnostic recipe pins the source
recipes, checkpoint files, dataset manifest, sample settings, and candidate ranges.

**Open decision.** Select a shared fixed knot grid after examining both ordinary
scores and rare high-weight keys. A candidate range is a calibration heuristic,
not a bound on future spline-model scores. A later spline pilot must check interval
occupancy and polynomial extrapolation. No spline training run has been allocated.

## Execution

Home cluster: Kennel. Execute from the isolated diagnostic worktree after syncing
its environment and committing the execution files:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m pytest -q experiments/transformer_pp/tests/test_calibrate_spline_scores.py
sbatch --parsable --job-name=spline-score-calibration \
  --output="$PWD/experiments/transformer_pp/runs/rb1a3a991da6c/training.log" \
  experiments/transformer_pp/calibrate_spline_scores.sbatch \
  "$PWD" "$(git rev-parse HEAD)" rb1a3a991da6c
```

The launcher requests one H100 in `standard`, freezes the execution set, and uses
the worktree's CUDA 12 environment. Runtime qualification checks the diagnostic
statistics against an explicit reference before processing model snapshots.

## Findings and regeneration

Pending the calibration run. Measurements remain under its run directory.
