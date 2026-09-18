# Polynomial attention throughput audit

Attention was the main avoidable cost. The original fused implementation still spent about 70% of degree-2 and 77% of degree-16 update time inside attention. Removing needless FP32 conversions and using three BF16 components for mixed products reduced complete update time again, while retaining FP32 polynomial/normalization arithmetic and the FP32 output saved for backward.

Final source: `0f655fb7098ae73e5923fc295fe0b5c3012184ee`; successful Slurm job `199635`, four H100 SXM GPUs on `dogo`, global batch 256, microbatch 64, four accumulation steps, 2,048 prediction positions, 199,906,304 parameters. Log: `/data/cl/u/rafah/home/projects/functional-memory/results/logs/polynomial_attention_throughput/slurm-199635.out`.

## Complete training update timings

| Variant | Original blocked (s) | First fused (s) | Revised fused (s) | Training only, 10B tokens | Wall-time budget |
|---|---:|---:|---:|---:|---:|
| Degree 2 | 1.644 | 0.939 | 0.464 | 2.46 h | 2.74–2.85 h |
| Degree 4 | — | — | 0.485 | 2.57 h | 2.85–2.96 h |
| Degree 8 | — | — | 0.533 | 2.83 h | 3.11–3.22 h |
| Degree 16 | 1.922 | 1.230 | 0.686 | 3.64 h | 3.92–4.03 h |
| Softmax (cuDNN) | — | 0.358 | 0.357 | 1.89 h | 2.17–2.29 h |

Compared with the original blocked full-step benchmark, the revised kernel is 3.55× faster at degree 2 and 2.80× at degree 16. The original blocked measurements are from the earlier successful four-SXM job 199623; the first-fused comparison was remeasured in allocation 199628. Final timings use three warmups and ten synchronized updates; profiles capture three additional updates after those timings.

## Profile and remaining work

| Variant | Attention time/update | Attention share of wall time | Collectives/update |
|---|---:|---:|---:|
| 2 | 175.9 ms | 37.9% | 11.0 ms |
| 4 | 200.2 ms | 41.1% | 10.7 ms |
| 8 | 251.4 ms | 46.9% | 9.1 ms |
| 16 | 416.3 ms | 59.8% | 6.4 ms |
| softmax | 60.2 ms | 16.8% | 9.3 ms |

The comparison leaves about 0.28–0.30 seconds per update outside attention across variants. Polynomial attention retains FP32 Horner evaluation and precision-preserving mixed products, so additional attention cost relative to cuDNN is expected. This is a measured, validated implementation; no claim of global optimality is made. CUDA command-buffer replay hides MLP/loss attribution; unidentified GEMMs remain explicitly unattributed. Recorded GPU activity does not measure SM utilization.

## Numerical and launch validation

All 56 GPU tests passed: independent FP64 output/QKV-gradient references, BF16 production context, odd widths and tail tiles, causality, tiny cotangents, invalid-score behavior, and overflow cases. Direct mixed-dot tests reject one- and two-component approximations and check finite values near the FP32 range limit. BF16 components can underflow; the implementation does not promise bitwise IEEE behavior for subnormals.

All 28 tile candidates (14 configurations × two degrees) passed parity. The existing 64×64/four-warp/two-stage setting won degree 2; a degree-16 alternative improved the isolated forward/backward call by only 2.1%. The shared existing tile setting was retained. Doubling global microbatch to 128 and halving accumulation to two changed full-step speed by less than 2%, so production batch layout remains unchanged.

## Real data and runtime projection

The 100-update degree-16 pilot read the actual packed manifests directly, performed two full seven-batch validations and four checkpoint saves, and kept the 19,074-step learning-rate horizon. Its wall time was 375.5 seconds. The first seven warmed production-loop samples had a median of 0.6997 seconds/update, versus 0.6865 seconds on resident synthetic input. The warmed evaluation took 3.59 seconds, and the waited-for checkpoint took 21.49 seconds. Startup (including dataset validation and compilation) is budgeted at 256.0 seconds.

The table adds the measured data/loop difference, 20 validations, startup, and a checkpoint budget ranging from 20 waited-for validation saves to all 39 saves charged serially. The latter is conservative because nonvalidation saves can overlap training. All variants use the observed degree-16 evaluation allowance. These are projections on the measured hardware, excluding queue delays, contention changes, and retries; they are not statistical confidence bounds. The compressed pilot cadence is not scaled directly to production.

The remaining five variants total approximately 14.8–15.4 hours when run sequentially with four H100 SXM GPUs each. The completed degree-zero experiment is preserved. No production sweep was restarted during this audit.

Artifacts: `train-step.json`, `baseline-train-step.json`, `profile-summary.json`, `baseline-profile-summary.json`, `tile-tuning.json`, `microbatch128.json`, `packed-pilot.json`, `packed-data-identity.json`, `packed-validation-metrics.jsonl`, `runtime-estimate.json`, `gpu-tests.xml`, `metadata.json`, and `run.sbatch`. Source hashes and raw samples are retained. Full traces and temporary pilot checkpoints remain under `.cache/polynomial_attention/throughput/`.
