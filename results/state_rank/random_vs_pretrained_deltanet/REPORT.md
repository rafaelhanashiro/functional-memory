# Gated DeltaNet: random versus pretrained state rank

All six conditions completed: the pinned pretrained model and five independent FLA reference initializations (seeds 0–4). The comparison covers all 24 layers and all eight heads on the exact four saved 512-token inputs, including BOS. No training was performed on the random controls.

The measured object is the native 256 × 256 recurrent state before output normalization, gating, and projection. State evolution uses FP32 fused recurrence with TF32 disabled; direct SVD and analysis use FP64. Full convolution and recurrent caches are preserved during inference.

## Terminal results at layer 12

Each number first averages metrics computed separately for the four inputs and eight heads. Random columns then summarize the five seed means; the seed range is descriptive, not a confidence interval. Threshold rank uses the strict cutoff σ > 10⁻⁴ σ₁.

| Metric | Pretrained | Random mean | Random seed range |
|---|---:|---:|---:|
| Participation ratio | 3.784 | 11.512 | 8.418–15.897 |
| Threshold rank | 74.875 | 111.106 | 81.188–135.812 |
| Modes for 90% energy | 8.312 | 16.275 | 11.438–22.781 |
| Modes for 99% energy | 21.250 | 33.737 | 24.125–46.562 |

## What changes across depth

Pretrained participation ratio is below the five-seed random mean in all 24 layers, and below every individual seed mean in 22 layers. This indicates stronger concentration by the participation-ratio measure on these inputs.

Numerical rank has a different pattern: pretrained threshold rank exceeds the random mean in nine layers (7, 10, 13, 16, 17, 18, 19, 20, and 22). At layer 20, pretrained participation ratio is **8.92 versus 16.59** random, while threshold rank is **228.13 versus 126.84** and the 99%-energy rank is **71.41 versus 45.59**. Thus stronger dominant modes can coexist with a larger weak-mode tail. A blanket claim that pretraining reduces every notion of rank is not supported.

These are fresh reference initializations, not the pretrained model’s original starting weights or a tracked training trajectory. The comparison is descriptive and limited to four fixed texts. It does not isolate effects of learned decay, update gates, projections, or normalization, and state rank does not establish language-model quality. DeltaNet’s finite recurrent state and Llama’s implicit softmax state have different definitions and widths; interpret absolute ranks within each architecture.

## Figures and reproduction

- [Rank through the input, layer 12](token_metrics.png) · [PDF](token_metrics.pdf)
- [Terminal rank across all 24 layers](terminal_rank_depth.png) · [PDF](terminal_rank_depth.pdf)
- [Relative spectra at selected prefixes](normalized_spectra.png) · [PDF](normalized_spectra.pdf)
- [State energy across depth](norm_energy_depth.png) · [PDF](norm_energy_depth.pdf)
- [All 144 condition/layer summaries](summary.csv)
- [Validation and provenance audit](validation.json)

Offline tests: **134 passed**, with three CUDA tests skipped on the login node; all **three native GPU tests then passed** before the pretrained measurement. All six Slurm jobs completed with exit code 0. The independent audit verified archive checksums, unique parameter fingerprints, matching inputs and axes, causal rank bounds, saved-spectrum reconstruction of rank metrics, all 144 CSV rows, and one matching successful Slurm log per condition. Maximum full-prefix versus tokenwise state error was 1.99 × 10⁻⁵, below the predeclared 2 × 10⁻⁴ tolerance.

The historical layer-12 baseline agrees to a maximum normalized-spectrum absolute error of 2.70 × 10⁻⁷. All terminal threshold ranks and all 90%/99%-energy ranks match. One of 16,384 threshold-rank measurements changed by one mode: Robert Boulter, prefix 280, head 2, from 242 to 241. The old boundary singular-value ratio was 0.000100003620747, only 3.62 × 10⁻⁹ above the cutoff; the run passed the predeclared narrow boundary-crossing check. Full current spectra at this unsaved prefix were transient; its comparison is recorded in the frozen measurement validation.

Execution source: `449739a463ae0a7805756fb8e20a7fe7e6843377`. `manifest.json` records the resolved configuration, source/input/model identity, seeds, initialization rules, and job IDs. Each condition retains metadata and an ignored numerical archive. No checkpoints, raw recurrent matrices, or convolution buffers are retained. The results passed independent review before retention.

Figure export layout was stabilized after measurement, using plotting source `f31dbed3b5781231c02414f0c64f4791ebad7d7d`; all nine plotting tests passed. Numeric archives are unchanged.
