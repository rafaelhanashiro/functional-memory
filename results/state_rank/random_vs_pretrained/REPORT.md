# Random versus pretrained Llama: completed comparison

All six conditions completed successfully: pinned pretrained Llama-3.2-1B and five complete, independently initialized Hugging Face reference models (seeds 0–4). Each condition measured all 16 layers and eight distinct KV heads on the exact same four saved 512-token WikiText inputs. Norm scales use the standard initialization of ones; projection weights are distinct across layers and KV heads.

## Layer 8 at 512 tokens

Metrics are computed per input/head and then averaged. The random mean weights the five seed means equally; the range is across seed means, not a confidence interval.

| Metric | Pretrained | Random mean | Random seed range |
|---|---:|---:|---:|
| Participation ratio | 1.388 | 4.765 | 4.208–5.322 |
| Threshold rank | 29.750 | 64.000 | 64.000–64.000 |
| Modes for 90% energy | 1.656 | 33.219 | 32.625–34.250 |
| Modes for 99% energy | 3.375 | 58.794 | 58.594–59.062 |
| Key L2 RMS at final token | 23.065 | 7.200 | 7.111–7.336 |
| Value L2 RMS at final token | 2.186 | 7.186 | 7.123–7.266 |

The pretrained state spectrum is substantially more concentrated than every random control at this layer on these inputs. The random controls retain all 64 modes above the relative singular-value threshold, whereas the pretrained threshold rank averages 29.75. This threshold rank is a numerical rank statistic, not an assertion that the trained state has lower exact algebraic rank.

The random networks also show concentration that changes with depth without any training. See the depth figure rather than interpreting random weights as an isotropic or flat-spectrum null model.

Key and value norms differ substantially between conditions. Because the diagnostic uses an exponential key–key kernel, the result combines changes in magnitude and geometry. It does not isolate which caused the spectral difference. These are architecture-matched reference initializations, not Meta's original initialization or a recorded training trajectory, and state rank alone does not measure language-model quality.

## Figures and reproduction

- [Rank through the input, layer 8](token_metrics.png) · [PDF](token_metrics.pdf)
- [Rank across all 16 layers](terminal_rank_depth.png) · [PDF](terminal_rank_depth.pdf)
- [Spectra at selected prefixes](normalized_spectra.png) · [PDF](normalized_spectra.pdf)
- [Norm and energy comparisons](norm_energy_depth.png) · [PDF](norm_energy_depth.pdf)
- [Per-condition/per-layer summary](summary.csv)

The pretrained layer-8 K/V arrays and all spectral metrics reproduce the retained historical experiment exactly for every saved input. Offline validation: 82 tests passed, 2 unrelated CUDA-only tests skipped. All six full-context GPU conditions passed; archive checksums, distinct parameter fingerprints, causal rank bounds, and saved-spectrum/metric consistency were checked after completion.

Execution source: `c5308deb23ed5e30b77a3d3491d482032322a235`. Configuration, input hashes, source identity, and job IDs are in `manifest.json`; each condition has detailed metadata and an ignored numerical archive. No random checkpoints or raw K/V arrays are retained. All six results passed independent review of archive integrity, metrics, summaries, provenance, and matching Slurm logs.
