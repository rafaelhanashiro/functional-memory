# State rank

The completed experiment compares pretrained and independently initialized Llama
and Gated DeltaNet attention states. This is the single scientific README for
methods, reproduction, findings, and retained artifact locations.

**Home cluster:** ORCD / Engaging, inferred from the recorded `mit_preemptable`
L40S launches, `node2902`/`node4200` hostnames, and original
`/home/rafah/projects/functional_memory` artifact root. That remote storage has
not been live-verified during this layout migration. Checkpoint retention is
**none**. The ignored per-condition `metrics.npz` archives and successful Slurm
logs remain home-cluster artifacts; cloning Git alone does not reproduce them.
The two older `spectra.npz` archives are retained through their existing Git LFS
records. See [provenance](provenance/manifest.json) for exact hashes and storage semantics.

Reusable code and tests live in [`scaffolds/state_rank/`](../../scaffolds/state_rank/).
Exact historical input and resolved recipes are in [`configs/`](configs/).
Historical manifests preserve their original source commits and execution paths;
those old paths describe past execution, not additional retained working copies.
New launchers resolve relocated historical baselines without rewriting recipes.
Use a fresh output directory to reproduce a comparison. The scientific metadata,
CSV values, figures, and numerical archives were not recomputed during migration.

Measure the tokenwise spectrum of an attention state before output processing.
Llama uses the implicit unnormalized softmax numerator state; Gated DeltaNet
uses its native recurrent state. Threshold rank counts singular values greater
than `1e-4` times the largest. Participation ratio uses squared singular values.

Run from the repository root using its existing PyTorch environment:

```bash
uv run --frozen scaffolds/state_rank/spectrum.py --layer 8 --tokens 512
uv run --frozen scaffolds/state_rank/spectrum.py --model gated-deltanet --layer 12
uv run --frozen python -m pytest scaffolds/state_rank/tests
```

For Slurm, synchronize the environment first and use `.venv/bin/python` with
the same script arguments. Keep the successful combined log at
`logs/<experiment>/slurm-<job_id>.out`.

Retained results live under `results/state_rank/`, with the two original
pretrained collections under `historical/`. Relocation preserves numerical
archives, inputs, scientific metadata, and figures byte for byte. To regenerate plots
intentionally, pass `--replot <result-directory>`; this rewrites result metadata
and should not be used merely to verify a source-code relocation.

The original pretrained Llama and DeltaNet result folders retain their numerical
archives, inputs, metadata, and aggregate figures. Their standalone `text-1`
through `text-4` figures have been removed. Use the unified model collections
below for the current sequence/seed-averaged presentation.

## Random versus pretrained Llama

The completed six-condition comparison is retained in
[the Llama findings below](#random-versus-pretrained-llama-completed-comparison).
Use a new `--output` directory for a new run; the retained manifest is a historical
execution record and protects the original source/configuration/input identity.

[`configs/random_vs_pretrained.json`](configs/random_vs_pretrained.json) compares the pinned Llama-3.2-1B checkpoint
with five complete Hugging Face reference initializations (seeds 0–4), across all
16 layers and all eight KV heads. Each random layer and head has distinct weight
draws; a seed identifies one fixed model used for every input. RMSNorm scales use
the standard initialization of ones. These are not Meta's original initial weights.

The comparison reuses the exact four saved 512-token sequences from the retained
layer-8 experiment. It uses FP32 eager inference with TF32 disabled, then FP64
analysis of the same unnormalized numerator state. It does not measure the rank
of an attention-probability matrix or evaluate language-model perplexity.

Develop and test in a separate experiment checkout. From that checkout, using the
existing root environment (set `RANK_PROJECT_ROOT` to the retained repository):

```bash
RANK_PROJECT_ROOT=/home/rafah/projects/functional_memory
"$RANK_PROJECT_ROOT/.venv/bin/python" -m pytest scaffolds/state_rank/tests
# Commit the tested scaffold before freezing its source identity and submitting.
"$RANK_PROJECT_ROOT/.venv/bin/python" -m scaffolds.state_rank.launch_comparison \
  --project-root "$RANK_PROJECT_ROOT" --submit
```

Without `--submit`, the launcher only prepares the manifest and exact input copy.
It submits six serial one-L40S jobs with four CPUs, 32 GiB host RAM, a one-hour
limit, and requeue support. The pretrained and random-seed-0 conditions serve as
full-context preflights; successful dependencies gate all subsequent conditions.
The first condition must reproduce the retained layer-8 KV arrays and spectra,
including identical discrete ranks, before random conditions execute. No separate
pilot model or reduced-context result is substituted for the scientific outputs.

Jobs execute from the frozen clean checkout and save scientific outputs outside
it under `results/state_rank/random_vs_pretrained/`. A preempted condition restarts
from its deterministic seed; complete, checksum-verified conditions are skipped.
One successful matching log per condition is retained under
`results/logs/state_rank/random_vs_pretrained/<condition>/slurm-<job_id>.out` on
ORCD; its canonical destination, `logs/state_rank/random_vs_pretrained/`, awaits a
verified migration. New launches write to `logs/`.

Each condition retains scalar rank/norm curves at every token and normalized
spectra at prefixes 1, 2, 8, 32, 128, and 512. Metrics have axes
`[layer,input,token,head]`; saved spectra use `[layer,input,prefix,head,mode]`.
Large numerical archives are ignored by Git. No random checkpoints or raw KV
arrays are retained. Metadata records parameter fingerprints, initialization,
inputs, pinned model configuration, package versions, and source provenance.

Regenerate the corresponding section of this README and four figure groups with:

```bash
"$RANK_PROJECT_ROOT/.venv/bin/python" -m scaffolds.state_rank.plot_comparison \
  --results "$RANK_PROJECT_ROOT/results/state_rank/random_vs_pretrained"
```

Plots aggregate metrics computed independently for each input/head, showing each
random seed's mean plus the random mean/range and the pretrained mean. Seeds are
the independent random-model replicates; heads are not paired semantically or
treated as independent seed replicates. Norm/energy plots help interpret changes
in exponential-kernel scale. Keep the experiment branch separate until the
completed results have been reviewed, then retain outputs, clean up temporary
files, merge, and remove the experiment checkout and merged branch.

## Random versus pretrained Gated DeltaNet

The completed six-condition comparison and its depth-wise findings are retained in
[the DeltaNet findings below](#gated-deltanet-random-versus-pretrained-state-rank).

[`configs/random_vs_pretrained_deltanet.json`](configs/random_vs_pretrained_deltanet.json) applies the same design to the pinned
`m-a-p/1.3B-100B-GatedDeltaNet-pure` model: one pretrained condition and five
independent complete initializations (seeds 0–4), all 24 layers, all eight heads,
and the exact four saved 512-token inputs from the historical DeltaNet experiment.
These are the same source articles as the Llama diagnostic, encoded with the
DeltaNet tokenizer. Token IDs are identical within the DeltaNet comparison.

Random models use the pinned FLA reference initialization. Linear, embedding,
and convolution weights use its normal initializer; normalization scales start
at one. The native decay and time-step parameters retain their specialized
initialization distributions. Each layer has distinct parameters, and each seed
identifies one fixed model used for all four inputs. These are fresh reference
initializations, not the pretrained model's original saved initialization.

The measured object is the native `[value,key]` recurrent state before output
normalization, gating, and projection. Full short-convolution and recurrent
caches propagate through every token. One model forward per token updates all
layers; independent direct FP64 SVDs measure every head at every token. State
evolution and model parameters are FP32, using FLA's fused recurrent kernel with
TF32 disabled. The 256-wide native states differ from Llama's implicit softmax
state, so absolute rank values should be interpreted within each architecture.

From a committed experiment checkout, launch with:

```bash
RANK_PROJECT_ROOT=/home/rafah/projects/functional_memory
"$RANK_PROJECT_ROOT/.venv/bin/python" -m scaffolds.state_rank.launch_comparison \
  --model gated-deltanet --project-root "$RANK_PROJECT_ROOT" --submit
```

The six serial jobs each request one L40S, eight CPUs, 48 GiB host RAM, and a
two-hour limit with requeue enabled. Four bounded SVD workers use single-threaded
BLAS. The pretrained condition checks the historical layer-12 spectra and rank
metrics; every condition independently checks full-prefix versus tokenwise
recurrent caches at prefixes 1, 2, 8, 32, 128, and 512. Successful pretrained and
seed-0 conditions gate the remaining conditions.

Results are written outside the checkout under
`results/state_rank/random_vs_pretrained_deltanet/`, with one matching successful
log per condition under `results/logs/state_rank/random_vs_pretrained_deltanet/`
on ORCD, likewise awaiting migration to `logs/`.
The same plotting command above accepts this results directory. Rank curves,
selected spectra, state energies, resolved configurations, and provenance are
retained; no checkpoints, raw recurrent states, or convolution buffers are
saved. Key/value vector norms are not collected for this diagnostic. Use a fresh
`--output` directory to rerun a retained comparison without changing its history.

## Unified analysis of both architectures and weight conditions

The [unified analysis below](#unified-state-rank-comparison) has
one collection for [Llama](unified_comparison/llama/heads.png)
and one for [Gated DeltaNet](unified_comparison/gated-deltanet/heads.png),
each placing pretrained and random models side by side. It reads the original
archives without rewriting their measurements, metadata, or logs.

- At the original selected layers (Llama 8, DeltaNet 12), each head has its own
  token-position curve, averaged over sequences and additionally over seeds
  for random models. Pretrained shading shows the sequence minimum/maximum;
  random shading shows the range of sequence-averaged seed curves.
- A second token-position view has one curve per layer, averaged over sequences
  and heads within each model, and additionally over random seeds. A shared
  colorbar identifies the layer.
- Terminal rank versus depth averages sequences and heads at token 512; random
  mean/range then summarizes independent model seeds. This shared overview
  retains its faint individual-seed curves. No separate sequence/seed figures
  or additional normalized-depth figure are retained in this collection.

All four rank metrics are included. These are averages of individually measured
state ranks, not ranks of averaged states or spectra. Random per-head means
average the same numerical head index across seeds; they describe the
initialization ensemble, not matched learned features. Inputs are the same four source
articles with model-native 512-token prefixes, not identical text spans across
architectures. The architectures also have different state definitions and
rank ceilings (64 and 256).

Run CPU postprocessing with the root environment and a fresh output directory:

```bash
uv run --frozen python -m scaffolds.state_rank.unify_analysis \
  --output /path/to/new-results
```

The default configuration is `results/state_rank/configs/unified_rank_analysis.json`; edit a copy
and pass `--config` to select other saved layers. `--project-root` locates the
retained source archives when executing from a separate checkout. The generator
requires all planned conditions, verifies archive/input checksums and matched
scientific conventions, and records source hashes before and after generation.
Outputs include five PNG/PDF figures, per-head and per-layer terminal CSVs, detailed
comparison statistics, resolved configuration, and a provenance manifest. The
CSV summaries retain individual seeds for auditing; the figures show seed
averages. The original per-token numerical archives remain the source of truth.

<!-- state-rank:llama-findings:start -->
## Random versus pretrained Llama: completed comparison

All six conditions completed successfully: pinned pretrained Llama-3.2-1B and five complete, independently initialized Hugging Face reference models (seeds 0–4). Each condition measured all 16 layers and eight distinct KV heads on the exact same four saved 512-token WikiText inputs. Norm scales use the standard initialization of ones; projection weights are distinct across layers and KV heads.

### Layer 8 at 512 tokens

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

### Figures and reproduction

- [Rank through the input, layer 8](random_vs_pretrained/token_metrics.png) · [PDF](random_vs_pretrained/token_metrics.pdf)
- [Rank across all 16 layers](random_vs_pretrained/terminal_rank_depth.png) · [PDF](random_vs_pretrained/terminal_rank_depth.pdf)
- [Spectra at selected prefixes](random_vs_pretrained/normalized_spectra.png) · [PDF](random_vs_pretrained/normalized_spectra.pdf)
- [Norm and energy comparisons](random_vs_pretrained/norm_energy_depth.png) · [PDF](random_vs_pretrained/norm_energy_depth.pdf)
- [Per-condition/per-layer summary](random_vs_pretrained/summary.csv)

The pretrained layer-8 K/V arrays and all spectral metrics reproduce the retained historical experiment exactly for every saved input. Offline validation: 82 tests passed, 2 unrelated CUDA-only tests skipped. All six full-context GPU conditions passed; archive checksums, distinct parameter fingerprints, causal rank bounds, and saved-spectrum/metric consistency were checked after completion.

Execution source: `c5308deb23ed5e30b77a3d3491d482032322a235`. Configuration, input hashes, source identity, and job IDs are in `manifest.json`; each condition has detailed metadata and an ignored numerical archive. No random checkpoints or raw K/V arrays are retained. All six results passed independent review of archive integrity, metrics, summaries, provenance, and matching Slurm logs.
<!-- state-rank:llama-findings:end -->

<!-- state-rank:deltanet-findings:start -->
## Gated DeltaNet: random versus pretrained state rank

All six conditions completed: the pinned pretrained model and five independent FLA reference initializations (seeds 0–4). The comparison covers all 24 layers and all eight heads on the exact four saved 512-token inputs, including BOS. No training was performed on the random controls.

The measured object is the native 256 × 256 recurrent state before output normalization, gating, and projection. State evolution uses FP32 fused recurrence with TF32 disabled; direct SVD and analysis use FP64. Full convolution and recurrent caches are preserved during inference.

### Terminal results at layer 12

Each number first averages metrics computed separately for the four inputs and eight heads. Random columns then summarize the five seed means; the seed range is descriptive, not a confidence interval. Threshold rank uses the strict cutoff σ > 10⁻⁴ σ₁.

| Metric | Pretrained | Random mean | Random seed range |
|---|---:|---:|---:|
| Participation ratio | 3.784 | 11.512 | 8.418–15.897 |
| Threshold rank | 74.875 | 111.106 | 81.188–135.812 |
| Modes for 90% energy | 8.312 | 16.275 | 11.438–22.781 |
| Modes for 99% energy | 21.250 | 33.737 | 24.125–46.562 |

### What changes across depth

Pretrained participation ratio is below the five-seed random mean in all 24 layers, and below every individual seed mean in 22 layers. This indicates stronger concentration by the participation-ratio measure on these inputs.

Numerical rank has a different pattern: pretrained threshold rank exceeds the random mean in nine layers (7, 10, 13, 16, 17, 18, 19, 20, and 22). At layer 20, pretrained participation ratio is **8.92 versus 16.59** random, while threshold rank is **228.13 versus 126.84** and the 99%-energy rank is **71.41 versus 45.59**. Thus stronger dominant modes can coexist with a larger weak-mode tail. A blanket claim that pretraining reduces every notion of rank is not supported.

These are fresh reference initializations, not the pretrained model’s original starting weights or a tracked training trajectory. The comparison is descriptive and limited to four fixed texts. It does not isolate effects of learned decay, update gates, projections, or normalization, and state rank does not establish language-model quality. DeltaNet’s finite recurrent state and Llama’s implicit softmax state have different definitions and widths; interpret absolute ranks within each architecture.

### Figures and reproduction

- [Rank through the input, layer 12](random_vs_pretrained_deltanet/token_metrics.png) · [PDF](random_vs_pretrained_deltanet/token_metrics.pdf)
- [Terminal rank across all 24 layers](random_vs_pretrained_deltanet/terminal_rank_depth.png) · [PDF](random_vs_pretrained_deltanet/terminal_rank_depth.pdf)
- [Relative spectra at selected prefixes](random_vs_pretrained_deltanet/normalized_spectra.png) · [PDF](random_vs_pretrained_deltanet/normalized_spectra.pdf)
- [State energy across depth](random_vs_pretrained_deltanet/norm_energy_depth.png) · [PDF](random_vs_pretrained_deltanet/norm_energy_depth.pdf)
- [All 144 condition/layer summaries](random_vs_pretrained_deltanet/summary.csv)
- [Validation and provenance audit](random_vs_pretrained_deltanet/validation.json)

Offline tests: **134 passed**, with three CUDA tests skipped on the login node; all **three native GPU tests then passed** before the pretrained measurement. All six Slurm jobs completed with exit code 0. The independent audit verified archive checksums, unique parameter fingerprints, matching inputs and axes, causal rank bounds, saved-spectrum reconstruction of rank metrics, all 144 CSV rows, and one matching successful Slurm log per condition. Maximum full-prefix versus tokenwise state error was 1.99 × 10⁻⁵, below the predeclared 2 × 10⁻⁴ tolerance.

The historical layer-12 baseline agrees to a maximum normalized-spectrum absolute error of 2.70 × 10⁻⁷. All terminal threshold ranks and all 90%/99%-energy ranks match. One of 16,384 threshold-rank measurements changed by one mode: Robert Boulter, prefix 280, head 2, from 242 to 241. The old boundary singular-value ratio was 0.000100003620747, only 3.62 × 10⁻⁹ above the cutoff; the run passed the predeclared narrow boundary-crossing check. Full current spectra at this unsaved prefix were transient; its comparison is recorded in the frozen measurement validation.

Execution source: `449739a463ae0a7805756fb8e20a7fe7e6843377`. `manifest.json` records the resolved configuration, source/input/model identity, seeds, initialization rules, and job IDs. Each condition retains metadata and an ignored numerical archive. No checkpoints, raw recurrent matrices, or convolution buffers are retained. The results passed independent review before retention.

Figure export layout was stabilized after measurement, using plotting source `f31dbed3b5781231c02414f0c64f4791ebad7d7d`; all nine plotting tests passed. Numeric archives are unchanged.
<!-- state-rank:deltanet-findings:end -->

<!-- state-rank:unified-comparison:start -->
## Unified state-rank comparison

The same analysis is applied to pretrained and randomly initialized Llama and Gated DeltaNet. All curves reuse the retained measurements; no inference or training was rerun.

### How to read the figures

- **Per head, across tokens:** fix a layer; each colored curve preserves a head index and averages over sequences. The random panel additionally averages the five seeds at each head index. Heads remain separate.
- **Per layer, across tokens:** each colored curve is one layer's rank averaged over sequences and heads, and additionally over seeds for random models, at each token position.
- **Across depth:** at the final token, average over sequences and heads within each model. Random means and ranges then summarize the five independently initialized models.
- Each architecture has one collection with pretrained and random panels side by side. No individual-sequence or individual-seed figures are retained in the collection.
- Random per-head curves average the same numerical head index across independently initialized models. This describes an initialization ensemble, not matched learned features across seeds or with the pretrained model.
- In per-head panels, pretrained shading is the sequence min/max; random shading is the min/max of the five sequence-averaged seed curves. Layer panels show means only. The retained depth overview also shows faint seed curves.
- Compute rank separately for each state before averaging. No rank is computed from an averaged state or spectrum. Bands are descriptive ranges, not confidence intervals.

All layer and head indices are zero-based. Token positions count the initial BOS token.

| Model | Layers | Heads | State rank ceiling | Selected layer |
|---|---:|---:|---:|---:|
| Llama | 16 | 8 | 64 | 8 |
| Gated DeltaNet | 24 | 8 | 256 | 12 |

[Terminal ranks versus layer](unified_comparison/depth_counts.png) · [PDF](unified_comparison/depth_counts.pdf)

Llama measures its implicit softmax numerator (eight KV heads, width 64); Gated DeltaNet measures its native 256 × 256 recurrent state before output processing.

The same four source articles are used in both architectures, with model-native 512-token prefixes. The text spans and token IDs differ across architectures; pretrained and random conditions within an architecture use exactly the same tokens.

### Model collections

Each figure has pretrained and random columns, with four rank measures and shared vertical scales. PNG links below also have sibling PDF files.

| Model collection | Per head | Per layer |
|---|---|---|
| Llama | [Layer 8](unified_comparison/llama/heads.png) | [All layers](unified_comparison/llama/layers.png) |
| Gated DeltaNet | [Layer 12](unified_comparison/gated-deltanet/heads.png) | [All layers](unified_comparison/gated-deltanet/layers.png) |

### Terminal comparisons

The table compares pretrained ranks with the mean of the five random models, after averaging over sequences and heads at token 512. It describes aggregate layers, not every individual head.

| Model | Metric | Pretrained lower | Equal | Pretrained higher |
|---|---|---:|---:|---:|
| Llama | Participation ratio | 16 | 0 | 0 |
| Llama | Threshold rank | 16 | 0 | 0 |
| Llama | Modes for 90% energy | 16 | 0 | 0 |
| Llama | Modes for 99% energy | 16 | 0 | 0 |
| Gated DeltaNet | Participation ratio | 24 | 0 | 0 |
| Gated DeltaNet | Threshold rank | 15 | 0 | 9 |
| Gated DeltaNet | Modes for 90% energy | 21 | 0 | 3 |
| Gated DeltaNet | Modes for 99% energy | 18 | 0 | 6 |

Llama's pretrained layer averages are lower for all four rank measures at every layer. DeltaNet's participation ratio is lower at every layer, but its threshold and energy-rank comparisons are mixed. Thus the result depends on the rank definition; it does not support a blanket claim that pretraining reduces all ranks.

These are cross-sectional comparisons with fresh reference initializations, not the actual initial checkpoints of the pretrained models or a measured trajectory through training.

Participation ratio is `(sum(sigma^2))^2 / sum(sigma^4)`. Threshold rank counts singular values strictly above `1e-4 * sigma_max`. Energy ranks count modes needed for 90% and 99% of squared singular-value energy.

### Data and provenance

- [Per-head terminal summaries](unified_comparison/per_head_terminal.csv): sequence mean/min/max at the selected layer; each seed remains separate.
- [Per-layer terminal summaries](unified_comparison/depth_summary.csv): sequence/head means and fractions of state width for each model and seed.
- [Detailed comparisons](unified_comparison/comparison_statistics.json), [resolved configuration](configs/unified_comparison.resolved.json), [provenance](unified_comparison/provenance.json), and [postprocessing validation](unified_comparison/validation.json).
- Full token curves remain in the original immutable `metrics.npz` archives; no duplicate archives or checkpoints were created.
- The original successful Slurm logs are recorded in provenance. This CPU postprocessing required no Slurm job.

Reproduce from the repository root with a fresh output directory:

```bash
uv run --frozen python -m scaffolds.state_rank.unify_analysis --output /path/to/new-results
```
<!-- state-rank:unified-comparison:end -->

<!-- state-rank:aggregation-conventions:start -->
## Detailed aggregation conventions

Participation ratio is (sum(sigma²))² / sum(sigma⁴). The 90% and 99% energy ranks count the minimum number of descending modes whose squared singular values reach the respective energy fraction.

Rank metrics are computed for each input/head state first, then averaged equally over inputs and heads. We never compute a rank from an averaged spectrum. Each faint random curve is one seed's input/head aggregate; the solid random curve is the equal-weight mean over completed seeds, and shading is the minimum–maximum range of those seed aggregates. It is a descriptive range, not a confidence interval. Inputs and heads are not treated as independent experimental replications, and corresponding pretrained/random heads are not assumed to be matched.

For each saved input/head state, relative singular values are computed as sqrt(lambda_i / lambda_1) from the normalized eigenvalues, and only then averaged over inputs and heads. A zero state contributes zeros. Spectral values below 10⁻⁶ are clipped only for logarithmic display; saved measurements and CSV values are unchanged.

Key/value norm panels show sqrt(mean(||vector||₂²)) across inputs and heads at the terminal token. These are RMS vector norms, not per-component RMS and not averages across all token positions. The energy panel averages log(trace(B)) over input/head states, then converts to base 10; it is not log of the arithmetic mean energy. Zero state energy is represented by negative infinity.

This is a descriptive diagnostic on a fixed, small set of inputs. The pretrained model is one checkpoint, while random weights are independent initialization seeds of the same architecture; seed ranges do not measure pretrained uncertainty. Matching the inputs and architecture isolates this comparison but does not identify a causal contribution of an individual head or establish generalization to other texts, checkpoints, model sizes, or training recipes. No language-model quality claim follows from state rank alone.


For Gated DeltaNet, key/value vector norms were not collected. Its state-energy panel uses the same mean-log convention.
<!-- state-rank:aggregation-conventions:end -->
