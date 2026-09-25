# State rank

How concentrated is an attention state's spectrum in a pretrained model, compared
with independently initialized models of the same architecture? This experiment
measures the tokenwise singular spectrum of Llama-3.2-1B's implicit softmax
numerator state and Gated DeltaNet's native recurrent state, for each pinned
pretrained checkpoint and five complete reference initializations (seeds 0–4).
This is the single scientific README for methods, reproduction, findings, and
retained artifact locations.

## Record

- **Home cluster:** Kennel, where the comparisons ran on 2026-09-24. An earlier
  ORCD / Engaging execution (L40S) is superseded; it is summarized under
  `superseded_execution` in [provenance](provenance/manifest.json), and its agreement
  with the Kennel runs is described [below](#kennel-execution-and-agreement-with-orcd).
- **Artifact root:** `/data/cl/u/rafah/home/projects/functional-memory` on Kennel.
  Git holds results only. Each run's numerical archive (`metrics.npz`, or
  `spectra.npz` for the two pretrained-only collections) stays beside its
  committed `metadata.json` in `runs/<run>/` of the primary checkout, ignored by
  Git, with its SHA256 in provenance. A clone therefore carries metadata,
  summaries, and figures, but not the archives.
- **Logs:** one successful log per condition at
  `logs/<comparison>/<condition>/slurm-<job_id>.out` on Kennel (ignored).
- **Checkpoints:** none. No random weights, raw K/V arrays, recurrent states, or
  convolution buffers are retained.

| Path | Contents |
|---|---|
| `runs/random_vs_pretrained-<condition>/` | Llama conditions: `metadata.json` (+ ignored `metrics.npz`) |
| `runs/random_vs_pretrained_deltanet-<condition>/` | Gated DeltaNet conditions, same layout |
| `runs/llama-3.2-1b-l8-kvall-t512/`, `runs/gated-deltanet-1.3b-l12-hall-t512/` | Original pretrained-only collections: fixed `inputs.json`, `metadata.json` (+ ignored `spectra.npz`); the comparisons' inputs and reproduction baselines |
| `summary.csv` | 240 rows (model × condition × layer): terminal rank metrics and fractions, plus terminal state energy and Llama key/value norms |
| `figures/` | Per-comparison, unified, and pretrained-collection figures (PNG and PDF) |
| `configs/` | Exact recipes: [`random_vs_pretrained.json`](configs/random_vs_pretrained.json), [`random_vs_pretrained_deltanet.json`](configs/random_vs_pretrained_deltanet.json), [`unified_rank_analysis.json`](configs/unified_rank_analysis.json) |
| `provenance/manifest.json` | Execution manifests, resolved configurations, jobs, hardware, acceptance, archive hashes, and the superseded ORCD execution |

Reusable code and tests live in [`scaffolds/state_rank/`](../../scaffolds/state_rank/); the launcher, record
tooling, and their tests are in [`adapter/`](adapter/).

## Method

Measure the tokenwise spectrum of an attention state before output processing.
Llama uses the implicit unnormalized softmax numerator state (post-RoPE keys);
Gated DeltaNet uses its native `[value,key]` recurrent state. Threshold rank
counts singular values strictly greater than `1e-4` times the largest.
Participation ratio uses squared singular values; energy ranks count the modes
needed for 90% and 99% of squared singular-value energy.

**Llama.** [`configs/random_vs_pretrained.json`](configs/random_vs_pretrained.json)
compares pinned Llama-3.2-1B with five complete Hugging Face reference
initializations across all 16 layers and all eight KV heads. Each random layer
and head has distinct weight draws; a seed identifies one fixed model used for
every input. RMSNorm scales start at one. These are not Meta's original initial
weights. The comparison reuses the exact four saved 512-token WikiText inputs of
the pretrained layer-8 collection, runs FP32 eager inference with TF32 disabled,
and analyzes the same unnormalized numerator state in FP64. It does not measure
the rank of an attention-probability matrix or evaluate perplexity.

**Gated DeltaNet.** [`configs/random_vs_pretrained_deltanet.json`](configs/random_vs_pretrained_deltanet.json)
applies the same design to pinned `m-a-p/1.3B-100B-GatedDeltaNet-pure`: all 24
layers, all eight heads, and the four saved 512-token inputs of the pretrained
layer-12 collection (the same source articles, encoded with the DeltaNet
tokenizer). Random models use the pinned FLA reference initialization: normal
Linear, embedding, and convolution weights, unit normalization scales, and the
specialized decay (`A_log`) and time-step (`dt_bias`) distributions. The measured
object is the native 256 × 256 state before output normalization, gating, and
projection. Full short-convolution and recurrent caches propagate through every
token; one forward per token updates all layers, and independent direct FP64 SVDs
measure every head at every token. State evolution is FP32 with FLA's fused
recurrent kernel and TF32 disabled. The 256-wide native states differ from
Llama's implicit softmax state, so interpret absolute ranks within each
architecture.

**Acceptance.** Each pretrained condition must reproduce its pretrained-only
collection before random conditions run: identical discrete ranks and
normalized spectra within tolerance (Llama also checks K/V arrays; DeltaNet also
checks log trace and uses a narrow boundary-crossing rule). Every DeltaNet
condition independently checks full-prefix versus tokenwise recurrent caches at
prefixes 1, 2, 8, 32, 128, and 512. The pretrained and seed-0 conditions gate the
remaining four. Each condition retains scalar rank/norm curves at every token
(`[layer,input,token,head]`) and normalized spectra at those prefixes
(`[layer,input,prefix,head,mode]`), plus metadata recording parameter
fingerprints, initialization, inputs, pinned model configuration, package
versions, and source provenance.

The unified analysis reads both comparisons without rewriting them. At the
selected layers (Llama 8, DeltaNet 12) each head keeps its own token curve,
averaged over sequences and, for random models, over seeds; a second view has one
curve per layer; terminal rank versus depth averages sequences and heads at token
512. These are averages of individually measured state ranks, never ranks of
averaged states or spectra. Random per-head means average the same numerical head
index across seeds; they describe the initialization ensemble, not matched
learned features.

## Reproduction

Regenerate every derived output (figures, `summary.csv`, and the generated section)
from `runs/` alone; this needs the ignored archives, so run it in the primary
checkout on Kennel:

```bash
uv sync --project scaffolds/state_rank --locked --link-mode hardlink
PYTHONPATH=. scaffolds/state_rank/.venv/bin/python -m experiments.state_rank.adapter.analysis.record regenerate
```

To rerun the comparisons, work from a new worktree and branch. Both models must be
present in the Hugging Face cache at their pinned revisions (Llama is gated:
authenticate with `hf auth login` and keep the token file at mode 600); jobs set
`HF_HUB_OFFLINE=1`. Each launcher submits six serial one-H100 jobs on `standard`
after checking that the frozen execution set matches the launch commit.

```bash
P=/data/cl/u/rafah/home/projects/functional-memory                  # primary checkout, holds the archives
git -C "$P" worktree add -b rerun/state_rank "$P/worktrees/state_rank" main
cd "$P/worktrees/state_rank"
uv sync --project scaffolds/state_rank --locked --link-mode hardlink
PY=scaffolds/state_rank/.venv/bin/python
srun -p standard --gres=gpu:1 -w dogo env PYTHONPATH=. $PY -m pytest -q \
  -c scaffolds/state_rank/pyproject.toml scaffolds/state_rank/tests experiments/state_rank/adapter/tests
W=.cache/state_rank/<rerun>/experiments/state_rank                  # execution workspace
PYTHONPATH=. $PY -m experiments.state_rank.adapter.execution.launch_comparison \
  --project-root "$P" --output "$W/random_vs_pretrained" --submit
PYTHONPATH=. $PY -m experiments.state_rank.adapter.execution.launch_comparison --model gated-deltanet \
  --project-root "$P" --output "$W/random_vs_pretrained_deltanet" --submit
# After all twelve conditions complete (CPU postprocessing):
PYTHONPATH=. $PY -m scaffolds.state_rank.unify_analysis --config experiments/state_rank/configs/unified_rank_analysis.json \
  --project-root .cache/state_rank/<rerun> --output "$W/unified_comparison"
PYTHONPATH=. $PY -m experiments.state_rank.adapter.analysis.record retain "$W"
```

`--project-root` locates the pretrained-only collections' archives under
`experiments/state_rank/runs/`. Without `--submit` the launcher only writes the
workspace manifest and exact input copy. A preempted condition restarts from its
deterministic seed; complete, checksum-verified conditions are skipped. `record
retain` copies each verified condition into `runs/`, figures into `figures/`, the
merged `summary.csv`, the generated README section below, and the execution
records into provenance.

The two pretrained-only collections were produced by the single-model spectrum
tool; a fresh measurement writes under `.cache/state_rank/spectrum/`:

```bash
PYTHONPATH=. $PY scaffolds/state_rank/spectrum.py --layer 8 --tokens 512
PYTHONPATH=. $PY scaffolds/state_rank/spectrum.py --model gated-deltanet --layer 12
```

## Findings

### Llama: random versus pretrained

All six conditions completed. At layer 8 and token 512, metrics are computed per
input/head and then averaged; the random mean weights the five seed means equally,
and the range is across seed means, not a confidence interval.

| Metric | Pretrained | Random mean | Random seed range |
|---|---:|---:|---:|
| Participation ratio | 1.388 | 4.765 | 4.208–5.322 |
| Threshold rank | 29.750 | 64.000 | 64.000–64.000 |
| Modes for 90% energy | 1.656 | 33.219 | 32.625–34.250 |
| Modes for 99% energy | 3.375 | 58.794 | 58.594–59.062 |
| Key L2 RMS at final token | 23.065 | 7.200 | 7.111–7.336 |
| Value L2 RMS at final token | 2.186 | 7.186 | 7.123–7.266 |

The pretrained state spectrum is substantially more concentrated than every
random control at this layer on these inputs. The random controls retain all 64
modes above the relative singular-value threshold, whereas the pretrained
threshold rank averages 29.75. This is a numerical rank statistic, not an
assertion that the trained state has lower exact algebraic rank.

The random networks also show concentration that changes with depth without any
training; see the depth figure rather than treating random weights as an
isotropic or flat-spectrum null model. Key and value norms differ substantially
between conditions. Because the diagnostic uses an exponential key–key kernel,
the result combines changes in magnitude and geometry and does not isolate which
caused the spectral difference.

- [Rank through the input, layer 8](figures/random_vs_pretrained-token_metrics.png) · [PDF](figures/random_vs_pretrained-token_metrics.pdf)
- [Rank across all 16 layers](figures/random_vs_pretrained-terminal_rank_depth.png) · [PDF](figures/random_vs_pretrained-terminal_rank_depth.pdf)
- [Spectra at selected prefixes](figures/random_vs_pretrained-normalized_spectra.png) · [PDF](figures/random_vs_pretrained-normalized_spectra.pdf)
- [Norm and energy comparisons](figures/random_vs_pretrained-norm_energy_depth.png) · [PDF](figures/random_vs_pretrained-norm_energy_depth.pdf)
- [Pretrained-only layer-8 collection](figures/llama-3.2-1b-l8-kvall-t512-effective_rank.png) · [PDF](figures/llama-3.2-1b-l8-kvall-t512-effective_rank.pdf)

### Gated DeltaNet: random versus pretrained

All six conditions completed; no training was performed on the random controls.
At layer 12 and token 512, each number first averages the four inputs and eight
heads; random columns summarize the five seed means.

| Metric | Pretrained | Random mean | Random seed range |
|---|---:|---:|---:|
| Participation ratio | 3.784 | 11.512 | 8.418–15.897 |
| Threshold rank | 74.875 | 111.106 | 81.188–135.812 |
| Modes for 90% energy | 8.312 | 16.275 | 11.438–22.781 |
| Modes for 99% energy | 21.250 | 33.737 | 24.125–46.562 |

Pretrained participation ratio is below the five-seed random mean in all 24
layers and below every individual seed mean in 22. This indicates stronger
concentration by the participation-ratio measure on these inputs.

Numerical rank has a different pattern: pretrained threshold rank exceeds the
random mean in nine layers (7, 10, 13, 16, 17, 18, 19, 20, and 22). At layer 20,
pretrained participation ratio is **8.92 versus 16.59** random, while threshold
rank is **228.13 versus 126.84** and the 99%-energy rank is **71.41 versus
45.59**. Stronger dominant modes can therefore coexist with a larger weak-mode
tail; a blanket claim that pretraining reduces every notion of rank is not
supported. Key/value vector norms are not collected for this diagnostic.

- [Rank through the input, layer 12](figures/random_vs_pretrained_deltanet-token_metrics.png) · [PDF](figures/random_vs_pretrained_deltanet-token_metrics.pdf)
- [Terminal rank across all 24 layers](figures/random_vs_pretrained_deltanet-terminal_rank_depth.png) · [PDF](figures/random_vs_pretrained_deltanet-terminal_rank_depth.pdf)
- [Relative spectra at selected prefixes](figures/random_vs_pretrained_deltanet-normalized_spectra.png) · [PDF](figures/random_vs_pretrained_deltanet-normalized_spectra.pdf)
- [State energy across depth](figures/random_vs_pretrained_deltanet-norm_energy_depth.png) · [PDF](figures/random_vs_pretrained_deltanet-norm_energy_depth.pdf)
- [Pretrained-only layer-12 collection](figures/gated-deltanet-1.3b-l12-hall-t512-effective_rank.png) · [PDF](figures/gated-deltanet-1.3b-l12-hall-t512-effective_rank.pdf)

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

[Terminal ranks versus layer](figures/unified_comparison-depth_counts.png) · [PDF](figures/unified_comparison-depth_counts.pdf)

Llama measures its implicit softmax numerator (eight KV heads, width 64); Gated DeltaNet measures its native 256 × 256 recurrent state before output processing.

The same four source articles are used in both architectures, with model-native 512-token prefixes. The text spans and token IDs differ across architectures; pretrained and random conditions within an architecture use exactly the same tokens.

### Model collections

Each figure has pretrained and random columns, with four rank measures and shared vertical scales. PNG links below also have sibling PDF files.

| Model collection | Per head | Per layer |
|---|---|---|
| Llama | [Layer 8](figures/unified_comparison-llama-heads.png) | [All layers](figures/unified_comparison-llama-layers.png) |
| Gated DeltaNet | [Layer 12](figures/unified_comparison-gated-deltanet-heads.png) | [All layers](figures/unified_comparison-gated-deltanet-layers.png) |

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

- Per-head terminal summaries: sequence mean/min/max at the selected layer; each seed remains separate.
- [Per-layer terminal summaries](summary.csv): sequence/head means and fractions of state width for each model and seed.
- Detailed comparisons, [resolved configuration](provenance/manifest.json), [provenance](provenance/manifest.json), and [postprocessing validation](provenance/manifest.json).
- Full token curves remain in the comparison runs' `metrics.npz` archives; no duplicate archives or checkpoints were created.
- The comparisons' successful Slurm logs are recorded in provenance. This CPU postprocessing required no Slurm job.

Reproduce from the repository root with a fresh output directory:

```bash
PYTHONPATH=. scaffolds/state_rank/.venv/bin/python -m experiments.state_rank.adapter.analysis.record regenerate
```
<!-- state-rank:unified-comparison:end -->

## Kennel execution and agreement with ORCD

All twelve conditions ran on one H100 each on `dogo` (Llama from source
`279fdd6`, DeltaNet from `5424ec8`); the unified analysis ran as CPU job 201750
from `279fdd6`. Before launch the full state-rank suite passed on an H100 (173
tests), and the three native DeltaNet GPU tests passed inside the pretrained
DeltaNet job. Jobs, attempts, hardware, and acceptance values are in
[provenance](provenance/manifest.json).

- **Baseline reproduction.** Llama reproduces the pretrained-only collection with
  zero discrete-rank mismatches, K/V relative error ≤ 2.6 × 10⁻⁶, and normalized
  spectra within 2.06 × 10⁻⁵. That collection was measured on L40S; the first
  Kennel attempt (job 201736) stopped at the original 2 × 10⁻⁵ spectrum tolerance,
  which was then relaxed to 5 × 10⁻⁵ for Llama only (`279fdd6`, recorded as
  `tolerance_amendment`). DeltaNet reproduces its collection within
  2.98 × 10⁻⁷ with zero rank mismatches at the unchanged 2 × 10⁻⁵ tolerance;
  maximum full-prefix versus tokenwise cache error is 1.93 × 10⁻⁵ (tolerance
  2 × 10⁻⁴).
- **Agreement with ORCD.** Llama: all threshold and energy ranks identical in 96
  condition/layer rows, continuous metrics within 3.7 × 10⁻⁶ relative, and
  identical parameter fingerprints. DeltaNet: energy ranks identical in all 144
  rows and threshold rank in 143; random seed 1 at layer 23 moves from 78.469 to
  78.5 because one of 32 input/head states gains a mode at the 10⁻⁴ σ₁ cutoff.
  Continuous metrics agree within 5.6 × 10⁻⁷. Every figure-level conclusion and
  terminal-comparison count is unchanged.
- **Random DeltaNet fingerprints.** The five random DeltaNet parameter
  fingerprints differ from ORCD although the initialization code, lock, and seeds
  are identical. The cause is CPU-vendor-dependent last-bit rounding of
  `log`/`exp`/`expm1` in FLA's `A_log` and `dt_bias` initialization: 10–22 of 384
  such elements per seed differ by one ULP between Intel and AMD hosts, and all
  other parameters are bit-identical. Seed 0 built on an Intel Xeon node
  reproduces the ORCD fingerprint exactly; Kennel's H100 hosts are AMD EPYC. The
  random DeltaNet metrics deviate from ORCD no more than the pretrained condition
  does, whose parameters are identical.

## Aggregation conventions and limitations

Rank metrics are computed for each input/head state first, then averaged equally
over inputs and heads. Each faint random curve is one seed's input/head
aggregate; the solid random curve is the equal-weight mean over seeds, and
shading is the minimum–maximum range of those seed aggregates, a descriptive
range rather than a confidence interval. Inputs and heads are not treated as
independent replications, and pretrained/random heads are not assumed matched.

For each saved input/head state, relative singular values are
sqrt(λᵢ / λ₁) from the normalized eigenvalues, averaged only afterward. A zero
state contributes zeros. Spectral values below 10⁻⁶ are clipped only for
logarithmic display. Key/value norm panels show sqrt(mean(‖vector‖₂²)) across
inputs and heads at the terminal token: RMS vector norms, not per-component RMS
and not averages over token positions. The energy panel averages log(trace(B))
over input/head states and converts to base 10; it is not the log of the
arithmetic mean energy.

This is a descriptive diagnostic on four fixed texts. The pretrained model is one
checkpoint, while random weights are independent initialization seeds of the
same architecture; seed ranges do not measure pretrained uncertainty. These are
fresh reference initializations, not the models' original starting weights or a
tracked training trajectory. The comparison does not identify a causal
contribution of an individual head or of learned decay, gates, projections, or
normalization, and does not establish generalization to other texts,
checkpoints, model sizes, or training recipes. State rank alone does not measure
language-model quality.
