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

The original pretrained Llama and DeltaNet result folders retain their numerical
archives, inputs, metadata, and aggregate figures. Their standalone `text-1`
through `text-4` figures have been removed. Use the unified model collections
below for the current sequence/seed-averaged presentation.

## Random versus pretrained Llama

The completed six-condition comparison is retained in
[`results/state_rank/random_vs_pretrained/REPORT.md`](../../results/state_rank/random_vs_pretrained/REPORT.md).
Use a new `--output` directory for a new run; the retained manifest is a historical
execution record and protects the original source/configuration/input identity.

`configs/random_vs_pretrained.json` compares the pinned Llama-3.2-1B checkpoint
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
"$RANK_PROJECT_ROOT/.venv/bin/python" -m pytest
# Commit the tested scaffold before freezing its source identity and submitting.
"$RANK_PROJECT_ROOT/.venv/bin/python" -m experiments.state_rank.launch_comparison \
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
`results/logs/state_rank/random_vs_pretrained/<condition>/slurm-<job_id>.out`.

Each condition retains scalar rank/norm curves at every token and normalized
spectra at prefixes 1, 2, 8, 32, 128, and 512. Metrics have axes
`[layer,input,token,head]`; saved spectra use `[layer,input,prefix,head,mode]`.
Large numerical archives are ignored by Git. No random checkpoints or raw KV
arrays are retained. Metadata records parameter fingerprints, initialization,
inputs, pinned model configuration, package versions, and source provenance.

Regenerate the compact report and four figure groups with:

```bash
"$RANK_PROJECT_ROOT/.venv/bin/python" -m experiments.state_rank.plot_comparison \
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
[`results/state_rank/random_vs_pretrained_deltanet/REPORT.md`](../../results/state_rank/random_vs_pretrained_deltanet/REPORT.md).

`configs/random_vs_pretrained_deltanet.json` applies the same design to the pinned
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
"$RANK_PROJECT_ROOT/.venv/bin/python" -m experiments.state_rank.launch_comparison \
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
log per condition under `results/logs/state_rank/random_vs_pretrained_deltanet/`.
The same plotting command above accepts this results directory. Rank curves,
selected spectra, state energies, resolved configurations, and provenance are
retained; no checkpoints, raw recurrent states, or convolution buffers are
saved. Key/value vector norms are not collected for this diagnostic. Use a fresh
`--output` directory to rerun a retained comparison without changing its history.

## Unified analysis of both architectures and weight conditions

The [unified report](../../results/state_rank/unified_comparison/REPORT.md) has
one collection for [Llama](../../results/state_rank/unified_comparison/llama/README.md)
and one for [Gated DeltaNet](../../results/state_rank/unified_comparison/gated-deltanet/README.md),
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
uv run --frozen python -m experiments.state_rank.unify_analysis \
  --output /path/to/new-results
```

The default configuration is `configs/unified_rank_analysis.json`; edit a copy
and pass `--config` to select other saved layers. `--project-root` locates the
retained source archives when executing from a separate checkout. The generator
requires all planned conditions, verifies archive/input checksums and matched
scientific conventions, and records source hashes before and after generation.
Outputs include five PNG/PDF figures, per-head and per-layer terminal CSVs, detailed
comparison statistics, resolved configuration, and a provenance manifest. The
CSV summaries retain individual seeds for auditing; the figures show seed
averages. The original per-token numerical archives remain the source of truth.
