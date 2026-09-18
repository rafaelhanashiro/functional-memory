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

## Random versus pretrained Llama

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
