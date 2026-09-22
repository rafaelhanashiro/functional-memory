# Unnormalized exponential attention

This experiment asks what changes when a 250M Transformer uses causal
`exp(QKᵀ / sqrt(d_K)) V` without dividing attention weights by their row sum.
**Status: setup and validation only; no completed unnormalized result is retained.**
The original source snapshot is `7a3e584`, before migration to the shared scaffold.
Live ORCD jobs and their ignored outputs have not been inspected or transferred.

**Home cluster: ORCD / Engaging.** The recorded project root is
`/home/rafah/projects/functional_memory`; the packed corpus is under
`/home/rafah/projects/ofm/datasets/fineweb_12b`. A clone provides code and the
records here, not remote checkpoints, environments, logs, or datasets. The
[provenance manifest](provenance/manifest.json) records source/configuration
identities and historical versus proposed artifact locations.

## Scientific recipe

The model has **253,654,016 parameters**: 14 layers, width 1024, 16 full-MHA
heads, `d_K=d_V=64`, SwiGLU width 3008, 32,000-token untied embeddings, 2,048
prediction positions, RoPE base 10,000, RMSNorm epsilon `1e-6`, and no dropout.
The block and final RMSNorm layers and vocabulary softmax remain unchanged.

The seed is 0. Packed FineWeb data, tokenizer, sequence order, AdamW parameters,
and learning-rate schedule match the historical OFM control. Global batch 256,
global microbatch 64, and four accumulation steps give **21,935 updates and
11,500,257,280 prediction tokens**. Warmup lasts 2,000 updates; peak/minimum
learning rates are `3e-4`/`3e-5`. Evaluation uses the first 1,792 held-out sequences
every 1,000 updates and at completion. Parameters and optimizer state are FP32;
model computation is BF16.

The [historical normalized baseline](baseline.json) used two H200s and reports
terminal/best validation NLL 2.591487 and PPL **13.3496** at update 21,935. These
rounded values come from OFM source `32ea743b7b3aed1d631116dfaa3c4b85101baf32`;
they are not results of a new normalized run. The comparison is single-seed and
uses different kernels: cuDNN for historical softmax, Pallas/Triton for the new
unnormalized operator. Primary endpoint: terminal held-out perplexity at the
matched token budget. Best-checkpoint metrics must be retained separately.

## Implementation and source comparability

The [shared Transformer scaffold](../../scaffolds/transformer/README.md) owns the
model, data pipeline, optimizer, trainer, evaluation, and checkpoints. Setting
`model.attention_normalization: false` selects the exponential mechanism;
`true` retains the existing normalized mechanisms. Unnormalized Taylor or learned
quadratic combinations are rejected because they are different experiments.
There is no score clipping, row-max subtraction, prefix-length division, added
Q/K normalization, or added attention-output normalization.

The fused implementation visits causal query/key tiles without materializing a
sequence-square attention matrix. Exponentials and accumulation are FP32. Mixed
FP32/BF16 products split the FP32 operand into three BF16 components; FP32/FP32
products use TF32x3. Its derivative is `dS = exp(S) * (dY Vᵀ)`, without the
softmax row-coupling term. The fused kernel was moved without changing its source
bytes. The shared blocked reference supplies independent scalar diagnostics.

`attention_backend: auto` selects the CPU reference or a supported NVIDIA fused
implementation; requested and resolved choices are recorded, and resume uses
the previously resolved choice. Explicit incompatible backends fail. Nonfinite
causal scores, exponential weights, outputs, gradients, or updated parameters
stop training with diagnostics. Weight sums are measurements only: underflow to
zero or overflow of a sum does not introduce a denominator or silently alter the
mathematics. Compared with the original trainer, the shared guard additionally
checks updated parameters; it does not modify valid finite updates.

The [exact original recipe](configs/historical.yaml) and
[original dependency lock](configs/historical.uv.lock) are unchanged. For exact
historical replay, use a separate checkout at `7a3e584`, its original environment,
and `experiments.unnormalized_attention.train`; its paths describe that execution
layout. The [canonical recipe](configs/base.yaml) preserves the scientific fields
and changes only automatic backend selection and result/checkpoint locations.
The shared lock preserves the same pinned training-library versions. New source
requires acceptance again; old GPU acceptance is not inherited by a refactor.

## Validation

From a dedicated checkout, prepare the shared environment according to cluster
policy, then run CPU acceptance:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.envs/shared_transformer"
export UV_CACHE_DIR="$PWD/.cache/uv"
uv sync --project scaffolds/transformer --locked --extra cuda
JAX_PLATFORMS=cpu "$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest \
  -c scaffolds/transformer/pyproject.toml \
  scaffolds/transformer/studies/unnormalized_attention/tests \
  --import-mode=importlib -m 'not gpu'
JAX_PLATFORMS=cpu "$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest \
  -c scaffolds/transformer/pyproject.toml \
  scaffolds/transformer/tests/test_frozen_parity.py -k unnormalized_attention
```

Tests cover independent FP64 outputs and Q/K/V gradients, causal boundaries,
prefix sums at zero scores, sensitivity to score shifts, underflow and visible
overflow, initialization, gradient accumulation, packed-data resume, and failure
records. Frozen-source checks export `7a3e584` from Git and compare initialization,
logits, every gradient, optimizer updates, and restoration/continuation of original
checkpoints. Full initialization acceptance fingerprints all 253,654,016 FP32
parameters against the independently exported original key-dimension/OFM port.
This [full-model CPU check](validation/initialization.json) passed: the parameter
SHA256 is `6e1b4dddd849bc14944f8f7e399aa405179deac85a32903f89dd3ca83b0b8b80`.

GPU acceptance must run in the intended allocation. It includes fused forward/VJP
tests, exact initialization verification, 20 full-context/full-microbatch updates,
validation, and a one-update checkpoint-resume check while preserving the full
learning-rate horizon. CPU validation alone does not establish GPU qualification.

## Prepare a new run on the home cluster

Consult [cluster operations](../../CLUSTERS.md). Freeze a clean dedicated source
checkout and its environment before submitting. Copy `configs/base.yaml` to a
new experiment's `results/<experiment>/configs/` and resolve its data paths,
source identity, home cluster, and artifact root. Use fresh durable output paths,
for example:

```yaml
train:
  source_commit: auto
  workdir: /home/rafah/projects/functional_memory/results/unnormalized_attention/runs/250m-exp-seed0
  checkpoint_dir: /home/rafah/projects/functional_memory/checkpoints/unnormalized_attention/250m-exp-seed0
```

Keep all other sections of the recipe. The ORCD adapters are
`scaffolds/transformer/studies/unnormalized_attention/bootstrap.sbatch` and
`launch.sbatch`. The latter accepts `MODE EXECUTION_ROOT PROJECT_ROOT SOURCE_COMMIT
GPU_COUNT GPU_TYPE CONFIG`; its original defaults request two H100s in
`mit_preemptable`, 32 CPUs, 256 GiB RAM, 48 hours, and requeue. Use `preflight`
first and gate `train` on its successful job. Preserve one matching successful
log at `results/logs/unnormalized_attention/<run>/slurm-<job_id>.out`.

Inside an existing allocation, the portable adapter runs acceptance and then the
same shared trainer. It does not submit jobs or choose a partition:

```bash
bash scaffolds/transformer/studies/unnormalized_attention/run_allocated.sh \
  /absolute/path/to/resolved-config.yaml \
  /home/rafah/projects/functional_memory/.cache/unnormalized_attention \
  2 H100 /home/rafah/projects/functional_memory/.envs/shared_transformer/bin/python
```

One host runs one Python process over all allocated GPUs. GPU count must divide
global microbatch 64; global batch and accumulation remain fixed. Reusing the
same source/config/output resumes the latest complete checkpoint, optimizer,
and data cursor. Keep latest and validation best while training can resume;
after review retain only validation best, terminal metrics, exact recipes,
provenance, and the matching successful log.

## Kennel handoff

The original handoff describes an **independent fresh run**, not a resumed ORCD
run. Kennel compatibility and production qualification have not been established
by the documentation. Assign a separate run identity/home cluster and verify that
no unintended duplicate experiment is active before launching. Do not transfer
ORCD job-selector helpers containing ORCD job IDs. No such launch or transfer was
performed by this refactor.

Use the portable allocation adapter with Kennel resource policy, four identical
H100s or H200s, and corresponding `GPU_COUNT`/`GPU_TYPE` arguments. Four-device
acceptance is supported. Preserve scheduler-provided device visibility; the
adapter requires a real Slurm job ID. The exact packed corpus has 90 training
shards, one evaluation shard, and two manifests totaling 24,031,780,324 bytes
(22.38 GiB). Expected manifest SHA256 values are:

- Train: `9de775ed836ab7c61857207335a981c09630caa3226d2dc0ee0b799cdcb2f4f4`.
- Evaluation: `0e8615538d9dccad17956d8d9839394d05a56f8ec210dc67716db829a543634f`.

Manifest shard paths are relative; verify all listed files after transfer.
Tokenizer downloads, raw corpus, pretrained checkpoints, and Hugging Face caches
are not runtime requirements. Resolve only local data/output paths in a fresh
copy of the recipe. Check driver compatibility and rerun all acceptance on the
new allocation. Checkpoints, data, caches, environments, and logs remain outside Git.
