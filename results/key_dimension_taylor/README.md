# Joint query/key dimension × Taylor degree

This completed collection asks how the accuracy of normalized Taylor attention
depends jointly on query/key width and polynomial degree. All 25 grid cells have
a committed terminal evaluation at 19,074 updates. Sixteen models were trained
for this collection; four degree sweeps at dK=64 and five softmax references are
reused from the retained [polynomial](../polynomial_attention/README.md) and
[key-dimension](../key_dimension/README.md) collections.

Increasing degree substantially narrows the gap to softmax, especially at small
query/key widths. Degree 8 is slightly better than degree 16 at dK=64 and 128;
the degree-8 advantage over softmax is only 0.0117 and 0.0190 perplexity there.
These are single-seed observations, not evidence of a statistically reliable
improvement. At dK=8,16,32, degree 16 comes closest to softmax while remaining
0.1426, 0.0537, and 0.0337 perplexity worse, respectively.

## Method and scientific identity

Rows use dK={8,16,32,64,128}; columns use even Taylor degree m={2,4,8,16} and
softmax. The polynomial is the unshifted truncated exponential
`P_m(s) = sum_{j=0}^m s^j/j!` applied to causally masked, scaled dot-product
scores, with row normalization. Production uses the fused Pallas/Triton
implementation, BF16 model computation, and the original explicit checks for
nonfinite scores, kernels, or normalizers. The diagnostic blocked implementation
evaluates a held-out sample and does not supply the reported validation loss.

The shared reference architecture has 10 transformer layers, hidden width 1024,
16 query/key/value heads, fixed value width 64, MLP width 3008, vocabulary 32000,
RoPE, RMSNorm, untied embeddings, and no dropout. Changing dK changes parameter
count: 181,556,224; 184,177,664; 189,420,544; 199,906,304; and 220,877,824 for the
five widths. “200M” denotes the reference recipe, not an equal parameter budget.

All cells use seed 0, FineWeb-Edu packed data, a global batch of 256, global
microbatch 64, four accumulation steps, and 2,048 next-token prediction positions.
Four GPUs cooperate on each new run. AdamW uses learning rate 0.0003, minimum
0.00003, 2,000 warmup updates, beta1=0.9, beta2=0.95, weight decay 0.1, and norm
clipping at 1. The common endpoint is 19,074 × 256 × 2,048 = 10,000,269,312
prediction tokens. Evaluation uses the fixed first seven global held-out batches.
Exact recipes, rather than this prose summary, are authoritative in [configs](configs/).

The original scientific source is
`4d942eae8981752f4666d4b6f8261b35e1338320`, with sweep identity
`7363664d1dafdf5d46c34dabb2d1567761279a0515c94c6bbc2f59505812b0b3`.
The frozen [dependency lock](configs/uv.lock), data-manifest hashes, per-run
sources and seeds, exact recipe hashes, metrics, selected checkpoints, execution
amendments, and diagnostic evidence are indexed by
[the canonical provenance manifest](provenance/manifest.json).

The softmax reference at dK=64 is the key-dimension result
**14.689767074685026**. The separate polynomial experiment's softmax result
**14.69417664861716** is not used in this grid or its deltas. All deltas subtract
softmax at the same width. Figures retain the final rendering from
`84b9f2296442d2f99b857cd6e4d3ec1e8c6576ea`: heatmaps have no reuse asterisks;
reuse remains explicit in the CSV, manifest, table, and learning curves.

<!-- BEGIN GENERATED RESULTS -->
The retained 25-cell terminal table is in [summary.csv](summary.csv).

| dK | m=2 | m=4 | m=8 | m=16 | Softmax |
|---:|---:|---:|---:|---:|---:|
| 8 | 20.5775 | 18.1785 | 17.1790 | 16.8479 | 16.7053 |
| 16 | 18.4989 | 16.5132 | 15.8001 | 15.6301 | 15.5764 |
| 32 | 17.1717 | 15.5698 | 15.0712 | 15.0190 | 14.9853 |
| 64 | 16.2810 | 14.9836 | 14.6781 | 14.7062 | 14.6898 |
| 128 | 15.7202 | 14.5300 | 14.3586 | 14.3831 | 14.3776 |

All cells are complete. The dK=64 polynomial row and all softmax cells are reused.
Full precision terminal NLL/perplexity, best-checkpoint results, source, reuse
status, and width-matched deltas remain in the CSV and provenance.

[Terminal heatmap](figures/terminal_ppl_heatmap.pdf) ·
[Delta heatmap](figures/delta_ppl_heatmap.pdf) ·
[Learning curves](figures/learning_curves.pdf)
<!-- END GENERATED RESULTS -->

## Execution, hardware validation, and diagnostic results

The home cluster for the sixteen new runs and the hardware checks is **Kennel**,
with artifact root `/data/cl/u/rafah/home/projects/functional-memory`. Production
used four H100s on dogo or four B300s on akita. The initial sweep became concurrent
independent jobs; the scientific recipes, batch, update budget, and seeds were
unchanged. Orchestration sources are `0dc83ae8` (parallel), `2b27b9e4` (A6000),
and `78b2f7f4` (H100-or-better queue). Full commit IDs and exact policy records
are retained in the manifest and [execution evidence](evidence/execution/).

Three runs—dK32/m16, dK128/m8, and dK128/m16—first reached step 500 on A6000s
using runtime `26faf4e00f7d2b6e54055a26343eb7b8f9d379eb` with host-staged transfer.
After operator cancellation, the same checkpoints resumed with the original
runtime on B300, B300, and H100 respectively. The handoff and independent
state/data/RNG restore audits are retained in their run records. This hardware
change can affect subsequent floating-point trajectories; no A100 pilot enters
the scientific grid.

Validation used the real full model and production batch, 32 training updates,
evaluation/checkpoint creation, and a separate restore from step 32 to 33.
Checks also compared initial model/input identities and reference attention
outputs/gradients. Hardware acceptance requires the full-model checks, not only
isolated kernel tests. The retained native kernel suite reported 105 passing
checks, even on machines whose end-to-end training failed.

| Allocation / transfer mode | Full-model outcome | Successful validation jobs | Evidence |
|---|---|---|---|
| H100, native | Original preflight and completed production | 200502 | [Preflight](evidence/preflight/) |
| B300, native | 20/20 coordinates passed; five checks reused the first qualification | 200733, 200753 | [First qualification](evidence/b300_initial/) and [grid](evidence/b300_native/) |
| A100, native | 1/20 passed; 19/20 failed with nonfinite validation at step 32 | 200754 | [Native grid](evidence/a100_native/) |
| A6000, host-staged | 20/20 passed, including checkpoint resume and replication audits | 200781–200786 | [Host-staged grid](evidence/a6000_host_staging/summary.json) |
| A100, host-staged | Representative dK8/m4 passed; full 20-cell grid was not rerun | 200853 | [Transport ablation](evidence/a100_transport/validation.json) |

A completed Slurm validation job means that its diagnostic harness finished;
it does not imply every tested coordinate passed. The A100 native failures
remain recorded, rather than being replaced by the later representative success.

On A6000, native peer transfers produced stale/corrupt values despite successful
CUDA calls. A 72-comparison transfer probe found six failed comparisons;
host-backed transfer was exact. Disabling NCCL peer-to-peer alone fixed the
collective hang but did not fix direct JAX GPU copies. The generic fix stages
state and batches through host memory and audits each replica. The attention
mathematics and precision were unchanged. Two early fixed-runtime pilots ran
out of memory before the first update; reserving 90% of the device pool allowed
the complete qualification to pass. Supporting records remain in
[A6000 transport evidence](evidence/a6000_transport/) and the
[memory-pool amendment](evidence/a6000_host_staging/memory-pool-adjustment.json).

The later A100 experiment on samoyed separated direct transfer corruption from
attention or optimizer instability. Native replication left GPU0 exact but
93 model/optimizer-state leaves on each of GPUs1–3 mismatched; all 131,136
training token positions on those replicas were corrupted. An out-of-range
evaluation token had the bit pattern of a previous floating-point loss value,
consistent with stale allocation contents. Native and NCCL-disable-only modes
failed, while host-staging alone and host-staging plus NCCL disabling both passed
replication and step32→33 restore checks. Their step-32 NLLs were 9.8021605 and
9.8021476 respectively. This is one coordinate, not full A100 production
qualification.

An independent direct-CUDA probe tested 12 directed GPU pairs at 4 KiB, 1 MiB,
and 16 MiB, both with and without peer access: all 36 peer-enabled copies were
corrupt and all 36 peer-disabled copies were exact. These calls did not use JAX
or NCCL. All four GPUs were in translated IOMMU DMA-FQ domains, without NVLink,
on a bare-metal node with driver 560.35.05. This is evidence for a host/driver
peer-transfer problem; the IOMMU configuration is a leading explanation, not a
proven causal diagnosis without an administrator-controlled A/B test. The
[exact direct-CUDA probe](evidence/a100_transport/cuda_peer_probe.py),
[copy results](evidence/a100_transport/cuda_peer_copy.json),
[allocation](evidence/a100_transport/allocation.json), and
[IOMMU evidence](evidence/a100_transport/iommu_domains.json) are preserved.
The committed training probe/launcher remain recoverable at source
`a1f1b2f04a3a625d8f48735aad0c990a09dbcc01`. Earlier temporary A6000 probe scripts
had already been removed; their recorded results/checksums remain, so those
diagnostics are not claimed to have a complete runnable source snapshot.

## Timing and interpretation

The sixteen successful terminal jobs took 1:49:03 to 4:58:18 each. For the three
handoff runs, that measures only the final allocation, excluding their preceding
A6000 work. These are scheduler elapsed times including startup/compilation,
training, evaluation, checkpoint I/O, and checks; they exclude queue time.
[scheduler.json](evidence/execution/scheduler.json) records the allocation,
duration, and job state. Completion records also retain trainer runtime and
individual training/evaluation/checkpoint timing samples.

These durations are operational observations, not a matched GPU benchmark:
width, degree, preceding progress, and allocation differ. The diagnostic
step32/33 wall times include fresh compilation and audits and must not be used
as steady-state throughput. For controlled kernel and full-update benchmarks,
including warmups, synchronization, and measurement exclusions, use the retained
[polynomial performance measurements](../polynomial_attention/README.md#performance-measurements).

## Retained artifacts and cleanup

Each of the sixteen new runs has one validation-best checkpoint at
`checkpoints/key_dimension_taylor/<run>/19074/` on Kennel. In this collection the
minimum committed validation NLL happens to occur at the terminal step for every
new run; selection is still by validation loss. The retained raw metrics and
completion records preserve the common endpoint independently of checkpoint
selection. Migration checks compared Orbax metrics, metadata hashes, file counts,
and byte sizes against the selected validation; they did not rehash every large
checkpoint payload or perform a fresh GPU restore. All sixteen checkpoint
directories total 34,130,735,208 bytes.

The four reused polynomial checkpoints remain owned by that original collection
on Kennel. The five reused softmax checkpoints and logs remain at their recorded
ORCD locations and were not locally reverified or migrated. Their committed
scientific metrics/configurations were verified. A Git clone supplies no
checkpoints, datasets, environments, or ignored logs.

One successful terminal log per new run and eleven successful validation-harness
logs remain under `results/logs/key_dimension_taylor/`, indexed with SHA256 in
the manifest. Superseded jobs, pilot checkpoints, compilation caches, duplicate
per-run reports, and mutable launch queues are not retained outputs. Exact
historical JSON records can mention removed execution paths or previously kept
step-500 checkpoints; those fields describe their recording time. The canonical
manifest and its evidence source-to-path index define present artifact locations.
No old-path alias is needed to analyze this retained collection.

## Reproduction

To regenerate figures from committed scientific artifacts, use the current
shared scaffold from the repository root:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.envs/shared_transformer"
uv sync --project scaffolds/transformer --locked --extra cuda
uv run --project scaffolds/transformer --locked --extra cuda python -m \
  scaffolds.transformer.studies.key_dimension_taylor.plot \
  --manifest results/key_dimension_taylor/provenance/manifest.json
```

The plotter verifies recipes, metrics, completion metadata, terminal identities,
and validation-best selections. It needs neither GPUs nor datasets/checkpoints.
Regeneration updates only the marked results block, preserving these methods,
execution findings, and limitations. The originally retained figures are
byte-identical copies; subsequent rendering may vary with plotting-library
versions and PDF timestamps.

For an exact historical implementation replay, use the original source and
locked environment in a fresh dedicated checkout. The following prepares an
isolated source archive without any dependency on deleted execution worktrees;
the final command runs inside an already obtained four-GPU allocation and does
not submit a job:

```bash
set -euo pipefail
REPO_ROOT="$PWD"
REPLAY_ROOT="$(mktemp -d /tmp/joint-replay.XXXXXX)"
git archive 4d942eae8981752f4666d4b6f8261b35e1338320 | tar -x -C "$REPLAY_ROOT"
export UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.envs/joint_historical_replay"
uv sync --project "$REPLAY_ROOT/experiments/polynomial_attention" --locked --extra cuda
REPLAY_OUTPUT="$REPO_ROOT/results/joint_replay/runs/dk008-m002-seed0"
REPLAY_CHECKPOINTS="$REPO_ROOT/checkpoints/joint_replay/dk008-m002-seed0"
test ! -e "$REPLAY_OUTPUT" && test ! -L "$REPLAY_OUTPUT"
test ! -e "$REPLAY_CHECKPOINTS" && test ! -L "$REPLAY_CHECKPOINTS"
mkdir -p "$REPLAY_OUTPUT" "$REPLAY_CHECKPOINTS"
ln -s "$REPLAY_CHECKPOINTS" "$REPLAY_OUTPUT/checkpoints"
cd "$REPLAY_ROOT"
export PYTHONPATH="$REPLAY_ROOT"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset POLYNOMIAL_ATTENTION_HOST_STAGING NCCL_P2P_DISABLE XLA_PYTHON_CLIENT_MEM_FRACTION
uv run --project experiments/polynomial_attention --locked --extra cuda python -m \
  experiments.polynomial_attention.train \
  "$REPO_ROOT/results/key_dimension_taylor/configs/200m-dk008-m002-seed0.yaml" \
  --workdir "$REPLAY_OUTPUT" --no-resume
```

Use fresh replay output/checkpoint names, and keep training artifacts outside the
temporary source archive. The retained recipe points to the Kennel packed corpus
at `/data/cl/u/rafah/home/datasets/fineweb_12b`; elsewhere, change only data paths
in a copied recipe and verify the manifest hashes before training. Check the
historical lock against the retained lock before installation. Run original
native production on its qualified H100/B300 path. Replaying the three mixed
hardware trajectories additionally requires their recorded step-500 handoff
sequence; the discarded intermediate checkpoints themselves are not retained.
Numerical bit identity across hardware is not asserted.

The direct-CUDA diagnostic is a standalone script. Inside a four-A100 allocation
and the historical Python 3.11/CUDA environment, run it with a fresh output path:

```bash
set -euo pipefail
CUDA_COPY_OUTPUT="$(mktemp -d /tmp/cuda-copy-replay.XXXXXX)/copies.json"
"$UV_PROJECT_ENVIRONMENT/bin/python" \
  "$REPO_ROOT/results/key_dimension_taylor/evidence/a100_transport/cuda_peer_probe.py" \
  "$CUDA_COPY_OUTPUT"
```

For the full A100 transfer ablation, the original supervisor verifies its Git
checkout identity. Recreate that exact source in a fresh temporary working
checkout and supply an isolated copy of its input policy whose configuration
paths point to the retained qualification recipes. This changes only local input
locations; the historical policy and recipes remain untouched:

```bash
set -euo pipefail
DIAGNOSTIC_ROOT="$(mktemp -d /tmp/a100-ablation-replay.XXXXXX)"
mkdir -p "$DIAGNOSTIC_ROOT/worktrees"
git -C "$REPO_ROOT" worktree add --detach "$DIAGNOSTIC_ROOT/worktrees/runtime" \
  a1f1b2f04a3a625d8f48735aad0c990a09dbcc01
"$UV_PROJECT_ENVIRONMENT/bin/python" - "$REPO_ROOT" "$DIAGNOSTIC_ROOT" <<'PY'
import json, sys
from pathlib import Path
repo, replay = map(Path, sys.argv[1:])
evidence = repo / 'results/key_dimension_taylor/evidence'
policy = json.loads((evidence / 'execution/hardware-policy.json').read_text())
for run in policy['runs']:
    run['config_path'] = str(evidence / 'qualification_configs' / f"{run['name']}.yaml")
target = replay / 'results/key_dimension_taylor/hardware_checks/policy.json'
target.parent.mkdir(parents=True)
target.write_text(json.dumps(policy))
PY
cd "$DIAGNOSTIC_ROOT/worktrees/runtime"
PYTHONPATH="$PWD" "$UV_PROJECT_ENVIRONMENT/bin/python" \
  experiments/key_dimension_taylor/a100_diagnostics/probe.py \
  --output "$DIAGNOSTIC_ROOT/evidence" \
  --commit a1f1b2f04a3a625d8f48735aad0c990a09dbcc01
```

The supervisor runs native, NCCL-disable-only, host-staged-only, and combined
modes with bounded step32/33 workers. Retain its output outside the temporary
source checkout before removing that checkout. This replay requires matching
packed data and a four-A100 allocation; it does not submit production jobs or
reuse the deleted execution directories.

For a new experiment, use a dedicated branch/checkout and the current
`scaffolds.transformer.train` entry point with fresh canonical result/checkpoint
paths, a new source identity, and the launcher-selected generic runtime settings
documented in the [shared scaffold](../../scaffolds/transformer/README.md) and
[cluster guide](../../CLUSTERS.md). Historical diagnostic launchers are recovered
from their recorded commits for forensic replay; do not reuse their old mutable
manifests or output paths. The completed results establish neither independent
seed uncertainty nor blanket support for every GPU/runtime configuration.

In a copied recipe for new shared-scaffold training, explicitly set
`train.source_commit: auto`, `train.workdir: results/<new-experiment>/runs/<new-run>`,
and `train.checkpoint_dir: checkpoints/<new-experiment>/<new-run>`. Resolve those
paths for the new experiment's home cluster, check that both destinations are
fresh, and record the resolved new configuration before launching. Leave the
historical retained recipe unchanged.
