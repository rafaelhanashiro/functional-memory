# Query/key dimension × Taylor degree

[Selected runs and source hashes](selection.yaml) · [Run index](../../experiments/key_dimension_taylor/runs.csv) · [Execution guide](../../experiments/key_dimension_taylor/README.md)

How does held-out perplexity depend on the per-head query/key width `dK` and on
the degree `m` of normalized Taylor attention, with softmax as the `m → ∞`
reference? This completed record holds **29 seed-0 runs** of the 200M reference
Transformer at a common 10,000,269,312-token budget. The heatmaps focus on the
complete 25-cell joint comparison: `dK` 8, 16, 32, 64, and 128, crossed with `m`
2, 4, 8, and 16 plus one softmax reference per width. Four further runs remain in
the record and summary CSV:

| Source | Coordinates | Runs | Executed |
| --- | --- | ---: | --- |
| Width sweep | softmax at dK 2, 4, 8, 16, 32, 128, 256 | 7 | ORCD, 2026-09-16 to 09-18 |
| Degree sweep | m = 0 at dK 64 | 1 | Kennel, 2026-09-17 |
| Joint sweep | m = 2, 4, 8, 16 at dK 8, 16, 32, 128 | 16 | Kennel, 2026-09-20 to 09-22 |
| Softmax baseline | softmax at dK 64, selective weight decay | 1 | Kennel, 2026-09-25 |
| Normalized Taylor rerun | m = 2, 4, 8, 16 at dK 64, QK-norm and selective weight decay | 4 | Kennel, 2026-09-26 to 09-27 |

Each run's source commit, hardware, and Slurm attempts are in its
[run record](../../experiments/key_dimension_taylor/runs/).

**Findings.** Softmax improves monotonically with query/key width in this family,
from PPL 22.35 at dK=2 to 14.17 at dK=256, with diminishing returns above dK=32
(14.99). Width also adds parameters (`178,934,784 + 327,680·dK`), so these
differences do not isolate width from model size. Increasing Taylor degree
narrows the gap to softmax, most strongly at small widths: at m=2 the gap falls
from +3.87 PPL at dK=8 to +1.34 at dK=128. At dK=128, degrees 8 and 16 are
within ±0.02 PPL of softmax (degree 8 lower by 0.0190), which one seed cannot
establish as an ordering. At dK 8, 16, and 32, degree 16 comes closest but remains
0.1426, 0.0537, and 0.0337 worse. Degree 0, uniform causal-prefix averaging,
reaches only 47.07; its Q/K projections receive no attention gradient.

The dK=64 row now uses newer training setups. The softmax reference
(`200m-softmax-seed0`, source `381f1fd`) is the repository's softmax baseline,
retrained on 2026-09-25 under the selective weight-decay policy. The Taylor runs at
m = 2, 4, 8, and 16 (source `a74e386`) were retrained on 2026-09-26/27 under the
scaffold's current defaults, QK-norm cosine scores and selective weight decay.
Degrees 8 and 16 reach 14.4978 and 14.4539: 0.17 and 0.21 below this softmax
reference, which has no QK-norm, and within 0.04 of the QK-norm softmax baseline
(14.4574, [QK-norm attention](../qk_norm_attention/README.md)). Every other run
here trained with decay on all parameters and without QK-norm.

<!-- BEGIN GENERATED RESULTS -->
Terminal held-out perplexity at 19,074 updates (10,000,269,312 prediction tokens), seed 0. 29 of 29 retained runs are complete. The heatmap shows the 25-cell joint comparison; other retained runs remain in the CSV and manifest.

| dK | m=2 | m=4 | m=8 | m=16 | Softmax |
|---:|---:|---:|---:|---:|---:|
| 8 | 20.5775 | 18.1785 | 17.1790 | 16.8479 | 16.7053 |
| 16 | 18.4989 | 16.5132 | 15.8001 | 15.6301 | 15.5764 |
| 32 | 17.1717 | 15.5698 | 15.0712 | 15.0190 | 14.9853 |
| 64 | 16.5107 | 14.8963 | 14.4978 | 14.4539 | 14.6655 |
| 128 | 15.7202 | 14.5300 | 14.3586 | 14.3831 | 14.3776 |

Runs retained outside the grid: `200m-dk002-seed0` (softmax, dK=2, PPL 22.3489); `200m-dk004-seed0` (softmax, dK=4, PPL 18.8928); `200m-m000-seed0` (m=0, dK=64, PPL 47.0653); `200m-dk256-seed0` (softmax, dK=256, PPL 14.1736). The grid uses one softmax reference per width; at dK=64 it is `200m-softmax-seed0` (PPL 14.6655).

Each delta in the CSV and delta heatmap subtracts the softmax endpoint at the same width. `original_experiment` in the CSV records which sweep trained each run; heatmaps do not mark it. FAILED means a recorded numerical failure; interrupted or unaudited runs remain INCOMPLETE. Source commits, exact recipes, and artifact checksums are in `experiments/key_dimension_taylor/sweep.yaml` and each run’s `record.json`.

Checkpoints and logs are not retained for 7 runs trained on another cluster (`200m-dk002-seed0`, `200m-dk004-seed0`, `200m-dk008-seed0`, `200m-dk016-seed0`, `200m-dk032-seed0`, `200m-dk128-seed0`, `200m-dk256-seed0`); their metrics, completion records, metadata, and recipes are retained and checksum-verified.

[Summary CSV](summary.csv) · [Terminal heatmap](terminal_ppl_heatmap.pdf) · [Delta heatmap](delta_ppl_heatmap.pdf) · [Learning curves](learning_curves.pdf)
<!-- END GENERATED RESULTS -->

## Method and scientific identity

Every run uses the 200M reference architecture: 10 layers, residual width 1024,
16 full-MHA heads, value width 64, SwiGLU width 3008, untied 32,000-token
embeddings, RMSNorm (epsilon `1e-6`), RoPE (base 10,000) at each model's true
query/key width, no dropout, BF16 computation, and FP32 parameters and optimizer
state. "200M" names the recipe, not an equal parameter budget: dK=64 has
199,906,304 parameters, each width step changes only the Q/K projections, and the
QK-norm runs add 160 per-head scales (199,906,464).

All runs use seed 0, the same packed FineWeb-Edu corpus and order, global batch
256 with microbatch 64 and four accumulation steps, 2,048 prediction positions,
and **19,074 updates**. AdamW uses peak/minimum learning rates `3e-4`/`3e-5`,
2,000 warmup updates, cosine decay, betas (0.9, 0.95), weight decay 0.1, and
clipping at 1.0 after accumulation. The dK=64 softmax and Taylor runs apply the
selective policy, with no decay on the token embedding, RMSNorm scales, biases, or
QK-norm scales; every other run decays all parameters. Evaluation uses the first 1,792 held-out
sequences (3,670,016 prediction tokens) every 1,000 updates and at completion;
perplexity is the exponential of mean token NLL. The primary endpoint is terminal
perplexity at the common budget; validation-best results are recorded separately.

`attention_degree: null` selects softmax. An integer `m` selects the unshifted
truncated exponential `P_m(s) = sum_{r=0}^m s^r / r!` on post-RoPE scores
`s = q·k / sqrt(dK)`, normalized over the causal prefix:
`sum(P_m(s) v) / sum(P_m(s))`. The dK=64 Taylor runs instead use QK-norm cosine
scores `s = exp(gamma) · cos(q, k)`, with `gamma` learned per layer and head from
`log(sqrt(dK))`. There is no score shifting, clipping, temperature, or added
normalization; invalid kernels or normalizers fail visibly. Degrees 2–16
used the fused Pallas/Triton kernel (FP32 Horner arithmetic, causal tiles only,
no sequence-square arrays); m=0 used the blocked reference; softmax used cuDNN.
The fused kernel and its benchmarks are documented in
[shared scaffold validation](../../scaffolds/transformer/validation/README.md).

[Exact recipes](../../experiments/key_dimension_taylor/runs/) are authoritative: one resolved configuration per run.

## Execution and provenance

Each [run record](../../experiments/key_dimension_taylor/runs/) holds its source
commit, recipe hash, hardware, Slurm attempts, selected checkpoint, and log.

- **Width sweep:** source `5e871d2` (dK=256: `3d330a2`, after one earlier
  attempt), two GPUs per run on ORCD, cuDNN softmax. Its checkpoints and logs
  stayed on ORCD and are **not retained**, by decision on 2026-09-24; metrics,
  completion records, metadata, and recipes are retained and checksum-verified.
- **Degree sweep:** source `2484f00` for the retained m=0 run, four H100 PCIe
  with the blocked backend. The sweep's m = 2–16 and softmax runs were replaced
  by the two entries below.
- **Softmax baseline:** source `381f1fd`, selective weight decay, four H100s and
  then four B300s, cuDNN.
- **Normalized Taylor rerun:** source `a74e386`, QK-norm and selective weight
  decay, fused kernel, four H100s or B300s per attempt, with the shared launcher's
  acceptance in every allocation. The runs were moved between partitions and
  preempted several times. m=2 trained on B300 and resumed on H100 from its
  update-6,500 checkpoint; m=4 and m=16 trained on B300, and m=8 on H100. The
  manifest records every attempt, partition change, and acceptance. The same
  launch produced the odd degrees in [odd Taylor attention](../odd_taylor_attention/README.md).
- **Joint sweep:** source `4d942ea`, sweep identity `7363664d…`, four H100s on
  `dogo` or four B300s on `akita`. Pre-launch acceptance (job 200502) ran every
  coordinate through 32 full-recipe updates, validation, and an exact step-32→33
  resume; all passed and are summarized per run in the manifest. Three runs
  (dK32/m16, dK128/m8, dK128/m16) reached step 500 on host-staged A6000s, then
  resumed from the same checkpoints on B300, B300, and H100; their handoff and
  restore audits are in their run records. Orchestration sources were
  `0dc83ae8`, `2b27b9e4`, and `78b2f7f4`.

Hardware and transfer qualification for these recipes (A100, A6000, B300) and
the Taylor kernel and throughput benchmarks describe the shared runtime and are
retained in [shared scaffold validation](../../scaffolds/transformer/validation/README.md).
Joint-sweep jobs took 1:49:03 to 4:58:18 each, including startup and checks; for
the three handoff runs this excludes the A6000 segment. These are operational
times, not a matched benchmark.

## Retained artifacts

On Kennel, 22 runs (16 joint, m=0, the dK=64 softmax, and the four dK=64 Taylor
reruns) each keep one validation-best checkpoint at `runs/<run_id>/checkpoints/<step>/` in
this experiment directory (47,174,560,306 bytes in total). Selection is by minimum validation NLL; it is the terminal step
for every run except m=0 (step 19,000). Each retained run has a matching `runs/<run_id>/training.log`; its record contains the checksum and acceptance history. Checkpoint migration checks compared Orbax
metrics, metadata hashes, file counts, and byte sizes, not every payload byte.
A Git clone supplies no checkpoints, datasets, environments, or logs.

## Reproduction

Regenerate the summary, figures, and results block from the committed runs; no GPU,
dataset, or checkpoint is needed:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m results.key_dimension_taylor.plot
```

The plotter verifies every recipe, metrics file, completion record, terminal
identity, and validation-best selection against the manifest, and rewrites only the
marked results block; figures regenerate byte-identically.

The exact executed code for each run is at its recorded source commit
(`git show <commit>:<path>`).

Use the [execution guide](../../experiments/key_dimension_taylor/README.md) to create and submit a fresh run with the current scaffold.

Recipes point to the Kennel corpus at `/data/cl/u/rafah/home/datasets/fineweb_12b`.
The three mixed-hardware joint trajectories also depended on their step-500 handoff,
whose intermediate checkpoints are not retained; bit identity across hardware is not
asserted.

## Limitations

Every coordinate has one seed. The focused heatmap grid is complete; the other
runs cover only softmax at dK 2, 4, and 256 and m=0 at dK=64. Width changes
parameter count. Hardware differed across sweeps and, for some runs, within a run.
The dK=64 row mixes training setups. Its Taylor runs use QK-norm and selective
decay, its softmax reference uses selective decay without QK-norm, and every other
width uses the original setup (decay on all parameters, no QK-norm). Deltas at
dK=64 therefore include QK-norm, which moves softmax by 0.21 PPL (14.6655 to
14.4574); comparisons across widths include both changes, and the decay policy
alone moves softmax by about 0.03 PPL.
