# Normalized fixed Taylor attention under the current defaults

This sweep reruns the `d_K = 64` column of the normalized fixed Taylor grid, degrees
**m = 1, 2, 3, 4, 8, 16**, under the scaffold's current defaults: QK-norm cosine
scores with a learned per-head scale, and selective weight decay (no decay on the
token embedding, RMSNorm scales, biases, or the QK-norm scales). The kernel is the
fixed Taylor polynomial

```text
y_i = sum_{j <= i} p_m(s_ij) v_j / sum_{j <= i} p_m(s_ij),   p_m(s) = sum_{r <= m} s^r / r!,
s_ij = exp(gamma_{l,h}) · <q_i, k_j> / (||q_i|| ||k_j||)
```

Every retained run of this family predates those defaults: `m = 2, 4, 8, 16` in the
[dK × Taylor-degree record](../key_dimension_taylor/README.md), and `m = 1, 3` in
[odd Taylor attention](../odd_taylor_attention/README.md), which failed numerically at
updates 9 and 17 when signed row sums crossed zero. Bounded cosine scores change the
range over which each polynomial is evaluated, so the odd degrees are the most
informative: with `|s| <= exp(gamma)` their kernels are still negative for scores
below a degree-dependent threshold. When the runs conclude they replace the legacy
runs in those records, as the rollout to the current defaults does for every
experiment.

**Status: running.** The results block below is written by the analysis module once
every run has concluded.

**Home cluster: Kennel.** The artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`. The
[provenance manifest](provenance/manifest.json) is the authoritative record of the
source commit, frozen execution set, acceptance, execution, and artifact locations.
The packed corpus is under `/data/cl/u/rafah/home/datasets/fineweb_12b`.

## Results

<!-- BEGIN GENERATED RESULTS -->
Pending: the runs have not concluded.
<!-- END GENERATED RESULTS -->

## Scientific recipe

Each [recipe](configs/) is the retained `d_K = 64` grid recipe
(`key_dimension_taylor/configs/200m-m002-seed0.yaml`) at its degree, with QK-norm
enabled and the decay policy stated; a configuration test enforces that nothing else
differs except output locations. The backbone is the repository's 200M model: 10
layers, width 1024, 16 heads, `d_K = d_V = 64`, untied embeddings, seed 0, packed
FineWeb in the fixed order, global batch 256, AdamW with peak/minimum learning rates
`3e-4`/`3e-5`, 2,000 warmup updates, **19,074 updates and 10,000,269,312 prediction
tokens**, evaluated on the first 1,792 held-out sequences every 1,000 updates and at
completion. All degrees use the scaffold's fused Pallas/Triton kernels: single-pass
for even degrees, and the signed two-pass kernel (final row sums first) for odd ones.

## Artifacts

| Artifact | Location in `experiments/normalized_taylor_rerun/` |
| --- | --- |
| Exact recipes | `configs/` |
| Committed metrics, metadata, and completion or failure records | `runs/<run>/` |
| Summary and figure | `summary.csv`, `figures/learning_curves.{png,pdf}` |
| Canonical manifest | `provenance/manifest.json` |
| Validation-best checkpoints (ignored) | `checkpoints/<run>/<step>/` |
| Slurm logs (ignored) | `logs/<run>/slurm-<job_id>.out` |

## Reproduction

Regenerate the summary, figure, and results block from the committed runs; no GPU,
dataset, or checkpoint is needed:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m experiments.normalized_taylor_rerun.adapter.analysis.report
```

The runs were submitted from this experiment's worktree root with the shared Kennel
launcher, which checks the frozen execution set against the launch commit, runs
acceptance (CPU tests, distributed CPU tests, the fused even and odd Taylor kernels
and QK-norm on the allocation's GPU type, initialization, a 20-update pilot, and an
exact resume audit), then trains; the manifest records each submission's partition
and resources.

## Limitations

One seed per degree. Each comparison with a legacy run changes two things at once
(QK-norm and the decay policy); the softmax baselines bound the decay policy's effect
at about 0.03 PPL.
