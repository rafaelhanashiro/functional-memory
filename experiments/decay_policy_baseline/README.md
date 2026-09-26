# Softmax baseline under selective weight decay

Does the scaffold's new weight-decay policy change the 200M softmax reference?
Every retained record was trained with AdamW decay on all parameters. Commit
`cb31558` made a selective policy canonical: decay still applies to every projection
matrix, including the output vocabulary projection, but no longer to the token
embedding, the RMSNorm scales, or biases. This experiment retrains the exact-softmax
reference at `d_K = 64` under that policy, with nothing else changed, and compares
it with the retained old-policy run. The outcome decides whether the existing
records stay comparable to new runs or are rerun under the new policy one
experiment at a time.

**Recipe.** [`configs/200m-softmax-seed0.yaml`](configs/200m-softmax-seed0.yaml) is
the retained [dK × Taylor-degree](../key_dimension_taylor/README.md) softmax recipe
with only its source commit (`auto`) and output paths changed: 10 layers, hidden
size 1024, 16 heads, `d_K = d_V = 64`, 199,906,304 parameters, cuDNN softmax
attention, global batch 256 × 2,048 tokens, 19,074 updates (10,000,269,312
prediction tokens), peak learning rate 3e-4 with 2,000 warmup updates and cosine
decay to 3e-5, weight decay 0.1, seed 0, on the packed FineWeb-Edu corpus at
`/data/cl/u/rafah/home/datasets/fineweb_12b`. The resolved configuration records
`optim.weight_decay_policy: exclude_embeddings_norms_biases_v1`.

**Control.** `experiments/key_dimension_taylor/runs/200m-softmax-seed0` (source
`1a8836a`, decay on all parameters): terminal validation NLL 2.687451, PPL 14.6942.

**Status: completed.** Under the selective policy the softmax reference ends at
NLL 2.685498 (PPL 14.6655) against the control's 2.687451 (PPL 14.6942): 0.0020 nats
and 0.029 PPL (0.20%) better. The gap is about six times the 0.0003-nat
hardware-only bound, so it is a real but small effect at this scale, and it favors
the new policy.

**Home cluster: Kennel.** The artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`. The
[provenance manifest](provenance/manifest.json) is the authoritative record of the
source commit, frozen execution set, acceptance, execution, and artifact locations.

## Results

<!-- BEGIN GENERATED RESULTS -->
Terminal held-out perplexity at 19,074 updates (10,000,269,312 prediction tokens), seed 0, 200M-parameter softmax reference. The control is the retained old-policy run; both use the same recipe, data order, and evaluation set.

| Weight-decay policy | Terminal NLL | Terminal PPL | Best step | Best PPL | Hardware |
|---|---:|---:|---:|---:|---|
| exclude_embeddings_norms_biases_v1 | 2.685498 | 14.6655 | 19,074 | 14.6655 | H100 dogo to update 1,000 checkpoint, then B300 akita |
| decay_all_parameters | 2.687451 | 14.6942 | 19,074 | 14.6942 | H100 and B300 (see the source record) |

Selective decay changes terminal NLL by -0.001953 nats and PPL by -0.0287 (-0.20%).

[Summary CSV](summary.csv) · [Learning curves](figures/learning_curves.pdf)
<!-- END GENERATED RESULTS -->

## Reading the result

The two runs share the recipe, data order, and evaluation set, and differ only in
which parameters AdamW decays. Their validation curves stay within 0.003 nats of
each other for the first 7,000 updates (the new policy is 0.006 nats behind at
update 1,000, its largest deficit), cross between updates 8,000 and 12,000, and then
the new policy pulls ahead steadily to 0.0020 nats at the end. That shape, a small
late advantage that grows as the learning rate decays, is what removing decay from
the embedding and normalization scales would be expected to give: those parameters
are no longer pulled toward zero once gradients become small.

One seed, so the comparison bounds only the policy's effect at this scale. The two
retained `d_K = 64` softmax runs on different hardware agreed within 0.0003 nats,
which is the scale of run-to-run noise here from hardware alone; the 0.0020-nat
difference clears that by about six times. Both runs also have mixed hardware
histories: the control moved between H100 and B300, and this run trained on H100
to its update-1,000 checkpoint and on B300 from there after a preemption.

For the planned rollout, the policy moves the softmax reference by 0.03 PPL. The
differences the retained records report between mechanisms, widths, and degrees
are tens to hundreds of times larger, so a record rerun under the new policy is
expected to change its numbers at the second decimal of PPL and none of its
orderings; each rerun still has to confirm that for its own mechanism.

## Artifacts

| Artifact | Location in `experiments/decay_policy_baseline/` |
| --- | --- |
| Exact recipe | `configs/200m-softmax-seed0.yaml` |
| Committed metrics, metadata, and completion record | `runs/200m-softmax-seed0/` |
| Summary and figure | `summary.csv`, `figures/learning_curves.{png,pdf}` |
| Canonical manifest | `provenance/manifest.json` |
| Validation-best checkpoint (ignored) | `checkpoints/200m-softmax-seed0/<step>/` |
| Slurm log (ignored) | `logs/200m-softmax-seed0/slurm-202190.out` |

## Reproduction

Regenerate the summary, figure, and results block from the committed runs; no GPU,
dataset, or checkpoint is needed:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m experiments.decay_policy_baseline.adapter.analysis.report
```

The run was submitted from this experiment's worktree root with the shared Kennel
launcher, which checks the frozen execution set against the launch commit, runs
acceptance (CPU tests, distributed CPU tests, cuDNN softmax GPU tests at the production widths,
initialization, a 20-update pilot, and an exact resume audit) in the allocation,
then trains:

```bash
RUN=200m-softmax-seed0
mkdir -p "experiments/decay_policy_baseline/logs/$RUN"
sbatch --job-name="$RUN" --output="experiments/decay_policy_baseline/logs/$RUN/slurm-%j.out" \
  scaffolds/transformer/launchers/kennel.sbatch experiments.decay_policy_baseline.adapter.execution.acceptance \
  "experiments/decay_policy_baseline/configs/$RUN.yaml" "$(git rev-parse HEAD)" 4 B300,H100
```

## Limitations

One seed per policy. Both runs mix GPU types, which the retained softmax pair bounds
at about 0.0003 nats. This run was preempted three times by higher-priority jobs on
a conference-deadline day; each requeue reused its passed acceptance stages or
resumed from the last checkpoint, and the manifest records every attempt. A single
softmax baseline does not show that the policy affects every mechanism equally;
the planned per-experiment reruns test that directly.
