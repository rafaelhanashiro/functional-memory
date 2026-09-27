# Factored degree-two Hermite attention

Can the normalized degree-two Hermite model pass qualification and train when its
fixed kernel is evaluated as a square? The earlier [expanded-polynomial run](../hermite_attention/README.md)
failed its first full-shape pilot update. A replay found a score near `-1` where
FP32 evaluation of the expanded polynomial returned zero although the true weight
was positive. The scientific polynomial is unchanged; its FP32 evaluation and
fused normalization arithmetic differ. This experiment uses one 200M model with
seed 0. It is a separate run, not a continuation of the
failed pilot.

**Home cluster: Kennel.** Its worktree and active artifact root are
`/data/cl/u/rafah/home/projects/functional-memory/worktrees/hermite_factored_m2`.
Packed FineWeb data live at `/data/cl/u/rafah/home/datasets/fineweb_12b`.
The [manifest](provenance/manifest.json) records configuration, source identity,
allocation and runtime, acceptance, Slurm attempts, and artifact checksums.

## Method

The probabilists' degree-two Hermite projection under standard-normal weighting is

```text
p_2(s) = exp(1/2) * (s+1)^2 / 2 = [a + a*s]^2 + 0^2
a = sqrt(exp(1/2) / 2) = 0.9079430793557843
s_ij = q_i · k_j / sqrt(d_K)
y_i = sum_(j<=i) p_2(s_ij) v_j / sum_(j<=i) p_2(s_ij)
```

The recipe stores fixed square factors `[a, a, 0]`, in ascending order within
each factor. It has no learned polynomial parameters. It does not supply expanded
coefficients to the attention implementation. The mathematical polynomial is
identical to the failed expanded recipe, while both weight arithmetic and the
normalized row reduction follow the existing SOS one-pass fused backend. Scores use post-RoPE Q and K with
`qk_norm: false`. The denominator is the nonnegative row sum of these weights,
without epsilon, clipping, or an arbitrary `0/0` fallback. Exact zero and
nonfinite denominators remain explicit numerical failures. Training also stops
on nonfinite forward loss, gradient norm, or updated parameters.

The model has 199,906,304 trainable parameters: ten layers, width 1024, 16 full
MHA heads with `d_K=d_V=64`, SwiGLU width 3008, untied 32,000-token embeddings,
context 2048, RoPE base 10000, no dropout, BF16 compute. It uses the same first
packed batch and order as the expanded degree-two run (seed 0), global batch 256,
global microbatch 64, four accumulation steps, AdamW peak/minimum rates
`3e-4`/`3e-5`, 2,000 warmup steps, weight decay 0.1 with
`exclude_embeddings_norms_biases_v1`, and gradient clipping 1. The 19,074 updates
cover 10,000,269,312 prediction tokens. Validation uses seven global batches
every 1,000 updates and at completion; held-out terminal perplexity is the primary
endpoint. Validation-best checkpoint selection is recorded separately.

## Results

<!-- BEGIN GENERATED RESULTS -->
| Variant | Degree | Status | Last update | Terminal PPL | Failure update |
| --- | ---: | --- | ---: | ---: | ---: |
| normalized | 2 | submitted |  |  |  |
<!-- END GENERATED RESULTS -->

The generated table uses only authoritative files under `runs/` for scientific
results. `submitted` indicates a recorded Slurm submission, not necessarily an
actively running job. `incomplete` indicates production metadata without a final
result. A numerical failure during the qualification pilot is recorded as
`acceptance-numerical-failure`, distinct from production and without terminal
perplexity. A concluded production failure records its stopping update, last valid
measurements, diagnostics, and matching Slurm log.

## Acceptance and reproduction

The generic Transformer attention interface selects the existing SOS one-pass
normalized fused implementation from `attention_fixed_square_factors`. CPU and GPU tests exercise
the factorization, including near-root scores, the normalized fused path, and its
gradients. Acceptance also checks backbone initialization against an exact-softmax
counterpart, two full-recipe training updates with evaluation, then a fresh-process
third update with checkpoint-state, next-batch, and RNG auditing. The short pilot
is a safety gate, not evidence of long-run stability. A failure there concludes
qualification; it does not start a 10B-token production run.

The branch `codex/hermite-factored-m2` runs entirely from its dedicated worktree.
After the shared implementation and these recipes are committed, run from that
worktree root:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
JAX_PLATFORMS=cpu PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m pytest \
  -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m 'not gpu' \
  scaffolds/transformer/tests/test_fixed_square_factors.py \
  experiments/hermite_factored_m2/adapter/tests
LAUNCH_COMMIT=$(git rev-parse HEAD)
PYTHONPATH=. scaffolds/transformer/.venv/bin/python \
  -m experiments.hermite_factored_m2.adapter.analysis.report \
  --refresh-provenance --launch-commit "$LAUNCH_COMMIT"
RUN=200m-normalized-factored-m002-seed0
CONFIG=experiments/hermite_factored_m2/configs/$RUN.yaml
mkdir -p "experiments/hermite_factored_m2/logs/$RUN"
sbatch --gres=gpu:2 --nodelist=dogo --exclude=mastiff --job-name="$RUN" \
  --output="experiments/hermite_factored_m2/logs/$RUN/slurm-%j.out" \
  scaffolds/transformer/launchers/kennel.sbatch \
  experiments.hermite_factored_m2.adapter.execution.acceptance \
  "$CONFIG" "$LAUNCH_COMMIT" 2 H100
```

The shared launcher uses Kennel's low-priority `sweep` partition, requeue,
append-mode logs, and a 48-hour limit. The two H100s on `dogo` match the failed
expanded run's allocation and local batch shape of `[32, 2048, 16, 64]`;
submission may queue until two GPUs are available. The manifest records actual
device and node identities, runtime settings, resolved backend, pilot outcomes,
and every requeue. The shared launcher's frozen-execution-set check prevents
changes to code, lock, recipes, or execution adapter while the job can run.

Metrics mirror to the private W&B project
`rafah-massachusetts-institute-of-technology/functional-memory`, group
`hermite_factored_m2`. Run files remain canonical. To consolidate execution
records and regenerate the summary and README table:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python \
  -m experiments.hermite_factored_m2.adapter.analysis.report --refresh-provenance
```

Omit `--refresh-provenance` to regenerate only the derived CSV and README table.
Use `--hash-checkpoints` at final retention to hash the selected checkpoint.
If the pilot stops numerically, retain its evidence only after the Slurm job
concludes:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python \
  -m experiments.hermite_factored_m2.adapter.analysis.report \
  --retain-qualification-failure 200m-normalized-factored-m002-seed0 \
  --failure-job-id "$JOB_ID"
```

## Artifacts and limitations

| Artifact | Location in `experiments/hermite_factored_m2/` |
| --- | --- |
| Exact recipe | `configs/` |
| Authoritative metrics, metadata, completion or failure | `runs/<run>/` |
| Derived summary and README table | `summary.csv`, generated block above |
| Authoritative execution record | `provenance/manifest.json` |
| Checkpoints, ignored | `checkpoints/<run>/<step>/` |
| Matching Slurm log, ignored | `logs/<run>/slurm-<job_id>.out` |
| Disposable preflight artifacts, ignored | `logs/<run>/acceptance/`, `logs/<run>/submissions/` |

While training, retain the latest resumable and validation-best checkpoints.
After completion, retain only the validation-best checkpoint and one matching
successful log; terminal metrics stay in `runs/` regardless of checkpoint step.
Concluded numerical failures retain one matching failure log and no diagnostic
checkpoint unless documented in this README and provenance. Once all jobs have
concluded, move retained artifacts into the primary checkout, verify hashes,
update the manifest, merge the branch, then remove the worktree. Do not remove a
worktree while a job can run or requeue.

The first comparison endpoint is whether the same two-step qualification pilot
passes. If it does, the full run tests whether it reaches scheduled validation
and the 10B-token endpoint with finite metrics. This run isolates the numerical
evaluation form of the fixed degree-two kernel.
It cannot by itself establish that all near-zero denominator or gradient issues
are resolved, or that factorized Hermite attention improves held-out perplexity.
The previous expanded run failed qualification, so it has no production endpoint
for an endpoint comparison. One seed and one GPU type limit generalization.
