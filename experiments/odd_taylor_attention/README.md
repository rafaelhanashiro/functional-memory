# Fixed odd Taylor attention

This experiment uses the shared Transformer implementation in
[`experiments/shared_transformer`](../shared_transformer/README.md). Its configuration,
sweep, preflight, plots, and retention policy stay in this experiment directory.
Use the shared locked environment for new runs; reproduce historical runs from
the source revision and dependency lock recorded in their provenance.

Two seed-0 models test the fixed Taylor polynomials `p₁(s)=1+s` and
`p₃(s)=1+s+s²/2+s³/6`. They match the completed 199,906,304-parameter
backbone, packed-data order, BF16 computation, 19,074-update learning-rate
schedule, and 10,000,269,312 prediction-token budget. No coefficient parameters
are added. Original fixed-quadratic and exact-softmax controls are referenced
with their original source commits, hardware histories, and SHA256 checksums.
Historical configurations and scientific results are never rewritten.

Scores remain unshifted after RoPE. Both odd polynomials can be negative, and
attention normalizes by the signed final row sum. Negative weights and negative
nonzero denominators are permitted. Zero/nonfinite denominators and nonfinite
computations fail explicitly. No denominator epsilon, clipping, or absolute-value
normalization is introduced. Production attention computes final row sums first,
then recomputes tiles to accumulate normalized weighted values; zero or negative
intermediate tile sums are valid.

Use a dedicated experiment branch and a clean frozen execution checkout distinct
from the retained project root. The locked interpreter is reused from
`PROJECT_ROOT/.envs/shared_transformer/bin/python`. Preview without writes:

```bash
/path/to/project/.envs/shared_transformer/bin/python \
  -m experiments.odd_taylor_attention.sweep \
  --project-root /path/to/project --execution-root /path/to/frozen-checkout --dry-run
```

Use `--launch` to submit the two-hour preflight on four H100s on `dogo`, followed
by independently eligible eight-hour production runs. Use `--preflight-only` to
validate without queuing production. `--submit` requires matching completed
preflight records. Production jobs run serially with `afterany` dependencies,
so numerical failure in one degree does not suppress the other.

Shared preflight checks cover CPU reference and training tests, GPU fused/reference
output and gradient agreement, signed cancellation cases, full-context and
full-microbatch workloads, and exact backbone initialization against the original
fixed-quadratic control. Each degree runs a two-update full-model pilot and a
fresh-process continuation to step three. Exact hashes compare the live state,
durable checkpoint, restored parameters/optimizer state, next training batch,
and RNG. CPU tests additionally check uninterrupted/resumed trajectory agreement.
Numerical or runtime failures reject only that degree; correctness or
infrastructure failures block production. Runtime estimates include evaluation,
checkpointing, three startup allowances, and 10% headroom against the eight-hour
allocation limit.

CPU checks have separate mandatory gates: ordinary attention/training tests run
without distributed tests, and distributed accumulation tests run in a fresh
process with two forced CPU devices. Every gate rejects failures or skipped
tests; distributed tests require both degree variants to execute.

Each production job first trains to update 5,000 with the full 19,074-update
schedule and unchanged validation cadence. It writes `stability_stage.json`,
requires finite committed step-5,000 validation, then independently restores and
hashes the state and actual next batch/RNG. A successful `stability_gate.json`
authorizes immediate continuation to update 19,074 in the same allocation.
`production_failure.json` and numerical-failure records distinguish failed runs
from terminal results. A requeue resumes the same run and preserves the gate.
The gate tests finiteness and restart integrity; it imposes no new loss threshold
or scientific hyperparameter.

Resolved configurations live under `experiments/odd_taylor_attention/configs/`.
Scientific results live under `results/odd_taylor_attention/<run>/`. Checkpoints
live outside the temporary checkout under
`experiments/odd_taylor_attention/checkpoints/<run>/`, with a compatibility symlink
in the result directory. Successful completion retains only the validation-best
checkpoint, choosing the latest on an exact tie, and preserves terminal metrics
separately. Retain the successful Slurm log matching metadata and remove
superseded debugging logs after resolution. Checkpoints, datasets, and caches
must remain ignored by Git.

Regenerate the four-way report with:

```bash
python -m experiments.odd_taylor_attention.plot \
  --manifest /path/to/project/results/odd_taylor_attention/sweep.json
```

The report includes terminal NLL/perplexity for completed runs, failure steps,
learning curves, score ranges, negative normalized-weight fractions, minimum
absolute denominators, cancellation ratios, and throughput. Diagnostics use the
original fixed held-out diagnostic sample. One seed does not estimate seed
variation; historical throughput reflects its original hardware allocations.
Keep the experiment separate from `main` until completion and scientific review,
then clean temporary outputs and retain indispensable artifacts outside the
checkout before merging and removing the branch/worktree.

Scaffold checks (no real job submissions):

```bash
JAX_PLATFORMS=cpu python -m pytest -c experiments/shared_transformer/pyproject.toml \
  experiments/odd_taylor_attention/tests -q
```
