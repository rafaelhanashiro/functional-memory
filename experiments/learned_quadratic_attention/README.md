# Learned quadratic attention

This sweep compares four seed-0 learned quadratic kernels against the completed
fixed Taylor quadratic and exact-softmax controls. All runs use the original
200M backbone, packed-data order, BF16 computation, 19,074 updates, and
10,000,269,312 prediction tokens. The controls are referenced with their original
source commits and SHA256 checksums; their configs and results are never rewritten.

| Family | Polynomial | Initial parameters | Sharing | Added parameters |
|---|---|---|---|---:|
| Positive | `0.5 * (1 + (a*s+b)^2)` | `a=b=1` | Global | 2 |
| Positive | `0.5 * (1 + (a*s+b)^2)` | `a=b=1` | Per head, per layer | 320 |
| Unrestricted | `1 + b*s + c*s^2` | `b=1, c=0.5` | Global | 2 |
| Unrestricted | `1 + b*s + c*s^2` | `b=1, c=0.5` | Per head, per layer | 320 |

Coefficients are FP32 parameters, trained with the existing learning-rate schedule
and global gradient clipping, without coefficient weight decay. Both families
start at the existing Taylor polynomial. Scores remain unshifted after RoPE.
Unrestricted kernels allow negative normalized weights and negative nonzero
denominators. Zero/nonfinite denominators and nonfinite computations fail; no
clipping, epsilon, or absolute-value denominator is introduced.

Develop on the dedicated experiment branch and use a **clean, frozen execution
checkout** distinct from the retained project root. The existing locked interpreter
at `PROJECT_ROOT/.envs/polynomial_attention/bin/python` is reused. From the frozen
checkout, first inspect the resolved manifest without writing or submitting:

```bash
/path/to/project/.envs/polynomial_attention/bin/python \
  -m experiments.learned_quadratic_attention.sweep \
  --project-root /path/to/project --execution-root /path/to/frozen-checkout --dry-run
```

Use `--launch` to submit a two-hour preflight on four H100s on `dogo`; it submits
eligible eight-hour production runs serially after completion. `--preflight-only`
stops after validation. `--submit` accepts only matching complete preflight records.
No production job is queued before the shared correctness checks finish. Every
production launcher rechecks its own pilot eligibility and the actual allocation.
Production jobs use `afterany` dependencies after the first run, so a failed run
does not suppress the remaining eligible variants.

Preflight checks CPU reference/gradient/optimizer/resume tests, fused GPU parity,
full-backbone initialization hashes against the fixed control, and coefficient
initialization/counts. Every variant runs full-context, full-microbatch training
and evaluation and an interrupted/resumed GPU pilot. Exact full-state hashes
match the live parameters, optimizer moments, and update count to the durable
checkpoint and the fresh process's restored state. The first resumed batch and
RNG must match the original iterator's next batch and RNG exactly. CPU tests also
require exact uninterrupted/resumed trajectory agreement for all four variants.
Independent GPU executions already differed before any restart in the initial
pilot, including embedding optimizer moments; their terminal hashes therefore
cannot isolate checkpoint fidelity. The audit uses no floating-point tolerance
and does not change production computation to enforce GPU determinism.
Runtime estimates include evaluation,
checkpoint overhead, and 10% headroom against the eight-hour allocation limit.
Numerical or runtime pilot failures reject only that variant; correctness and
infrastructure failures block the entire sweep. Small validation records and test
reports are retained under `results/learned_quadratic_attention/preflight/`.

Resolved rerun configs live under `experiments/learned_quadratic_attention/configs/`.
Scientific results live under `results/learned_quadratic_attention/<run>/`.
Checkpoints are stored outside the temporary checkout under
`experiments/learned_quadratic_attention/checkpoints/<run>/`, with a compatibility
symlink from the result directory. On successful completion the retention command
keeps exactly the validation-best checkpoint, retains terminal metrics separately,
and preserves the successful Slurm log matching metadata. Checkpoints, caches,
datasets, and logs must remain ignored by Git.

Regenerate comparisons with:

```bash
python -m experiments.learned_quadratic_attention.plot \
  --manifest /path/to/project/results/learned_quadratic_attention/sweep.json
```

The six-label report includes terminal NLL/perplexity, learning curves, raw/effective
coefficient records, coefficient trajectories, signed-weight/denominator stability,
score ranges, and throughput. Coefficients must be interpreted with score scales:
Q/K rescaling can absorb some movement. A single seed does not estimate seed
variation. The positive and unrestricted families use different optimization
coordinates, so equal learning rates do not isolate positivity alone.
Keep the experiment separate from `main` until completion and review;
only merge after removing temporary outputs and confirming retained artifacts are
safe outside the checkout.

Scaffold checks (no Slurm submissions):

```bash
JAX_PLATFORMS=cpu python -m pytest -c experiments/polynomial_attention/pyproject.toml \
  experiments/learned_quadratic_attention/tests -q
```
