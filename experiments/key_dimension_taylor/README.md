# Joint key dimension × Taylor degree

This sweep trains the Cartesian product `dK = 8,16,32,64,128` and fixed Taylor
degree `m = 2,4,8,16`, reusing the four completed `dK=64` Taylor experiments.
Five completed softmax runs from `key_dimension` provide width-matched
references. There are 16 new training runs and 25 displayed cells.

The model has 10 layers, hidden width 1024, 16 heads, value width 64, MLP width
3008, and untied 32,000-token embeddings. Parameter counts are 181,556,224;
184,177,664; 189,420,544; 199,906,304; and 220,877,824 in ascending key width.
This is a fixed-backbone comparison, not a parameter-matched comparison.
All cells use seed 0; the experiment does not estimate seed variation.

Attention uses the existing polynomial trainer and fused kernel. Scores are
unshifted post-RoPE dot products scaled by the true `1/sqrt(dK)`, including
when the fused implementation pads a small width internally. Coefficients are
the fixed Taylor coefficients. No score clipping or denominator epsilon is
introduced. Odd degrees, learned coefficients, degree zero, and width 256 are
outside this sweep.

The unchanged recipe is 19,074 updates (10,000,269,312 prediction tokens),
context 2048, batch 256, global microbatch 64, four accumulation steps, BF16
computation, FP32 parameters/optimizer, AdamW `(0.9,0.95)`, weight decay 0.1,
global gradient clipping at 1, and learning rate `3e-4` to `3e-5` with 2,000
warmup updates. Evaluation uses the same 1,792 packed sequences, every 1,000
updates and at the endpoint. Checkpoint boundaries are every 500 updates.
Packed-data manifest checksums and resolved configurations are part of the
sweep identity.

## Launch

Use a dedicated, clean, committed checkout as the execution root. Results and
checkpoints live in canonical storage outside that checkout. From the frozen
checkout, using the existing locked polynomial-attention Python environment:

```bash
python -m experiments.key_dimension_taylor.sweep \
  --project-root /absolute/path/to/functional-memory \
  --execution-root /absolute/path/to/functional-memory/worktrees/key_dimension_taylor \
  --key-dimensions 8 16 32 64 128 --degrees 2 4 8 16 --dry-run

python -m experiments.key_dimension_taylor.sweep \
  --project-root /absolute/path/to/functional-memory \
  --execution-root /absolute/path/to/functional-memory/worktrees/key_dimension_taylor \
  --launch
```

`--launch` submits preflight and automatically submits independently eligible
production runs after validation. `--preflight-only` stops after qualification;
`--submit` submits from the saved, qualified manifest. Pair-qualified names
such as `200m-dk008-m002-seed0` prevent collisions with older sweeps.

Validation checks all selected kernel pairs against an independent FP64
reference, including gradients, causal masking, partial tiles, small-width
padding, negative scores, and degree-16 arithmetic. GPU checks also use full
production per-device Q/K shapes `(16,2048,16,dK)` and V shape
`(16,2048,16,64)`. Each new model must pass 32 full-model updates, evaluation,
checkpoint save, and a fresh-process continuation to update 33. Pilots preserve
the full training schedule; production starts from scratch. Checkpoint audits
compare parameter and optimizer bytes, data batches, and RNG across restart.

Production uses four H100s on `dogo`, serial jobs, and eight-hour allocations.
Synchronized pilot timings include startup, evaluation, and checkpointing.
Only 80% of the allocation is budgeted. Runs that need splitting stop at the
largest 500-update boundary that fits and resume without resetting the schedule
or token count. Numerical failure of one configuration does not suppress other
eligible configurations. Numerical and infrastructure failures have distinct
structured records.

## Results and provenance

Scientific outputs live under `results/key_dimension_taylor/`; configurations
and validation-best checkpoints live under `experiments/key_dimension_taylor/`.
Historical artifacts remain at their original paths, with their original
source commits and checksums. References are validated before reuse; unavailable
historical files are explicitly recorded rather than attributed to this run.
The softmax reference at `dK=64` is **14.689767074685026** from `key_dimension`.
The separate polynomial-sweep softmax result **14.69417664861716** is documented
but never substituted or averaged into the grid.

```bash
python -m experiments.key_dimension_taylor.plot \
  --manifest /absolute/path/to/functional-memory/results/key_dimension_taylor/sweep.json
```

The reporter exports annotated terminal-PPL and width-matched delta-PPL
heatmaps as PNG/PDF, learning curves, `summary.csv`, and a results `README.md`. It accepts
only committed terminal metrics at update 19,074. Pending, incomplete, and
failed cells receive labels instead of imputed PPL. The softmax column is
visually separated but shares the linear PPL color scale; historical cells
are marked with an asterisk.

Completed new runs retain one validation-best checkpoint (latest on an exact
tie), terminal metrics independently of checkpoint selection, and one matching
successful Slurm log. Segment histories and failures remain structured records.
Temporary pilots, caches, superseded logs, and intermediate checkpoints are
removed after validation/completion. Keep the experiment branch separate until
the completed results are reviewed; then follow the repository retention and
merge lifecycle.
