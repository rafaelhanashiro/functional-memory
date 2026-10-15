# State-rank scaffold qualification

This directory records software correctness, hardware qualification, and runtime
benchmarks for the reusable scaffold. Scientific measurements and findings belong
to the experiment studies and result collections.

## Frozen-head capture and sparse polynomial fitting

No GPU qualification result has been recorded yet. Before production, run the
qualification in the experiment's own locked environment on one Kennel H100:

```bash
mkdir -p scaffolds/state_rank/validation/logs
sbatch scaffolds/state_rank/validation/qualify_sparse_heads.sbatch \
  <committed-launch-source> <prepared-corpus-directory> \
  <manifest-sha256> <tokens-sha256>
```

The launcher requests one H100 on `dogo` in `debug` for at most two hours. It
checks the frozen attention-study and state-rank execution set on every start,
also checks the qualification scripts against the launch commit, verifies the
frozen corpus checksums, and uses the worktree's own environment. Cached pretrained
files must already exist; the job performs no network downloads. CPU tests and
temporary files execute in the allocation with node-local temporary storage.

The checks include the focused CPU tests, a tiny CUDA Llama capture, the CUDA
feature Gram identity, CPU-FP64/CUDA-FP32 forward and gradient agreement for all
four selection modes, and native head-output reconstruction for the six selected heads of the pinned FP32
Llama-3.2-1B model on the first two training windows. These are software checks;
the qualification does not fit or score a scientific comparison run.

The benchmark uses one captured head with 512 keys. It times ten synchronized
forward/backward iterations at query batch eight for dense features and fixed,
random, and dynamic selection at 128, 512, and 2,048 features. Optimizer time is
recorded separately. Evaluation uses query batch sixteen and cached key features;
key-feature construction is timed separately. Two warmup updates precede timings.
Outputs, losses, gradients, and updated parameters must remain finite during the
training-safe computations. Each mode saves its model and optimizer, reloads the
model into a fresh module, and must reproduce all returned tensor outputs exactly.

Results, tests, timings, memory peaks, source and data hashes, native-output
checks, and checkpoint replay evidence are written under the ignored
`logs/sparse-head-<job>-<restart>/`. A retained checkpoint in that directory is
qualification evidence, not an experiment checkpoint. Summarize the actual
qualification here once it completes; skipped or failed GPU checks do not qualify
the implementation. Timings cover a single head and 512-token inputs; runtime
estimates must separately include six-head fitting, validation frequency,
longer-context evaluation, capture, and I/O.
