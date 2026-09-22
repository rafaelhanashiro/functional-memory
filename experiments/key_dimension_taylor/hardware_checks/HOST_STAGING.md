# A6000 transfers and qualification

On `doodle`, direct GPU peer communication hung, and copies of arrays computed
on GPU 0 delivered stale values to other GPUs. Disabling NCCL peer transfers
fixed collective communication but did not fix JAX array placement. This caused
an incorrect initial training loss and a nonfinite held-out evaluation during
the original A6000 pilot. The initial model itself matched the H100 state hash.

Set both execution options for the A6000 compatibility checks:

```text
NCCL_P2P_DISABLE=1
POLYNOMIAL_ATTENTION_HOST_STAGING=1
```

Host staging places initialized/restored parameters, optimizer state, input
batches, and RNG arguments directly from host memory on their target GPUs.
Every physical state replica is checked byte-for-byte before training starts or
resumes. This changes transfer routing only: model arrays, BF16 computation,
attention kernels, optimizer, data order, clipping, and learning-rate schedule
remain unchanged. The default execution path remains unchanged when the host
staging option is absent or `0`.

`host_staging.sbatch` runs only short compatibility checks. Set
`HOST_STAGING_RUNTIME_ROOT` to a clean, frozen checkout containing this fix and
`HOST_STAGING_PYTHON` to the pinned polynomial-attention Python executable. Pass:

```text
--policy /absolute/path/to/results/key_dimension_taylor/hardware_checks/policy.json
--runtime-commit <full commit of the frozen checkout>
--names 200m-dk008-m002-seed0
--output-root /absolute/path/to/results/key_dimension_taylor/hardware_checks/a6000_host_staging/<shard>
```

Use separate output subdirectories for concurrent shards. Each selected pair
starts from scratch for 32 updates, evaluates, and saves a checkpoint. A fresh
process restores that checkpoint and completes update 33. The full 19,074-update
schedule remains in force. Both child processes have a 15-minute deadline;
failed coordinates do not prevent subsequent coordinates from being checked.

Records preserve the original scientific source and policy separately from the
fixed runtime commit and environment. The original 105-case A6000 kernel gate
is reusable only while both attention implementation files are byte-identical.
For available H100 pilots, initial state, first input batch, and RNG fingerprints
must match. State/data/RNG continuity across the restart must be exact.

Successful checks include synchronized timing and an eight-hour allocation plan
with 20% headroom, split at 500-update boundaries when necessary. Temporary
checkpoints are removed after recording evidence. The launcher never submits
production training or changes the original validation policy.
