# Concurrent hardware execution amendments

The original sweep manifest, training source, configurations, dependencies, data,
seeds, and token budgets remain authoritative. This adapter adds a separately
hashed execution policy and records its own frozen source commit. Invoke these
scripts directly. Original workers import from `training_root`; an explicitly
qualified runtime amendment instead executes its separately frozen `runtime_root`.
Workers never import training code from the orchestration checkout.

On 2026-09-21 the user requested concurrent B300 allocations alongside H100s.
Remaining lower-degree coordinates are assigned to four B300s on `akita` in
`sweep`; the other pending coordinates retain four H100s on `dogo` in `standard`.
Both use eight-hour allocations, the unchanged per-device microbatch, and 20%
runtime headroom. Different coordinates have no scheduler dependencies. Only
successive checkpoint segments of the same coordinate depend on one another.
Slurm can therefore run multiple B300 allocations if resources permit, alongside
the H100 allocation. The `sweep` partition is preemptible; requeued jobs validate
and resume their durable optimizer state, batch cursor, and RNG.

`schedule.py --policy PATH --prepare MANIFEST --b300-names NAME,NAME,...`
creates an immutable amendment for pending, unstarted coordinates. The existing
running allocation remains unchanged. Then `--migrate` holds the pending legacy
chain, verifies no training artifacts exist, cancels only those pending jobs,
and preserves their IDs in a migration ledger. `--submit h100` queues the
independently qualified H100 coordinates. `--submit qualification` starts the
B300 hardware qualification and automatically submits eligible B300 production
afterward. `--submit b300` is also an idempotent recovery operation.

B300 qualification reruns all 105 original GPU correctness tests, including
FP64 comparisons and production tensor shapes. Each assigned configuration gets
a full-model 32-update pilot, evaluation, checkpoint, and fresh-process update 33.
Exact restored state, next input, and RNG audits are required. Pilot timings use
the original runtime estimator and 500-update segment boundaries. Production
starts from scratch; pilot checkpoints are removed after evidence is retained.
Numerical or runtime ineligibility is recorded per coordinate and does not
suppress other qualified coordinates. Infrastructure failures are separate and
retain debugging evidence for recovery.

Submissions start held. A durable scheduler ledger, canonical manifest job IDs,
and launch history are reconciled before release. Unique Slurm comments recover
orphan held submissions. Per-run inherited file locks and process-group signal
handling exclude duplicate checkpoint writers. Completed runs use the original
validation-best checkpoint and successful-log retention implementation.

Reports are generated under an execution-policy-specific directory with a
renderer lock and staged file replacement. Canonical figures are published only
after the original running launcher's job becomes terminal, so its legacy
renderer cannot race the new renderer. Original scientific source identity stays
unchanged; execution attempts separately record orchestration commit, hardware,
node, partition, job, restart count, and qualification evidence. Hardware can
change floating-point trajectories and throughput, so this execution distinction
must remain explicit in the final scientific report.

The execution amendment and mutable job ledger live under
`results/key_dimension_taylor/`, outside either frozen checkout. Keep both
checkouts until all jobs and possible requeues finish. Complete the existing
review, retention, and merge lifecycle afterward.

## A6000 host-staging amendment

On 2026-09-22 the user requested moving the remaining queued H100 coordinates to
available A6000s. All 20 Cartesian configurations passed full-model 32-update
pilots, evaluation, checkpoint save, and fresh-process update 33 under runtime
`26faf4e00f7d2b6e54055a26343eb7b8f9d379eb`. This runtime enables verified host
placement, disables NCCL peer-to-peer transport, and permits a 90% GPU allocator
pool. Model mathematics, original configurations, and the full training schedule
remain unchanged.

`schedule.py --policy NEW_PATH --prepare-a6000 MANIFEST --a6000-names NAME,NAME,...`
creates a separate immutable execution policy. It references the original pilot
records by path and checksum; it does not rewrite their validation-only policy
as production authorization. `--migrate --submit a6000` transfers ownership only
from the explicitly selected pending jobs, then queues independent four-A6000
chains in `sweep`. Jobs do not pin a hostname. Only segments of the same run have
dependencies. Eight-hour allocations retain 20% timing headroom and use the
pilot's checkpoint-aligned segmentation.

For this amendment, `source_commit` in the unchanged scientific configuration,
metrics, and completion records identifies the original scientific/configuration
source (`4d942ea...`). **The actual executed training checkout is separately
identified by `runtime_commit` and `runtime_root` in execution provenance.**
The orchestration commit is a third, distinct source. Reports must show this
distinction; A6000 execution must not be described as using unchanged training
code. Every attempt records the runtime settings and checksummed qualification.

A6000 pilot estimates for the three migrated coordinates are approximately
19–22 hours each, slower than four H100s. Starting concurrently avoids leaving
the A6000s idle; it does not establish that completing entirely on A6000s gives
the shortest sweep finish time. Durable 500-update checkpoints preserve the
option of a separately recorded handoff to faster hardware. This amendment
does not automatically migrate running jobs or create duplicate training copies.

## Flexible H100/B300 checkpoint handoff

The user subsequently requested stopping the A6000 allocations and queueing the
three unfinished coordinates on any available GPU at least as fast as H100.
The current inventory supports H100 and B300 for this request. All jobs still
require four homogeneous GPUs on one node, the original batch sizes, and the
full 19,074-update schedule.

`fast_queue.py --policy NEW_PATH --prepare MANIFEST --from-policy A6000_POLICY`
records a separate immutable execution policy. `--migrate --submit` first holds
the old continuation jobs and waits for all selected runs to have a finalized
checkpoint at update 500 or later. It cancels only their A6000 chains, waits for
Slurm termination and release of the per-run execution locks, and records each
checkpoint handoff before releasing replacement jobs. Checkpoints, scientific
results, and old execution histories remain in their canonical locations.

Each replacement is a single independent eight-hour job in `sweep`, requesting
`--gres=gpu:4` with exclusions for A100/A6000 nodes. This admits Dogo and Mastiff
H100s and Akita B300s; Mastiff may become usable if repaired. No hostname is
required, and there is no cross-run dependency or fixed future start time.
Jobs are submitted longest projected H100 runtime first. Slurm chooses actual
start times and hardware based on availability, priority, and other users.

H100 qualification already exists for every selected coordinate. While a
specific B300 qualification is missing, that job also excludes Akita, retaining
immediate H100 eligibility. A temporary CPU-only controller
`JAX_PLATFORMS=cpu .../fast_queue.py --policy NEW_PATH --watch` checks every 30
seconds and removes Akita from each pending job's exclusions when that exact
coordinate passes its B300 pilot. It never changes a running job or waits for
unrelated coordinates. It remains available for up to 24 hours if an H100 job
could be preempted and requeued before B300 eligibility is recorded, and exits
once no queued or requeueable job needs that update. `.watch.json` records its
PID and status. Keep its small log until
the handoff is resolved, then remove temporary controller outputs at retention.

At allocation time, the worker detects H100 or B300, verifies that family's
original qualification and checksums, and records the actual hardware. The fast
worker executes the original frozen scientific checkout and clears the A6000
host-staging/NCCL/allocator overrides. It independently restores and audits model
and optimizer arrays, data cursor, and RNG before continuing from the retained
checkpoint. The subsequent floating-point trajectory may differ across hardware;
the earlier A6000 execution remains explicitly recorded in provenance. Training
metrics preserve the original scientific source and common terminal budget.

The B300 qualification records remain under their original validation-only
policy and worker commit. The fast execution policy references those records
without relabeling their source or modifying frozen checkouts. A failed B300
pilot leaves the coordinate eligible for H100. The three fast jobs each fit one
allocation at the pilot's 80% wall-time budget; no A6000 continuation jobs remain.
