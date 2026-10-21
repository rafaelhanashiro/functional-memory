# Cluster reference

Last checked: **2026-09-22**. Kennel details combine the supplied wiki PDFs with
live Slurm checks; ORCD details come from its official documentation, without a
live account check. Recheck limits before launching a large sweep.

Each experiment has one home cluster. Record its cluster and artifact paths in
provenance; Git shares code and small results, not checkpoints or datasets.
See [AGENTS.md](AGENTS.md) for retention and migration rules.

## Kennel (CSAIL)

- **Login:** `ssh rafah@kennel.csail.mit.edu`. The login node has no GPUs;
  run experiments through Slurm, not directly on compute nodes.
- **Repository:** `/data/cl/u/rafah/home/projects/functional-memory`.
- **Datasets:** `/data/cl/u/rafah/home/datasets` (alongside `projects`);
  packed FineWeb is in `fineweb_12b`.
- **Storage:** keep persistent files on NFS under `/data/cl/u/rafah`.
- **Artifact root:** the primary checkout; retained checkpoints and logs live in
  `experiments/<study>/runs/<run_id>/` there.
  Avoid AFS for jobs because authentication expires. `/data/cl/scratch` is
  expendable; node-local `/scratch` is for temporary, I/O-heavy data.
  Current training configs read FineWeb from NFS; stage locally when needed.
- **Quota:** personal quota and backup coverage are unverified. `df` reports
  filesystem free space, not a guaranteed personal allocation.

| Partition | Time limit | Use |
| --- | --- | --- |
| `debug` | 2 hours | Quick checks; at most 2 GPUs per user |
| `standard` | 24 hours | Normal jobs; at most 4 GPUs per user |
| `long` | 7 days | Longer/larger jobs; avoid monopolizing the cluster |
| `sweep` | 7 days | Low-priority sweeps using spare capacity |

GPU types currently include A6000, A100, H100, and B300; use `sinfo` to inspect
nodes before choosing hardware. On 2026-09-22 they were: `akita` 8×B300;
`dogo` and `mastiff` 8×H100 each; `collie` 2×A100 and `samoyed` 10×A100;
`doodle` 9×A6000 and `terrier` 10×A6000. Nodes define no Slurm features, so
accept either B300 or H100 by requesting `--gres=gpu:N` and excluding the
A100/A6000 nodes; A6000 training is markedly slower in this project's tests. General priority is `debug > standard > long > sweep`.
Use periodic checkpoints and `--requeue`. Live `standard`/`long` grace time is
60 seconds (the supplied PDFs say 120); `sweep` has no grace period.
The Transformer Kennel launcher requeues ten minutes before the time limit and
resumes the latest checkpoint. For `standard`, submit with
`TRANSFORMER_ALLOCATION_HOURS=24` and `--time=24:00:00`; acceptance checks that
durable checkpoint progress fits each allocation and records the total projected
training time across allocations. Clear inherited `LD_LIBRARY_PATH` at submission
so system CUDA libraries cannot override the locked environment's wheels.

### Kennel transfer policy

The shared launcher helper `scaffolds/transformer/launchers/kennel_runtime.sh`
selects `TRANSFORMER_HOST_STAGING=1` and `NCCL_P2P_DISABLE=1` for A100/A6000
allocations. A6000 also defaults to `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`.
Explicit environment settings can override these choices for isolated diagnostics.
The shared runtime owns placement and replica verification; model and optimizer
code do not select hardware policy.

Kennel launchers default to `XLA_PYTHON_CLIENT_PREALLOCATE=false` and preserve
explicit allocator settings supplied with `sbatch --export`. Acceptance records
the resolved preallocation setting and memory fraction and qualifies that runtime.

Kennel launchers put disposable compiler and test-capture temporary files in a
unique job directory under `SLURM_TMPDIR` or `/tmp`, and remove it at exit.
Using NFS for these files can leave processes waiting on NFS RPCs with an idle GPU.
Recipes, logs, checkpoints, environments, and persistent compilation caches stay
in the worktree on NFS.

Node CUDA toolkits differ: on 2026-09-24 `dogo` had CUDA 12.6 in `/usr/local/cuda`
and `akita` CUDA 13.0. jaxlib assembles Triton custom-call PTX with that `ptxas`
unless `CUDA_HOME` names another toolkit; `scaffolds/transformer/launchers/kennel.sbatch`
points it at the toolkit inside the selected environment (`nvidia/cuda_nvcc`, CUDA 12.9,
or `nvidia/cu13`, CUDA 13.2), since 12.6 rejects Triton's PTX. Triton 3.7 picks its PTX
version from its bundled assembler (CUDA 13.1 for Blackwell, PTX 9.1), which CUDA 12.9's
`ptxas` rejects, so the launcher also sets `TRITON_PTXAS_PATH` and
`TRITON_PTXAS_BLACKWELL_PATH` to the environment's `ptxas`.

cuBLAS 12.8–13.1 has an NVIDIA bug on Blackwell: some kernels free tensor memory twice,
which can silently corrupt data when a cuBLAS kernel runs concurrently with another
kernel. B300 jobs therefore use a second environment from the same lock: `.venv-cuda13`
(extra `cuda13`: JAX's CUDA 13 plugin, CUDA 13.2, cuBLAS 13.2.2.2, and the same cuDNN
9.22.0.52 as the CUDA 12 environment). `kennel_runtime.sh` selects it on B300 and the
CUDA 12 `.venv` on every other GPU, and runs record the choice as
`TRANSFORMER_CUDA_ENVIRONMENT`; a run requeued between B300 and H100 changes environment
with its hardware. CUDA 13 needs driver >= 580: `akita` had 580.126.20 on 2026-10-15,
while `dogo` had 560 (CUDA 12.6) on 2026-10-04, so H100 stays on CUDA 12, where the bug
does not apply. Sync both environments in every worktree that may run on B300:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
UV_PROJECT_ENVIRONMENT=$PWD/scaffolds/transformer/.venv-cuda13 \
  uv sync --project scaffolds/transformer --locked --extra cuda13 --link-mode hardlink
```

On CUDA 12, B300 runs stopped on infinite gradient norms at finite losses, also with
the gradient all-reduce on the compute stream (`--xla_gpu_disable_async_collectives=ALLCOLLECTIVES`,
still set on B300 at no measurable cost) and on a single GPU; a checkpoint that failed
twice on CUDA 12 replayed cleanly twice on CUDA 13. On B300 the cuDNN softmax rejects
width 256 with `d_qk != d_v`, and FLA's Gated DeltaNet kernel crashes at width 128 in
float32 in either environment. Evidence is in
[`scaffolds/transformer/validation/README.md`](scaffolds/transformer/validation/README.md#b300-cuda-13-environment).

The retained joint-sweep checks cover all 20 B300 and host-staged A6000
configurations and 16 new H100 configurations. On `samoyed`, direct peer transfers
corrupted replicas and batches; the host-staged A100 pilot passed training,
evaluation, and fresh-process resume for `d_K=8,m=4`. This does not qualify the
remaining A100 grid or a new attention mechanism. Evidence and limitations are
recorded in `scaffolds/transformer/validation/README.md`.

## ORCD / Engaging (MIT)

- **Login:** `ssh rafah@orcd-login.mit.edu`, using MIT authentication/Duo.
  See the [SSH guide](https://orcd-docs.mit.edu/accessing-orcd/ssh-login/).
- **Paths:** record the current checkout and dataset paths in each experiment.
  Packed FineWeb is at `/home/rafah/projects/ofm/datasets/fineweb_12b`; its train
  and eval manifests match Kennel's `fineweb_12b` byte for byte (checked 2026-09-28).
- **Execution storage:** `/home/rafah` has held training checkpoints to its quota
  before. Put experiment worktrees under `/orcd/pool/006/rafah/functional_memory/worktrees/`
  with `UV_CACHE_DIR=/orcd/pool/006/rafah/.cache/uv`, so hardlinked environments
  and checkpoints share the pool filesystem.
- **Artifact root:** `/orcd/pool/006/rafah/functional_memory/artifacts`, outside any
  worktree, because the primary checkout (`/home/rafah/projects/functional_memory`)
  is on `/home`. Retained checkpoints and logs live in
  `experiments/<study>/runs/<run_id>/` there; it shares the worktrees' filesystem, so
  completion moves are renames.

| Storage | Documented default quota | Intended use / backup |
| --- | --- | --- |
| `/home/rafah` | 200 GB | Code and important files; snapshots |
| `/home/rafah/orcd/pool` | 1 TB | Larger datasets; no backup |
| `/home/rafah/orcd/scratch` | 1 TB | Active-job data; no backup |

Check **your actual usage and quota** with `cat ~/orcd/.quota` (updated every
30 minutes). Scratch may be deleted after six months without logging in;
keep retained checkpoints in durable storage with a backup plan.
Source: [filesystems](https://orcd-docs.mit.edu/filesystems-file-transfer/filesystems/).

| GPU partition | Documented time limit | Base GPU allowance |
| --- | --- | --- |
| `mit_normal_gpu` | 6 hours | 2 GPUs |
| `mit_preemptable` | 48 hours | 4 GPUs; checkpoint and use `--requeue` |

Documented GPU types include L40S, A100, H100, and H200, depending on partition.
Account/group entitlements may differ; verify before submitting. Live QOS limits
on 2026-09-28 (`sacctmgr show qos`) were per user across all jobs: 2 GPUs and 32
CPUs in `mit_normal_gpu`, 4 GPUs in `mit_preemptable`. H200 nodes have 8 GPUs,
120 CPUs, and 2 TB of memory; nodes define no GPU-type features. Sources:
[scheduler](https://orcd-docs.mit.edu/running-jobs/overview/),
[GPU requests](https://orcd-docs.mit.edu/running-jobs/requesting-resources/).
[Full documentation](https://orcd-docs.mit.edu/); help: `orcd-help-engaging@mit.edu`.

### ORCD launch policy

`scaffolds/transformer/launchers/orcd.sbatch` requests 2×H200 in either
`mit_preemptable` or `mit_normal_gpu`, whichever frees first. Because the caps are
per user, 2-GPU jobs can hold three allocations (six GPUs) at once, while 4-GPU jobs
fit only in `mit_preemptable`. Every job asks for 12 hours with
`--time-min=06:00:00`, so a whole 200M run fits one `mit_preemptable` allocation, and
Slurm's backfill can still start the job in `mit_normal_gpu`, trimming its limit to
that partition's 6-hour maximum (on 2026-10-05 a 12-hour request with a lower
`--time-min` started there with a 6-hour limit). Ten minutes before its limit a job
requeues itself and resumes from its latest checkpoint, losing at most one checkpoint
interval. `mit_preemptable` preempts without grace time (`GraceTime=0`) whenever a
higher-tier job needs the node, which also loses up to one checkpoint interval;
`mit_normal_gpu` never preempts. On 2026-10-08 and 09, GPU 0 of `node3009` and of
`node3209` (8×H100 each) ran at 90–93°C, throttled to 345 MHz, and slowed 2-GPU jobs
about 3.5×; exclude both until they are repaired. The 2×H100 NVL nodes (`node1708`,
`node1709`) train the 200M recipe about 1.6× slower than H100 80GB HBM3.
A requeue or preemption resets the job's submit time and with it its queue age, so
later runs of the same sweep can overtake it. H200 uses direct device transfer. Order a
sweep's runs with `sbatch --nice`: 1000 priority points equal about 8 hours of queue age.

Acceptance shares passed test stages through the worktree's
`.cache/<study>/acceptance-tests/`, so a sweep runs each selection of CPU and GPU tests
once per source commit, environment, runtime, and GPU model (H100 SXM and NVL apart);
every run still performs its own initialization check, pilot, and resume in its
allocation, and repeats them when a requeue lands on another GPU model.

## Everyday use

```bash
sinfo -o '%P %l %G'                         # Partitions, time limits, GPU types
scontrol show partition standard           # Kennel; substitute an ORCD partition there
squeue -u "$USER"                           # Your queued/running jobs
sacct -j JOB_ID -o JobID,State,Elapsed,ExitCode
```

Submit from the experiment's worktree after syncing that worktree's own environment
(`uv sync --project scaffolds/<capability> --locked --link-mode hardlink`). Environments
and caches stay outside Git; never modify an environment while jobs use it. Transformer
experiments submit `scaffolds/transformer/launchers/kennel.sbatch` with their study’s
acceptance module; each README gives the exact command.

Both Transformer launchers end a job whose `training.log` has not grown for 20 minutes
(set `TRANSFORMER_STALL_MINUTES` when submitting to change it), well above normal
silences such as the first update's compilation (at most 3.5 minutes, for Gated
DeltaNet). A GPU that fails, for example by running out of memory, can otherwise leave
its data-parallel peer waiting in a collective until the time limit, and a rare kernel
deadlock can freeze a GPU. The job logs `nvidia-smi` state, exits with code 124, and
requeues; the run resumes from its latest checkpoint and its record lists the stalled
attempt. A run's third stall is recorded as `failed` with a stall error and not requeued,
since repeated stalls point to a fault a requeue may meet again: resubmit it, excluding
the node if its GPUs were at fault.

To move an experiment, transfer required artifacts separately (for example with
`rsync`), verify checksums, recreate the environment, and update provenance before
resuming. Keep the original artifacts until the transferred copy is verified.
