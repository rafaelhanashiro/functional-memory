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

### Kennel transfer policy

The shared launcher helper `scaffolds/transformer/launchers/kennel_runtime.sh`
selects `TRANSFORMER_HOST_STAGING=1` and `NCCL_P2P_DISABLE=1` for A100/A6000
allocations. A6000 also defaults to `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`.
Explicit environment settings can override these choices for isolated diagnostics.
The shared runtime owns placement and replica verification; model and optimizer
code do not select hardware policy.

Node CUDA toolkits differ: on 2026-09-24 `dogo` had CUDA 12.6 in `/usr/local/cuda`
and `akita` CUDA 13.0. jaxlib assembles Triton custom-call PTX with that `ptxas`
unless `CUDA_HOME` names another toolkit; `scaffolds/transformer/launchers/kennel.sbatch`
points it at the locked environment's `nvidia/cuda_nvcc` (CUDA 12.9), since 12.6
rejects Triton 3.6's PTX.

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
fit only in `mit_preemptable`. Every job asks for 6 hours, the `mit_normal_gpu`
maximum. Ten minutes before the limit it requeues itself and resumes from its latest
checkpoint, so it loses at most one checkpoint interval. H200 uses direct device
transfer. Order a sweep's runs with `sbatch --nice`: 1000 priority points equal about
8 hours of queue age.

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

To move an experiment, transfer required artifacts separately (for example with
`rsync`), verify checksums, recreate the environment, and update provenance before
resuming. Keep the original artifacts until the transferred copy is verified.
