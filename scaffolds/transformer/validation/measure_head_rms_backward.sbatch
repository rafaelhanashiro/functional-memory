#!/bin/bash
# Measure the FP32 head RMSNorm backward on run checkpoints (measure_head_rms_backward.py) on the granted GPUs, of any
# type: the blocked backend runs every kernel. One compilation takes minutes on one core and bounds the time, so each
# run list gets its own process; the processes share the GPUs in turn (about 12 GB of GPU memory and one core each):
#
#   sbatch --partition=long --nodelist=<node> --gres=gpu:<n> --cpus-per-task=<lists> --mem=<16 GB per list> \
#     --output=scaffolds/transformer/validation/logs/head-rms-backward-%j.log \
#     scaffolds/transformer/validation/measure_head_rms_backward.sbatch "<runs-a.json> <runs-b.json>" [script options]
#
# Each list's result is logs/head-rms-backward-<job>-<list name>.json, rewritten after every run. No run or FP64
# model starts later than DEADLINE_MINUTES (default 25) into the job, so a 30-minute limit ends cleanly.
# Submit to long, to a node whose idle CPUs and GPUs already fit the request (long can preempt sweep jobs otherwise).
#SBATCH --job-name=head-rms-backward
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
STARTED=$(date +%s)
cd "${SLURM_SUBMIT_DIR:?Submit from the worktree}"
read -r -a LISTS <<< "${1:?Supply the run lists, space separated in one argument}"; shift
# GPUs that fail CUDA initialization (CLUSTERS.md): dogo's H100 at Slurm index 4 and mastiff's at indices 1 and 7.
FAULTY_GPUS="GPU-355f34a2-be0c-ba44-022b-1a056f643b80 GPU-86f64c73-c368-bcfb-49e5-6f24ab5eb7a5"
FAULTY_GPUS+=" GPU-cea82b77-b090-2e61-f425-c4b1608e8739"
GPUS=()
while read -r uuid; do
  case " $FAULTY_GPUS " in *" $uuid "*) echo "Skipping the known faulty GPU $uuid" ;; *) GPUS+=("$uuid") ;; esac
done < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[[ ${#GPUS[@]} -gt 0 ]] || { echo "Every granted GPU is a known faulty one" >&2; exit 3; }
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda,cpu PYTHONUNBUFFERED=1
# The processes share the GPUs: no preallocation, and an equal share of 90% of a GPU's memory as each one's ceiling.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=$(awk -v n="${#LISTS[@]}" -v g="${#GPUS[@]}" \
    'BEGIN { per = int((n + g - 1) / g); printf "%.3f", 0.9 / per }')
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
TASK_TMP="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/head-rms-backward-${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf -- "$TASK_TMP"' EXIT
export TMPDIR="$TASK_TMP"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/head-rms-backward/jax"
OUTPUT="scaffolds/transformer/validation/logs"
mkdir -p "$OUTPUT" "$JAX_COMPILATION_CACHE_DIR"
nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version,compute_cap,memory.total --format=csv
echo "Job $SLURM_JOB_ID on $(hostname), environment $TRANSFORMER_ENVIRONMENT_DIR, source $(git rev-parse HEAD), GPUs ${GPUS[*]}, memory fraction $XLA_PYTHON_CLIENT_MEM_FRACTION, lists ${LISTS[*]}, options $*"
DEADLINE=$((STARTED + 60 * ${DEADLINE_MINUTES:-25}))
PIDS=()
INDEX=0
for list in "${LISTS[@]}"; do
  name=$(basename "$list" .json)
  CUDA_VISIBLE_DEVICES="${GPUS[$((INDEX % ${#GPUS[@]}))]}" \
  "$PYTHON" -m scaffolds.transformer.validation.measure_head_rms_backward --runs "$list" \
      --output "$OUTPUT/head-rms-backward-${SLURM_JOB_ID}-${name}.json" --deadline "$DEADLINE" "$@" \
      > "$OUTPUT/head-rms-backward-${SLURM_JOB_ID}-${name}.log" 2>&1 &
  PIDS+=($!)
  INDEX=$((INDEX + 1))
done
STATUS=0
for pid in "${PIDS[@]}"; do wait "$pid" || STATUS=$?; done
for list in "${LISTS[@]}"; do
  name=$(basename "$list" .json)
  echo "=== $name"; grep -v 'probe RMS' "$OUTPUT/head-rms-backward-${SLURM_JOB_ID}-${name}.log" | tail -n 30
done
echo "Finished after $(( $(date +%s) - STARTED )) s with status $STATUS"
exit "$STATUS"
