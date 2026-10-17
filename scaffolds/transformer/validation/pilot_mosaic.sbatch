#!/bin/bash
# Train a study recipe for a few hundred updates on one GPU with a chosen attention backend,
# so Mosaic GPU and fused Triton runs from the same initialization and data can be compared.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
LAUNCH_COMMIT="${1:?Expected committed source}"
GPU_TYPE="${2:?Expected H100 or B300}"
RUN_ID="${3:?Expected an attention_scaling run ID}"
BACKEND="${4:?Expected fused or mosaic}"
STEPS="${5:-1000}"
REPEAT="${6:-}"  # optional label for a repeated run of the same backend (run-to-run noise)
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment attention_scaling --scaffold transformer
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
NAME="${GPU_TYPE,,}-$RUN_ID-$BACKEND${REPEAT:+-$REPEAT}"
WORKDIR="$PWD/.cache/mosaic-pilot/$NAME"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/mosaic-pilot/jax"
# A requeue resumes from the latest checkpoint (every 100 updates); a fresh start clears old output.
if [[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]; then rm -rf "$WORKDIR"; fi
mkdir -p "$WORKDIR" scaffolds/transformer/validation/runs/mosaic-pilot
"$PYTHON" - "$RUN_ID" "$BACKEND" "$WORKDIR" "$STEPS" <<'PY'
import sys
from pathlib import Path
import yaml
run_id, backend, workdir, steps = sys.argv[1:]
recipe = yaml.safe_load(Path(f"experiments/attention_scaling/runs/{run_id}/config.yaml").read_text())
recipe["model"]["attention_backend"] = backend
recipe["train"].update(workdir=workdir, checkpoint_dir=f"{workdir}/checkpoints", micro_batch_size=16,
                       eval_every=100, ckpt_every=100, wandb_mode="disabled", wandb_run_name=None)
Path(workdir, "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
PY
"$PYTHON" -m scaffolds.transformer.train "$WORKDIR/recipe.yaml" --no-resume --stop-after-step "$STEPS" \
  --wandb-mode disabled
"$PYTHON" - "$WORKDIR" "$NAME" "$RUN_ID" "$BACKEND" "$LAUNCH_COMMIT" <<'PY'
import json, os, sys
from pathlib import Path
import jax
workdir, name, run_id, backend, commit = sys.argv[1:]
rows = [json.loads(line) for line in Path(workdir, "metrics.jsonl").read_text().splitlines()]
keys = ("step", "validation_nll", "train_nll_sample_mean", "train_gradient_norm_mean", "train_gradient_norm_max",
        "train_gradient_clipped_fraction", "tokens_per_second")
summary = {
    "recipe": f"experiments/attention_scaling/runs/{run_id}/config.yaml", "attention_backend": backend,
    "overrides": {"gpu_count": 1, "micro_batch_size": 16, "eval_every": 100}, "source_commit": commit,
    "slurm_job_id": os.environ["SLURM_JOB_ID"], "node": os.environ["SLURMD_NODENAME"],
    "device_kinds": [d.device_kind for d in jax.local_devices()],
    "runtime_environment": {key: os.environ.get(key) for key in (
        "TRANSFORMER_CUDA_ENVIRONMENT", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE")},
    "validation": [{key: r.get(key) for key in keys} for r in rows if r.get("event") == "validation"],
}
Path("scaffolds/transformer/validation/runs/mosaic-pilot", f"{name}.json").write_text(
    json.dumps(summary, indent=1, sort_keys=True) + "\n")
PY
rm -rf "$WORKDIR/checkpoints"
