#!/bin/bash
# Train a PaTH study recipe for 1,000 updates on one GPU with a chosen attention backend
# (blocked: the JAX implementation; fused: FLA's kernels), from the recipe's initialization
# and data, so the two implementations' loss curves and update times can be compared.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
LAUNCH_COMMIT="${1:?Expected committed source}"
GPU_TYPE="${2:?Expected H100 or B300}"
RUN_ID="${3:?Expected an attention_scaling run ID}"
BACKEND="${4:?Expected blocked or fused}"
STEPS="${5:-1000}"
REPEAT="${6:-}"  # optional label for a repeated run of the same backend (run-to-run noise)
MICRO="${7:-}"   # optional microbatch override (the global batch is unchanged)
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment attention_scaling --scaffold transformer
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
# The custom-call bridge executes outside XLA command buffers (acceptance's runtime for fused PaTH).
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer="
NAME="${GPU_TYPE,,}-$RUN_ID-$BACKEND${REPEAT:+-$REPEAT}"
WORKDIR="$PWD/.cache/path-pilot/$NAME"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/path-pilot/jax" EJKERNEL_TRITON_CACHE_DIR="$PWD/.cache/path-pilot/triton"
# A requeue resumes from the latest checkpoint (every 100 updates); a fresh start clears old output.
if [[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]; then rm -rf "$WORKDIR"; fi
mkdir -p "$WORKDIR" scaffolds/transformer/validation/runs/path-pilot
"$PYTHON" - "$RUN_ID" "$BACKEND" "$WORKDIR" "$MICRO" <<'PY'
import sys
from pathlib import Path
import yaml
run_id, backend, workdir, micro = sys.argv[1:]
recipe = yaml.safe_load(Path(f"experiments/attention_scaling/runs/{run_id}/config.yaml").read_text())
recipe["model"]["attention_backend"] = backend
recipe["train"].update(workdir=workdir, checkpoint_dir=f"{workdir}/checkpoints", eval_every=100, ckpt_every=100,
                       wandb_mode="disabled", wandb_run_name=None)
if micro:
    recipe["train"]["micro_batch_size"] = int(micro)
Path(workdir, "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
PY
"$PYTHON" -m scaffolds.transformer.train "$WORKDIR/recipe.yaml" --no-resume --stop-after-step "$STEPS" \
  --wandb-mode disabled
"$PYTHON" - "$WORKDIR" "$NAME" "$RUN_ID" "$BACKEND" "$LAUNCH_COMMIT" <<'PY'
import json, os, sys
from pathlib import Path
import jax
import numpy as np
import yaml
workdir, name, run_id, backend, commit = sys.argv[1:]
micro = yaml.safe_load(Path(workdir, "recipe.yaml").read_text())["train"]["micro_batch_size"]
rows = [json.loads(line) for line in Path(workdir, "metrics.jsonl").read_text().splitlines()]
validation = [r for r in rows if r.get("event") == "validation"]
keys = ("step", "validation_nll", "train_nll_sample_mean", "train_gradient_norm_mean", "train_gradient_norm_max",
        "train_gradient_clipped_fraction", "tokens_per_second", "training_interval_seconds", "training_window_start_step")
# Seconds per update over each validation interval; the first includes compilation.
per_update = [r["training_interval_seconds"] / (r["step"] - r["training_window_start_step"])
              for r in validation[1:] if r.get("training_interval_seconds") and r["step"] > r["training_window_start_step"]]
summary = {
    "recipe": f"experiments/attention_scaling/runs/{run_id}/config.yaml", "attention_backend": backend,
    "overrides": {"gpu_count": 1, "eval_every": 100, "ckpt_every": 100, "micro_batch_size": micro}, "source_commit": commit,
    "slurm_job_id": os.environ["SLURM_JOB_ID"], "node": os.environ["SLURMD_NODENAME"],
    "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", 0)),
    "device_kinds": [d.device_kind for d in jax.local_devices()],
    "runtime_environment": {key: os.environ.get(key) for key in (
        "TRANSFORMER_CUDA_ENVIRONMENT", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE")},
    "seconds_per_update_median": float(np.median(per_update)) if per_update else None,
    "validation": [{key: r.get(key) for key in keys} for r in validation],
}
Path("scaffolds/transformer/validation/runs/path-pilot", f"{name}.json").write_text(
    json.dumps(summary, indent=1, sort_keys=True) + "\n")
PY
rm -rf "$WORKDIR/checkpoints"
