#!/bin/bash
# Time one training microbatch (16 x 2048, synthetic tokens) with the fused and Mosaic
# attention backends on the same GPU, twice each, for study recipes (RUNS; default a query
# tanh and a Taylor recipe) read from RECIPES (default this checkout's attention_scaling runs).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -cs 'A-Za-z0-9' '-' | sed 's/-$//')"
OUT="scaffolds/transformer/validation/runs/mosaic-step/$GPU-$SLURM_JOB_ID"
SCRATCH="$(mktemp -d)"
trap 'rm -rf -- "$SCRATCH"' EXIT
mkdir -p "$OUT"
for round in 1 2; do
  for run in ${RUNS:-r0bc57959ef6c raf32fab4ba52}; do
    for backend in fused mosaic; do
      "$PYTHON" - "$run" "$backend" "$SCRATCH" <<'PY'
import os
import sys
from pathlib import Path
import yaml
run, backend, out = sys.argv[1:]
recipe = yaml.safe_load(Path(os.environ.get("RECIPES", "experiments/attention_scaling/runs"), run, "config.yaml").read_text())
recipe["model"]["attention_backend"] = backend
recipe["train"]["micro_batch_size"] = 16
Path(out, f"{run}-{backend}.yaml").write_text(yaml.safe_dump(recipe))
PY
      label="$backend"
      if [[ "$backend" == mosaic ]]; then
        RUN_ID_FOR_LABEL="$run"
        label="mosaic-"$("$PYTHON" - "$RUN_ID_FOR_LABEL" <<'PY'
import os
import sys
from pathlib import Path
import yaml
from scaffolds.transformer.attention.backends.mosaic.api import default_precision
model = yaml.safe_load(Path(os.environ.get("RECIPES", "experiments/attention_scaling/runs"), sys.argv[1], "config.yaml").read_text())["model"]
family = "taylor" if model.get("attention_normalization", True) else "query"
print(default_precision(family, model["attention_degree"]))
PY
)""
      fi
      "$PYTHON" -m scaffolds.transformer.benchmark "$SCRATCH/$run-$backend.yaml" \
        --output "$OUT/$run-$label-$round.json" --warmup 5 --iterations 30
    done
  done
done
