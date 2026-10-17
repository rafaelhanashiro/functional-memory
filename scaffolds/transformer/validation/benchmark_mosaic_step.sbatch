#!/bin/bash
# Time one training microbatch (16 x 2048, synthetic tokens) with the fused and Mosaic
# attention backends on the same GPU, twice each, for a query tanh and a Taylor recipe.
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
OUT=scaffolds/transformer/validation/runs/mosaic-step
SCRATCH="$(mktemp -d)"
trap 'rm -rf -- "$SCRATCH"' EXIT
mkdir -p "$OUT"
for round in 1 2; do
  for run in r0bc57959ef6c raf32fab4ba52; do
    for backend in fused mosaic; do
      "$PYTHON" - "$run" "$backend" "$SCRATCH" <<'PY'
import sys
from pathlib import Path
import yaml
run, backend, out = sys.argv[1:]
recipe = yaml.safe_load(Path(f"experiments/attention_scaling/runs/{run}/config.yaml").read_text())
recipe["model"]["attention_backend"] = backend
recipe["train"]["micro_batch_size"] = 16
Path(out, f"{run}-{backend}.yaml").write_text(yaml.safe_dump(recipe))
PY
      "$PYTHON" -m scaffolds.transformer.benchmark "$SCRATCH/$run-$backend.yaml" \
        --output "$OUT/$GPU-$run-$backend-$round.json" --warmup 5 --iterations 30
    done
  done
done
