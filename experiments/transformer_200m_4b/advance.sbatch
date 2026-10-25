#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
if [[ "${1:-advance}" == report ]]; then
  exec scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.pipeline report
fi
exec scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.pipeline advance --submit
