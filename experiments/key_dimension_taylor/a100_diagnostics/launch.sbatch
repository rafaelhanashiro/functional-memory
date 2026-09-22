#!/bin/bash
set -euo pipefail
ROOT=/data/cl/u/rafah/home/projects/functional-memory/worktrees/a100_diagnostics
PYTHON=/data/cl/u/rafah/home/projects/functional-memory/.envs/polynomial_attention/bin/python
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
cd "$ROOT"
exec "$PYTHON" experiments/key_dimension_taylor/a100_diagnostics/probe.py "$@"
