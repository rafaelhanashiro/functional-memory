#!/bin/bash
# Kennel resource policy stays in the launcher, outside model/training code.
set -euo pipefail
EXECUTION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$EXECUTION_ROOT"
source "$EXECUTION_ROOT/scaffolds/transformer/launchers/kennel_runtime.sh"
exec bash "$EXECUTION_ROOT/scaffolds/transformer/studies/gla_key_dimension/run_allocated.sh" "$@"
