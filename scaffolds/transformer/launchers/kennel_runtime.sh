#!/usr/bin/env bash
# Source from Kennel launchers after entering the frozen execution checkout.
# GPU qualification remains a separate preflight requirement.
transformer_configure_kennel_runtime() {
  local gpu_names
  gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)" || return
  case "${TRANSFORMER_HOST_STAGING:-auto}" in
    auto)
      case "$gpu_names" in
        *A100*|*A6000*) export TRANSFORMER_HOST_STAGING=1 ;;
        *) export TRANSFORMER_HOST_STAGING=0 ;;
      esac
      ;;
    0|1) export TRANSFORMER_HOST_STAGING ;;
    *) echo 'TRANSFORMER_HOST_STAGING must be auto, 0, or 1 in a Kennel launcher' >&2; return 1 ;;
  esac
  if [[ "$TRANSFORMER_HOST_STAGING" == 1 ]]; then
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  fi
  case "$gpu_names" in
    *A6000*) export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}" ;;
  esac
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
}
transformer_configure_kennel_runtime
