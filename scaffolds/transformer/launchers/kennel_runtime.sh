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
    *B300*)
      # cuBLAS 12.8-13.1 can silently corrupt data on Blackwell when a cuBLAS kernel runs
      # concurrently with another kernel. Keep collectives on the compute stream until the
      # B300 environment has cuBLAS >= 13.2 (CUDA 13); see CLUSTERS.md.
      case " ${XLA_FLAGS:-} " in
        *" --xla_gpu_disable_async_collectives="*) ;;
        *) export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_disable_async_collectives=ALLCOLLECTIVES" ;;
      esac
      ;;
  esac
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
}
transformer_configure_kennel_runtime
