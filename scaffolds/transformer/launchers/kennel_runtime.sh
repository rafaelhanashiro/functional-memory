#!/usr/bin/env bash
# Source from Kennel launchers after entering the frozen execution checkout.
# GPU qualification remains a separate preflight requirement.
transformer_configure_kennel_runtime() {
  local gpu_names
  gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)" || return
  # One lock, two environments: B300 uses CUDA 13 (cuBLAS >= 13.2, which fixes Blackwell
  # data corruption when a cuBLAS kernel overlaps another kernel); other GPUs use CUDA 12.
  case "$gpu_names" in
    *B300*) export TRANSFORMER_CUDA_ENVIRONMENT=cuda13 TRANSFORMER_ENVIRONMENT_DIR=.venv-cuda13 ;;
    *) export TRANSFORMER_CUDA_ENVIRONMENT=cuda12 TRANSFORMER_ENVIRONMENT_DIR=.venv ;;
  esac
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
      # Collectives on the compute stream: the workaround used before the CUDA 13
      # environment; it costs no measurable throughput and is kept; see CLUSTERS.md.
      case " ${XLA_FLAGS:-} " in
        *" --xla_gpu_disable_async_collectives="*) ;;
        *) export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_disable_async_collectives=ALLCOLLECTIVES" ;;
      esac
      ;;
  esac
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
}
# The CUDA toolkit installed in an environment (CUDA 12 wheels: nvidia/cuda_nvcc,
# CUDA 13 wheels: nvidia/cu13), so jaxlib and Triton assemble with its ptxas.
transformer_cuda_home() {
  "$1" -c 'import importlib.util as u; s = u.find_spec("nvidia.cuda_nvcc") or u.find_spec("nvidia.cu13"); print(next(iter(s.submodule_search_locations)))'
}
transformer_configure_kennel_runtime
