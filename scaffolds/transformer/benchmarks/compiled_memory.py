"""Device memory of a recipe's training update, from XLA's compiled program, without running it.

The update is the trainer's own one-device program (``training.loop.accumulated_train_step`` with the
options the loop passes: update checks, and telemetry unless the model rematerializes) on the recipe's
state, accumulation and micro-batch. XLA reports its arguments (state, batches, RNG keys), outputs (the
updated state and checks), aliases and temporaries; an update needs their sum on the device at once,
beside whatever else the process holds (for example a gradient-event capture's pre-update state on
several devices). The allocator grants XLA_PYTHON_CLIENT_MEM_FRACTION of the device (0.75 by default,
also without preallocation), so a program whose sum exceeds that fails with RESOURCE_EXHAUSTED.

``--target local`` compiles for this process's first GPU (inside an allocation; autotuned, as training
compiles). ``--target h100`` or ``b200`` compiles for that GPU from any host through a compile-only CUDA
topology (no GPU and no autotuning; the B200's bandwidth, L2 size and clock are nominal). On an H100 the
trained programs' temporaries were 3-5% above the compile-only estimate (score-bias strips, 2026-11-11).
``--micro-batch`` compiles another micro-batch at the recipe's global batch, to explore a recipe change.

    python -m scaffolds.transformer.benchmarks.compiled_memory RECIPE --target b200 [--micro-batch 8] \\
        [--output REPORT.json]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

H100 = """
gpu_device_info {
  threads_per_block_limit: 1024 threads_per_warp: 32 shared_memory_per_block: 49152
  shared_memory_per_core: 233472 threads_per_core_limit: 2048 core_count: 132 fpus_per_core: 128
  block_dim_limit_x: 2147483647 block_dim_limit_y: 65535 block_dim_limit_z: 65535
  memory_bandwidth: 3352320000000 l2_cache_size: 52428800 clock_rate_ghz: 1.98
  device_memory_size: 85049802752 shared_memory_per_block_optin: 232448
  cuda_compute_capability { major: 9 minor: 0 }
  registers_per_core_limit: 65536 registers_per_block_limit: 65536
}
platform_name: "CUDA"
dnn_version_info { major: 9 minor: 22 patch: 0 }
runtime_version { major: 12 minor: 9 }
device_description_str: "NVIDIA H100 80GB HBM3"
"""
B200 = """
gpu_device_info {
  threads_per_block_limit: 1024 threads_per_warp: 32 shared_memory_per_block: 49152
  shared_memory_per_core: 233472 threads_per_core_limit: 2048 core_count: 148 fpus_per_core: 128
  block_dim_limit_x: 2147483647 block_dim_limit_y: 65535 block_dim_limit_z: 65535
  memory_bandwidth: 8000000000000 l2_cache_size: 132644864 clock_rate_ghz: 1.965
  device_memory_size: 192265846784 shared_memory_per_block_optin: 232448
  cuda_compute_capability { major: 10 minor: 0 }
  registers_per_core_limit: 65536 registers_per_block_limit: 65536
}
platform_name: "CUDA"
dnn_version_info { major: 9 minor: 22 patch: 0 }
runtime_version { major: 13 minor: 2 }
device_description_str: "NVIDIA B200"
"""
TARGETS = {"h100": H100, "b200": B200}
FIELDS = ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes",
          "generated_code_size_in_bytes", "host_temp_size_in_bytes")
GIB = 2 ** 30


def compiled_update_memory(config, target: str) -> dict:
    """XLA's memory analysis of ``config``'s one-device accumulated update compiled for ``target``."""
    import jax
    import jax.numpy as jnp

    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.numerics import requires_update_checks
    from scaffolds.transformer.training.state import create_state

    micro = config.train.micro_batch_size
    if config.data.batch_size % micro:
        raise ValueError(f"Micro-batch {micro} does not divide the batch {config.data.batch_size}")
    accumulation = config.data.batch_size // micro
    if accumulation < 2:
        raise ValueError("The loop runs one-micro-batch updates through train_step, not the accumulated update")
    if target == "local":
        device, options = jax.local_devices()[0], {}
    else:
        from jax.experimental import topologies

        device = topologies.get_topology_desc(target, "cuda", target_config=TARGETS[target],
                                              topology="1x1x1").devices[0]
        options = {"xla_gpu_autotune_level": "0"}  # Autotuning needs the device.
    sharding = jax.sharding.SingleDeviceSharding(device)
    rng = jax.random.PRNGKey(config.train.seed)
    with jax.default_device(jax.devices("cpu")[0]):
        abstract = jax.eval_shape(lambda: create_state(config, rng))
    state = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding), abstract)
    batches = jax.ShapeDtypeStruct((accumulation, micro, config.data.sequence_length), jnp.int32, sharding=sharding)
    rngs = jax.ShapeDtypeStruct((accumulation, 2), jnp.uint32, sharding=sharding)
    static = {"check_all_gradients": requires_update_checks(config.model), "collect_telemetry": not config.model.remat}
    started = time.perf_counter()
    lowered = loop.accumulated_train_step.lower(state, batches, rngs, **static)
    compiled = lowered.compile(options)
    seconds = time.perf_counter() - started
    analysis = compiled.memory_analysis()
    memory = {name: getattr(analysis, name, None) for name in FIELDS}
    total = (memory["argument_size_in_bytes"] + memory["output_size_in_bytes"] - memory["alias_size_in_bytes"]
             + memory["temp_size_in_bytes"])
    report = {"target": target, "device_kind": getattr(device, "device_kind", None), "micro_batch_size": micro,
              "accumulation_steps": accumulation, "static_options": static, "compile_options": options,
              "lower_and_compile_seconds": seconds, "memory": memory, "update_bytes": total,
              "update_GiB": total / GIB, "temp_GiB": memory["temp_size_in_bytes"] / GIB}
    if target == "local" and device.memory_stats():
        report["allocator_bytes_limit"] = device.memory_stats().get("bytes_limit")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--target", choices=("local", *TARGETS), required=True)
    parser.add_argument("--micro-batch", type=int, help="compile this micro-batch at the recipe's global batch")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    import jax

    from scaffolds.transformer.configs import load_config

    config = load_config(args.recipe)
    if args.micro_batch:
        config.train.micro_batch_size = args.micro_batch
    report = {"recipe": str(args.recipe), **compiled_update_memory(config, args.target),
              "xla_flags": os.environ.get("XLA_FLAGS"), "jax": jax.__version__}
    text = json.dumps(report, indent=1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
