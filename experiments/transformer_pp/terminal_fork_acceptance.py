"""Qualify the two forks of the completed exponential run r19dfa87cb7ed at its terminal update, 7,570.

Both children start from the parent's terminal weights (PARENT_CHECKPOINT; see the study README) with a fresh AdamW
at update zero on the parent's own schedule (455 warmup updates, peak 2e-3, cosine to 2e-4 over 7,570 updates), and
both read the parent's packed training stream from its next sequence, 1,937,920 (``continuation.optimizer: fresh``):

- exp -> exp keeps the parent's recipe and every parameter (``continuation.preserve_all_parameters``);
- exp -> query takes r90776a5d4175's attention, the degree-eight query polynomial with direct tanh coefficients on raw
  scores and W decay, unnormalized with per-head RMSNorm. Its new coefficient arrays are that recipe's own initial draw,
  so every query starts at T_8(s) = sum_{r<=8} s^r / r!, and every other array is the parent's.

A child's recipe is its partner's (the parent's or r90776a5d4175's) but for execution keys, the pinned continuation
block and ``train.retain_terminal_checkpoint``, which keeps the last resumable checkpoint for future continuation.
Shared acceptance checks the restored backbone's hash and the fresh optimizer. This adapter also checks, on the CPU
worker, that every shared array is the parent's bit for bit and that the new coefficients are the query recipe's
initial arrays. Its pilot check pins the batches both children record and read: the parent's last batch and their
first, the sequences after it. Both train on one of ORCD's H100s, the partners' GPU; the pinned first-update hash is
the one-device layout.
"""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path
import sys

from experiments.transformer_pp import key_width_acceptance
from scaffolds.transformer.acceptance import attention_spec, comparison_recipe, main as run_acceptance
from scaffolds.transformer.configs import ContinuationConfig, config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

STUDY = Path(__file__).resolve().parent
RUNS = STUDY / "runs"
PARENT = "r19dfa87cb7ed"
PARENT_CONFIG = f"experiments/transformer_pp/runs/{PARENT}/config.yaml"
PARENT_CONFIG_SHA256 = "358e43cd6e2e507f3995d6c17245e2bf8fe20116491edb18766ba49ae6ee64c5"
PARENT_DATA_IDENTITY_SHA256 = "6aad8549d983f9bc8dfb2e3bd7ff15954d112c6aeb15656635a91019c0b2c1ba"
PARENT_STEP = 7570
BATCH_SIZE = 256
DATA_START_SEQUENCE = PARENT_STEP * BATCH_SIZE
# The parent's step-7,570 state, rebuilt by replaying its last 70 updates once from its retained step-7,500 state
# (Slurm job 30686778; reconstructed_checkpoints/7570.json beside it records the replay and its comparison).
PARENT_CHECKPOINT = (f"/orcd/pool/006/rafah/functional_memory/artifacts/experiments/transformer_pp/runs/{PARENT}/"
                     f"reconstructed_checkpoints/{PARENT_STEP}")
PARENT_CHECKPOINT_SHA256 = "59c7f06fb08eae083959bba7978293ff6d4acb1fb5ba4e41f58a2e9c8b965e94"
# Each branch's partner: its run, recipe SHA256, and whether the child keeps every parent parameter.
PARTNERS = {
    "exp": (PARENT, PARENT_CONFIG_SHA256, True),
    "query": ("r90776a5d4175", "359f489b5f78dc047ee290354150f7d336f27acdc08586c9ea16cfb458f062bd", False),
}
QUERY_DEGREE = 8
# The first update's packed rows, sequences 1,937,920-1,938,175 of the train manifest (9de775ed...): as the trainer's
# checkpoint audit hashes them on one device (16 accumulation steps of 16 sequences, int32), and as one update
# (256 x 2049, int32), the first_batch the trainer records in continuation.json and checks its first update against.
# The parent's last update read 1,937,664-1,937,919, recorded as parent_last_batch.
FIRST_UPDATE_BATCH_SHA256 = "3ebd52ed42de3de391c0f4c5a3a5c235bbb350113d0c0d57090044b88ea1156d"
FIRST_BATCH_SHA256 = "fbe146caa7ab17d437febf8ddcfe7a695bc9c8d21a9fe60f9c9bba81ead9371d"
PARENT_LAST_BATCH_SHA256 = "53233277115349feb7834876817e2c0952fdf6bbba3b3a390a5f0d2e812e6631"
GPU_TYPES = ("H100",)
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir",
                  "wandb_mode")


def branch(config) -> str:
    degree = config.model.attention_degree
    if degree is None:
        return "exp"
    if degree == QUERY_DEGREE:
        return "query"
    raise ValueError(f"A fork of {PARENT} is the exponential or the degree-{QUERY_DEGREE} query polynomial, "
                     f"not degree {degree}")


def partner_config(name):
    run, digest, _ = PARTNERS[name]
    path = RUNS / run / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"The partner recipe {run} changed")
    return load_config(path)


def expected_continuation(name) -> dict:
    if PARENT_CHECKPOINT_SHA256 is None:
        raise ValueError("The fork checkpoint is not pinned yet")
    return {"parent_config": PARENT_CONFIG, "parent_config_sha256": PARENT_CONFIG_SHA256,
            "parent_checkpoint": PARENT_CHECKPOINT, "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "parent_data_identity_sha256": PARENT_DATA_IDENTITY_SHA256, "parent_step": PARENT_STEP,
            "new_parameter_learning_rate": None, "preserve_all_parameters": PARTNERS[name][2],
            "optimizer": "fresh", "data_start_sequence": DATA_START_SEQUENCE}


def fork_recipe(name):
    """A child's recipe: its partner's, with the pinned fork and the kept terminal checkpoint. Allocation
    (``scaffolds.records create``) writes the run's own directory, W&B name and source commit."""
    config = partner_config(name)
    config.continuation = ContinuationConfig(**expected_continuation(name))
    config.train.retain_terminal_checkpoint = True
    config.train.wandb_run_name = None
    validate_config(config)
    return config


def _changed(actual: dict, expected: dict) -> list[str]:
    return sorted(f"{section}.{key}" for section in set(actual) | set(expected)
                  for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                  if actual.get(section, {}).get(key) != expected.get(section, {}).get(key))


def check_recipe(config):
    validate_config(config)
    name = branch(config)
    partner = partner_config(name)
    actual, expected = comparison_recipe(config), comparison_recipe(partner)
    for entry in (actual, expected):
        for key in EXECUTION_KEYS:
            entry["train"].pop(key, None)
        entry["train"].pop("retain_terminal_checkpoint", None)
        entry.pop("continuation", None)
    if actual != expected:
        raise ValueError(f"Settings differ from {PARTNERS[name][0]}: {_changed(actual, expected)}")
    # The comparison holds head RMSNorm's evaluation at its historical value; the child keeps its partner's.
    if config.model.attention_head_rms_prescale != partner.model.attention_head_rms_prescale:
        raise ValueError("A fork evaluates head RMSNorm as its partner trained it")
    if config.train.retain_terminal_checkpoint is not True:
        raise ValueError("A fork keeps its terminal checkpoint for further training")
    continuation = config_to_dict(config)["continuation"]
    target = expected_continuation(name)
    if continuation != target:
        raise ValueError(f"The fork's continuation differs from the pinned fork: "
                         f"{sorted(key for key in set(continuation) | set(target) if continuation.get(key) != target.get(key))}")


def check_initialization(config):
    """Every shared array the parent's, bit for bit; the query child's new arrays the query recipe's initial draw."""
    import jax
    import numpy as np
    from flax import nnx

    from scaffolds.transformer.training.continuation import initialize_continuation_state
    from scaffolds.transformer.training.optim import CONTINUATION_PARAMETERS
    from scaffolds.transformer.training.state import create_state
    from scaffolds.transformer.training.checkpointing import restore_state

    name = branch(config)
    config.model.attention_backend = "blocked"
    state, audit = initialize_continuation_state(config)

    def flat(params):
        return {"/".join(str(part) for part in path): np.asarray(value[...])
                for path, value in nnx.to_flat_state(params)}

    child = flat(state.params)
    parent_config = load_config(PARENT_CONFIG)
    parent_config.model.attention_backend = "blocked"
    parent = flat(restore_state(PARENT_CHECKPOINT, create_state(parent_config, jax.random.PRNGKey(
        parent_config.train.seed))).params)
    introduced = {path for path in child if path.split("/")[0] in CONTINUATION_PARAMETERS}
    shared = set(child) - introduced
    if shared != set(parent) or any(child[path].dtype != parent[path].dtype or child[path].shape != parent[path].shape
                                    or not np.array_equal(child[path], parent[path]) for path in shared):
        raise RuntimeError(f"A fork's shared arrays differ from the parent {PARENT}'s")
    groups = {}
    for path in shared:
        key = path.split("/")[-2] if path.startswith("layers_") else path.split("/")[0]
        groups[key] = groups.get(key, 0) + int(child[path].size)
    if name == "query":
        query_recipe = partner_config("query")
        original = key_width_acceptance.flat_initial_parameters(query_recipe)
        if introduced != {"query_polynomial_bias", "query_polynomial_amplitude", "query_polynomial_kernel"} or any(
                not np.array_equal(child[path], original[path]) for path in introduced):
            raise RuntimeError("The query child's new coefficients differ from r90776a5d4175's initial arrays")
    elif introduced:
        raise RuntimeError("The exp child introduces no arrays")
    return {"branch": name, "parent_run_id": PARENT, "shared_arrays": len(shared),
            "shared_parameters": sum(int(child[path].size) for path in shared),
            "introduced_parameters": sum(int(child[path].size) for path in introduced),
            "shared_parameter_groups": dict(sorted(groups.items())),
            "all_shared_arrays_identical": True, "optimizer": audit["optimizer"],
            "optimizer_counts": audit["optimizer_counts"], "data_start_sequence": audit["data_start_sequence"]}


def check_pilot(config, outcome, validation):
    """The pilot is the fork's own first updates: from update zero, at the parent's next sequence and batch."""
    first = (outcome.get("checkpoint_audit") or {}).get("first_update") or {}
    expected_batches = {
        "parent_last_batch": {"sequence_start": DATA_START_SEQUENCE - BATCH_SIZE, "sequence_stop": DATA_START_SEQUENCE,
                              "sha256": PARENT_LAST_BATCH_SHA256},
        "first_batch": {"sequence_start": DATA_START_SEQUENCE, "sequence_stop": DATA_START_SEQUENCE + BATCH_SIZE,
                        "sha256": FIRST_BATCH_SHA256}}
    if (outcome.get("start_step") != 0 or outcome.get("restored_step") is not None
            or outcome.get("data_start_sequence") != DATA_START_SEQUENCE or first.get("step") != 1
            or first.get("batch_sha256") != FIRST_UPDATE_BATCH_SHA256
            or outcome.get("fork_batches") != expected_batches):
        raise RuntimeError(f"The fork's pilot did not start at update zero on the parent's next batch: {outcome}")
    return {"first_update_batch_sha256": FIRST_UPDATE_BATCH_SHA256, "data_start_sequence": DATA_START_SEQUENCE,
            **expected_batches}


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    extra = key_width_acceptance.GPU_TESTS["exponential"] if branch(config) == "exp" else ()
    return replace(
        shared, check_config=check_recipe, required_backend=config.model.attention_backend,
        check_initialization=check_initialization, check_pilot=check_pilot,
        cpu_tests=(*shared.cpu_tests, "experiments/transformer_pp/tests/test_terminal_fork_acceptance.py"),
        gpu_tests=(*shared.gpu_tests, *(test for test in extra if test not in shared.gpu_tests)))


def require_gpu_type(argv) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("config")
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    if args.runtime_env:
        return
    if args.gpu_type.upper() not in GPU_TYPES:
        raise ValueError(f"The forks of {PARENT} qualify and train on {' or '.join(GPU_TYPES)} only, "
                         f"not {args.gpu_type}")
    if args.gpu_count != 1:
        raise ValueError(f"The forks of {PARENT} train on one GPU: the pinned first-update hash is the one-device "
                         f"layout, not {args.gpu_count} devices")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    require_gpu_type(argv)
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
