"""Compile-only H100 checks of the causal strips (no GPU; they skip without the CUDA plugin).

Kept apart from test_blocked_strips.py, which acceptance runs for strips recipes and where a
skip would fail acceptance. test_blocked_strips.py checks the memory property structurally
(every strip's checkpoint prevents CSE).
"""
from collections import defaultdict, deque
from functools import partial
import re

import jax
import jax.numpy as jnp
import pytest

from scaffolds.transformer.attention.backends import reference
from scaffolds.transformer.attention.backends.reference import strip_dot_product_attention


def _h100_sharding():
    from scaffolds.transformer.validation.test_blocked_attention_benchmark import h100_sharding

    return h100_sharding()


def test_strip_memory_does_not_grow_with_layers_on_h100():
    """Compile-only H100 program (no GPU): each extra layer adds its saved inputs, not its strips'
    [strip x keys] intermediates (with prevent_cse=False it adds about 2 of them per layer)."""
    sharding = _h100_sharding()
    shape = (1, 1024, 2, 64)

    def temporary_bytes(layers):
        def loss(x, w):
            for layer in range(layers):
                x = strip_dot_product_attention(x * w[layer], x, x, strip_size=256) + x
            return jnp.sum(x.astype(jnp.float32) ** 2)

        arguments = (jax.ShapeDtypeStruct(shape, jnp.bfloat16, sharding=sharding),
                     jax.ShapeDtypeStruct((layers,), jnp.bfloat16, sharding=sharding))
        compiled = jax.jit(jax.grad(loss, argnums=(0, 1))).lower(*arguments).compile({"xla_gpu_autotune_level": "0"})
        return compiled.memory_analysis().temp_size_in_bytes

    strips_score_bytes = 4 * 2 * 256 * sum(range(256, 1025, 256))  # one FP32 [strip x prefix] array per strip
    growth = (temporary_bytes(3) - temporary_bytes(1)) / 2
    assert growth < 0.5 * strips_score_bytes, (growth, strips_score_bytes)


def _closing(text, start):
    depth = 0
    for index in range(start, len(text)):
        depth += {"(": 1, ")": -1}.get(text[index], 0)
        if depth == 0:
            return index
    raise ValueError(f"Unbalanced HLO: {text[start:start + 80]}")


def _dataflow(text):
    """An optimized HLO module's instructions and their users. Fusions connect their operands to
    the fused computation's parameters and its root to their own result."""
    computations, current = {}, None
    for line in text.splitlines():
        header = re.match(r"(?:ENTRY )?%([\w.\-]+) .*\{$", line)
        if header:
            current = computations.setdefault(header.group(1), {})
        elif line.startswith("}"):
            current = None
        elif current is not None and line.startswith("  ") and " = " in line:
            body = line.strip()
            name, rest = body.removeprefix("ROOT ").split(" = ", 1)
            kind_end = _closing(rest, 0) + 1 if rest.startswith("(") else rest.index(" ")
            kind, rest = rest[:kind_end], rest[kind_end:].lstrip()
            opcode = re.match(r"[\w\-$]+", rest).group(0)
            end = _closing(rest, rest.index("("))
            attributes = rest[end + 1:]
            called = re.search(r"\bcalls=%([\w.\-]+)", attributes)
            parameter = re.match(r"parameter\((\d+)\)", rest)
            op_name = re.search(r'op_name="([^"]*)"', attributes)
            current[name.lstrip("%")] = {
                "root": body.startswith("ROOT "), "opcode": opcode, "operands": re.findall(r"%([\w.\-]+)", rest[:end]),
                "calls": called and called.group(1), "parameter": parameter and int(parameter.group(1)),
                "op_name": op_name.group(1) if op_name else "",
                # bf16 results and reduce-precision to BF16 hold rounded values.
                "rounds": kind.startswith("bf16[") or (opcode == "reduce-precision" and "mantissa_bits=7" in attributes),
            }
    users = defaultdict(set)
    for computation, instructions in computations.items():
        for name, instruction in instructions.items():
            fused = computations.get(instruction["calls"]) if instruction["opcode"] == "fusion" else None
            for index, operand in enumerate(instruction["operands"]):
                if fused is None:
                    users[(computation, operand)].add((computation, name))
                else:
                    users[(computation, operand)] |= {(instruction["calls"], inner) for inner, value in fused.items()
                                                      if value["parameter"] == index}
            if fused is not None:
                for inner, value in fused.items():
                    if value["root"]:
                        users[(instruction["calls"], inner)].add((computation, name))
    instructions = {(computation, name): value for computation, values in computations.items()
                    for name, value in values.items()}
    entry = re.search(r"^ENTRY %([\w.\-]+) ", text, re.MULTILINE).group(1)
    return instructions, users, entry


def _unrounded(instructions, users, seeds, *, stop=True):
    """Instructions reached from ``seeds`` along values that no BF16 rounding has passed through."""
    seen, queue = set(seeds), deque(seeds)
    while queue:
        for user in users.get(queue.popleft(), ()):
            if user not in seen:
                seen.add(user)
                if not (stop and instructions[user]["rounds"]):
                    queue.append(user)
    return seen


def _rounding_paths(length):
    """Products of the strips that read Q/K/V not rounded to BF16, and whether a consumer that upcasts the
    output (head RMSNorm) reads it unrounded; Q/K/V are rounded to BF16 from FP32 math, as QK RMSNorm and
    RoPE produce them. Forward and backward at ``length`` tokens in strips of 256."""
    sharding = _h100_sharding()

    def forward(q, k, v):
        q, k, v = ((x * scale).astype(jnp.bfloat16) for x, scale in ((q, 1.5), (k, 0.5), (v, 2.0)))
        y = partial(strip_dot_product_attention, strip_size=256)(q, k, v).astype(jnp.float32)
        return y * jax.lax.rsqrt(jnp.mean(y * y, axis=-1, keepdims=True) + 1e-6)

    arguments = [jax.ShapeDtypeStruct((1, length, 2, 64), jnp.float32, sharding=sharding)] * 3
    result = {}
    for label, program in (("forward", forward),
                           ("backward", jax.grad(lambda *x: jnp.sum(forward(*x)), argnums=(0, 1, 2)))):
        text = jax.jit(program).lower(*arguments).compile({"xla_gpu_autotune_level": "0"}).as_text()
        instructions, users, entry = _dataflow(text)
        inputs = [(entry, name) for (computation, name), value in instructions.items()
                  if computation == entry and value["parameter"] is not None]
        products = {key for key, value in instructions.items() if value["opcode"] == "dot"}
        assert products & _unrounded(instructions, users, inputs, stop=False), "no product reads Q/K/V"
        result[label] = sorted(instructions[key]["op_name"] for key in products & _unrounded(instructions, users, inputs))
        if label == "forward":
            values = {key for key in products if "bhqk,bkhd->bqhd" in instructions[key]["op_name"]}
            consumers = {key for key, value in instructions.items() if value["opcode"] == "rsqrt"}
            assert len(values) == -(-length // 256) and consumers & _unrounded(instructions, users, values, stop=False)
            result["output"] = bool(consumers & _unrounded(instructions, users, values))
    return result


@pytest.mark.parametrize("length", (1024, 256))
def test_bf16_roundings_survive_xla_gpu(length):
    """Every strip product (score, P V, and the backward's) reads BF16-rounded Q/K/V, and the output
    is rounded before a consumer upcasts it, with four strips and with one."""
    assert _rounding_paths(length) == {"forward": [], "backward": [], "output": False}


def test_plain_conversions_lose_their_bf16_rounding_on_xla_gpu(monkeypatch):
    """The hazard the explicit rounding removes: with plain conversions XLA's excess-precision
    simplification feeds the products unrounded FP32 Q/K/V and, without a concatenation between
    them (one strip), hands the unrounded output to its consumer."""
    monkeypatch.setattr(reference, "_rounded_to", lambda x, dtype: x)
    four, one = _rounding_paths(1024), _rounding_paths(256)
    assert four["forward"] and four["backward"] and one["output"]
