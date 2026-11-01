# Native FLA runtime

This package trains the upstream `GatedDeltaNetForCausalLM` directly. It lives in
the Transformer scaffold as a reusable model-family runtime, with an independent
CUDA PyTorch environment. It does not instantiate the JAX Transformer backbone or
load its configuration defaults.

`config.py` pins the upstream architecture, initialization and stock Flame
optimizer recipe. `reference_recipe()` returns a complete template; callers fill
dataset manifests, output paths and tracking identity. `validate_config()` rejects
undeclared scientific overrides. `reference_recipe(total_steps=...)` explicitly
adapts the token budget: warmup keeps the reference's 5% fraction (rounded half up
to whole updates, minimum one), and evaluation cadence scales by the same factor.
All other optimizer values and the regular checkpoint cadence remain fixed.
Native FLA owns all layers and their
initialization; the runtime preserves FP32 master parameters and uses BF16
autocast. Model execution is eager, without `torch.compile`.

```bash
uv sync --project scaffolds/transformer/native_fla --locked --link-mode hardlink
scaffolds/transformer/native_fla/.venv/bin/python -m pytest -q \
  scaffolds/transformer/native_fla/tests
```

`train.py` reads checksum-verified packed sequences through the shared NumPy
reader. From each row of T+1 tokens it passes the first T tokens to FLA and applies
FLA's fused cross-entropy to the last T tokens; passing those targets through
FLA's `labels` argument would shift them twice. Gradient accumulation keeps the
configured global batch constant on one GPU. AdamW uses the stock Flame parameter
group, including weight decay on every parameter. The schedule reproduces the
pinned TorchTitan LambdaLR indexing, including the initial warmup update and
virtual stable step.

`checkpointing.py` atomically publishes full model, optimizer, RNG and packed-data
cursor state. Resume verifies recipe identity and discards measurements beyond the
restored step. `retention.py` keeps latest and validation best during training, then
only validation best after completion. A numerical failure records its stopping
point separately from completion. SIGUSR1/SIGTERM request a checkpoint at an update
boundary and exit 75 for the launcher to requeue.

`acceptance.py` runs the full model/global batch/context and compares uninterrupted
training with a checkpoint resumed in a fresh process, including optimizer and
data state. Its disposable outputs remain under the run's `.runtime/`; a passed
identity is reusable only with matching source, recipe, environment and hardware.
Production launch runs this gate before training. Runtime qualification evidence
belongs in the scaffold's [validation record](../validation/README.md).

```bash
# Use the native launcher after committing the frozen execution inputs.
scaffolds/transformer/native_fla/.venv/bin/python \
  -m scaffolds.transformer.native_fla.launch --run experiments/<study>/runs/<run_id>
```

The launcher records allocation, source and frozen-set identity, appends the
retained log, and rechecks frozen inputs on every start/requeue. Cluster resource
policy and supported hardware are documented in [CLUSTERS.md](../../../CLUSTERS.md).
Scientific recipes and run selections belong to their study, not this package.
