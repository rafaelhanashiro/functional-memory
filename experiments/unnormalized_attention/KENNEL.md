# Fresh run on Kennel

This handoff is for a separate run from initialization. Do not transfer the MIT
checkpoints or its `.cache` launch helpers: those helpers contain MIT job IDs and
coordinate the two MIT allocations. All model and training code, including the
independent initialization reference, is already in this repository.

The Kennel agents should adapt the scheduler submission and verify the installed
NVIDIA driver against the locked CUDA environment. Kennel compatibility has not
been tested from MIT. Keep the scientific configuration unchanged.

## Checkout and environment

```bash
git clone --branch codex/unnormalized-attention-250m \
  git@github.com:rafaelhanashiro/functional-memory.git
cd functional-memory
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv sync --project experiments/unnormalized_attention --locked --extra cuda
```

The lock selects Python 3.11 and JAX's CUDA 12 extra. Perform environment setup
under the cluster's usual resource policy. Commit any required local code changes
before acceptance; the launcher requires a clean, frozen checkout.

## Packed data

Copy the existing packed corpus from MIT, preserving both manifests and every
listed shard. Its source is:

```text
/home/rafah/projects/ofm/datasets/fineweb_12b/
  train/manifest.json
  train/shard-00000.npy ... shard-00089.npy
  eval/manifest.json
  eval/shard-00000.npy
```

These 91 shards and two manifests total **24,031,780,324 bytes** (22.38 GiB).
The small `README.md` and `corpus.json` are also useful provenance. No tokenizer
download, Hugging Face cache, raw corpus, or model checkpoint is needed at runtime.
All manifest-listed shards are validated, including trailing training shards.

Manifest paths are relative to their own directories, so the corpus can move
without changing either JSON file. Expected SHA256:

| Manifest | SHA256 |
| --- | --- |
| `train/manifest.json` | `9de775ed836ab7c61857207335a981c09630caa3226d2dc0ee0b799cdcb2f4f4` |
| `eval/manifest.json` | `0e8615538d9dccad17956d8d9839394d05a56f8ec210dc67716db829a543634f` |

Use a separate empty output directory. From the repository root, create an
external config that changes only the data and output paths:

```bash
data_root=/absolute/kennel/path/fineweb_12b
run_root=/absolute/kennel/path/250m-exp-seed0
mkdir -p "$run_root"
"$UV_PROJECT_ENVIRONMENT/bin/python" - "$data_root" "$run_root" <<'PY'
from pathlib import Path
import sys
from experiments.unnormalized_attention.configs import load_config, save_config

data_root, run_root = (Path(p).resolve() for p in sys.argv[1:])
config = load_config('experiments/unnormalized_attention/configs/base.yaml')
config.data.train_manifest = str(data_root / 'train/manifest.json')
config.data.eval_manifest = str(data_root / 'eval/manifest.json')
config.train.workdir = run_root / 'training'
output = run_root / 'config.yaml'
if output.exists() or config.train.workdir.exists():
    raise RuntimeError('Choose a new output directory for the fresh Kennel run')
save_config(config, output)
print(output)
PY
```

## GPU allocation and execution

The current scaffold uses **one host**, with one Python process driving all its
GPUs. For the intended four-GPU run use four H100s or four H200s of one type, with
32 CPUs and 256 GiB host RAM as a tested starting allocation. GPU count must
divide global microbatch 64; global batch remains 256 and accumulation remains 4.

Use Kennel's partition/account, GPU resource syntax, and time limits. The MIT
`launch.sbatch` and `bootstrap.sbatch` contain MIT defaults and absolute paths;
use the portable entry point below instead. For Slurm, explicitly request the
GPU count per node as well as one node and one task. Preserve the GPUs exposed
by the scheduler; acceptance requires the requested count and type to be visible.

Run this command **inside the allocated GPU job**, from the repository root:

```bash
bash experiments/unnormalized_attention/run_allocated.sh \
  "$run_root/config.yaml" "$run_root/runtime" 4 H200 \
  "$UV_PROJECT_ENVIRONMENT/bin/python"
```

Use `H100` for an H100 allocation. The script runs CPU/GPU tests, verifies all
253,654,016 initial parameters against the independent reference, runs a
20-update full-context pilot and one-update resume check, then starts production
in the same allocation. It checks the actual GPU count and type. The distributed
unit fixture supports both two and four devices without restricting visibility.
Acceptance records are isolated by source, config hash, and allocation.

If Kennel does not use Slurm, adapt the scheduler-dependent checks explicitly;
the current preflight requires a real `SLURM_JOB_ID`. Keep the hardware and
numerical acceptance checks. Do not change the checkout or environment while
its job can run or requeue.

The production budget is 21,935 updates / 11,500,257,280 tokens. Seed, optimizer,
learning-rate schedule, packed-data order, evaluation set, and attention formula
must match `configs/base.yaml`. The formula is the unshifted causal
`exp(QK^T / sqrt(d_k)) V`, with no attention denominator or added normalization.
The historical normalized reference has validation PPL **13.3496**; its record is
in `results/unnormalized_attention/baseline.json`.

Outputs go to `$run_root/training`. A fresh directory starts at step zero;
subsequent launches into that same directory resume its latest complete
checkpoint, including optimizer and data position. Checkpoints are written every
500 updates and at validation. Keep best and latest while active, then retain
only the validation-best checkpoint after completion/review, as required by
`AGENTS.md`. Keep the dataset, caches, and checkpoints outside Git.

MIT and Kennel are independent allocations. The MIT alternative-job selector
does not coordinate with Kennel; stop the unwanted cluster's jobs separately if
only one run is desired once an allocation starts.
