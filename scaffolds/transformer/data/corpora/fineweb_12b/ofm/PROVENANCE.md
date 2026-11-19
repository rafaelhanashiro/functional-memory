# fineweb_12b builder provenance

These are the `ofm` project files that built `fineweb_12b`, copied unchanged from ofm's Git
objects to their ofm-relative paths. Slurm job 20811682 built the corpus at ofm commit `712d451`,
in a worktree whose builder files were unmodified. Later ofm commits changed the materializer and its launcher and added the
configuration `configs/data/fineweb_12b.yaml`. The checked-out versions of those three files are
copied unchanged to [`../ofm-8977f529/`](../ofm-8977f529/); they did not build this corpus.

## Source

| | |
| --- | --- |
| Project | `/home/rafah/projects/ofm` on ORCD |
| Remote | `origin`, `git@github.com:rafaelhanashiro/ofm.git` (fetch and push) |
| Checked-out commit | `8977f529e83efc688daf96cdcfd365ebbce2afa8` on `main`, 2026-09-11, equal to `origin/main` as last fetched (2026-09-12) |
| Working tree | Clean: `git status --short` prints nothing, and `git diff` and `git diff --cached` of the builder files are empty |
| Build commit | `712d45152331dea665ff18860b58d0474e4d9cba`, 2026-08-20 00:40:45 -0400, "Specify deterministic FineWeb 12B corpus" |

## Build

- Commit `712d451` added the launcher [`scripts/fineweb_12b/materialize.sbatch`](scripts/fineweb_12b/materialize.sbatch)
  and the build report. `git worktree add -b data-12b worktrees/data_12b main` then created a
  worktree at `712d451`, and its `git status` showed no changes.
- From that worktree, `sbatch scripts/fineweb_12b/materialize.sbatch` submitted job 20811682
  (Slurm's `SubmitLine` and `WorkDir`, below). The launcher runs
  [`scripts/materialize_fineweb/materialize_fineweb.py`](scripts/materialize_fineweb/materialize_fineweb.py)
  with command-line flags in the environment `/home/rafah/projects/ofm/.envs/cuda`, with
  `HF_HOME=/home/rafah/projects/ofm/.caches/hf`, and writes `runs/fineweb_12b/data` in the
  worktree.
- The job ran once, with no restarts, from 00:42:58 to 02:20:40 -0400 on `node1900`; the 91
  shards' file times run from 00:44 to 02:20.
- Job 20814071 then validated the corpus (below). The report was committed on `data-12b` as
  `b29cf63` and reached `main` as `f2f1a28`, with the same `REPORT.md` blob.
- The corpus is now at `/home/rafah/projects/ofm/datasets/fineweb_12b`. The worktree and the
  `data-12b` branch no longer exist.

The corpus was built from committed builder files. The only tracked file changed in the
worktree before the job ended was `runs/fineweb_12b/REPORT.md`; the worktree's `git status`
after the job lists nothing else, so the builder files and `uv.lock` were the `712d451` versions
and `uv run` did not rewrite the lock. Five later commits changed the builder: `8972d52`
(2026-08-20 18:21:25 -0400), `7cacb84`, `04c8894`, `3acec9b` and `cbd08f8`.

## Files

`sha256sum -c`, run from `scaffolds/transformer/data/corpora/fineweb_12b`:

```text
dfee0ee60f147043260f29a64b48f1ae503dbbeb583be0d0e1f8efdfc1aec376  ofm/scripts/fineweb_12b/materialize.sbatch
e4511502d81b55490e61fb1115787bbafdad3bba46fa28a24cfd2e6bddf450ac  ofm/scripts/materialize_fineweb/materialize_fineweb.py
f0ae68215a6772ae45bb57c4270b3dce769cd9331ed445b59ce2354b8e32aa43  ofm/data/__init__.py
664df08ef60a13d6e69ce7ca4b13d11fe2f3d152c53b3dcff7448ef7294d5cca  ofm/data/packed.py
0f9715a21c1a75dffa0b54f06804d43d019830e136639b0e1b6d9b32858a9dcb  ofm/pyproject.toml
bbf095db881866fc7f62537c340945a5ade40b7f70459ac93bab2d67e2160f88  ofm/uv.lock
abefde8473a7633c133d0c594a3a889a1b67873d900bf0984b51cf496e71b92e  ofm/runs/fineweb_12b/REPORT.md
f48aa68487ba7cd3f8fe7c27c1888dbe69926b5b640d42f4301ba09e51e62d3a  ofm-8977f529/configs/data/fineweb_12b.yaml
815bbd729200a0e27f3c29a9ce850ccfa05f53ebde52616a5427b144d4705066  ofm-8977f529/scripts/materialize_fineweb/materialize_fineweb.py
ef58b472df8c2d381c8e2725403cece5a2025064b58c5c557ff2f3d99fa75287  ofm-8977f529/scripts/materialize_fineweb/materialize_fineweb.sbatch
```

| Copy | ofm source | Git blob | State |
| --- | --- | --- | --- |
| `ofm/scripts/fineweb_12b/materialize.sbatch` | `712d451` | `9b62f79e3d2b24de9437b0f871ab967c67446e1b` | Committed in `712d451`; the launcher job 20811682 ran. Deleted in `8972d52`. Mode 755. |
| `ofm/scripts/materialize_fineweb/materialize_fineweb.py` | `712d451` | `daf65b214b84817c29240f1ab821698e04bb8a4a` | Committed (`40bb0c5`); ran unmodified. Changed after the build in `8972d52`. |
| `ofm/data/__init__.py` | `712d451` | `b7bc140a4319091d6c2bbb05a112d6bbb12069cd` | Committed (`a815279`); ran unmodified; unchanged at `8977f529`. |
| `ofm/data/packed.py` | `712d451` | `571bac5e58cf171fa91b5f5b42208984647c4b94` | Committed (`2b51692`); ran unmodified; unchanged at `8977f529`. |
| `ofm/pyproject.toml` | `712d451` | `76615bf67a40595613e4afee367ec37bf1b91774` | Committed (`0ebf24a`); the build's environment definition. Changed after the build in `3acec9b` and `cbd08f8`. |
| `ofm/uv.lock` | `712d451` | `18568ff4942729dadfa5c5e3f2af966495ea05c9` | Committed (`0ebf24a`); the build's lock. Changed after the build in `3acec9b` and `cbd08f8`. |
| `ofm/runs/fineweb_12b/REPORT.md` | `f2f1a28` | `3249e636085bf26d3f93c5129160a766e6534c46` | The build's own record, committed after validation. Deleted in `ca88edc`. |
| `ofm-8977f529/configs/data/fineweb_12b.yaml` | `8977f529` | `feeaa8596b0f51c99499a721d87f9cc2c300edb2` | Committed after the build in `8972d52`; unmodified in the checkout. |
| `ofm-8977f529/scripts/materialize_fineweb/materialize_fineweb.py` | `8977f529` | `725d9f73e7df4d9a9b84c02e291327429545b709` | Committed after the build in `8972d52`; unmodified in the checkout. |
| `ofm-8977f529/scripts/materialize_fineweb/materialize_fineweb.sbatch` | `8977f529` | `99688d3857cd80f23f02197e180ea45cec670ab3` | Committed after the build in `8972d52`, `7cacb84` and `04c8894`; unmodified in the checkout. |

Each copy's `git hash-object` equals its blob. The copies from `8977f529`, and those of
`data/__init__.py` and `data/packed.py`, are also byte-identical to the checkout (`cmp`). At
`712d451`, `scripts/materialize_fineweb/materialize_fineweb.sbatch` (blob
`bd4123e31f62a77d0b127db4b761a9390b28eec9`) existed but did not run; it is not copied.

### History

```text
$ git log --format='%H %ci %s' -- scripts/fineweb_12b/materialize.sbatch
8972d52e68375b7076e9efb00501140bbe07de05 2026-08-20 18:21:25 -0400 Align scaffold with shared dataset layout
712d45152331dea665ff18860b58d0474e4d9cba 2026-08-20 00:40:45 -0400 Specify deterministic FineWeb 12B corpus

$ git log --format='%H %ci %s' -- scripts/materialize_fineweb/materialize_fineweb.py
8972d52e68375b7076e9efb00501140bbe07de05 2026-08-20 18:21:25 -0400 Align scaffold with shared dataset layout
40bb0c5dfac007f50d593df786abf1177400f881 2026-07-24 08:52:18 -0400 Add 50M Chinchilla packed-data runs

$ git log --format='%H %ci %s' -- scripts/materialize_fineweb/materialize_fineweb.sbatch
04c8894c0373c681bb5b06f32eaadcaf51442a33 2026-08-21 19:40:50 -0400 Preserve Slurm command status during temp cleanup
7cacb8466735fec80a2f31a5a7e7e47c9bea99e8 2026-08-21 16:57:19 -0400 Modernize verification and clean transient job state
8972d52e68375b7076e9efb00501140bbe07de05 2026-08-20 18:21:25 -0400 Align scaffold with shared dataset layout
40bb0c5dfac007f50d593df786abf1177400f881 2026-07-24 08:52:18 -0400 Add 50M Chinchilla packed-data runs

$ git log --format='%H %ci %s' -- configs/data/fineweb_12b.yaml
8972d52e68375b7076e9efb00501140bbe07de05 2026-08-20 18:21:25 -0400 Align scaffold with shared dataset layout

$ git log --format='%H %ci %s' -- data/__init__.py
a815279b358b3c1da2b0a3be8529dbd6bc6269cb 2026-05-20 00:04:18 -0400 Refactor project layout for language model experiments

$ git log --format='%H %ci %s' -- data/packed.py
2b51692749652cdece42e2a4e1ea2734508a9911 2026-08-19 23:51:40 -0400 Avoid redundant packed-data hashing
40bb0c5dfac007f50d593df786abf1177400f881 2026-07-24 08:52:18 -0400 Add 50M Chinchilla packed-data runs

$ git log --format='%H %ci %s' -- pyproject.toml
cbd08f8c65b0c8a027936eadfc861977668a467d 2026-08-23 12:00:47 -0400 Keep CPU CI free of unused CUDA dependencies
3acec9ba0bc0365e7ad28d36e8474785e9c6cc54 2026-08-21 15:49:33 -0400 Remove obsolete training entrypoint
0ebf24a647ecb40e63b07a5f51e0d26c95380f49 2026-05-25 20:26:29 -0400 Refactor training scaffold and experiment configs
2c144db444f8f1bf61b109d74e4718bb81834ba2 2026-05-20 00:16:08 -0400 Fix pytest import path in CI
a815279b358b3c1da2b0a3be8529dbd6bc6269cb 2026-05-20 00:04:18 -0400 Refactor project layout for language model experiments
3e78f1854221d1b0529d164cb4ca869623a70df5 2026-05-14 21:42:34 -0400 Add configurable tqdm progress display
033319a6db0a9b9b6d07d6469523d79bd1e5f33a 2026-05-14 20:27:59 -0400 Add online functional memory copy experiment

$ git log --format='%H %ci %s' -- uv.lock
cbd08f8c65b0c8a027936eadfc861977668a467d 2026-08-23 12:00:47 -0400 Keep CPU CI free of unused CUDA dependencies
3acec9ba0bc0365e7ad28d36e8474785e9c6cc54 2026-08-21 15:49:33 -0400 Remove obsolete training entrypoint
0ebf24a647ecb40e63b07a5f51e0d26c95380f49 2026-05-25 20:26:29 -0400 Refactor training scaffold and experiment configs
a815279b358b3c1da2b0a3be8529dbd6bc6269cb 2026-05-20 00:04:18 -0400 Refactor project layout for language model experiments
3e78f1854221d1b0529d164cb4ca869623a70df5 2026-05-14 21:42:34 -0400 Add configurable tqdm progress display
033319a6db0a9b9b6d07d6469523d79bd1e5f33a 2026-05-14 20:27:59 -0400 Add online functional memory copy experiment

$ git log --format='%H %ci %s' -- runs/fineweb_12b/REPORT.md
ca88edc23c25cdc5e69771131fd3c0d1f8edc884 2026-08-20 17:44:58 -0400 Remove dataset report from experiment runs
f2f1a285d6522d0031e56aec329156d7374e556e 2026-08-20 02:28:52 -0400 Record validated FineWeb 12B corpus
712d45152331dea665ff18860b58d0474e4d9cba 2026-08-20 00:40:45 -0400 Specify deterministic FineWeb 12B corpus
```

## Imports

The materializer imports one ofm module, `data.packed`, through the package `data/__init__.py`,
which holds only a docstring. `data/packed.py` imports the standard library and `numpy`. The
materializer's other imports are the standard library and `tqdm` and, inside functions,
`huggingface_hub.HfApi`, `datasets.load_dataset` and `transformers.AutoTokenizer`. The launcher
puts the project directory on `PYTHONPATH`; the project itself is not installed, since
`pyproject.toml` declares no build system.

## Environment

| Component | Version |
| --- | --- |
| Python | CPython 3.11.15, uv-managed: `3.11.15 (main, Mar 10 2026, 18:16:52) [Clang 21.1.4 ]` |
| uv | 0.10.10 |
| datasets | 4.8.5 |
| tokenizers | 0.22.2 |
| transformers | 5.8.1 |
| huggingface_hub | 1.15.0 |
| numpy | 2.4.6 |
| pyarrow | 24.0.0 |

- The launcher runs `uv run --python 3.11 --extra cuda` with
  `UV_PROJECT_ENVIRONMENT=/home/rafah/projects/ofm/.envs/cuda`. uv 0.10.10 created that
  environment on 2026-05-25. Its interpreter resolves through uv's link
  `cpython-3.11-linux-x86_64-gnu` (recreated 2026-09-12) to
  `/home/rafah/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu`, installed 2026-05-23
  and the only Python 3.11 installed. During the build, job 20812051 (2026-08-20 01:00:02, in
  the parallel Codex transcript of that night) logged `Using CPython 3.11.15 interpreter at:
  /home/rafah/projects/ofm/.envs/cuda/bin/python`. `/home/rafah/.local/bin/uv` is 0.10.10,
  dated 2026-03-13.
- The build ran on `node1900`: x86_64, Linux `4.18.0-553.147.1.el8_10.x86_64` according to
  `scontrol` on 2026-11-18. The installed native wheels are manylinux x86_64 builds, for example
  `tokenizers` `cp39-abi3-manylinux_2_17_x86_64`, `numpy` `cp311-cp311-manylinux_2_27_x86_64` and
  `pyarrow` `cp311-cp311-manylinux_2_28_x86_64`.
- Slurm recorded neither the batch script nor the submission environment
  (`sacct --batch-script` and `--env-vars` print `NONE`). The launcher's own exports are the
  known environment; `sbatch` passed on the rest of the submitting shell's environment, which is
  unrecorded.
- All 134 installed distributions date from 2026-05-25: none was installed or replaced after
  that. Every installed version appears in [`uv.lock`](uv.lock). Of the lock's packages, only
  `colorama`, `nest-asyncio`, `tzdata` and the project itself are not installed. `uv run` also
  installs the default `dev` group (`pytest`).
- At `8977f529`, `uv.lock` (blob `d354281988523dc77d5943b0eb3e6c642c385850`) removes 29
  packages (`torch`, `triton`, `tyro` and their dependencies) and changes no version, and
  `pyproject.toml` drops `torch` and `tyro`.

<details>
<summary>Installed packages: <code>uv pip freeze --python /home/rafah/projects/ofm/.envs/cuda/bin/python</code> (the environment has no pip)</summary>

```text
absl-py==2.4.0
aiofiles==25.1.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.5
aiosignal==1.4.0
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.13.0
array-record==0.8.3
attrs==26.1.0
certifi==2026.4.22
charset-normalizer==3.4.7
click==8.4.0
cloudpickle==3.1.2
contourpy==1.3.3
cuda-bindings==13.2.0
cuda-pathfinder==1.5.4
cuda-toolkit==13.0.2
cycler==0.12.1
datasets==4.8.5
dill==0.4.1
docstring-parser==0.18.0
etils==1.14.0
filelock==3.29.0
flax==0.12.7
fonttools==4.63.0
frozenlist==1.8.0
fsspec==2026.2.0
gitdb==4.0.12
gitpython==3.1.50
grain==0.2.16
h11==0.16.0
hf-xet==1.5.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==1.15.0
humanize==4.15.0
idna==3.15
iniconfig==2.3.0
jax==0.10.0
jax-cuda12-pjrt==0.10.0
jax-cuda12-plugin==0.10.0
jaxlib==0.10.0
jinja2==3.1.6
kiwisolver==1.5.0
markdown-it-py==4.2.0
markupsafe==3.0.3
matplotlib==3.10.9
mdurl==0.1.2
ml-dtypes==0.5.4
mpmath==1.3.0
msgpack==1.1.2
multidict==6.7.1
multiprocess==0.70.19
networkx==3.6.1
numpy==2.4.6
nvidia-cublas==13.1.1.3
nvidia-cublas-cu12==12.9.2.10
nvidia-cuda-cccl-cu12==12.9.27
nvidia-cuda-cupti==13.0.85
nvidia-cuda-cupti-cu12==12.9.79
nvidia-cuda-nvcc-cu12==12.9.86
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-nvrtc-cu12==12.9.86
nvidia-cuda-runtime==13.0.96
nvidia-cuda-runtime-cu12==12.9.79
nvidia-cudnn-cu12==9.22.0.52
nvidia-cudnn-cu13==9.20.0.48
nvidia-cufft==12.0.0.61
nvidia-cufft-cu12==11.4.1.4
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusolver-cu12==11.7.5.82
nvidia-cusparse==12.6.3.3
nvidia-cusparse-cu12==12.5.10.65
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu12==2.30.4
nvidia-nccl-cu13==2.29.7
nvidia-nvjitlink==13.0.88
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvshmem-cu12==3.6.5
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
opt-einsum==3.4.0
optax==0.2.8
orbax-checkpoint==0.11.40
packaging==26.2
pandas==3.0.3
pillow==12.2.0
platformdirs==4.9.6
pluggy==1.6.0
prometheus-client==0.25.0
propcache==0.5.2
protobuf==7.35.0
psutil==7.2.2
pyarrow==24.0.0
pydantic==2.13.4
pydantic-core==2.46.4
pygments==2.20.0
pyparsing==3.3.2
pytest==9.0.3
python-dateutil==2.9.0.post0
pyyaml==6.0.3
regex==2026.5.9
requests==2.34.2
rich==15.0.0
safetensors==0.7.0
scipy==1.17.1
sentry-sdk==2.60.0
setuptools==81.0.0
shellingham==1.5.4
simplejson==4.1.1
six==1.17.0
smmap==5.0.3
sympy==1.14.0
tensorstore==0.1.84
tokenizers==0.22.2
torch==2.12.0
tqdm==4.67.3
transformers==5.8.1
treescope==0.1.10
triton==3.7.0
typeguard==4.5.2
typer==0.25.1
typing-extensions==4.15.0
typing-inspection==0.4.2
tyro==1.0.13
urllib3==2.7.0
uvloop==0.22.1
wandb==0.27.0
xxhash==3.7.0
yarl==1.24.2
zipp==4.1.0
```

</details>

## Hugging Face inputs

- Dataset `HuggingFaceFW/fineweb-edu`, configuration `sample-100BT`, split `train`, revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, streamed; ofm's datasets cache is empty. The
  dataset card at that revision, `README.md`, which maps `sample-100BT` to its data files, is in
  ofm's hub cache (Git blob `16e64d907529c7f742578a2fee4b72039d64211d`, 26,354 bytes, SHA256
  `a0cc8998a20499432b28b6575f3046b714938eb8e11b8d59a1d25ddf3716061e`, cached 2026-05-25). At
  2026-08-20 00:43:33 the build recorded in the same cache that the revision has no
  `dataset_infos.json`, `fineweb-edu.py` or `.huggingface.yaml`.
- Tokenizer `NousResearch/Llama-2-7b-hf`, revision `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`,
  read from ofm's Hugging Face cache, where its files date from 2026-05-25:

```text
f7b50bcf6d6672eade5e43514d48e9c1e4e63a56aef7b14acdaca94ce93436f7  tokenizer.json
9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347  tokenizer.model
37951d5a1e0c6a77f88c61f79972f17e889b62258aadff4c61eacd3fb567fe72  tokenizer_config.json
2ea88203b955648951a454e73a73967ff77ba4e79ae8ceb4400d4ab2cb040777  special_tokens_map.json
3c66a81e29ce617a74ff2e4055e7fb275b5c9457103864745693fbce59b368a2  config.json
```

## Evidence

Slurm accounting, queried 2026-11-18:

```text
$ sacct -D -X -P -j 20811682,20814071 -o JobID,JobName,Partition,State,ExitCode,Restarts,Submit,Start,End,Elapsed,NodeList,AllocCPUS,ReqMem,WorkDir,SubmitLine
JobID|JobName|Partition|State|ExitCode|Restarts|Submit|Start|End|Elapsed|NodeList|AllocCPUS|ReqMem|WorkDir|SubmitLine
20811682|ofm_fineweb_12b|mit_preemptable|COMPLETED|0:0|0|2026-08-20T00:42:13|2026-08-20T00:42:58|2026-08-20T02:20:40|01:37:42|node1900|16|64G|/home/rafah/projects/ofm/worktrees/data_12b|sbatch scripts/fineweb_12b/materialize.sbatch
20814071|validate_fw12b|mit_quicktest|COMPLETED|0:0|0|2026-08-20T02:22:46|2026-08-20T02:22:48|2026-08-20T02:24:32|00:01:44|node1603|2|8G|/home/rafah/projects/ofm/worktrees/data_12b|sbatch /home/rafah/projects/ofm/.caches/tmp/validate_fineweb_12b.sbatch
$ sacct -P -j 20811682,20814071 -o JobID,State,ExitCode,Elapsed,MaxRSS
JobID|State|ExitCode|Elapsed|MaxRSS
20811682|COMPLETED|0:0|01:37:42|
20811682.batch|COMPLETED|0:0|01:37:42|34996344K
20811682.extern|COMPLETED|0:0|01:37:42|
20814071|COMPLETED|0:0|00:01:44|
20814071.batch|COMPLETED|0:0|00:01:44|8384116K
20814071.extern|COMPLETED|0:0|00:01:44|
```

- [`runs/fineweb_12b/REPORT.md`](runs/fineweb_12b/REPORT.md): the build's own record of both
  jobs, the manifests' SHA256s and the validation.
- The transcript of the Codex agent that ran the build:
  `/home/rafah/.codex/sessions/2026/08/20/rollout-2026-08-20T00-37-46-01a01d75-f9d3-7241-b1e8-d5dcdbb3ad7f.jsonl.zst`,
  367,190 bytes (1,334,433 uncompressed), SHA256
  `e6ec9ca98cfd77aaa6064f56be0de365c7c35a31a5d7a459f2784d4d58161bd2`. It is not copied: it is a
  conversation log that also covers unrelated work. It records the worktree's creation at a
  clean `712d451`, the submission, tails of the job's standard error (tqdm progress only), and
  the validation script and its output.
- Validation job 20814071 ran `/home/rafah/projects/ofm/.caches/tmp/validate_fineweb_12b.sbatch`
  from the worktree. It loaded both manifests with `data.packed.load_packed_manifest`, which
  recomputes every shard's SHA256 and checks row continuity and counts. It also compared the
  manifests' SHA256s with `corpus.json`, checked the manifests' sequence length, dtype, counts
  and source and tokenizer metadata, required each split directory to hold only its shards, `manifest.json` and
  `progress.json`, and found no `*.tmp` file. Its output, from the transcript, follows;
  `preserved_eval` is the older `sample-10BT` evaluation split, kept for this comparison and
  then deleted.

```json
{
  "eval": {
    "manifest_sha256": "0e8615538d9dccad17956d8d9839394d05a56f8ec210dc67716db829a543634f",
    "num_sequences": 4888,
    "num_shards": 1,
    "prediction_tokens": 10010624,
    "stored_tokens": 10015512
  },
  "preserved_eval": {
    "manifest_sha256": "40e17c20aff7df1264d53ce3394a58af54178470dcd3a4cd290172345e8e6889",
    "matches_new_eval_manifest": false,
    "matches_new_eval_shard": false,
    "num_sequences": 4888,
    "num_shards": 1,
    "prediction_tokens": 10010624,
    "stored_tokens": 10015512
  },
  "total_bytes": 24035486562,
  "train": {
    "manifest_sha256": "9de775ed836ab7c61857207335a981c09630caa3226d2dc0ee0b799cdcb2f4f4",
    "num_sequences": 5859375,
    "num_shards": 90,
    "prediction_tokens": 12000000000,
    "stored_tokens": 12005859375
  }
}
```

- Job 20869563 (`ofm_materialize`, 2026-08-20 18:04:42 to 18:16:42 -0400 on `node2116`) ran
  `sbatch --parsable scripts/materialize_fineweb/materialize_fineweb.sbatch configs/data/fineweb_12b.yaml`
  in the ofm checkout, whose `main` (`3d6000a`) did not yet contain that configuration: it ran
  uncommitted versions of the later files, which `8972d52` committed at 18:21 and whose text
  then is unrecorded. It completed `0:0` with 51.5 s of CPU and `MaxDiskRead` 21792.72M, and
  changed no corpus file (every file's change time is from the build). That matches the
  materializer's check of a completed corpus, which re-hashes every shard against the manifests
  and compares the configuration with the manifests' metadata. It did not build the corpus.
- ofm tracks `datasets/fineweb_12b/README.md`, `corpus.json` and both manifests (added in
  `8972d52`; equal to the files on disk). That README, which the builder does not write, names
  the later configuration and launcher under "Construction". Of the build's output only the
  shards, the manifests and `corpus.json` remain on disk.

Not found: the job's logs (`worktrees/data_12b/runs/fineweb_12b/data/logs/materialize-20811682.out`
and `.err`; Slurm's own output went to `/dev/null`), its resume state
(`runs/fineweb_12b/data/train/progress.json` and `eval/progress.json`), and the validation
script and its logs (`.caches/tmp/validate_fineweb_12b.sbatch` and
`.caches/tmp/fineweb-12b-validation-20814071.out` and `.err`, deleted after validation). None
remains under `/home/rafah/projects/ofm`, searched to depth 7 outside `.git`, the environments,
the caches and checkpoints.

## Manifest check

On ORCD, 2026-11-18:

```text
$ sha256sum /home/rafah/projects/ofm/datasets/fineweb_12b/train/manifest.json /home/rafah/projects/ofm/datasets/fineweb_12b/eval/manifest.json
9de775ed836ab7c61857207335a981c09630caa3226d2dc0ee0b799cdcb2f4f4  /home/rafah/projects/ofm/datasets/fineweb_12b/train/manifest.json
0e8615538d9dccad17956d8d9839394d05a56f8ec210dc67716db829a543634f  /home/rafah/projects/ofm/datasets/fineweb_12b/eval/manifest.json
```

Both equal the SHA256s this repository records. The two manifests and `corpus.json`
(`540b74482f962907392216580d128ff581a7fc112d28ce1178c0841500eacea1`) are byte-identical to
[`../train/manifest.json`](../train/manifest.json), [`../eval/manifest.json`](../eval/manifest.json)
and [`../corpus.json`](../corpus.json).

## Later versions

[`../ofm-8977f529/`](../ofm-8977f529/) holds the checked-out versions of the builder files
changed or added after the build:

- `configs/data/fineweb_12b.yaml`: the build's parameters as a configuration. Every value
  equals the build's flags, with the materializer's default `text_field: text`, except
  `output_dir: /home/rafah/projects/ofm/datasets/fineweb_12b`. The manifests and `corpus.json`
  record only relative paths.
- `scripts/materialize_fineweb/materialize_fineweb.py`: takes that configuration instead of
  flags, so it imports `yaml`, and deletes `progress.json` after writing `corpus.json`. Its
  partition, shuffle, tokenization and writing code is the build's.
- `scripts/materialize_fineweb/materialize_fineweb.sbatch`: runs the materializer on a
  configuration, `sbatch scripts/materialize_fineweb/materialize_fineweb.sbatch configs/data/fineweb_12b.yaml`,
  logging to `.caches/tmp/materialize-<job>/`, which it deletes on success.

They use the same `data/` package. Their environment differs from the build's only by the
removals above.

## Running the copies

- The launchers hard-code ORCD's project root, `/home/rafah/projects/ofm` (for `.caches` and
  `.envs`), and partition, `mit_preemptable`; the configuration holds ORCD's output path. The
  materializers and `data/` hold no cluster paths.
- The build launcher writes its output and logs to `runs/fineweb_12b/data` under the directory
  it is submitted from, about 24 GB; inside this repository that path is not ignored.
- No uv configuration applied to the build: ofm has no `uv.toml` or `[tool.uv]`, and there is
  no user or system `uv.toml`. The copied `pyproject.toml` has no `[tool.uv]` either, so uv run
  inside this repository also reads `scaffolds/transformer/pyproject.toml`'s: from `ofm/`,
  `uv pip freeze --show-settings` lists its `pytorch-cpu` index, and from the ofm project it
  lists no index.
- The launcher's `uv run` does not pass `--locked`. At the build the lock was current, and
  the job left `uv.lock` unchanged.
