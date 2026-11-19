#!/bin/bash
# Rebuilds fineweb_12b from Hugging Face with the builder that produced it (ofm/, see ofm/PROVENANCE.md),
# then compares the result with this directory's manifests and corpus.json.
#
#   rebuild.sh WORK [--setup-only] [--python PYTHON]
#
# WORK is expendable storage outside the repository with about 40 GB free: the corpus (about 24 GB) goes to
# WORK/data, and any environment, Python and caches stay under WORK. Run the build on a CPU node that reaches
# huggingface.co, not on a login node: 16 CPUs and 64 GB (peak RSS 30-34 GiB) take about 2 hours. The builder
# streams FineWeb-Edu for the whole run and resumes an interrupted build, so a requeue may rerun this script.
#
# By default the build's own environment is installed under WORK and checked against its freeze: uv 0.10.10,
# CPython 3.11.15 and the 134 locked distributions (needs PyPI and uv's Python downloads). --python PYTHON
# instead runs the builder with an existing interpreter and its installed packages; it reports how the
# packages that determine the bytes differ from the build's, and the final comparison decides. --setup-only
# stops after the environment check, which may run on a login node.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
WORK="$(realpath -m "${1:?usage: rebuild.sh WORK [--setup-only] [--python PYTHON]}")"
shift
SETUP_ONLY=0
PY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --setup-only) SETUP_ONLY=1 ;;
    --python) PY="$(command -v "${2:?--python needs an interpreter}")"; shift ;;
    *) echo "usage: rebuild.sh WORK [--setup-only] [--python PYTHON]" >&2; exit 2 ;;
  esac
  shift
done
case "$WORK/" in
  "$REPO/"*) echo "WORK must be outside the repository: uv would read its [tool.uv] settings" >&2; exit 1 ;;
esac
OUT="$WORK/data"
CPUS="${SLURM_CPUS_PER_TASK:-16}"

block() {  # the first text block after the line starting with $1 in PROVENANCE.md
  awk -v start="$1" 'index($0, start) == 1 {found = 1} found && /^```text$/ {inside = 1; next}
    inside && /^```$/ {exit} inside {print}' "$HERE/ofm/PROVENANCE.md"
}

# The builder's files, unmodified, run from a copy outside the repository.
(cd "$HERE" && sha256sum --check --quiet <(block '## Files'))
mkdir -p "$WORK"
rm -rf "$WORK/ofm"
cp -a "$HERE/ofm" "$WORK/ofm"
cd "$WORK/ofm"

if [ -z "$PY" ]; then
  # uv 0.10.10 and CPython 3.11.15 made the build's environment; neither touches the shared installs.
  export UV_CACHE_DIR="$WORK/uv-cache" UV_PYTHON_INSTALL_DIR="$WORK/python" UV_PROJECT_ENVIRONMENT="$WORK/env"
  export UV_LINK_MODE=copy
  if [ ! -x "$WORK/uv/bin/uv" ]; then
    python3 -m venv "$WORK/uv"
    "$WORK/uv/bin/pip" install --quiet uv==0.10.10
  fi
  UV="$WORK/uv/bin/uv"
  test "$("$UV" --version | cut -d' ' -f2)" = 0.10.10
  "$UV" python install --no-config 3.11.15
  "$UV" sync --no-config --locked --extra cuda --python 3.11.15
  # The environment must equal the build's, distribution for distribution.
  diff <("$UV" pip freeze --no-config --python "$WORK/env/bin/python") <(block '<summary>Installed packages')
  echo "environment: $("$UV" pip freeze --no-config --python "$WORK/env/bin/python" | wc -l) distributions equal the build's"
  PY="$WORK/env/bin/python"
  RUN=("$UV" run --no-config --locked --python 3.11.15 --extra cuda python)
else
  # An existing environment: the builder's imports must exist; versions are compared, not enforced.
  "$PY" - <(block '<summary>Installed packages') <<'EOF'
import importlib.metadata as metadata, sys
built = dict(line.strip().split("==") for line in open(sys.argv[1]) if "==" in line)
print("python:", sys.version.split()[0], "(build: 3.11.15)")
for name in ("datasets", "pyarrow", "fsspec", "huggingface-hub", "tokenizers", "transformers", "numpy", "tqdm"):
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        sys.exit(f"{name} is not installed; the builder needs it")
    print(f"{name} {version} (build: {built[name]})" + ("" if version == built[name] else "  DIFFERS"))
EOF
  RUN=("$PY")
fi
"$PY" -c 'import sys; print("python:", sys.version)'
[ "$SETUP_ONLY" = 1 ] && exit 0

# The build launcher's exports and flags (ofm/scripts/fineweb_12b/materialize.sbatch); only the output differs.
export OMP_NUM_THREADS="$CPUS" MKL_NUM_THREADS="$CPUS" TOKENIZERS_PARALLELISM=true
export HF_HOME="$WORK/hf" HF_DATASETS_CACHE="$WORK/hf/datasets" TRANSFORMERS_CACHE="$WORK/hf/transformers"
export XDG_CACHE_HOME="$WORK/xdg" TMPDIR="$WORK/tmp/${SLURM_JOB_ID:-local}" PYTHONPATH="$WORK/ofm"
mapfile -t FLAGS < <(sed -n '/materialize_fineweb\.py/,$p' scripts/fineweb_12b/materialize.sbatch |
  grep -oE -- '--[a-z-]+ [^ \\]+' | tr ' ' '\n')
test "${#FLAGS[@]}" -eq 30
mkdir -p "$OUT/logs" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"
"$PY" -c 'import urllib.request; urllib.request.urlopen("https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/revision/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", timeout=60)'
echo "$(date -u +%FT%TZ) materialize ${FLAGS[*]}"
"${RUN[@]}" scripts/materialize_fineweb/materialize_fineweb.py "$OUT" "${FLAGS[@]}" \
  >"$OUT/logs/materialize-${SLURM_JOB_ID:-local}.out" 2>"$OUT/logs/materialize-${SLURM_JOB_ID:-local}.err"
echo "$(date -u +%FT%TZ) materialized"

# The rebuild must equal the corpus byte for byte.
sha256sum "$OUT/train/manifest.json" "$OUT/eval/manifest.json" "$OUT/corpus.json"
cmp "$OUT/train/manifest.json" "$HERE/train/manifest.json"
cmp "$OUT/eval/manifest.json" "$HERE/eval/manifest.json"
cmp "$OUT/corpus.json" "$HERE/corpus.json"
(cd "$HF_HOME/hub/models--NousResearch--Llama-2-7b-hf/snapshots/8efe6c9b93655b934e27bd9981e3ec13e55aee9d" &&
  sha256sum --check --ignore-missing <(block '- Tokenizer '))
echo "fineweb_12b rebuilt byte for byte"
