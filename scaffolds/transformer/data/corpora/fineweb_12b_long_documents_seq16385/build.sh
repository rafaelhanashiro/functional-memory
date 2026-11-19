#!/bin/bash
# Builds fineweb_12b_long_documents_seq16385 from the long-documents-dedup worktree checkout at 2ee60d77a3581f2a75f318612d8351f951671c77.
set -euo pipefail
cd "/data/cl/u/rafah/home/projects/functional-memory/worktrees/long-documents-dedup"
echo "source_commit=$(git rev-parse HEAD)"
test "$(git rev-parse HEAD)" = 2ee60d77a3581f2a75f318612d8351f951671c77
git diff --quiet HEAD -- scaffolds/transformer/data
for split in eval train; do
  echo "$(date -u +%FT%TZ) building $split"
  PYTHONPATH="/data/cl/u/rafah/home/projects/functional-memory/worktrees/long-documents-dedup" JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m scaffolds.transformer.data.long_documents \
    /data/cl/u/rafah/home/datasets/fineweb_12b/$split/manifest.json /data/cl/u/rafah/home/datasets/fineweb_12b_long_documents_seq16385/$split --sequence-length 16385 --opening-tokens 1024
done
echo "$(date -u +%FT%TZ) done"
