# Corpora

Every packed-data run in this repository (the `attention_scaling` and `transformer_pp`
studies) trains and validates on `fineweb_12b`, a packed FineWeb-Edu corpus, or on a corpus
derived from it for longer contexts and long-document evaluation. The corpora stay outside Git. This
directory holds each corpus's manifests, byte copies of those on disk, and its construction
record. A manifest lists every shard with its SHA256, so a corpus whose manifests equal these
byte for byte is the same corpus.

| Corpus | Built from | Train rows × length | Eval rows × length | Shards | Manifest SHA256, train / eval |
| --- | --- | --- | --- | ---: | --- |
| `fineweb_12b` | FineWeb-Edu | 5,859,375 × 2,049 | 4,888 × 2,049 | 90 / 1 | `9de775ed` / `0e861553` |
| `fineweb_12b_seq4097` | `fineweb_12b`, repacked | 2,930,402 × 4,097 | 2,444 × 4,097 | 90 / 1 | `ed22293a` / `fed8b3f8` |
| `fineweb_12b_seq16385` | `fineweb_12b` eval, repacked | – | 611 × 16,385 | – / 1 | – / `ec83c04e` |
| `fineweb_12b_long_documents_seq16385` | `fineweb_12b`, long documents | 27,962 × 16,385 | 18 × 16,385 | 28 / 1 | `db96b415` / `355b76e9` |

On Kennel each corpus is at `/data/cl/u/rafah/home/datasets/<corpus>/`; ORCD's `fineweb_12b`
has the same manifests ([CLUSTERS.md](../../../../CLUSTERS.md)).

## fineweb_12b

Its manifests' `metadata` records the build:

- Source: Hugging Face `HuggingFaceFW/fineweb-edu`, configuration `sample-100BT`, split
  `train`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, field `text`, one pass over
  the source (`source_epochs_scanned: 1`).
- Order: a streaming shuffle with a 10,000-document buffer and seed 0.
- Split: each document goes to train or eval by the SHA256 of its text
  (`partition.algorithm: sha256-text`), 1% to eval.
- Tokens: `NousResearch/Llama-2-7b-hf` at revision `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`
  (vocabulary 32,000, stored as uint16). Each document contributes its tokens without a
  beginning-of-sequence token (id 1), followed by the end-of-sequence token 2. There is no
  padding, and each split's stream begins with a document's first token.
- Rows: each split's stream cut into consecutive, disjoint 2,049-token rows: 2,048 prediction
  positions, with a row's targets its own tokens shifted by one. Train: 5,859,375 rows
  (12,000,000,000 prediction positions) in shards of 65,536 rows, the 90th holding 26,671.
  Eval: 4,888 rows (10,015,512 tokens) in one shard.
- Files: `<split>/shard-NNNNN.npy`, uint16 `[rows, 2049]`; `<split>/manifest.json`, format
  `ofm-packed-sequences` version 1, listing the shards in order with their first row, row count
  and SHA256; `corpus.json`, naming both manifests with their SHA256s.

The builder that produced it is in [`fineweb_12b/ofm/`](fineweb_12b/ofm/), copied unchanged from
the `ofm` project at commit `712d451`: the Slurm launcher `scripts/fineweb_12b/materialize.sbatch`,
which runs `scripts/materialize_fineweb/materialize_fineweb.py` with `data/packed.py`, and its
locked environment, `pyproject.toml` and `uv.lock`. [`PROVENANCE.md`](fineweb_12b/ofm/PROVENANCE.md)
records the build. [`fineweb_12b/ofm-8977f529/`](fineweb_12b/ofm-8977f529/) holds later ofm
versions of the materializer, its launcher and the configuration `configs/data/fineweb_12b.yaml`,
which did not build it. The facts above do not determine the build alone: the rule that maps a
text's hash to the 1% eval split, the shuffle and the tokenizer call belong to the builder, their
exact behavior to its environment's library versions, and the row counts (5,859,375 train, 4,888
eval) to its launcher's flags.

## Derived corpora

Built from `fineweb_12b` with this scaffold's tools. Each manifest's metadata
(`repacked_from`, `long_documents_from`) records its source manifest and that manifest's
SHA256.

- `fineweb_12b_seq4097` and `fineweb_12b_seq16385`: the same stream re-cut into longer rows,
  the trailing partial row dropped ([README](fineweb_12b_seq4097/README.md)):
  `python -m scaffolds.transformer.data.repack SOURCE/manifest.json DESTINATION --sequence-length 4097`
  (16385 for the eval-only `fineweb_12b_seq16385`).
- `fineweb_12b_long_documents_seq16385`: one row per document of at least 16,384 tokens,
  built by [`build.sh`](fineweb_12b_long_documents_seq16385/build.sh)
  ([README](fineweb_12b_long_documents_seq16385/README.md)).

## How runs read a corpus

- A recipe's `data` section names the corpus: `source: packed`, `train_manifest` and
  `eval_manifest` (absolute paths on the run's cluster), `sequence_length` equal to the
  manifest's row length, and `batch_size` in rows. `dataset_name`, `dataset_config`,
  `tokenizer_name` and `text_field` describe the corpus; the reader uses only the manifests.
  `shuffle_seed` affects synthetic test data only: packed rows are read in stored order.
- On load, [`packed.py`](../packed.py) checks each manifest's format and every shard's row
  count and SHA256, and refuses on any difference. The trainer writes the run's
  `data_identity.json` (both manifests' SHA256s, batch size, row length); the study's
  `sweep.yaml` pins the same SHA256s under `data_manifests`, and the launcher compares them at
  every start.
- Training: update `u`, counting from 0, reads train rows `[u·B, (u+1)·B)` for batch size `B`.
  Every run starts at row 0, so runs with one batch size read the same rows in the same order,
  and a run of `T` updates reads the first `T·B` rows; the train split holds 45,776 updates at
  `B` = 128 and 22,888 at 256. A resumed run continues from row `step·B`, and a continuation
  from row `parent_step·B` of its parent's corpus, whose identity it pins
  (`parent_data_identity_sha256`).
- Validation: the first `eval_batches·B` rows of the eval split at every evaluation.
  Transformer++ recipes evaluate 9,961,472 prediction positions: 4,864 rows of 2,049 (19
  batches of 256 or 38 of 128), or 2,432 rows of 4,097 at 4,096-token context. The
  `attention_scaling` recipes evaluate 1,792 rows of 2,049 (7 batches of 256).

## Reproducing a corpus elsewhere

1. Build `fineweb_12b` with [`fineweb_12b/rebuild.sh`](fineweb_12b/rebuild.sh) on a CPU node that
   reaches huggingface.co, not a login node: 16 CPUs, 64 GB, about 40 GB of expendable storage
   outside the repository, and about 2 hours.

   ```bash
   scaffolds/transformer/data/corpora/fineweb_12b/rebuild.sh WORK                  # the build's environment
   scaffolds/transformer/data/corpora/fineweb_12b/rebuild.sh WORK --python PYTHON  # an existing one
   ```

   It runs the builder in [`fineweb_12b/ofm/`](fineweb_12b/ofm/) from a copy under `WORK`, with its
   launcher's flags and exports, and writes the corpus to `WORK/data`. By default it installs the
   build's environment under `WORK`: uv 0.10.10, CPython 3.11.15 and the 134 distributions of
   `ofm/uv.lock`, which must equal the freeze in [`PROVENANCE.md`](fineweb_12b/ofm/PROVENANCE.md).
   That needs PyPI and uv's Python downloads. `--python` uses an existing interpreter and its
   packages instead, and reports how `datasets`, `pyarrow`, `fsspec`, `huggingface-hub`,
   `tokenizers`, `transformers` and `numpy` differ from the build's; other versions may change the
   bytes. Either way the script ends by comparing the manifests and `corpus.json` with this
   directory's byte for byte and the downloaded tokenizer files with their recorded SHA256s.
2. Check early: each finished shard's entry appears in `WORK/data/<split>/progress.json`, the
   eval shard and train shard 0 within about 10 minutes, and must equal the same shard in this
   directory's manifest. On a mismatch, the first differing shard locates it:

   ```bash
   python -c 'import json, sys; a, b = (json.load(open(p))["shards"] for p in sys.argv[1:]); \
   print(next(((x["path"], x["sha256"], y["sha256"]) for x, y in zip(a, b) if x != y), "shards equal"))' \
     NEW/train/manifest.json scaffolds/transformer/data/corpora/fineweb_12b/train/manifest.json
   ```
3. Build the derived corpora with the commands above. Their manifests' metadata records the
   source manifest's absolute path, so compare their shard entries and `num_tokens` rather than
   the whole file.
4. Point new recipes' `train_manifest` and `eval_manifest` at the new cluster's paths and record
   them in [CLUSTERS.md](../../../../CLUSTERS.md). The manifests' SHA256s, not the paths,
   identify the data.

Verified on Kennel on 2026-11-19: Slurm job 7262 (`mastiff`, 16 CPUs and 64 GB, 2 h 10 min, peak
RSS 30.0 GiB) ran `rebuild.sh` in the build's environment, equal to the freeze, and reproduced
`fineweb_12b` byte for byte: all 91 shards, both manifests (`9de775ed`, `0e861553`), `corpus.json`
(`540b7448`) and the five tokenizer files.
