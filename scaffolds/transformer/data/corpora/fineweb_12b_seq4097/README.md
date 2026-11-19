# FineWeb-Edu packed at 4,097 tokens

The canonical `fineweb_12b` corpus re-cut into rows of 4,097 tokens (4,096 prediction
positions) for 4,096-token context runs. It is read-only and shared in place; it is not an
experiment run.

## Construction

`fineweb_12b` stores consecutive, disjoint 2,049-token slices of one token stream
(documents joined by EOS). Concatenating its rows in manifest order and cutting the stream
into 4,097-token rows gives the same tokens in the same order:

```bash
python -m scaffolds.transformer.data.repack \
  /data/cl/u/rafah/home/datasets/fineweb_12b/train/manifest.json \
  /data/cl/u/rafah/home/datasets/fineweb_12b_seq4097/train --sequence-length 4097
python -m scaffolds.transformer.data.repack \
  /data/cl/u/rafah/home/datasets/fineweb_12b/eval/manifest.json \
  /data/cl/u/rafah/home/datasets/fineweb_12b_seq4097/eval --sequence-length 4097
```

Each source shard was checked against its manifest checksum while it was read. The
trailing partial row was dropped. Every manifest's `metadata.repacked_from` records the
source manifest, its SHA256, its row length, and the dropped token count.

| Split | Rows | Tokens | Shards | Dropped tail tokens | Source manifest SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| train | 2,930,402 | 12,005,856,994 | 90 | 2,381 | `9de775ed836ab7c6...` |
| eval | 2,444 | 10,013,068 | 1 | 2,444 | (fineweb_12b/eval) |

Rows 0–1,999 of the train split equal rows 0–3,999 of the source byte for byte, and row
1,500,000 equals the source stream at token 6,145,500,000.

The train-split build log (Slurm job 231938) is `logs/repack-train.log` beside the corpus on disk.
