# FineWeb-Edu long documents at 16,385 tokens

Single-document rows for long-context evaluation. Every complete document of at least 16,384
tokens in the canonical `fineweb_12b` streams becomes one row, unless an earlier document in
the stream begins with the same 1,024 tokens: an end-of-document token, as precedes every
document in the stream, then the document's first 16,384 tokens. Position 0 is therefore a
document start, as in training, and no row mixes documents. The corpus is read-only and
shared in place; it is not an experiment run.

## Construction

[`build.sh`](build.sh) ran `scaffolds/transformer/data/long_documents.py` at
commit `2ee60d77` as Slurm job 234087 (CPU, 5 min 11 s; log in `logs/build.log` beside the corpus on disk):

```bash
python -m scaffolds.transformer.data.long_documents \
  /data/cl/u/rafah/home/datasets/fineweb_12b/train/manifest.json \
  /data/cl/u/rafah/home/datasets/fineweb_12b_long_documents_seq16385/train \
  --sequence-length 16385 --opening-tokens 1024
python -m scaffolds.transformer.data.long_documents \
  /data/cl/u/rafah/home/datasets/fineweb_12b/eval/manifest.json \
  /data/cl/u/rafah/home/datasets/fineweb_12b_long_documents_seq16385/eval \
  --sequence-length 16385 --opening-tokens 1024
```

Each source shard was checked against its manifest checksum while it was read. Every
manifest's `metadata.long_documents_from` records the source manifest and its SHA256, the
stream's token count, the number of indexed documents, the long documents left out because
their opening appeared earlier, the tokens after the last end-of-document token (a document
the stream cuts off, not indexed), and the SHA256s of the two index files. After the build,
document lengths, separators and that tail accounted for every stream token; all 27,962
training rows had distinct openings; and 38 training rows (random ones, the first and last,
both sides of the 1x boundary, and documents that cross a source shard) matched the source
stream token for token.

Web crawls repeat pages, sometimes revised: FineWeb removes duplicates within each Common
Crawl snapshot but not across them. In the training stream 161,170 of the 10,208,216
documents (1.6%) begin with the same 1,024 tokens as an earlier document, and 1,470 long
documents were left out as repeats.

| Split | Rows | Tokens | Shards | Indexed documents | Long repeats left out | Stream tokens | Unindexed tail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 27,962 | 458,157,370 | 28 | 10,208,216 | 1,470 | 12,005,859,375 | 764 |
| eval | 18 | 294,930 | 1 | 8,589 | 0 | 10,015,512 | 4,142 |

## Layout

- `<split>/manifest.json` and `<split>/shard-*.npy`: the rows, in the `ofm-packed-sequences`
  format that `PackedBatchIterator` reads (16,385 tokens per row).
- `<split>/documents.npy`: int64 `[documents, 3]`, every complete document's stream offset of
  its first token, its length without the end-of-document token, and the offset where its
  first 1,024 tokens first appear (its own offset if it is shorter than that), in stream order.
- `<split>/rows.npy`: int64 `[rows]`, each row's document (an index into `documents.npy`).

## Documents a run never saw

Every row is the first appearance of its 1,024-token opening, rows keep stream order, and
offsets count stream tokens, which repacking preserves. A run that trained on rows `[0, n)` of
a corpus with row length `L` (`n` = updates × batch rows, continuations included) consumed
stream tokens `[0, n * L)`. The rows it never saw by position, none of which repeats the
opening of a document it trained on, are the suffix from `first_unseen_row(manifest, n * L)` in
`scaffolds.transformer.data.long_documents`; `row_documents(manifest)` gives each row's
`[start, length, first_start]`. Revised copies whose opening differs are not detected: of the
first 608 rows unseen by the 1x runs (9,657–10,264), a scan of 64-token windows against the
first 3.97B stream tokens found 31 (5%) sharing at least half their windows with one earlier
document, 8 of them almost verbatim. The 18 held-out rows are unseen by every run but too few
for per-position curves.

| Training | Consumed stream tokens | First unseen row | Unseen rows |
| --- | ---: | ---: | ---: |
| 7,570 updates × 256 × 2,049 (1x Chinchilla at 2k) | 3,970,798,080 | 9,657 | 18,305 |
| 7,570 updates × 128 × 4,097 (1x Chinchilla at 4k) | 3,969,829,120 | 9,657 | 18,305 |
| 9,500 updates × 256 × 2,049 (continuations to update 9,500) | 4,983,168,000 | 12,155 | 15,807 |
| 18,925 updates × 256 × 2,049 (2.5x Chinchilla) | 9,926,995,200 | 23,297 | 4,665 |
| 20,000 updates × 256 × 2,049 | 10,490,880,000 | 24,608 | 3,354 |
