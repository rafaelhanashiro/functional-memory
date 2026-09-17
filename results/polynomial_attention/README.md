# Polynomial attention sweep on Kennel

The active sweep requests **four H100 GPUs per run** on one node, in the
`long` partition. Preflight job **199422** is queued for resources. After all
four-GPU checks pass, it automatically submits the six production runs in
sequence: Taylor degrees 0, 2, 4, 8, 16, then exact softmax.

- Frozen source: `2484f00e9a32f1571b330fff5788d9a86175235f`.
- Each model has 199,906,304 parameters and trains for 19,074 updates,
  totaling 10,000,269,312 prediction tokens.
- Global batch 256, global microbatch 64, four accumulation steps, and
  2,048 prediction positions are unchanged. Four devices use 16 sequences
  per device per microbatch.
- Data, environments, caches, results, and checkpoints remain on NFS.
- Dataset verification passed for all 91 shards; the unchanged source passed
  138 CPU tests. Four-GPU acceptance must pass separately before production.

See `sweep.json` for the current immutable run specifications and job IDs,
and `allocation_change.json` for the allocation change provenance. The earlier
two-GPU jobs 199416–199421 were cancelled. Degree zero had completed about 202
updates without a checkpoint; it will restart from the same seed on four GPUs.
The superseded attempt is preserved under the archive path in that record.

The seven-day production limit, checkpoint/resume behavior, evaluation subset,
optimizer, initialization, and scientific recipe remain unchanged. Production
submission is blocked until every GPU pilot and both resume checks pass with
a projected runtime below six days per model.
