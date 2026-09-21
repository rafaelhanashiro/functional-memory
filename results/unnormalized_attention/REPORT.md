# 250M Transformer with unnormalized exponential attention

This source snapshot records the experiment setup. Live execution status and
metrics are written to `250m-exp-seed0/REPORT.md` and `250m-exp-seed0/metadata.json`
in the dedicated experiment checkout. A completed result requires its
`completion.json`; this setup report does not establish completion.

Question: what changes when the 250M Transformer uses `exp(QK^T/sqrt(d_k))V`
without the softmax row denominator?

The experiment uses 253,654,016 parameters, seed 0, 14 layers, width 1024,
16 full-MHA heads, FFN width 3008, and a 2,048-token context. Its canonical
configuration is `experiments/unnormalized_attention/configs/base.yaml`.
The 21,935-update recipe consumes 11,500,257,280 prediction tokens and matches
the historical OFM softmax control at source
`32ea743b7b3aed1d631116dfaa3c4b85101baf32`; see `baseline.json`.

The intended scientific change is removal of attention row normalization.
The implementation uses unshifted exponential weights with FP32 accumulation;
block RMSNorm, vocabulary softmax, optimizer, initialization, and data recipe
remain those of the baseline. A fused Pallas/Triton implementation makes the
unnormalized computation feasible without storing quadratic attention matrices.

Primary endpoint: terminal held-out perplexity at the matched token budget.
Historical normalized PPL is 13.3496. No result for the new model is available
yet. The comparison is single-seed and changes numerical kernel execution
(cuDNN for historical softmax, Pallas/Triton for unnormalized attention).

Next: complete output/gradient tests, exact initialization verification, and
full-recipe training/validation/resume preflight, then launch on 2 H100 GPUs.
The historical control used 2 H200s; allocation differences are recorded.
Retain validation-best weights and terminal metrics separately after review.
