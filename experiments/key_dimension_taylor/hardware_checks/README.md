# Validation across Blackwell and Ampere

This validation-only wrapper checks all 20 pairs of dK=8,16,32,64,128 and
m=2,4,8,16 on B300, A100, and RTX A6000. It cannot submit production training.
Scientific code is imported from the original frozen training checkout. Its
configuration, data, reference checksums, and source remain authoritative.

Each GPU model runs the existing 105 attention tests, then a 32-update full-model
pilot for every coordinate, evaluation/checkpoint save, and a fresh-process
continuation to update 33. Pilots keep four GPUs, the production model and
per-device batch/context, and the original full learning-rate schedule. The
existing B300 kernel suite and five full-model checks are reused by immutable
references and SHA-256 checksums. Four dK64 pilot configs are resolved from the
current recipe; their historical training results and provenance remain intact.

Prepare the policy with `manage.py --policy PATH --prepare ORIGINAL_MANIFEST
--reuse-policy B300_EXECUTION_POLICY`, then queue with `--policy PATH --submit`.
The checkout must be committed and clean before preparing the policy. Three
independent four-GPU jobs use `sweep` without a hostname pin; each validation
allocation is four hours. This permits A6000 allocation on either supported node.
The combined allocation retains its GPUs between kernel tests and model pilots.
Existing production jobs are not moved, cancelled, or duplicated.

Results, resolved pilot configs, audits, and a summary live under
`results/key_dimension_taylor/hardware_checks/`. Pilot work directories are
isolated from all production metrics and checkpoints. Hardware/source evidence
and per-coordinate outcomes are retained; temporary pilot checkpoints are
removed after recording evidence. Numerical errors, insufficient GPU memory,
runtime ineligibility, and infrastructure failures are distinct outcomes.
Successful preliminary validation does not establish full-training stability
or produce terminal perplexity measurements.
