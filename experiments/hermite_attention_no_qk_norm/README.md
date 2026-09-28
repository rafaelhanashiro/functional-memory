# Fixed Hermite attention without QK-norm

Fixed Hermite kernels (`p_m(s) = exp(1/2) sum_(r<=m) He_r(s)/r!`) at `m = 1–4` with plain
scores `s = q·k/sqrt(d_K)` (`qk_norm: false`) and selective weight decay, 200M, seed 0.
Normalized `m = 2` is evaluated as the square `exp(1/2) (s+1)^2 / 2`. The matched
QK-norm comparison, and a one-line summary of these runs, is in
[hermite_attention](../hermite_attention/README.md). These runs are not plotted.

Home cluster: Kennel. [Run index](runs.csv) · [Study facts](sweep.yaml). Each run's
recipe, record, metrics, retained checkpoint (completed runs only), and
`training.log` are under `runs/<run_id>/`; the source commits in the records hold
the executed code.
