# Reference initializations

Opened 2026-10-02. The reference mechanisms and every normalized fixed Taylor column
(`m = 1, 2, 3, 4, 8, 16`) in [`results/attention_scaling/`](../../attention_scaling/README.md)
now use these reference-initialization runs. Open decision: rerun the degree-chart
families, which still use Flax defaults.

The scaffold's Flax defaults draw Linear weights truncated normal with variance
`1/fan_in` and no depth scaling. Do they handicap the models against the recipes their
authors train with? Softmax and normalized fixed Taylor `m = 2`, `4`, `8`, and `16` at
every `d_K`, and softmax no QK-norm, unnormalized exp, Taylor `m = 1` and `3`, and GLA at
`d_K = 64`, repeat the
Flax-default recipes and change only
`model.initialization`
([scaffold](../../../scaffolds/transformer/README.md#weight-initialization)):

- **softmax, unnormalized exp, and Taylor**: `transformer_reference`, GPT-2/Megatron: N(0, 0.02), attention
  and MLP output projections N(0, 0.02/√(2L)).
- **GLA**: `gla_reference`, official GLA code: N(0, 0.02), Xavier gain 2^-2.5 on the
  query, key, and gate projections.

Runs and roles: [`selection.yaml`](selection.yaml).

## Results

Terminal validation perplexity:

| Model | d_K | Flax defaults | Reference init | Change |
| --- | ---: | ---: | ---: | ---: |
| Softmax | 8 | 17.224 | **15.639** | −1.59 |
| Softmax | 16 | 15.591 | **14.335** | −1.26 |
| Softmax | 32 | 14.865 | **13.762** | −1.10 |
| Softmax | 64 | 14.457 | **13.514** | −0.94 |
| Softmax | 128 | 14.225 | **13.303** | −0.92 |
| Softmax no QK-norm | 64 | 14.665 | **13.932** | −0.73 |
| Unnormalized exp | 64 | 18.304 | **15.885** | −2.42 |
| Taylor m = 1 | 64 | fail @31 | fail @35 | |
| Taylor m = 2 | 8 | 22.440 | **19.491** | −2.95 |
| Taylor m = 2 | 16 | 19.310 | **17.154** | −2.16 |
| Taylor m = 2 | 32 | 17.559 | **15.794** | −1.77 |
| Taylor m = 2 | 64 | 16.511 | **14.911** | −1.60 |
| Taylor m = 2 | 128 | 15.800 | **14.329** | −1.47 |
| Taylor m = 3 | 64 | fail @419 | fail @555 | |
| Taylor m = 4 | 8 | 19.067 | **17.063** | −2.00 |
| Taylor m = 4 | 16 | 16.756 | **15.212** | −1.54 |
| Taylor m = 4 | 32 | 15.607 | **14.278** | −1.33 |
| Taylor m = 4 | 64 | 14.896 | **13.774** | −1.12 |
| Taylor m = 4 | 128 | 14.438 | **13.455** | −0.98 |
| Taylor m = 8 | 8 | 17.743 | **16.081** | −1.66 |
| Taylor m = 8 | 16 | 15.871 | **14.563** | −1.31 |
| Taylor m = 8 | 32 | 15.018 | **13.854** | −1.16 |
| Taylor m = 8 | 64 | 14.498 | **13.543** | −0.96 |
| Taylor m = 8 | 128 | 14.199 | **13.288** | −0.91 |
| Taylor m = 16 | 8 | 17.341 | **15.749** | −1.59 |
| Taylor m = 16 | 16 | 15.649 | **14.357** | −1.29 |
| Taylor m = 16 | 32 | 14.890 | **13.775** | −1.12 |
| Taylor m = 16 | 64 | 14.454 | **13.519** | −0.94 |
| Taylor m = 16 | 128 | 14.221 | **13.307** | −0.91 |
| GLA | 64 | 15.895 | **15.376** | −0.52 |

Gap by update at `d_K = 64` (reference minus Flax defaults):

| Update | 1,000 | 5,000 | 10,000 | 15,000 | 19,074 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Softmax | −38.5 | −2.28 | −1.18 | −0.98 | −0.94 |
| GLA | −8.56 | −1.86 | −0.80 | −0.56 | −0.52 |

- **The reference initializations train better, not just faster.** The gap shrinks
  early but levels off: softmax stays near −0.95 over the last 4,000 updates.
- **Softmax and Taylor `m = 8`, `16` gain at every width, most at small `d_K`**
  (−1.59 to −1.66 at 8, −0.91 to −0.92 at 128). The gains match within 0.1 at each
  width, so Taylor's standing against softmax is unchanged: `m = 16` stays within 0.03
  of softmax at `d_K >= 16`.
- **Taylor `m = 2` and `4` gain more** (−2.95 and −2.00 at `d_K = 8`, −1.47 and −0.98
  at 128), so they close part of their gap to softmax: at `d_K = 8` from 5.22 to 3.85
  (`m = 2`) and from 1.84 to 1.42 (`m = 4`).
- **Unnormalized exp gains the most (−2.42).** Its attention output sums over all
  visible keys, so with Flax defaults the residual stream reaches about 1,300 at
  initialization (length 512); the depth-scaled output projections shrink that about 10×.
- **Against Gated DeltaNet** (paper initialization), reference-initialized softmax
  is ahead at `d_K >= 32` (by 0.19, 0.19, 0.08), level at 16 (+0.05), and behind
  at 8 (+0.79).
- **Odd-degree normalized Taylor still fails** (row sums cross zero), a few updates to
  136 updates later than with Flax defaults.
- **The degree-chart polynomial kernels are likely handicapped too.** Every normalized
  fixed Taylor degree that finishes gained as much as softmax (within 0.01) or more; until those
  families are rerun, compare them with the references only at equal initialization.

## Limits

- One seed per run.
- Reference-initialization softmax and unnormalized exp runs trained on B300, except
  softmax `d_K = 8`, softmax no QK-norm, and GLA (H100). The Taylor `m = 1`, `3`, `8`, and
  `16` runs trained on H100, except `d_K = 16`, `m = 16` and `d_K = 64, 128`, `m = 8`,
  which resumed across H100 and B300 after preemptions. Taylor `m = 2` and `4` trained
  on ORCD (2×H200); their Flax-default controls trained on Kennel. The Flax-default GLA run and softmax at
  `d_K = 8, 16, 32, 128` trained on ORCD.
