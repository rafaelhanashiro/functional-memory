# Learning rate

Opened 2026-10-04. Open decision: find the best shared peak learning rate and rerun the
canonical comparisons with it, or discard.

The shared recipe peaks at 3e-4 (cosine to 3e-5), a scaffold default never tuned at this
scale or for the reference initialization. GPT-3 trained 125M and 350M models at 6e-4 and
3e-4; GLA's 340M runs used 3e-4, Gated DeltaNet's 1.3B runs 4e-4. One run repeats the
canonical QK-norm softmax at `d_K = 64` with the transformer reference initialization and
changes only the peak to 6e-4 (floor 6e-5, the same 10%). Runs and roles:
[`selection.yaml`](selection.yaml).

## Results

| Peak learning rate | Terminal PPL |
| --- | ---: |
| 3e-4 (`rbde7fc4c2135`, canonical) | 13.514 |
| 6e-4 (`ra846aac754ee`) | **13.015** |

Gap by update (6e-4 minus 3e-4):

| Update | 1,000 | 2,000 | 5,000 | 10,000 | 14,000 | 18,000 | 19,074 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gap | −17.0 | −2.86 | −0.84 | −0.48 | −0.47 | −0.50 | −0.50 |

- **3e-4 is too low for softmax.** Doubling it gains 0.50, steady over the last 9,000
  updates, so it is not only faster training. Neither run has a loss spike.
- **The optimum may be higher still.** Only one point above 3e-4 was run.
- **Whether the gain is shared is unknown.** If other mechanisms gain differently, a
  shared retune could change comparisons, as the initialization did; unnormalized
  `m = 1` already shows loss spikes at 3e-4.

## Limits

- One seed per run.
- The 6e-4 run trained on H100 and the 3e-4 run on B300.
