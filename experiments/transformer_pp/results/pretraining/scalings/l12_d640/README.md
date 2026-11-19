# The 12-layer width-640 shape

Transformer++ results at the 12-layer width-640 shape: 12 layers of width 640 with 10 heads (98,977,016
parameters for softmax), whose 1× Chinchilla budget is 7,551 updates of 128 × 2,048 tokens. Each recipe is a
partner's recipe derived with [`derive_recipe.py`](../../../../derive_recipe.py); the runs live in the study's
`runs/` and `runs.csv` like every other run. Two folders compare them:

| Collection | Question | Selected runs |
| --- | --- | ---: |
| [Shape pairs](shape_pairs/README.md) | Do the attention-variant comparisons hold at this shape? Eight variants at 1×, each against its partner at the 15-layer width-896 reference shape. | 16 |
| [Budgets](budgets/README.md) | How do the variants compare as the budget grows? At 2×, the reference shape's 1× token count, against exp, their 1× runs and the reference shape at the same tokens; at 4×, the five degree-twelve RoPE polynomials and softmax against exp and their 2× runs. | 40 |

Three other folders also hold runs at this shape, beside their reference-shape runs:

- [softplus_sos](../../parametrizations/sos/softplus_sos/README.md): softplus inputs composed with query sum of squares, at
  1×, 2× and 4× here.
- [score_transforms](../../../exploratory/score_transforms/README.md): the degree-eight query polynomial evaluated at
  `softplus(s)` or `|s|` in place of the score.
- [score_biases](../../../exploratory/score_biases/README.md): a signed score shift for every order of the degree-eight
  query polynomial, kept as a negative check.
