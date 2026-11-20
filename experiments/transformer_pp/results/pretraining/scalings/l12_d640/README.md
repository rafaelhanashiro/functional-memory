# The 12-layer width-640 shape

Transformer++ results at the 12-layer width-640 shape: 12 layers of width 640 with 10 heads (98,977,016
parameters for softmax), whose 1× Chinchilla budget is 7,551 updates of 128 × 2,048 tokens. Each recipe is a
partner's recipe derived with [`derive_recipe.py`](../../../../derive_recipe.py); the runs live in the study's
`runs/` and `runs.csv` like every other run. Every pretraining comparison at this shape has its folder here,
whatever its mechanism; one that also ran at the reference shape keeps those runs in its mechanism's group,
and each part links the other.

| Folder | Question | Selected runs |
| --- | --- | ---: |
| [Shape pairs](shape_pairs/README.md) | Do the attention-variant comparisons hold at this shape? Eight variants at 1×, each against its partner at the 15-layer width-896 reference shape. | 16 |
| [Budgets](budgets/README.md) | How do the variants compare as the budget grows? At 2×, the reference shape's 1× token count, against exp, their 1× runs and the reference shape at the same tokens; at 4×, the five degree-twelve RoPE polynomials and softmax against exp and their 2× runs. | 40 |
| [Softplus SoS](softplus_sos/README.md) | How do the two degree-twelve compositions of softplus inputs with query sum of squares compare with exp and the other polynomials at 1×, 2× and 4×? Reference shape: [softplus_sos](../../parametrizations/sos/softplus_sos/README.md). | 18 |
| [Score transforms](score_transforms/README.md) | Does the degree-eight query polynomial improve evaluated at `softplus(s)` or the score's absolute value in place of the score? Reference shape: [softplus_inputs](../../parametrizations/sos/softplus_inputs/README.md). | 4 |
| [Score biases](score_biases/README.md) | Does a signed score shift for every order of the degree-eight query polynomial help? Kept as a negative check. | 3 |
