# Structural zero constants

Opened 2026-11-18. Does a polynomial improve when its constant is structurally zero throughout
training, rather than starting at zero or being removed by an input prefactor? Degree eight at the
15-layer width-896 reference shape, on softplus inputs and on raw scores.

Added 2026-11-18. The three new degree-eight runs test whether the constant must
remain zero throughout training, rather than merely starting at zero or removing
the baseline through an extra input prefactor. Their
[selection](selection.yaml) pins each run and its controls;
[methods and launch instructions](../../../../README.md#softplus-polynomials-with-structural-zero-constants)
specify the shared recipe.

| Variant | Run | Weight, z=softplus(s) | Learned generator coordinates |
| --- | --- | --- | --- |
| Query | `rb5b604ae829a` | sum from r=1 to 8 of c_r(x) z^r | r1..r8 |
| Static | `rc28ef4dbe26d` | sum from r=1 to 8 of c_r z^r | r1..r8 |
| Query SoS | `r7d5baf55104e` | U(z)² + V(z)², U0=V0=0 | U1..U4, V1..V3 |

All three have total degree eight, softplus inputs, RoPE, no L2 coefficient-input
normalization, and no prefactor. They retain the reference-shape recipe and
established WD policy. The removed constants have no trainable parameters, and
all surviving initial values, including W draws, match the recipe parents.
The two direct forms start at `T8(z)-1`. The SoS start is the sum of the squared
Taylor factors after removing each factor's constant; removing these constants
also changes higher expanded coefficients. Its expanded c0 and c1 remain zero.

All three trained on one B200. Validation perplexity, best with terminal in parentheses:

| Variant | Run | Best (terminal) | Control, best | vs control | vs exp |
| --- | --- | ---: | --- | ---: | ---: |
| Query | `rb5b604ae829a` | 14.250 (14.253) | `ra3b54e2489f8` 14.262 | −0.09% | −0.05% |
| Static | `rc28ef4dbe26d` | 14.192 (14.196) | `r9c69dd7cfc8d` 14.294 | −0.71% | −0.46% |
| Query SoS | `r7d5baf55104e` | 14.128 (14.130) | `r41a4c74c5f58` 14.339 | −1.47% | −0.91% |

- **The zero constant helps SoS** by 1.47% against the learned constant, to within
  0.23% of the degree-nine prefactor run (`r841f35a24241`, 14.095).
- **It does not help query:** −0.09% against the learnable constant that starts at zero.
- **Static** is 0.71% below its raw-score parent, a comparison that changes both the input
  and the constant.

## Raw scores

Added 2026-11-19. The same three structural zero constants on raw scores (`attention_score_transform:
identity`, every other setting of the softplus run), on the plain Mosaic query kernels. Each is compared with
its raw-score control, whose constant is learnable, and with its softplus version:

| Variant | Run | Raw control, best | Softplus version, best |
| --- | --- | --- | --- |
| Query | `r81b5ebb23a51` | `r25fffc16c2f3` 14.323 | `rb5b604ae829a` 14.250 |
| Static | `r37fb1c7bb6db` | `r9c69dd7cfc8d` 14.294 | `rc28ef4dbe26d` 14.192 |
| Query SoS | `rc379a8f30f5b` | `reeefbcc19abf` 14.196 | `r7d5baf55104e` 14.128 |

On raw scores the direct polynomials are signed, as their controls are; the SoS polynomial
s²(U(s)² + V(s)²) stays nonnegative. Query trained on one B300, static on one H100 PCIe, SoS on one B200.

| Variant | Run | Best (terminal) | vs raw control | vs softplus version |
| --- | --- | ---: | ---: | ---: |
| Query | `r81b5ebb23a51` | 16.960 (16.966) | +18.41% | +19.02% |
| Static | `r37fb1c7bb6db` | 17.359 (17.365) | +21.44% | +22.32% |
| Query SoS | `rc379a8f30f5b` | 14.726 (14.732) | +3.74% | +4.24% |

- **On raw scores the zero constant hurts every form**: the direct forms by 18–21%, SoS by 3.7%.
- The direct forms start at T_8(s) − 1, about s near zero: negative for every key with a negative score,
  with no constant to keep small scores' weights positive. Their gap is largest early (about +200% at
  update 500, +33% for query at update 1,500) and narrows without closing. SoS stays nonnegative.
- The zero constant pays off only on softplus inputs, where z ≥ 0 keeps T_8(z) − 1 nonnegative.
- Limits: one seed; the three ran on different GPU types.

The comparisons and their limits: Query is matched to the existing softplus Query control
`ra3b54e2489f8`, whose constant starts at zero but is learnable. Static's parent
`r9c69dd7cfc8d` has raw-score inputs, so that comparison changes both the transform
and constant policy. SoS is compared with the learned-constant softplus run
`r41a4c74c5f58` and prefactor run `r841f35a24241`; the prefactor has total degree
nine, unlike the new degree-eight SoS. One seed does not establish significance
or isolate optimization from the changed function class and initialization.
