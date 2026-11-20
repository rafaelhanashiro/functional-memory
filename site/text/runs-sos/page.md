> [!note] SoS model
>
> Let $s$ be the attention score. The SoS model is given by
> $$h(s;x) = q_1^2(s;x)+q_2^2(s;x)$$
> where $\text{deg}(q_1)=m/2$ and $\text{deg}(q_2)=m/2-1$.

Let $z=\text{softplus}(s)$. We consider three models:

| Variant | Attention weight | Note |
|---|---|---|
| SoS | $h(s)$ | |
| SoS + softplus | $h(z)$ | |
| SoS + softplus prefactor | $z h(z)$ | Raises degree by 1 |

Intuition:
- Softplus: negative scores are first transformed into a small number *and then* fed into $h$. E.g., without it, $x^2$ would treat $-100$ and $100$ equally.
- Softplus prefactor: we additionally get the property that $\text{weight}\to0$ as $s\to-\infty$.
