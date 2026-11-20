This project asks whether we can learn a better kernel, potentially input-dependent, other than exp.

Concretely, an attention output is given by
$$
o_t=\sum_{s\le t}p(z_{ts})\,v_s
$$
where $z_{ts} \sim q_t^\top k_s$ is the attention score. In a standard transformer, we have $p=\text{softmax}$.

Our goal is to learn a function $p$ based on the data, which can potentially depend on the activation that goes into the attention layer. We will primarily focus on polynomials, but also test out different function classes.
