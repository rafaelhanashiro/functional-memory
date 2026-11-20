This project asks whether polynomial attention can match or beat exp and softmax attention when the rest of the Transformer stays the same. Each head scores query $t$ against key $s$ with $z_{ts}=q_t^\top k_s/\sqrt{d_K}$ and weights the values by a polynomial of degree $m$ instead of $e^{z}$:

$$o_t=\operatorname{RMSNorm}\Big(\sum_{s\le t}p(z_{ts})\,v_s\Big),\qquad p(z)=\sum_{r=0}^{m}c_r z^r.$$

Exp is the same with $p(z)=e^z$; softmax instead divides by the causal row sum. The coefficients $c_r$ are fixed at Taylor's $1/r!$, learned per head (static), or computed from each query (query).
