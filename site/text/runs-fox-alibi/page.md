> [!info] Models
> Suppressing layer and head indices, let $x_t$ be the pre-attention RMSNorm activation and $s_{t,i} = q_t^\top k_i/\sqrt{d_K}$ the score of key $i \le t$. The query-dependent polynomial is
> $$p_t(s) = \sum_{r=0}^{4} \frac{b_r + a_r \tanh(W_r^\top x_t)}{r!}\, s^r.$$
> Each gate decays a key's weight with its distance,
> $$\omega_{t,i} = p_t(s_{t,i})\, e^{-\gamma G_{t,i}}, \qquad \gamma = e^{A_{\log}} > 0,$$
> and the output is $o_t = \mathrm{RMSNorm}_{\mathrm{head}}\big(\sum_{i \le t} \omega_{t,i} v_i\big)$.
>
> | Gate | $G_{t,i}$ |
> |---|---|
> | FoX | $\sum_{j=i+1}^{t} \mathrm{softplus}(w_q^\top x_j + \beta)$ |
> | Query gate | $(t-i)\, \mathrm{softplus}(w_q^\top x_t + \beta)$ |
> | Query gate, sigmoid | $(t-i)\, \sigma(w_q^\top x_t + \beta)$ |
> | Query–key gate | $(t-i)\, \mathrm{softplus}(w_q^\top x_t + w_k^\top x_i + \beta)$ |
> | ALiBi | $t-i$ |
>
> - FoX multiplies GDN-style per-token gates $\alpha_j = e^{-\gamma\, \mathrm{softplus}(w_q^\top x_j + \beta)}$ along the sequence; the other gates apply one rate, from the query or the query–key pair, across the whole distance. With $A_{\log} = 0$, $\alpha_j = \sigma(-w_q^\top x_j - \beta)$: the original FoX sigmoid gate, up to the sign of $w_q$ and $\beta$.
> - ALiBi learns its rate $\gamma = e^{A}$ per layer and head, from ALiBi's slopes and without weight decay; it reads no token. The original ALiBi keeps the slopes fixed.
> - The query–key gate reads the key; the polynomial coefficients read the query only.
> - Per degree: each degree $r$ has its own gate, $\omega_{t,i} = \sum_r c_{t,r}\, s_{t,i}^r\, e^{-\gamma_r G^{(r)}_{t,i}}$.
