> [!info] Models
> All models compute 
> $$
>     o_t=\operatorname{RMSNorm}\left(\sum_{s\le t}p(z_{ts})\,v_s\right)
> $$
and differ in $p$:
>
> - Exp: $p(z)=e^{z}$
> - Taylor: $p(z)=\sum_{r=0}^{m}z^r/r!$, exp's Taylor series to degree $m$
> - Hermite: $p(z)=e^{1/2}\sum_{r=0}^{m}\mathrm{He}_r(z)/r!$, exp's Hermite series to degree $m$
> - Static: $p(z)=\sum_{r=0}^{m}c_r z^r$, with $c_r$ learned per head
> - Query: $p(z)=\sum_{r=0}^{m}c_r(x_t)\,z^r$, with $c_r(x_t)=b_r+a_r\tanh(w_r^\top x_t)$ and $x_t$ the attention layer's input
> 
> Unless otherwise stated, all applicable models use weight decay on $w_r$.
