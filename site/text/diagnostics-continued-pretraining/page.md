> [!info] Models
> - Query temperature: QK RMSNorm's per-head query scale $\exp(\gamma_h)$ becomes $\exp(\gamma_h + a_h \cos(w_h, x_t))$, a bounded temperature for each token and head read from the attention input $x_t$; $W$ decay is weight decay on $w_h$.
