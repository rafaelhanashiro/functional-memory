Every run on Kennel since the yellow box's runs, Claude's and Codex's (Codex's are marked), without the stability reruns.

#### In progress

- **Learning rate ×4**, $d_K = 4$
  - Exp
  - $z\,h_t(z)$ with a degree-16 core
- **Pure tanh square factors** $F_j = f_j \tanh(\hat x\, w_j + \beta_j)$ **for $z\,h_t(z)$ with a degree-16 core**, $d_K = 4$
  - Learning rate ×2
- **Query SoS with separate negative-score branches**, $m = 8$, $d_K = 64$ · Codex. All use $h(s)$ for $s \ge 0$; for $s < 0$, with $u = -s$:
  - $e^s h(s)$ (queued)
  - $e^s h(0)$
  - $h(0)/D(u)$, positive quartic coefficients (queued)
  - $h(0)/D(u)$, quartic SoS denominator
  - $h(0)/D(u)$, unrestricted quartic coefficients

#### Finished

- **Degree 16 on the shifted kernel**, $d_K = 4$
  - Query
  - Fixed Taylor (restarted after an over-strict validation check stopped it at update 5,500)
- **Learning rate ×½ and ×2**, $d_K = 4$
  - Exp ×½
  - Exp ×2
  - $z\,h_t(z)$ with a degree-16 core ×½
  - $z\,h_t(z)$ with a degree-16 core ×2
- **Pure tanh square factors** $F_j = f_j \tanh(\hat x\, w_j + \beta_j)$ **for $z\,h_t(z)$ with a degree-16 core**, $d_K = 4$
  - Learning rate as the partner's
- **Pure tanh query coefficients** $c_r = \tanh(\hat x\, w_r + \beta_r)/r!$, $d_K = 4$
  - $P_t(s)$, $m = 8$
  - $P_t(s)$, $m = 16$
  - $\mathrm{softplus}(s)\,P_t(s)$, $m = 8$
  - $\mathrm{softplus}(s)\,P_t(s)$, $m = 16$
  - $e^s P_t(s)$, $m = 8$
  - $e^s P_t(s)$, $m = 16$
- **Query SoS with the softplus prefactor** $z\,h_t(z)$, $d_K = 64$ · Codex
  - Core 4
  - Core 2
- **Even powers 2–8 with query sigmoid gates**, $d_K = 4$ · Codex
  - QK score, gates from 1/2
  - Cosine score, gates from 1/2
  - QK score, gates from Taylor
- **The same three plus a gated exponential**, $d_K = 4$
  - $m = 8$: all three
  - $m = 16$, even powers to 16: all three · Codex
- **Cosine score, gates from 1/2, gate learning rate by order**, $d_K = 4$
  - $2/r$
  - $r/2$
- **Parts of $z\,h_t(z)$**, $d_K = 4$
  - $h_t(s)$
  - $h_t(z)$
  - $z\,h_t(z)$ with a degree-16 core
- **Static $z\,h(z)$**, $d_K = 4$ · Codex
- **Cosine score with a sigmoid gate on every power**, $d_K = 4$ · Codex
  - $m = 8$
  - $m = 16$

#### Cancelled before training (at your request)

- **Exponential prefactors**, $d_K = 4$ · Codex
  - $\mathrm{softplus}(s)\,e^s$
  - $\mathrm{sigmoid}(s)\,e^s$
- **Endpoint gates, NoPE**, $d_K = 4$ · Codex
  - Exp with a query–key distance gate
  - Query, $m = 4$, with per-order query–key gates

#### The yellow box's runs (all finished)

- **Fixed kernels**
  - Swish
- **Scaling $d_K = 4, 8, 16$**, each at all three widths
  - Query
  - Static
  - Taylor
  - SoS + prefactor
  - Exp
  - Softmax
- **Query**
  - Softplus + prefactor
- **Static**
  - Softplus + prefactor
  - Softplus
- **Prefactor**
  - softplus · SoS(score)
  - exp · SoS(score)
  - exp · SoS(softplus)
  - sigmoid · SoS(score)
  - sigmoid · SoS(softplus)
  - swish · SoS(score)
  - swish · SoS(softplus)
