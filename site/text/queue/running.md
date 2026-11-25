*Updated 2026-11-25 at 16:25 EST.*

Every run on Kennel since the yellow box's runs, Claude's and Codex's (Codex's are marked), without the stability reruns.

#### In progress

- **Twice the Chinchilla budget**, $d_K = 4$ (15,140 updates)
  - Exp
  - Exp, learning rate ×2
  - $z\,h_t(z)$ with a degree-16 core, learning rate ×2
  - $z\,h_t(z)$ with a degree-16 core, learning rate ×4
- **$\mathrm{softplus}(s)\,h_t(s)$**, query SoS core 8 on the raw score, $d_K = 4$
  - Learning rate as the recipe's
  - Learning rate ×2 (queued)
  - Learning rate ×4 (queued)
- **Exponential prefactors**, $d_K = 4$ (the two cancelled recipes, now training)
  - $\mathrm{softplus}(s)\,e^s$
  - $\mathrm{sigmoid}(s)\,e^s$ (queued)
- **$e^s\,h_t(z)$ with a degree-16 core** (affine query SoS on $z = \mathrm{softplus}(s)$), $d_K = 4$ · Codex
  - Learning rate as the recipe's
  - Learning rate ×2
  - Learning rate ×4
- **Fixed Taylor $e^s\,T_8(z)$**, $z = \mathrm{softplus}(s)$, learning rate ×2, $d_K = 64$ · Codex (queued)

#### Finished

- **Learning rate**, $d_K = 4$
  - Exp ×½
  - Exp ×2
  - Exp ×4
  - $z\,h_t(z)$ with a degree-16 core ×½
  - $z\,h_t(z)$ with a degree-16 core ×2
  - $z\,h_t(z)$ with a degree-16 core ×4
  - $z\,h_t(z)$ with a degree-16 core ×8
  - $z\,h_t(z)$ with a degree-8 core ×2
  - $z\,h_t(z)$ with a degree-8 core ×4
  - $z\,h_t(z)$ with a degree-8 core ×8
- **Learning rate**, $d_K = 64$
  - Exp ×½
  - $z\,h_t(z)$ with a degree-8 core ×½
  - $f(s)\,h_t(s)$, query SoS core 8 on the raw score (ORCD runs the softplus-input counterparts):
    - exp prefactor ×2 and ×4
    - softplus prefactor ×2 and ×4
    - sigmoid prefactor ×2 and ×4
- **Query polynomial under a prefactor of the raw score**, $m = 8$, $d_K = 64$
  - $e^s\,p_t(s)$
  - $\mathrm{softplus}(s)\,p_t(s)$
- **Separate negative-score branches**, $m = 8$, $d_K = 64$. Each keeps its kernel for $s \ge 0$; for $s < 0$, with $u = -s$:
  - Query polynomial $p_t$: $e^s p_t(s)$, $e^s p_t(0)$, $p_t(0)/D(u)$ with positive quartic coefficients
  - Query SoS $h$ · Codex: $e^s h(s)$, $e^s h(0)$, $h(0)/D(u)$ with positive quartic coefficients, with a quartic SoS denominator, and with unrestricted quartic coefficients (numerical failure at update 2,136)
- **Unscaled cosine score** (queries and keys unit-normalized, no head scale), $d_K = 4$
  - Exp
  - Affine tanh query, $m = 8$ · Codex
  - $z\,h_t(z)$, core 8 · Codex
- **Pure tanh square factors** $F_j = f_j \tanh(\hat x\, w_j + \beta_j)$ **for $z\,h_t(z)$ with a degree-16 core**, $d_K = 4$
  - Learning rate as the partner's
  - Learning rate ×2
- **Degree 16 on the shifted kernel**, $d_K = 4$
  - Query
  - Fixed Taylor (restarted after an over-strict validation check stopped it at update 5,500)
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
