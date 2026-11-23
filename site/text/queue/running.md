#### In progress

- **Degree 16 on the shifted kernel**, $d_K = 4$
  - Query
  - Fixed Taylor
- **Pure tanh query coefficients** $c_r = \tanh(\hat x\, w_r + \beta_r)/r!$, $d_K = 4$
  - $P_t(s)$, $m = 8$ (queued)
  - $P_t(s)$, $m = 16$
  - $\mathrm{softplus}(s)\,P_t(s)$, $m = 8$ (queued)
  - $\mathrm{softplus}(s)\,P_t(s)$, $m = 16$
  - $e^s P_t(s)$, $m = 8$ (queued)
  - $e^s P_t(s)$, $m = 16$ (queued)
- **Query SoS with the softplus prefactor** $z\,h_t(z)$, $d_K = 64$
  - Core 4
  - Core 2 (queued)

#### Finished

Launched after the yellow box's runs; stability reruns left out.

- **Even powers 2–8 with query sigmoid gates**, $d_K = 4$
  - QK score, gates from 1/2
  - Cosine score, gates from 1/2
  - QK score, gates from Taylor
- **The same three plus a gated exponential**, $d_K = 4$
  - $m = 8$: all three
  - $m = 16$, even powers to 16: all three
- **Cosine score, gates from 1/2, gate learning rate by order**, $d_K = 4$
  - $2/r$
  - $r/2$
- **Parts of $z\,h_t(z)$**, $d_K = 4$
  - $h_t(s)$
  - $h_t(z)$
  - $z\,h_t(z)$ with a degree-16 core
- **Static $z\,h(z)$**, $d_K = 4$
- **Cosine score with a sigmoid gate on every power**, $d_K = 4$
  - $m = 8$
  - $m = 16$

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
