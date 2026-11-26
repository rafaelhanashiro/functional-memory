*Updated 2026-11-25 at 23:46 EST.*

Every run on ORCD not yet on the site.

#### In progress

Nothing is training on ORCD.

#### Finished

- **Learning rate at two and four times the schedule**, $d_K = 64$
  - Query, $m = 8$: $\times 2$, $\times 4$
  - Exp: $\times 2$, $\times 4$
  - $z\,h_t(z)$ (softplus prefactor), core 8: $\times 2$, $\times 4$
  - $\mathrm{softplus}(s)\,e^s$: $\times 2$, $\times 4$
  - $e^s\,h_t(z)$ (exp prefactor), core 8: $\times 2$, $\times 4$
  - $\sigma(s)\,h_t(z)$ (sigmoid prefactor), core 8: $\times 2$, $\times 4$
- **Pure tanh query coefficients** $c_r = \tanh(\hat x\, w_r + \beta_r)/r!$, $d_K = 64$
  - $P_t(s)$, $m = 8$
  - $P_t(s)$, $m = 16$
- **Tanh query** $c_r = b_r + a_r \tanh(w_r^\top x)$, their affine counterpart, $d_K = 64$
  - $p_t(s)$, $m = 16$
