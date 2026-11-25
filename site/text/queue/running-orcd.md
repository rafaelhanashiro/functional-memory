*Updated 2026-11-25 at 16:26 EST.*

Every run on ORCD not yet on the site.

#### In progress

- **Learning rate at two and four times the schedule**, $d_K = 64$ (all four kernels qualified on ORCD's H100s). Query SoS core 8 on $z = \mathrm{softplus}(s)$ under each prefactor, and exp under the softplus prefactor:
  - $\sigma(s)\,h_t(z)$ (sigmoid prefactor): $\times 4$ (preempted a third time at update 6,057; queued to resume from update 6,000)

#### Finished

- **Learning rate at two and four times the schedule**, $d_K = 64$
  - Query, $m = 8$: $\times 2$, $\times 4$
  - Exp: $\times 2$, $\times 4$
  - $z\,h_t(z)$ (softplus prefactor), core 8: $\times 2$, $\times 4$
  - $\mathrm{softplus}(s)\,e^s$: $\times 2$, $\times 4$
  - $e^s\,h_t(z)$ (exp prefactor), core 8: $\times 2$, $\times 4$
  - $\sigma(s)\,h_t(z)$ (sigmoid prefactor), core 8: $\times 2$
- **Pure tanh query coefficients** $c_r = \tanh(\hat x\, w_r + \beta_r)/r!$, $d_K = 64$
  - $P_t(s)$, $m = 8$
  - $P_t(s)$, $m = 16$
- **Tanh query** $c_r = b_r + a_r \tanh(w_r^\top x)$, their affine counterpart, $d_K = 64$
  - $p_t(s)$, $m = 16$
