Every run on ORCD not yet on the site.

#### In progress

- **Learning rate at two and four times the schedule**, $d_K = 64$
  - $z\,h_t(z)$, core 8: $\times 2$, $\times 4$ (kernel qualified; waiting for your go-ahead)

#### Finished

- **Learning rate at two and four times the schedule**, $d_K = 64$
  - Query, $m = 8$: $\times 2$, $\times 4$
  - Exp: $\times 2$, $\times 4$
- **Pure tanh query coefficients** $c_r = \tanh(\hat x\, w_r + \beta_r)/r!$, $d_K = 64$
  - $P_t(s)$, $m = 8$
  - $P_t(s)$, $m = 16$
- **Tanh query** $c_r = b_r + a_r \tanh(w_r^\top x)$, their affine counterpart, $d_K = 64$
  - $p_t(s)$, $m = 16$
