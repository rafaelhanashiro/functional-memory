### Polynomial gradient dynamics

How do attention coefficients and the backbone adapt to each other during training? These Transformer++ diagnostics use degree eight and $d_K=64$. [Runs and measurements](https://github.com/rafaelhanashiro/functional-memory/tree/main/experiments/transformer_pp/staging/polynomial_gradient_dynamics)

#### In progress

- **Exponential prefactor with a query-conditioned sum of squares** (`r11215998f5f6`). Recording coefficient gradients and optimizer updates for $e^s h_t(\mathrm{softplus}(s))$ at twice the base learning rate.

#### Finished

- **Query-conditioned tanh coefficients** (`r1422e8000207`). Recorded coefficient gradients and optimizer updates. Best validation PPL **14.313**.
- **Learned static coefficients** (`r6f55fe172f41`). Recorded the same measurements for the static model. Best validation PPL **14.352**.
- **Taylor reset** (`ra4d05272d53c`). On the original learned-static checkpoint, resetting only its coefficients to Taylor raised PPL from **14.294 to 15.532**. Loss rose on all 4,864 packed validation sequences. The separately trained Taylor checkpoint scored **14.218** on the same text.
