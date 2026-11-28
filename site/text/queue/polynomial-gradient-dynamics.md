### Polynomial gradient dynamics

How do attention coefficients and the backbone adapt to each other during training? These Transformer++ diagnostics use degree eight and $d_K=64$. [Runs and measurements](https://github.com/rafaelhanashiro/functional-memory/tree/main/experiments/transformer_pp/staging/polynomial_gradient_dynamics)

#### In progress

- **Freeze or continue learning the static coefficients.** Both arms start at step 500 with validation PPL **49.520**. Kennel job 16393 (`r3c1d740f28d1`) trains the whole model. Job 16397 (`r8111c7eda7d3`) freezes the learned polynomial coefficients and trains everything else. Both use the same batches and optimizer schedule through step 2,000. [Shared checkpoint on W&B](https://wandb.ai/rafah-massachusetts-institute-of-technology/functional-memory/runs/eb6e2da7cfb5469db8c7a5ebbcec548c)

#### Finished

- **Exponential prefactor with a query-conditioned sum of squares** (`r11215998f5f6`). The gradient diagnostic reached best validation PPL **14.046** at step 7,500. The original reached **14.001** at the same step. [Diagnostic on W&B](https://wandb.ai/rafah-massachusetts-institute-of-technology/functional-memory/runs/b1ecbc8cc042446fbcb540a4357b38a1)
- **Query-conditioned tanh coefficients** (`r1422e8000207`). Recorded coefficient gradients and optimizer updates. Best validation PPL **14.313**.
- **Learned static coefficients** (`r6f55fe172f41`). Recorded the same measurements for the static model. Best validation PPL **14.352**.
- **Taylor reset** (`ra4d05272d53c`). On the original learned-static checkpoint, resetting only its coefficients to Taylor raised PPL from **14.294 to 15.532**. Loss rose on all 4,864 packed validation sequences. The separately trained Taylor checkpoint scored **14.218** on the same text.
