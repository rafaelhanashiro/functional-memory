### Polynomial gradient dynamics

How do attention coefficients and the backbone adapt to each other during training? These Transformer++ diagnostics use degree eight and $d_K=64$. [Runs and measurements](https://github.com/rafaelhanashiro/functional-memory/tree/main/experiments/transformer_pp/staging/polynomial_gradient_dynamics)

#### Finished

- **Freeze or continue learning the static coefficients.** From the same step-500 checkpoint, continuing to train all parameters (`r3c1d740f28d1`) reached **19.83580** validation PPL at step 2,000. Freezing the learned coefficients and training everything else (`r8111c7eda7d3`) reached **19.84958**. Both arms reached their best validation PPL on that final update. [Train all parameters](https://wandb.ai/rafah-massachusetts-institute-of-technology/functional-memory/runs/a7ececb182634f5bb3bdbca73c8d50ad) · [Freeze coefficients](https://wandb.ai/rafah-massachusetts-institute-of-technology/functional-memory/runs/443da0437f424047aae769a5449ecae4)
- **Exponential prefactor with a query-conditioned sum of squares** (`r11215998f5f6`). The gradient diagnostic reached best validation PPL **14.046** at step 7,500. The original reached **14.001** at the same step. [Diagnostic on W&B](https://wandb.ai/rafah-massachusetts-institute-of-technology/functional-memory/runs/b1ecbc8cc042446fbcb540a4357b38a1)
- **Query-conditioned tanh coefficients** (`r1422e8000207`). Recorded coefficient gradients and optimizer updates. Best validation PPL **14.313**.
- **Learned static coefficients** (`r6f55fe172f41`). Recorded the same measurements for the static model. Best validation PPL **14.352**.
- **Taylor reset** (`ra4d05272d53c`). On the original learned-static checkpoint, resetting only its coefficients to Taylor raised PPL from **14.294 to 15.532**. Loss rose on all 4,864 packed validation sequences. The separately trained Taylor checkpoint scored **14.218** on the same text.
