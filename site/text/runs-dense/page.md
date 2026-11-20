> [!info] Dense model
> $$c_r(x_t) = \frac{1}{r!}\left[b_r + \frac{1}{\sqrt{K}} \sum_{j=1}^{K} V_{rj} \tanh(w_j^\top x_t)\right]$$
>
> We also test a sigmoid variant:
>
> $$c_r(x_t) = \frac{1}{r!\sqrt{K}} \sum_{j=1}^{K} V_{rj}\, \sigma(w_j^\top x_t)$$
