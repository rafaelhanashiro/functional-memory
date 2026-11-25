# LR
At $d_k=4$,
- Higher LR was better (even more so for SoS). This may be due to the model needing to better exploit score scaling when the geometry is needed; higher LR helps reach larger scaling.

# Gradient spikes
At $d_k=4$,
- Query had more spikes than SoS, and the latter performed much better

# Prefactors
- For the most part, it seems the best recipe is prefactor(score)*poly(score)
- exp*softplus works well. **Why is that?**
