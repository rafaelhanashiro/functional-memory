> [!tip] Notes
> - Attending only to the last 2k tokens greatly improves performance.
> - Clamping RoPE at distance 2k keeps PPL stable from lengths 2k–4k, but then blows up past that.
