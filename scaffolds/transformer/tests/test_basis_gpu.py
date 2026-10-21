"""GPU qualification of scalar RBF kernels against analytic float64 derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.basis import fit_exp, rbf
from scaffolds.transformer.configs import Config
from scaffolds.transformer.validation.qualify_basis import check


@pytest.mark.gpu
@pytest.mark.parametrize("count", (3, 5, 9, 20))
@pytest.mark.parametrize("normalized", (False, True))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_rbf_gpu_analytic_output_gradients_and_causality(count, normalized, dtype):
    if jax.default_backend() != "gpu":
        pytest.skip("requires GPU")
    config = Config()
    config.model.qk_head_dim = config.model.value_head_dim = 64
    config.model.qk_norm = True
    config.model.attention_normalization = normalized
    config.model.attention_basis = fit_exp(
        [rbf(float(c), 1.0) for c in np.linspace(-6, 6, count)],
        quadrature_order=128, check_order=256,
    ).to_config()
    check(config, dtype)
