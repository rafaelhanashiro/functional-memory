"""GPU qualification of scalar RBF kernels against analytic float64 derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.basis import ExpProjection, rbf
from scaffolds.transformer.configs import Config
from scaffolds.transformer.validation.qualify_basis import check


# Frozen N(0,1) fits, 128/256 quadrature nodes. GPU-only processes must
# restore coefficients as training does, without invoking the CPU fitter.
COEFFICIENTS = {
    3: (0.0038884559389006693, 1.568715063994447, 632.8648352636275),
    5: (5.339937464674399, -0.23400716082225384, 0.9312125373360598, 13.07024840694441, 370.9794096959457),
    9: (0.7842980450462962, -0.1523710791787486, 0.06293250466768532, 0.057376729248309444, 0.38759739950223415, 1.574309886972801, 7.516279945564808, 30.74141419377182, 176.47302400187155),
    20: (-74.74475072727894, 179.88680628933096, -260.7306403593473, 302.49568692478897, -314.9063755603312, 312.5870638613478, -306.80418087187644, 304.8888346187574, -311.22098178268914, 329.8137995839666, -363.83776424992607, 419.90930789273443, -502.4878860855093, 625.5809248785472, -788.6226831409843, 1011.026828235822, -1236.6295459656237, 1448.4336311832224, -1360.1420060895953, 989.4471959252),
}


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
    config.model.attention_basis = ExpProjection(
        tuple(rbf(float(c), 1.0) for c in np.linspace(-6, 6, count)),
        COEFFICIENTS[count], 0.0, 1.0, 128, 256, 1e-12, {},
    ).to_config()
    check(config, dtype)
