import io
import math

import pytest
import torch

from scaffolds.state_rank.sparse_polynomial import PolynomialFeatures, SparsePolynomialHead


@pytest.mark.parametrize("dimension,degree", [(1, 1), (1, 3), (3, 2), (3, 3), (7, 3)])
def test_feature_inner_product_is_exact_truncated_exponential(dimension, degree):
    generator = torch.Generator().manual_seed(14)
    q = torch.randn(5, dimension, generator=generator, dtype=torch.float64)
    k = torch.randn(8, dimension, generator=generator, dtype=torch.float64)
    bank = PolynomialFeatures(dimension, degree)
    score = q @ k.T / math.sqrt(dimension)
    expected = sum(score.pow(order) / math.factorial(order) for order in range(degree + 1))
    torch.testing.assert_close(bank(q) @ bank(k).T, expected, rtol=2e-14, atol=2e-14)
    assert bank(q).dtype == torch.float64
    assert bank.width == math.comb(dimension + degree, degree)
    for order in range(degree + 1):
        expected_count = math.comb(dimension + order - 1, order)
        assert (bank.degrees == order).sum().item() == expected_count
    torch.testing.assert_close(bank(q)[:, 0], torch.ones(5, dtype=torch.float64), rtol=0, atol=0)


def test_cubic_feature_multiplicities_and_real_head_width():
    bank = PolynomialFeatures(3, 3)
    x = torch.tensor([[2.0, 3.0, 5.0]], dtype=torch.float64)
    features = bank(x)[0]
    for coordinates, expected in [
        ([0, 0, 0], 2 ** 3 / math.sqrt(6)),
        ([0, 0, 1], 2 ** 2 * 3 / math.sqrt(2)),
        ([0, 1, 2], 2 * 3 * 5),
    ]:
        index = (bank.indices == torch.tensor(coordinates)).all(-1).nonzero().item()
        torch.testing.assert_close(features[index], torch.tensor(expected * 3 ** (-3 / 4),
                                                                 dtype=torch.float64))
    real_bank = PolynomialFeatures(64, 3)
    assert real_bank.width == 47_905
    assert (real_bank.degrees == 3).sum().item() == 45_760


def sample_inputs():
    generator = torch.Generator().manual_seed(19)
    # Small inputs keep all selected polynomial kernels positive, so the test
    # compares the exact normalization instead of the invalid-row policy.
    q = torch.randn(4, 3, generator=generator, dtype=torch.float64) * 0.2
    k = torch.randn(7, 3, generator=generator, dtype=torch.float64) * 0.2
    v = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    return q, k, v, torch.tensor([0, 2, 4, 6])


@pytest.mark.parametrize("mode,budget", [("dense", 20)] + [
    (mode, budget) for mode in ("fixed", "random", "dynamic") for budget in (1, 7, 20)
])
def test_exact_budget_constant_and_causal_outputs_match_explicit_mask(mode, budget):
    bank = PolynomialFeatures(3, 3)
    head = SparsePolynomialHead(bank, mode, budget, seed=32).double().eval()
    head.set_calibration(torch.linspace(0.3, 2.0, bank.width, dtype=torch.float64),
                         torch.linspace(1.8, 0.4, bank.width, dtype=torch.float64))
    q, k, v, positions = sample_inputs()
    result = head(q, k, v, positions)
    assert result["support"].shape == (len(q), budget)
    assert (result["support"][:, 0] == 0).all()
    assert all(len(row.unique()) == budget for row in result["support"])
    if mode in {"fixed", "random"}:
        assert torch.equal(result["support"], result["support"][:1].expand(len(q), -1))
    feature_mask = torch.zeros(len(q), bank.width, dtype=q.dtype)
    feature_mask.scatter_(1, result["support"], 1.0)
    expected_kernel = ((bank(q) * feature_mask) * head.weights) @ bank(k).T
    allowed = torch.arange(len(k))[None, :] <= positions[:, None]
    expected_kernel = expected_kernel.masked_fill(~allowed, 0.0)
    expected_attention = expected_kernel / expected_kernel.sum(-1, keepdim=True)
    assert result["valid"].all()
    torch.testing.assert_close(result["kernel"], expected_kernel, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(result["attention"], expected_attention, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(result["output"], expected_attention @ v, rtol=1e-14, atol=1e-14)
    assert (result["attention"][~allowed] == 0).all()
    torch.testing.assert_close(result["output"][0], v[0], rtol=1e-14, atol=1e-14)

    # A dense key-written recurrent state and sparse query reads must reproduce
    # this pairwise reference, including arbitrary selected causal positions.
    key_features, query_features = bank(k), bank(q)
    numerator_state = torch.zeros(bank.width, v.shape[-1], dtype=v.dtype)
    denominator_state = torch.zeros(bank.width, dtype=v.dtype)
    for position in range(len(k)):
        numerator_state += key_features[position, :, None] * v[position]
        denominator_state += key_features[position]
        for query_index in (positions == position).nonzero().flatten():
            support = result["support"][query_index]
            read = query_features[query_index, support] * head.weights[support]
            recurrent_output = ((read @ numerator_state[support])
                                / (read @ denominator_state[support]))
            torch.testing.assert_close(result["output"][query_index], recurrent_output,
                                       rtol=1e-14, atol=1e-14)


def test_random_support_is_seeded_and_fixed_support_uses_calibration():
    bank = PolynomialFeatures(3, 3)
    a = SparsePolynomialHead(bank, "random", 5, seed=5)
    b = SparsePolynomialHead(bank, "random", 5, seed=5)
    c = SparsePolynomialHead(bank, "random", 5, seed=6)
    assert torch.equal(a.support, b.support)
    assert not torch.equal(a.support, c.support)
    fixed = SparsePolynomialHead(bank, "fixed", 3, initial_score_temperature=1)
    query_rms = torch.ones(bank.width)
    query_rms[8], query_rms[13] = 20, 10
    fixed.set_calibration(query_rms, torch.ones(bank.width))
    assert fixed.support.tolist() == [0, 8, 13]


def test_dynamic_support_varies_by_query_and_reads_no_key_or_value():
    bank = PolynomialFeatures(2, 1)
    head = SparsePolynomialHead(bank, "dynamic", 2, initial_score_temperature=1)
    qf = bank(torch.tensor([[10.0, 0.1], [0.1, 10.0]]))
    assert head.query_support(qf).tolist() == [[0, 1], [0, 2]]
    # Calibration is frozen across examples; current keys, values and state
    # are not inputs to this selection API.
    head.key_rms[1] = 1000
    assert head.query_support(qf).tolist() == [[0, 1], [0, 1]]


def test_dynamic_training_forward_matches_hard_evaluation_and_learns_router():
    head = SparsePolynomialHead(PolynomialFeatures(3, 3), "dynamic", 7).double()
    q, k, v, positions = sample_inputs()
    head.train()
    trained = head(q, k, v, positions, training_safe=True)
    head.eval()
    evaluated = head(q, k, v, positions, training_safe=False)
    for key in ("kernel", "attention", "output", "denominator"):
        torch.testing.assert_close(trained[key], evaluated[key], rtol=1e-14, atol=1e-14)
    assert torch.equal(trained["support"], evaluated["support"])
    loss = trained["output"].square().mean() + trained["denominator_penalty"]
    loss.backward()
    assert torch.isfinite(head.log_relevance.grad).all()
    assert head.log_relevance.grad[1:].abs().sum() > 0
    assert torch.isfinite(head.log_weights.grad).all()
    assert head.log_weights.grad.abs().sum() > 0


def test_negative_normalizer_is_visible_in_evaluation_and_safe_for_training():
    head = SparsePolynomialHead(PolynomialFeatures(1, 1), "dense", 2,
                                initial_score_temperature=1).double()
    q = torch.tensor([[-2.0]], dtype=torch.float64)
    k = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
    v = torch.tensor([[2.0], [4.0]], dtype=torch.float64)
    positions = torch.tensor([1])
    result = head(q, k, v, positions)
    torch.testing.assert_close(result["kernel"], torch.tensor([[-1.0, -1.0]], dtype=torch.float64))
    assert result["denominator"].item() == -2
    assert not result["valid"].any()
    assert torch.isnan(result["attention"]).all()
    assert torch.isnan(result["output"]).all()
    safe = head(q, k, v, positions, training_safe=True)
    assert not safe["valid"].any()
    assert torch.isfinite(safe["output"]).all()
    assert safe["denominator"].item() == -2  # The actual denominator stays observable.
    assert safe["denominator_penalty"].item() > 1
    (safe["output"].square().mean() + safe["denominator_penalty"]).backward()
    assert torch.isfinite(head.log_weights.grad).all()


def test_signed_attention_is_not_silently_clipped_or_renormalized():
    head = SparsePolynomialHead(PolynomialFeatures(1, 1), "dense", 2,
                                initial_score_temperature=1).double()
    result = head(torch.tensor([[1.0]], dtype=torch.float64),
                  torch.tensor([[-2.0], [3.0]], dtype=torch.float64),
                  torch.tensor([[1.0], [5.0]], dtype=torch.float64), torch.tensor([1]))
    assert result["valid"].all()
    expected = torch.tensor([[-1.0 / 3, 4.0 / 3]], dtype=torch.float64)
    torch.testing.assert_close(result["attention"], expected)
    torch.testing.assert_close(result["output"], torch.tensor([[19.0 / 3]], dtype=torch.float64))


def test_small_positive_normalizer_relative_to_signed_mass_is_invalid():
    head = SparsePolynomialHead(PolynomialFeatures(1, 1), "dense", 2,
                                initial_score_temperature=1).double()
    result = head(torch.tensor([[1.0]], dtype=torch.float64),
                  torch.tensor([[-2.0], [0.00001]], dtype=torch.float64),
                  torch.tensor([[1.0], [5.0]], dtype=torch.float64), torch.tensor([1]))
    assert result["denominator"].item() > 0
    assert not result["valid"].any()
    assert torch.isnan(result["output"]).all()
    assert result["denominator_penalty"].item() > 0


@pytest.mark.parametrize("mode", ["dense", "fixed", "random", "dynamic"])
def test_fitting_steps_keep_loss_parameters_and_gradients_finite(mode):
    bank = PolynomialFeatures(3, 3)
    head = SparsePolynomialHead(bank, mode, bank.width if mode == "dense" else 7).double()
    q, k, v, positions = sample_inputs()
    causal = torch.arange(len(k))[None, :] <= positions[:, None]
    teacher = (q @ k.T / math.sqrt(3)).masked_fill(~causal, -torch.inf).softmax(-1) @ v
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        result = head(q, k, v, positions, training_safe=True)
        loss = (result["output"] - teacher).square().mean() + result["denominator_penalty"]
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in head.parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        optimizer.step()
        assert all(torch.isfinite(parameter).all() for parameter in head.parameters())
        losses.append(loss.item())
    assert losses[-1] < losses[0]


@pytest.mark.parametrize("mode", ["fixed", "random", "dynamic"])
def test_checkpoint_roundtrip_preserves_support_calibration_weights_and_predictions(mode):
    original = SparsePolynomialHead(PolynomialFeatures(3, 3), mode, 7, seed=8).double().eval()
    with torch.no_grad():
        original.log_weights.add_(torch.linspace(-0.1, 0.1, original.bank.width))
        original.log_relevance.copy_(torch.linspace(-0.3, 0.4, original.bank.width))
    original.set_calibration(torch.linspace(1, 2, original.bank.width),
                             torch.linspace(2, 1, original.bank.width))
    buffer = io.BytesIO()
    torch.save(original.state_dict(), buffer)
    buffer.seek(0)
    restored = SparsePolynomialHead(PolynomialFeatures(3, 3), mode, 7, seed=99).double().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    for name, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
    expected, actual = original(*sample_inputs()), restored(*sample_inputs())
    for key in ("support", "kernel", "output", "valid"):
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_feature_bank_and_new_head_follow_requested_device(device):
    bank = PolynomialFeatures(3, 3).to(device)
    head = SparsePolynomialHead(bank, "dynamic", 4).double()
    assert all(buffer.device.type == device for buffer in head.buffers())
    assert all(parameter.device.type == device for parameter in head.parameters())
    assert bank.indices.dtype == torch.long
    q, k, v, positions = (value.to(device) for value in sample_inputs())
    result = head(q, k, v, positions, training_safe=True)
    assert result["output"].device.type == device
    assert result["output"].dtype == torch.float64


@pytest.mark.parametrize("kwargs", [{"d": 0}, {"d": 2, "degree": 0}, {"d": 2, "degree": 4}])
def test_invalid_feature_bank_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PolynomialFeatures(**kwargs)


@pytest.mark.parametrize("mode,budget", [("other", 3), ("dense", 2), ("fixed", 0), ("random", 21)])
def test_invalid_mode_and_budget_are_rejected(mode, budget):
    with pytest.raises(ValueError):
        SparsePolynomialHead(PolynomialFeatures(3, 3), mode, budget)
