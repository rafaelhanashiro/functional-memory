"""Symmetric polynomial features and fixed-budget query-side selection.

Feature inner products equal the degree-truncated Taylor exponential at
q.dot(k)/sqrt(d). Query-dependent hard selection preserves separability in
queries and keys. It does not sparsify key writes.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations_with_replacement
import math

import torch
from torch import nn


class PolynomialFeatures(nn.Module):
    def __init__(self, d: int, degree: int = 3):
        super().__init__()
        if d < 1 or degree not in (1, 2, 3):
            raise ValueError("Require positive width and degree 1, 2 or 3")
        self.d, self.degree = d, degree
        indices, degrees, factors = [], [], []
        for order in range(degree + 1):
            for coordinates in combinations_with_replacement(range(d), order):
                multiplicity_factorial = math.prod(math.factorial(n) for n in Counter(coordinates).values())
                indices.append([*coordinates, *([d] * (degree - order))])
                degrees.append(order)
                factors.append(d ** (-order / 4) / math.sqrt(multiplicity_factorial))
        self.width = len(indices)
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.register_buffer("degrees", torch.tensor(degrees, dtype=torch.long))
        self.register_buffer("factors", torch.tensor(factors, dtype=torch.float64))

    def forward(self, x):
        if x.shape[-1] != self.d:
            raise ValueError(f"Expected dimension {self.d}, got {x.shape[-1]}")
        augmented = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        values = self.factors.to(dtype=x.dtype).expand(*x.shape[:-1], self.width)
        for axis in range(self.degree):
            values = values * augmented[..., self.indices[:, axis]]
        return values


class SparsePolynomialHead(nn.Module):
    def __init__(self, bank: PolynomialFeatures, mode: str, budget: int,
                 seed: int = 0, denominator_margin: float = 0.001,
                 initial_score_temperature: float = 0.1):
        super().__init__()
        if mode not in {"dense", "fixed", "random", "dynamic"}:
            raise ValueError(f"Unknown selection mode {mode}")
        if not 1 <= budget <= bank.width or (mode == "dense" and budget != bank.width):
            raise ValueError("Invalid hard feature budget")
        if not 0 < denominator_margin < 1 or initial_score_temperature <= 0:
            raise ValueError("Invalid normalization margin or temperature")
        self.bank, self.mode, self.budget = bank, mode, budget
        self.denominator_margin = denominator_margin
        self.log_weights = nn.Parameter(bank.degrees.float() * math.log(initial_score_temperature))
        self.log_relevance = nn.Parameter(torch.zeros(bank.width, device=bank.indices.device),
                                         requires_grad=mode == "dynamic")
        self.register_buffer("query_rms", torch.ones(bank.width, device=bank.indices.device))
        self.register_buffer("key_rms", torch.ones(bank.width, device=bank.indices.device))
        generator = torch.Generator().manual_seed(seed)
        support = torch.cat((torch.zeros(1, dtype=torch.long),
                             torch.randperm(bank.width - 1, generator=generator)[:budget - 1] + 1))
        self.register_buffer("support", support.to(bank.indices.device))

    @property
    def weights(self):
        # The bounded positive parameterization is part of the scientific recipe.
        return self.log_weights.clamp(-12, 12).exp()

    @torch.no_grad()
    def load_dense_weights(self, state):
        self.log_weights.copy_(state["log_weights"].to(self.log_weights))
        self.log_relevance.zero_()

    @torch.no_grad()
    def set_calibration(self, query_rms, key_rms):
        if query_rms.shape != self.query_rms.shape or key_rms.shape != self.key_rms.shape:
            raise ValueError("Feature calibration shape mismatch")
        self.query_rms.copy_(query_rms)
        self.key_rms.copy_(key_rms)
        if self.mode == "fixed":
            scores = self.query_rms * self.key_rms * self.weights
            indices = torch.topk(scores[1:], self.budget - 1, sorted=True).indices + 1
            self.support.copy_(torch.cat((indices.new_zeros(1), indices)))

    def query_support(self, query_features):
        if self.mode == "dense":
            return torch.arange(self.bank.width, device=query_features.device).expand(len(query_features), -1)
        if self.mode != "dynamic":
            return self.support.expand(len(query_features), -1)
        # Expected magnitude against a unit-RMS key coordinate; teacher/state independent.
        scores = (query_features.abs() * self.key_rms * self.weights.detach()
                  * self.log_relevance.clamp(-12, 12).exp())
        indices = torch.topk(scores[:, 1:], self.budget - 1, dim=-1, sorted=True).indices + 1
        return torch.cat((indices.new_zeros((len(indices), 1)), indices), dim=-1)

    def forward_features(self, qf, kf, v, query_positions, training_safe=False):
        if qf.ndim != 2 or kf.ndim != 2 or v.ndim != 2 or len(kf) != len(v):
            raise ValueError("Expected query/key feature matrices and matching values")
        if qf.shape[-1] != self.bank.width or kf.shape[-1] != self.bank.width:
            raise ValueError("Polynomial feature widths differ")
        if query_positions.shape != (len(qf),):
            raise ValueError("One causal position is required per query")
        support = self.query_support(qf)
        if self.mode == "dense":
            kernel = (qf * self.weights) @ kf.T
        elif self.mode == "dynamic" and self.training and training_safe:
            # Forward uses the exact hard budget. Backward uses sigmoid gates so
            # every relevance score can learn, including currently unselected rows.
            magnitude = (qf.detach().abs() * self.key_rms * self.weights.detach()).clamp_min(1e-30)
            log_scores = magnitude.log() + self.log_relevance.clamp(-12, 12)
            hard = torch.zeros_like(qf).scatter(1, support, 1.0)
            if self.budget == 1:
                gate = hard
            else:
                boundary = log_scores.gather(1, support[:, -1:]).detach()
                soft = torch.sigmoid(log_scores - boundary)
                soft = torch.cat((torch.ones_like(soft[:, :1]), soft[:, 1:]), dim=-1)
                gate = hard + (soft - soft.detach())
            kernel = (qf * self.weights * gate) @ kf.T
        elif self.mode in {"fixed", "random"}:
            kernel = (qf[:, self.support] * self.weights[self.support]) @ kf[:, self.support].T
        else:
            query = qf.gather(1, support) * self.weights[support]
            selected_keys = kf[:, support].permute(1, 0, 2)
            kernel = torch.einsum("qm,qkm->qk", query, selected_keys)
        allowed = torch.arange(len(kf), device=qf.device)[None, :] <= query_positions[:, None]
        kernel = kernel.masked_fill(~allowed, 0.0)
        denominator = kernel.sum(-1)
        mass = kernel.abs().sum(-1)
        margin = self.denominator_margin * mass.detach().clamp_min(torch.finfo(kernel.dtype).tiny)
        valid = ((denominator > margin) & torch.isfinite(denominator)
                 & torch.isfinite(kernel).all(-1) & (mass > 0))
        relative_denominator = denominator / mass.detach().clamp_min(torch.finfo(kernel.dtype).tiny)
        penalty = torch.relu(self.denominator_margin - relative_denominator).square().mean()
        if training_safe:
            divisor = torch.maximum(denominator, margin)
        else:
            # Invalid rows remain explicitly invalid rather than silently clipped.
            divisor = torch.where(valid, denominator, torch.ones_like(denominator))
        attention = kernel / divisor[:, None]
        output = attention @ v
        valid = valid & torch.isfinite(output).all(-1)
        if not training_safe:
            attention = attention.masked_fill(~valid[:, None], torch.nan)
            output = output.masked_fill(~valid[:, None], torch.nan)
        return {"kernel": kernel, "denominator": denominator, "attention": attention,
                "output": output, "valid": valid, "denominator_penalty": penalty,
                "support": support}

    def forward(self, q, k, v, query_positions=None, training_safe=False):
        if query_positions is None:
            query_positions = torch.arange(len(q), device=q.device)
        return self.forward_features(self.bank(q), self.bank(k), v, query_positions, training_safe)
