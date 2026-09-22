"""OFM Transformer port with independently controlled query/key head widths."""

from experiments.shared_transformer.models.registry import build_model, model_parameter_count

__all__ = ["build_model", "model_parameter_count"]
