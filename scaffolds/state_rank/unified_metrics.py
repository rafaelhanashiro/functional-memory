"""Read-only aggregation of retained state-rank archives for unified figures.

Input/head rank statistics are already measured in each archive. This module
averages those scalar statistics directly, never an averaged spectrum. Token
curves can retain each random model for audit or average random models at a
fixed positional head index for grouped plots. Depth curves reduce inputs and
heads at the terminal token within one model condition.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from scaffolds.state_rank.plot_comparison import (
    Condition, RANK_METRICS, SCALAR_METRICS, load_conditions,
)

RANK_KEYS = tuple(key for key, _ in RANK_METRICS)


@dataclass
class ModelAnalysis:
    key: str
    label: str
    results: Path
    manifest: dict
    conditions: list[Condition]
    layer: int
    inputs: list[dict]

    @property
    def width(self) -> int:
        """Number of singular modes: 64 for this Llama, 256 for this DeltaNet."""
        return int(self.conditions[0].arrays["normalized_eigenvalues"].shape[-1])

    @property
    def layers(self) -> np.ndarray:
        return self.conditions[0].arrays["layers"]

    @property
    def head_indices(self) -> np.ndarray:
        return self.conditions[0].arrays["head_indices"]

    @property
    def prefixes(self) -> np.ndarray:
        return self.conditions[0].arrays["prefixes"]

    @property
    def model_depth(self) -> int:
        value = self.conditions[0].metadata.get("model_config", {}).get("num_hidden_layers")
        return int(value) if value is not None else int(self.layers[-1]) + 1

    @property
    def relative_depth(self) -> np.ndarray:
        """Zero-based model layer / (full model depth - 1), not measured-row index."""
        if self.model_depth == 1:
            return np.zeros(len(self.layers), dtype=float)
        return self.layers.astype(float) / (self.model_depth - 1)

    @property
    def head_count(self) -> int:
        return len(self.head_indices)

    @property
    def input_count(self) -> int:
        return len(self.inputs)


def load_analysis(results: str | Path, key: str, label: str, layer: int | None = None) -> ModelAnalysis:
    """Require a complete matched comparison and the immutable saved input records.

    Token equality is checked inside one architecture. Different model-native
    tokenizers across separate ModelAnalysis objects are intentionally allowed.
    ``load_conditions`` verifies archive checksums and condition identities.
    """
    results = Path(results).expanduser().resolve()
    manifest, conditions, statuses = load_conditions(results)
    if not statuses or not conditions:
        raise ValueError(f"{key}: no completed comparison conditions")
    incomplete = [entry["name"] for entry in statuses if entry["status"] != "complete"]
    if incomplete or len(conditions) != len(statuses):
        raise ValueError(f"{key}: all planned conditions must be complete; incomplete: {incomplete}")
    pretrained = [condition for condition in conditions if condition.weights == "pretrained"]
    random = [condition for condition in conditions if condition.weights == "random"]
    if len(pretrained) != 1 or not random or pretrained[0].seed is not None:
        raise ValueError(f"{key}: comparison requires one pretrained model and at least one random model")
    seeds = [condition.seed for condition in random]
    if (any(type(seed) is not int or seed < 0 for seed in seeds) or len(seeds) != len(set(seeds))):
        raise ValueError(f"{key}: random model seeds must be distinct nonnegative integers")
    config = manifest.get("config", {})
    arrays = conditions[0].arrays
    for config_key, array_key in (("layers", "layers"), ("spectrum_prefixes", "spectrum_prefixes")):
        if config_key in config and not np.array_equal(config[config_key], arrays[array_key]):
            raise ValueError(f"{key}: measured {array_key} differ from the resolved configuration")
    if "tokens" in config and config["tokens"] != len(arrays["prefixes"]):
        raise ValueError(f"{key}: measured token count differs from the resolved configuration")
    if "seeds" in config and config["seeds"] != seeds:
        raise ValueError(f"{key}: random seeds differ from the resolved configuration")
    inputs_path = results / "inputs.json"
    raw_inputs = inputs_path.read_bytes()
    if manifest.get("inputs_sha256") and hashlib.sha256(raw_inputs).hexdigest() != manifest["inputs_sha256"]:
        raise ValueError(f"{key}: saved input records fail their immutable checksum")
    inputs = json.loads(raw_inputs)
    if not isinstance(inputs, list) or not inputs or any(not isinstance(item, dict) for item in inputs):
        raise ValueError(f"{key}: inputs.json must contain a nonempty list of input records")
    tokens = [item.get("token_ids") for item in inputs]
    if any(not isinstance(ids, list) or any(type(token) is not int for token in ids) for ids in tokens):
        raise ValueError(f"{key}: input records must contain integer token_ids")
    if any(len(ids) != arrays["token_ids"].shape[1] for ids in tokens):
        raise ValueError(f"{key}: saved input token lengths differ from measured token IDs")
    if not np.array_equal(np.asarray(tokens), arrays["token_ids"]):
        raise ValueError(f"{key}: saved input token IDs differ from measured token IDs")
    if "input_count" in config and config["input_count"] != len(inputs):
        raise ValueError(f"{key}: input count differs from the resolved configuration")
    layer = config.get("comparison_layer") if layer is None else layer
    if type(layer) is not int or layer not in arrays["layers"]:
        raise ValueError(f"{key}: selected layer {layer!r} is not a measured model layer")
    depths = [condition.metadata.get("model_config", {}).get("num_hidden_layers") for condition in conditions]
    recorded = [depth for depth in depths if depth is not None]
    if recorded:
        if len(recorded) != len(conditions) or any(type(depth) is not int or depth < 1 for depth in recorded):
            raise ValueError(f"{key}: full model depth is missing or invalid in condition metadata")
        if len(set(recorded)) != 1 or arrays["layers"][-1] >= recorded[0]:
            raise ValueError(f"{key}: model depth differs between conditions or excludes a measured layer")
    elif not np.array_equal(arrays["layers"], np.arange(len(arrays["layers"]))):
        raise ValueError(f"{key}: subset layer archives require the full model depth in metadata")
    # Avoid accidental in-memory writes by a plotting caller. No original files
    # are opened for writing, and each aggregation below produces fresh arrays.
    for condition in conditions:
        for values in condition.arrays.values():
            values.setflags(write=False)
    return ModelAnalysis(key, label, results, manifest, conditions, layer, inputs)


def _layer_index(condition: Condition, layer: int) -> int:
    if type(layer) is not int:
        raise ValueError("Layer must be a zero-based integer model index")
    matches = np.flatnonzero(condition.arrays["layers"] == layer)
    if len(matches) != 1:
        raise ValueError(f"Layer {layer} is not uniquely present in {condition.name}")
    return int(matches[0])


def head_curves(condition: Condition, layer: int, input_index: int | None = None) -> dict[str, dict[str, np.ndarray]]:
    """[T,H] curves: reduce sequences only, preserving all heads of this one model.

    With ``input_index``, all three summaries are copies of that input's curve;
    the band then has zero width. Head indices are never matched across seeds.
    """
    index = _layer_index(condition, layer)
    if input_index is not None and (type(input_index) is not int or not 0 <= input_index < condition.arrays["token_ids"].shape[0]):
        raise ValueError("input_index must be a valid zero-based saved input index")
    result = {}
    for metric in SCALAR_METRICS:
        if metric not in condition.arrays:
            continue
        values = condition.arrays[metric][index]  # [I,T,H]
        if input_index is not None:
            selected = values[input_index]
            result[metric] = {key: selected.copy() for key in ("mean", "min", "max")}
        else:
            result[metric] = {"mean": values.mean(axis=0), "min": values.min(axis=0), "max": values.max(axis=0)}
    return result



def grouped_curves(analysis: ModelAnalysis, view: str = "heads") -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Average saved scalar metrics into pretrained/random plotting groups.

    ``heads`` returns [T,H] arrays at ``analysis.layer``. Pretrained mean/min/max
    reduce sequences only. Each random model first averages its sequences, then
    mean/min/max reduce those model-level curves at each positional head index.
    The same index across seeds is not asserted to identify a shared head role.

    ``layers`` returns [L,T] arrays. Pretrained first averages heads within each
    sequence, then mean/min/max reduce sequences. Each random model averages
    sequences and heads; mean/min/max then reduce those per-seed layer curves.
    Bands therefore describe sequence spread for pretrained and seed spread for
    random models. These are descriptive ranges, not confidence intervals.

    Per-condition helpers and CSV exports remain separate audit records. No
    rank statistic is recomputed from pooled or averaged spectra.
    """
    if view not in ("heads", "layers"):
        raise ValueError("view must be 'heads' or 'layers'")
    pretrained = [condition for condition in analysis.conditions if condition.weights == "pretrained"]
    random = [condition for condition in analysis.conditions if condition.weights == "random"]
    if len(pretrained) != 1 or not random:
        raise ValueError("Grouped curves require one pretrained model and random controls")
    reference = pretrained[0]
    groups = {"pretrained": {}, "random": {}}
    for metric in SCALAR_METRICS:
        if metric not in reference.arrays:
            continue
        if view == "heads":
            # [I,T,H] samples for the pretrained sequence range.
            reference_samples = reference.arrays[metric][_layer_index(reference, analysis.layer)]
            # [S,T,H] samples for the random-model range after sequence means.
            random_samples = np.stack([
                condition.arrays[metric][_layer_index(condition, analysis.layer)].mean(axis=0)
                for condition in random
            ])
        else:
            # [I,L,T] sequence samples after within-sequence head averaging.
            reference_samples = reference.arrays[metric].mean(axis=3).transpose(1, 0, 2)
            # [S,L,T] model samples after averaging sequences and heads.
            random_samples = np.stack([
                condition.arrays[metric].mean(axis=(1, 3)) for condition in random
            ])
        for group, samples in (("pretrained", reference_samples), ("random", random_samples)):
            groups[group][metric] = {
                "mean": samples.mean(axis=0), "min": samples.min(axis=0), "max": samples.max(axis=0),
            }
    return groups


def depth_curves(condition: Condition) -> dict[str, np.ndarray]:
    """[L] curves: terminal-token scalar metric, then mean over inputs and heads."""
    return {metric: condition.arrays[metric][:, :, -1, :].mean(axis=(1, 2))
            for metric in SCALAR_METRICS if metric in condition.arrays}


def comparison_statistics(analysis: ModelAnalysis) -> dict[str, dict]:
    """Descriptive depth statistics across model-level seed means, never matched heads."""
    pretrained = [condition for condition in analysis.conditions if condition.weights == "pretrained"]
    random = [condition for condition in analysis.conditions if condition.weights == "random"]
    if len(pretrained) != 1 or not random:
        raise ValueError("Comparison statistics require one pretrained model and random controls")
    reference = depth_curves(pretrained[0])
    random_curves = [depth_curves(condition) for condition in random]
    result = {}
    for metric in RANK_KEYS:
        values = np.stack([curves[metric] for curves in random_curves])
        average = values.mean(axis=0)
        # No inferential tolerance or significance test: classify these saved
        # descriptive means by their exact floating-point ordering.
        above, below, equal = reference[metric] > average, reference[metric] < average, reference[metric] == average
        result[metric] = {
            "layers": analysis.layers.tolist(), "pretrained": reference[metric].tolist(),
            "random_mean": average.tolist(), "random_min": values.min(axis=0).tolist(),
            "random_max": values.max(axis=0).tolist(),
            "random_by_condition": {condition.name: values[index].tolist() for index, condition in enumerate(random)},
            "pretrained_above_random_mean": analysis.layers[above].tolist(),
            "pretrained_below_random_mean": analysis.layers[below].tolist(),
            "pretrained_equal_random_mean": analysis.layers[equal].tolist(),
        }
    return result


def summary_rows(analyses: Sequence[ModelAnalysis]) -> dict[str, list[dict]]:
    if len({analysis.key for analysis in analyses}) != len(analyses):
        raise ValueError("Model analysis keys must be distinct")
    head_rows, depth_rows = [], []
    for analysis in analyses:
        for condition in analysis.conditions:
            identity = {"model": analysis.key, "model_label": analysis.label, "condition": condition.name,
                        "weights": condition.weights, "seed": condition.seed, "state_width": analysis.width,
                        "inputs": analysis.input_count, "terminal_prefix": int(analysis.prefixes[-1])}
            heads = head_curves(condition, analysis.layer)
            for head_index, head in enumerate(analysis.head_indices):
                row = identity | {"layer": analysis.layer, "head": int(head)}
                for metric in RANK_KEYS:
                    for statistic in ("mean", "min", "max"):
                        row[f"{metric}_{statistic}"] = float(heads[metric][statistic][-1, head_index])
                head_rows.append(row)
            depth = depth_curves(condition)
            for layer_index, layer in enumerate(analysis.layers):
                row = identity | {"layer": int(layer), "relative_depth": float(analysis.relative_depth[layer_index]),
                                  "heads": analysis.head_count}
                for metric in RANK_KEYS:
                    row[metric] = float(depth[metric][layer_index])
                    row[f"{metric}_fraction"] = row[metric] / analysis.width
                depth_rows.append(row)
    return {"per_head_terminal": head_rows, "depth_summary": depth_rows}


def write_summaries(output: str | Path, analyses: Sequence[ModelAnalysis]) -> dict[str, int]:
    """Write only the new derived CSVs; never write inside original result packages."""
    output = Path(output).expanduser().resolve()
    if any(output == analysis.results or analysis.results in output.parents for analysis in analyses):
        raise ValueError("Derived summaries must be outside original result directories")
    rows = summary_rows(analyses)
    base = ["model", "model_label", "condition", "weights", "seed", "state_width", "inputs", "terminal_prefix"]
    fields = {
        "per_head_terminal": base + ["layer", "head"] + [f"{metric}_{statistic}" for metric in RANK_KEYS for statistic in ("mean", "min", "max")],
        "depth_summary": base + ["layer", "relative_depth", "heads"] + [field for metric in RANK_KEYS for field in (metric, f"{metric}_fraction")],
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, entries in rows.items():
        destination = output / f"{name}.csv"
        temporary = destination.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as stream:
            # Match the repository's CSV LF rule so provenance hashes survive
            # staging, merging and a fresh Git checkout byte-for-byte.
            writer = csv.DictWriter(stream, fieldnames=fields[name], lineterminator="\n")
            writer.writeheader()
            writer.writerows(entries)
        temporary.replace(destination)
    return {f"{name}_rows": len(entries) for name, entries in rows.items()}
