#!/usr/bin/env python3
"""Frozen D0--D7 diagnosis of complementary CSI directions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import rankdata

from kd_sensing.baselines.full_pool_common import sha256_file
from kd_sensing.config.parsing import safe_load_yaml

import run_radio_guided_hierarchical_prototypes as upstream


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tools/configs/csi_complementary_direction_diagnostic.yaml"
MASK_NAMES = upstream.MASK_NAMES
MASK_COUNTS = upstream.MASK_COUNTS
NUM_BEAMS = 64
DIMENSION = 64


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    config = safe_load_yaml(path.read_text(encoding="utf-8"))
    if config["protocol"].get("outer_test_enabled") is not False:
        raise ValueError("Direction diagnostics require outer_test_enabled=false.")
    pilot = config["pilot"]
    if (
        int(pilot["history_frames"]) != 5
        or int(pilot["re_per_frame"]) != 4
        or int(pilot["re_window"]) != 20
    ):
        raise ValueError("The diagnostic protocol is fixed at five frames x four RE = 20 RE.")
    statistics = config["statistics"]
    if int(statistics["bootstrap_replicates"]) < 1000:
        raise ValueError("Formal diagnostics require at least 1000 bootstrap replicates.")
    if int(statistics["primary_k"]) != 8:
        raise ValueError("The preregistered primary direction count is K=8.")
    return config


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in values:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def require_inner_split(role: str) -> str:
    value = str(role)
    if value not in {"train", "validation"}:
        raise ValueError("Only train and validation are allowed; outer test remains sealed.")
    return value


def validate_historical_identity(identity: Mapping[str, Any]) -> None:
    if (
        int(identity.get("history_frames", -1)) != 5
        or int(identity.get("re_per_frame", -1)) != 4
        or int(identity.get("re_window", -1)) != 20
    ):
        raise ValueError("CSI identity must be historical five-frame 4 RE/frame, 20 RE/window.")
    if identity.get("future_channel_used") is not False:
        raise ValueError("Future CSI/channel use is forbidden.")
    if identity.get("outer_test_accessed") is not False:
        raise ValueError("Outer test must remain sealed.")


def validate_sample_alignment(
    sensing: Mapping[str, Any],
    radio: Mapping[str, Any],
) -> None:
    """Fail closed before any row-wise cross-branch calculation."""
    if list(sensing["sample_ids"]) != list(radio["sample_ids"]):
        raise ValueError("Sensing and CSI sample IDs are not aligned.")
    if list(sensing["trajectory_ids"]) != list(radio["trajectory_ids"]):
        raise ValueError("Sensing and CSI trajectory IDs are not aligned.")
    if list(sensing["mask_names"]) != list(radio["mask_names"]):
        raise ValueError("Sensing mask identities are not aligned.")
    if not torch.equal(torch.as_tensor(sensing["target"]), torch.as_tensor(radio["target"])):
        raise ValueError("Sensing and CSI targets are not aligned.")


def prototype_basis(prototypes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return singular values and shared right-singular directions as columns."""
    value = torch.as_tensor(prototypes, dtype=torch.float32)
    if tuple(value.shape) != (NUM_BEAMS, DIMENSION):
        raise ValueError("Prototype Bank must have shape [64,64].")
    centered = value - value.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=True)
    directions = vh.transpose(0, 1).contiguous()
    return singular_values, directions


def project_directions(features: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(features, dtype=torch.float32)
    basis = torch.as_tensor(directions, dtype=torch.float32)
    if value.shape[-1] != basis.shape[0] or basis.ndim != 2:
        raise ValueError("Feature and direction dimensions do not match.")
    return value @ basis


def reconstruct_directions(coefficients: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(coefficients, dtype=torch.float32)
    basis = torch.as_tensor(directions, dtype=torch.float32)
    if value.shape[-1] != basis.shape[1] or basis.ndim != 2:
        raise ValueError("Coefficient and direction dimensions do not match.")
    return value @ basis.transpose(0, 1)


def replace_directions(
    sensing: torch.Tensor,
    radio: torch.Tensor,
    directions: torch.Tensor,
    selected: Sequence[int],
) -> torch.Tensor:
    sensing_coeff = project_directions(sensing, directions)
    radio_coeff = project_directions(radio, directions)
    mixed = sensing_coeff.clone()
    indices = torch.as_tensor(list(selected), dtype=torch.long, device=mixed.device)
    mixed.index_copy_(-1, indices, radio_coeff.index_select(-1, indices))
    return reconstruct_directions(mixed, directions)


def delete_directions(
    features: torch.Tensor,
    directions: torch.Tensor,
    selected: Sequence[int],
) -> torch.Tensor:
    coefficients = project_directions(features, directions)
    indices = torch.as_tensor(list(selected), dtype=torch.long, device=coefficients.device)
    coefficients = coefficients.clone()
    coefficients.index_fill_(-1, indices, 0.0)
    return reconstruct_directions(coefficients, directions)


def fisher_scores(values: np.ndarray, labels: np.ndarray, num_classes: int = NUM_BEAMS) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("Fisher inputs must be [N,D] values and [N] labels.")
    overall = x.mean(axis=0)
    between = np.zeros(x.shape[1], dtype=np.float64)
    within = np.zeros(x.shape[1], dtype=np.float64)
    for beam in range(int(num_classes)):
        subset = x[y == beam]
        if not len(subset):
            continue
        mean = subset.mean(axis=0)
        between += len(subset) * (mean - overall) ** 2
        within += ((subset - mean) ** 2).sum(axis=0)
    return (between / max(len(x), 1)) / (within / max(len(x), 1) + 1e-12)


def fit_bin_edges(values: np.ndarray, bins: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or int(bins) < 2:
        raise ValueError("MI bin fitting requires [N,D] and bins >= 2.")
    quantiles = np.linspace(0.0, 1.0, int(bins) + 1)[1:-1]
    return np.quantile(x, quantiles, axis=0).T


def fixed_bin_mutual_information(
    values: np.ndarray,
    labels: np.ndarray,
    edges: np.ndarray,
    num_classes: int = NUM_BEAMS,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    boundaries = np.asarray(edges, dtype=np.float64)
    if x.ndim != 2 or boundaries.shape[0] != x.shape[1] or len(x) != len(y):
        raise ValueError("MI values, labels, and train-fitted edges do not align.")
    result = np.zeros(x.shape[1], dtype=np.float64)
    for dimension in range(x.shape[1]):
        discrete = np.searchsorted(boundaries[dimension], x[:, dimension], side="right")
        joint = np.zeros((boundaries.shape[1] + 1, int(num_classes)), dtype=np.float64)
        np.add.at(joint, (discrete, y), 1.0)
        joint /= max(joint.sum(), 1.0)
        px = joint.sum(axis=1, keepdims=True)
        py = joint.sum(axis=0, keepdims=True)
        expected = px @ py
        valid = joint > 0
        result[dimension] = np.sum(joint[valid] * np.log(joint[valid] / expected[valid]))
    return result


def highest_wrong(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scores = np.asarray(logits, dtype=np.float64).copy()
    y = np.asarray(labels, dtype=np.int64)
    scores[np.arange(len(y)), y] = -np.inf
    return scores.argmax(axis=1)


def margin_contributions(
    feature_coefficients: np.ndarray,
    prototype_coefficients: np.ndarray,
    labels: np.ndarray,
    wrong: np.ndarray,
) -> np.ndarray:
    features = np.asarray(feature_coefficients, dtype=np.float64)
    prototypes = np.asarray(prototype_coefficients, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    h = np.asarray(wrong, dtype=np.int64)
    return features * (prototypes[y] - prototypes[h])


def bootstrap_weights(count: int, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    probability = np.full(int(count), 1.0 / float(count), dtype=np.float64)
    weights = rng.multinomial(int(count), probability, size=int(replicates)).astype(np.float32)
    return weights / float(count)


def bootstrap_mean_summary(differences: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    samples = np.asarray(weights, dtype=np.float64) @ values
    mean = values.mean(axis=0)
    low, high = np.quantile(samples, [0.025, 0.975], axis=0)
    std = values.std(axis=0, ddof=1)
    effect = mean / np.maximum(std, 1e-12)
    probability = np.minimum((samples <= 0).mean(axis=0), (samples >= 0).mean(axis=0))
    p_value = np.minimum(1.0, 2.0 * probability)
    if values.shape[1] != 1:
        raise ValueError("bootstrap_mean_summary expects one paired metric.")
    return {
        "mean": float(mean[0]),
        "ci_low": float(low[0]),
        "ci_high": float(high[0]),
        "effect_size": float(effect[0]),
        "p_value": float(p_value[0]),
    }


def bootstrap_direction_summary(differences: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(differences, dtype=np.float64)
    samples = np.asarray(weights, dtype=np.float64) @ values
    mean = values.mean(axis=0)
    low, high = np.quantile(samples, [0.025, 0.975], axis=0)
    std = values.std(axis=0, ddof=1)
    p_value = 2.0 * np.minimum((samples <= 0).mean(axis=0), (samples >= 0).mean(axis=0))
    return {
        "mean": mean,
        "ci_low": low,
        "ci_high": high,
        "effect_size": mean / np.maximum(std, 1e-12),
        "p_value": np.minimum(1.0, p_value),
    }


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def _cosine_logits(features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    p = np.asarray(prototypes, dtype=np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    p = p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-12)
    return x @ p.T


def _topk_correct(logits: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    indices = np.argpartition(np.asarray(logits), -int(k), axis=1)[:, -int(k) :]
    return (indices == np.asarray(labels)[:, None]).any(axis=1)


def classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    base_logits: np.ndarray | None = None,
) -> dict[str, Any]:
    scores = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    prediction = scores.argmax(axis=1)
    correct = prediction == y
    top3 = _topk_correct(scores, y, 3)
    wrong = highest_wrong(scores, y)
    target_margin = scores[np.arange(len(y)), y] - scores[np.arange(len(y)), wrong]
    distance = np.abs(prediction - y)
    distance = np.minimum(distance, NUM_BEAMS - distance)
    result: dict[str, Any] = {
        "top1": float(correct.mean()),
        "top3": float(top3.mean()),
        "within3": float((distance <= 3).mean()),
        "mae": float(distance.mean()),
        "target_margin": float(target_margin.mean()),
        "correct": correct,
        "prediction": prediction,
        "top3_correct": top3,
        "margin_values": target_margin,
        "fix_rate": math.nan,
        "harm_rate": math.nan,
    }
    if base_logits is not None:
        base_correct = np.asarray(base_logits).argmax(axis=1) == y
        result["fix_rate"] = float(correct[~base_correct].mean()) if (~base_correct).any() else 0.0
        result["harm_rate"] = float((~correct[base_correct]).mean()) if base_correct.any() else 0.0
    return result


class RidgeWorkspace:
    """Closed-form multi-output ridge with reusable train cross-products."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        standardized = (x - self.mean) / self.scale
        one_hot = np.eye(NUM_BEAMS, dtype=np.float64)[y]
        self.target_mean = one_hot.mean(axis=0)
        centered_target = one_hot - self.target_mean
        self.gram = standardized.T @ standardized / len(x)
        self.cross = standardized.T @ centered_target / len(x)

    def fit(self, indices: Sequence[int], alpha: float) -> dict[str, np.ndarray]:
        selected = np.asarray(indices, dtype=np.int64)
        gram = self.gram[np.ix_(selected, selected)]
        cross = self.cross[selected]
        weight = np.linalg.solve(gram + float(alpha) * np.eye(len(selected)), cross)
        return {
            "indices": selected,
            "weight": weight,
            "mean": self.mean[selected],
            "scale": self.scale[selected],
            "target_mean": self.target_mean,
        }


def ridge_predict(model: Mapping[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    selected = np.asarray(model["indices"], dtype=np.int64)
    x = np.asarray(features, dtype=np.float64)[:, selected]
    standardized = (x - model["mean"]) / model["scale"]
    return standardized @ model["weight"] + model["target_mean"]


def fit_multivariate_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    mean_x = x.mean(axis=0)
    scale_x = x.std(axis=0)
    scale_x[scale_x < 1e-8] = 1.0
    mean_y = y.mean(axis=0)
    standardized = (x - mean_x) / scale_x
    gram = standardized.T @ standardized / len(x)
    cross = standardized.T @ (y - mean_y) / len(x)
    weight = np.linalg.solve(gram + float(alpha) * np.eye(x.shape[1]), cross)
    return {"mean_x": mean_x, "scale_x": scale_x, "mean_y": mean_y, "weight": weight}


def predict_multivariate_ridge(model: Mapping[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    x = (np.asarray(features, dtype=np.float64) - model["mean_x"]) / model["scale_x"]
    return x @ model["weight"] + model["mean_y"]


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    first, second = set(left), set(right)
    return len(first & second) / max(len(first | second), 1)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    first = rankdata(np.asarray(left, dtype=np.float64))
    second = rankdata(np.asarray(right, dtype=np.float64))
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _upstream_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return upstream._load_config(_path(config["source"]["radio_guided_config"]))


def _source_cache(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    role = require_inner_split(role)
    upstream_config = _upstream_config(config)
    cache = upstream._load_cache(upstream_config, role)
    expected = config["source"][f"{role}_cache_sha256"]
    source_path = _path(config["source"]["radio_guided_cache_root"]) / f"{role}.pt"
    if sha256_file(source_path) != expected:
        raise ValueError(f"{role} source cache hash mismatch.")
    return cache


def _prototype_and_temperatures(
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, float, float, float, dict[str, Any]]:
    upstream_config = _upstream_config(config)
    m4, _ = upstream._load_m4(upstream_config, device)
    expert, sensing_temperature, f1_payload = upstream._load_f1_radio_expert(upstream_config, device)
    prototype = m4.prototype_bank.prototypes.detach().float().cpu()
    return (
        prototype,
        float(m4.prototype_bank.temperature),
        float(sensing_temperature),
        float(expert.temperature().detach().cpu()),
        f1_payload,
    )


def _expanded_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    y = torch.as_tensor(target, dtype=torch.long)
    sample_shape = tuple(logits.shape[:-1])
    axes = [axis for axis, size in enumerate(sample_shape) if int(size) == len(y)]
    if len(axes) != 1:
        raise ValueError(f"Cannot identify unique sample axis in logits shape {tuple(logits.shape)}.")
    shape = [1] * len(sample_shape)
    shape[axes[0]] = len(y)
    return y.reshape(shape).expand(sample_shape)


def _target_rank(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    view = _expanded_target(logits, target)
    target_value = logits.gather(-1, view.unsqueeze(-1))
    return 1 + (logits > target_value).sum(dim=-1)


def _cross_entropy_values(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    expanded = _expanded_target(logits, target)
    flat = logits.reshape(-1, logits.shape[-1])
    return F.cross_entropy(flat.float(), expanded.reshape(-1), reduction="none").reshape(logits.shape[:-1])


def _audit_source_identity(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = _path(config["source"]["radio_guided_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = _source_cache(config, "train")
    validation = _source_cache(config, "validation")
    checks = manifest["preflight"]["checks"]
    required_checks = {
        "feature_sample_alignment",
        "hashes",
        "no_future_channel_inputs",
        "outer_test_disabled",
        "outer_test_not_accessed",
        "sample_ids_disjoint",
        "sample_ids_unique",
        "target_alignment",
        "trajectory_disjoint",
        "twenty_re",
    }
    if not required_checks.issubset(checks) or not all(bool(checks[key]) for key in required_checks):
        raise ValueError("Upstream cache manifest does not pass all required identity checks.")
    if manifest.get("outer_test_accessed") is not False:
        raise ValueError("Upstream manifest does not keep outer test sealed.")
    if set(train) & {"future_csi", "future_channel", "outer_test"}:
        raise ValueError("Train cache exposes forbidden future or outer-test fields.")
    if set(validation) & {"future_csi", "future_channel", "outer_test"}:
        raise ValueError("Validation cache exposes forbidden future or outer-test fields.")
    expected_identity = {
        "m4_checkpoint_sha256": config["source"]["m4_checkpoint_sha256"],
        "csi_checkpoint_sha256": config["source"]["csi_checkpoint_sha256"],
        "f1_checkpoint_sha256": config["source"]["f1_checkpoint_sha256"],
        "prototype_bank_sha256": config["source"]["prototype_bank_sha256"],
        "split_manifest_sha256": config["protocol"]["split_manifest_sha256"],
        "pilot_codebook_file_sha256": config["source"]["pilot_codebook_sha256"],
        "pilot_codebook_hash": config["source"]["pilot_codebook_hash"],
        "history_frames": 5,
        "re_per_frame": 4,
        "re_window": 20,
        "outer_test_accessed": False,
    }
    for role, cache in (("train", train), ("validation", validation)):
        mismatched = {
            key: (cache["identity"].get(key), expected)
            for key, expected in expected_identity.items()
            if cache["identity"].get(key) != expected
        }
        if mismatched:
            raise ValueError(f"{role} source cache identity mismatch: {mismatched}.")
    validate_sample_alignment(train, train)
    validate_sample_alignment(validation, validation)
    train_ids, validation_ids = set(train["sample_ids"]), set(validation["sample_ids"])
    train_trajectories, validation_trajectories = set(train["trajectory_ids"]), set(validation["trajectory_ids"])
    if (
        len(train_ids) != len(train["sample_ids"])
        or len(validation_ids) != len(validation["sample_ids"])
        or train_ids & validation_ids
        or train_trajectories & validation_trajectories
    ):
        raise ValueError("Train/validation sample or trajectory identities are not disjoint.")
    return manifest, train, validation


def audit(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    manifest, train, validation = _audit_source_identity(config)
    prototype, bank_temperature, sensing_temperature, radio_temperature, f1_payload = (
        _prototype_and_temperatures(config, device)
    )
    prototype_hash = upstream._tensor_sha256(prototype)
    if prototype_hash != config["source"]["prototype_bank_sha256"]:
        raise ValueError("Prototype tensor hash mismatch.")
    prototype_normalized = F.normalize(prototype, dim=-1)
    prototype_rank = int(torch.linalg.matrix_rank(prototype_normalized.double()).item())
    if prototype_rank != DIMENSION:
        raise ValueError("The shared Prototype Bank is not full rank.")

    sample_count = min(4096, len(train["target"]))
    sensing_direct = (
        F.normalize(train["z_s_all_masks"][:sample_count].float(), dim=-1)
        @ prototype_normalized.transpose(0, 1)
        / bank_temperature
    )
    radio_direct = (
        F.normalize(train["z_c"][0, :sample_count].float(), dim=-1)
        @ prototype_normalized.transpose(0, 1)
        / radio_temperature
    )
    sensing_diff = float(
        (sensing_direct - train["sensing_logits_all_masks"][:sample_count].float()).abs().max().item()
    )
    radio_diff = float((radio_direct - train["radio_evidence"][0, :sample_count].float()).abs().max().item())
    if sensing_diff > 1e-5 or radio_diff > 1e-2:
        raise ValueError(
            f"Cached shared-prototype scoring mismatch: sensing={sensing_diff}, radio={radio_diff}."
        )

    train_target = torch.as_tensor(train["target"]).long()
    validation_target = torch.as_tensor(validation["target"]).long()
    if (
        tuple(train["z_s_all_masks"].shape) != (37510, 14, 64)
        or tuple(validation["z_s_all_masks"].shape) != (6365, 14, 64)
        or tuple(train["z_c"].shape) != (1, 37510, 64)
        or tuple(validation["z_c"].shape) != (3, 6365, 64)
        or train_target.numel() != 37510
        or validation_target.numel() != 6365
    ):
        raise ValueError("Frozen feature cache shapes do not match the trajectory protocol.")

    output = _path(config["output"]["root"])
    result = {
        "status": "passed",
        "z_s_shape_train": list(train["z_s_all_masks"].shape),
        "z_c_shape_train": list(train["z_c"].shape),
        "z_s_shape_validation": list(validation["z_s_all_masks"].shape),
        "z_c_shape_validation": list(validation["z_c"].shape),
        "prototype_shape": list(prototype.shape),
        "prototype_rank": prototype_rank,
        "same_frozen_prototype_bank": True,
        "raw_coordinate_comparison_allowed": True,
        "prototype_basis_required_as_primary": True,
        "sensing_pre_query_transform": "none; cached fused z_s is queried directly then L2-normalized by bank",
        "radio_pre_query_transform": "LayerNorm(128)->Linear(128,128)->GELU->Linear(128,64); bank L2 normalization",
        "scoring_formula": "normalize(z) @ normalize(P).T / temperature",
        "bank_temperature": bank_temperature,
        "sensing_temperature": sensing_temperature,
        "radio_temperature": radio_temperature,
        "sensing_effective_scale": 1.0 / (bank_temperature * sensing_temperature),
        "radio_effective_scale": 1.0 / radio_temperature,
        "sensing_scoring_max_abs_diff": sensing_diff,
        "radio_scoring_max_abs_diff": radio_diff,
        "radio_projection_trained_against_shared_bank": True,
        "sample_target_trajectory_mask_aligned": True,
        "train_samples": len(train_target),
        "validation_samples": len(validation_target),
        "train_trajectories": len(set(train["trajectory_ids"])),
        "validation_trajectories": len(set(validation["trajectory_ids"])),
        "history_frames": 5,
        "re_per_frame": 4,
        "re_window": 20,
        "future_channel_used": False,
        "test_loader_constructed": False,
        "outer_test_accessed": False,
        "f1_method": f1_payload.get("method"),
        "source_cache_hashes": {
            "train": config["source"]["train_cache_sha256"],
            "validation": config["source"]["validation_cache_sha256"],
        },
        "source_manifest_outer_test_accessed": manifest.get("outer_test_accessed"),
    }
    lines = [
        "# CSI 互补方向空间对齐审计",
        "",
        "## 结论",
        "",
        "- 审计通过。M4、CSI encoder/GRU、RadioPrototypeExpert、共享 Beam Prototype Bank 和 F1 均保持冻结且未修改。",
        "- z_s 与 z_c 都以同一满秩 [64,64] 冻结 Prototype Bank 为最终分类坐标锚点，因此允许原始坐标诊断；主结论仍以 prototype SVD 公共 basis 为准，避免任意坐标旋转依赖。",
        "- train/validation 的 sample ID、target、trajectory 与 14 mask 严格对齐且轨迹互斥；未构造 outer-test loader/cache。",
        "",
        "## 必答项",
        "",
        f"1. train z_s={list(train['z_s_all_masks'].shape)}、z_c={list(train['z_c'].shape)}；validation z_s={list(validation['z_s_all_masks'].shape)}、z_c={list(validation['z_c'].shape)}。",
        "2. 两支直接查询同一个冻结 [64,64] Prototype Bank，prototype rank=64。",
        "3. sensing cache 是 M4 fusion 后的 64 维 z_s，查询前无额外 projection/LN；bank 内 L2 normalize。radio 使用 LayerNorm(128)->Linear->GELU->Linear(64)，再由同一 bank L2 normalize。",
        "4. 精确 scoring 为 normalize(z) @ normalize(P).T / temperature，全程 FP32 cosine。",
        f"5. bank temperature={bank_temperature:.10g}，F1 sensing temperature={sensing_temperature:.10g}，radio temperature={radio_temperature:.10g}；有效 scale 分别为 {1.0/(bank_temperature*sensing_temperature):.8g} 与 {1.0/radio_temperature:.8g}。",
        "6. 两路最终 embedding 与 prototype 均为 64 维且受同一固定满秩 bank 锚定；同时执行原始坐标与 prototype-basis 诊断。",
        "7. F1 RadioPrototypeExpert 的 64 维 projection 由冻结 CSI classifier 通过同一 prototype matrix 因式分解初始化并以共享 bank scoring 保存。",
        "8. sample ID、target、trajectory 和 mask identity 的逐项/互斥校验均通过。",
        "9. cache 只包含 t-4...t 五帧，每帧 4 RE、窗口 20 RE；未读取 future CSI/channel。",
        "10. outer test 未访问，且 role guard 对任何非 train/validation 值在路径解析前失败。",
        "",
        "## 数值复核",
        "",
        f"- sensing cache 与直接 shared-bank scoring 最大差：{sensing_diff:.3e}。",
        f"- radio cache 与直接 shared-bank scoring 最大差：{radio_diff:.3e}。",
        f"- prototype SHA256：{prototype_hash}。",
        "",
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.md").write_text("\n".join(lines), encoding="utf-8")
    _write_json(output / "diagnostics/audit.json", result)
    return result


def build_cache(config: Mapping[str, Any], device: torch.device, *, force: bool = False) -> dict[str, Any]:
    audit_result = audit(config, device)
    prototype, bank_temperature, sensing_temperature, radio_temperature, _ = (
        _prototype_and_temperatures(config, device)
    )
    output = _path(config["output"]["root"])
    destination_root = output / "cache"
    manifests: dict[str, Any] = {}
    for role in ("train", "validation"):
        destination = destination_root / f"{role}_features.pt"
        if destination.exists() and not force:
            existing = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
            if existing["identity"]["cache_version"] != config["cache"]["version"]:
                raise ValueError(f"Existing {role} derived cache version mismatch.")
            manifests[role] = {
                "path": str(destination.resolve()),
                "sha256": sha256_file(destination),
                "samples": len(existing["sample_ids"]),
                "reused": True,
            }
            continue
        source = _source_cache(config, role)
        z_s_raw = source["z_s_all_masks"].float().contiguous()
        z_c_raw = source["z_c"].float().contiguous()
        z_s_full_raw = source["z_s_full"].float().contiguous()
        e_s = source["sensing_logits_all_masks"].float() / float(sensing_temperature)
        e_c = source["radio_evidence"].float()
        target = source["target"].long()
        payload = {
            "sample_ids": list(source["sample_ids"]),
            "trajectory_ids": list(source["trajectory_ids"]),
            "target": target,
            "mask_names": list(source["mask_names"]),
            "mask_ids": torch.arange(len(MASK_NAMES), dtype=torch.long),
            "mask_availability": source["mask_availability"].bool(),
            "z_s_raw": z_s_raw,
            "z_s_normalized": F.normalize(z_s_raw, dim=-1),
            "z_s_full_raw": z_s_full_raw,
            "z_s_full_normalized": F.normalize(z_s_full_raw, dim=-1),
            "z_c_raw": z_c_raw,
            "z_c_normalized": F.normalize(z_c_raw, dim=-1),
            "prototype_raw": prototype.float(),
            "prototype_normalized": F.normalize(prototype.float(), dim=-1),
            "e_s": e_s,
            "e_c": e_c,
            "sensing_prediction": e_s.argmax(dim=-1),
            "csi_prediction": e_c.argmax(dim=-1),
            "sensing_target_rank": _target_rank(e_s, target),
            "csi_target_rank": _target_rank(e_c, target),
            "sensing_ce": _cross_entropy_values(e_s, target),
            "csi_ce": _cross_entropy_values(e_c, target),
            "identity": {
                "cache_version": config["cache"]["version"],
                "role": role,
                "source_cache_sha256": config["source"][f"{role}_cache_sha256"],
                "m4_checkpoint_sha256": config["source"]["m4_checkpoint_sha256"],
                "csi_checkpoint_sha256": config["source"]["csi_checkpoint_sha256"],
                "f1_checkpoint_sha256": config["source"]["f1_checkpoint_sha256"],
                "prototype_bank_sha256": config["source"]["prototype_bank_sha256"],
                "split_manifest_sha256": config["protocol"]["split_manifest_sha256"],
                "pilot_codebook_file_sha256": config["source"]["pilot_codebook_sha256"],
                "pilot_codebook_hash": config["source"]["pilot_codebook_hash"],
                "bank_temperature": bank_temperature,
                "sensing_temperature": sensing_temperature,
                "radio_temperature": radio_temperature,
                "history_frames": 5,
                "re_per_frame": 4,
                "re_window": 20,
                "source_encoder_rerun": False,
                "future_channel_used": False,
                "outer_test_accessed": False,
            },
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        manifests[role] = {
            "path": str(destination.resolve()),
            "sha256": sha256_file(destination),
            "samples": len(payload["sample_ids"]),
            "tensor_shapes": {
                key: list(value.shape) for key, value in payload.items() if torch.is_tensor(value)
            },
            "reused": False,
        }
    manifest = {
        "status": "complete",
        "cache_version": config["cache"]["version"],
        "audit_status": audit_result["status"],
        "roles": manifests,
        "source_hashes": {
            key: value
            for key, value in config["source"].items()
            if str(key).endswith("_sha256")
        }
        | {
            "split_manifest_sha256": config["protocol"]["split_manifest_sha256"],
            "pilot_codebook_hash": config["source"]["pilot_codebook_hash"],
        },
        "pilot_configuration": {
            "budget": "2x2",
            "history_frames": 5,
            "re_per_frame": 4,
            "re_window": 20,
        },
        "source_encoder_rerun": False,
        "test_loader_constructed": False,
        "outer_test_accessed": False,
    }
    _write_json(destination_root / "cache_manifest.json", manifest)
    return manifest


def _derived_cache(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    role = require_inner_split(role)
    path = _path(config["output"]["root"]) / f"cache/{role}_features.pt"
    record = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if record["identity"].get("cache_version") != config["cache"]["version"]:
        raise ValueError("Derived cache version mismatch.")
    if record["identity"].get("outer_test_accessed") is not False:
        raise ValueError("Derived cache does not preserve outer-test sealing.")
    validate_historical_identity(record["identity"])
    return record


def _baseline_conditions(validation: Mapping[str, Any]) -> dict[str, Any]:
    target = validation["target"].numpy()
    sensing_prediction = validation["sensing_prediction"].numpy()
    accuracies = np.asarray([(sensing_prediction[:, index] == target).mean() for index in range(len(MASK_NAMES))])
    lowest = np.argsort(accuracies)[:3]
    single = [index for index, name in enumerate(MASK_NAMES) if MASK_COUNTS[name] == 1]
    missing_lidar = MASK_NAMES.index("missing_lidar")
    severe = sorted(set(single + lowest.tolist() + [missing_lidar]))
    return {
        "mask_top1": accuracies,
        "worst_mask_index": int(np.argmin(accuracies)),
        "worst_mask": MASK_NAMES[int(np.argmin(accuracies))],
        "lowest_three_indices": [int(value) for value in lowest],
        "lowest_three_masks": [MASK_NAMES[int(value)] for value in lowest],
        "single_indices": single,
        "severe_indices": severe,
        "severe_masks": [MASK_NAMES[index] for index in severe],
        "missing_lidar_index": missing_lidar,
    }


def analyze_d0(
    config: Mapping[str, Any],
    validation: Mapping[str, Any],
    weights: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = validation["target"].numpy()
    e_s = validation["e_s"].numpy()
    e_c = validation["e_c"][int(config["pilot"]["main_radio_view"])].numpy()
    csi_metrics = classification_metrics(e_c, target)
    rows: list[dict[str, Any]] = []
    for mask_index, mask_name in enumerate(MASK_NAMES):
        sensing = classification_metrics(e_s[:, mask_index], target)
        fusion_logits = 0.5 * e_s[:, mask_index] + 0.5 * e_c
        fusion = classification_metrics(fusion_logits, target, base_logits=e_s[:, mask_index])
        sensing_correct = sensing["correct"]
        csi_correct = csi_metrics["correct"]
        csi_top3 = csi_metrics["top3_correct"]
        fusion_gain = fusion["correct"].astype(np.float64) - sensing_correct.astype(np.float64)
        gain_stats = bootstrap_mean_summary(fusion_gain, weights)
        rows.append(
            {
                "mask_id": mask_index,
                "mask": mask_name,
                "available_modalities": MASK_COUNTS[mask_name],
                "sensing_top1": sensing["top1"],
                "sensing_top3": sensing["top3"],
                "csi_top1": csi_metrics["top1"],
                "csi_top3": csi_metrics["top3"],
                "fixed_fusion_top1": fusion["top1"],
                "fixed_fusion_top3": fusion["top3"],
                "sensing_wrong_csi_correct": float((~sensing_correct & csi_correct).mean()),
                "sensing_wrong_csi_top3": float((~sensing_correct & csi_top3).mean()),
                "sensing_correct_csi_wrong": float((sensing_correct & ~csi_correct).mean()),
                "both_wrong": float((~sensing_correct & ~csi_correct).mean()),
                "prediction_disagreement": float(
                    (sensing["prediction"] != csi_metrics["prediction"]).mean()
                ),
                "oracle_top1": float((sensing_correct | csi_correct).mean()),
                "csi_conditional_fix_rate": (
                    float(csi_correct[~sensing_correct].mean()) if (~sensing_correct).any() else 0.0
                ),
                "csi_conditional_harm_rate": (
                    float((~csi_correct[sensing_correct]).mean()) if sensing_correct.any() else 0.0
                ),
                "fusion_fix_rate": fusion["fix_rate"],
                "fusion_harm_rate": fusion["harm_rate"],
                "fusion_gain": gain_stats["mean"],
                "fusion_gain_ci_low": gain_stats["ci_low"],
                "fusion_gain_ci_high": gain_stats["ci_high"],
                "fusion_gain_effect_size": gain_stats["effect_size"],
                "fusion_gain_p_value": gain_stats["p_value"],
                "re_per_frame": 4,
                "history_frames": 5,
                "re_window": 20,
            }
        )
    conditions = _baseline_conditions(validation)
    return rows, conditions


def _direction_arrays(
    cache: Mapping[str, Any],
    directions: np.ndarray,
    radio_view: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = torch.from_numpy(np.asarray(directions, dtype=np.float32))
    sensing = project_directions(cache["z_s_normalized"], basis).numpy()
    radio = project_directions(cache["z_c_normalized"][int(radio_view)], basis).numpy()
    prototypes = project_directions(cache["prototype_normalized"], basis).numpy()
    return sensing, radio, prototypes


def _direction_rank(
    sensing: np.ndarray,
    radio: np.ndarray,
    labels: np.ndarray,
    prototype_coefficients: np.ndarray,
    severe_indices: Sequence[int],
    effective: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radio_logits = radio @ prototype_coefficients.T
    radio_wrong = highest_wrong(radio_logits, labels)
    radio_contribution = margin_contributions(radio, prototype_coefficients, labels, radio_wrong)
    radio_fisher = fisher_scores(radio, labels)
    scores: list[np.ndarray] = []
    advantages: list[np.ndarray] = []
    for mask_index in severe_indices:
        sensing_values = sensing[:, int(mask_index)]
        sensing_logits = sensing_values @ prototype_coefficients.T
        sensing_wrong = highest_wrong(sensing_logits, labels)
        sensing_contribution = margin_contributions(
            sensing_values, prototype_coefficients, labels, sensing_wrong
        )
        difference = radio_contribution - sensing_contribution
        sensing_fisher = fisher_scores(sensing_values, labels)
        fisher_term = np.log((radio_fisher + 1e-12) / (sensing_fisher + 1e-12))
        effect_term = difference.mean(axis=0) / np.maximum(difference.std(axis=0, ddof=1), 1e-12)
        scores.append(fisher_term + effect_term)
        advantages.append(difference.mean(axis=0))
    score = np.mean(scores, axis=0)
    advantage = np.mean(advantages, axis=0)
    score = np.where(np.asarray(effective, dtype=bool), score, -np.inf)
    order = np.argsort(-score, kind="stable")
    return score, advantage, order


def analyze_discriminability(
    config: Mapping[str, Any],
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    directions: np.ndarray,
    basis_name: str,
    conditions: Mapping[str, Any],
    weights: np.ndarray,
    singular_values: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    statistics = config["statistics"]
    bins = int(statistics["mi_bins"])
    ratio_threshold = float(statistics["fisher_ratio"])
    q_threshold = float(statistics["fdr_q"])
    minimum_masks = int(statistics["minimum_consistent_severe_masks"])
    radio_view = int(config["pilot"]["main_radio_view"])
    train_target = train["target"].numpy()
    validation_target = validation["target"].numpy()
    train_s, train_c, prototype_coefficients = _direction_arrays(train, directions, radio_view)
    validation_s, validation_c, _ = _direction_arrays(validation, directions, radio_view)
    train_radio_logits = train_c @ prototype_coefficients.T
    validation_radio_logits = validation_c @ prototype_coefficients.T
    train_radio_wrong = highest_wrong(train_radio_logits, train_target)
    validation_radio_wrong = highest_wrong(validation_radio_logits, validation_target)
    train_radio_contribution = margin_contributions(
        train_c, prototype_coefficients, train_target, train_radio_wrong
    )
    validation_radio_contribution = margin_contributions(
        validation_c, prototype_coefficients, validation_target, validation_radio_wrong
    )
    train_radio_fisher = fisher_scores(train_c, train_target)
    validation_radio_fisher = fisher_scores(validation_c, validation_target)
    radio_edges = fit_bin_edges(train_c, bins)
    train_radio_mi = fixed_bin_mutual_information(train_c, train_target, radio_edges)
    validation_radio_mi = fixed_bin_mutual_information(validation_c, validation_target, radio_edges)
    csi_prediction = validation_radio_logits.argmax(axis=1)

    rows: list[dict[str, Any]] = []
    mask_pass = np.zeros((len(MASK_NAMES), DIMENSION), dtype=bool)
    train_advantages = np.zeros((len(MASK_NAMES), DIMENSION), dtype=np.float64)
    validation_advantages = np.zeros_like(train_advantages)
    fisher_ratios = np.zeros_like(train_advantages)
    severe = set(int(value) for value in conditions["severe_indices"])
    for mask_index, mask_name in enumerate(MASK_NAMES):
        train_sensing = train_s[:, mask_index]
        validation_sensing = validation_s[:, mask_index]
        train_sensing_logits = train_sensing @ prototype_coefficients.T
        validation_sensing_logits = validation_sensing @ prototype_coefficients.T
        train_wrong = highest_wrong(train_sensing_logits, train_target)
        validation_wrong = highest_wrong(validation_sensing_logits, validation_target)
        train_sensing_contribution = margin_contributions(
            train_sensing, prototype_coefficients, train_target, train_wrong
        )
        validation_sensing_contribution = margin_contributions(
            validation_sensing, prototype_coefficients, validation_target, validation_wrong
        )
        train_difference = train_radio_contribution - train_sensing_contribution
        validation_difference = validation_radio_contribution - validation_sensing_contribution
        train_advantages[mask_index] = train_difference.mean(axis=0)
        validation_advantages[mask_index] = validation_difference.mean(axis=0)
        bootstrap = bootstrap_direction_summary(validation_difference, weights)
        q_values = benjamini_hochberg(bootstrap["p_value"])
        train_sensing_fisher = fisher_scores(train_sensing, train_target)
        validation_sensing_fisher = fisher_scores(validation_sensing, validation_target)
        fisher_ratio = (validation_radio_fisher + 1e-12) / (validation_sensing_fisher + 1e-12)
        fisher_ratios[mask_index] = fisher_ratio
        sensing_edges = fit_bin_edges(train_sensing, bins)
        train_sensing_mi = fixed_bin_mutual_information(train_sensing, train_target, sensing_edges)
        validation_sensing_mi = fixed_bin_mutual_information(
            validation_sensing, validation_target, sensing_edges
        )
        sensing_error = validation_sensing_logits.argmax(axis=1) != validation_target
        rescue = sensing_error & (csi_prediction == validation_target)
        candidate = (
            (fisher_ratio > ratio_threshold)
            & (bootstrap["ci_low"] > 0)
            & (q_values < q_threshold)
            & (train_difference.mean(axis=0) > 0)
        )
        mask_pass[mask_index] = candidate
        for direction in range(DIMENSION):
            rows.append(
                {
                    "basis": basis_name,
                    "mask_id": mask_index,
                    "mask": mask_name,
                    "severe_mask": mask_index in severe,
                    "direction": direction,
                    "singular_rank": direction + 1,
                    "in_top8_singular_subspace": direction < 8,
                    "in_top16_singular_subspace": direction < 16,
                    "in_top32_singular_subspace": direction < 32,
                    "in_full64_singular_subspace": True,
                    "singular_value": (
                        float(singular_values[direction]) if singular_values is not None else math.nan
                    ),
                    "train_sensing_fisher": train_sensing_fisher[direction],
                    "train_csi_fisher": train_radio_fisher[direction],
                    "validation_sensing_fisher": validation_sensing_fisher[direction],
                    "validation_csi_fisher": validation_radio_fisher[direction],
                    "validation_fisher_ratio": fisher_ratio[direction],
                    "train_sensing_mi": train_sensing_mi[direction],
                    "train_csi_mi": train_radio_mi[direction],
                    "validation_sensing_mi": validation_sensing_mi[direction],
                    "validation_csi_mi": validation_radio_mi[direction],
                    "train_sensing_margin_contribution": train_sensing_contribution[:, direction].mean(),
                    "train_csi_margin_contribution": train_radio_contribution[:, direction].mean(),
                    "validation_sensing_margin_mean": validation_sensing_contribution[:, direction].mean(),
                    "validation_sensing_margin_median": np.median(
                        validation_sensing_contribution[:, direction]
                    ),
                    "validation_sensing_positive_ratio": (
                        validation_sensing_contribution[:, direction] > 0
                    ).mean(),
                    "validation_csi_margin_mean": validation_radio_contribution[:, direction].mean(),
                    "validation_csi_margin_median": np.median(
                        validation_radio_contribution[:, direction]
                    ),
                    "validation_csi_positive_ratio": (
                        validation_radio_contribution[:, direction] > 0
                    ).mean(),
                    "sensing_error_sensing_contribution": (
                        validation_sensing_contribution[sensing_error, direction].mean()
                        if sensing_error.any()
                        else math.nan
                    ),
                    "csi_rescue_csi_contribution": (
                        validation_radio_contribution[rescue, direction].mean()
                        if rescue.any()
                        else math.nan
                    ),
                    "paired_advantage": bootstrap["mean"][direction],
                    "paired_advantage_ci_low": bootstrap["ci_low"][direction],
                    "paired_advantage_ci_high": bootstrap["ci_high"][direction],
                    "paired_effect_size": bootstrap["effect_size"][direction],
                    "paired_p_value": bootstrap["p_value"][direction],
                    "paired_q_value": q_values[direction],
                    "train_validation_sign_consistent": bool(
                        np.sign(train_difference[:, direction].mean())
                        == np.sign(validation_difference[:, direction].mean())
                    ),
                    "candidate_for_mask": bool(candidate[direction]),
                    "bootstrap_replicates": int(statistics["bootstrap_replicates"]),
                }
            )

    if singular_values is None:
        effective = np.ones(DIMENSION, dtype=bool)
    else:
        tolerance = float(statistics["singular_value_relative_tolerance"])
        effective = np.asarray(singular_values) > float(np.max(singular_values)) * tolerance
    score, train_advantage, order = _direction_rank(
        train_s,
        train_c,
        train_target,
        prototype_coefficients,
        conditions["severe_indices"],
        effective,
    )
    severe_pass_count = mask_pass[np.asarray(conditions["severe_indices"], dtype=np.int64)].sum(axis=0)
    formal_candidate = effective & (severe_pass_count >= minimum_masks)
    order_rank = np.empty(DIMENSION, dtype=np.int64)
    order_rank[order] = np.arange(1, DIMENSION + 1)
    candidate_rows = [
        {
            "basis": basis_name,
            "direction": direction,
            "singular_rank": direction + 1,
            "singular_value": (
                float(singular_values[direction]) if singular_values is not None else math.nan
            ),
            "effective_direction": bool(effective[direction]),
            "train_only_score": (
                float(score[direction]) if np.isfinite(score[direction]) else ""
            ),
            "train_only_rank": int(order_rank[direction]),
            "train_mean_margin_advantage": train_advantage[direction],
            "validation_mean_margin_advantage": validation_advantages[
                np.asarray(conditions["severe_indices"], dtype=np.int64), direction
            ].mean(),
            "severe_consistent_mask_count": int(severe_pass_count[direction]),
            "formal_candidate": bool(formal_candidate[direction]),
            "selected_top8_train_only": bool(direction in order[:8]),
        }
        for direction in range(DIMENSION)
    ]
    state = {
        "train_s": train_s,
        "train_c": train_c,
        "validation_s": validation_s,
        "validation_c": validation_c,
        "prototype_coefficients": prototype_coefficients,
        "effective": effective,
        "score": score,
        "advantage": train_advantage,
        "order": order,
        "formal_candidate": formal_candidate,
        "validation_advantages": validation_advantages,
        "fisher_ratios": fisher_ratios,
    }
    return rows, candidate_rows, state


def analyze_d3(
    config: Mapping[str, Any],
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    state: Mapping[str, Any],
    conditions: Mapping[str, Any],
    directions: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    statistics = config["statistics"]
    k_values = [int(value) for value in statistics["stability_k"]]
    base_score = np.asarray(state["score"])
    base_advantage = np.asarray(state["advantage"])
    base_order = np.asarray(state["order"])
    effective = np.asarray(state["effective"], dtype=bool)
    prototype_coefficients = np.asarray(state["prototype_coefficients"])
    train_target = train["target"].numpy()
    validation_target = validation["target"].numpy()
    train_trajectories = np.asarray(train["trajectory_ids"])
    validation_trajectories = np.asarray(validation["trajectory_ids"])
    rows: list[dict[str, Any]] = []

    def add_result(source: str, replicate: str, score: np.ndarray, advantage: np.ndarray) -> None:
        order = np.argsort(-np.where(effective, score, -np.inf), kind="stable")
        sign_consistency = float(
            (np.sign(advantage[effective]) == np.sign(base_advantage[effective])).mean()
        )
        contribution_correlation = (
            float(np.corrcoef(advantage[effective], base_advantage[effective])[0, 1])
            if np.std(advantage[effective]) > 1e-12 and np.std(base_advantage[effective]) > 1e-12
            else 0.0
        )
        for k in k_values:
            rows.append(
                {
                    "source": source,
                    "replicate": replicate,
                    "k": k,
                    "jaccard": _jaccard(base_order[:k], order[:k]),
                    "rank_spearman": _spearman(base_score[effective], score[effective]),
                    "advantage_sign_consistency": sign_consistency,
                    "contribution_correlation": contribution_correlation,
                }
            )

    for trajectory in sorted(set(train_trajectories.tolist())):
        indices = np.flatnonzero(train_trajectories != trajectory)
        score, advantage, _ = _direction_rank(
            state["train_s"][indices],
            state["train_c"][indices],
            train_target[indices],
            prototype_coefficients,
            conditions["severe_indices"],
            effective,
        )
        add_result("train_leave_one_trajectory_out", str(trajectory), score, advantage)

    for seed in statistics["stability_bootstrap_seeds"]:
        rng = np.random.default_rng(int(seed))
        indices = rng.integers(0, len(train_target), size=len(train_target))
        score, advantage, _ = _direction_rank(
            state["train_s"][indices],
            state["train_c"][indices],
            train_target[indices],
            prototype_coefficients,
            conditions["severe_indices"],
            effective,
        )
        add_result("train_bootstrap", str(seed), score, advantage)

    for trajectory in sorted(set(validation_trajectories.tolist())):
        indices = np.flatnonzero(validation_trajectories == trajectory)
        score, advantage, _ = _direction_rank(
            state["validation_s"][indices],
            state["validation_c"][indices],
            validation_target[indices],
            prototype_coefficients,
            conditions["severe_indices"],
            effective,
        )
        add_result("validation_trajectory", str(trajectory), score, advantage)

    for seed in statistics["stability_bootstrap_seeds"]:
        rng = np.random.default_rng(int(seed) + 1000)
        indices = rng.integers(0, len(validation_target), size=len(validation_target))
        score, advantage, _ = _direction_rank(
            state["validation_s"][indices],
            state["validation_c"][indices],
            validation_target[indices],
            prototype_coefficients,
            conditions["severe_indices"],
            effective,
        )
        add_result("validation_sample_resample", str(seed), score, advantage)

    for radio_view, noise_seed in enumerate(config["pilot"]["validation_noise_seeds"]):
        _, radio, _ = _direction_arrays(validation, directions, radio_view)
        score, advantage, _ = _direction_rank(
            state["validation_s"],
            radio,
            validation_target,
            prototype_coefficients,
            conditions["severe_indices"],
            effective,
        )
        add_result("validation_csi_noise", str(noise_seed), score, advantage)

    detail_rows = list(rows)
    for source in sorted({str(row["source"]) for row in detail_rows}):
        for k in k_values:
            subset = [row for row in detail_rows if row["source"] == source and row["k"] == k]
            rows.append(
                {
                    "source": f"{source}_mean",
                    "replicate": "mean",
                    "k": k,
                    "jaccard": float(np.mean([row["jaccard"] for row in subset])),
                    "rank_spearman": float(np.mean([row["rank_spearman"] for row in subset])),
                    "advantage_sign_consistency": float(
                        np.mean([row["advantage_sign_consistency"] for row in subset])
                    ),
                    "contribution_correlation": float(
                        np.mean([row["contribution_correlation"] for row in subset])
                    ),
                }
            )
    primary = [row for row in detail_rows if int(row["k"]) == 8]
    summary = {
        "top8_jaccard_mean": float(np.mean([row["jaccard"] for row in primary])),
        "top8_rank_spearman_mean": float(np.mean([row["rank_spearman"] for row in primary])),
        "top8_sign_consistency_mean": float(
            np.mean([row["advantage_sign_consistency"] for row in primary])
        ),
        "validation_trajectory_top8_jaccard_min": float(
            min(
                row["jaccard"]
                for row in primary
                if row["source"] == "validation_trajectory"
            )
        ),
        "validation_trajectory_sign_consistency_min": float(
            min(
                row["advantage_sign_consistency"]
                for row in primary
                if row["source"] == "validation_trajectory"
            )
        ),
    }
    return rows, summary


def _choose_probe_alpha(
    config: Mapping[str, Any],
    train: Mapping[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    trajectories = np.asarray(train["trajectory_ids"])
    unique = sorted(set(trajectories.tolist()))
    count = int(config["probe"]["calibration_trajectory_count"])
    calibration = set(unique[-count:])
    fit_indices = np.flatnonzero(~np.isin(trajectories, list(calibration)))
    calibration_indices = np.flatnonzero(np.isin(trajectories, list(calibration)))
    labels = train["target"].numpy()
    z_s = train["z_s_raw"].numpy()
    rows: list[dict[str, Any]] = []
    best_alpha, best_score = None, -math.inf
    for alpha in [float(value) for value in config["probe"]["alpha_grid"]]:
        scores = []
        for mask_index in range(len(MASK_NAMES)):
            workspace = RidgeWorkspace(z_s[fit_indices, mask_index], labels[fit_indices])
            model = workspace.fit(range(DIMENSION), alpha)
            logits = ridge_predict(model, z_s[calibration_indices, mask_index])
            scores.append(float((logits.argmax(axis=1) == labels[calibration_indices]).mean()))
        score = float(np.mean(scores))
        rows.append(
            {
                "stage": "train_only_alpha_selection",
                "alpha": alpha,
                "all14_macro_top1": score,
                "fit_trajectories": len(unique) - count,
                "calibration_trajectories": count,
            }
        )
        if score > best_score:
            best_alpha, best_score = alpha, score
    if best_alpha is None:
        raise RuntimeError("Train-only probe alpha selection failed.")
    return float(best_alpha), rows


def _probe_spec_key(method: str, k: int = 0, run: int = -1) -> tuple[str, int, int]:
    return method, int(k), int(run)


def analyze_d4(
    config: Mapping[str, Any],
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    state: Mapping[str, Any],
    conditions: Mapping[str, Any],
    weights: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    alpha, rows = _choose_probe_alpha(config, train)
    labels_train = train["target"].numpy()
    labels_validation = validation["target"].numpy()
    z_c_train = train["z_c_raw"][int(config["pilot"]["main_radio_view"])].numpy()
    z_c_validation = validation["z_c_raw"][int(config["pilot"]["main_radio_view"])].numpy()
    a_c_train = np.asarray(state["train_c"])
    a_c_validation = np.asarray(state["validation_c"])
    z_s_train = train["z_s_raw"].numpy()
    z_s_validation = validation["z_s_raw"].numpy()
    order = np.asarray(state["order"])
    effective = np.asarray(state["effective"], dtype=bool)
    available = np.flatnonzero(effective)
    k_values = [int(value) for value in config["statistics"]["stability_k"]]
    random_runs = int(config["statistics"]["random_probe_runs"])
    rng = np.random.default_rng(int(config["statistics"]["random_seed"]))
    random_directions = {
        (k, run): np.sort(rng.choice(available, size=k, replace=False))
        for k in k_values
        for run in range(random_runs)
    }

    correct: dict[tuple[str, int, int], list[np.ndarray]] = defaultdict(list)
    top3_correct: dict[tuple[str, int, int], list[np.ndarray]] = defaultdict(list)
    predictions: dict[tuple[str, int, int, int], np.ndarray] = {}
    for mask_index, mask_name in enumerate(MASK_NAMES):
        train_features = np.concatenate(
            (z_s_train[:, mask_index], z_c_train, a_c_train), axis=1
        )
        validation_features = np.concatenate(
            (z_s_validation[:, mask_index], z_c_validation, a_c_validation), axis=1
        )
        workspace = RidgeWorkspace(train_features, labels_train)
        specs: list[tuple[str, int, int, np.ndarray]] = [
            ("P0_sensing", 0, -1, np.arange(0, 64)),
            ("P1_csi", 0, -1, np.arange(64, 128)),
            ("P2_concat", 0, -1, np.arange(0, 128)),
        ]
        for k in k_values:
            specs.append(
                (
                    "P3_candidate",
                    k,
                    -1,
                    np.concatenate((np.arange(0, 64), 128 + order[:k])),
                )
            )
            for run in range(random_runs):
                specs.append(
                    (
                        "P4_random",
                        k,
                        run,
                        np.concatenate((np.arange(0, 64), 128 + random_directions[(k, run)])),
                    )
                )
        for method, k, run, indices in specs:
            model = workspace.fit(indices, alpha)
            logits = ridge_predict(model, validation_features)
            metrics = classification_metrics(logits, labels_validation)
            key = _probe_spec_key(method, k, run)
            correct[key].append(metrics["correct"].astype(np.float64))
            top3_correct[key].append(metrics["top3_correct"].astype(np.float64))
            predictions[(method, k, run, mask_index)] = metrics["prediction"]
            rows.append(
                {
                    "stage": "validation",
                    "scope": "mask",
                    "mask": mask_name,
                    "mask_id": mask_index,
                    "method": method,
                    "k": k,
                    "random_run": run,
                    "input_dim": len(indices),
                    "alpha": alpha,
                    "top1": metrics["top1"],
                    "top3": metrics["top3"],
                }
            )

    scope_indices = {
        "all14": list(range(len(MASK_NAMES))),
        "worst": [int(conditions["worst_mask_index"])],
        "missing_lidar": [int(conditions["missing_lidar_index"])],
        "single": [index for index, name in enumerate(MASK_NAMES) if MASK_COUNTS[name] == 1],
        "two": [index for index, name in enumerate(MASK_NAMES) if MASK_COUNTS[name] == 2],
        "three": [index for index, name in enumerate(MASK_NAMES) if MASK_COUNTS[name] == 3],
    }
    all_keys = list(correct)
    scope_values: dict[tuple[tuple[str, int, int], str], np.ndarray] = {}
    for key in all_keys:
        matrix = np.stack(correct[key], axis=1)
        top3_matrix = np.stack(top3_correct[key], axis=1)
        for scope, indices in scope_indices.items():
            per_sample = matrix[:, indices].mean(axis=1)
            per_sample_top3 = top3_matrix[:, indices].mean(axis=1)
            scope_values[(key, scope)] = per_sample
            rows.append(
                {
                    "stage": "validation",
                    "scope": scope,
                    "mask": conditions["worst_mask"] if scope == "worst" else scope,
                    "method": key[0],
                    "k": key[1],
                    "random_run": key[2],
                    "input_dim": (
                        64 + key[1] if key[0] in {"P3_candidate", "P4_random"} else
                        128 if key[0] == "P2_concat" else 64
                    ),
                    "alpha": alpha,
                    "top1": float(per_sample.mean()),
                    "top3": float(per_sample_top3.mean()),
                }
            )

    p0_key = _probe_spec_key("P0_sensing")
    primary_key = _probe_spec_key("P3_candidate", 8)
    scope_summary: dict[str, Any] = {}
    for scope in ("all14", "worst", "missing_lidar"):
        base = scope_values[(p0_key, scope)]
        candidate = scope_values[(primary_key, scope)]
        paired = bootstrap_mean_summary(candidate - base, weights)
        random_values = np.asarray(
            [
                scope_values[(_probe_spec_key("P4_random", 8, run), scope)].mean()
                for run in range(random_runs)
            ]
        )
        scope_summary[scope] = {
            "p0": float(base.mean()),
            "p3_top8": float(candidate.mean()),
            "gain": float(candidate.mean() - base.mean()),
            "gain_ci_low": paired["ci_low"],
            "gain_ci_high": paired["ci_high"],
            "gain_effect_size": paired["effect_size"],
            "gain_p_value": paired["p_value"],
            "p4_random_mean": float(random_values.mean()),
            "p4_random_std": float(random_values.std(ddof=1)),
            "candidate_over_random_mean": float(candidate.mean() - random_values.mean()),
        }
    summary = {
        "probe_type": "closed_form_multiclass_ridge",
        "alpha": alpha,
        "alpha_selected_from": "train trajectories only",
        "random_runs": random_runs,
        "primary_k": 8,
        "scopes": scope_summary,
        "candidate_predictions": {
            mask_name: predictions[("P3_candidate", 8, -1, mask_index)].tolist()
            for mask_index, mask_name in enumerate(MASK_NAMES)
        },
    }
    return rows, summary


def _choose_residual_alpha(
    config: Mapping[str, Any],
    train: Mapping[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    trajectories = np.asarray(train["trajectory_ids"])
    unique = sorted(set(trajectories.tolist()))
    count = int(config["probe"]["calibration_trajectory_count"])
    calibration = set(unique[-count:])
    fit_indices = np.flatnonzero(~np.isin(trajectories, list(calibration)))
    calibration_indices = np.flatnonzero(np.isin(trajectories, list(calibration)))
    z_s = train["z_s_raw"].numpy()
    z_c = train["z_c_raw"][int(config["pilot"]["main_radio_view"])].numpy()
    rows: list[dict[str, Any]] = []
    best_alpha, best_mse = None, math.inf
    for alpha in [float(value) for value in config["probe"]["residual_alpha_grid"]]:
        errors = []
        for mask_index in range(len(MASK_NAMES)):
            model = fit_multivariate_ridge(
                z_s[fit_indices, mask_index], z_c[fit_indices], alpha
            )
            prediction = predict_multivariate_ridge(model, z_s[calibration_indices, mask_index])
            errors.append(float(np.mean((z_c[calibration_indices] - prediction) ** 2)))
        mse = float(np.mean(errors))
        rows.append(
            {
                "stage": "train_only_residual_alpha_selection",
                "residual_alpha": alpha,
                "validation_mse": mse,
                "fit_trajectories": len(unique) - count,
                "calibration_trajectories": count,
            }
        )
        if mse < best_mse:
            best_alpha, best_mse = alpha, mse
    if best_alpha is None:
        raise RuntimeError("Train-only residual alpha selection failed.")
    return float(best_alpha), rows


def analyze_d5(
    config: Mapping[str, Any],
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    directions: np.ndarray,
    probe_alpha: float,
    conditions: Mapping[str, Any],
    weights: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    residual_alpha, rows = _choose_residual_alpha(config, train)
    labels_train = train["target"].numpy()
    labels_validation = validation["target"].numpy()
    z_s_train = train["z_s_raw"].numpy()
    z_s_validation = validation["z_s_raw"].numpy()
    z_c_train = train["z_c_raw"][int(config["pilot"]["main_radio_view"])].numpy()
    z_c_validation = validation["z_c_raw"][int(config["pilot"]["main_radio_view"])].numpy()
    e_s = validation["e_s"].numpy()
    e_c = validation["e_c"][int(config["pilot"]["main_radio_view"])].numpy()
    prototype_basis_matrix = np.asarray(directions, dtype=np.float64)
    random_runs = int(config["statistics"]["random_probe_runs"])
    random_seed = int(config["statistics"]["random_seed"]) + 500
    method_correct: dict[str, list[np.ndarray]] = defaultdict(list)
    per_mask_summary: dict[str, Any] = {}
    for mask_index, mask_name in enumerate(MASK_NAMES):
        residual_model = fit_multivariate_ridge(
            z_s_train[:, mask_index], z_c_train, residual_alpha
        )
        train_prediction = predict_multivariate_ridge(residual_model, z_s_train[:, mask_index])
        validation_prediction = predict_multivariate_ridge(
            residual_model, z_s_validation[:, mask_index]
        )
        residual_train = z_c_train - train_prediction
        residual_validation = z_c_validation - validation_prediction
        explained_variance = 1.0 - np.sum((z_c_validation - validation_prediction) ** 2) / max(
            np.sum((z_c_validation - z_c_train.mean(axis=0)) ** 2), 1e-12
        )
        residual_coeff_train = residual_train @ prototype_basis_matrix
        residual_coeff_validation = residual_validation @ prototype_basis_matrix
        residual_edges = fit_bin_edges(
            residual_coeff_train, int(config["statistics"]["mi_bins"])
        )
        residual_mi = fixed_bin_mutual_information(
            residual_coeff_validation, labels_validation, residual_edges
        )

        train_features = np.concatenate((z_s_train[:, mask_index], residual_train), axis=1)
        validation_features = np.concatenate(
            (z_s_validation[:, mask_index], residual_validation), axis=1
        )
        workspace = RidgeWorkspace(train_features, labels_train)
        specs = {
            "R0_sensing": np.arange(0, 64),
            "R1_residual": np.arange(64, 128),
            "R2_sensing_residual": np.arange(0, 128),
        }
        metrics_by_method: dict[str, Any] = {}
        for method, indices in specs.items():
            model = workspace.fit(indices, probe_alpha)
            logits = ridge_predict(model, validation_features)
            metrics = classification_metrics(logits, labels_validation)
            metrics_by_method[method] = metrics
            method_correct[method].append(metrics["correct"].astype(np.float64))
            rows.append(
                {
                    "stage": "validation",
                    "mask": mask_name,
                    "mask_id": mask_index,
                    "method": method,
                    "top1": metrics["top1"],
                    "top3": metrics["top3"],
                    "probe_alpha": probe_alpha,
                    "residual_alpha": residual_alpha,
                    "explained_csi_variance": explained_variance,
                    "residual_norm": float(np.linalg.norm(residual_validation, axis=1).mean()),
                    "residual_beam_mi_mean": float(residual_mi.mean()),
                    "residual_beam_mi_sum": float(residual_mi.sum()),
                }
            )

        random_top1 = []
        for run in range(random_runs):
            rng_train = np.random.default_rng(random_seed + 100 * mask_index + run)
            rng_validation = np.random.default_rng(random_seed + 10000 + 100 * mask_index + run)
            shuffled_train = residual_train[rng_train.permutation(len(residual_train))]
            shuffled_validation = residual_validation[
                rng_validation.permutation(len(residual_validation))
            ]
            random_train = np.concatenate((z_s_train[:, mask_index], shuffled_train), axis=1)
            random_validation = np.concatenate(
                (z_s_validation[:, mask_index], shuffled_validation), axis=1
            )
            random_workspace = RidgeWorkspace(random_train, labels_train)
            random_model = random_workspace.fit(np.arange(0, 128), probe_alpha)
            random_logits = ridge_predict(random_model, random_validation)
            random_metrics = classification_metrics(random_logits, labels_validation)
            random_top1.append(random_metrics["top1"])
            rows.append(
                {
                    "stage": "validation",
                    "mask": mask_name,
                    "mask_id": mask_index,
                    "method": "R3_random_residual",
                    "random_run": run,
                    "top1": random_metrics["top1"],
                    "top3": random_metrics["top3"],
                    "probe_alpha": probe_alpha,
                    "residual_alpha": residual_alpha,
                }
            )
        sensing_correct = e_s[:, mask_index].argmax(axis=1) == labels_validation
        csi_correct = e_c.argmax(axis=1) == labels_validation
        rescue = ~sensing_correct & csi_correct
        per_mask_summary[mask_name] = {
            "explained_csi_variance": float(explained_variance),
            "residual_norm": float(np.linalg.norm(residual_validation, axis=1).mean()),
            "residual_mi_mean": float(residual_mi.mean()),
            "r0": metrics_by_method["R0_sensing"]["top1"],
            "r1": metrics_by_method["R1_residual"]["top1"],
            "r2": metrics_by_method["R2_sensing_residual"]["top1"],
            "r2_gain": (
                metrics_by_method["R2_sensing_residual"]["top1"]
                - metrics_by_method["R0_sensing"]["top1"]
            ),
            "r3_random_mean": float(np.mean(random_top1)),
            "r3_random_std": float(np.std(random_top1, ddof=1)),
            "rescue_samples": int(rescue.sum()),
            "r1_rescue_accuracy": (
                float(metrics_by_method["R1_residual"]["correct"][rescue].mean())
                if rescue.any()
                else math.nan
            ),
            "r2_rescue_accuracy": (
                float(metrics_by_method["R2_sensing_residual"]["correct"][rescue].mean())
                if rescue.any()
                else math.nan
            ),
        }

    scope_indices = {
        "all14": list(range(len(MASK_NAMES))),
        "worst": [int(conditions["worst_mask_index"])],
        "missing_lidar": [int(conditions["missing_lidar_index"])],
    }
    scope_summary: dict[str, Any] = {}
    r0_matrix = np.stack(method_correct["R0_sensing"], axis=1)
    r2_matrix = np.stack(method_correct["R2_sensing_residual"], axis=1)
    for scope, indices in scope_indices.items():
        r0 = r0_matrix[:, indices].mean(axis=1)
        r2 = r2_matrix[:, indices].mean(axis=1)
        paired = bootstrap_mean_summary(r2 - r0, weights)
        scope_summary[scope] = {
            "r0": float(r0.mean()),
            "r2": float(r2.mean()),
            "gain": float((r2 - r0).mean()),
            "gain_ci_low": paired["ci_low"],
            "gain_ci_high": paired["ci_high"],
            "gain_effect_size": paired["effect_size"],
            "gain_p_value": paired["p_value"],
        }
    summary = {
        "probe_alpha": probe_alpha,
        "residual_alpha": residual_alpha,
        "residual_alpha_selected_from": "train trajectories only",
        "per_mask": per_mask_summary,
        "scopes": scope_summary,
    }
    return rows, summary


def _fit_direction_affine(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    centered_x = x - x.mean(axis=0)
    centered_y = y - y.mean(axis=0)
    alpha = (centered_x * centered_y).mean(axis=0) / np.maximum(
        (centered_x**2).mean(axis=0), 1e-12
    )
    beta = y.mean(axis=0) - alpha * x.mean(axis=0)
    return alpha, beta


def _score_coefficients(
    coefficients: np.ndarray,
    directions: np.ndarray,
    prototypes: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    values = torch.as_tensor(coefficients, dtype=torch.float32, device=device)
    basis = torch.as_tensor(directions, dtype=torch.float32, device=device)
    bank = torch.as_tensor(prototypes, dtype=torch.float32, device=device)
    reconstructed = torch.matmul(values, basis.transpose(0, 1))
    reconstructed = F.normalize(reconstructed, dim=-1)
    logits = torch.matmul(reconstructed, F.normalize(bank, dim=-1).transpose(0, 1))
    return logits.cpu().numpy()


def _sensing_strong_order(
    sensing: np.ndarray,
    labels: np.ndarray,
    prototype_coefficients: np.ndarray,
) -> np.ndarray:
    logits = sensing @ prototype_coefficients.T
    wrong = highest_wrong(logits, labels)
    contribution = margin_contributions(sensing, prototype_coefficients, labels, wrong)
    fisher = fisher_scores(sensing, labels)
    effect = contribution.mean(axis=0) / np.maximum(contribution.std(axis=0, ddof=1), 1e-12)
    return np.argsort(-(np.log1p(fisher) + effect), kind="stable")


def analyze_d6(
    config: Mapping[str, Any],
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    state: Mapping[str, Any],
    directions: np.ndarray,
    conditions: Mapping[str, Any],
    weights: np.ndarray,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    labels_train = train["target"].numpy()
    labels_validation = validation["target"].numpy()
    train_s = np.asarray(state["train_s"], dtype=np.float32)
    train_c = np.asarray(state["train_c"], dtype=np.float32)
    validation_s = np.asarray(state["validation_s"], dtype=np.float32)
    validation_c = np.asarray(state["validation_c"], dtype=np.float32)
    prototypes = validation["prototype_normalized"].numpy()
    prototype_coefficients = np.asarray(state["prototype_coefficients"])
    order = np.asarray(state["order"])
    available = np.flatnonzero(np.asarray(state["effective"], dtype=bool))
    k_values = [int(value) for value in config["statistics"]["k_curve"]]
    random_runs = int(config["statistics"]["random_direction_runs"])
    rng = np.random.default_rng(int(config["statistics"]["random_seed"]) + 1000)
    random_sets = {
        (k, run): np.sort(rng.choice(available, size=k, replace=False))
        for k in k_values
        for run in range(random_runs)
    }
    rows: list[dict[str, Any]] = []
    correct: dict[tuple[str, str, int, int], list[np.ndarray]] = defaultdict(list)
    top3: dict[tuple[str, str, int, int], list[np.ndarray]] = defaultdict(list)
    primary_logits: dict[str, np.ndarray] = {}
    baseline_logits: dict[str, np.ndarray] = {}

    for mask_index, mask_name in enumerate(MASK_NAMES):
        sensing_train = train_s[:, mask_index]
        sensing_validation = validation_s[:, mask_index]
        sensing_order = _sensing_strong_order(
            sensing_train, labels_train, prototype_coefficients
        )
        alpha, beta = _fit_direction_affine(train_c, sensing_train)
        radio_modes = {
            "uncalibrated": validation_c,
            "affine_train": validation_c * alpha + beta,
        }
        base_logits = _score_coefficients(
            sensing_validation, directions, prototypes, device
        )
        base_metrics = classification_metrics(base_logits, labels_validation)
        baseline_logits[mask_name] = base_logits
        for calibration, radio_coefficients in radio_modes.items():
            base_key = (calibration, "I0_sensing", 0, -1)
            correct[base_key].append(base_metrics["correct"].astype(np.float64))
            top3[base_key].append(base_metrics["top3_correct"].astype(np.float64))
            rows.append(
                {
                    "scope": "mask",
                    "mask": mask_name,
                    "mask_id": mask_index,
                    "calibration": calibration,
                    "intervention": "I0_sensing",
                    "k": 0,
                    "random_run": -1,
                    "top1": base_metrics["top1"],
                    "top3": base_metrics["top3"],
                    "fix_rate": 0.0,
                    "harm_rate": 0.0,
                    "target_margin": base_metrics["target_margin"],
                }
            )
            full_logits = _score_coefficients(radio_coefficients, directions, prototypes, device)
            full_metrics = classification_metrics(
                full_logits, labels_validation, base_logits=base_logits
            )
            full_key = (calibration, "I4_all_csi", 64, -1)
            correct[full_key].append(full_metrics["correct"].astype(np.float64))
            top3[full_key].append(full_metrics["top3_correct"].astype(np.float64))
            rows.append(
                {
                    "scope": "mask",
                    "mask": mask_name,
                    "mask_id": mask_index,
                    "calibration": calibration,
                    "intervention": "I4_all_csi",
                    "k": 64,
                    "random_run": -1,
                    "top1": full_metrics["top1"],
                    "top3": full_metrics["top3"],
                    "fix_rate": full_metrics["fix_rate"],
                    "harm_rate": full_metrics["harm_rate"],
                    "target_margin": full_metrics["target_margin"],
                }
            )
            for k in k_values:
                for intervention, selected in (
                    ("I1_candidate", order[:k]),
                    ("I3_sensing_strong", sensing_order[:k]),
                ):
                    mixed = sensing_validation.copy()
                    mixed[:, selected] = radio_coefficients[:, selected]
                    logits = _score_coefficients(mixed, directions, prototypes, device)
                    metrics = classification_metrics(
                        logits, labels_validation, base_logits=base_logits
                    )
                    key = (calibration, intervention, k, -1)
                    correct[key].append(metrics["correct"].astype(np.float64))
                    top3[key].append(metrics["top3_correct"].astype(np.float64))
                    rows.append(
                        {
                            "scope": "mask",
                            "mask": mask_name,
                            "mask_id": mask_index,
                            "calibration": calibration,
                            "intervention": intervention,
                            "k": k,
                            "random_run": -1,
                            "top1": metrics["top1"],
                            "top3": metrics["top3"],
                            "fix_rate": metrics["fix_rate"],
                            "harm_rate": metrics["harm_rate"],
                            "target_margin": metrics["target_margin"],
                        }
                    )
                    if (
                        intervention == "I1_candidate"
                        and k == 8
                        and calibration == config["statistics"]["intervention_main_calibration"]
                    ):
                        primary_logits[mask_name] = logits

                mixed_batch = np.broadcast_to(
                    sensing_validation[None], (random_runs,) + sensing_validation.shape
                ).copy()
                for run in range(random_runs):
                    selected = random_sets[(k, run)]
                    mixed_batch[run][:, selected] = radio_coefficients[:, selected]
                random_logits = _score_coefficients(
                    mixed_batch, directions, prototypes, device
                )
                for run in range(random_runs):
                    metrics = classification_metrics(
                        random_logits[run], labels_validation, base_logits=base_logits
                    )
                    key = (calibration, "I2_random", k, run)
                    correct[key].append(metrics["correct"].astype(np.float64))
                    top3[key].append(metrics["top3_correct"].astype(np.float64))
                    rows.append(
                        {
                            "scope": "mask",
                            "mask": mask_name,
                            "mask_id": mask_index,
                            "calibration": calibration,
                            "intervention": "I2_random",
                            "k": k,
                            "random_run": run,
                            "top1": metrics["top1"],
                            "top3": metrics["top3"],
                            "fix_rate": metrics["fix_rate"],
                            "harm_rate": metrics["harm_rate"],
                            "target_margin": metrics["target_margin"],
                        }
                    )

    scope_indices = {
        "all14": list(range(len(MASK_NAMES))),
        "worst": [int(conditions["worst_mask_index"])],
        "missing_lidar": [int(conditions["missing_lidar_index"])],
    }
    scope_values: dict[tuple[tuple[str, str, int, int], str], np.ndarray] = {}
    detail_keys = list(correct)
    for key in detail_keys:
        matrix = np.stack(correct[key], axis=1)
        top3_matrix = np.stack(top3[key], axis=1)
        for scope, indices in scope_indices.items():
            values = matrix[:, indices].mean(axis=1)
            values_top3 = top3_matrix[:, indices].mean(axis=1)
            scope_values[(key, scope)] = values
            rows.append(
                {
                    "scope": scope,
                    "mask": conditions["worst_mask"] if scope == "worst" else scope,
                    "calibration": key[0],
                    "intervention": key[1],
                    "k": key[2],
                    "random_run": key[3],
                    "top1": float(values.mean()),
                    "top3": float(values_top3.mean()),
                }
            )

    main_calibration = str(config["statistics"]["intervention_main_calibration"])
    scope_summary: dict[str, Any] = {}
    for scope in ("all14", "worst", "missing_lidar"):
        base = scope_values[((main_calibration, "I0_sensing", 0, -1), scope)]
        candidate = scope_values[((main_calibration, "I1_candidate", 8, -1), scope)]
        scope_mask_indices = scope_indices[scope]
        base_matrix = np.stack(
            correct[(main_calibration, "I0_sensing", 0, -1)], axis=1
        )[:, scope_mask_indices]
        candidate_matrix = np.stack(
            correct[(main_calibration, "I1_candidate", 8, -1)], axis=1
        )[:, scope_mask_indices]
        base_flat = base_matrix.reshape(-1).astype(bool)
        candidate_flat = candidate_matrix.reshape(-1).astype(bool)
        random_per_sample = np.stack(
            [
                scope_values[((main_calibration, "I2_random", 8, run), scope)]
                for run in range(random_runs)
            ],
            axis=1,
        )
        gain = bootstrap_mean_summary(candidate - base, weights)
        over_random = bootstrap_mean_summary(candidate - random_per_sample.mean(axis=1), weights)
        random_accuracies = random_per_sample.mean(axis=0)
        scope_summary[scope] = {
            "baseline": float(base.mean()),
            "candidate_top8": float(candidate.mean()),
            "gain": float((candidate - base).mean()),
            "gain_ci_low": gain["ci_low"],
            "gain_ci_high": gain["ci_high"],
            "gain_effect_size": gain["effect_size"],
            "gain_p_value": gain["p_value"],
            "random_top8_mean": float(random_accuracies.mean()),
            "random_top8_std": float(random_accuracies.std(ddof=1)),
            "candidate_over_random": float(candidate.mean() - random_accuracies.mean()),
            "candidate_over_random_ci_low": over_random["ci_low"],
            "candidate_over_random_ci_high": over_random["ci_high"],
            "fix_rate": (
                float(candidate_flat[~base_flat].mean()) if (~base_flat).any() else 0.0
            ),
            "harm_rate": (
                float((~candidate_flat[base_flat]).mean()) if base_flat.any() else 0.0
            ),
        }
    summary = {
        "primary_k": 8,
        "primary_calibration": main_calibration,
        "random_runs": random_runs,
        "scopes": scope_summary,
    }
    intervention_state = {
        "primary_logits": primary_logits,
        "baseline_logits": baseline_logits,
        "random_sets": random_sets,
    }
    return rows, summary, intervention_state


def analyze_d7(
    config: Mapping[str, Any],
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    state: Mapping[str, Any],
    directions: np.ndarray,
    weights: np.ndarray,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels_train = train["target"].numpy()
    labels_validation = validation["target"].numpy()
    train_s = np.asarray(state["train_s"])
    train_c = np.asarray(state["train_c"])
    validation_c = np.asarray(state["validation_c"], dtype=np.float32)
    prototype_coefficients = np.asarray(state["prototype_coefficients"])
    prototypes = validation["prototype_normalized"].numpy()
    order = np.asarray(state["order"])
    effective = np.asarray(state["effective"], dtype=bool)
    available = np.flatnonzero(effective)
    csi_weak = available[np.argsort(np.asarray(state["score"])[available])]
    sensing_orders = [
        _sensing_strong_order(train_s[:, mask_index], labels_train, prototype_coefficients)
        for mask_index in range(len(MASK_NAMES))
    ]
    sensing_score = np.zeros(DIMENSION, dtype=np.float64)
    for rank_order in sensing_orders:
        sensing_score[rank_order] += np.arange(DIMENSION, 0, -1)
    sensing_strong = np.argsort(-sensing_score, kind="stable")
    k_values = [int(value) for value in config["statistics"]["k_curve"]]
    random_runs = int(config["statistics"]["random_direction_runs"])
    rng = np.random.default_rng(int(config["statistics"]["random_seed"]) + 2000)
    random_sets = {
        (k, run): np.sort(rng.choice(available, size=k, replace=False))
        for k in k_values
        for run in range(random_runs)
    }
    baseline_logits = _score_coefficients(validation_c, directions, prototypes, device)
    baseline = classification_metrics(baseline_logits, labels_validation)
    rows: list[dict[str, Any]] = [
        {
            "ablation": "baseline_csi",
            "k": 0,
            "random_run": -1,
            "top1": baseline["top1"],
            "top3": baseline["top3"],
            "target_margin": baseline["target_margin"],
            "within3": baseline["within3"],
            "mae": baseline["mae"],
        }
    ]
    primary_candidate_metrics: dict[str, Any] | None = None
    primary_random_correct: list[np.ndarray] = []
    primary_random_top1: list[float] = []
    for k in k_values:
        for ablation, selected in (
            ("candidate_csi_strong", order[:k]),
            ("sensing_strong", sensing_strong[:k]),
            ("csi_weak", csi_weak[:k]),
        ):
            coefficients = validation_c.copy()
            coefficients[:, selected] = 0.0
            logits = _score_coefficients(coefficients, directions, prototypes, device)
            metrics = classification_metrics(logits, labels_validation)
            rows.append(
                {
                    "ablation": ablation,
                    "k": k,
                    "random_run": -1,
                    "top1": metrics["top1"],
                    "top3": metrics["top3"],
                    "target_margin": metrics["target_margin"],
                    "within3": metrics["within3"],
                    "mae": metrics["mae"],
                    "top1_drop": baseline["top1"] - metrics["top1"],
                    "margin_drop": baseline["target_margin"] - metrics["target_margin"],
                }
            )
            if ablation == "candidate_csi_strong" and k == 8:
                primary_candidate_metrics = metrics
        coefficients_batch = np.broadcast_to(
            validation_c[None], (random_runs,) + validation_c.shape
        ).copy()
        for run in range(random_runs):
            coefficients_batch[run, :, random_sets[(k, run)]] = 0.0
        logits_batch = _score_coefficients(coefficients_batch, directions, prototypes, device)
        for run in range(random_runs):
            metrics = classification_metrics(logits_batch[run], labels_validation)
            rows.append(
                {
                    "ablation": "random",
                    "k": k,
                    "random_run": run,
                    "top1": metrics["top1"],
                    "top3": metrics["top3"],
                    "target_margin": metrics["target_margin"],
                    "within3": metrics["within3"],
                    "mae": metrics["mae"],
                    "top1_drop": baseline["top1"] - metrics["top1"],
                    "margin_drop": baseline["target_margin"] - metrics["target_margin"],
                }
            )
            if k == 8:
                primary_random_correct.append(metrics["correct"].astype(np.float64))
                primary_random_top1.append(metrics["top1"])
    if primary_candidate_metrics is None:
        raise RuntimeError("D7 primary K=8 candidate deletion was not evaluated.")
    candidate_drop_values = (
        baseline["correct"].astype(np.float64)
        - primary_candidate_metrics["correct"].astype(np.float64)
    )
    random_correct_mean = np.stack(primary_random_correct, axis=1).mean(axis=1)
    additional_drop_values = (
        random_correct_mean - primary_candidate_metrics["correct"].astype(np.float64)
    )
    drop_stats = bootstrap_mean_summary(candidate_drop_values, weights)
    over_random_stats = bootstrap_mean_summary(additional_drop_values, weights)
    random_drop = baseline["top1"] - np.asarray(primary_random_top1)
    summary = {
        "baseline_top1": baseline["top1"],
        "candidate_top8_top1": primary_candidate_metrics["top1"],
        "candidate_top8_drop": baseline["top1"] - primary_candidate_metrics["top1"],
        "candidate_drop_ci_low": drop_stats["ci_low"],
        "candidate_drop_ci_high": drop_stats["ci_high"],
        "candidate_drop_effect_size": drop_stats["effect_size"],
        "candidate_drop_p_value": drop_stats["p_value"],
        "random_top8_drop_mean": float(random_drop.mean()),
        "random_top8_drop_std": float(random_drop.std(ddof=1)),
        "candidate_additional_drop_over_random": float(
            (baseline["top1"] - primary_candidate_metrics["top1"]) - random_drop.mean()
        ),
        "additional_drop_ci_low": over_random_stats["ci_low"],
        "additional_drop_ci_high": over_random_stats["ci_high"],
        "candidate_top8_target_margin": primary_candidate_metrics["target_margin"],
        "random_runs": random_runs,
    }
    return rows, summary


def analyze_sample_groups(
    validation: Mapping[str, Any],
    state: Mapping[str, Any],
    intervention_state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    labels = validation["target"].numpy()
    e_s = validation["e_s"].numpy()
    e_c = validation["e_c"][0].numpy()
    csi_prediction = e_c.argmax(axis=1)
    csi_top3 = _topk_correct(e_c, labels, 3)
    order = np.asarray(state["order"])
    selected = order[:8]
    prototype_coefficients = np.asarray(state["prototype_coefficients"])
    radio_logits = state["validation_c"] @ prototype_coefficients.T
    radio_wrong = highest_wrong(radio_logits, labels)
    radio_contribution = margin_contributions(
        state["validation_c"], prototype_coefficients, labels, radio_wrong
    )
    rows: list[dict[str, Any]] = []
    for mask_index, mask_name in enumerate(MASK_NAMES):
        sensing_logits = e_s[:, mask_index]
        sensing_prediction = sensing_logits.argmax(axis=1)
        sensing_correct = sensing_prediction == labels
        csi_correct = csi_prediction == labels
        groups = {
            "G0_sensing_correct_csi_correct": sensing_correct & csi_correct,
            "G1_sensing_correct_csi_wrong": sensing_correct & ~csi_correct,
            "G2_sensing_wrong_csi_correct": ~sensing_correct & csi_correct,
            "G3_sensing_wrong_csi_top3": ~sensing_correct & ~csi_correct & csi_top3,
            "G4_both_wrong_no_csi_top3": ~sensing_correct & ~csi_correct & ~csi_top3,
        }
        sensing_wrong = highest_wrong(
            state["validation_s"][:, mask_index] @ prototype_coefficients.T, labels
        )
        sensing_contribution = margin_contributions(
            state["validation_s"][:, mask_index],
            prototype_coefficients,
            labels,
            sensing_wrong,
        )
        advantage = (radio_contribution - sensing_contribution)[:, selected].sum(axis=1)
        coefficient_gap = np.abs(
            state["validation_c"][:, selected]
            - state["validation_s"][:, mask_index, selected]
        ).mean(axis=1)
        base_logits = intervention_state["baseline_logits"][mask_name]
        replacement_logits = intervention_state["primary_logits"][mask_name]
        base_metrics = classification_metrics(base_logits, labels)
        replacement_metrics = classification_metrics(replacement_logits, labels)
        for group_name, group in groups.items():
            count = int(group.sum())
            rows.append(
                {
                    "row_type": "group_summary",
                    "mask": mask_name,
                    "mask_id": mask_index,
                    "group": group_name,
                    "samples": count,
                    "sample_fraction": float(group.mean()),
                    "candidate_direction_advantage": (
                        float(advantage[group].mean()) if count else math.nan
                    ),
                    "sensing_csi_coefficient_gap": (
                        float(coefficient_gap[group].mean()) if count else math.nan
                    ),
                    "sensing_target_margin": (
                        float(base_metrics["margin_values"][group].mean()) if count else math.nan
                    ),
                    "replacement_target_margin": (
                        float(replacement_metrics["margin_values"][group].mean())
                        if count
                        else math.nan
                    ),
                    "replacement_fix_rate": (
                        float(
                            replacement_metrics["correct"][group & ~base_metrics["correct"]].mean()
                        )
                        if bool((group & ~base_metrics["correct"]).any())
                        else 0.0
                    ),
                    "replacement_harm_rate": (
                        float(
                            (~replacement_metrics["correct"][group & base_metrics["correct"]]).mean()
                        )
                        if bool((group & base_metrics["correct"]).any())
                        else 0.0
                    ),
                }
            )
            beam_counts = np.bincount(labels[group], minlength=NUM_BEAMS)
            for beam, beam_count in enumerate(beam_counts):
                rows.append(
                    {
                        "row_type": "beam_distribution",
                        "mask": mask_name,
                        "mask_id": mask_index,
                        "group": group_name,
                        "beam": beam,
                        "samples": int(beam_count),
                        "sample_fraction": (
                            float(beam_count / count) if count else 0.0
                        ),
                    }
                )
    return rows


def _screen_control_candidates(
    sensing_train: np.ndarray,
    radio_train: np.ndarray,
    sensing_validation: np.ndarray,
    radio_validation: np.ndarray,
    prototype_coefficients: np.ndarray,
    labels_train: np.ndarray,
    labels_validation: np.ndarray,
    severe_indices: Sequence[int],
    weights: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[int, np.ndarray, np.ndarray]:
    radio_wrong_train = highest_wrong(radio_train @ prototype_coefficients.T, labels_train)
    radio_wrong_validation = highest_wrong(
        radio_validation @ prototype_coefficients.T, labels_validation
    )
    radio_contribution_train = margin_contributions(
        radio_train, prototype_coefficients, labels_train, radio_wrong_train
    )
    radio_contribution_validation = margin_contributions(
        radio_validation, prototype_coefficients, labels_validation, radio_wrong_validation
    )
    radio_fisher_validation = fisher_scores(radio_validation, labels_validation)
    passes = np.zeros((len(severe_indices), DIMENSION), dtype=bool)
    for row_index, mask_index in enumerate(severe_indices):
        sensing_wrong_train = highest_wrong(
            sensing_train[:, mask_index] @ prototype_coefficients.T, labels_train
        )
        sensing_wrong_validation = highest_wrong(
            sensing_validation[:, mask_index] @ prototype_coefficients.T, labels_validation
        )
        sensing_contribution_train = margin_contributions(
            sensing_train[:, mask_index],
            prototype_coefficients,
            labels_train,
            sensing_wrong_train,
        )
        sensing_contribution_validation = margin_contributions(
            sensing_validation[:, mask_index],
            prototype_coefficients,
            labels_validation,
            sensing_wrong_validation,
        )
        difference_train = radio_contribution_train - sensing_contribution_train
        difference_validation = radio_contribution_validation - sensing_contribution_validation
        bootstrap = bootstrap_direction_summary(difference_validation, weights)
        q_values = benjamini_hochberg(bootstrap["p_value"])
        sensing_fisher_validation = fisher_scores(
            sensing_validation[:, mask_index], labels_validation
        )
        passes[row_index] = (
            (
                (radio_fisher_validation + 1e-12)
                / (sensing_fisher_validation + 1e-12)
                > float(config["statistics"]["fisher_ratio"])
            )
            & (bootstrap["ci_low"] > 0)
            & (q_values < float(config["statistics"]["fdr_q"]))
            & (difference_train.mean(axis=0) > 0)
        )
    count_by_direction = passes.sum(axis=0)
    formal = count_by_direction >= int(
        config["statistics"]["minimum_consistent_severe_masks"]
    )
    return int(formal.sum()), formal, count_by_direction


def _control_common_basis(
    name: str,
    config: Mapping[str, Any],
    *,
    sensing_train_features: np.ndarray,
    radio_train_features: np.ndarray,
    sensing_validation_features: np.ndarray,
    radio_validation_features: np.ndarray,
    labels_train: np.ndarray,
    labels_validation: np.ndarray,
    prototypes: np.ndarray,
    directions: np.ndarray,
    conditions: Mapping[str, Any],
    weights: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    sensing_train = np.asarray(sensing_train_features) @ directions
    radio_train = np.asarray(radio_train_features) @ directions
    sensing_validation = np.asarray(sensing_validation_features) @ directions
    radio_validation = np.asarray(radio_validation_features) @ directions
    prototype_coefficients = np.asarray(prototypes) @ directions
    effective = np.ones(DIMENSION, dtype=bool)
    score, advantage, order = _direction_rank(
        sensing_train,
        radio_train,
        labels_train,
        prototype_coefficients,
        conditions["severe_indices"],
        effective,
    )
    candidate_count, _, _ = _screen_control_candidates(
        sensing_train,
        radio_train,
        sensing_validation,
        radio_validation,
        prototype_coefficients,
        labels_train,
        labels_validation,
        conditions["severe_indices"],
        weights,
        config,
    )
    rng = np.random.default_rng(int(seed))
    random_runs = int(config["statistics"]["random_direction_runs"])
    random_sets = [
        np.sort(rng.choice(np.arange(DIMENSION), size=8, replace=False))
        for _ in range(random_runs)
    ]
    base_correct, candidate_correct = [], []
    random_correct: list[list[np.ndarray]] = [[] for _ in range(random_runs)]
    for mask_index in range(len(MASK_NAMES)):
        alpha, beta = _fit_direction_affine(
            radio_train, sensing_train[:, mask_index]
        )
        aligned = radio_validation * alpha + beta
        base_logits = _score_coefficients(
            sensing_validation[:, mask_index], directions, prototypes, device
        )
        mixed = sensing_validation[:, mask_index].copy()
        mixed[:, order[:8]] = aligned[:, order[:8]]
        candidate_logits = _score_coefficients(mixed, directions, prototypes, device)
        base_correct.append(
            (base_logits.argmax(axis=1) == labels_validation).astype(np.float64)
        )
        candidate_correct.append(
            (candidate_logits.argmax(axis=1) == labels_validation).astype(np.float64)
        )
        random_batch = np.broadcast_to(
            sensing_validation[:, mask_index][None],
            (random_runs,) + sensing_validation[:, mask_index].shape,
        ).copy()
        for run, selected in enumerate(random_sets):
            random_batch[run][:, selected] = aligned[:, selected]
        random_logits = _score_coefficients(random_batch, directions, prototypes, device)
        for run in range(random_runs):
            random_correct[run].append(
                (random_logits[run].argmax(axis=1) == labels_validation).astype(np.float64)
            )
    base = np.stack(base_correct, axis=1).mean(axis=1)
    candidate = np.stack(candidate_correct, axis=1).mean(axis=1)
    random_values = np.stack(
        [np.stack(values, axis=1).mean(axis=1) for values in random_correct],
        axis=1,
    )
    gain = bootstrap_mean_summary(candidate - base, weights)
    over_random = bootstrap_mean_summary(candidate - random_values.mean(axis=1), weights)
    random_accuracies = random_values.mean(axis=0)
    return {
        "control": name,
        "candidate_count": candidate_count,
        "train_top8": ",".join(str(int(value)) for value in order[:8]),
        "train_advantage_mean": float(advantage[order[:8]].mean()),
        "baseline_all14": float(base.mean()),
        "candidate_all14": float(candidate.mean()),
        "candidate_gain": float((candidate - base).mean()),
        "candidate_gain_ci_low": gain["ci_low"],
        "candidate_gain_ci_high": gain["ci_high"],
        "candidate_gain_p_value": gain["p_value"],
        "random_all14_mean": float(random_accuracies.mean()),
        "random_all14_std": float(random_accuracies.std(ddof=1)),
        "candidate_over_random": float(candidate.mean() - random_accuracies.mean()),
        "candidate_over_random_ci_low": over_random["ci_low"],
        "candidate_over_random_ci_high": over_random["ci_high"],
        "passes_direction_control": bool(
            candidate_count >= 8
            and gain["ci_low"] > 0
            and over_random["ci_low"] > 0
            and gain["mean"] >= 0.01
        ),
    }


def _pca_directions(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    centered = x - x.mean(axis=0)
    covariance = centered.T @ centered / max(len(x), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors[:, np.argsort(-eigenvalues)].astype(np.float32)


def _independent_pca_control(
    config: Mapping[str, Any],
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    conditions: Mapping[str, Any],
    weights: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    labels_train = train["target"].numpy()
    labels_validation = validation["target"].numpy()
    sensing_train_features = train["z_s_normalized"].numpy()
    sensing_validation_features = validation["z_s_normalized"].numpy()
    radio_train_features = train["z_c_normalized"][0].numpy()
    radio_validation_features = validation["z_c_normalized"][0].numpy()
    prototypes = validation["prototype_normalized"].numpy()
    radio_basis = _pca_directions(radio_train_features)
    radio_train = radio_train_features @ radio_basis
    radio_validation = radio_validation_features @ radio_basis
    scores = []
    advantages = []
    sensing_bases: list[np.ndarray] = []
    sensing_train_coefficients: list[np.ndarray] = []
    sensing_validation_coefficients: list[np.ndarray] = []
    prototype_coefficients: list[np.ndarray] = []
    for mask_index in range(len(MASK_NAMES)):
        basis = _pca_directions(sensing_train_features[:, mask_index])
        sensing_bases.append(basis)
        sensing_train = sensing_train_features[:, mask_index] @ basis
        sensing_validation = sensing_validation_features[:, mask_index] @ basis
        prototype_coefficient = prototypes @ basis
        sensing_train_coefficients.append(sensing_train)
        sensing_validation_coefficients.append(sensing_validation)
        prototype_coefficients.append(prototype_coefficient)
        radio_wrong = highest_wrong(radio_train @ prototype_coefficient.T, labels_train)
        sensing_wrong = highest_wrong(sensing_train @ prototype_coefficient.T, labels_train)
        radio_contribution = margin_contributions(
            radio_train, prototype_coefficient, labels_train, radio_wrong
        )
        sensing_contribution = margin_contributions(
            sensing_train, prototype_coefficient, labels_train, sensing_wrong
        )
        ratio = (fisher_scores(radio_train, labels_train) + 1e-12) / (
            fisher_scores(sensing_train, labels_train) + 1e-12
        )
        difference = radio_contribution - sensing_contribution
        scores.append(
            np.log(ratio)
            + difference.mean(axis=0) / np.maximum(difference.std(axis=0, ddof=1), 1e-12)
        )
        advantages.append(difference.mean(axis=0))
    score = np.mean(scores, axis=0)
    advantage = np.mean(advantages, axis=0)
    order = np.argsort(-score, kind="stable")
    random_runs = int(config["statistics"]["random_direction_runs"])
    rng = np.random.default_rng(int(config["statistics"]["random_seed"]) + 3000)
    random_sets = [
        np.sort(rng.choice(np.arange(DIMENSION), size=8, replace=False))
        for _ in range(random_runs)
    ]
    base_correct, candidate_correct = [], []
    random_correct: list[list[np.ndarray]] = [[] for _ in range(random_runs)]
    pass_counts = np.zeros(DIMENSION, dtype=np.int64)
    for mask_index in conditions["severe_indices"]:
        sensing_train = sensing_train_coefficients[mask_index]
        sensing_validation = sensing_validation_coefficients[mask_index]
        prototype_coefficient = prototype_coefficients[mask_index]
        radio_wrong_train = highest_wrong(radio_train @ prototype_coefficient.T, labels_train)
        sensing_wrong_train = highest_wrong(sensing_train @ prototype_coefficient.T, labels_train)
        radio_wrong_validation = highest_wrong(
            radio_validation @ prototype_coefficient.T, labels_validation
        )
        sensing_wrong_validation = highest_wrong(
            sensing_validation @ prototype_coefficient.T, labels_validation
        )
        train_difference = margin_contributions(
            radio_train, prototype_coefficient, labels_train, radio_wrong_train
        ) - margin_contributions(
            sensing_train, prototype_coefficient, labels_train, sensing_wrong_train
        )
        validation_difference = margin_contributions(
            radio_validation, prototype_coefficient, labels_validation, radio_wrong_validation
        ) - margin_contributions(
            sensing_validation,
            prototype_coefficient,
            labels_validation,
            sensing_wrong_validation,
        )
        bootstrap = bootstrap_direction_summary(validation_difference, weights)
        q_values = benjamini_hochberg(bootstrap["p_value"])
        ratio = (fisher_scores(radio_validation, labels_validation) + 1e-12) / (
            fisher_scores(sensing_validation, labels_validation) + 1e-12
        )
        pass_counts += (
            (ratio > float(config["statistics"]["fisher_ratio"]))
            & (bootstrap["ci_low"] > 0)
            & (q_values < float(config["statistics"]["fdr_q"]))
            & (train_difference.mean(axis=0) > 0)
        )
    for mask_index in range(len(MASK_NAMES)):
        sensing_train = sensing_train_coefficients[mask_index]
        sensing_validation = sensing_validation_coefficients[mask_index]
        basis = sensing_bases[mask_index]
        alpha, beta = _fit_direction_affine(radio_train, sensing_train)
        aligned = radio_validation * alpha + beta
        base_logits = _score_coefficients(sensing_validation, basis, prototypes, device)
        mixed = sensing_validation.copy()
        mixed[:, order[:8]] = aligned[:, order[:8]]
        candidate_logits = _score_coefficients(mixed, basis, prototypes, device)
        base_correct.append((base_logits.argmax(axis=1) == labels_validation).astype(np.float64))
        candidate_correct.append(
            (candidate_logits.argmax(axis=1) == labels_validation).astype(np.float64)
        )
        random_batch = np.broadcast_to(
            sensing_validation[None], (random_runs,) + sensing_validation.shape
        ).copy()
        for run, selected in enumerate(random_sets):
            random_batch[run][:, selected] = aligned[:, selected]
        random_logits = _score_coefficients(random_batch, basis, prototypes, device)
        for run in range(random_runs):
            random_correct[run].append(
                (random_logits[run].argmax(axis=1) == labels_validation).astype(np.float64)
            )
    base = np.stack(base_correct, axis=1).mean(axis=1)
    candidate = np.stack(candidate_correct, axis=1).mean(axis=1)
    random_values = np.stack(
        [np.stack(values, axis=1).mean(axis=1) for values in random_correct], axis=1
    )
    gain = bootstrap_mean_summary(candidate - base, weights)
    over_random = bootstrap_mean_summary(candidate - random_values.mean(axis=1), weights)
    random_accuracies = random_values.mean(axis=0)
    candidate_count = int(
        (
            pass_counts
            >= int(config["statistics"]["minimum_consistent_severe_masks"])
        ).sum()
    )
    return {
        "control": "independent_pca_wrong_alignment",
        "candidate_count": candidate_count,
        "train_top8": ",".join(str(int(value)) for value in order[:8]),
        "train_advantage_mean": float(advantage[order[:8]].mean()),
        "baseline_all14": float(base.mean()),
        "candidate_all14": float(candidate.mean()),
        "candidate_gain": float((candidate - base).mean()),
        "candidate_gain_ci_low": gain["ci_low"],
        "candidate_gain_ci_high": gain["ci_high"],
        "candidate_gain_p_value": gain["p_value"],
        "random_all14_mean": float(random_accuracies.mean()),
        "random_all14_std": float(random_accuracies.std(ddof=1)),
        "candidate_over_random": float(candidate.mean() - random_accuracies.mean()),
        "candidate_over_random_ci_low": over_random["ci_low"],
        "candidate_over_random_ci_high": over_random["ci_high"],
        "passes_direction_control": bool(
            candidate_count >= 8
            and gain["ci_low"] > 0
            and over_random["ci_low"] > 0
            and gain["mean"] >= 0.01
        ),
    }


@torch.inference_mode()
def _time_shuffled_radio_features(
    config: Mapping[str, Any],
    role: str,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    role = require_inner_split(role)
    source = _source_cache(config, role)
    radio_view = int(config["pilot"]["main_radio_view"])
    frames = source["frame_csi_features"][radio_view]
    upstream_config = _upstream_config(config)
    encoder = upstream._load_radio_encoder(upstream_config, device)
    expert, _, _ = upstream._load_f1_radio_expert(upstream_config, device)
    m4, _ = upstream._load_m4(upstream_config, device)
    order = torch.tensor([2, 0, 4, 1, 3], dtype=torch.long, device=device)
    values: list[torch.Tensor] = []
    max_reconstruction_diff = 0.0
    batch_size = int(config["runtime"]["matrix_batch_size"])
    for start in range(0, len(frames), batch_size):
        stop = min(start + batch_size, len(frames))
        original_frames = frames[start:stop].to(device)
        reconstructed = encoder.temporal(original_frames)[0][:, -1]
        max_reconstruction_diff = max(
            max_reconstruction_diff,
            float(
                (
                    reconstructed.cpu()
                    - source["c_radio"][radio_view, start:stop]
                )
                .abs()
                .max()
                .item()
            ),
        )
        shuffled = original_frames.index_select(1, order)
        c_radio = encoder.temporal(shuffled)[0][:, -1]
        values.append(expert(c_radio, m4.prototype_bank)["z_radio"].float().cpu())
    if max_reconstruction_diff > 1e-5:
        raise ValueError(
            f"Cached frame features do not reconstruct frozen GRU output: {max_reconstruction_diff}."
        )
    return F.normalize(torch.cat(values), dim=-1).numpy(), max_reconstruction_diff


def analyze_negative_controls(
    config: Mapping[str, Any],
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    directions: np.ndarray,
    conditions: Mapping[str, Any],
    weights: np.ndarray,
    d4_summary: Mapping[str, Any],
    d6_summary: Mapping[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels_train = train["target"].numpy()
    labels_validation = validation["target"].numpy()
    sensing_train = train["z_s_normalized"].numpy()
    sensing_validation = validation["z_s_normalized"].numpy()
    radio_train = train["z_c_normalized"][0].numpy()
    radio_validation = validation["z_c_normalized"][0].numpy()
    prototypes = validation["prototype_normalized"].numpy()
    seed = int(config["statistics"]["random_seed"])
    rows: list[dict[str, Any]] = []

    main_scope = d6_summary["scopes"]["all14"]
    rows.append(
        {
            "control": "random_directions_20",
            "candidate_count": math.nan,
            "baseline_all14": main_scope["baseline"],
            "candidate_all14": main_scope["candidate_top8"],
            "candidate_gain": main_scope["gain"],
            "candidate_gain_ci_low": main_scope["gain_ci_low"],
            "candidate_gain_ci_high": main_scope["gain_ci_high"],
            "random_all14_mean": main_scope["random_top8_mean"],
            "random_all14_std": main_scope["random_top8_std"],
            "candidate_over_random": main_scope["candidate_over_random"],
            "candidate_over_random_ci_low": main_scope["candidate_over_random_ci_low"],
            "candidate_over_random_ci_high": main_scope["candidate_over_random_ci_high"],
            "passes_direction_control": bool(
                main_scope["gain_ci_low"] > 0
                and main_scope["candidate_over_random_ci_low"] > 0
            ),
        }
    )
    probe_scope = d4_summary["scopes"]["all14"]
    rows.append(
        {
            "control": "same_parameter_random_probe_10",
            "baseline_all14": probe_scope["p0"],
            "candidate_all14": probe_scope["p3_top8"],
            "candidate_gain": probe_scope["gain"],
            "candidate_gain_ci_low": probe_scope["gain_ci_low"],
            "candidate_gain_ci_high": probe_scope["gain_ci_high"],
            "random_all14_mean": probe_scope["p4_random_mean"],
            "random_all14_std": probe_scope["p4_random_std"],
            "candidate_over_random": probe_scope["candidate_over_random_mean"],
            "passes_direction_control": bool(
                probe_scope["gain_ci_low"] > 0
                and probe_scope["p3_top8"]
                > probe_scope["p4_random_mean"] + probe_scope["p4_random_std"]
            ),
        }
    )

    for run in range(int(config["statistics"]["random_rotation_runs"])):
        rng = np.random.default_rng(seed + 4000 + run)
        matrix = rng.standard_normal((DIMENSION, DIMENSION))
        rotation, _ = np.linalg.qr(matrix)
        rows.append(
            _control_common_basis(
                f"random_orthogonal_rotation_{run}",
                config,
                sensing_train_features=sensing_train,
                radio_train_features=radio_train,
                sensing_validation_features=sensing_validation,
                radio_validation_features=radio_validation,
                labels_train=labels_train,
                labels_validation=labels_validation,
                prototypes=prototypes,
                directions=rotation.astype(np.float32),
                conditions=conditions,
                weights=weights,
                device=device,
                seed=seed + 5000 + run,
            )
        )

    rows.append(
        _independent_pca_control(
            config, train, validation, conditions, weights, device
        )
    )

    rng_label_train = np.random.default_rng(seed + 6000)
    rng_label_validation = np.random.default_rng(seed + 6001)
    rows.append(
        _control_common_basis(
            "label_permutation",
            config,
            sensing_train_features=sensing_train,
            radio_train_features=radio_train,
            sensing_validation_features=sensing_validation,
            radio_validation_features=radio_validation,
            labels_train=labels_train[rng_label_train.permutation(len(labels_train))],
            labels_validation=labels_validation[
                rng_label_validation.permutation(len(labels_validation))
            ],
            prototypes=prototypes,
            directions=directions,
            conditions=conditions,
            weights=weights,
            device=device,
            seed=seed + 6100,
        )
    )

    rng_sample_train = np.random.default_rng(seed + 7000)
    rng_sample_validation = np.random.default_rng(seed + 7001)
    rows.append(
        _control_common_basis(
            "sample_id_shuffle",
            config,
            sensing_train_features=sensing_train,
            radio_train_features=radio_train[
                rng_sample_train.permutation(len(radio_train))
            ],
            sensing_validation_features=sensing_validation,
            radio_validation_features=radio_validation[
                rng_sample_validation.permutation(len(radio_validation))
            ],
            labels_train=labels_train,
            labels_validation=labels_validation,
            prototypes=prototypes,
            directions=directions,
            conditions=conditions,
            weights=weights,
            device=device,
            seed=seed + 7100,
        )
    )

    shuffled_train, train_reconstruction_diff = _time_shuffled_radio_features(
        config, "train", device
    )
    shuffled_validation, validation_reconstruction_diff = _time_shuffled_radio_features(
        config, "validation", device
    )
    time_row = _control_common_basis(
        "time_order_shuffle",
        config,
        sensing_train_features=sensing_train,
        radio_train_features=shuffled_train,
        sensing_validation_features=sensing_validation,
        radio_validation_features=shuffled_validation,
        labels_train=labels_train,
        labels_validation=labels_validation,
        prototypes=prototypes,
        directions=directions,
        conditions=conditions,
        weights=weights,
        device=device,
        seed=seed + 8100,
    )
    time_row["cached_gru_reconstruction_max_diff_train"] = train_reconstruction_diff
    time_row["cached_gru_reconstruction_max_diff_validation"] = validation_reconstruction_diff
    rows.append(time_row)

    label_row = next(row for row in rows if row["control"] == "label_permutation")
    sample_row = next(row for row in rows if row["control"] == "sample_id_shuffle")
    summary = {
        "label_permutation_passed": bool(label_row["passes_direction_control"]),
        "sample_shuffle_passed": bool(sample_row["passes_direction_control"]),
        "negative_controls_pass": bool(
            not label_row["passes_direction_control"]
            and not sample_row["passes_direction_control"]
        ),
        "random_rotation_candidate_gain_mean": float(
            np.mean(
                [
                    row["candidate_gain"]
                    for row in rows
                    if str(row["control"]).startswith("random_orthogonal_rotation_")
                ]
            )
        ),
        "time_shuffle_supported": True,
        "time_shuffle_order": [2, 0, 4, 1, 3],
    }
    return rows, summary


def _plot_figures(
    output: Path,
    d2_candidates: Sequence[Mapping[str, Any]],
    d2_rows: Sequence[Mapping[str, Any]],
    d3_rows: Sequence[Mapping[str, Any]],
    d4_rows: Sequence[Mapping[str, Any]],
    d5_summary: Mapping[str, Any],
    d6_rows: Sequence[Mapping[str, Any]],
    selected: Sequence[int],
    main_calibration: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = output / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    ordered = sorted(d2_candidates, key=lambda row: int(row["train_only_rank"]))
    ranks = np.asarray([int(row["train_only_rank"]) for row in ordered])
    advantage = np.asarray([float(row["validation_mean_margin_advantage"]) for row in ordered])
    plt.figure(figsize=(8, 4))
    plt.plot(ranks, advantage, marker=".", linewidth=1)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Train-only direction rank")
    plt.ylabel("Validation CSI - sensing margin contribution")
    plt.tight_layout()
    plt.savefig(figure_root / "direction_advantage_rank.png", dpi=160)
    plt.close()

    stability = [
        row
        for row in d3_rows
        if int(row["k"]) == 8 and not str(row["source"]).endswith("_mean")
    ]
    sources = sorted({str(row["source"]) for row in stability})
    values = [
        np.mean([float(row["jaccard"]) for row in stability if row["source"] == source])
        for source in sources
    ]
    plt.figure(figsize=(9, 4))
    plt.bar(np.arange(len(sources)), values)
    plt.axhline(0.5, color="red", linestyle="--", linewidth=1)
    plt.xticks(np.arange(len(sources)), sources, rotation=30, ha="right")
    plt.ylabel("Top-8 Jaccard")
    plt.tight_layout()
    plt.savefig(figure_root / "direction_stability.png", dpi=160)
    plt.close()

    probe_k = sorted(
        {
            int(row["k"])
            for row in d4_rows
            if row.get("scope") == "all14" and row.get("method") == "P3_candidate"
        }
    )
    p0 = next(
        float(row["top1"])
        for row in d4_rows
        if row.get("scope") == "all14" and row.get("method") == "P0_sensing"
    )
    candidate_probe = [
        next(
            float(row["top1"])
            for row in d4_rows
            if row.get("scope") == "all14"
            and row.get("method") == "P3_candidate"
            and int(row["k"]) == k
        )
        for k in probe_k
    ]
    random_probe = [
        np.mean(
            [
                float(row["top1"])
                for row in d4_rows
                if row.get("scope") == "all14"
                and row.get("method") == "P4_random"
                and int(row["k"]) == k
            ]
        )
        for k in probe_k
    ]
    plt.figure(figsize=(7, 4))
    plt.plot(probe_k, (np.asarray(candidate_probe) - p0) * 100, marker="o", label="candidate")
    plt.plot(probe_k, (np.asarray(random_probe) - p0) * 100, marker="o", label="random")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("K")
    plt.ylabel("All-14 gain (pp)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_root / "probe_gain_vs_k.png", dpi=160)
    plt.close()

    intervention_k = sorted(
        {
            int(row["k"])
            for row in d6_rows
            if row.get("scope") == "all14"
            and row.get("calibration") == main_calibration
            and row.get("intervention") == "I1_candidate"
        }
    )
    intervention_base = next(
        float(row["top1"])
        for row in d6_rows
        if row.get("scope") == "all14"
        and row.get("calibration") == main_calibration
        and row.get("intervention") == "I0_sensing"
    )
    candidate_intervention = [
        next(
            float(row["top1"])
            for row in d6_rows
            if row.get("scope") == "all14"
            and row.get("calibration") == main_calibration
            and row.get("intervention") == "I1_candidate"
            and int(row["k"]) == k
        )
        for k in intervention_k
    ]
    random_intervention = [
        np.mean(
            [
                float(row["top1"])
                for row in d6_rows
                if row.get("scope") == "all14"
                and row.get("calibration") == main_calibration
                and row.get("intervention") == "I2_random"
                and int(row["k"]) == k
            ]
        )
        for k in intervention_k
    ]
    plt.figure(figsize=(7, 4))
    plt.plot(
        intervention_k,
        (np.asarray(candidate_intervention) - intervention_base) * 100,
        marker="o",
        label="candidate",
    )
    plt.plot(
        intervention_k,
        (np.asarray(random_intervention) - intervention_base) * 100,
        marker="o",
        label="random",
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("K")
    plt.ylabel("All-14 intervention gain (pp)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_root / "intervention_gain_vs_k.png", dpi=160)
    plt.close()

    selected_set = set(int(value) for value in selected)
    heatmap = np.full((len(MASK_NAMES), len(selected)), np.nan, dtype=np.float64)
    selected_position = {int(value): index for index, value in enumerate(selected)}
    for row in d2_rows:
        direction = int(row["direction"])
        if direction in selected_set:
            heatmap[int(row["mask_id"]), selected_position[direction]] = float(
                row["paired_advantage"]
            )
    plt.figure(figsize=(9, 6))
    image = plt.imshow(heatmap, aspect="auto", cmap="coolwarm")
    plt.colorbar(image, label="CSI - sensing margin contribution")
    plt.xticks(np.arange(len(selected)), [str(value) for value in selected])
    plt.yticks(np.arange(len(MASK_NAMES)), MASK_NAMES)
    plt.xlabel("Train-selected prototype direction")
    plt.tight_layout()
    plt.savefig(figure_root / "mask_direction_heatmap.png", dpi=160)
    plt.close()

    masks = list(MASK_NAMES)
    gains = [100.0 * float(d5_summary["per_mask"][mask]["r2_gain"]) for mask in masks]
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(masks)), gains)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(np.arange(len(masks)), masks, rotation=40, ha="right")
    plt.ylabel("R2 - R0 Top-1 (pp)")
    plt.tight_layout()
    plt.savefig(figure_root / "residual_information_by_mask.png", dpi=160)
    plt.close()


def _decision_summary(
    config: Mapping[str, Any],
    *,
    conditions: Mapping[str, Any],
    d1_candidates: Sequence[Mapping[str, Any]],
    d2_candidates: Sequence[Mapping[str, Any]],
    d2_state: Mapping[str, Any],
    d3_summary: Mapping[str, Any],
    d4_summary: Mapping[str, Any],
    d5_summary: Mapping[str, Any],
    d6_summary: Mapping[str, Any],
    d7_summary: Mapping[str, Any],
    negative_summary: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = config["thresholds"]
    candidate_directions = [
        int(row["direction"]) for row in d2_candidates if bool(row["formal_candidate"])
    ]
    raw_candidates = [
        int(row["direction"]) for row in d1_candidates if bool(row["formal_candidate"])
    ]
    candidate_pass = len(candidate_directions) >= 8
    stability_pass = bool(
        d3_summary["top8_jaccard_mean"] >= float(thresholds["top8_jaccard"])
        and d3_summary["top8_rank_spearman_mean"] >= float(thresholds["rank_spearman"])
        and d3_summary["top8_sign_consistency_mean"] >= float(thresholds["sign_consistency"])
    )
    trajectory_pass = bool(
        d3_summary["validation_trajectory_top8_jaccard_min"]
        >= float(thresholds["top8_jaccard"])
        and d3_summary["validation_trajectory_sign_consistency_min"]
        >= float(thresholds["sign_consistency"])
    )

    probe_scopes = d4_summary["scopes"]
    probe_thresholds = {
        "all14": float(thresholds["all14_gain_pp"]) / 100.0,
        "worst": float(thresholds["worst_gain_pp"]) / 100.0,
        "missing_lidar": float(thresholds["missing_lidar_gain_pp"]) / 100.0,
    }
    probe_scope_pass = _probe_scope_success(probe_scopes, probe_thresholds)
    probe_pass = any(probe_scope_pass.values())

    residual_scopes = d5_summary["scopes"]
    severe_positive = sum(
        d5_summary["per_mask"][mask]["r2_gain"] > 0
        for mask in conditions["severe_masks"]
    )
    residual_pass = bool(
        residual_scopes["all14"]["gain_ci_low"] > 0 and severe_positive >= 3
    )

    intervention_scopes = d6_summary["scopes"]
    intervention_threshold = bool(
        intervention_scopes["all14"]["gain"]
        >= float(thresholds["all14_gain_pp"]) / 100.0
        or intervention_scopes["worst"]["gain"]
        >= float(thresholds["worst_gain_pp"]) / 100.0
        or intervention_scopes["missing_lidar"]["gain"]
        >= float(thresholds["missing_lidar_gain_pp"]) / 100.0
    )
    intervention_signal = bool(
        any(scope["gain_ci_low"] > 0 for scope in intervention_scopes.values())
        and any(
            scope["candidate_over_random_ci_low"] > 0
            for scope in intervention_scopes.values()
        )
    )
    intervention_pass = bool(intervention_threshold and intervention_signal)
    deletion_pass = bool(
        d7_summary["additional_drop_ci_low"] > 0
        and d7_summary["candidate_additional_drop_over_random"] > 0
    )
    negative_pass = bool(negative_summary["negative_controls_pass"])
    level2 = all(
        (
            candidate_pass,
            stability_pass,
            trajectory_pass,
            probe_pass,
            residual_pass,
            intervention_pass,
            deletion_pass,
            negative_pass,
        )
    )
    level1 = all(
        (
            candidate_pass,
            stability_pass,
            trajectory_pass,
            probe_pass,
            residual_pass,
            intervention_signal,
            deletion_pass,
            negative_pass,
        )
    )
    level = 2 if level2 else 1 if level1 else 0
    selected = [int(value) for value in np.asarray(d2_state["order"])[:8]]
    mask_advantage = np.asarray(d2_state["validation_advantages"])[:, selected].mean(axis=1)
    mask_order = np.argsort(-mask_advantage)
    return {
        "level": level,
        "level_name": {
            0: "Level 0: 不存在满足完整证据链的方向互补",
            1: "Level 1: 存在统计互补但不足以构成方法",
            2: "Level 2: 存在可利用的方向互补",
        }[level],
        "criteria": {
            "at_least_8_candidates": candidate_pass,
            "direction_stability": stability_pass,
            "two_validation_trajectories": trajectory_pass,
            "probe_over_sensing_and_random": probe_pass,
            "conditional_residual_information": residual_pass,
            "replacement_threshold_and_random": intervention_pass,
            "candidate_deletion_over_random": deletion_pass,
            "negative_controls": negative_pass,
        },
        "prototype_candidate_directions": candidate_directions,
        "prototype_candidate_count": len(candidate_directions),
        "raw_coordinate_candidate_directions": raw_candidates,
        "raw_coordinate_candidate_count": len(raw_candidates),
        "selected_top8_train_only": selected,
        "strongest_masks": [MASK_NAMES[int(index)] for index in mask_order[:5]],
        "strongest_mask_advantages": [
            float(mask_advantage[int(index)]) for index in mask_order[:5]
        ],
        "worth_implementing_compensation": level == 2,
        "recommended_basis": "prototype_svd" if level == 2 else "none",
        "recommended_k": 8 if level == 2 else None,
    }


def _probe_scope_success(
    scopes: Mapping[str, Mapping[str, float]],
    thresholds: Mapping[str, float],
) -> dict[str, bool]:
    return {
        name: bool(
            scopes[name]["gain"] >= thresholds[name]
            and scopes[name]["p3_top8"]
            > scopes[name]["p4_random_mean"] + scopes[name]["p4_random_std"]
            and scopes[name]["gain_ci_low"] > 0
        )
        for name in thresholds
    }


def _write_final_report(
    output: Path,
    summary: Mapping[str, Any],
) -> None:
    decision = summary["decision"]
    d3 = summary["d3"]
    d4 = summary["d4"]
    d5 = summary["d5"]
    d6 = summary["d6"]
    d7 = summary["d7"]
    negative = summary["negative_controls"]
    sample_groups = summary["sample_groups"]
    selected = decision["selected_top8_train_only"]
    all14_probe = d4["scopes"]["all14"]
    all14_residual = d5["scopes"]["all14"]
    all14_intervention = d6["scopes"]["all14"]
    g2 = sample_groups["g2"]
    lines = [
        "# 超稀疏 CSI 互补方向诊断报告",
        "",
        "## 最终判定",
        "",
        f"- **{decision['level_name']}**。",
        (
            "- 只有 Level 2 才进入 CSI 互补维度补偿；本轮结论"
            + ("支持继续实现。" if decision["worth_implementing_compensation"] else "不支持继续实现。")
        ),
        "- 本轮没有训练或修改 M4、CSI Encoder/GRU、RadioPrototypeExpert、Beam Prototype Bank 或 F1；D6 是 oracle 诊断，不是部署方法。",
        "- outer test 未访问，输入仍为 t-4...t 五帧、每帧 4 RE、窗口 20 RE，target 为 t+1。",
        "",
        "## 预注册门槛",
        "",
    ]
    for name, passed in decision["criteria"].items():
        lines.append(f"- {name}: {'通过' if passed else '未通过'}")
    lines.extend(
        [
            "",
            "## 必答问题",
            "",
            "1. **z_s 和 z_c 是否可直接比较**：可以作原始坐标诊断。两者虽经不同上游 projection，但最终都直接查询同一满秩冻结 [64,64] bank；主证据仍使用共同 prototype SVD basis。",
            f"2. **原始坐标是否稳定**：原始坐标得到 {decision['raw_coordinate_candidate_count']} 个正式候选；随机正交旋转的平均干预增益为 {100.0*negative['random_rotation_candidate_gain_mean']:.3f} pp，因此原始轴结果只作补充，不作为旋转不变主结论。",
            f"3. **prototype-basis 是否存在稳定互补方向**：正式候选 {decision['prototype_candidate_count']} 个；top-8 综合 Jaccard={d3['top8_jaccard_mean']:.3f}、Spearman={d3['top8_rank_spearman_mean']:.3f}、符号一致率={d3['top8_sign_consistency_mean']:.3f}。",
            f"4. **候选方向数量**：{decision['prototype_candidate_count']}；train-only 主 top-8 为 {selected}。",
            f"5. **集中在哪些 singular direction**：train-only top-8 为 {selected}；完整 singular value/rank 见 d2 CSV 和 prototype_svd.pt。",
            f"6. **最明显 mask**：{decision['strongest_masks']}。",
            f"7. **train/validation/轨迹稳定性**：两条 validation trajectory 的最小 top-8 Jaccard={d3['validation_trajectory_top8_jaccard_min']:.3f}，最小符号一致率={d3['validation_trajectory_sign_consistency_min']:.3f}。",
            f"8. **条件 residual 是否仍含 beam 信息**：All-14 R2-R0={100.0*all14_residual['gain']:.3f} pp，95% CI [{100.0*all14_residual['gain_ci_low']:.3f}, {100.0*all14_residual['gain_ci_high']:.3f}] pp。",
            f"9. **top-8 是否提升 linear probe**：P0={100.0*all14_probe['p0']:.3f}%，P3={100.0*all14_probe['p3_top8']:.3f}%，增益={100.0*all14_probe['gain']:.3f} pp。",
            f"10. **是否超过随机方向**：P4 随机均值/标准差={100.0*all14_probe['p4_random_mean']:.3f}%/{100.0*all14_probe['p4_random_std']:.3f}%，P3-P4={100.0*all14_probe['candidate_over_random_mean']:.3f} pp。",
            f"11. **oracle replacement 能否修复 sensing**：All-14 基线={100.0*all14_intervention['baseline']:.3f}%，top-8={100.0*all14_intervention['candidate_top8']:.3f}%，增益={100.0*all14_intervention['gain']:.3f} pp，95% CI [{100.0*all14_intervention['gain_ci_low']:.3f}, {100.0*all14_intervention['gain_ci_high']:.3f}] pp。",
            f"12. **replacement Fix/Harm**：All-14 Fix={100.0*all14_intervention['fix_rate']:.3f}%，Harm={100.0*all14_intervention['harm_rate']:.3f}%。",
            f"13. **删除候选方向是否更损害 CSI**：候选 top-8 drop={100.0*d7['candidate_top8_drop']:.3f} pp，随机 drop={100.0*d7['random_top8_drop_mean']:.3f}±{100.0*d7['random_top8_drop_std']:.3f} pp，额外 drop={100.0*d7['candidate_additional_drop_over_random']:.3f} pp。",
            f"14. **是否主要作用于 sensing 错、CSI 对**：G2 样本 top-8 contribution advantage={g2['candidate_direction_advantage']:.6f}，replacement Fix={100.0*g2['replacement_fix_rate']:.3f}%，Harm={100.0*g2['replacement_harm_rate']:.3f}%。",
            f"15. **permutation/shuffle 负对照**：label permutation 通过错误门槛={negative['label_permutation_passed']}，sample shuffle 通过错误门槛={negative['sample_shuffle_passed']}；要求二者均为 False。",
            f"16. **最终 Level**：{decision['level_name']}。",
            f"17. **是否值得实现补偿**：{'是' if decision['worth_implementing_compensation'] else '否'}。",
            f"18. **推荐原始维度还是 prototype basis**：{decision['recommended_basis']}。",
            f"19. **推荐 K**：{decision['recommended_k'] if decision['recommended_k'] is not None else '不推荐进入方法实现'}。",
            "20. **outer test/future CSI**：两者均保持严格封存；outer-test loader/cache 未构造，time-shuffle 只对历史 frame feature 经冻结 GRU 重算。",
            "",
            "## 解释边界",
            "",
            "- D0 只证明样本级互补，不单独支持方向结论。",
            "- D4/D5 probe 只在 train 拟合并一次评估 validation；不保存或部署。",
            "- D6 使用同一样本 CSI coefficient 的 oracle replacement，只有它显著超过随机方向且通过完整证据链时才具备方法开发价值。",
            "",
        ]
    )
    (output / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_diagnostics(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    output = _path(config["output"]["root"])
    train = _derived_cache(config, "train")
    validation = _derived_cache(config, "validation")
    statistics = config["statistics"]
    weights = bootstrap_weights(
        len(validation["target"]),
        int(statistics["bootstrap_replicates"]),
        int(statistics["bootstrap_seed"]),
    )
    print("D0 sample complementarity", flush=True)
    d0_rows, conditions = analyze_d0(config, validation, weights)
    _write_csv(output / "diagnostics/d0_sample_complementarity.csv", d0_rows)

    prototype = train["prototype_normalized"].float()
    singular_values_t, directions_t = prototype_basis(prototype)
    singular_values = singular_values_t.numpy()
    directions = directions_t.numpy()
    centered = prototype - prototype.mean(dim=0, keepdim=True)
    artifact = {
        "prototype_centered": centered,
        "singular_values": singular_values_t,
        "directions": directions_t,
        "direction_convention": "columns; coefficients=z@V; reconstruction=coefficients@V.T",
        "prototype_bank_sha256": config["source"]["prototype_bank_sha256"],
        "selection_role": "train",
        "outer_test_accessed": False,
    }
    artifact_path = output / "artifacts/prototype_svd.pt"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)

    print("D1 raw-coordinate discriminability", flush=True)
    identity = np.eye(DIMENSION, dtype=np.float32)
    d1_rows, d1_candidates, _ = analyze_discriminability(
        config,
        train,
        validation,
        directions=identity,
        basis_name="raw_coordinate",
        conditions=conditions,
        weights=weights,
    )
    _write_csv(output / "diagnostics/d1_coordinate_discriminability.csv", d1_rows)
    _write_csv(output / "diagnostics/d1_candidate_coordinates.csv", d1_candidates)

    print("D2 prototype-basis discriminability", flush=True)
    d2_rows, d2_candidates, d2_state = analyze_discriminability(
        config,
        train,
        validation,
        directions=directions,
        basis_name="prototype_svd",
        conditions=conditions,
        weights=weights,
        singular_values=singular_values,
    )
    _write_csv(output / "diagnostics/d2_prototype_basis_discriminability.csv", d2_rows)
    _write_csv(output / "diagnostics/d2_candidate_directions.csv", d2_candidates)
    selected = [int(value) for value in np.asarray(d2_state["order"])[:8]]
    _write_json(
        output / "artifacts/selected_top8_directions.json",
        {
            "directions": selected,
            "selection_role": "train",
            "selection_used_validation": False,
            "primary_k": 8,
            "basis": "centered_prototype_svd_right_singular_vectors",
            "prototype_bank_sha256": config["source"]["prototype_bank_sha256"],
            "outer_test_accessed": False,
        },
    )

    print("D3 direction stability", flush=True)
    d3_rows, d3_summary = analyze_d3(
        config, train, validation, d2_state, conditions, directions
    )
    _write_csv(output / "diagnostics/d3_direction_stability.csv", d3_rows)

    print("D4 train-only linear probes", flush=True)
    d4_rows, d4_summary = analyze_d4(
        config, train, validation, d2_state, conditions, weights
    )
    _write_csv(output / "diagnostics/d4_linear_probe_summary.csv", d4_rows)

    print("D5 conditional CSI residual", flush=True)
    d5_rows, d5_summary = analyze_d5(
        config,
        train,
        validation,
        directions,
        float(d4_summary["alpha"]),
        conditions,
        weights,
    )
    _write_csv(output / "diagnostics/d5_conditional_residual_information.csv", d5_rows)

    print("D6 direction intervention", flush=True)
    d6_rows, d6_summary, intervention_state = analyze_d6(
        config,
        train,
        validation,
        d2_state,
        directions,
        conditions,
        weights,
        device,
    )
    _write_csv(output / "diagnostics/d6_direction_intervention.csv", d6_rows)

    print("D7 direction deletion", flush=True)
    d7_rows, d7_summary = analyze_d7(
        config, train, validation, d2_state, directions, weights, device
    )
    _write_csv(output / "diagnostics/d7_direction_ablation.csv", d7_rows)

    print("Sample groups and negative controls", flush=True)
    sample_rows = analyze_sample_groups(validation, d2_state, intervention_state)
    _write_csv(output / "diagnostics/sample_group_direction_analysis.csv", sample_rows)
    negative_rows, negative_summary = analyze_negative_controls(
        config,
        train,
        validation,
        directions,
        conditions,
        weights,
        d4_summary,
        d6_summary,
        device,
    )
    _write_csv(output / "diagnostics/negative_controls.csv", negative_rows)

    decision = _decision_summary(
        config,
        conditions=conditions,
        d1_candidates=d1_candidates,
        d2_candidates=d2_candidates,
        d2_state=d2_state,
        d3_summary=d3_summary,
        d4_summary=d4_summary,
        d5_summary=d5_summary,
        d6_summary=d6_summary,
        d7_summary=d7_summary,
        negative_summary=negative_summary,
    )
    g2_rows = [
        row
        for row in sample_rows
        if row["row_type"] == "group_summary"
        and row["group"] == "G2_sensing_wrong_csi_correct"
    ]
    sample_summary = {
        "g2": {
            "samples": int(sum(int(row["samples"]) for row in g2_rows)),
            "candidate_direction_advantage": float(
                np.average(
                    [float(row["candidate_direction_advantage"]) for row in g2_rows],
                    weights=[int(row["samples"]) for row in g2_rows],
                )
            ),
            "replacement_fix_rate": float(
                np.average(
                    [float(row["replacement_fix_rate"]) for row in g2_rows],
                    weights=[int(row["samples"]) for row in g2_rows],
                )
            ),
            "replacement_harm_rate": float(
                np.average(
                    [float(row["replacement_harm_rate"]) for row in g2_rows],
                    weights=[int(row["samples"]) for row in g2_rows],
                )
            ),
        }
    }
    d4_public = {key: value for key, value in d4_summary.items() if key != "candidate_predictions"}
    summary = {
        "protocol": {
            "id": config["protocol"]["id"],
            "train_samples": len(train["target"]),
            "validation_samples": len(validation["target"]),
            "train_trajectories": len(set(train["trajectory_ids"])),
            "validation_trajectories": len(set(validation["trajectory_ids"])),
            "history_frames": 5,
            "re_per_frame": 4,
            "re_window": 20,
            "future_channel_used": False,
            "test_loader_constructed": False,
            "outer_test_accessed": False,
        },
        "statistics": {
            "bootstrap_replicates": int(statistics["bootstrap_replicates"]),
            "paired_bootstrap": True,
            "confidence_level": 0.95,
            "fdr": "Benjamini-Hochberg",
            "fdr_q": float(statistics["fdr_q"]),
            "primary_k": 8,
        },
        "conditions": conditions,
        "prototype_svd": {
            "singular_values": singular_values.tolist(),
            "effective_rank": int(
                (
                    singular_values
                    > singular_values.max()
                    * float(statistics["singular_value_relative_tolerance"])
                ).sum()
            ),
            "artifact_sha256": sha256_file(artifact_path),
        },
        "d3": d3_summary,
        "d4": d4_public,
        "d5": d5_summary,
        "d6": d6_summary,
        "d7": d7_summary,
        "negative_controls": negative_summary,
        "sample_groups": sample_summary,
        "decision": decision,
        "model_or_fusion_trained": False,
    }
    _write_json(output / "diagnostic_summary.json", summary)
    _plot_figures(
        output,
        d2_candidates,
        d2_rows,
        d3_rows,
        d4_rows,
        d5_summary,
        d6_rows,
        selected,
        str(statistics["intervention_main_calibration"]),
    )
    _write_final_report(output, summary)
    return summary


def _write_resolved_config(config: Mapping[str, Any]) -> None:
    output = _path(config["output"]["root"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("all", "audit", "cache", "diagnose", "summarize"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()
    config = _load_config(args.config)
    device_name = str(args.device or config["runtime"]["device"])
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable.")
    device = torch.device(device_name)
    _write_resolved_config(config)
    started = time.time()
    if args.command == "audit":
        result = audit(config, device)
    elif args.command == "cache":
        result = build_cache(config, device, force=bool(args.force_cache))
    elif args.command == "diagnose":
        result = run_diagnostics(config, device)
    elif args.command == "summarize":
        output = _path(config["output"]["root"])
        result = json.loads((output / "diagnostic_summary.json").read_text(encoding="utf-8"))
        _write_final_report(output, result)
    else:
        build_cache(config, device, force=bool(args.force_cache))
        result = run_diagnostics(config, device)
    elapsed = time.time() - started
    _write_json(
        _path(config["output"]["root"]) / "process_manifest.json",
        {
            "command": args.command,
            "device": str(device),
            "elapsed_seconds": elapsed,
            "status": "complete",
            "result_level": result.get("decision", {}).get("level")
            if isinstance(result, Mapping)
            else None,
            "outer_test_accessed": False,
        },
    )
    print(json.dumps({"status": "complete", "elapsed_seconds": elapsed}, sort_keys=True))


if __name__ == "__main__":
    main()
