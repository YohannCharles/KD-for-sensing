"""Train-only geometry for the sensing prototype-deformation diagnostic."""

from __future__ import annotations

from collections import OrderedDict
from itertools import combinations
from typing import Mapping

import numpy as np
import torch


MODALITIES = ("image", "lidar", "radar", "gps")
MASKS: OrderedDict[str, tuple[int, int, int, int]] = OrderedDict(
    (
        ("full", (1, 1, 1, 1)),
        ("missing_image", (0, 1, 1, 1)),
        ("missing_radar", (1, 1, 0, 1)),
        ("missing_gps", (1, 1, 1, 0)),
        ("missing_lidar", (1, 0, 1, 1)),
        ("missing_image_radar", (0, 1, 0, 1)),
        ("missing_image_gps", (0, 1, 1, 0)),
        ("missing_image_lidar", (0, 0, 1, 1)),
        ("missing_radar_gps", (1, 1, 0, 0)),
        ("missing_lidar_radar", (1, 0, 0, 1)),
        ("missing_lidar_gps", (1, 0, 1, 0)),
        ("image_only", (1, 0, 0, 0)),
        ("radar_only", (0, 0, 1, 0)),
        ("gps_only", (0, 0, 0, 1)),
        ("lidar_only", (0, 1, 0, 0)),
    )
)
SINGLE_MISSING = {
    "image": "missing_image",
    "lidar": "missing_lidar",
    "radar": "missing_radar",
    "gps": "missing_gps",
}


def mask_metadata() -> list[dict[str, object]]:
    """Return the pre-registered mask order and its availability groups."""
    rows = []
    for index, (name, bits) in enumerate(MASKS.items()):
        available = [modality for modality, bit in zip(MODALITIES, bits) if bit]
        missing = [modality for modality, bit in zip(MODALITIES, bits) if not bit]
        rows.append(
            {
                "mask_id": index,
                "mask": name,
                "bits": list(bits),
                "available_modalities": available,
                "missing_modalities": missing,
                "available_count": len(available),
                "missing_count": len(missing),
                "group": "Full" if not missing else {1: "Three", 2: "Two", 3: "Single"}[len(available)],
            }
        )
    return rows


def normalize(value: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize the final dimension without producing NaN for empty sums."""
    return value / value.norm(dim=-1, keepdim=True).clamp_min(float(eps))


def spherical_log_map(base: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Map unit-sphere targets into the tangent plane at ``base``."""
    base = normalize(torch.as_tensor(base))
    target = normalize(torch.as_tensor(target, device=base.device, dtype=base.dtype))
    dot = (base * target).sum(dim=-1, keepdim=True)
    clipped = dot.clamp(-1.0 + float(eps), 1.0 - float(eps))
    theta = torch.acos(clipped)
    tangent = target - dot * base
    scale = theta / torch.sin(theta).clamp_min(float(eps))
    mapped = scale * tangent
    return torch.where(tangent.norm(dim=-1, keepdim=True) <= float(eps), torch.zeros_like(mapped), mapped)


def spherical_exp_map(base: torch.Tensor, tangent: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Map tangent vectors back to the unit sphere."""
    base = normalize(torch.as_tensor(base))
    tangent = torch.as_tensor(tangent, device=base.device, dtype=base.dtype)
    tangent = tangent - (tangent * base).sum(dim=-1, keepdim=True) * base
    length = tangent.norm(dim=-1, keepdim=True)
    direction = tangent / length.clamp_min(float(eps))
    mapped = torch.cos(length) * base + torch.sin(length) * direction
    mapped = torch.where(length <= float(eps), base, mapped)
    return normalize(mapped)


def estimate_centers(
    features: torch.Tensor,
    labels: torch.Tensor,
    learned_prototypes: torch.Tensor,
    *,
    kappa: float = 20.0,
    num_beams: int = 64,
) -> dict[str, torch.Tensor]:
    """Estimate Euclidean, spherical, and learned-bank shrinkage centers."""
    values = torch.as_tensor(features, dtype=torch.float32)
    target = torch.as_tensor(labels, dtype=torch.long, device=values.device).reshape(-1)
    prototypes = normalize(torch.as_tensor(learned_prototypes, dtype=torch.float32, device=values.device))
    if values.ndim != 3 or values.shape[0] != target.numel() or values.shape[1] != len(MASKS):
        raise ValueError(f"features must have shape [N, {len(MASKS)}, D] and labels [N].")
    if prototypes.shape != (int(num_beams), values.shape[-1]):
        raise ValueError("learned prototype shape does not match the beam and feature dimensions.")
    if not bool(((target >= 0) & (target < int(num_beams))).all()):
        raise ValueError("labels are outside the configured beam range.")

    counts = torch.bincount(target, minlength=int(num_beams)).to(dtype=torch.float32)
    raw = torch.zeros(len(MASKS), int(num_beams), values.shape[-1], dtype=torch.float32, device=values.device)
    unit = torch.zeros_like(raw)
    normalized_features = normalize(values)
    for mask_index in range(len(MASKS)):
        raw[mask_index].index_add_(0, target, values[:, mask_index])
        unit[mask_index].index_add_(0, target, normalized_features[:, mask_index])
    denominator = counts.clamp_min(1.0)[None, :, None]
    euclidean = raw / denominator
    spherical = normalize(unit)
    unsupported = counts.eq(0)[None, :, None]
    spherical = torch.where(unsupported, prototypes[None], spherical)
    euclidean = torch.where(unsupported, prototypes[None], euclidean)
    weight = counts / (counts + float(kappa))
    shrinkage = normalize(weight[None, :, None] * spherical + (1.0 - weight)[None, :, None] * prototypes[None])
    return {
        "counts": counts.to(dtype=torch.long),
        "euclidean": euclidean,
        "spherical": spherical,
        "shrinkage": shrinkage,
        "learned": prototypes,
        "supported": counts.gt(0),
    }


def tangent_shifts(centers: torch.Tensor) -> torch.Tensor:
    """Return every mask center's tangent shift from the Full center."""
    values = normalize(torch.as_tensor(centers, dtype=torch.float32))
    if values.ndim != 3 or values.shape[0] != len(MASKS):
        raise ValueError(f"centers must have shape [{len(MASKS)}, B, D].")
    return spherical_log_map(values[0:1].expand_as(values), values)


def euclidean_shifts(centers: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(centers, dtype=torch.float32)
    if values.ndim != 3 or values.shape[0] != len(MASKS):
        raise ValueError(f"centers must have shape [{len(MASKS)}, B, D].")
    return values - values[0:1]


def additive_deformation(shifts: torch.Tensor) -> torch.Tensor:
    """Compose every mask only from the four registered single-missing shifts."""
    values = torch.as_tensor(shifts, dtype=torch.float32)
    if values.ndim != 3 or values.shape[0] != len(MASKS):
        raise ValueError(f"shifts must have shape [{len(MASKS)}, B, D].")
    by_modality = {name: values[list(MASKS).index(mask)] for name, mask in SINGLE_MISSING.items()}
    result = torch.zeros_like(values)
    for mask_index, metadata in enumerate(mask_metadata()):
        for modality in metadata["missing_modalities"]:
            result[mask_index] += by_modality[str(modality)]
    return result


def count_deformation(shifts: torch.Tensor) -> torch.Tensor:
    """Fit one per-beam contribution from single-missing conditions only."""
    values = torch.as_tensor(shifts, dtype=torch.float32)
    singles = torch.stack([values[list(MASKS).index(mask)] for mask in SINGLE_MISSING.values()])
    alpha = singles.mean(dim=0)
    counts = torch.tensor([row["missing_count"] for row in mask_metadata()], dtype=values.dtype, device=values.device)
    return counts[:, None, None] * alpha[None]


def pairwise_deformation(shifts: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fit single and pair terms, using only single and double missing masks."""
    values = torch.as_tensor(shifts, dtype=torch.float32)
    additive = additive_deformation(values)
    pair_terms: dict[str, torch.Tensor] = {}
    for left, right in combinations(MODALITIES, 2):
        wanted = {left, right}
        mask_index = next(
            index for index, row in enumerate(mask_metadata()) if set(row["missing_modalities"]) == wanted
        )
        pair_terms[f"{left}+{right}"] = values[mask_index] - additive[mask_index]
    result = additive.clone()
    for mask_index, metadata in enumerate(mask_metadata()):
        missing = set(metadata["missing_modalities"])
        for left, right in combinations(MODALITIES, 2):
            if {left, right}.issubset(missing):
                result[mask_index] += pair_terms[f"{left}+{right}"]
    return result, pair_terms


def centers_from_deformation(full_centers: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Reconstruct mask centers through the spherical exponential map."""
    base = normalize(torch.as_tensor(full_centers, dtype=torch.float32))
    values = torch.as_tensor(shifts, dtype=torch.float32, device=base.device)
    return spherical_exp_map(base[None].expand_as(values), values)


def topology_adjacency(distance: torch.Tensor) -> torch.Tensor:
    """Build adjacency strictly from the audited distance matrix."""
    value = torch.as_tensor(distance, dtype=torch.float32)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not bool(torch.allclose(value, value.t())):
        raise ValueError("topology distance must be a square symmetric matrix.")
    adjacency = value.eq(value[value.gt(0)].min()).to(dtype=torch.float32)
    adjacency.fill_diagonal_(0)
    return adjacency


def smooth_deformation(
    shifts: torch.Tensor,
    full_centers: torch.Tensor,
    adjacency: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    """Apply graph-Laplacian smoothing and reproject into each beam tangent plane."""
    values = torch.as_tensor(shifts, dtype=torch.float32)
    base = normalize(torch.as_tensor(full_centers, dtype=torch.float32, device=values.device))
    graph = torch.as_tensor(adjacency, dtype=torch.float32, device=values.device)
    laplacian = torch.diag(graph.sum(dim=1)) - graph
    system = torch.eye(graph.shape[0], device=values.device) + float(coefficient) * laplacian
    smoothed = torch.linalg.solve(system, values.transpose(0, 1).reshape(graph.shape[0], -1)).reshape_as(values.transpose(0, 1)).transpose(0, 1)
    return smoothed - (smoothed * base[None]).sum(dim=-1, keepdim=True) * base[None]


def prototype_logits(features: torch.Tensor, prototypes: torch.Tensor, *, temperature: float = 0.1) -> torch.Tensor:
    """Score shared [B,D] or mask-conditioned [M,B,D] prototypes."""
    values = normalize(torch.as_tensor(features, dtype=torch.float32))
    banks = normalize(torch.as_tensor(prototypes, dtype=torch.float32, device=values.device))
    if values.ndim == 2 and banks.ndim == 2:
        return values @ banks.t() / float(temperature)
    if values.ndim == 3 and banks.ndim == 2:
        return torch.einsum("nmd,bd->nmb", values, banks) / float(temperature)
    if values.ndim == 3 and banks.ndim == 3 and values.shape[1] == banks.shape[0]:
        return torch.einsum("nmd,mbd->nmb", values, banks) / float(temperature)
    raise ValueError("expected features/prototypes as [N,D]/[B,D] or [N,M,D]/[M,B,D].")


def weighted_r2(observed: torch.Tensor, predicted: torch.Tensor, weights: torch.Tensor | None = None) -> float:
    """Compute an element-wise weighted coefficient of determination."""
    truth = torch.as_tensor(observed, dtype=torch.float64).reshape(-1)
    estimate = torch.as_tensor(predicted, dtype=torch.float64).reshape(-1)
    weight = torch.ones_like(truth) if weights is None else torch.as_tensor(weights, dtype=torch.float64).expand_as(observed).reshape(-1)
    valid = torch.isfinite(truth) & torch.isfinite(estimate) & torch.isfinite(weight) & weight.gt(0)
    truth, estimate, weight = truth[valid], estimate[valid], weight[valid]
    if truth.numel() == 0:
        return float("nan")
    mean = (truth * weight).sum() / weight.sum()
    residual = ((truth - estimate).square() * weight).sum()
    total = ((truth - mean).square() * weight).sum()
    return float((1.0 - residual / total.clamp_min(torch.finfo(torch.float64).eps)).item())


def benjamini_hochberg(p_values: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return monotone Benjamini-Hochberg adjusted q-values."""
    values = np.asarray(p_values, dtype=np.float64)
    flat = values.reshape(-1)
    result = np.full_like(flat, np.nan)
    valid = np.isfinite(flat)
    selected = flat[valid]
    if selected.size:
        order = np.argsort(selected)
        ranked = selected[order] * selected.size / np.arange(1, selected.size + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
        restored = np.empty_like(ranked)
        restored[order] = ranked
        result[valid] = restored
    return result.reshape(values.shape)


def validate_mask_contract(masks: Mapping[str, tuple[int, int, int, int]] = MASKS) -> None:
    """Reject missing, duplicate, empty, or wrongly ordered availability masks."""
    if tuple(masks) != tuple(MASKS) or len(masks) != 15:
        raise ValueError("the diagnostic requires the pre-registered 15-mask order.")
    values = [tuple(int(value) for value in bits) for bits in masks.values()]
    if values[0] != (1, 1, 1, 1) or len(set(values)) != 15 or any(sum(bits) == 0 for bits in values):
        raise ValueError("availability masks must be unique, non-empty, and start with Full.")


__all__ = [
    "MASKS",
    "MODALITIES",
    "SINGLE_MISSING",
    "additive_deformation",
    "benjamini_hochberg",
    "centers_from_deformation",
    "count_deformation",
    "estimate_centers",
    "euclidean_shifts",
    "mask_metadata",
    "normalize",
    "pairwise_deformation",
    "prototype_logits",
    "smooth_deformation",
    "spherical_exp_map",
    "spherical_log_map",
    "tangent_shifts",
    "topology_adjacency",
    "validate_mask_contract",
    "weighted_r2",
]
