"""Deterministic assignment rules for the Full-pool BTMA causal ablation."""

from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np

from kd_sensing.baselines.full_pool_candidate12 import (
    MODALITIES,
    assignment_diagnostics,
    capacity_constrained_assignment,
    percentile_ranks,
)


BTMA_METHODS = (
    "b0_random_balanced",
    "b1_fixed_weak_schedule",
    "b2_kl_capacity",
    "b3_topology_risk_only",
    "b4_margin_only",
    "b5_risk_margin_full",
)
CAPACITY_METHODS = frozenset(BTMA_METHODS) - {"b1_fixed_weak_schedule"}


def _hash_uniform(sample_id: str, modality: int, seed: int) -> float:
    payload = f"btma-v1:{seed}:{sample_id}:{modality}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def random_balanced_scores(sample_ids: Sequence[str], *, seed: int = 2026) -> np.ndarray:
    """Return fixed, sample-id-bound pseudo-random scores without labels."""

    return np.asarray(
        [[_hash_uniform(str(sample_id), modality, seed) for modality in range(4)] for sample_id in sample_ids],
        dtype=np.float64,
    )


def fixed_proportion_assignment(sample_ids: Sequence[str], proportions: Mapping[str, float]) -> np.ndarray:
    """Assign hash-sorted samples to exact integer quotas from historical proportions."""

    values = np.asarray([float(proportions[name]) for name in MODALITIES], dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("BTMA fixed proportions must be four finite non-negative values.")
    count = len(sample_ids)
    normalized = values / values.sum()
    raw = normalized * count
    quotas = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - quotas), kind="stable")[: count - int(quotas.sum())]:
        quotas[index] += 1
    order = np.asarray(
        sorted(range(count), key=lambda index: (_hash_uniform(str(sample_ids[index]), 0, 731), str(sample_ids[index]))),
        dtype=np.int64,
    )
    result = np.empty(count, dtype=np.int64)
    start = 0
    for modality, quota in enumerate(quotas):
        result[order[start : start + int(quota)]] = modality
        start += int(quota)
    if start != count:
        raise AssertionError("BTMA fixed quota assignment left samples unassigned.")
    return result


def score_assignment(
    method: str,
    *,
    logits: np.ndarray,
    features: np.ndarray,
    prototypes: np.ndarray,
    labels: np.ndarray,
    topology_distance: np.ndarray,
    sample_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, Mapping[str, np.ndarray]]:
    """Produce method-local scores and capacity-constrained assignments."""

    if method not in {"b2_kl_capacity", "b3_topology_risk_only", "b4_margin_only", "b5_risk_margin_full"}:
        raise ValueError(f"BTMA score assignment does not support {method}.")
    diagnostics = assignment_diagnostics(logits, features, prototypes, labels, topology_distance, sample_ids)
    if method == "b2_kl_capacity":
        scores = percentile_ranks(-diagnostics["kl_uniform"], sample_ids)
    elif method == "b3_topology_risk_only":
        scores = diagnostics["risk_rank"]
    elif method == "b4_margin_only":
        # Raw negative margin provides the protocol-required deterministic secondary order.
        scores = diagnostics["margin_rank"] + 1e-12 * percentile_ranks(-diagnostics["margin"], sample_ids)
    else:
        scores = diagnostics["combined_hardness"]
    return scores, capacity_constrained_assignment(scores, sample_ids), diagnostics


__all__ = [
    "BTMA_METHODS",
    "CAPACITY_METHODS",
    "fixed_proportion_assignment",
    "random_balanced_scores",
    "score_assignment",
]
