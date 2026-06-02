from __future__ import annotations

import math
from typing import Iterable

import torch

from kd_sensing.data.geometry_residual import angle_to_beam as _angle_to_beam
from kd_sensing.data.geometry_residual import circular_beam_distance as _circular_beam_distance
from kd_sensing.data.geometry_residual import normalize_angle_degrees


def angle_to_beam(
    angle_degrees: float,
    *,
    num_classes: int,
    start_degrees: float = 0.0,
    direction: int = 1,
    beam_offset: int = 0,
) -> int:
    signed_angle = float(angle_degrees) * (1.0 if int(direction) >= 0 else -1.0)
    beam = _angle_to_beam(signed_angle, num_beams=int(num_classes), start_degrees=float(start_degrees))
    return int((beam + int(beam_offset)) % int(num_classes))


def circular_beam_distance(left: int, right: int, *, num_classes: int) -> int:
    return _circular_beam_distance(int(left), int(right), num_beams=int(num_classes))


def signed_angle_delta_degrees(next_angle: float, prev_angle: float) -> float:
    return float((float(next_angle) - float(prev_angle) + 180.0) % 360.0 - 180.0)


def circular_mean_degrees(angles: Iterable[float]) -> float:
    values = [float(item) for item in angles]
    if not values:
        raise ValueError("circular_mean_degrees requires at least one angle.")
    sin_sum = sum(math.sin(math.radians(item)) for item in values)
    cos_sum = sum(math.cos(math.radians(item)) for item in values)
    return normalize_angle_degrees(math.degrees(math.atan2(sin_sum, cos_sum)))


def circular_velocity_degrees(angles: Iterable[float]) -> float:
    values = [float(item) for item in angles]
    if len(values) < 2:
        return 0.0
    deltas = [signed_angle_delta_degrees(values[idx], values[idx - 1]) for idx in range(1, len(values))]
    return float(sum(deltas) / max(len(deltas), 1))


def topk_neighbors(center_beam: int, *, num_classes: int, k: int) -> tuple[int, ...]:
    beams = list(range(int(num_classes)))
    beams.sort(key=lambda item: (circular_beam_distance(item, center_beam, num_classes=num_classes), item))
    return tuple(int(item) for item in beams[: max(1, min(int(k), int(num_classes)))])


def beam_score_kernel(
    center_beams: Iterable[int],
    *,
    num_classes: int,
    width: float = 2.0,
    temperature: float = 1.0,
    neighbor_top_k: int | None = None,
) -> torch.Tensor:
    centers = [int(item) % int(num_classes) for item in center_beams]
    if not centers:
        return torch.empty(0, int(num_classes), dtype=torch.float32)
    sigma = max(float(width), 1e-6)
    temp = max(float(temperature), 1e-6)
    rows = []
    for center in centers:
        distances = torch.tensor(
            [circular_beam_distance(idx, center, num_classes=num_classes) for idx in range(int(num_classes))],
            dtype=torch.float32,
        )
        score = -(distances**2) / (2.0 * sigma * sigma * temp)
        if neighbor_top_k is not None and int(neighbor_top_k) > 0 and int(neighbor_top_k) < int(num_classes):
            keep = set(topk_neighbors(center, num_classes=num_classes, k=int(neighbor_top_k)))
            mask = torch.tensor([idx not in keep for idx in range(int(num_classes))], dtype=torch.bool)
            score[mask] = score[mask] - 1_000.0
        rows.append(score)
    return torch.stack(rows, dim=0)
