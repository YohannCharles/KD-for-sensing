from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping

import torch

from kd_sensing.baselines.gps_window.geometry import (
    angle_to_beam,
    beam_score_kernel,
    circular_mean_degrees,
    circular_velocity_degrees,
    signed_angle_delta_degrees,
    topk_neighbors,
)
from kd_sensing.baselines.gps_window.types import GpsWindowBaselineConfig, GpsWindowPrediction, GpsWindowSample


class CalibrationState:
    def __init__(self, *, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self.majority_beam = 0
        self.transition_delta = 0
        self.transition_scores = torch.zeros(self.num_classes, dtype=torch.float32)
        self.sample_count = 0
        self.beam_direction = 1
        self.beam_offset = 0
        self.beam_mapping_score = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": int(self.sample_count),
            "majority_beam": int(self.majority_beam),
            "transition_delta": int(self.transition_delta),
            "beam_direction": int(self.beam_direction),
            "beam_offset": int(self.beam_offset),
            "beam_mapping_score": float(self.beam_mapping_score),
        }


def build_calibration_state(samples: list[GpsWindowSample], cfg: GpsWindowBaselineConfig) -> CalibrationState:
    state = CalibrationState(num_classes=cfg.num_classes)
    labels: list[int] = []
    deltas: list[int] = []
    transition_hist = torch.ones(int(cfg.num_classes), dtype=torch.float32) * 1e-3
    for sample in samples:
        state.sample_count += 1
        labels.extend(int(item) % int(cfg.num_classes) for item in sample.target_beams if int(item) >= 0)
        if sample.history_beams and sample.target_beams:
            last = int(sample.history_beams[-1]) % int(cfg.num_classes)
            target = int(sample.target_beams[0]) % int(cfg.num_classes)
            delta = (target - last) % int(cfg.num_classes)
            if delta > int(cfg.num_classes) // 2:
                delta -= int(cfg.num_classes)
            deltas.append(int(delta))
            transition_hist[(last + int(delta)) % int(cfg.num_classes)] += 1.0
    if labels:
        state.majority_beam = int(Counter(labels).most_common(1)[0][0])
    if deltas:
        state.transition_delta = int(Counter(deltas).most_common(1)[0][0])
    state.transition_scores = torch.log(transition_hist / transition_hist.sum())
    if cfg.auto_calibrate_beam_mapping:
        direction, offset, score = estimate_beam_mapping(samples, cfg)
        state.beam_direction = direction
        state.beam_offset = offset
        state.beam_mapping_score = score
    else:
        state.beam_direction = 1 if int(cfg.beam_direction) >= 0 else -1
        state.beam_offset = int(cfg.beam_offset) % int(cfg.num_classes)
    return state


def estimate_beam_mapping(samples: list[GpsWindowSample], cfg: GpsWindowBaselineConfig) -> tuple[int, int, float]:
    directions = (1, -1) if cfg.auto_calibrate_beam_direction else (1 if int(cfg.beam_direction) >= 0 else -1,)
    best = (1 if int(cfg.beam_direction) >= 0 else -1, int(cfg.beam_offset) % int(cfg.num_classes), -1.0)
    for direction in directions:
        for offset in range(int(cfg.num_classes)):
            score = _mapping_score(samples, cfg, direction=int(direction), offset=int(offset))
            if score > best[2]:
                best = (int(direction), int(offset), float(score))
    return best


def predict_sample(
    sample: GpsWindowSample,
    cfg: GpsWindowBaselineConfig,
    calibration: CalibrationState | None = None,
) -> GpsWindowPrediction:
    history = sample.available_history[-max(int(cfg.history_window), 1) :]
    diagnostics: dict[str, Any] = {
        "algorithm": cfg.algorithm,
        "beam_offset": int(cfg.beam_offset),
        "effective_beam_direction": int(calibration.beam_direction if calibration is not None else cfg.beam_direction),
        "effective_beam_offset": int(calibration.beam_offset if calibration is not None else cfg.beam_offset),
        "beam_mapping_score": float(calibration.beam_mapping_score if calibration is not None else 0.0),
        "gps_coverage": sample.gps_coverage,
        "history_count": len(history),
        "config": cfg.to_dict(),
    }
    reason = _fallback_reason(sample, cfg, history)
    if reason:
        centers, fallback_status = _fallback_center_beams(sample, cfg, calibration, reason=reason)
    else:
        centers = _predict_center_beams(history, cfg, calibration)
        fallback_status = "none"
    scores = beam_score_kernel(
        centers,
        num_classes=cfg.num_classes,
        width=cfg.score_width,
        temperature=cfg.score_temperature,
        neighbor_top_k=cfg.neighbor_top_k,
    )
    if fallback_status.startswith("transition") and calibration is not None:
        scores = scores + float(cfg.fallback_weight) * calibration.transition_scores.reshape(1, -1)
    topk = tuple(
        topk_neighbors(int(center), num_classes=cfg.num_classes, k=cfg.neighbor_top_k)
        for center in centers
    )
    diagnostics["fallback_status"] = fallback_status
    diagnostics["center_beams"] = list(centers)
    diagnostics["score_width"] = float(cfg.score_width)
    return GpsWindowPrediction(
        sample_id=sample.sample_id,
        scenario=sample.scenario,
        split=sample.split,
        scores=scores,
        topk_beams=topk,
        center_beams=tuple(int(item) for item in centers),
        fallback_status=fallback_status,
        gps_coverage=sample.gps_coverage,
        diagnostics=diagnostics,
    )


def _fallback_reason(
    sample: GpsWindowSample,
    cfg: GpsWindowBaselineConfig,
    history: tuple[dict[str, Any], ...],
) -> str | None:
    if len(history) < max(1, int(cfg.min_history)):
        return "history_window_insufficient"
    if cfg.low_confidence_range is not None:
        last_range = _float_value(history[-1], "relative_range")
        if last_range is None or last_range > float(cfg.low_confidence_range):
            return "geometry_low_confidence"
    if cfg.algorithm == "constant_velocity" and len(history) < 2:
        return "velocity_history_insufficient"
    return None


def _fallback_center_beams(
    sample: GpsWindowSample,
    cfg: GpsWindowBaselineConfig,
    calibration: CalibrationState | None,
    *,
    reason: str,
) -> tuple[list[int], str]:
    mode = str(cfg.fallback or "majority").strip().lower()
    horizon = max(int(cfg.horizon), 1)
    if mode == "last_beam" and sample.history_beams:
        center = int(sample.history_beams[-1]) % int(cfg.num_classes)
        return [center] * horizon, f"last_beam:{reason}"
    if mode == "transition" and sample.history_beams and calibration is not None:
        center = (int(sample.history_beams[-1]) + int(calibration.transition_delta)) % int(cfg.num_classes)
        return [center] * horizon, f"transition:{reason}"
    center = int(calibration.majority_beam if calibration is not None else 0) % int(cfg.num_classes)
    return [center] * horizon, f"majority:{reason}"


def _predict_center_beams(
    history: tuple[dict[str, Any], ...],
    cfg: GpsWindowBaselineConfig,
    calibration: CalibrationState | None = None,
) -> list[int]:
    angles = _predict_angles(history, cfg)
    direction = int(calibration.beam_direction if calibration is not None else cfg.beam_direction)
    offset = int(calibration.beam_offset if calibration is not None else cfg.beam_offset)
    return _angles_to_beams(angles, cfg, direction=direction, offset=offset)


def _predict_angles(history: tuple[dict[str, Any], ...], cfg: GpsWindowBaselineConfig) -> list[float]:
    algorithm = str(cfg.algorithm or "geometry_last").strip().lower()
    horizon = max(int(cfg.horizon), 1)
    if algorithm == "constant_velocity":
        return _constant_velocity_angles(history, horizon=horizon, velocity_decay=float(cfg.velocity_decay))
    base_angle = _last_or_smoothed_angle(history, cfg)
    angular_velocity = circular_velocity_degrees(_history_angles(history)) * float(cfg.angular_velocity_weight)
    return [base_angle + angular_velocity * (idx + 1) for idx in range(horizon)]


def _angles_to_beams(
    angles: list[float],
    cfg: GpsWindowBaselineConfig,
    *,
    direction: int,
    offset: int,
) -> list[int]:
    return [
        angle_to_beam(
            angle,
            num_classes=cfg.num_classes,
            start_degrees=cfg.beam_start_degrees,
            direction=direction,
            beam_offset=offset,
        )
        for angle in angles
    ]


def _last_or_smoothed_angle(history: tuple[dict[str, Any], ...], cfg: GpsWindowBaselineConfig) -> float:
    angles = _history_angles(history)
    if cfg.angle_smoothing or int(cfg.smoothing_window) > 1:
        window = int(cfg.smoothing_window) if int(cfg.smoothing_window) > 1 else len(angles)
        return circular_mean_degrees(angles[-window:])
    return float(angles[-1])


def _constant_velocity_angles(history: tuple[dict[str, Any], ...], *, horizon: int, velocity_decay: float) -> list[float]:
    xs = [_float_value(item, "relative_x", "local_x") for item in history]
    ys = [_float_value(item, "relative_y", "local_y") for item in history]
    points = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(points) < 2:
        angle = _last_or_smoothed_angle(history, GpsWindowBaselineConfig(horizon=horizon))
        return [angle] * int(horizon)
    vx = points[-1][0] - points[-2][0]
    vy = points[-1][1] - points[-2][1]
    x0, y0 = points[-1]
    decay = float(velocity_decay)
    angles = []
    for idx in range(int(horizon)):
        step = idx + 1
        factor = decay**idx
        angles.append(math.degrees(math.atan2(y0 + vy * step * factor, x0 + vx * step * factor)))
    return angles


def _history_angles(history: tuple[dict[str, Any], ...]) -> list[float]:
    angles = []
    for item in history:
        value = _float_value(item, "relative_azimuth", "azimuth")
        if value is not None:
            angles.append(float(value))
    if not angles:
        x = _float_value(history[-1], "relative_x", "local_x")
        y = _float_value(history[-1], "relative_y", "local_y")
        if x is None or y is None:
            raise ValueError("available GPS geometry is missing relative azimuth and relative x/y.")
        angles.append(math.degrees(math.atan2(y, x)))
    return angles


def _float_value(payload: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        try:
            if value is None or value == "":
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _mapping_score(
    samples: list[GpsWindowSample],
    cfg: GpsWindowBaselineConfig,
    *,
    direction: int,
    offset: int,
) -> float:
    total = 0
    score = 0.0
    top1 = 0
    top3 = 0
    for sample in samples:
        history = sample.available_history[-max(int(cfg.history_window), 1) :]
        if _fallback_reason(sample, cfg, history):
            continue
        try:
            beams = _angles_to_beams(_predict_angles(history, cfg), cfg, direction=direction, offset=offset)
        except ValueError:
            continue
        for pred, truth in zip(beams, sample.target_beams):
            if int(truth) < 0:
                continue
            neighbors = topk_neighbors(int(pred), num_classes=cfg.num_classes, k=3)
            linear_distances = [min(abs(int(item) - int(truth)) / 5.0, 1.0) for item in neighbors]
            best_so_far = 1.0
            dba_terms = []
            for distance in linear_distances:
                best_so_far = min(best_so_far, distance)
                dba_terms.append(1.0 - best_so_far)
            score += sum(dba_terms) / max(len(dba_terms), 1)
            top1 += int(int(pred) == int(truth))
            top3 += int(any(int(item) == int(truth) for item in neighbors))
            total += 1
    if total == 0:
        return -1.0
    return float(score / total + 0.05 * top3 / total + 0.02 * top1 / total)


def error_buckets(predictions: torch.Tensor, labels: torch.Tensor, *, num_classes: int) -> dict[str, int]:
    pred = predictions.detach().cpu().reshape(-1).to(torch.long)
    truth = labels.detach().cpu().reshape(-1).to(torch.long)
    buckets = defaultdict(int)
    for item_pred, item_truth in zip(pred.tolist(), truth.tolist()):
        if item_truth < 0:
            continue
        diff = abs(int(item_pred) % int(num_classes) - int(item_truth) % int(num_classes))
        dist = min(diff, int(num_classes) - diff)
        if dist == 0:
            buckets["exact"] += 1
        elif dist <= 2:
            buckets["near_1_2"] += 1
        elif dist <= 5:
            buckets["near_3_5"] += 1
        else:
            buckets["far_gt5"] += 1
    return dict(buckets)
