from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class G2DDiagnosticsAccumulator:
    def __init__(self, *, num_pred: int, horizon_names: list[str]):
        self.num_pred = int(num_pred)
        self.horizon_names = list(horizon_names)
        self.count = 0
        self.loss_sums: dict[str, float] = {}
        self.teacher_confidence_sums: dict[str, dict[str, float]] = {}
        self.student_confidence_sums: dict[str, dict[str, float]] = {}
        self.ratio_sums: dict[str, dict[str, float]] = {}
        self.latest_ranking: dict[str, list[str]] = {}
        self.latest_active_modalities: list[str] = []

    def update(self, diagnostics: dict[str, Any]) -> None:
        self.count += 1
        for key, value in (diagnostics.get("loss") or {}).items():
            if isinstance(value, (int, float)):
                self.loss_sums[key] = self.loss_sums.get(key, 0.0) + float(value)
        _accumulate_nested(self.teacher_confidence_sums, diagnostics.get("teacher_confidence"))
        _accumulate_nested(self.student_confidence_sums, diagnostics.get("student_branch_confidence"))
        _accumulate_nested(self.ratio_sums, diagnostics.get("confidence_ratio"))
        ranking = diagnostics.get("modality_ranking_weak_to_strong")
        if isinstance(ranking, dict):
            self.latest_ranking = {str(key): [str(item) for item in value] for key, value in ranking.items()}
        active = diagnostics.get("active_modalities")
        if isinstance(active, (list, tuple)):
            self.latest_active_modalities = [str(item) for item in active]

    def finalize(self, *, epoch: int) -> dict[str, Any]:
        denom = max(self.count, 1)
        payload = {
            "epoch": int(epoch),
            "num_pred": self.num_pred,
            "horizon_names": list(self.horizon_names),
            "teacher_confidence": _mean_nested(self.teacher_confidence_sums, denom),
            "modality_ranking_weak_to_strong": self.latest_ranking,
            "active_modalities": list(self.latest_active_modalities),
            "loss": {key: float(value / denom) for key, value in self.loss_sums.items()},
        }
        student = _mean_nested(self.student_confidence_sums, denom)
        if student:
            payload["student_branch_confidence"] = student
        ratio = _mean_nested(self.ratio_sums, denom)
        if ratio:
            payload["confidence_ratio"] = ratio
        return payload

    def write_epoch(self, output_dir: str | Path, *, epoch: int) -> Path:
        target_dir = Path(output_dir) / "diagnostics"
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"g2d_epoch_{int(epoch)}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.finalize(epoch=epoch), f, indent=2)
        return path


def _accumulate_nested(target: dict[str, dict[str, float]], values: Any) -> None:
    if not isinstance(values, dict):
        return
    for outer_key, item in values.items():
        if not isinstance(item, dict):
            continue
        bucket = target.setdefault(str(outer_key), {})
        for inner_key, value in item.items():
            if isinstance(value, (int, float)):
                bucket[str(inner_key)] = bucket.get(str(inner_key), 0.0) + float(value)


def _mean_nested(values: dict[str, dict[str, float]], denom: int) -> dict[str, dict[str, float]]:
    return {
        outer_key: {inner_key: float(value / denom) for inner_key, value in item.items()}
        for outer_key, item in values.items()
    }


__all__ = ["G2DDiagnosticsAccumulator"]
