from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def compare_adapter_to_source(
    *,
    target_scene: Any,
    budget: Any,
    seed: Any,
    variant: str,
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    baseline_variant: str = "v1_hierarchical",
) -> dict[str, Any]:
    base = {
        "comparison": "adapter_vs_source_only",
        "target_scene": target_scene,
        "budget": budget,
        "seed": seed,
        "baseline_variant": baseline_variant,
        "candidate_variant": variant,
    }
    missing = _missing_comparison_inputs({baseline_variant: baseline, variant: candidate})
    if missing:
        return {**base, "status": "inconclusive", "missing": missing}
    deltas = _accuracy_deltas(candidate, baseline)
    return {
        **base,
        "status": "complete",
        "accuracy_deltas": deltas,
        "efficiency": _efficiency_summary(candidate),
        "candidate_better_than_source_only": _is_better(deltas),
    }


def compare_proto_to_full(
    *,
    target_scene: Any,
    budget: Any,
    seed: Any,
    proto: dict[str, Any] | None,
    full: dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "comparison": "adapter_proto_vs_full_finetune",
        "target_scene": target_scene,
        "budget": budget,
        "seed": seed,
        "candidate_variant": "v5_adapter_proto",
        "baseline_variant": "v6_full_finetune",
    }
    missing = _missing_comparison_inputs({"v5_adapter_proto": proto, "v6_full_finetune": full})
    if missing:
        return {**base, "status": "inconclusive", "missing": missing}
    deltas = _accuracy_deltas(proto, full)
    trainable_delta = _numeric_delta(proto, full, "trainable_ratio")
    time_delta = _numeric_delta(proto, full, "adaptation_time_seconds")
    return {
        **base,
        "status": "complete",
        "accuracy_deltas": deltas,
        "trainable_ratio_delta": trainable_delta,
        "adaptation_time_seconds_delta": time_delta,
        "adapter_proto_better_than_full_finetune": _is_better(deltas)
        and (trainable_delta is None or trainable_delta <= 0)
        and (time_delta is None or time_delta <= 0),
    }


def _missing_comparison_inputs(items: Mapping[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
    missing = []
    for variant, row in items.items():
        if row is None:
            missing.append({"variant": variant, "reason": "missing_run"})
            continue
        if row.get("run_status") != "completed":
            missing.append({"variant": variant, "reason": row.get("failure_reason") or "run_not_completed"})
            continue
        if bool(row.get("main_conclusion_eligible", True)) is False:
            missing.append(
                {
                    "variant": variant,
                    "reason": "run_excluded_from_main_conclusion",
                    "eligibility_reasons": list(row.get("eligibility_reasons", []) or []),
                    "run_path": row.get("metrics_path"),
                }
            )
            continue
        if all(row.get(metric) is None for metric in ("top1", "top3", "top5", "coarse_accuracy", "fine_accuracy")):
            missing.append({"variant": variant, "reason": "metrics_missing"})
    return missing


def _accuracy_deltas(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        metric: _numeric_delta(candidate, baseline, metric)
        for metric in (
            "top1",
            "top3",
            "top5",
            "coarse_accuracy",
            "fine_accuracy",
            "radio_semantic_accuracy",
            "path_semantic_accuracy",
        )
    }


def _numeric_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any], metric: str) -> float | None:
    lhs = candidate.get(metric)
    rhs = baseline.get(metric)
    if not isinstance(lhs, (int, float)) or not isinstance(rhs, (int, float)):
        return None
    return float(lhs) - float(rhs)


def _is_better(deltas: Mapping[str, float | None]) -> bool:
    available = [value for value in deltas.values() if value is not None]
    return bool(available) and all(value >= 0 for value in available) and any(value > 0 for value in available)


def _efficiency_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trainable_ratio": row.get("trainable_ratio"),
        "adaptation_time_seconds": row.get("adaptation_time_seconds"),
        "adaptation_time_per_epoch": row.get("adaptation_time_per_epoch"),
    }


__all__ = [
    "compare_adapter_to_source",
    "compare_proto_to_full",
]
