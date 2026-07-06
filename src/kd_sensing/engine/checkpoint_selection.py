import math
from typing import Any

SUPPORTED_SELECTION_METRICS = {"val_acc", "avg_missing_top1", "worst_pattern_top1"}


def resolve_checkpoint_selection_metric(cfg: dict[str, Any]) -> str:
    checkpoint_cfg = cfg.get("checkpoint", {}) if isinstance(cfg.get("checkpoint"), dict) else {}
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    raw = checkpoint_cfg.get("selection_metric", training_cfg.get("selection_metric", training_cfg.get("save_best_metric", "val_acc")))
    metric = _normalize_selection_metric(raw)
    if metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(
            "checkpoint selection_metric must be one of "
            f"{sorted(SUPPORTED_SELECTION_METRICS)}, got {raw!r}."
        )
    return metric


def checkpoint_selection_score(epoch_log: dict[str, Any], metric: str, *, val_acc: float | None = None) -> float:
    metric = _normalize_selection_metric(metric)
    if metric == "val_acc":
        return _finite_float(epoch_log.get("val_acc", val_acc), "val_acc")
    if metric == "avg_missing_top1":
        value = _first_float(epoch_log, "avg_missing_top1", "val/subset/avg_missing/top1")
        if value is None:
            value = _mean(_subset_top1_values(epoch_log, include_aggregate=False, exclude_full=True))
        return _finite_float(value, metric)
    if metric == "worst_pattern_top1":
        value = _first_float(epoch_log, "worst_pattern_top1", "val/subset/worst_pattern/top1")
        if value is None:
            values = _subset_top1_values(epoch_log, include_aggregate=False, exclude_full=True)
            value = min(values) if values else None
        return _finite_float(value, metric)
    raise ValueError(f"Unsupported checkpoint selection metric: {metric}")


def select_best_checkpoint_epoch(epoch_logs: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for row in epoch_logs:
        score = checkpoint_selection_score(row, metric)
        if best is None or score > float(best["score"]):
            best = {"epoch": int(row.get("epoch", 0)), "score": score, "metric": _normalize_selection_metric(metric)}
    if best is None:
        raise ValueError("epoch_logs must contain at least one row.")
    return best


def _normalize_selection_metric(raw: Any) -> str:
    metric = str(raw or "val_acc").strip().lower()
    aliases = {
        "val_top1": "val_acc",
        "val_acc_top1": "val_acc",
        "best_val_top1": "val_acc",
        "best_top1": "val_acc",
        "avg_missing": "avg_missing_top1",
        "best_avg_missing_top1": "avg_missing_top1",
        "worst_pattern": "worst_pattern_top1",
    }
    return aliases.get(metric, metric)


def _subset_top1_values(
    epoch_log: dict[str, Any],
    *,
    include_aggregate: bool,
    exclude_full: bool,
) -> list[float]:
    values: list[float] = []
    for key, value in epoch_log.items():
        if not (isinstance(key, str) and key.startswith("val/subset/") and key.endswith("/top1")):
            continue
        name = key.removeprefix("val/subset/").removesuffix("/top1")
        if exclude_full and name in {"all", "full"}:
            continue
        if not include_aggregate and name in {"avg_missing", "balanced", "overall_mean"}:
            continue
        parsed = _float_or_none(value)
        if parsed is not None:
            values.append(parsed)
    nested = epoch_log.get("validation_metrics", {})
    subset_metrics = nested.get("modality_subsets") if isinstance(nested, dict) else None
    if isinstance(subset_metrics, dict):
        for name, metrics in subset_metrics.items():
            if exclude_full and name in {"all", "full"}:
                continue
            if not include_aggregate and name in {"avg_missing", "balanced", "overall_mean"}:
                continue
            if not isinstance(metrics, dict):
                continue
            value = _first_float(metrics, "top1", "val_acc")
            if value is not None:
                values.append(value)
    return values


def _first_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _finite_float(value: Any, name: str) -> float:
    numeric = _float_or_none(value)
    if numeric is None:
        raise ValueError(f"Checkpoint selection metric '{name}' is not available as a finite numeric value.")
    return numeric


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
