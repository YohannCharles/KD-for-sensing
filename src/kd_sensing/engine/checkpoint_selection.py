import math
from typing import Any

SUPPORTED_SELECTION_METRICS = {"val_acc", "avg_missing_top1", "worst_pattern_top1"}


def model_selection_enabled(cfg: dict[str, Any]) -> bool:
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    raw = training_cfg.get("model_selection")
    if isinstance(raw, dict):
        return bool(raw.get("enabled", True))
    if raw is None:
        return bool(training_cfg.get("use_early_stopping", True))
    return bool(raw)


def config_declares_independent_validation(cfg: dict[str, Any]) -> bool:
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    internal = data_cfg.get("validation_from_train")
    if isinstance(internal, dict):
        if bool(internal.get("enabled", False)):
            return True
    elif bool(internal):
        return True

    dataset_cfg = data_cfg.get("dataset", {}) if isinstance(data_cfg.get("dataset"), dict) else {}
    protocol = str(dataset_cfg.get("split_protocol") or "").strip().lower()
    if protocol in {
        "stratified_80_10_10",
        "deepsense6g_2604_stratified_80_10_10",
        "2604_stratified_80_10_10",
    }:
        return True

    domains = dataset_cfg.get("domains")
    if isinstance(domains, list) and domains:
        return all(isinstance(domain, dict) and _has_distinct_validation_csv(domain) for domain in domains)
    return _has_distinct_validation_csv(dataset_cfg)


def _has_distinct_validation_csv(dataset_cfg: dict[str, Any]) -> bool:
    validation_csv = dataset_cfg.get("val_csv_name") or dataset_cfg.get("validation_csv_name")
    if not validation_csv:
        return False
    test_csv = dataset_cfg.get("test_csv_name")
    return not test_csv or str(validation_csv) != str(test_csv)


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
