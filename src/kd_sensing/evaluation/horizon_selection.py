from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def metric_horizons_from_config(cfg: dict[str, Any], *, num_pred: int) -> tuple[int, ...]:
    evaluation_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), dict) else {}
    raw = evaluation_cfg.get("metric_horizons", evaluation_cfg.get("aggregate_horizons"))
    return normalize_metric_horizons(raw, num_pred=num_pred, field_name="evaluation.metric_horizons")


def metric_horizons_from_metrics(metrics: dict[str, Any], *, num_pred: int) -> tuple[int, ...]:
    raw = metrics.get("metric_horizons")
    return normalize_metric_horizons(raw, num_pred=num_pred, field_name="metric_horizons")


def normalize_metric_horizons(raw: Any, *, num_pred: int, field_name: str) -> tuple[int, ...]:
    if raw is None or raw is False:
        return tuple(range(1, int(num_pred) + 1))
    if isinstance(raw, str):
        items = _items_from_string(raw)
    elif isinstance(raw, Iterable) and not isinstance(raw, (bytes, bytearray, dict)):
        items = list(raw)
    else:
        items = [raw]

    horizons: list[int] = []
    for item in items:
        if item is None or item == "":
            continue
        try:
            horizon = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must contain integer 1-based horizons, got {item!r}.") from exc
        if horizon not in horizons:
            horizons.append(horizon)

    if not horizons:
        return tuple(range(1, int(num_pred) + 1))

    invalid = [horizon for horizon in horizons if horizon < 1 or horizon > int(num_pred)]
    if invalid:
        raise ValueError(
            f"{field_name} contains horizons outside 1..{int(num_pred)}: {invalid}. "
            "Regenerate sequence CSVs and set model.num_pred/data.dataset.num_pred large enough."
        )
    return tuple(horizons)


def horizon_indices(horizons: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(horizon) - 1 for horizon in horizons)


def selected_horizon_mean(values: Any, totals: Any, *, horizons: Iterable[int] | None = None) -> float:
    values_list = _as_float_list(values)
    totals_list = _as_float_list(totals)
    length = min(len(values_list), len(totals_list))
    if length == 0:
        return 0.0
    selected = set(horizon_indices(horizons or range(1, length + 1)))
    valid_values = [
        values_list[index]
        for index in range(length)
        if totals_list[index] > 0 and index in selected
    ]
    if not valid_values:
        return 0.0
    return float(sum(valid_values) / len(valid_values))


def aggregate_topk_and_dba(metrics: dict[str, Any], *, topk_values: Iterable[int] = (1, 3, 5)) -> dict[str, float]:
    total = metrics.get("total", [])
    horizons = metric_horizons_from_metrics(metrics, num_pred=len(total))
    topk = metrics.get("topk", {})
    result: dict[str, float] = {}
    for k in topk_values:
        values = topk.get(str(k), topk.get(k, [])) if isinstance(topk, dict) else []
        result[f"top{k}"] = selected_horizon_mean(values, total, horizons=horizons)
    result["dba"] = selected_horizon_mean(metrics.get("dba", []), total, horizons=horizons)
    result["adba"] = result["dba"]
    return result


def metric_horizon_source_from_config(cfg: dict[str, Any]) -> str:
    evaluation_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), dict) else {}
    if "metric_horizons" in evaluation_cfg:
        return "evaluation.metric_horizons"
    if "aggregate_horizons" in evaluation_cfg:
        return "evaluation.aggregate_horizons"
    return "default_all_horizons"


def _as_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        return [float(values)]
    return [float(value) for value in values]


def _items_from_string(raw: str) -> list[str]:
    text = raw.strip()
    if not text or text.lower() in {"all", "none", "null"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [item.strip() for item in text.replace(";", ",").split(",")]


__all__ = [
    "aggregate_topk_and_dba",
    "horizon_indices",
    "metric_horizon_source_from_config",
    "metric_horizons_from_config",
    "metric_horizons_from_metrics",
    "normalize_metric_horizons",
    "selected_horizon_mean",
]
