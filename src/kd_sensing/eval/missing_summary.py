import csv
import json
import math
from pathlib import Path
from typing import Any


_CORE_METRICS = (
    "top1",
    "top3",
    "top5",
    "within_3",
    "mae",
    "adba",
    "nrp",
    "mean_beamforming_gain",
    "beam_loss_db",
)
_LOWER_IS_BETTER = frozenset(("mae", "beam_loss_db", "loss"))


def summarize_missing_patterns(
    results: list[dict[str, Any]],
    *,
    modality_count: int = 4,
) -> dict[str, Any]:
    by_mask: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in results:
        mask = _mask(row.get("mask"), modality_count=modality_count)
        if mask is not None:
            by_mask[mask] = row
    expected = (1 << int(modality_count)) - 1
    groups = {
        "Single": [row for mask, row in by_mask.items() if sum(mask) == 1],
        "Double": [row for mask, row in by_mask.items() if sum(mask) == 2],
        "Triple": [row for mask, row in by_mask.items() if sum(mask) == 3],
    }
    full = next((row for mask, row in by_mask.items() if sum(mask) == modality_count), None)
    incomplete = [row for mask, row in by_mask.items() if 0 < sum(mask) < modality_count]
    summary: dict[str, Any] = {
        "complete": len(by_mask) == expected and full is not None,
        "expected_pattern_count": expected,
        "actual_pattern_count": len(by_mask),
        "Full": _metric_values(full) if full is not None else None,
    }
    for name, rows in groups.items():
        summary[f"{name} Macro"] = _aggregate(rows, mode="macro")
        summary[f"{name} Worst"] = _aggregate(rows, mode="worst")
    summary["All-14 Macro"] = _aggregate(incomplete, mode="macro")
    summary["All-14 Worst"] = _aggregate(incomplete, mode="worst")
    return summary


def save_missing_summary(summary: dict[str, Any], path_prefix: str | Path) -> dict[str, str]:
    prefix = Path(path_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    markdown_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    rows = [
        {"section": section, "metric": metric, "value": value}
        for section, metrics in summary.items()
        if isinstance(metrics, dict)
        for metric, value in metrics.items()
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "metric", "value"))
        writer.writeheader()
        writer.writerows(rows)
    metrics = sorted({row["metric"] for row in rows})
    lines = [
        "| summary | " + " | ".join(metrics) + " |",
        "| --- | " + " | ".join("---" for _ in metrics) + " |",
    ]
    for section, values in summary.items():
        if not isinstance(values, dict):
            continue
        cells = [section] + [_format(values.get(metric)) for metric in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(markdown_path)}


def _aggregate(rows: list[dict[str, Any]], *, mode: str) -> dict[str, float] | None:
    if not rows:
        return None
    result: dict[str, float] = {}
    for metric in _CORE_METRICS:
        values = [float(row[metric]) for row in rows if _finite(row.get(metric))]
        if not values:
            continue
        if mode == "macro":
            result[metric] = sum(values) / len(values)
        elif mode == "worst":
            result[metric] = max(values) if metric in _LOWER_IS_BETTER else min(values)
        else:
            raise ValueError(f"Unknown missing-pattern aggregation mode {mode!r}.")
    return result


def _metric_values(row: dict[str, Any] | None) -> dict[str, float]:
    if row is None:
        return {}
    return {metric: float(row[metric]) for metric in _CORE_METRICS if _finite(row.get(metric))}


def _mask(value: Any, *, modality_count: int) -> tuple[int, ...] | None:
    if not isinstance(value, str):
        return None
    parts = value.split(",")
    if len(parts) != int(modality_count):
        return None
    try:
        mask = tuple(int(part) for part in parts)
    except ValueError:
        return None
    return mask if any(mask) and all(item in {0, 1} for item in mask) else None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _format(value: Any) -> str:
    return "" if value is None else f"{float(value):.6g}"


__all__ = ["save_missing_summary", "summarize_missing_patterns"]
