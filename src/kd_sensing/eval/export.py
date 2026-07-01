import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_COLUMNS = [
    "pattern",
    "mask",
    "num_samples",
    "sample_count",
    "count",
    "loss",
    "top1",
    "top3",
    "top5",
    "adba",
    "mae",
    "mean_confidence",
    "mean_global_reliability",
    "mean_global_reliability_correct",
    "mean_global_reliability_wrong",
    "mean_modality_reliability",
    "mean_available_modality_reliability",
    "ece",
]


def save_results_csv(results: list[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = _columns(results)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in columns})


def save_results_json(results: list[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def save_results_markdown(results: list[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(format_results_markdown(results), encoding="utf-8")


def format_results_markdown(results: list[dict]) -> str:
    columns = [
        "pattern",
        "mask",
        "top1",
        "top3",
        "top5",
        "adba",
        "mae",
        "mean_confidence",
        "mean_global_reliability",
        "mean_available_modality_reliability",
        "ece",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in results:
        lines.append("| " + " | ".join(_format_cell(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _columns(results: list[dict[str, Any]]) -> list[str]:
    extras = sorted({key for row in results for key in row if key not in DEFAULT_COLUMNS})
    return DEFAULT_COLUMNS + extras


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
