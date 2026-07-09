#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import yaml


SUMMARY_COLUMNS = [
    "experiment",
    "seed",
    "history_window",
    "prediction_window",
    "temporal_missing_mode",
    "temporal_missing_prob",
    "full",
    "avg_missing",
    "missing_image",
    "drop1",
    "drop2",
    "drop3",
    "single_modality_mean",
    "radar_only",
    "within3",
    "MAE",
    "temporal_available_rate",
    "modality_temporal_available_rate",
    "num_all_missing_fixed",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize temporal missing v1 outputs.")
    parser.add_argument("--root", default="outputs/temporal_missing_v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    root = Path(build_parser().parse_args(argv).root)
    rows = [_row_for_run(path, root) for path in sorted(root.glob("*/seed*")) if path.is_dir()]
    rows = [row for row in rows if row]
    _write_csv(root / "summary.csv", rows, SUMMARY_COLUMNS)
    _write_csv(root / "temporal_mask_stats.csv", rows, ["experiment", "seed", "temporal_available_rate", "modality_temporal_available_rate", "num_all_missing_fixed"])
    (root / "summary.md").write_text(_markdown(rows), encoding="utf-8")
    print(f"wrote {root / 'summary.csv'}")
    print(f"wrote {root / 'summary.md'}")
    print(f"wrote {root / 'temporal_mask_stats.csv'}")
    return 0


def _row_for_run(run_dir: Path, root: Path) -> dict:
    cfg = _load_yaml(run_dir / "final_config.yaml")
    temporal = cfg.get("temporal_missing", {}) if isinstance(cfg.get("temporal_missing"), dict) else {}
    metrics = _load_json(run_dir / "metrics.json") or _load_json(run_dir / "test_report.json")
    eval_rows = _load_eval_rows(run_dir)
    row = {column: "" for column in SUMMARY_COLUMNS}
    row["experiment"] = run_dir.parent.name
    row["seed"] = run_dir.name.removeprefix("seed")
    row["history_window"] = temporal.get("history_window", cfg.get("model", {}).get("history_window", ""))
    row["prediction_window"] = temporal.get("prediction_window", cfg.get("model", {}).get("prediction_window", ""))
    row["temporal_missing_mode"] = temporal.get("mode", "none")
    row["temporal_missing_prob"] = temporal.get("prob", 0.0)
    for pattern, target in (
        ("full", "full"),
        ("avg_missing", "avg_missing"),
        ("missing_image", "missing_image"),
        ("radar_only", "radar_only"),
    ):
        row[target] = _pattern_metric(eval_rows, pattern, "top1")
    row["drop1"] = _mean_patterns(eval_rows, ["missing_image", "missing_radar", "missing_lidar", "missing_gps"], "top1")
    row["drop2"] = _mean_missing_count(eval_rows, 2)
    row["drop3"] = _mean_missing_count(eval_rows, 3)
    row["single_modality_mean"] = _mean_patterns(eval_rows, ["image_only", "radar_only", "lidar_only", "gps_only"], "top1")
    row["within3"] = _first_metric(metrics, ("within_3", "top3", "val_beam_top3"))
    row["MAE"] = _first_metric(metrics, ("mae", "val_beam_mae", "avg_missing_mae"))
    stats = _temporal_stats(run_dir)
    row.update(stats)
    return row


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_eval_rows(run_dir: Path) -> list[dict]:
    candidates = list((run_dir / "eval").glob("*_missing_patterns.csv")) + [run_dir / "eval_matrix.csv"]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as f:
                return [dict(row) for row in csv.DictReader(f)]
    return []


def _pattern_metric(rows: list[dict], pattern: str, metric: str) -> str:
    for row in rows:
        if str(row.get("pattern", "")) == pattern:
            return row.get(metric, "")
    return ""


def _mean_patterns(rows: list[dict], patterns: list[str], metric: str) -> str:
    values = [_float(_pattern_metric(rows, pattern, metric)) for pattern in patterns]
    values = [value for value in values if value is not None]
    return "" if not values else f"{sum(values) / len(values):.6g}"


def _mean_missing_count(rows: list[dict], count: int) -> str:
    values = []
    for row in rows:
        mask = str(row.get("mask", ""))
        if mask and mask != "aggregate":
            missing = sum(1 for item in mask.split(",") if item.strip() == "0")
            value = _float(row.get("top1"))
            if missing == count and value is not None:
                values.append(value)
    return "" if not values else f"{sum(values) / len(values):.6g}"


def _first_metric(metrics: dict, names: tuple[str, ...]) -> str:
    for name in names:
        value = metrics.get(name)
        if value is not None:
            return value
    return ""


def _temporal_stats(run_dir: Path) -> dict:
    stats = {"temporal_available_rate": "", "modality_temporal_available_rate": "", "num_all_missing_fixed": ""}
    for path in run_dir.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        text = json.dumps(payload)
        for key in list(stats):
            if key in text and stats[key] == "":
                stats[key] = _find_scalar(payload, key)
    return stats


def _find_scalar(value, key: str):
    if isinstance(value, dict):
        if key in value and not isinstance(value[key], (dict, list)):
            return value[key]
        for item in value.values():
            found = _find_scalar(item, key)
            if found != "":
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_scalar(item, key)
            if found != "":
                return found
    return ""


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _markdown(rows: list[dict]) -> str:
    lines = ["# Temporal Missing v1 Summary", ""]
    if not rows:
        lines.append("- 暂无可汇总 run。")
        return "\n".join(lines) + "\n"
    effective = all(str(row.get("history_window")) == "5" and str(row.get("prediction_window")) == "1" for row in rows)
    injected = any(str(row.get("temporal_missing_mode")) != "none" for row in rows)
    lines.append(f"- history_window=5 / prediction_window=1 生效: {effective}")
    lines.append(f"- temporal missing 正确注入: {injected}")
    lines.append("- frame / modality-frame / block 难度比较: 需要完整 eval matrix 后按 avg_missing 或 MAE 排序。")
    lines.append("- 时序缺失是否放大模态缺失问题: 查看 tm0 与 tm1/tm2/tm3 的 avg_missing 差值。")
    lines.append("- C2 temporal missing 鲁棒性: 需要对应 checkpoint/eval rows 后判断。")
    lines.extend(["", "| experiment | seed | mode | full | avg_missing | MAE |", "| --- | ---: | --- | ---: | ---: | ---: |"])
    for row in rows:
        lines.append(
            f"| {row.get('experiment','')} | {row.get('seed','')} | {row.get('temporal_missing_mode','')} | "
            f"{row.get('full','')} | {row.get('avg_missing','')} | {row.get('MAE','')} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
