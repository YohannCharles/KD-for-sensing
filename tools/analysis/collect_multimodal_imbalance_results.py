#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODALITIES = ("image", "radar", "gps", "lidar", "mmwave")
HORIZONS = ("t+1", "t+2", "t+3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect G2D multimodal imbalance diagnostics into a CSV.")
    parser.add_argument("--outputs-root", default="outputs", help="Root directory containing scene/run outputs.")
    parser.add_argument(
        "--output",
        default="outputs/analysis/multimodal_imbalance_summary.csv",
        help="Output CSV path.",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    outputs_root = Path(args.outputs_root)
    rows = collect_rows(outputs_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames()
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return output_path


def collect_rows(outputs_root: Path) -> list[dict[str, Any]]:
    rows = []
    for diagnostics_dir in outputs_root.glob("**/diagnostics"):
        latest = _latest_g2d_diagnostic(diagnostics_dir)
        if latest is None:
            continue
        run_dir = diagnostics_dir.parent
        diagnostics = _load_json(latest)
        metrics = _load_json(run_dir / "metrics.json")
        train_log = _load_json(run_dir / "train_log.json")
        rows.append(_row(run_dir, diagnostics, metrics, train_log))
    return rows


def _row(run_dir: Path, diagnostics: dict[str, Any], metrics: dict[str, Any], train_log: dict[str, Any]) -> dict[str, Any]:
    scene = run_dir.parent.name if run_dir.parent.name.startswith("scene") else ""
    row: dict[str, Any] = {
        "scene": scene,
        "run_name": run_dir.name,
        "method": _method_from_config(train_log, run_dir.name),
        "top1_t1": metrics.get("val_top1_t1"),
        "top1_t2": metrics.get("val_top1_t2"),
        "top1_t3": metrics.get("val_top1_t3"),
        "top1_avg": metrics.get("val_top1_avg"),
        "top3_avg": metrics.get("val_top3_avg"),
        "top5_avg": metrics.get("val_top5_avg"),
        "ranking_avg": _join(diagnostics.get("modality_ranking_weak_to_strong", {}).get("avg")),
        "ranking_t1": _join(diagnostics.get("modality_ranking_weak_to_strong", {}).get("t+1")),
        "ranking_t2": _join(diagnostics.get("modality_ranking_weak_to_strong", {}).get("t+2")),
        "ranking_t3": _join(diagnostics.get("modality_ranking_weak_to_strong", {}).get("t+3")),
        "final_active_modalities": _join(diagnostics.get("active_modalities")),
    }
    teacher_conf = diagnostics.get("teacher_confidence") or {}
    for modality in MODALITIES:
        values = teacher_conf.get(modality) or {}
        for horizon in HORIZONS:
            suffix = horizon.replace("+", "")
            row[f"teacher_conf_{modality}_{suffix}"] = values.get(horizon)
        row[f"teacher_conf_{modality}_avg"] = values.get("avg")
    return row


def _fieldnames() -> list[str]:
    fields = [
        "scene",
        "run_name",
        "method",
        "top1_t1",
        "top1_t2",
        "top1_t3",
        "top1_avg",
        "top3_avg",
        "top5_avg",
    ]
    for modality in MODALITIES:
        fields.extend(
            [
                f"teacher_conf_{modality}_t1",
                f"teacher_conf_{modality}_t2",
                f"teacher_conf_{modality}_t3",
                f"teacher_conf_{modality}_avg",
            ]
        )
    fields.extend(["ranking_avg", "ranking_t1", "ranking_t2", "ranking_t3", "final_active_modalities"])
    return fields


def _latest_g2d_diagnostic(path: Path) -> Path | None:
    candidates = sorted(path.glob("g2d_epoch_*.json"), key=_epoch_number)
    return candidates[-1] if candidates else None


def _epoch_number(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _method_from_config(train_log: dict[str, Any], fallback: str) -> str:
    cfg = train_log.get("runtime", {}) if isinstance(train_log, dict) else {}
    run_dir = str(cfg.get("run_dir") or "")
    if run_dir:
        return Path(run_dir).name
    return fallback


def _join(values: Any) -> str:
    if not isinstance(values, (list, tuple)):
        return ""
    return "|".join(str(value) for value in values)


if __name__ == "__main__":
    main()
