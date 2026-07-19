#!/usr/bin/env python3
"""Fail-closed pooled summary for DeepSense6G secondary evidence."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("T2", "masktrain_cls", "amber_full", "rmbp_mm", "amr_net_4m")
SEEDS = (1, 2, 3)
METRICS = (
    "adba", "top1", "top3", "top5", "within_1", "within_3", "mae",
    "normalized_gain", "gain_loss_db",
    "spectral_efficiency_ratio_0db", "spectral_efficiency_loss_0db",
    "spectral_efficiency_ratio_10db", "spectral_efficiency_loss_10db",
    "spectral_efficiency_ratio_20db", "spectral_efficiency_loss_20db",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize complete DeepSense6G TWC fixed-mask evidence.")
    parser.add_argument("--root", default="outputs/deepsense6g_twc_secondary_v1")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    root = _path(args.root)
    output = _path(args.output_dir) if args.output_dir else root / "summary"
    result = summarize(root / "eval_fixed", output)
    print(json.dumps({"status": "complete", **result}, indent=2))
    return 0


def summarize(eval_root: Path, output_dir: Path) -> dict[str, Any]:
    units = []
    expected_identity = None
    for method in METHODS:
        for seed in SEEDS:
            path = eval_root / method / f"seed{seed}" / "metrics.csv"
            if not path.is_file():
                raise FileNotFoundError(f"DeepSense6G pooled evidence cell is missing: {path}")
            rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
            if not rows or any(row["coverage_status"] != "complete" for row in rows):
                raise ValueError(f"DeepSense6G pooled evidence cell is empty or partial: {path}")
            if any(row["dataset_scope"] != "deepsense6g_scene31_34_pooled_v1" for row in rows):
                raise ValueError(f"DeepSense6G evidence is not pooled Scene31--34: {path}")
            identity = tuple((row["eval_family"], row["pattern"], row["mask_type"], row["mask_digest"]) for row in rows)
            if expected_identity is None:
                expected_identity = identity
            elif identity != expected_identity:
                raise ValueError(f"DeepSense6G fixed-mask identity differs: {path}")
            units.append((method, seed, rows))
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    conditions = list(expected_identity or ())
    for method, seed, rows in units:
        for index, row in enumerate(rows):
            groups[(method, index)].append(row)
    by_seed_rows = [row for _method, _seed, rows in units for row in rows]
    summary_rows = []
    for (method, index), rows in sorted(groups.items()):
        if len(rows) != len(SEEDS):
            raise ValueError(f"DeepSense6G pooled seed set is incomplete for {method} condition {index}.")
        family, pattern, mask_type, digest = conditions[index]
        payload = {
            "method": method,
            "condition_index": index,
            "eval_family": family,
            "pattern": pattern,
            "mask_type": mask_type,
            "mask_digest": digest,
            "seed_count": len(rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            payload[f"{metric}_mean"] = statistics.fmean(values)
            payload[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "pooled_metrics_by_seed.csv", by_seed_rows)
    _write_csv(output_dir / "pooled_method_summary.csv", summary_rows)
    payload = {
        "schema_version": 1,
        "artifact_kind": "deepsense6g_twc_pooled_summary_v1",
        "dataset_scope": "deepsense6g_scene31_34_pooled_v1",
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "scene_ids": [31, 32, 33, 34],
        "complete_unit_count": len(units),
        "fixed_condition_count": len(conditions),
        "pooled_summary_row_count": len(summary_rows),
        "primary_metric": "adba",
        "secondary_metric": "top1",
        "adba_definition": "progressive_top3_minimum_circular_beam_distance",
        "adba_delta": 5.0,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "# DeepSense6G TWC 次级证据\n\n"
        f"Scene31--34 合并为一个完整数据集；完整覆盖 {len(METHODS)} 方法 x {len(SEEDS)} seeds = {len(units)} 个 checkpoint。"
        "与 MMW 分开汇总，所有 baseline 均按本项目四模态适配范围解释。"
        "主指标为 circular progressive Top-3 ADBA（delta=5），Top-1 为支线。\n",
        encoding="utf-8",
    )
    return {"output_dir": str(output_dir), **payload}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
