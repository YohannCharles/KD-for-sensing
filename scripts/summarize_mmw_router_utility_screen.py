#!/usr/bin/env python3
"""Summarize fixed-mask and Oracle-Gap evidence for the Router utility screen."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from launch_mmw_router_utility_screen import CANDIDATES, PROTOCOL_ID


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/mmw_router_expected_utility_screen_v3")
    args = parser.parse_args()
    summarize(Path(args.root).resolve())
    return 0


def summarize(root: Path) -> None:
    manifest = json.loads((root / "training_manifest_router_utility_seed1.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL_ID or any(
        job.get("oracle_gap_status") != "done" for job in manifest.get("jobs", [])
    ):
        raise ValueError("Router utility screen is incomplete or protocol-mismatched.")
    rows = []
    for candidate in CANDIDATES:
        fixed = _read_csv(root / "eval_inner" / candidate / "metrics.csv")
        oracle = json.loads((root / "oracle_gap" / candidate / "summary.json").read_text(encoding="utf-8"))
        learned = {
            item["condition"]: item
            for item in oracle["condition_summary"]
            if item["fusion"] == "learned"
        }
        weights = oracle["router_weight_response"]
        corrupt = [value for key, value in learned.items() if key != "clean"]
        rows.append(
            {
                "candidate": candidate,
                "fixed_mask_adba_mean": _mean(fixed, "adba"),
                "fixed_mask_top1_mean": _mean(fixed, "top1"),
                "oracle_clean_normalized_gain": learned["clean"]["normalized_gain"],
                "oracle_corrupt_normalized_gain_mean": statistics.fmean(
                    float(item["normalized_gain"]) for item in corrupt
                ),
                "oracle_corrupt_gap_closure_mean": statistics.fmean(
                    float(item["normalized_gap_closure_normalized_gain"]) for item in corrupt
                ),
                "oracle_corrupt_soft_regret_mean": statistics.fmean(
                    float(item["router_soft_oracle_regret"]) for item in corrupt
                ),
                "monotonic_sensor_count": sum(bool(item["mean_weight_monotonic"]) for item in weights),
                "gps_clean_to_s3_weight_delta": next(
                    float(item["severity3_weight"]) - float(item["clean_weight"])
                    for item in weights
                    if item["affected_modality"] == "gps"
                ),
            }
        )
    _write_csv(root / "router_utility_screen_summary.csv", rows)
    best = max(rows, key=lambda item: (item["monotonic_sensor_count"], item["oracle_corrupt_normalized_gain_mean"]))
    lines = [
        "# Expected-Utility Router Screen v3 (seed1, inner-only)",
        "",
        "该表仅用于开发集筛选；ADBA 为固定 mask 主指标，Oracle Gap 的 normalized gain、regret 与权重单调性用于机制判断。",
        "",
        "| Candidate | Fixed ADBA | Fixed Top1 | Clean gain | Corrupt gain | Gap closure | Regret | Monotonic | GPS Δw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['fixed_mask_adba_mean']:.3f} | {row['fixed_mask_top1_mean']:.3f} | "
            f"{row['oracle_clean_normalized_gain']:.4f} | {row['oracle_corrupt_normalized_gain_mean']:.4f} | "
            f"{row['oracle_corrupt_gap_closure_mean']:.3f} | {row['oracle_corrupt_soft_regret_mean']:.4f} | "
            f"{row['monotonic_sensor_count']}/4 | {row['gps_clean_to_s3_weight_delta']:+.4f} |"
        )
    lines.extend(["", f"机械排序首位：`{best['candidate']}`。正式选择仍需结合对 CurrentControl 的差值和后续 seed。", ""])
    (root / "ROUTER_UTILITY_SCREEN.md").write_text("\n".join(lines), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: list[dict[str, str]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
