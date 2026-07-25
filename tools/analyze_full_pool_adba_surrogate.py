#!/usr/bin/env python3
"""Independently recompute Full-pool ADBA-surrogate Adapter results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.baselines.prototype_decision_adapter import MASKS, aggregate, numpy_metrics, write_json


RUNS = {"b1": "a1", "b4": "a4", "b6": "a6", "b7": "a7"}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric_max_difference(left: Any, right: Any) -> float:
    if isinstance(left, dict):
        return max((_numeric_max_difference(value, right[key]) for key, value in left.items()), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0


def _recompute(directory: Path, reference_ids: dict[str, np.ndarray]) -> tuple[dict[str, Any], float]:
    report = _read(directory / "metrics.json")
    internal = {row["key"]: row for row in report["mask_metrics"]}
    rows = []
    maximum = 0.0
    masks = {}
    for key, label, raw_mask in MASKS:
        with np.load(directory / "predictions" / f"{key}.npz") as payload:
            sample_ids = payload["sample_id"]
            if key in reference_ids and not np.array_equal(reference_ids[key], sample_ids):
                raise ValueError(f"Paired sample identity mismatch: {directory.name}/{key}")
            reference_ids.setdefault(key, sample_ids.copy())
            metrics = numpy_metrics(payload["new_logits"], payload["ground_truth"])
        maximum = max(
            maximum,
            *(abs(metrics[name] - float(internal[key]["new"][name])) for name in metrics),
        )
        masks[key] = metrics
        rows.append({"key": key, "label": label, "mask": list(raw_mask), "new": metrics})
    grouped = aggregate(rows)
    maximum = max(
        maximum,
        _numeric_max_difference(grouped["aggregates"], report["aggregates"]),
        _numeric_max_difference(grouped["retention"], report["retention"]),
        abs(float(grouped["spa_macro"]) - float(report["spa_macro"])),
    )
    equivalence = report["full_equivalence"]
    if (
        float(equivalence["max_abs_logit_diff"]) > 1e-7
        or int(equivalence["argmax_mismatch_count"]) != 0
        or float(equivalence["top1_difference"]) != 0.0
    ):
        raise ValueError(f"Full equivalence failed: {directory}")
    return {"masks": masks, **grouped, "training": report["training"], "loss_profile": report.get("loss_profile")}, maximum


def _summary(result: dict[str, Any]) -> dict[str, float]:
    all14 = result["aggregates"]["all14_macro"]
    return {
        "top1": float(all14["top1"]),
        "top3": float(all14["top3"]),
        "top5": float(all14["top5"]),
        "within3": float(all14["within3"]),
        "mae": float(all14["mae"]),
        "adba": float(all14["adba"]),
        "radar_gps_adba": float(result["masks"]["radar_gps"]["adba"]),
        "no_image_adba": float(result["masks"]["no_image"]["adba"]),
        "no_lidar_adba": float(result["masks"]["no_lidar"]["adba"]),
    }


def analyze(root: Path) -> dict[str, Any]:
    reference_ids: dict[str, np.ndarray] = {}
    directories = {
        "a0": root / "stage2/a0",
        **{method: root / "stage2" / method for method in RUNS.values()},
        **{key: root / "adba_surrogate" / key for key in RUNS},
    }
    recomputed = {}
    maximum = 0.0
    for key, directory in directories.items():
        recomputed[key], difference = _recompute(directory, reference_ids)
        maximum = max(maximum, difference)
    if maximum > 1e-7:
        raise ValueError(f"Independent metric difference exceeds 1e-7: {maximum}")
    summaries = {key: _summary(value) for key, value in recomputed.items()}
    deltas = {}
    for key, ce_method in RUNS.items():
        deltas[key] = {
            "vs_a0": {metric: summaries[key][metric] - summaries["a0"][metric] for metric in summaries[key]},
            "vs_ce_counterpart": {
                metric: summaries[key][metric] - summaries[ce_method][metric] for metric in summaries[key]
            },
        }
    payload = {
        "schema_version": 1,
        "outer_test_accessed": False,
        "independent_metric_max_abs_difference": maximum,
        "summaries": summaries,
        "deltas": deltas,
        "b6_minus_b7": {metric: summaries["b6"][metric] - summaries["b7"][metric] for metric in summaries["b6"]},
        "recomputed": recomputed,
    }
    output_root = root / "adba_surrogate"
    write_json(output_root / "independent_recompute.json", payload)
    fields = ("method", "top1", "top3", "top5", "within3", "mae", "adba", "radar_gps_adba", "no_image_adba", "no_lidar_adba")
    with (output_root / "main_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in ("a0", "a1", "b1", "a4", "b4", "a6", "b6", "a7", "b7"):
            writer.writerow({"method": key.upper(), **summaries[key]})
    print(json.dumps({"status": "completed", "max_abs_difference": maximum, "b6_minus_b7": payload["b6_minus_b7"]}, indent=2))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    analyze(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
