#!/usr/bin/env python3
"""Independently analyze local circular Beam transport against fixed baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kd_sensing.baselines.prototype_decision_adapter import MASKS, write_json
from kd_sensing.utils.checkpoint import load_torch_payload
from analyze_full_pool_adba_surrogate import _recompute, _summary


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _circular_convolution(values: np.ndarray, kernel: np.ndarray, radius: int) -> np.ndarray:
    return sum(weight * np.roll(values, shift) for weight, shift in zip(kernel, range(-radius, radius + 1)))


def _kernel_audit(checkpoint: Path, metrics_path: Path) -> dict[str, Any]:
    state = load_torch_payload(checkpoint, map_location="cpu")["state_dict"]
    raw = state.get("transport_kernel_logits")
    identity_bias = state.get("transport_identity_bias")
    if not torch.is_tensor(raw) or not torch.is_tensor(identity_bias):
        raise ValueError("Circular transport checkpoint is missing local-kernel state.")
    if raw.shape != identity_bias.shape or raw.ndim != 2 or raw.shape[0] != 4 or raw.shape[1] % 2 != 1:
        raise ValueError("Circular transport kernel shape is invalid.")
    radius = (raw.shape[1] - 1) // 2
    kernels = torch.softmax(raw.float() + identity_bias.float(), dim=-1).numpy()
    shifts = np.arange(-radius, radius + 1, dtype=np.int64)
    rows = []
    for index, modality in enumerate(("image", "radar", "gps", "lidar")):
        kernel = kernels[index]
        rows.append(
            {
                "modality": modality,
                "shifts": shifts.tolist(),
                "probabilities": kernel.tolist(),
                "mass": float(kernel.sum()),
                "mass_error": float(abs(kernel.sum() - 1.0)),
                "mean_shift": float(np.dot(kernel, shifts)),
                "entropy": float(-(kernel * np.log(np.maximum(kernel, np.finfo(np.float32).tiny))).sum()),
            }
        )
    composed = {}
    max_composed_mass_error = 0.0
    for key, _, raw_mask in MASKS:
        probability = np.zeros(64, dtype=np.float64)
        probability[0] = 1.0
        for modality_index, available in enumerate(raw_mask):
            if not available:
                probability = _circular_convolution(probability, kernels[modality_index], radius)
        mass_error = float(abs(probability.sum() - 1.0))
        max_composed_mass_error = max(max_composed_mass_error, mass_error)
        composed[key] = {"mass": float(probability.sum()), "mass_error": mass_error}
    reported = _read(metrics_path).get("transport_kernel_audit")
    if not isinstance(reported, dict):
        raise ValueError("Circular transport metrics are missing their kernel audit.")
    reported_rows = reported.get("kernels")
    if not isinstance(reported_rows, list) or len(reported_rows) != len(rows):
        raise ValueError("Circular transport run-level kernel audit is malformed.")
    report_difference = max(
        float(np.max(np.abs(np.asarray(left["probabilities"]) - np.asarray(right["probabilities"]))))
        for left, right in zip(rows, reported_rows)
    )
    return {
        "radius": radius,
        "kernels": rows,
        "composed_by_mask": composed,
        "max_kernel_mass_error": max(row["mass_error"] for row in rows),
        "max_composed_mass_error": max_composed_mass_error,
        "run_audit_max_abs_difference": report_difference,
    }


def analyze(root: Path) -> dict[str, Any]:
    run_root = root / "circular_transport"
    directories = {
        "a0": root / "stage2/a0",
        "mask_mlp": root / "adba_surrogate/b1",
        "factorized_bias": run_root / "factorized_all_seen",
        "circular_transport": run_root / "circular_transport",
    }
    reference_ids: dict[str, np.ndarray] = {}
    recomputed: dict[str, dict[str, Any]] = {}
    maximum = 0.0
    for key, directory in directories.items():
        recomputed[key], difference = _recompute(directory, reference_ids)
        maximum = max(maximum, difference)
    summaries = {key: _summary(value) for key, value in recomputed.items()}
    transport_audit = _kernel_audit(
        run_root / "circular_transport/checkpoints/last.pth",
        run_root / "circular_transport/metrics.json",
    )
    maximum = max(maximum, transport_audit["run_audit_max_abs_difference"])
    if maximum > 1e-7:
        raise ValueError(f"Independent metric or kernel-audit difference exceeds 1e-7: {maximum}")
    transport = summaries["circular_transport"]
    factorized = summaries["factorized_bias"]
    a0 = summaries["a0"]
    criteria = {
        "adba_above_a0": transport["adba"] > a0["adba"],
        "adba_above_factorized": transport["adba"] > factorized["adba"],
        "mae_not_worse_than_a0": transport["mae"] <= a0["mae"],
    }
    payload = {
        "schema_version": 1,
        "outer_test_accessed": False,
        "independent_metric_max_abs_difference": maximum,
        "summaries": summaries,
        "deltas": {
            key: {
                "vs_a0": {metric: summaries[key][metric] - a0[metric] for metric in summaries[key]},
                "vs_factorized": {metric: summaries[key][metric] - factorized[metric] for metric in summaries[key]},
                "vs_mask_mlp": {metric: summaries[key][metric] - summaries["mask_mlp"][metric] for metric in summaries[key]},
            }
            for key in ("factorized_bias", "circular_transport")
        },
        "transport_kernel_audit": transport_audit,
        "success_criteria": {**criteria, "status": "passed" if all(criteria.values()) else "not_met"},
        "recomputed": recomputed,
    }
    write_json(run_root / "independent_recompute.json", payload)
    fields = ("method", "top1", "top3", "top5", "within3", "mae", "adba", "radar_gps_adba", "no_image_adba", "no_lidar_adba")
    with (run_root / "all_seen_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in ("a0", "factorized_bias", "mask_mlp", "circular_transport"):
            writer.writerow({"method": key, **summaries[key]})
    print(json.dumps({"status": "completed", "success_criteria": payload["success_criteria"]}, indent=2))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    analyze(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
