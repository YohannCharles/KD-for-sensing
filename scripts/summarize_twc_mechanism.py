#!/usr/bin/env python3
"""Summarize quantitative BPA/router mechanism traces without PCA claims."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize TWC clean/Block80 mechanism traces.")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--protocol-manifest", default="outputs/cache/mmw_twc_outer_v1/protocol_manifest.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = summarize(_path(args.eval_dir), _path(args.protocol_manifest), _path(args.output_dir))
    print(json.dumps({"status": "complete", **result}, indent=2))
    return 0


def summarize(eval_dir: Path, protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cache = json.loads(Path(protocol["fixed_mask_cache"]["path"]).read_text(encoding="utf-8"))
    clean, missing = _condition_indices(cache["conditions"])
    traces = []
    for path in sorted(eval_dir.glob("*/seed*/mechanism_trace.jsonl")):
        method, seed_name = path.parent.parent.name, path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row.update(method=method, seed=int(seed_name.removeprefix("seed")))
            traces.append(row)
    if not traces:
        raise ValueError(f"No mechanism_trace.jsonl files found under {eval_dir}.")
    if any(int(row["condition_index"]) not in {clean, missing} for row in traces):
        raise ValueError("Mechanism trace contains conditions outside clean/canonical Block80.")

    summary_rows = _condition_summary(traces, clean=clean)
    drift_rows = _paired_drift(traces, clean=clean, missing=missing)
    cdf_rows = _cdf_rows(traces, clean=clean)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "mechanism_summary.csv", summary_rows)
    _write_csv(output_dir / "clean_to_missing_drift.csv", drift_rows)
    _write_csv(output_dir / "physical_error_cdf.csv", cdf_rows)
    payload = {
        "schema_version": 1,
        "artifact_kind": "twc_quantitative_mechanism_v1",
        "protocol_manifest": str(protocol_path),
        "clean_condition_index": clean,
        "missing_condition_index": missing,
        "trace_row_count": len(traces),
        "condition_summary": summary_rows,
        "drift_summary": _aggregate_drift(drift_rows),
    }
    (output_dir / "mechanism_summary.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    _plots(cdf_rows, drift_rows, output_dir)
    return {"output_dir": str(output_dir), "trace_rows": len(traces), "paired_rows": len(drift_rows)}


def _condition_indices(conditions: list[dict[str, Any]]) -> tuple[int, int]:
    clean = next(index for index, item in enumerate(conditions) if item["family"] == "whole_modality" and float(item["requested_missing_rate"]) == 0.0)
    missing = next(index for index, item in enumerate(conditions) if item["family"] == "temporal_missing" and item["mask_type"] == "block" and math.isclose(float(item["requested_missing_rate"]), 0.8))
    return clean, missing


def _condition_summary(rows: list[dict[str, Any]], *, clean: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        condition = "clean" if int(row["condition_index"]) == clean else "block80"
        groups[(row["method"], int(row["seed"]), condition)].append(row)
    result = []
    for (method, seed, condition), selected in sorted(groups.items()):
        errors = [_distance(row["prediction"], row["target"]) for row in selected]
        alignments = [float(row["router_oracle_aligned"]) for row in selected if "router_oracle_aligned" in row]
        result.append(
            {
                "method": method,
                "seed": seed,
                "condition": condition,
                "sample_count": len(selected),
                "mean_physical_codebook_error": statistics.fmean(errors),
                "far_error_gt3": statistics.fmean(error > 3 for error in errors),
                "far_error_gt5": statistics.fmean(error > 5 for error in errors),
                "within_neighbor_rate": statistics.fmean(error <= 1 for error in errors),
                "prototype_neighbor_margin": statistics.fmean(float(row["prototype_neighbor_margin"]) for row in selected),
                "router_oracle_alignment": statistics.fmean(alignments) if alignments else math.nan,
            }
        )
    return result


def _paired_drift(rows: list[dict[str, Any]], *, clean: int, missing: int) -> list[dict[str, Any]]:
    paired: dict[tuple[str, int, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(row["method"], int(row["seed"]), row["domain_id"], row["sample_id"])][int(row["condition_index"])] = row
    result = []
    for (method, seed, domain, sample), values in sorted(paired.items()):
        if set(values) != {clean, missing}:
            raise ValueError(f"Mechanism trace pair is incomplete for {method} seed{seed} {domain} {sample}.")
        first, second = values[clean], values[missing]
        a = np.asarray(first["output_features"], dtype=np.float64)
        b = np.asarray(second["output_features"], dtype=np.float64)
        denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), np.finfo(np.float64).tiny)
        result.append(
            {
                "method": method,
                "seed": seed,
                "domain_id": domain,
                "sample_id": sample,
                "feature_l2_drift": float(np.linalg.norm(a - b)),
                "feature_cosine_drift": float(1.0 - np.dot(a, b) / denominator),
                "prediction_codebook_shift": _distance(first["prediction"], second["prediction"]),
                "missing_error": _distance(second["prediction"], second["target"]),
            }
        )
    return result


def _cdf_rows(rows: list[dict[str, Any]], *, clean: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        condition = "clean" if int(row["condition_index"]) == clean else "block80"
        groups[(row["method"], condition)].append(_distance(row["prediction"], row["target"]))
    return [
        {"method": method, "condition": condition, "threshold": threshold, "cdf": statistics.fmean(error <= threshold for error in errors)}
        for (method, condition), errors in sorted(groups.items())
        for threshold in range(33)
    ]


def _aggregate_drift(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    return [
        {
            "method": method,
            "sample_count": len(selected),
            "feature_l2_drift": statistics.fmean(row["feature_l2_drift"] for row in selected),
            "feature_cosine_drift": statistics.fmean(row["feature_cosine_drift"] for row in selected),
            "prediction_codebook_shift": statistics.fmean(row["prediction_codebook_shift"] for row in selected),
        }
        for method, selected in sorted(groups.items())
    ]


def _plots(cdf_rows: list[dict[str, Any]], drift_rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cdf_rows:
        groups[(row["method"], row["condition"])].append(row)
    for (method, condition), selected in groups.items():
        axis.plot([row["threshold"] for row in selected], [row["cdf"] for row in selected], label=f"{method} {condition}")
    axis.set(xlabel="Physical codebook-step error", ylabel="CDF", xlim=(0, 16), ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "physical_error_cdf.png", dpi=220)
    figure.savefig(output_dir / "physical_error_cdf.pdf")
    plt.close(figure)

    aggregate = _aggregate_drift(drift_rows)
    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    axis.bar([row["method"] for row in aggregate], [row["feature_cosine_drift"] for row in aggregate])
    axis.set(ylabel="Clean-to-Block80 cosine drift")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output_dir / "feature_drift.png", dpi=220)
    figure.savefig(output_dir / "feature_drift.pdf")
    plt.close(figure)


def _distance(first: int, second: int) -> int:
    absolute = abs(int(first) - int(second)) % 64
    return min(absolute, 64 - absolute)


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
