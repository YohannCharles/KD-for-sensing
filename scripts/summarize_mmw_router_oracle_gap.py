#!/usr/bin/env python3
"""Summarize the fixed-checkpoint MMW Router Oracle Gap Test."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eval_mmw_router_oracle_gap import CONDITIONS, DEFAULT_OUTPUT, PROTOCOL_ID


METRICS = (
    "adba",
    "top1",
    "normalized_gain",
    "spectral_efficiency_ratio_0db",
    "spectral_efficiency_ratio_10db",
    "spectral_efficiency_ratio_20db",
)
SENSORS = ("image_occlusion", "radar_noise", "lidar_sparsify", "gps_noise")
AFFECTED_MODALITY = {"image_occlusion": "image", "radar_noise": "radar", "lidar_sparsify": "lidar", "gps_noise": "gps"}
MODALITY_INDEX = {"image": 0, "radar": 1, "gps": 2, "lidar": 3}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    summarize(Path(args.root).resolve())
    return 0


def summarize(root: Path) -> None:
    manifest = json.loads((root / "evaluation_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest["request"].get("protocol") != PROTOCOL_ID:
        raise ValueError("Oracle Gap manifest is not complete and protocol-matched.")
    rows = []
    for condition in CONDITIONS:
        if not (root / condition / "complete.json").is_file():
            raise ValueError(f"Missing complete condition: {condition}")
        rows.extend(read_csv(root / condition / "domain_metrics.csv"))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["fusion"])].append(row)
    summary = []
    for condition in CONDITIONS:
        fusion_metrics = {}
        for fusion in ("uniform", "learned", "oracle"):
            selected = grouped[(condition, fusion)]
            if len(selected) != 15:
                raise ValueError(f"{condition}/{fusion} expected 15 domain rows, got {len(selected)}")
            fusion_metrics[fusion] = {metric: mean(float(row[metric]) for row in selected) for metric in METRICS}
        for fusion in ("uniform", "learned", "oracle"):
            item: dict[str, Any] = {"condition": condition, "fusion": fusion, "domain_count": 15}
            item.update(fusion_metrics[fusion])
            for metric in METRICS:
                denominator = fusion_metrics["oracle"][metric] - fusion_metrics["uniform"][metric]
                item[f"normalized_gap_closure_{metric}"] = (
                    (fusion_metrics[fusion][metric] - fusion_metrics["uniform"][metric]) / denominator
                    if denominator > 0
                    else math.nan
                )
            selected = grouped[(condition, fusion)]
            item["router_soft_oracle_regret"] = mean(float(row["router_soft_oracle_regret"]) for row in selected)
            item["router_selection_oracle_regret"] = mean(
                float(row["router_selection_oracle_regret"]) for row in selected
            )
            summary.append(item)
    weight_rows = weight_response(root)
    quality_rows = affected_modality_quality(root)
    write_csv(root / "condition_summary.csv", summary)
    write_csv(root / "router_weight_response.csv", weight_rows)
    write_csv(root / "affected_modality_quality.csv", quality_rows)
    plot_metrics(root, summary)
    plot_weights(root, weight_rows)
    (root / "README.md").write_text(render_markdown(summary, weight_rows, quality_rows, manifest), encoding="utf-8")
    (root / "summary.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL_ID,
                "claim_eligible": False,
                "condition_count": len(CONDITIONS),
                "domain_count": 15,
                "condition_summary": summary,
                "router_weight_response": weight_rows,
                "affected_modality_quality": quality_rows,
            },
            indent=2,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )


def weight_response(root: Path) -> list[dict[str, Any]]:
    clean = load_condition_trace(root / "clean")
    results = []
    for sensor in SENSORS:
        modality = AFFECTED_MODALITY[sensor]
        index = MODALITY_INDEX[modality]
        traces = [clean] + [load_condition_trace(root / f"{sensor}_s{severity}") for severity in (1, 2, 3)]
        reference = traces[0]["sample_key"]
        if any(not np.array_equal(reference, trace["sample_key"]) for trace in traces[1:]):
            raise ValueError(f"Sample identities differ across severities for {sensor}.")
        weights = np.stack([trace["router_weights"][:, index] for trace in traces], axis=1)
        domains = traces[0]["domain_id"]
        domain_ids = np.unique(domains)
        means = np.stack([weights[domains == domain].mean(axis=0) for domain in domain_ids]).mean(axis=0)
        monotonic_by_domain = [
            np.mean(np.all(np.diff(weights[domains == domain], axis=1) <= 1e-8, axis=1)) for domain in domain_ids
        ]
        results.append(
            {
                "sensor": sensor,
                "affected_modality": modality,
                "clean_weight": float(means[0]),
                "severity1_weight": float(means[1]),
                "severity2_weight": float(means[2]),
                "severity3_weight": float(means[3]),
                "delta_clean_to_s1": float(means[1] - means[0]),
                "delta_s1_to_s2": float(means[2] - means[1]),
                "delta_s2_to_s3": float(means[3] - means[2]),
                "mean_weight_monotonic": bool(np.all(np.diff(means) <= 1e-8)),
                "mean_weight_spearman": float(spearman(np.arange(4, dtype=float), means)),
                "sample_monotonic_fraction": float(np.mean(monotonic_by_domain)),
                "sample_count": int(weights.shape[0]),
            }
        )
    return results


def affected_modality_quality(root: Path) -> list[dict[str, Any]]:
    results = []
    for sensor in SENSORS:
        modality = AFFECTED_MODALITY[sensor]
        index = MODALITY_INDEX[modality]
        for severity, condition in enumerate(["clean"] + [f"{sensor}_s{level}" for level in (1, 2, 3)]):
            domain_rows = []
            for file in sorted((root / condition / "traces").glob("*.npz")):
                with np.load(file) as payload:
                    logits = payload["unimodal_logits"][:, index].astype(np.float64)
                    shifted = logits - logits.max(axis=1, keepdims=True)
                    probability = np.exp(shifted)
                    probability /= probability.sum(axis=1, keepdims=True)
                    sorted_logits = np.sort(logits, axis=1)
                    domain_rows.append(
                        {
                            "normalized_gain": float(payload["unimodal_normalized_gain"][:, index].mean()),
                            "router_weight": float(payload["router_weights"][:, index].mean()),
                            "oracle_selection_rate": float(np.mean(payload["oracle_modality"] == index)),
                            "confidence": float(probability.max(axis=1).mean()),
                            "normalized_entropy": float(
                                np.mean(-np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=1) / np.log(64))
                            ),
                            "top2_margin": float(np.mean(sorted_logits[:, -1] - sorted_logits[:, -2])),
                        }
                    )
            if len(domain_rows) != 15:
                raise ValueError(f"Expected 15 domains for affected-modality quality: {condition}")
            results.append(
                {
                    "sensor": sensor,
                    "affected_modality": modality,
                    "severity": severity,
                    "condition": condition,
                    **{key: mean(row[key] for row in domain_rows) for key in domain_rows[0]},
                }
            )
    return results


def load_condition_trace(path: Path) -> dict[str, np.ndarray]:
    files = sorted((path / "traces").glob("*.npz"))
    if len(files) != 15:
        raise ValueError(f"Expected 15 trace files under {path}, got {len(files)}")
    keys, domains, routers = [], [], []
    for file in files:
        with np.load(file) as payload:
            domain = str(payload["domain_id"].item())
            ids = payload["sample_id"].astype(str)
            keys.append(np.asarray([f"{domain}::{sample_id}" for sample_id in ids]))
            domains.append(np.repeat(domain, len(ids)))
            routers.append(payload["router_weights"])
    order = np.argsort(np.concatenate(keys))
    return {
        "sample_key": np.concatenate(keys)[order],
        "domain_id": np.concatenate(domains)[order],
        "router_weights": np.concatenate(routers, axis=0)[order],
    }


def plot_metrics(root: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, sensor in zip(axes.flat, SENSORS):
        conditions = ["clean"] + [f"{sensor}_s{severity}" for severity in (1, 2, 3)]
        for fusion, marker in (("uniform", "o"), ("learned", "s"), ("oracle", "^")):
            values = [next(row["normalized_gain"] for row in rows if row["condition"] == condition and row["fusion"] == fusion) for condition in conditions]
            axis.plot(range(4), values, marker=marker, label=fusion.title())
        axis.set_title(sensor.replace("_", " "))
        axis.set_xticks(range(4), ("Clean", "S1", "S2", "S3"))
        axis.set_ylabel("Normalized gain")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False)
    fig.savefig(root / "oracle_gap_normalized_gain.png", dpi=220)
    plt.close(fig)


def plot_weights(root: Path, rows: list[dict[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for row in rows:
        values = [row["clean_weight"], row["severity1_weight"], row["severity2_weight"], row["severity3_weight"]]
        axis.plot(range(4), values, marker="o", label=row["affected_modality"].title())
    axis.set_xticks(range(4), ("Clean", "S1", "S2", "S3"))
    axis.set_ylabel("Router weight on corrupted modality")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.savefig(root / "router_weight_response.png", dpi=220)
    plt.close(fig)


def render_markdown(
    rows: list[dict[str, Any]],
    weights: list[dict[str, Any]],
    quality: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> str:
    indexed = {(row["condition"], row["fusion"]): row for row in rows}
    lines = [
        "# MMW Router Oracle Gap Test",
        "",
        f"该结果固定同一个 {manifest['request'].get('candidate', 'Router')} seed1 inner-validation checkpoint，属于机制诊断（claim_eligible=false），不是独立 outer 主结果。ADBA 为主指标，Top-1 为辅；Oracle 由逐样本实际 beam-power normalized gain 选择最佳单模态。",
        "",
        "## Normalized Gain",
        "",
        "| Condition | Uniform | Learned | Oracle | Learned gap closure | Soft regret |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        uniform = indexed[(condition, "uniform")]
        learned = indexed[(condition, "learned")]
        oracle = indexed[(condition, "oracle")]
        lines.append(
            f"| {condition} | {uniform['normalized_gain']:.4f} | {learned['normalized_gain']:.4f} | "
            f"{oracle['normalized_gain']:.4f} | {learned['normalized_gap_closure_normalized_gain']:.3f} | "
            f"{learned['router_soft_oracle_regret']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Router Weight Response",
            "",
            "| Sensor | Modality | Clean | S1 | S2 | S3 | Mean monotonic | Spearman | Sample monotonic |",
            "|---|---|---:|---:|---:|---:|:---:|---:|---:|",
        ]
    )
    for row in weights:
        lines.append(
            f"| {row['sensor']} | {row['affected_modality']} | {row['clean_weight']:.4f} | "
            f"{row['severity1_weight']:.4f} | {row['severity2_weight']:.4f} | {row['severity3_weight']:.4f} | "
            f"{str(row['mean_weight_monotonic'])} | {row['mean_weight_spearman']:.3f} | "
            f"{row['sample_monotonic_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Affected-Modality Quality Signals",
            "",
            "| Sensor | Level | Gain | Confidence | Entropy | Margin | Router weight | Oracle selected |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in quality:
        lines.append(
            f"| {row['sensor']} | {row['severity']} | {row['normalized_gain']:.4f} | "
            f"{row['confidence']:.4f} | {row['normalized_entropy']:.4f} | {row['top2_margin']:.4f} | "
            f"{row['router_weight']:.4f} | {row['oracle_selection_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Manifest SHA identity: `{manifest['request_sha256']}`",
            f"- Checkpoint SHA256: `{manifest['request']['checkpoint_sha256']}`",
            "- Conditions: Clean + 4 sensors x 3 paired severities; all 15 domains; four modalities remain available.",
            "- Full logits, weights, fused outputs and beam powers are stored in each condition's `traces/*.npz`.",
            "",
        ]
    )
    return "\n".join(lines)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr, yr = ranks(x), ranks(y)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return math.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values) -> float:
    items = list(values)
    return float(sum(items) / len(items))


if __name__ == "__main__":
    raise SystemExit(main())
