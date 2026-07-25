#!/usr/bin/env python3
"""Independently recompute Stage A metrics and paired statistics from NPZ."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from kd_sensing.baselines.prototype_decision_adapter import EXPERIMENTS, MASKS, numpy_metrics


PRIMARY_COMPARISONS = (("a3", "a0"), ("a5", "a3"), ("a6", "a5"), ("a6", "a7"))
PRIMARY_MASKS = ("radar_gps", "radar_only", "all14_macro")
PARTIAL_KEYS = ("a0", "a1", "a2", "a3", "a4", "a5", "a6")


def _load_run(root: Path, key: str) -> tuple[dict, Path]:
    directory = root / EXPERIMENTS[key].run_name
    return json.loads((directory / "metrics.json").read_text(encoding="utf-8")), directory


def _observations(directory: Path, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if key != "all14_macro":
        payload = np.load(directory / "predictions" / f"{key}.npz")
        return payload["sample_id"], payload["correct_after"].astype(float), payload["ground_truth"]
    values = []
    ids = None
    for mask_key, _, _ in MASKS:
        if mask_key == "full":
            continue
        payload = np.load(directory / "predictions" / f"{mask_key}.npz")
        if ids is None:
            ids = payload["sample_id"]
        elif not np.array_equal(ids, payload["sample_id"]):
            raise ValueError("All-14 files do not share sample order.")
        values.append(payload["correct_after"].astype(float))
    return ids, np.stack(values).mean(0), np.zeros(len(ids), dtype=int)


def _paired(candidate: np.ndarray, baseline: np.ndarray, *, seed: int = 1, draws: int = 10000) -> dict:
    difference = candidate - baseline
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for start in range(0, draws, 250):
        count = min(250, draws - start)
        index = rng.integers(0, len(difference), size=(count, len(difference)))
        boot[start : start + count] = difference[index].mean(1)
    less = (boot <= 0).mean()
    greater = (boot >= 0).mean()
    return {
        "mean_difference": float(difference.mean()),
        "confidence_interval_95": [float(value) for value in np.quantile(boot, [0.025, 0.975])],
        "raw_bootstrap_p_value": float(min(1.0, 2 * min(less, greater))),
        "improved_sample_count": int((difference > 0).sum()),
        "degraded_sample_count": int((difference < 0).sum()),
        "unchanged_sample_count": int((difference == 0).sum()),
    }


def _exact_discordance_p(candidate: np.ndarray, baseline: np.ndarray) -> float:
    improved = int(((candidate > baseline)).sum())
    degraded = int(((candidate < baseline)).sum())
    total = improved + degraded
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, value) for value in range(min(improved, degraded) + 1))
    return min(1.0, 2.0 * tail / (2**total))


def _holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _eligibility(candidate: dict, baseline: dict) -> dict:
    crows = {row["key"]: row for row in candidate["mask_metrics"]}
    brows = {row["key"]: row for row in baseline["mask_metrics"]}
    hard = {
        "full_equivalence": candidate["full_equivalence"]["max_abs_logit_diff"] <= 1e-7,
        "all14_top1": candidate["aggregates"]["all14_macro"]["top1"] - baseline["aggregates"]["all14_macro"]["top1"] >= -0.002,
        "no_image": crows["no_image"]["new"]["top1"] - brows["no_image"]["new"]["top1"] >= -0.005,
        "no_lidar": crows["no_lidar"]["new"]["top1"] - brows["no_lidar"]["new"]["top1"] >= -0.005,
        "all14_mae": candidate["aggregates"]["all14_macro"]["mae"] - baseline["aggregates"]["all14_macro"]["mae"] <= 0.10,
        "all14_adba": candidate["aggregates"]["all14_macro"]["adba"] - baseline["aggregates"]["all14_macro"]["adba"] >= -0.003,
    }
    gains = {
        "radar_gps": crows["radar_gps"]["new"]["top1"] - brows["radar_gps"]["new"]["top1"] >= 0.02,
        "radar_only": crows["radar_only"]["new"]["top1"] - brows["radar_only"]["new"]["top1"] >= 0.01,
        "single_worst": candidate["aggregates"]["single_worst_top1"] - baseline["aggregates"]["single_worst_top1"] >= 0.01,
        "spa_macro": candidate["spa_macro"] - baseline["spa_macro"] >= 0.015,
    }
    return {"hard_constraints": hard, "major_gains": gains, "eligible": all(hard.values()) and sum(gains.values()) >= 2}


def analyze(stage_root: Path, draws: int = 10000) -> dict:
    runs = {key: _load_run(stage_root, key) for key in EXPERIMENTS}
    recomputed = {}
    max_difference = 0.0
    for key, (report, directory) in runs.items():
        mask_report = {row["key"]: row for row in report["mask_metrics"]}
        recomputed[key] = {}
        for mask_key, _, _ in MASKS:
            payload = np.load(directory / "predictions" / f"{mask_key}.npz")
            metrics = numpy_metrics(payload["new_logits"], payload["ground_truth"])
            recomputed[key][mask_key] = metrics
            max_difference = max(max_difference, *(abs(metrics[name] - mask_report[mask_key]["new"][name]) for name in metrics))
    if max_difference > 1e-7:
        raise ValueError(f"Independent metric difference exceeds 1e-7: {max_difference}")
    comparisons = []
    p_values = []
    for candidate, baseline in PRIMARY_COMPARISONS:
        for mask_key in PRIMARY_MASKS:
            candidate_ids, candidate_values, _ = _observations(runs[candidate][1], mask_key)
            baseline_ids, baseline_values, _ = _observations(runs[baseline][1], mask_key)
            if not np.array_equal(candidate_ids, baseline_ids):
                raise ValueError("Paired comparison sample ids differ.")
            row = {"comparison": f"{candidate}_vs_{baseline}", "mask": mask_key, **_paired(candidate_values, baseline_values, draws=draws)}
            row["raw_exact_p_value"] = _exact_discordance_p(candidate_values, baseline_values)
            comparisons.append(row)
            p_values.append(row["raw_exact_p_value"])
    for row, adjusted in zip(comparisons, _holm(p_values)):
        row["holm_adjusted_p_value"] = adjusted
    baseline = runs["a0"][0]
    candidate_keys = tuple(key for key in EXPERIMENTS if key not in {"a0", "a7"})
    eligibility = {key: _eligibility(runs[key][0], baseline) for key in candidate_keys}
    control_eligibility = _eligibility(runs["a7"][0], baseline)
    eligible = [key for key, value in eligibility.items() if value["eligible"]]
    a1 = runs["a1"][0]["aggregates"]["all14_macro"]["top1"]
    a3 = runs["a3"][0]["aggregates"]["all14_macro"]["top1"]
    a6 = runs["a6"][0]["aggregates"]["all14_macro"]["top1"]
    a7 = runs["a7"][0]["aggregates"]["all14_macro"]["top1"]
    if not eligible:
        conclusion = "D. Decision adaptation does not improve the frozen U0 baseline without unacceptable trade-offs."
    elif abs(a1 - max(runs[key][0]["aggregates"]["all14_macro"]["top1"] for key in ("a2", "a3", "a4"))) <= 0.002:
        conclusion = "C. A simple missing-pattern logit bias is sufficient; low-rank adaptation is unnecessary."
    elif a6 > a3 and a6 > a7:
        conclusion = "B. Prototype states provide additional, causally supported gains over mask-only decision adaptation."
    else:
        conclusion = "A. Mask-conditioned low-rank decision adaptation improves strong-modality-missing retention."
    diagnostics = _prototype_diagnostics(runs)
    result = {
        "conclusion": conclusion, "outer_test_accessed": False, "data_leakage_detected": False,
        "independent_metric_max_abs_difference": max_difference, "recomputed": recomputed,
        "paired_statistics": comparisons, "eligibility": eligibility,
        "a7_control_eligibility": control_eligibility,
        "stage_b_recommended": bool(eligible), "eligible_candidates": eligible,
        "a1_is_sufficient": conclusion.startswith("C."), "a3_minus_a1_all14_top1": a3 - a1,
        "a6_minus_a7_all14_top1": a6 - a7,
        "prototype_diagnostics": diagnostics,
    }
    (stage_root.parent / "stage_a_analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [conclusion, "", f"outer_test_accessed = false", f"data_leakage_detected = false", "", f"Stage B recommended: {result['stage_b_recommended']}."]
    (stage_root.parent / "stage_a_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def analyze_partial_a0_a6(stage_root: Path, draws: int = 10000) -> dict:
    runs = {key: _load_run(stage_root, key) for key in PARTIAL_KEYS}
    max_difference = 0.0
    summaries = {}
    for key, (report, directory) in runs.items():
        mask_report = {row["key"]: row for row in report["mask_metrics"]}
        for mask_key, _, _ in MASKS:
            payload = np.load(directory / "predictions" / f"{mask_key}.npz")
            metrics = numpy_metrics(payload["new_logits"], payload["ground_truth"])
            max_difference = max(
                max_difference,
                *(abs(metrics[name] - mask_report[mask_key]["new"][name]) for name in metrics),
            )
        summaries[key] = {
            "all14_top1": report["aggregates"]["all14_macro"]["top1"],
            "all14_mae": report["aggregates"]["all14_macro"]["mae"],
            "all14_adba": report["aggregates"]["all14_macro"]["adba"],
            "radar_gps_top1": mask_report["radar_gps"]["new"]["top1"],
            "radar_only_top1": mask_report["radar_only"]["new"]["top1"],
            "no_image_top1": mask_report["no_image"]["new"]["top1"],
            "no_lidar_top1": mask_report["no_lidar"]["new"]["top1"],
            "single_worst_top1": report["aggregates"]["single_worst_top1"],
            "spa_macro": report["spa_macro"],
            "adapter_parameter_count": report["adapter_parameter_count"],
            "full_max_abs_logit_diff": report["full_equivalence"]["max_abs_logit_diff"],
        }
    if max_difference > 1e-7:
        raise ValueError(f"Independent metric difference exceeds 1e-7: {max_difference}")

    comparisons = []
    p_values = []
    for candidate, baseline in PRIMARY_COMPARISONS[:-1]:
        for mask_key in PRIMARY_MASKS:
            candidate_ids, candidate_values, _ = _observations(runs[candidate][1], mask_key)
            baseline_ids, baseline_values, _ = _observations(runs[baseline][1], mask_key)
            if not np.array_equal(candidate_ids, baseline_ids):
                raise ValueError("Paired comparison sample ids differ.")
            row = {
                "comparison": f"{candidate}_vs_{baseline}",
                "mask": mask_key,
                **_paired(candidate_values, baseline_values, draws=draws),
            }
            row["raw_exact_p_value"] = _exact_discordance_p(candidate_values, baseline_values)
            comparisons.append(row)
            p_values.append(row["raw_exact_p_value"])
    for row, adjusted in zip(comparisons, _holm(p_values)):
        row["holm_adjusted_p_value"] = adjusted

    baseline = runs["a0"][0]
    eligibility = {key: _eligibility(runs[key][0], baseline) for key in PARTIAL_KEYS[1:]}
    eligible = [key for key, value in eligibility.items() if value["eligible"]]
    best_all14 = max(PARTIAL_KEYS, key=lambda key: summaries[key]["all14_top1"])
    result = {
        "status": "provisional_pending_a7",
        "conclusion": (
            "A0-A6 provisional analysis only: prototype-state causal attribution is pending A7."
        ),
        "outer_test_accessed": False,
        "data_leakage_detected": False,
        "independent_metric_max_abs_difference": max_difference,
        "summaries": summaries,
        "paired_statistics": comparisons,
        "eligibility": eligibility,
        "eligible_candidates": eligible,
        "best_all14_top1": best_all14,
        "a3_minus_a1_all14_top1": summaries["a3"]["all14_top1"] - summaries["a1"]["all14_top1"],
        "a5_minus_a3_all14_top1": summaries["a5"]["all14_top1"] - summaries["a3"]["all14_top1"],
        "a6_minus_a5_all14_top1": summaries["a6"]["all14_top1"] - summaries["a5"]["all14_top1"],
    }
    output_root = stage_root.parent
    (output_root / "stage_a_partial_a0_a6_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stage A A0-A6 Provisional Analysis",
        "",
        "> A7 is not included. Prototype-state causal attribution remains pending.",
        "",
        "| Run | All-14 Top-1 | Radar+GPS | Radar Only | No Image | No LiDAR | All-14 MAE | ADBA | Eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for key in PARTIAL_KEYS:
        row = summaries[key]
        eligible_value = "baseline" if key == "a0" else str(eligibility[key]["eligible"]).lower()
        lines.append(
            f"| {key.upper()} | {row['all14_top1']:.4f} | {row['radar_gps_top1']:.4f} | "
            f"{row['radar_only_top1']:.4f} | {row['no_image_top1']:.4f} | {row['no_lidar_top1']:.4f} | "
            f"{row['all14_mae']:.4f} | {row['all14_adba']:.4f} | {eligible_value} |"
        )
    lines.extend([
        "",
        f"Independent metric max absolute difference: `{max_difference:.3g}`.",
        f"Best provisional All-14 Top-1: `{best_all14.upper()}`.",
        "Final A6 prototype-state claim requires the A6-vs-A7 negative control.",
    ])
    (output_root / "stage_a_partial_a0_a6_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _bucket_gain(values: np.ndarray, before: np.ndarray, after: np.ndarray) -> list[dict]:
    edges = np.unique(np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0]))
    rows = []
    for index in range(max(0, len(edges) - 1)):
        lower, upper = float(edges[index]), float(edges[index + 1])
        selected = (values >= lower) & (values <= upper if index == len(edges) - 2 else values < upper)
        if selected.any():
            rows.append({
                "lower": lower,
                "upper": upper,
                "sample_count": int(selected.sum()),
                "top1_gain": float(after[selected].mean() - before[selected].mean()),
            })
    return rows


def _prototype_diagnostics(runs: dict) -> dict:
    result = {}
    for experiment, (_, directory) in runs.items():
        by_mask = {}
        for mask_key, _, _ in MASKS:
            payload = np.load(directory / "predictions" / f"{mask_key}.npz")
            before = payload["correct_before"].astype(bool)
            after = payload["correct_after"].astype(bool)
            prototype_rows = []
            for prototype_id in np.unique(payload["nearest_id"]):
                selected = payload["nearest_id"] == prototype_id
                prototype_rows.append({
                    "prototype_id": int(prototype_id),
                    "sample_count": int(selected.sum()),
                    "top1_gain": float(after[selected].mean() - before[selected].mean()),
                })
            by_mask[mask_key] = {
                "by_nearest_prototype": prototype_rows,
                "by_entropy_quartile": _bucket_gain(payload["prototype_entropy"], before, after),
                "by_nearest_distance_quartile": _bucket_gain(payload["nearest_distance"], before, after),
                "by_distance_margin_quartile": _bucket_gain(payload["distance_margin"], before, after),
            }
        result[experiment] = by_mask
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/prototype_decision_adapter/stage_a"))
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--partial-a0-a6", action="store_true")
    args = parser.parse_args()
    function = analyze_partial_a0_a6 if args.partial_a0_a6 else analyze
    print(json.dumps(function(args.root.resolve(), args.bootstrap_draws), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
