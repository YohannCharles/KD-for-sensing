#!/usr/bin/env python3
"""Finalize the user-approved post-hoc SMSL fold-0 futility stop."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from normalize_smsl_artifacts import normalize_process_manifest
from run_smsl_r5 import (
    DEFAULT_CONFIG,
    SMSL_ARMS,
    _aggregate_run_artifacts,
    _baseline_g3_cohort_metrics,
    _load_config,
    _output,
    _rejected_run_incidents,
    _write_csv,
    _write_training_manifest,
)
from kd_sensing.baselines.full_pool_common import write_json


CONCLUSION = "全部失败，停止R5路线"


def _metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    scopes = summary["aggregate"]
    masks = {row["mask"]: row for row in summary["masks"]}
    return {
        "single_worst_top1": float(scopes["Single"]["top1_worst"]),
        "severe_worst_top1": float(scopes["Severe"]["top1_worst"]),
        "severe_macro_top1": float(scopes["Severe"]["top1_macro"]),
        "all14_worst_top1": float(scopes["All14"]["top1_worst"]),
        "all14_macro_top1": float(scopes["All14"]["top1_macro"]),
        "severe_within3_worst": float(scopes["Severe"]["within3_worst"]),
        "severe_mae_worst": float(scopes["Severe"]["mae_worst"]),
        "severe_g1_far_error_macro": float(scopes["Severe"]["g1_far_error_macro"]),
        "full_top1": float(masks["full"]["top1"]),
        "image_only_top1": float(masks["image_only"]["top1"]),
        "missing_lidar_top1": float(masks["missing_lidar"]["top1"]),
        "missing_lidar_radar_top1": float(masks["missing_lidar_radar"]["top1"]),
        "missing_lidar_gps_top1": float(masks["missing_lidar_gps"]["top1"]),
    }


def _risk_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bucket_metrics(rows: list[dict[str, str]], bucket: str) -> dict[str, float]:
    selected = [row for row in rows if int(row["available_count"]) <= 2 and row["risk_bucket"] == bucket]
    return {
        "top1": float(np.mean([float(row["top1"]) for row in selected])),
        "g1_rate": float(np.mean([float(row["g1_rate"]) for row in selected])),
    }


def _same_mask(rows: list[dict[str, str]]) -> dict[str, Any]:
    masks = sorted({row["mask"] for row in rows if int(row["available_count"]) <= 2})
    gaps = []
    for name in masks:
        high = next(float(row["g1_rate"]) for row in rows if row["mask"] == name and row["risk_bucket"] == "high_risk_top20")
        low = next(float(row["g1_rate"]) for row in rows if row["mask"] == name and row["risk_bucket"] == "lower_risk_bottom80")
        gaps.append(high - low)
    return {
        "mask_count": len(masks),
        "positive_mask_count": sum(gap > 0.0 for gap in gaps),
        "mean_g1_rate_gap": float(np.mean(gaps)),
    }


def finalize(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_config(config_path.resolve())
    output = _output(config)
    fold_dir = output / "screening/fold_0"
    summaries = {arm: json.loads((fold_dir / arm / "evaluation_summary.json").read_text(encoding="utf-8")) for arm in SMSL_ARMS}
    for arm, summary in summaries.items():
        if summary.get("role") != "train_heldout" or summary.get("validation_accessed") is not False:
            raise ValueError(f"Fold-0 result is not train-only: {arm}")
    metrics = {arm: _metrics(summary) for arm, summary in summaries.items()}
    baseline = metrics["a0"]
    risk = {arm: _risk_rows(fold_dir / arm / "risk_bucket_metrics.csv") for arm in SMSL_ARMS}
    buckets = {
        arm: {name: _bucket_metrics(rows, name) for name in ("high_risk_top20", "lower_risk_bottom80")} for arm, rows in risk.items()
    }
    a2_wins = [
        arm
        for arm in ("a1", "c1", "c2")
        if metrics["a2"]["single_worst_top1"] > metrics[arm]["single_worst_top1"]
        and metrics["a2"]["severe_worst_top1"] > metrics[arm]["severe_worst_top1"]
    ]
    evidence = {
        "status": "posthoc_futility_stop_after_fold0",
        "formal_four_fold_gate_evaluated": False,
        "folds_completed": [0],
        "folds_not_run": [1, 2, 3],
        "seed": int(config["training"]["base_seed"]),
        "a2_wins_against": a2_wins,
        "a2_beats_two_of_three_controls": len(a2_wins) >= 2,
        "a3_beats_c3": metrics["a3"]["single_worst_top1"] > metrics["c3"]["single_worst_top1"],
        "metrics": metrics,
        "risk_buckets": buckets,
        "within_mask_f2_separation": _same_mask(risk["a0"]),
        "outer_test_accessed": False,
        "development_validation_accessed": False,
        "csi_or_channel_accessed": False,
        "conclusion": CONCLUSION,
        "limitation": "User-approved post-hoc single-fold futility stop; not a completed pre-registered four-fold CV result.",
    }
    rows = [
        {
            "fold": 0,
            "arm": arm,
            "seed": int(config["training"]["base_seed"]),
            "formal_cv_complete": False,
            **metrics[arm],
        }
        for arm in SMSL_ARMS
    ]
    _write_csv(output / "screening_summary.csv", rows)
    _write_csv(
        output / "full_seed_summary.csv",
        [
            {
                "status": "not_run_posthoc_fold0_futility",
                "seed_count": 0,
                "reason": "Fold 0 provided no F2 attribution signal; user approved compute-saving stop.",
                "outer_test_accessed": False,
            }
        ],
    )
    _write_training_manifest(config, output)
    _aggregate_run_artifacts(output)
    process_manifest = output / "process_manifest.json"
    if process_manifest.is_file():
        normalize_process_manifest(process_manifest)
    record_paths = {arm: [fold_dir / arm / "evaluation_records.pt"] for arm in SMSL_ARMS}
    g3 = _baseline_g3_cohort_metrics(record_paths, SMSL_ARMS)
    evidence["a0_g3_cohort"] = g3
    write_json(output / "screening_gate.json", evidence, sort_keys=True)
    write_json(
        output / "failure_manifest.json",
        {
            "route": "SMSL_R5",
            **evidence,
            "phase2_run": False,
            "rejected_runs": _rejected_run_incidents(output),
        },
        sort_keys=True,
    )
    write_json(output / "final_conclusion.json", {"conclusion": CONCLUSION, **evidence}, sort_keys=True)

    def pp(left: str, right: str, metric: str) -> float:
        return 100.0 * (metrics[left][metric] - metrics[right][metric])

    high_gain_a2 = 100.0 * (buckets["a2"]["high_risk_top20"]["top1"] - buckets["a0"]["high_risk_top20"]["top1"])
    low_gain_a2 = 100.0 * (buckets["a2"]["lower_risk_bottom80"]["top1"] - buckets["a0"]["lower_risk_bottom80"]["top1"])
    high_gain_a3 = 100.0 * (buckets["a3"]["high_risk_top20"]["top1"] - buckets["a0"]["high_risk_top20"]["top1"])
    low_gain_a3 = 100.0 * (buckets["a3"]["lower_risk_bottom80"]["top1"] - buckets["a0"]["lower_risk_bottom80"]["top1"])
    table = [
        "| Arm | Single Worst | Severe Worst | Severe Macro | All-14 Macro | Full |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {arm.upper()} | {100 * metrics[arm]['single_worst_top1']:.3f}% | {100 * metrics[arm]['severe_worst_top1']:.3f}% | {100 * metrics[arm]['severe_macro_top1']:.3f}% | {100 * metrics[arm]['all14_macro_top1']:.3f}% | {100 * metrics[arm]['full_top1']:.3f}% |"
            for arm in SMSL_ARMS
        ],
    ]
    same_mask = evidence["within_mask_f2_separation"]
    report = [
        "# SMSL R5 最终报告",
        "",
        f"最终结论：**{CONCLUSION}**。",
        "",
        "> 证据边界：按用户批准的计算节约规则，在 fold 0 后执行 post-hoc futility stop。未完成预注册四折 CV，因此本报告只能否定继续计算的必要性，不能声称完整四折统计结论。",
        "",
        "下表均为 fold 0 train-heldout 指标，不是 development validation 或既有生产指标；本轮只用它比较同管线 arms 的相对差异。",
        "",
        *table,
        "",
        f"1. F2 是否优于普通 loss hard mining：否。A2-A1 Single Worst 为 `{pp('a2', 'a1', 'single_worst_top1'):+.3f} pp`，Severe Macro 为 `{pp('a2', 'a1', 'severe_macro_top1'):+.3f} pp`。",
        f"2. F2 是否优于固定 mask 权重：否。A2-C1 Single Worst 为 `{pp('a2', 'c1', 'single_worst_top1'):+.3f} pp`。",
        f"3. 随机打乱后效果是否消失：否。A2-C2 Single Worst 仅 `{pp('a2', 'c2', 'single_worst_top1'):+.3f} pp`，不构成有意义优势。",
        f"4. A3 是否优于普通 margin KD：否。A3-C3 Single Worst 为 `{pp('a3', 'c3', 'single_worst_top1'):+.3f} pp`，Severe Macro 为 `{pp('a3', 'c3', 'severe_macro_top1'):+.3f} pp`。",
        f"5. 提升是否集中在 F2 高风险样本：否。A2 high20/lower80 相对 A0 为 `{high_gain_a2:+.3f}/{low_gain_a2:+.3f} pp`；A3 为 `{high_gain_a3:+.3f}/{low_gain_a3:+.3f} pp`。",
        f"6. 同一 mask 内 F2 是否有效：F2 风险本身在 `{same_mask['positive_mask_count']}/{same_mask['mask_count']}` 个 Severe mask 中令 high20 G1 rate 高于 lower80，平均差 `{100 * same_mask['mean_g1_rate_gap']:+.3f} pp`；但这种识别能力没有转化为 A2/A3 精度收益。",
        f"7. image-only 与 missing-LiDAR：A2 相对 A0 分别为 `{pp('a2', 'a0', 'image_only_top1'):+.3f}` / `{pp('a2', 'a0', 'missing_lidar_top1'):+.3f} pp`；A3 为 `{pp('a3', 'a0', 'image_only_top1'):+.3f}` / `{pp('a3', 'a0', 'missing_lidar_top1'):+.3f} pp`，均未改善。",
        f"8. far error 是否下降：A2 Severe G1 far-error macro 相对 A0 为 `{pp('a2', 'a0', 'severe_g1_far_error_macro'):+.3f} pp`，明显恶化；A3 为 `{pp('a3', 'a0', 'severe_g1_far_error_macro'):+.3f} pp`，但伴随 Top-1 下降。",
        f"9. Full、Within-3、MAE 是否退化：A2 相对 A0 为 Full `{pp('a2', 'a0', 'full_top1'):+.3f} pp`、Severe Within-3 worst `{pp('a2', 'a0', 'severe_within3_worst'):+.3f} pp`、MAE `{metrics['a2']['severe_mae_worst'] - baseline['severe_mae_worst']:+.4f}`；A3 为 `{pp('a3', 'a0', 'full_top1'):+.3f} pp`、`{pp('a3', 'a0', 'severe_within3_worst'):+.3f} pp`、`{metrics['a3']['severe_mae_worst'] - baseline['severe_mae_worst']:+.4f}`。A0-G3 固定 cohort 仅 `{g3['a0']['baseline_a0_g3_cohort_count']}` 个样本，A2/A3 Top-1 为 `{g3['a2']['top1']:.6f}/{g3['a3']['top1']:.6f}`，样本过少，不作稳定退化结论。",
        "10. 证据是否支持第二创新点：不支持。A2 未胜普通 hard mining/固定 mask，A3 未胜普通 margin KD，且收益不集中于高风险样本；按用户批准的 futility 规则停止后续 folds、Phase 2 与 validation。",
        "",
        "未访问 development validation 或 outer test；未使用 CSI/channel。多 seed、trajectory bootstrap 95% CI 与正式四折 gate 均未运行。",
    ]
    (output / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return evidence


if __name__ == "__main__":
    print(json.dumps(finalize(), sort_keys=True))
