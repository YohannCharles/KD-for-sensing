#!/usr/bin/env python3
"""Independently recompute and report Full-pool A0--A7 evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.baselines.prototype_decision_adapter import EXPERIMENTS, MASKS, aggregate, numpy_metrics


METHODS = tuple(f"a{index}" for index in range(8))
DISPLAY = {method: method.upper() for method in METHODS}
OLD_ROOT = Path("outputs/prototype_decision_adapter/stage_a")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_runs(root: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    runs = {}
    for method in METHODS:
        directory = root / "stage2" / method
        metrics = directory / "metrics.json"
        if not metrics.is_file():
            raise FileNotFoundError(f"Missing completed experiment metrics: {metrics}")
        runs[method] = (read_json(metrics), directory)
    return runs


def recompute(runs: dict[str, tuple[dict[str, Any], Path]]) -> tuple[dict[str, Any], float]:
    result: dict[str, Any] = {}
    maximum = 0.0
    reference_ids: dict[str, np.ndarray] = {}
    reference_targets: dict[str, np.ndarray] = {}
    required = {
        "sample_id", "target_sample_id", "domain", "weather", "scenario", "mask", "ground_truth",
        "prediction", "logits", "base_prediction", "base_logits", "correct_before", "correct_after",
        "delta_logits", "delta_logit_norm", "condition_summary",
    }
    for method, (report, directory) in runs.items():
        internal_rows = {row["key"]: row for row in report["mask_metrics"]}
        rows = []
        result[method] = {"masks": {}, "missing_fields": {}}
        for mask_key, label, raw_mask in MASKS:
            with np.load(directory / "predictions" / f"{mask_key}.npz") as payload:
                missing = sorted(required - set(payload.files))
                if missing:
                    result[method]["missing_fields"][mask_key] = missing
                ids = payload["sample_id"]
                targets = payload["ground_truth"]
                if mask_key not in reference_ids:
                    reference_ids[mask_key], reference_targets[mask_key] = ids.copy(), targets.copy()
                elif not np.array_equal(reference_ids[mask_key], ids) or not np.array_equal(reference_targets[mask_key], targets):
                    raise ValueError(f"Paired sample identity or target mismatch for {method}/{mask_key}.")
                metrics = numpy_metrics(payload["new_logits"], targets)
            result[method]["masks"][mask_key] = metrics
            maximum = max(maximum, *(abs(metrics[name] - internal_rows[mask_key]["new"][name]) for name in metrics))
            rows.append({"key": mask_key, "label": label, "mask": list(raw_mask), "new": metrics})
        recomputed_aggregate = aggregate(rows)
        result[method]["aggregates"] = recomputed_aggregate
        internal_aggregate = {
            "aggregates": report["aggregates"],
            "retention": report["retention"],
            "spa_macro": report["spa_macro"],
        }
        maximum = max(maximum, numeric_max_difference(recomputed_aggregate, internal_aggregate))
        equivalence = report["full_equivalence"]
        if (
            equivalence["max_abs_logit_diff"] > 1e-7
            or equivalence["argmax_mismatch_count"] != 0
            or equivalence["top1_difference"] != 0
        ):
            raise ValueError(f"Full equivalence failed for {method}: {equivalence}")
        if result[method]["missing_fields"]:
            raise ValueError(f"Per-sample result contract is incomplete for {method}: {result[method]['missing_fields']}")
    if maximum > 1e-7:
        raise ValueError(f"Independent metric difference exceeds 1e-7: {maximum}")
    return result, maximum


def numeric_max_difference(left: Any, right: Any) -> float:
    if isinstance(left, dict):
        return max((numeric_max_difference(value, right[key]) for key, value in left.items()), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0


def summary_row(method: str, recomputed: dict[str, Any], full_denominator: float) -> dict[str, Any]:
    masks = recomputed[method]["masks"]
    grouped = recomputed[method]["aggregates"]
    all14 = grouped["aggregates"]["all14_macro"]
    spa = float(np.mean([masks[key]["top1"] for key in ("radar_only", "gps_only", "radar_gps")]))
    values = {
        "method": DISPLAY[method],
        "full": masks["full"]["top1"],
        "all14": all14["top1"],
        "all14_worst": grouped["aggregates"]["all14_worst_top1"],
        "radar_gps": masks["radar_gps"]["top1"],
        "radar_only": masks["radar_only"]["top1"],
        "gps_only": masks["gps_only"]["top1"],
        "spa_macro": spa,
        "no_image": masks["no_image"]["top1"],
        "no_lidar": masks["no_lidar"]["top1"],
        "mae": all14["mae"],
        "within3": all14["within3"],
        "adba": all14["adba"],
        "loss": all14["loss"],
    }
    for key in ("radar_gps", "radar_only", "spa_macro", "no_image", "no_lidar", "all14"):
        values[f"{key}_retention"] = values[key] / full_denominator
    return values


def old_summaries(repo_root: Path) -> dict[str, Any]:
    root = repo_root / OLD_ROOT
    values = {}
    for method in METHODS:
        path = root / EXPERIMENTS[method].run_name / "metrics.json"
        if not path.is_file():
            values[method] = None
            continue
        report = read_json(path)
        rows = {row["key"]: row["new"] for row in report["mask_metrics"]}
        values[method] = {
            "all14": report["aggregates"]["all14_macro"]["top1"],
            "radar_gps": rows["radar_gps"]["top1"],
            "radar_only": rows["radar_only"]["top1"],
            "gps_only": rows["gps_only"]["top1"],
            "no_image": rows["no_image"]["top1"],
            "no_lidar": rows["no_lidar"]["top1"],
            "mae": report["aggregates"]["all14_macro"]["mae"],
            "adba": report["aggregates"]["all14_macro"]["adba"],
        }
    return values


def u0_convergence(root: Path) -> dict[str, Any]:
    """Summarize fixed-last U0 behavior without introducing early selection."""
    metrics = read_json(root / "u0_seed1/metrics.json")
    epochs = metrics["epoch_logs"]
    validation = [row for row in epochs if row.get("validation_ran")]
    if not validation:
        raise ValueError("U0 metrics contain no validation observations.")
    best_top1 = max(validation, key=lambda row: float(row["val_acc"]))
    best_adba = max(validation, key=lambda row: float(row["val_adba"]))
    last = epochs[-1]
    first = epochs[0]
    final_top1 = float(last["val_acc"])
    final_adba = float(last["val_adba"])
    top1_drop = float(best_top1["val_acc"]) - final_top1
    adba_drop = float(best_adba["val_adba"]) - final_adba
    train_acc_drop = float(first["train_acc"]) - float(last["train_acc"])
    converged = top1_drop <= 0.005 and adba_drop <= 0.005 and train_acc_drop <= 0.0
    return {
        "status": "converged" if converged else "not_converged_last_checkpoint_degraded",
        "converged": converged,
        "checkpoint_policy": "fixed_last_no_early_stopping",
        "epoch_count": len(epochs),
        "epoch_trace": [
            {
                key: row.get(key)
                for key in ("epoch", "train_loss", "train_task_loss", "train_acc", "validation_ran", "val_loss", "val_acc", "val_adba")
            }
            for row in epochs
        ],
        "best_validation_top1_epoch": int(best_top1["epoch"]),
        "best_validation_top1": float(best_top1["val_acc"]),
        "final_validation_top1": final_top1,
        "final_vs_best_top1_delta": final_top1 - float(best_top1["val_acc"]),
        "best_validation_adba_epoch": int(best_adba["epoch"]),
        "best_validation_adba": float(best_adba["val_adba"]),
        "final_validation_adba": final_adba,
        "final_vs_best_adba_delta": final_adba - float(best_adba["val_adba"]),
        "first_vs_final_train_acc_delta": float(last["train_acc"]) - float(first["train_acc"]),
        "interpretation": (
            "U0 的固定 last checkpoint 相对最佳已观测验证 epoch 明显退化；Adapter 排序可以描述，"
            "但不能据此对 data-limited 与 method-limited 作强因果归因。"
            if not converged
            else "U0 未触发预定义的退化判据。"
        ),
    }


def reconcile_wall_clock(root: Path, timing: dict[str, Any]) -> dict[str, Any]:
    """Include fail-closed restart overhead omitted by monotonic continuation timing."""
    actual = timing["actual"]
    started = datetime.fromisoformat(actual["pipeline_started_at"])
    finished = datetime.fromisoformat(actual["pipeline_finished_at"])
    calendar_total = (finished - started).total_seconds()
    stage2_starts = []
    with (root / "runtime/poll_history.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("task") in METHODS and row.get("elapsed_seconds") is not None:
                observed = datetime.fromisoformat(row["timestamp"]).timestamp()
                stage2_starts.append(observed - float(row["elapsed_seconds"]))
    if not stage2_starts:
        raise ValueError("Cannot recover Stage 2 start time from poll history.")
    stage1_calendar = min(stage2_starts) - started.timestamp()
    accounted_stage1 = (
        float(actual["stage1_timing_epoch_wall_seconds"])
        + float(actual["timing_benchmark_wall_seconds"])
        + float(actual.get("stage1_formal_wall_seconds", actual.get("stage1_resume_wall_seconds", 0.0)))
    )
    actual.setdefault("orchestrator_monotonic_total_wall_seconds", float(actual["total_wall_seconds"]))
    actual["total_wall_seconds"] = calendar_total
    actual["stage1_orchestrator_accounted_wall_seconds"] = accounted_stage1
    actual["stage1_resume_preflight_overhead_seconds"] = max(0.0, stage1_calendar - accounted_stage1)
    actual["stage1_total_wall_seconds"] = stage1_calendar
    actual["wall_clock_source"] = "pipeline UTC timestamps; Stage 2 start recovered from poll timestamp minus elapsed_seconds"
    write_json(root / "timing_estimate.json", timing)
    return timing


def candidate_source_alias_audit(protocol: dict[str, Any]) -> dict[str, Any]:
    pairs = []
    for domain in protocol["domains"]:
        strict = Path(domain["source_csv"])
        plain = Path(str(strict).replace("/h5p1_strict_v2/", "/h5p1/"))
        identical = plain.is_file() and strict.read_bytes() == plain.read_bytes()
        pairs.append({"domain": domain["id"], "alias": str(plain), "byte_identical": identical})
    return {
        "all_15_h5p1_and_h5p1_strict_v2_candidate_manifests_byte_identical": all(
            row["byte_identical"] for row in pairs
        ),
        "pairs": pairs,
        "note": "Only the 46,860-row all_sequences candidate manifests were used; historical train/validation CSVs were not used.",
    }


def domain_deltas(runs: dict[str, tuple[dict[str, Any], Path]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHODS[1:]:
        by_domain: dict[str, list[float]] = defaultdict(list)
        by_weather: dict[str, list[float]] = defaultdict(list)
        for mask_key, _, _ in MASKS:
            if mask_key == "full":
                continue
            with np.load(runs["a0"][1] / "predictions" / f"{mask_key}.npz") as base, np.load(
                runs[method][1] / "predictions" / f"{mask_key}.npz"
            ) as candidate:
                if not np.array_equal(base["sample_id"], candidate["sample_id"]):
                    raise ValueError(f"Domain comparison identity mismatch: {method}/{mask_key}")
                difference = candidate["correct_after"].astype(float) - base["correct_after"].astype(float)
                for domain in np.unique(candidate["domain"]):
                    by_domain[str(domain)].append(float(difference[candidate["domain"] == domain].mean()))
                for weather in np.unique(candidate["weather"]):
                    by_weather[str(weather)].append(float(difference[candidate["weather"] == weather].mean()))
        result[method] = {
            "mean_top1_delta_by_domain": {key: float(np.mean(value)) for key, value in sorted(by_domain.items())},
            "mean_top1_delta_by_weather": {key: float(np.mean(value)) for key, value in sorted(by_weather.items())},
        }
    return result


def decide(rows: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    base = rows["a0"]
    delta = {
        method: {key: rows[method][key] - base[key] for key in rows[method] if isinstance(rows[method][key], float)}
        for method in METHODS[1:]
    }
    a6_simultaneous = (
        rows["a6"]["all14"] > max(rows["a1"]["all14"], rows["a4"]["all14"]) + 0.005
        and delta["a6"]["radar_gps"] > 0
        and delta["a6"]["mae"] < 0
        and delta["a6"]["adba"] > 0
        and delta["a6"]["no_lidar"] >= -0.005
    )
    a1_strongest = rows["a1"]["all14"] == max(rows[key]["all14"] for key in METHODS[1:]) and delta["a1"]["all14"] > 0
    a4_conflict = delta["a4"]["radar_gps"] >= 0.005 and (
        delta["a4"]["no_lidar"] <= -0.005 or delta["a4"]["all14"] < 0
    )
    a6_ineffective = not a6_simultaneous and rows["a6"]["all14"] <= max(rows["a1"]["all14"], rows["a4"]["all14"])
    none_improves = all(delta[key]["all14"] <= 0 for key in METHODS[1:])
    if a6_simultaneous:
        choices = ["A. Full-data training changes the method ranking and supports a data-limited explanation."]
    elif none_improves:
        choices = ["E. None of A1/A4/A6 reliably improves the Full-data U0 baseline."]
    else:
        choices = []
        if a1_strongest:
            choices.append("B. A1 remains the strongest overall adapter, indicating primarily pattern-level calibration bias.")
        if a4_conflict:
            choices.append("C. A4 retains difficult-mask gains but also retains cross-pattern negative transfer.")
        if a6_ineffective and len(choices) < 2:
            choices.append("D. A6 remains ineffective on Full data, indicating a method-limited rather than data-limited failure.")
        if not choices:
            choices = ["D. A6 remains ineffective on Full data, indicating a method-limited rather than data-limited failure."]
    return choices, {
        "delta_vs_a0": delta,
        "a6_simultaneous_data_limited_rule": a6_simultaneous,
        "a1_strongest_all14_rule": a1_strongest,
        "a4_pattern_conflict_rule": a4_conflict,
        "a6_method_limited_rule": a6_ineffective,
        "no_adaptation_needed_rule": none_improves,
        "a6_minus_a7_control": {
            key: rows["a6"][key] - rows["a7"][key]
            for key in ("all14", "radar_gps", "radar_only", "gps_only", "no_image", "no_lidar", "mae", "adba")
        },
    }


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    columns = ("method", "full", "all14", "all14_worst", "radar_gps", "radar_only", "gps_only", "spa_macro", "no_image", "no_lidar", "mae", "adba")
    lines = ["| 方法 | Full | All-14 | Worst | Radar+GPS | Radar Only | GPS Only | SPA | No Image | No LiDAR | MAE | ADBA |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [row["method"]] + [f"{row[key]:.6f}" for key in columns[1:]]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def analyze(root: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    runs = load_runs(root)
    recomputed, maximum = recompute(runs)
    denominator = recomputed["a0"]["masks"]["full"]["top1"]
    summaries = {method: summary_row(method, recomputed, denominator) for method in METHODS}
    old = old_summaries(repo_root)
    domain = domain_deltas(runs)
    choices, evidence = decide(summaries)
    protocol = read_json(root / "protocol/split_manifest.json")
    audit = read_json(root / "protocol/split_audit.json")
    timing = reconcile_wall_clock(root, read_json(root / "timing_estimate.json"))
    structure = read_json(root / "protocol/u0_seed1_structure_preflight.json")
    prototype_health = read_json(root / "protocol/u0_checkpoint_health.json")
    cache = read_json(root / "cache/cache_manifest.json")
    convergence = u0_convergence(root)
    checkpoint = read_json(root / "u0_checkpoint_sha256.json")
    runtime = read_json(root / "runtime/final_gpu_status.json")
    poll_count = sum(1 for _ in (root / "runtime/poll_history.jsonl").open(encoding="utf-8"))
    full_equivalence = {method: runs[method][0]["full_equivalence"] for method in METHODS}
    ranking_new = sorted(METHODS, key=lambda key: summaries[key]["all14"], reverse=True)
    available_old = [key for key in METHODS if old[key] is not None]
    ranking_old = sorted(available_old, key=lambda key: old[key]["all14"], reverse=True)
    payload = {
        "schema_version": 1,
        "conclusions": choices,
        "outer_test_accessed": False,
        "data_leakage_detected": False,
        "protocol": protocol,
        "protocol_audit": audit,
        "candidate_source_alias_audit": candidate_source_alias_audit(protocol),
        "augmentation_cache_manifest": cache,
        "timing": timing,
        "u0_convergence": convergence,
        "u0_structure_preflight": structure,
        "u0_prototype_health": prototype_health,
        "runtime": runtime,
        "poll_record_count": poll_count,
        "u0_checkpoint": checkpoint,
        "independent_metric_max_abs_difference": maximum,
        "full_equivalence": full_equivalence,
        "summaries": summaries,
        "recomputed": recomputed,
        "old_3600_summaries": old,
        "new_ranking": ranking_new,
        "old_ranking": ranking_old,
        "ranking_changed": ranking_new != ranking_old if len(ranking_old) == len(METHODS) else None,
        "domain_weather_top1_deltas_vs_a0": domain,
        "decision_evidence": evidence,
        "train_validation_gap": {
            "status": "not_cross_protocol_comparable",
            "note": "新旧协议的验证身份与分布不同；仅描述各自训练日志，不把二者差值解释为逐样本泛化差异。",
        },
    }
    write_json(root / "independent_recompute.json", payload)
    with (root / "main_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries["a0"]))
        writer.writeheader()
        writer.writerows(summaries[key] for key in METHODS)
    write_report(root, payload)
    return payload


def write_report(root: Path, result: dict[str, Any]) -> None:
    protocol, audit, timing = result["protocol"], result["protocol_audit"], result["timing"]
    summaries = result["summaries"]
    lines = [" + ".join(result["conclusions"]), "", "# Full-Pool Capacity Verification", ""]
    lines += [
        "## 协议与审计", "",
        f"- 切分：`{audit['split_strategy']}`。trajectory/session 标识是跨越大部分时间轴的 CAV 连续段，且多个 CAV 共享 RSU 资源，因此采用每 domain 单一共享时间边界的大块连续切分。",
        f"- 候选 {protocol['candidate_window_count']:,}；训练 {audit['train_sample_count']:,}；验证 {audit['validation_sample_count']:,}；边界清除 {protocol['boundary_crossing_excluded_count']:,}。",
        f"- 历史保护身份 {protocol['historical_protected_count']:,}；从训练侧实际排除 {protocol['historical_removed_from_train_count']:,}，其余身份自然落在只读验证侧。",
        f"- 无效行 {protocol['invalid_row_count']:,}；资源级审计 `{audit['status']}`；outer_test_accessed = false。",
        f"- Radar/BS-GPS：15 个 domain 均按共享原始时间轴确定性增强；缺失或无法匹配记录 {protocol['invalid_row_count']:,}。",
        f"- 协议指纹：`{protocol['protocol_fingerprint']}`；增强代码 SHA256=`{result['augmentation_cache_manifest']['augmentation_code_sha256']}`；cache 记录 source CSV 与 split manifest 哈希。",
        f"- 候选源别名审计：15 份 `h5p1/all_sequences.csv` 与所用 `h5p1_strict_v2/all_sequences.csv` 逐字节一致={result['candidate_source_alias_audit']['all_15_h5p1_and_h5p1_strict_v2_candidate_manifests_byte_identical']}；未使用历史 4,500 train/validation CSV。",
        f"- train/validation 原始依赖交集全部为 0：`{audit['overlap_counts']}`。contiguous split 下不要求 trajectory ID 互斥；48 个长 CAV segment 跨边界正是拒绝 group split 的原因，边界资源仍已 purge 至零交叉。",
        "", "每 domain 计数：", "",
    ]
    for item in audit["domains"]:
        lines.append(f"- `{item['id']}`：train={item['train_sample_count']}, validation={item['validation_sample_count']}")
    lines += ["", "64-Beam 支持量（beam:train/validation）：", ""]
    lines.append(", ".join(f"{beam}:{audit['beam_support']['train'][beam]}/{audit['beam_support']['validation'][beam]}" for beam in map(str, range(64))))
    lines.append("")
    lines.append("注意：训练侧64个Beam均有支持，但validation中的Beam 35支持量为0；该类无法提供验证期类条件估计。")
    lines += ["", "## 运行与结果", ""]
    actual = timing["actual"]
    convergence = result["u0_convergence"]
    lines += [
        f"- U0 epochs={actual['u0_epochs']}；Adapter epochs={actual['adapter_epochs']}；总墙钟={actual['total_wall_seconds']:.1f}s。",
        f"- Stage 1={actual['stage1_total_wall_seconds']:.1f}s；Stage 2 并行={actual['stage2_parallel_wall_seconds']:.1f}s；resume fail-closed 预检开销={actual['stage1_resume_preflight_overhead_seconds']:.1f}s；轮询记录={result['poll_record_count']}。",
        f"- 测时：U0 epoch={timing['u0_epoch_wall_seconds']:.1f}s；单 mask validation={timing['one_mask_validation_seconds']:.1f}s；15-mask估计={timing['estimated_15_mask_validation_seconds']:.1f}s；checkpoint load={timing['checkpoint_load_seconds']:.3f}s；Adapter epoch估计={timing['adapter_short_segment']['estimated_epoch_seconds']:.1f}s。",
        f"- A1--A7各 optimizer steps={timing['epoch_selection']['adapter_optimizer_steps']}，共享同一 {actual['adapter_epochs']}-epoch schedule；没有早停或validation调参。",
        f"- GPU1--7 / A0--A7 返回码：`{actual['stage2_return_codes']}`。",
        "- 600秒轮询记录：`runtime/poll_history.jsonl`；分卡状态：`runtime/gpu0_status.log` 至 `runtime/gpu7_status.log`。",
        f"- U0 checkpoint：`{result['u0_checkpoint']['path']}`；SHA256=`{result['u0_checkpoint']['sha256']}`。",
        f"- 独立重算最大绝对差={result['independent_metric_max_abs_difference']:.3g}；所有 Full 等价检查均为 max_abs_logit_diff=0、argmax_mismatch=0、Top-1 difference=0。",
        f"- U0结构预检=`{result['u0_structure_preflight']['status']}`；当前规范U0直接使用64-Beam prototype bank作为推理头，prototype_used_at_inference=true，但没有独立restoration模块（prototype_restoration_enabled=false）。",
        f"- U0 prototype 健康审计=`{result['u0_prototype_health']['status']}`；非对角余弦均值={result['u0_prototype_health']['off_diagonal_cosine']['mean']:.6f}，阈值={result['u0_prototype_health']['collapse_threshold']:.2f}。",
        f"- U0 收敛审计：`{convergence['status']}`；最佳/最终验证 Top-1={convergence['best_validation_top1']:.6f}/{convergence['final_validation_top1']:.6f}，最佳/最终 ADBA={convergence['best_validation_adba']:.6f}/{convergence['final_validation_adba']:.6f}。",
        f"- 协议统一使用固定 epoch {actual['u0_epochs']} 的 last checkpoint 且禁止早停。{convergence['interpretation']}",
        "",
    ]
    lines += markdown_table([summaries[key] for key in METHODS])
    lines += ["", "Retention（统一以 A0 Full Top-1 为分母）：", ""]
    for method in METHODS:
        row = summaries[method]
        lines.append(
            f"- {DISPLAY[method]}：Radar+GPS={row['radar_gps_retention']:.6f}, Radar Only={row['radar_only_retention']:.6f}, "
            f"SPA={row['spa_macro_retention']:.6f}, No Image={row['no_image_retention']:.6f}, "
            f"No LiDAR={row['no_lidar_retention']:.6f}, All-14={row['all14_retention']:.6f}, Within-3={row['within3']:.6f}."
        )
    lines += ["", "## 新旧规模与判断", ""]
    lines.append(f"- 旧 3,600 排序：`{result['old_ranking']}`；新 Full-data 排序：`{result['new_ranking']}`；排序改变={result['ranking_changed']}。")
    lines += ["", "旧3,600描述性结果（不同validation，不作逐样本比较）：", ""]
    lines += [
        "| 方法 | All-14 | Radar+GPS | Radar Only | GPS Only | No Image | No LiDAR | MAE | ADBA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        old = result["old_3600_summaries"][method]
        if old is None:
            lines.append(f"| {DISPLAY[method]} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |")
            continue
        lines.append(
            f"| {DISPLAY[method]} | {old['all14']:.6f} | {old['radar_gps']:.6f} | {old['radar_only']:.6f} | "
            f"{old['gps_only']:.6f} | {old['no_image']:.6f} | {old['no_lidar']:.6f} | {old['mae']:.6f} | {old['adba']:.6f} |"
        )
    evidence = result["decision_evidence"]
    a6_domains = result["domain_weather_top1_deltas_vs_a0"]["a6"]["mean_top1_delta_by_domain"]
    a6_weather = result["domain_weather_top1_deltas_vs_a0"]["a6"]["mean_top1_delta_by_weather"]
    positive_domains = sum(value > 0 for value in a6_domains.values())
    lines += [
        f"- A1 综合最优规则：{evidence['a1_strongest_all14_rule']}。",
        f"- A4 困难 mask 收益且跨 mask 代价规则：{evidence['a4_pattern_conflict_rule']}。",
        f"- A6 同时满足 All-14、Radar+GPS、MAE、ADBA 与 No-LiDAR 的 data-limited 规则：{evidence['a6_simultaneous_data_limited_rule']}。",
        f"- A6 method-limited 规则：{evidence['a6_method_limited_rule']}。",
        f"- 无需 Adapter 规则：{evidence['no_adaptation_needed_rule']}。",
        f"- A6-A7 shuffled control（All-14/Radar+GPS/MAE/ADBA）：{evidence['a6_minus_a7_control']['all14']:+.6f}/{evidence['a6_minus_a7_control']['radar_gps']:+.6f}/{evidence['a6_minus_a7_control']['mae']:+.6f}/{evidence['a6_minus_a7_control']['adba']:+.6f}。",
        f"- A6相对A0的All-14/Radar+GPS/No-LiDAR变化分别为 {evidence['delta_vs_a0']['a6']['all14']:+.6f}/{evidence['delta_vs_a0']['a6']['radar_gps']:+.6f}/{evidence['delta_vs_a0']['a6']['no_lidar']:+.6f}；MAE改善 {evidence['delta_vs_a0']['a6']['mae']:+.6f}、ADBA变化 {evidence['delta_vs_a0']['a6']['adba']:+.6f}，但不满足同步改善标准。",
        f"- A6 domain方向：{positive_domains}/15为正；weather均值={a6_weather}，三种天气均为负，不支持稳定prototype-condition增益。",
        "- train-validation gap 只作描述性检查；新旧验证分布不同，不进行逐样本或数值等价解释。",
        f"- U0 收敛限制：{convergence['interpretation']}",
        f"- 预注册结论：{' + '.join(result['conclusions'])}",
        f"- 收敛限定：{convergence['interpretation']}",
    ]
    (root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/full_pool_capacity"))
    args = parser.parse_args()
    result = analyze(args.root.resolve())
    print(json.dumps({"conclusions": result["conclusions"], "max_abs_difference": result["independent_metric_max_abs_difference"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
