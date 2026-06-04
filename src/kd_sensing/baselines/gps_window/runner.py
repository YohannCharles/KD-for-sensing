from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kd_sensing.baselines.gps_window.adapter import (
    ALLOWED_PREDICTION_FIELDS,
    discover_ready_mmw_scenarios,
    guard_no_target_oracle,
    load_beam_power_vectors,
    load_samples_from_csv,
    split_csv_path,
)
from kd_sensing.baselines.gps_window.artifacts import (
    collapse_diagnostics,
    prediction_histogram,
    write_json,
    write_predictions_csv,
)
from kd_sensing.baselines.gps_window.predictors import (
    build_calibration_state,
    error_buckets,
    predict_sample,
)
from kd_sensing.baselines.gps_window.support_split import split_calibration_support
from kd_sensing.baselines.gps_window.types import (
    GpsWindowBaselineConfig,
    GpsWindowRunMetadata,
    GpsWindowSample,
    normalize_scenarios,
)
from kd_sensing.config.io import deep_merge
from kd_sensing.evaluation.metrics import beam_power_metrics, calculate_beam_group_metrics, calculate_dba_score, calculate_topk_accuracy


def run_gps_window_baseline(
    cfg: dict[str, Any],
    *,
    scenes: list[str] | None = None,
    source_scenes: list[str] | None = None,
    target_scenes: list[str] | None = None,
    execute: bool = False,
    sweep: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    baseline_cfg = GpsWindowBaselineConfig.from_mapping(cfg.get("gps_window", {}))
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    data_root = Path(data_cfg.get("data_root", "dataset/MMW/sunny"))
    split_tag = str(data_cfg.get("split_tag", baseline_cfg.split_tag))
    targets = list(target_scenes or scenes or normalize_scenarios(data_cfg.get("target_scenes")) or normalize_scenarios(data_cfg.get("scenes")))
    if not targets:
        targets = discover_ready_mmw_scenarios(data_root, split_tag=split_tag)
    sources = list(source_scenes or normalize_scenarios(data_cfg.get("source_scenes")))
    out_dir = Path(output_dir or cfg.get("output", {}).get("dir", "outputs/gps_window_baseline"))
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = _parameter_grid(cfg, baseline_cfg, sweep=sweep)
    plan = _build_plan(
        cfg,
        data_root=data_root,
        output_dir=out_dir,
        baseline_cfg=baseline_cfg,
        target_scenes=targets,
        source_scenes=sources,
        split_tag=split_tag,
        grid=grid,
    )
    plan_path = write_json(out_dir / "gps_window_plan.json", plan)
    if not execute:
        return {"mode": "plan_only", "plan_path": str(plan_path), **plan}
    scene_results = []
    all_iterations = []
    for target in targets:
        scene_sources = [item for item in sources if item != target] if sources else [item for item in targets if item != target]
        result = _run_target_scene(
            cfg,
            data_root=data_root,
            output_dir=out_dir,
            target_scene=target,
            source_scenes=scene_sources,
            baseline_cfg=baseline_cfg,
            split_tag=split_tag,
            grid=grid,
        )
        scene_results.append(result)
        all_iterations.extend(result.get("iterations", []))
    summary = _summary(scene_results)
    write_json(out_dir / "summary.json", summary)
    next_candidate = _next_candidate_summary(all_iterations)
    write_json(out_dir / "next_candidate_summary.json", next_candidate)
    return {
        "mode": "execute",
        "plan_path": str(plan_path),
        "summary_path": str(out_dir / "summary.json"),
        "next_candidate_summary_path": str(out_dir / "next_candidate_summary.json"),
        "scene_results": scene_results,
        "summary": summary,
        "next_candidate_summary": next_candidate,
    }


def _run_target_scene(
    cfg: dict[str, Any],
    *,
    data_root: Path,
    output_dir: Path,
    target_scene: str,
    source_scenes: list[str],
    baseline_cfg: GpsWindowBaselineConfig,
    split_tag: str,
    grid: list[dict[str, Any]],
) -> dict[str, Any]:
    eval_samples = load_samples_from_csv(
        split_csv_path(data_root, target_scene, "test", split_tag=split_tag),
        scenario=target_scene,
        split="target_test",
        cfg=baseline_cfg,
        max_samples=baseline_cfg.max_samples,
    )
    support_samples = _calibration_samples(
        data_root=data_root,
        target_scene=target_scene,
        source_scenes=source_scenes,
        baseline_cfg=baseline_cfg,
        split_tag=split_tag,
    )
    fit_samples, selection_samples, calibration_info = _split_calibration_support(support_samples, baseline_cfg)
    if not eval_samples:
        return {"target_scene": target_scene, "status": "skipped", "reason": "empty_target_test"}
    scene_dir = output_dir / str(target_scene)
    scene_dir.mkdir(parents=True, exist_ok=True)
    iterations = []
    best: dict[str, Any] | None = None
    for idx, params in enumerate(grid):
        run_cfg = GpsWindowBaselineConfig.from_mapping(deep_merge(baseline_cfg.to_dict(), params))
        run_id = f"run_{idx:03d}_{run_cfg.algorithm}_w{run_cfg.history_window}_o{run_cfg.beam_offset}"
        run_dir = scene_dir / run_id
        calibration_metrics = _evaluate_samples(
            selection_samples,
            run_cfg,
            data_root=data_root,
            output_dir=run_dir / "calibration",
            calibration_samples=fit_samples,
            write_artifacts=False,
            calibration_split_label=calibration_info["fit_split"],
            selection_split_label=calibration_info["selection_split"],
            evaluation_split_label=calibration_info["selection_split"],
            selection_sample_count=len(selection_samples),
        )
        eval_result = _evaluate_samples(
            eval_samples,
            run_cfg,
            data_root=data_root,
            output_dir=run_dir,
            calibration_samples=fit_samples,
            write_artifacts=True,
            calibration_split_label=calibration_info["fit_split"],
            selection_split_label=calibration_info["selection_split"],
            evaluation_split_label="target_test",
            selection_sample_count=len(selection_samples),
        )
        iteration = {
            "run_id": run_id,
            "target_scene": target_scene,
            "source_scenes": list(source_scenes),
            "parameters": run_cfg.to_dict(),
            "calibration_metrics": calibration_metrics.get("metrics", {}),
            "final_eval_metrics": eval_result.get("metrics", {}),
            "error_buckets": eval_result.get("error_buckets", {}),
            "prediction_histogram": eval_result.get("prediction_histogram", {}),
            "selection_split": calibration_info["selection_split"],
            "support_selection_split": calibration_info["selection_split"],
            "support_fit_sample_count": len(fit_samples),
            "support_selection_sample_count": len(selection_samples),
            "calibration_holdout": calibration_info,
            "used_target_test_for_calibration": False,
        }
        iterations.append(iteration)
        _append_jsonl(scene_dir / "iteration_report.jsonl", iteration)
        score = float(calibration_metrics.get("metrics", {}).get("top1_avg", -1.0))
        if best is None or score > float(best.get("calibration_score", -1.0)):
            best = {"calibration_score": score, "run_id": run_id, "metrics": eval_result.get("metrics", {})}
    write_json(scene_dir / "iteration_report.json", {"iterations": iterations})
    return {
        "target_scene": target_scene,
        "source_scenes": list(source_scenes),
        "status": "completed",
        "run_count": len(iterations),
        "best_by_calibration": best,
        "scene_dir": str(scene_dir),
        "iterations": iterations,
    }


def _evaluate_samples(
    samples: list[GpsWindowSample],
    cfg: GpsWindowBaselineConfig,
    *,
    data_root: Path,
    output_dir: Path,
    calibration_samples: list[GpsWindowSample],
    write_artifacts: bool,
    calibration_split_label: str | None = None,
    selection_split_label: str | None = None,
    evaluation_split_label: str | None = None,
    selection_sample_count: int = 0,
) -> dict[str, Any]:
    calibration = build_calibration_state(calibration_samples, cfg)
    predictions = [predict_sample(sample, cfg, calibration) for sample in samples]
    if not predictions:
        return {"metrics": {"sample_count": 0}, "metadata": calibration.to_dict()}
    outputs = torch.stack([item.scores for item in predictions], dim=0)
    labels = torch.tensor([list(sample.target_beams[: int(cfg.horizon)]) for sample in samples], dtype=torch.long)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    metrics = _metrics(outputs, labels, cfg=cfg)
    pred = outputs.argmax(dim=-1)
    power_np = load_beam_power_vectors(samples, data_root=data_root, horizon=cfg.horizon, num_classes=cfg.num_classes)
    power = torch.from_numpy(power_np) if power_np is not None else None
    metrics.update(beam_power_metrics(pred[:, 0], labels[:, 0], power))
    fallback_count = sum(1 for item in predictions if item.fallback_status != "none")
    coverage = [float(item.gps_coverage) for item in predictions]
    guard = guard_no_target_oracle(
        split=samples[0].split if samples else "",
        phase="prediction",
        used_fields=ALLOWED_PREDICTION_FIELDS,
        calibration_split=calibration_split_label or ("source" if cfg.calibration_mode == "source" else "target_adapt_support"),
    )
    calibration_split = calibration_split_label or ("source" if cfg.calibration_mode == "source" else "target_adapt_support")
    metadata = GpsWindowRunMetadata(
        config=cfg.to_dict(),
        used_fields=tuple(ALLOWED_PREDICTION_FIELDS),
        used_target_oracle_fields=tuple(guard["used_target_oracle_fields"]),
        eligible_for_main_claim=bool(guard["eligible_for_main_claim"]),
        ineligible_reason=guard["ineligible_reason"],
        calibration_split=calibration_split,
        calibration_sample_count=len(calibration_samples),
        selection_split=selection_split_label,
        selection_sample_count=int(selection_sample_count),
        evaluation_split=evaluation_split_label,
        used_target_test_for_calibration=False,
    ).to_dict()
    metrics.update(
        {
            "sample_count": len(samples),
            "algorithm": cfg.algorithm,
            "algorithm_parameters": cfg.to_dict(),
            "gps_coverage": float(sum(coverage) / max(len(coverage), 1)),
            "fallback_count": int(fallback_count),
            "fallback_rate": float(fallback_count / max(len(predictions), 1)),
            "calibration_state": calibration.to_dict(),
            "effective_beam_direction": int(calibration.beam_direction),
            "effective_beam_offset": int(calibration.beam_offset),
            "effective_boresight_angle_degrees": float(calibration.boresight_angle_degrees),
            "beam_mapping_score": float(calibration.beam_mapping_score),
            "boresight_score": float(calibration.boresight_score),
            "calibration_split": calibration_split,
            "calibration_sample_count": len(calibration_samples),
            "selection_split": selection_split_label,
            "selection_sample_count": int(selection_sample_count),
            "evaluation_split": evaluation_split_label,
            "oracle_guard": guard,
            "used_target_oracle_fields": metadata["used_target_oracle_fields"],
        }
    )
    histogram = prediction_histogram(pred, labels, num_classes=cfg.num_classes)
    collapse = collapse_diagnostics(pred, labels, num_classes=cfg.num_classes)
    buckets = error_buckets(pred, labels, num_classes=cfg.num_classes)
    if write_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "metrics.json", metrics | {"run_metadata": metadata})
        write_json(output_dir / "prediction_hist.json", histogram)
        write_json(output_dir / "collapse_diagnostics.json", collapse)
        write_predictions_csv(output_dir / "predictions.csv", samples, predictions, labels=labels, top_k=cfg.neighbor_top_k)
        write_json(output_dir / "run_metadata.json", metadata)
    return {
        "metrics": metrics,
        "metadata": metadata,
        "prediction_histogram": histogram,
        "collapse_diagnostics": collapse,
        "error_buckets": buckets,
    }


def _metrics(outputs: torch.Tensor, labels: torch.Tensor, *, cfg: GpsWindowBaselineConfig) -> dict[str, Any]:
    topk, totals = calculate_topk_accuracy(outputs, labels, k_values=(1, 3, 5))
    dba = calculate_dba_score(outputs, labels)
    group_metrics = calculate_beam_group_metrics(outputs, labels, group_size=cfg.group_size, num_classes=cfg.num_classes)
    payload = {
        "top1_by_horizon": _float_list(topk[1]),
        "top3_by_horizon": _float_list(topk[3]),
        "top5_by_horizon": _float_list(topk[5]),
        "top1_avg": float(np.mean(topk[1])) if len(topk[1]) else 0.0,
        "top3_avg": float(np.mean(topk[3])) if len(topk[3]) else 0.0,
        "top5_avg": float(np.mean(topk[5])) if len(topk[5]) else 0.0,
        "dba_by_horizon": _float_list(dba),
        "dba_avg": float(np.mean(dba)) if len(dba) else 0.0,
        "valid_by_horizon": [int(item) for item in totals.tolist()],
    }
    payload.update(group_metrics)
    return payload


def _calibration_samples(
    *,
    data_root: Path,
    target_scene: str,
    source_scenes: list[str],
    baseline_cfg: GpsWindowBaselineConfig,
    split_tag: str,
) -> list[GpsWindowSample]:
    if baseline_cfg.calibration_mode == "target_adapt":
        samples = load_samples_from_csv(
            split_csv_path(data_root, target_scene, "train", split_tag=split_tag),
            scenario=target_scene,
            split="target_adapt_support",
            cfg=baseline_cfg,
            max_samples=baseline_cfg.support_samples or baseline_cfg.max_samples,
        )
        return samples[: int(baseline_cfg.support_samples)] if baseline_cfg.support_samples else samples
    result: list[GpsWindowSample] = []
    for scene in source_scenes:
        result.extend(
            load_samples_from_csv(
                split_csv_path(data_root, scene, "train", split_tag=split_tag),
                scenario=scene,
                split="source_calibration",
                cfg=baseline_cfg,
                max_samples=baseline_cfg.max_samples,
            )
        )
    return result


def _split_calibration_support(
    samples: list[GpsWindowSample],
    baseline_cfg: GpsWindowBaselineConfig,
) -> tuple[list[GpsWindowSample], list[GpsWindowSample], dict[str, Any]]:
    return split_calibration_support(
        samples,
        calibration_mode=baseline_cfg.calibration_mode,
        holdout_fraction=baseline_cfg.calibration_holdout_fraction,
        holdout_min_samples=baseline_cfg.calibration_holdout_min_samples,
        holdout_strategy=baseline_cfg.calibration_holdout_strategy,
    )


def _parameter_grid(cfg: dict[str, Any], baseline_cfg: GpsWindowBaselineConfig, *, sweep: bool) -> list[dict[str, Any]]:
    sweep_cfg = cfg.get("sweep", {}) if isinstance(cfg.get("sweep"), dict) else {}
    if not sweep and not sweep_cfg.get("enabled", False):
        return [baseline_cfg.to_dict()]
    keys = [
        "history_window",
        "smoothing_window",
        "velocity_decay",
        "score_width",
        "fallback",
        "fallback_weight",
        "beam_offset",
        "beam_direction",
        "boresight_angle_degrees",
        "auto_calibrate_boresight_angle",
        "auto_calibrate_beam_mapping",
        "angle_smoothing",
        "algorithm",
        "calibration_holdout_strategy",
    ]
    values = {key: sweep_cfg.get(key, [baseline_cfg.to_dict()[key]]) for key in keys}
    normalized = {key: value if isinstance(value, list) else [value] for key, value in values.items()}
    grid = [dict(zip(keys, combo)) for combo in itertools.product(*(normalized[key] for key in keys))]
    max_runs = sweep_cfg.get("max_runs")
    if max_runs is not None:
        grid = grid[: int(max_runs)]
    return grid


def _build_plan(
    cfg: dict[str, Any],
    *,
    data_root: Path,
    output_dir: Path,
    baseline_cfg: GpsWindowBaselineConfig,
    target_scenes: list[str],
    source_scenes: list[str],
    split_tag: str,
    grid: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": "gps_window_baseline_plan_v1",
        "algorithm": baseline_cfg.algorithm,
        "parameters": baseline_cfg.to_dict(),
        "parameter_grid": grid,
        "scenes": list(target_scenes),
        "source_scenes": list(source_scenes),
        "target_scenes": list(target_scenes),
        "split": {
            "split_tag": split_tag,
            "calibration_mode": baseline_cfg.calibration_mode,
            "calibration_holdout_fraction": baseline_cfg.calibration_holdout_fraction,
            "calibration_holdout_min_samples": baseline_cfg.calibration_holdout_min_samples,
            "calibration_holdout_strategy": baseline_cfg.calibration_holdout_strategy,
        },
        "claim_scope": baseline_cfg.claim_scope,
        "output_dir": str(output_dir),
        "data_root": str(data_root),
        "uses_neural_network": False,
        "uses_checkpoint": False,
        "plan_only_evaluates": False,
        "config": cfg,
    }


def _summary(scene_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for result in scene_results:
        best = result.get("best_by_calibration") or {}
        metrics = best.get("metrics") if isinstance(best.get("metrics"), dict) else {}
        rows.append(
            {
                "target_scene": result.get("target_scene"),
                "status": result.get("status"),
                "run_count": result.get("run_count", 0),
                "sample_count": metrics.get("sample_count", 0),
                "top1_avg": metrics.get("top1_avg"),
                "top3_avg": metrics.get("top3_avg"),
                "top5_avg": metrics.get("top5_avg"),
                "dba_avg": metrics.get("dba_avg"),
                "gps_coverage": metrics.get("gps_coverage"),
                "fallback_rate": metrics.get("fallback_rate"),
                "majority_last_transition_comparison": {
                    "available": False,
                    "reason": "not_run_as_separate_comparison_in_this_summary",
                },
                "best_run_id": best.get("run_id"),
            }
        )
    return {
        "version": "gps_window_baseline_summary_v1",
        "scene_count": len(rows),
        "cross_scene_claim_allowed": False,
        "claim_scope_note": "Per-scenario GPS-only baseline; do not claim town/weather generalization from sunny Town10 alone.",
        "scenes": rows,
    }


def _next_candidate_summary(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        iterations,
        key=lambda item: float(item.get("calibration_metrics", {}).get("top1_avg", -1.0)),
        reverse=True,
    )
    if not ranked:
        return {"available": False, "reason": "no_comparable_iterations"}
    best = ranked[0]
    params = dict(best.get("parameters", {}))
    buckets = best.get("error_buckets", {}) if isinstance(best.get("error_buckets"), dict) else {}
    suggestion = dict(params)
    if int(buckets.get("far_gt5", 0)) > int(buckets.get("near_1_2", 0)):
        suggestion["beam_offset"] = int(params.get("beam_offset", 0)) + 1
    suggestion["score_width"] = max(1.0, float(params.get("score_width", 2.0)) * 0.9)
    return {
        "available": True,
        "basis": "calibration_ranking_and_error_buckets",
        "best_run_id": best.get("run_id"),
        "best_calibration_top1_avg": best.get("calibration_metrics", {}).get("top1_avg"),
        "candidate_parameters": suggestion,
        "target_test_claim_note": "Candidate is not selected from target_test metrics.",
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _float_list(values: Any) -> list[float]:
    return [float(item) for item in list(values)]
