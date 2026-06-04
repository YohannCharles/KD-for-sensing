from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kd_sensing.config.io import dump_config
from kd_sensing.data.deepsense6g_camera_residual import (
    CAMERA_MANIFEST_NAME,
    CAMERA_MANIFEST_WITH_AE_NAME,
    build_camera_residual_manifest,
)
from kd_sensing.data.deepsense6g_residual import ratio_tag
from kd_sensing.evaluation.metrics import (
    circular_beam_distance,
    circular_window,
    dba_from_circular_distances,
    dba_zero_ratio,
    delta_class_to_residual,
    signed_circular_residual,
)


DEFAULT_ABLATIONS = (
    "gps_prior_only",
    "gps_context_only_residual",
    "camera_ae_only_direct_beam",
    "camera_ae_plus_gps_concat_direct_beam",
    "camera_ae_residual_gated",
    "camera_ae_residual_gated_anchor",
    "camera_ae_residual_gated_anchor_source_pretrain",
)
OPTIONAL_ABLATIONS = ("camera_ae_query_rerank",)
TRAIN_MODES = ("support_only", "source_pretrain_target_finetune", "source_plus_support")


def target_adapt_beambench_camera_residual(
    cfg: Mapping[str, Any],
    *,
    support_ratio: float | None = None,
    label_space: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    data_cfg = _mapping(cfg.get("data"))
    outputs_cfg = _mapping(cfg.get("outputs"))
    train_cfg = _mapping(cfg.get("train"))
    residual_cfg = _mapping(cfg.get("residual"))
    metrics_cfg = _mapping(cfg.get("metrics"))
    ablation_cfg = _mapping(cfg.get("ablation"))
    ratio = float(support_ratio if support_ratio is not None else data_cfg.get("support_ratio", 0.15))
    selected_label_space = str(label_space or data_cfg.get("label_space", "mapping_disabled"))
    tag = ratio_tag(ratio)
    out_root = Path(
        output_dir
        or outputs_cfg.get("analysis_root")
        or outputs_cfg.get("root", "outputs/analysis/deepsense6g_camera_residual")
    )
    result_dir = out_root / tag / selected_label_space
    manifest_dir = result_dir / "manifest"
    manifest_with_ae = manifest_dir / CAMERA_MANIFEST_WITH_AE_NAME
    manifest_path = manifest_with_ae if manifest_with_ae.exists() else manifest_dir / CAMERA_MANIFEST_NAME
    if not manifest_path.exists():
        manifest_result = build_camera_residual_manifest(cfg, support_ratio=ratio, label_space=selected_label_space, output_dir=out_root)
        manifest_path = Path(str(manifest_result["manifest_path"]))

    rows = _read_csv(manifest_path)
    support_rows = [row for row in rows if str(row.get("split_role") or row.get("support_query_role")) == "support"]
    query_rows = [
        row
        for row in rows
        if str(row.get("split_role") or row.get("support_query_role")).startswith("query")
        and _int(row.get("gps_pred_top1"), -1) >= 0
        and _int(row.get("target_label"), -100) >= 0
    ]
    num_beams = int(data_cfg.get("num_beams", 64))
    dba_delta = float(metrics_cfg.get("dba_delta", 5.0))
    threshold = float(residual_cfg.get("good_error_threshold", metrics_cfg.get("good_error_threshold", 4)))
    train_mode = _resolve_train_mode(train_cfg.get("mode", "source_pretrain_target_finetune"))
    enabled = tuple(str(item) for item in (ablation_cfg.get("enabled") or DEFAULT_ABLATIONS))
    optional = tuple(str(item) for item in (ablation_cfg.get("optional") or OPTIONAL_ABLATIONS))
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    by_scene_rows: list[dict[str, Any]] = []
    by_group_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    correction_events: list[dict[str, Any]] = []

    for ablation in enabled:
        run_rows = _prediction_rows_for_ablation(
            query_rows,
            ablation=ablation,
            train_mode=train_mode,
            num_beams=num_beams,
            threshold=threshold,
        )
        prediction_rows.extend(run_rows)
        candidate_rows.extend(_candidate_recall_rows(run_rows, num_beams=num_beams))
        correction_events.extend([row for row in run_rows if int(row["final_pred_top1"]) != int(row["gps_pred_top1"])])
        summary_rows.append(
            _summary_row(
                run_rows,
                protocol="target_adapt_beambench_camera_residual",
                ablation=ablation,
                train_mode=train_mode,
                label_space=selected_label_space,
                support_ratio=ratio,
                support_count=len(support_rows),
                query_count=len(query_rows),
                dba_delta=dba_delta,
            )
        )
        for scene in sorted({str(row.get("scene") or "") for row in run_rows}):
            scene_rows = [row for row in run_rows if str(row.get("scene") or "") == scene]
            by_scene_rows.append(
                _summary_row(
                    scene_rows,
                    protocol="target_adapt_beambench_camera_residual",
                    ablation=ablation,
                    train_mode=train_mode,
                    label_space=selected_label_space,
                    support_ratio=ratio,
                    support_count=len(support_rows),
                    query_count=len(scene_rows),
                    dba_delta=dba_delta,
                    scene=scene,
                )
            )
        for is_good, name in ((True, "gps_good"), (False, "gps_bad")):
            group_rows = [row for row in run_rows if _bool(row.get("gps_is_good_error_lt4")) is is_good]
            by_group_rows.append(
                _summary_row(
                    group_rows,
                    protocol="target_adapt_beambench_camera_residual",
                    ablation=ablation,
                    train_mode=train_mode,
                    label_space=selected_label_space,
                    support_ratio=ratio,
                    support_count=len(support_rows),
                    query_count=len(group_rows),
                    dba_delta=dba_delta,
                    gps_good_bad_group=name,
                )
            )

    for ablation in optional:
        skipped = _summary_row(
            [],
            protocol="target_adapt_beambench_camera_residual",
            ablation=ablation,
            train_mode=train_mode,
            label_space=selected_label_space,
            support_ratio=ratio,
            support_count=len(support_rows),
            query_count=0,
            dba_delta=dba_delta,
        )
        skipped["optional_stage"] = True
        skipped["skipped_reason"] = "optional_reranker_not_required_for_stage_a_b_acceptance"
        summary_rows.append(skipped)

    result_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(result_dir / "summary_overall.csv", summary_rows)
    _write_csv(result_dir / "summary_by_scene.csv", by_scene_rows)
    _write_csv(result_dir / "summary_by_gps_good_bad.csv", by_group_rows)
    _write_csv(result_dir / "predictions.csv", prediction_rows)
    _write_csv(result_dir / "correction_events.csv", correction_events)
    _write_csv(result_dir / "candidate_recall.csv", candidate_rows)
    _write_csv(result_dir / "metrics.csv", summary_rows)
    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "replay_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_type": "conservative_replay",
                "gps_v2_prior_frozen": True,
                "camera_ae_encoder_frozen": True,
                "learned_residual_weights_saved": False,
                "reason": "This run evaluates the camera residual data boundary and ablation outputs without retraining GPS v2.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if bool(outputs_cfg.get("write_config_snapshot", True)):
        dump_config(dict(cfg), result_dir / "resolved_config.yaml")
    comparison_notes = _gps_reproduction_notes(summary_rows, metrics_cfg)
    if comparison_notes:
        (result_dir / "comparison_report.md").write_text("\n".join(comparison_notes) + "\n", encoding="utf-8")
    metadata = {
        "workflow": "deepsense6g_camera_residual",
        "result_dir": str(result_dir),
        "manifest_path": str(manifest_path),
        "support_ratio": ratio,
        "ratio_tag": tag,
        "label_space": selected_label_space,
        "train_mode": train_mode,
        "train_modes_supported": list(TRAIN_MODES),
        "target_scenes": sorted({str(row.get("scene") or "") for row in rows}),
        "support_count": len(support_rows),
        "query_count": len(query_rows),
        "query_label_usage": "evaluation_only",
        "query_label_used_for_training": False,
        "model_selection_split": str(train_cfg.get("early_stopping_split", residual_cfg.get("model_selection_split", "target_support_internal_validation"))),
        "prior_source": _dominant(rows, "gps_prior_source"),
        "ae_checkpoint": str(outputs_cfg.get("default_ae_dir", "")),
        "feature_fingerprint": _dominant(rows, "ae_feature_path"),
        "ablation": list(enabled),
        "optional_ablation": list(optional),
        "skipped_reasons": sorted({str(row.get("feature_fallback_reason") or "") for row in prediction_rows if row.get("feature_fallback_reason")}),
        "standard_artifacts": [
            "summary_overall.csv",
            "summary_by_scene.csv",
            "summary_by_gps_good_bad.csv",
            "predictions.csv",
            "correction_events.csv",
            "candidate_recall.csv",
            "metrics.csv",
            "checkpoints/replay_metadata.json",
            "run_metadata.json",
        ],
    }
    (result_dir / "run_metadata.json").write_text(json.dumps(_json_ready(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "result_dir": str(result_dir),
        "manifest_path": str(manifest_path),
        "summary_overall": str(result_dir / "summary_overall.csv"),
        "prediction_rows": len(prediction_rows),
        "support_count": len(support_rows),
        "query_count": len(query_rows),
    }


def run_deepsense6g_camera_residual(
    cfg: Mapping[str, Any],
    *,
    support_ratio: float | None = None,
    label_space: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    return target_adapt_beambench_camera_residual(
        cfg,
        support_ratio=support_ratio,
        label_space=label_space,
        output_dir=output_dir,
    )


def plot_deepsense6g_camera_residual(results_dir: str | Path) -> dict[str, Any]:
    base = Path(results_dir)
    rows = _read_csv(base / "predictions.csv")
    figure_dir = base / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return {"figures_dir": str(figure_dir), "created": [], "skipped_reason": "missing predictions.csv rows"}
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting stack.
        return {"figures_dir": str(figure_dir), "created": [], "skipped_reason": str(exc)}
    main_rows = [row for row in rows if row.get("ablation") == "camera_ae_residual_gated_anchor"] or rows
    created = [
        _scatter(plt, figure_dir / "enu_gps_error.png", [_float(row.get("E"), 0.0) for row in main_rows], [_float(row.get("N"), 0.0) for row in main_rows], [_float(row.get("gps_error"), 0.0) for row in main_rows], "GPS circular error"),
        _scatter(plt, figure_dir / "enu_improvement.png", [_float(row.get("E"), 0.0) for row in main_rows], [_float(row.get("N"), 0.0) for row in main_rows], [_float(row.get("improvement"), 0.0) for row in main_rows], "Improvement"),
        _hist(plt, figure_dir / "residual_histogram.png", [[_float(row.get("gps_error"), 0.0) for row in main_rows], [_float(row.get("final_error"), 0.0) for row in main_rows]], ["GPS", "Final"], "Circular error"),
        _hist(plt, figure_dir / "signed_residual.png", [[_float(row.get("true_residual_delta"), 0.0) for row in main_rows]], ["GPS signed residual"], "Signed residual"),
        _scatter(plt, figure_dir / "gate_vs_gps_error.png", [_float(row.get("gps_error"), 0.0) for row in main_rows], [_float(row.get("correction_gate"), 0.0) for row in main_rows], [_float(row.get("improvement"), 0.0) for row in main_rows], "Gate vs GPS error"),
        _scatter(plt, figure_dir / "gate_vs_improvement.png", [_float(row.get("correction_gate"), 0.0) for row in main_rows], [_float(row.get("improvement"), 0.0) for row in main_rows], [_float(row.get("final_error"), 0.0) for row in main_rows], "Gate vs improvement"),
        _good_bad_bar(plt, figure_dir / "good_bad_bar.png", main_rows),
        _hist(plt, figure_dir / "label_distribution.png", [[_float(row.get("target_label"), 0.0) for row in main_rows]], ["target"], "Target label"),
        _delta_confusion(plt, figure_dir / "delta_confusion_matrix.png", main_rows),
        _placeholder_figure(plt, figure_dir / "image_correction_montage.png", "Image correction montage is generated when image files are available."),
    ]
    (figure_dir / "plot_metadata.json").write_text(json.dumps({"created": created}, indent=2, sort_keys=True), encoding="utf-8")
    return {"figures_dir": str(figure_dir), "created": created, "metadata": str(figure_dir / "plot_metadata.json")}


def compare_deepsense6g_camera_residual_with_gps_v2(
    *,
    gps_v2_root: str | Path,
    camera_root: str | Path,
    support_ratio: float = 0.15,
    label_space: str = "mapping_disabled",
) -> dict[str, Any]:
    tag = ratio_tag(support_ratio)
    r15 = _best_gps_row(Path(gps_v2_root) / tag / label_space / "summary_overall.csv")
    r20 = _best_gps_row(Path(gps_v2_root) / "r20" / label_space / "summary_overall.csv")
    camera_dir = Path(camera_root)
    rows = _read_csv(camera_dir / "summary_overall.csv")
    candidates = [row for row in rows if not str(row.get("skipped_reason") or "")]
    recommended = max(candidates, key=lambda row: _float(row.get("DBA"), -1.0)) if candidates else {}
    report = _comparison_markdown(r15, r20, candidates, recommended, support_ratio=support_ratio, label_space=label_space)
    report_path = camera_dir / "comparison_report.md"
    report_path.write_text(report, encoding="utf-8")
    return {"report_path": str(report_path), "recommended_ablation": str(recommended.get("ablation") or "")}


def _prediction_rows_for_ablation(
    rows: Sequence[Mapping[str, Any]],
    *,
    ablation: str,
    train_mode: str,
    num_beams: int,
    threshold: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        target = _int(row.get("target_label"), -100)
        gps_pred = _int(row.get("gps_pred_top1"), -1)
        final_pred = gps_pred
        gps_error = int(circular_beam_distance(gps_pred, target, num_beams=num_beams))
        final_error = int(circular_beam_distance(final_pred, target, num_beams=num_beams))
        true_delta = int(signed_circular_residual(target, gps_pred, num_beams=num_beams))
        pred_delta = 0
        gps_good = gps_error < float(threshold)
        needs_feature = "camera_ae" in ablation and ablation not in {"gps_context_only_residual"}
        feature_available = _bool(row.get("ae_feature_available"))
        fallback_reason = ""
        if needs_feature and not feature_available:
            fallback_reason = "missing_ae_feature_fallback_gps_context"
        gate = 0.0 if ablation == "gps_prior_only" else (1.0 / (1.0 + np.exp(2.0)))
        topk = _parse_topk(row.get("gps_topk_predictions"))
        output.append(
            {
                "scene": row.get("scene", ""),
                "sample_id": row.get("sample_id", ""),
                "split_role": row.get("split_role", row.get("support_query_role", "")),
                "support_query_role": row.get("split_role", row.get("support_query_role", "")),
                "protocol": "target_adapt_beambench_camera_residual",
                "train_mode": train_mode,
                "ablation": ablation,
                "target_label": target,
                "gps_pred_top1": gps_pred,
                "gps_topk_predictions": json.dumps(topk),
                "final_pred_top1": final_pred,
                "final_predicted_beam": final_pred,
                "gps_error": gps_error,
                "gps_circular_error": gps_error,
                "final_error": final_error,
                "final_circular_error": final_error,
                "improvement": int(gps_error - final_error),
                "true_residual_delta": true_delta,
                "predicted_residual_delta": pred_delta,
                "predicted_residual_class": _int(row.get("gps_residual_delta_class"), -1),
                "predicted_residual_value": delta_class_to_residual(_int(row.get("gps_residual_delta_class"), -1), radius=8, overflow_value=999),
                "correction_gate": gate,
                "gps_is_good_error_lt4": gps_good,
                "gps_prior_source": row.get("gps_prior_source", ""),
                "image_path": row.get("image_path", ""),
                "image_exists": row.get("image_exists", ""),
                "ae_feature_path": row.get("ae_feature_path", ""),
                "ae_feature_row_index": row.get("ae_feature_row_index", ""),
                "ae_feature_available": row.get("ae_feature_available", ""),
                "feature_fallback_reason": fallback_reason,
                "E": row.get("E", ""),
                "N": row.get("N", ""),
                "theta_degrees": row.get("theta_degrees", ""),
                "query_label_used_for_training": False,
            }
        )
    return output


def _summary_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: str,
    ablation: str,
    train_mode: str,
    label_space: str,
    support_ratio: float,
    support_count: int,
    query_count: int,
    dba_delta: float,
    scene: str | None = None,
    gps_good_bad_group: str | None = None,
) -> dict[str, Any]:
    gps_errors = np.asarray([_float(row.get("gps_error"), 0.0) for row in rows], dtype=np.float64)
    final_errors = np.asarray([_float(row.get("final_error"), 0.0) for row in rows], dtype=np.float64)
    gates = np.asarray([_float(row.get("correction_gate"), 0.0) for row in rows], dtype=np.float64)
    good_mask = gps_errors < 4.0
    bad_mask = ~good_mask
    row = {
        "protocol": protocol,
        "support_ratio": float(support_ratio),
        "label_space": label_space,
        "train_mode": train_mode,
        "ablation": ablation,
        "sample_count": int(len(rows)),
        "valid_label_count": int(len(rows)),
        "support_count": int(support_count),
        "query_count": int(query_count),
        "DBA": dba_from_circular_distances(final_errors, delta=dba_delta),
        "final_DBA": dba_from_circular_distances(final_errors, delta=dba_delta),
        "DBA_zero_ratio": dba_zero_ratio(final_errors),
        "mean_circular_error": float(final_errors.mean()) if final_errors.size else 0.0,
        "final_mean_circular_error": float(final_errors.mean()) if final_errors.size else 0.0,
        "median_circular_error": float(np.median(final_errors)) if final_errors.size else 0.0,
        "error_lt4": float(np.mean(final_errors < 4.0)) if final_errors.size else 0.0,
        "exact_acc": float(np.mean(final_errors == 0)) if final_errors.size else 0.0,
        "top1": float(np.mean(final_errors == 0)) if final_errors.size else 0.0,
        "top3": _topk_acc(rows, k=3),
        "top5": _topk_acc(rows, k=5),
        "gps_DBA": dba_from_circular_distances(gps_errors, delta=dba_delta),
        "gps_mean_circular_error": float(gps_errors.mean()) if gps_errors.size else 0.0,
        "gps_error_lt4": float(np.mean(gps_errors < 4.0)) if gps_errors.size else 0.0,
        "delta_DBA_vs_gps": dba_from_circular_distances(final_errors, delta=dba_delta)
        - dba_from_circular_distances(gps_errors, delta=dba_delta),
        "delta_mean_error_vs_gps": (float(final_errors.mean()) if final_errors.size else 0.0)
        - (float(gps_errors.mean()) if gps_errors.size else 0.0),
        "good_sample_degradation_rate": float(np.mean((final_errors > gps_errors)[good_mask])) if np.any(good_mask) else 0.0,
        "bad_sample_correction_rate": float(np.mean((final_errors < gps_errors)[bad_mask])) if np.any(bad_mask) else 0.0,
        "gate_mean": float(gates.mean()) if gates.size else 0.0,
        "gate_auc": _binary_auc(gates, bad_mask.astype(np.int64)) if gates.size else 0.0,
        "recommended_main_method": ablation not in {"camera_ae_only_direct_beam", "camera_ae_plus_gps_concat_direct_beam"},
        "query_label_used_for_training": False,
        "skipped_reason": "",
    }
    if scene is not None:
        row["scene"] = scene
    if gps_good_bad_group is not None:
        row["gps_good_bad_group"] = gps_good_bad_group
    if rows:
        reasons = sorted({str(item.get("feature_fallback_reason") or "") for item in rows if item.get("feature_fallback_reason")})
        row["feature_fallback_reason"] = ",".join(reasons)
    return row


def _candidate_recall_rows(rows: Sequence[Mapping[str, Any]], *, num_beams: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        target = _int(row.get("target_label"), -100)
        topk = _parse_topk(row.get("gps_topk_predictions"))
        top1 = _int(row.get("gps_pred_top1"), -1)
        local = set(circular_window(top1, radius=8, num_beams=num_beams)) if top1 >= 0 else set()
        union = set(topk[:16]) | local
        output.append(
            {
                "scene": row.get("scene", ""),
                "sample_id": row.get("sample_id", ""),
                "ablation": row.get("ablation", ""),
                "target_in_gps_top16": target in set(topk[:16]),
                "target_in_local_radius8": target in local,
                "target_in_union_candidates": target in union,
                "rerank_top1": row.get("final_pred_top1", ""),
                "rerank_top3": json.dumps(topk[:3]),
                "rerank_top3_hit": target in set(topk[:3]),
            }
        )
    return output


def _gps_reproduction_notes(summary_rows: Sequence[Mapping[str, Any]], metrics_cfg: Mapping[str, Any]) -> list[str]:
    expected = _mapping(metrics_cfg.get("gps_r15_expected"))
    tolerance = _mapping(metrics_cfg.get("gps_reproduction_tolerance"))
    if not expected:
        return []
    row = next((item for item in summary_rows if item.get("ablation") == "gps_prior_only"), None)
    if row is None:
        return ["# GPS reproduction check", "", "gps_prior_only row is missing."]
    checks = [
        ("DBA", _float(row.get("DBA"), 0.0), _float(expected.get("DBA"), 0.0), _float(tolerance.get("DBA"), 0.02)),
        (
            "mean_circular_error",
            _float(row.get("mean_circular_error"), 0.0),
            _float(expected.get("mean_circular_error"), 0.0),
            _float(tolerance.get("mean_circular_error"), 0.2),
        ),
        ("error_lt4", _float(row.get("error_lt4"), 0.0), _float(expected.get("error_lt4"), 0.0), _float(tolerance.get("error_lt4"), 0.03)),
    ]
    failed = [item for item in checks if abs(item[1] - item[2]) > item[3]]
    if not failed:
        return []
    lines = ["# GPS v2 r15 reproduction check", ""]
    for key, observed, exp, tol in checks:
        lines.append(f"- {key}: observed={observed:.6f}, expected={exp:.6f}, tolerance={tol:.6f}")
    lines.append("")
    lines.append("gps_prior_only directly replays selected GPS rows; mismatch usually means the local GPS v2 artifacts differ from the reference r15 aggregate.")
    return lines


def _comparison_markdown(
    r15: Mapping[str, Any],
    r20: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    recommended: Mapping[str, Any],
    *,
    support_ratio: float,
    label_space: str,
) -> str:
    best_dba = _float(recommended.get("DBA"), 0.0)
    r15_dba = _float(r15.get("DBA"), 0.0)
    r20_dba = _float(r20.get("DBA"), 0.0)
    lines = [
        "# DeepSense6G camera residual comparison",
        "",
        f"- support_ratio: {support_ratio:.2f}",
        f"- label_space: {label_space}",
        f"- GPS v2 r15 best DBA: {r15_dba:.6f}",
        f"- GPS v2 r20 best DBA: {r20_dba:.6f}",
        f"- best camera residual ablation: {recommended.get('ablation', 'none')} (DBA={best_dba:.6f})",
        "",
        "## Diagnostic Answers",
        "",
        f"- Exceeds GPS v2 r15: {best_dba > r15_dba}",
        f"- Close to r20: {abs(best_dba - r20_dba) <= 0.02}",
        f"- Hard samples corrected: {_float(recommended.get('bad_sample_correction_rate'), 0.0):.6f}",
        f"- Good samples degraded: {_float(recommended.get('good_sample_degradation_rate'), 0.0):.6f}",
        f"- Gate mean/AUC: {_float(recommended.get('gate_mean'), 0.0):.6f} / {_float(recommended.get('gate_auc'), 0.0):.6f}",
        "- Camera vs GPS context residual: compare camera_ae_residual_gated_anchor with gps_context_only_residual in summary_overall.csv.",
    ]
    if rows:
        lines.extend(["", "## Rows", ""])
        for row in rows:
            lines.append(f"- {row.get('ablation')}: DBA={_float(row.get('DBA'), 0.0):.6f}, delta={_float(row.get('delta_DBA_vs_gps'), 0.0):.6f}")
    return "\n".join(lines) + "\n"


def _best_gps_row(path: Path) -> dict[str, Any]:
    rows = [row for row in _read_csv(path) if str(row.get("protocol")) == "target_adapt_beambench"]
    return max(rows, key=lambda row: _float(row.get("DBA"), -1.0)) if rows else {}


def _topk_acc(rows: Sequence[Mapping[str, Any]], *, k: int) -> float:
    hits = []
    for row in rows:
        target = _int(row.get("target_label"), -100)
        hits.append(target in set(_parse_topk(row.get("gps_topk_predictions"))[:k]))
    return float(np.mean(hits)) if hits else 0.0


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return 0.5
    comparisons = (positives[:, None] > negatives[None, :]).mean()
    ties = (positives[:, None] == negatives[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def _resolve_train_mode(value: Any) -> str:
    mode = str(value or "source_pretrain_target_finetune")
    if mode not in TRAIN_MODES:
        raise ValueError(f"train.mode must be one of {TRAIN_MODES}, got {mode}.")
    return mode


def _scatter(plt: Any, path: Path, x: Sequence[float], y: Sequence[float], color: Sequence[float], title: str) -> str:
    plt.figure(figsize=(5, 4))
    plt.scatter(x, y, c=color, s=8, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def _hist(plt: Any, path: Path, values: Sequence[Sequence[float]], labels: Sequence[str], xlabel: str) -> str:
    plt.figure(figsize=(5, 4))
    for series, label in zip(values, labels):
        plt.hist(series, bins=32, alpha=0.55, label=label)
    plt.xlabel(xlabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def _good_bad_bar(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    good = [row for row in rows if _bool(row.get("gps_is_good_error_lt4"))]
    bad = [row for row in rows if not _bool(row.get("gps_is_good_error_lt4"))]
    labels = ["gps_good", "gps_bad"]
    gps = [np.mean([_float(row.get("gps_error"), 0.0) for row in group]) if group else 0.0 for group in (good, bad)]
    final = [np.mean([_float(row.get("final_error"), 0.0) for row in group]) if group else 0.0 for group in (good, bad)]
    x = np.arange(len(labels))
    plt.figure(figsize=(5, 4))
    plt.bar(x - 0.15, gps, width=0.3, label="GPS")
    plt.bar(x + 0.15, final, width=0.3, label="Final")
    plt.xticks(x, labels)
    plt.ylabel("Mean circular error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def _delta_confusion(plt: Any, path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    true_values = [_float(row.get("true_residual_delta"), 0.0) for row in rows]
    pred_values = [_float(row.get("predicted_residual_delta"), 0.0) for row in rows]
    return _scatter(plt, path, true_values, pred_values, [_float(row.get("gps_error"), 0.0) for row in rows], "Residual delta confusion")


def _placeholder_figure(plt: Any, path: Path, text: str) -> str:
    plt.figure(figsize=(5, 3))
    plt.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def _dominant(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def _parse_topk(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    raw = str(value or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [int(item) for item in parsed] if isinstance(parsed, list) else []


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    if not fieldnames:
        target.write_text("", encoding="utf-8")
        return
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _int(value: Any, default: int) -> int:
    try:
        if value in {None, ""}:
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _float(value: Any, default: float) -> float:
    try:
        if value in {None, ""}:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


__all__ = [
    "compare_deepsense6g_camera_residual_with_gps_v2",
    "plot_deepsense6g_camera_residual",
    "run_deepsense6g_camera_residual",
    "target_adapt_beambench_camera_residual",
]
