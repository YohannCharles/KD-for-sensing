from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kd_sensing.config.io import dump_config
from kd_sensing.data.deepsense6g_residual import (
    FALLBACK_PRIOR_SOURCE,
    build_residual_manifest,
    inspect_residual_inputs,
    ratio_tag,
)
from kd_sensing.evaluation.metrics import (
    circular_beam_distance,
    circular_window,
    dba_from_circular_distances,
    dba_zero_ratio,
    signed_circular_residual,
)


DEFAULT_ABLATIONS = (
    "gps_prior_only",
    "gps_context_only_residual",
    "gps_plus_residual_no_gate",
    "gps_plus_residual_gated",
    "gps_plus_residual_gated_anchor",
    "gps_topk_rerank",
)
OPTIONAL_ABLATIONS = {
    "image_plus_gps_residual": ("image",),
    "lidar_plus_gps_residual": ("lidar",),
    "radar_plus_gps_residual": ("radar",),
    "all_available_modalities_residual": ("image", "lidar", "radar"),
}


def run_deepsense6g_residual_fusion(
    cfg: Mapping[str, Any],
    *,
    support_ratio: float | None = None,
    label_space: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    data_cfg = _mapping(cfg.get("data"))
    outputs_cfg = _mapping(cfg.get("outputs"))
    ablation_cfg = _mapping(cfg.get("ablation"))
    residual_cfg = _mapping(cfg.get("residual"))
    metrics_cfg = _mapping(cfg.get("metrics"))
    ratio = float(support_ratio if support_ratio is not None else data_cfg.get("support_ratio", 0.15))
    selected_label_space = str(label_space or data_cfg.get("label_space", "mapping_disabled"))
    tag = ratio_tag(ratio)
    out_root = Path(output_dir or outputs_cfg.get("root", "outputs/analysis/deepsense6g_residual_fusion"))
    result_dir = out_root / tag / selected_label_space
    manifest_path = result_dir / "manifest" / "residual_manifest.csv"
    if not manifest_path.exists():
        build_residual_manifest(cfg, support_ratio=ratio, label_space=selected_label_space, output_dir=out_root)
    rows = _read_csv(manifest_path)
    query_rows = [
        row
        for row in rows
        if str(row.get("support_query_role", "")).startswith("query")
        and _int(row.get("gps_pred_top1"), -1) >= 0
        and _int(row.get("target_label"), -100) >= 0
    ]
    support_rows = [row for row in rows if str(row.get("support_query_role")) == "support"]
    support_count = len(support_rows)
    query_count = len(query_rows)
    support_prior_available_count = sum(1 for row in support_rows if _bool(row.get("gps_prior_available")))
    query_prior_available_count = sum(1 for row in query_rows if _bool(row.get("gps_prior_available")))
    missing_support_prior_count = support_count - support_prior_available_count
    missing_query_prior_count = query_count - query_prior_available_count
    num_beams = int(data_cfg.get("num_beams", 64))
    dba_delta = float(metrics_cfg.get("dba_delta", 5.0))
    threshold = float(residual_cfg.get("good_error_threshold", metrics_cfg.get("good_error_threshold", 4)))
    enabled = tuple(ablation_cfg.get("enabled") or DEFAULT_ABLATIONS)
    optional_enabled = tuple(ablation_cfg.get("optional_modalities") or OPTIONAL_ABLATIONS)
    availability = _modality_availability(rows)
    prediction_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    correction_events: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    by_scene_rows: list[dict[str, Any]] = []
    by_group_rows: list[dict[str, Any]] = []

    for ablation in enabled:
        modalities = _modalities_for_ablation(str(ablation), availability)
        run_rows = _prediction_rows_for_ablation(
            query_rows,
            ablation=str(ablation),
            modalities=modalities,
            num_beams=num_beams,
            threshold=threshold,
        )
        prediction_rows.extend(run_rows)
        candidate_rows.extend(_candidate_recall_rows(run_rows, num_beams=num_beams))
        correction_events.extend([row for row in run_rows if int(row["gps_pred_top1"]) != int(row["final_predicted_beam"])])
        summary_rows.append(
            _summary_row(
                run_rows,
                protocol="target_adapt_beambench_residual",
                ablation=str(ablation),
                modalities=modalities,
                label_space=selected_label_space,
                support_ratio=ratio,
                support_count=support_count,
                query_count=query_count,
                dba_delta=dba_delta,
            )
        )
        by_scene_rows.extend(
            _summary_row(
                [row for row in run_rows if row["scene"] == scene],
                protocol="target_adapt_beambench_residual",
                ablation=str(ablation),
                modalities=modalities,
                label_space=selected_label_space,
                support_ratio=ratio,
                support_count=support_count,
                query_count=sum(1 for row in run_rows if row["scene"] == scene),
                dba_delta=dba_delta,
                scene=scene,
            )
            for scene in sorted({row["scene"] for row in run_rows})
        )
        by_group_rows.extend(
            _summary_row(
                [row for row in run_rows if bool(row["gps_is_good_error_lt4"]) is is_good],
                protocol="target_adapt_beambench_residual",
                ablation=str(ablation),
                modalities=modalities,
                label_space=selected_label_space,
                support_ratio=ratio,
                support_count=support_count,
                query_count=sum(1 for row in run_rows if bool(row["gps_is_good_error_lt4"]) is is_good),
                dba_delta=dba_delta,
                gps_good_bad_group="gps_good" if is_good else "gps_bad",
            )
            for is_good in (True, False)
        )

    for ablation in optional_enabled:
        modalities = ("gps_context", *OPTIONAL_ABLATIONS.get(str(ablation), ()))
        missing = [
            name
            for name in modalities
            if name != "gps_context" and not availability.get(name, {}).get("available", False)
        ]
        if missing:
            summary_rows.append(
                _skipped_summary(
                    ablation=str(ablation),
                    modalities=modalities,
                    label_space=selected_label_space,
                    support_ratio=ratio,
                    reason=f"missing_or_unstable_modalities:{','.join(missing)}",
                )
            )
            continue
        run_rows = _prediction_rows_for_ablation(
            query_rows,
            ablation=str(ablation),
            modalities=modalities,
            num_beams=num_beams,
            threshold=threshold,
        )
        prediction_rows.extend(run_rows)
        summary_rows.append(
            _summary_row(
                run_rows,
                protocol="target_adapt_beambench_residual",
                ablation=str(ablation),
                modalities=modalities,
                label_space=selected_label_space,
                support_ratio=ratio,
                support_count=support_count,
                query_count=query_count,
                dba_delta=dba_delta,
            )
        )

    upper_bound = _skipped_summary(
        ablation="within_scene_residual_upper_bound",
        modalities=("gps_context",),
        label_space=selected_label_space,
        support_ratio=ratio,
        reason="sanity upper bound protocol only; not part of main conclusion",
    )
    upper_bound["protocol"] = "within_scene_residual_upper_bound"
    upper_bound["upper_bound_protocol"] = True
    summary_rows.append(upper_bound)

    result_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(result_dir / "summary_overall.csv", summary_rows)
    _write_csv(result_dir / "summary_by_scene.csv", by_scene_rows)
    _write_csv(result_dir / "summary_by_gps_good_bad.csv", by_group_rows)
    _write_csv(result_dir / "predictions.csv", prediction_rows)
    _write_csv(
        result_dir / "correction_events.csv",
        correction_events,
        fieldnames=[
            "scene",
            "sample_id",
            "target_label",
            "gps_pred_top1",
            "final_predicted_beam",
            "gps_circular_error",
            "final_circular_error",
            "improvement",
            "correction_gate",
            "delta",
            "gps_is_good_error_lt4",
            "ablation",
        ],
    )
    _write_csv(result_dir / "candidate_recall.csv", candidate_rows)
    if bool(outputs_cfg.get("write_config_snapshot", True)):
        dump_config(dict(cfg), result_dir / "resolved_config.yaml")
    metadata = {
        "workflow": "deepsense6g_gps_residual_fusion",
        "result_dir": str(result_dir),
        "manifest_path": str(manifest_path),
        "support_ratio": ratio,
        "ratio_tag": tag,
        "label_space": selected_label_space,
        "train_mode": "support_only",
        "degraded_reason": (
            "source prior predictions incomplete for support/query split"
            if missing_support_prior_count or missing_query_prior_count
            else "conservative replay residual runner; learned residual checkpoint training is not enabled"
        ),
        "support_count": support_count,
        "query_count": query_count,
        "support_prior_available_count": support_prior_available_count,
        "query_prior_available_count": query_prior_available_count,
        "missing_support_prior_count": missing_support_prior_count,
        "missing_query_prior_count": missing_query_prior_count,
        "query_label_used_for_training": False,
        "model_selection_split": str(residual_cfg.get("model_selection_split", "target_support_internal_validation")),
        "prior_source": _dominant(query_rows, "gps_prior_source") or FALLBACK_PRIOR_SOURCE,
        "enabled_modalities": availability,
        "standard_artifacts": [
            "summary_overall.csv",
            "summary_by_scene.csv",
            "summary_by_gps_good_bad.csv",
            "predictions.csv",
            "correction_events.csv",
            "candidate_recall.csv",
            "run_metadata.json",
            "resolved_config.yaml",
        ],
    }
    (result_dir / "run_metadata.json").write_text(json.dumps(_json_ready(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "result_dir": str(result_dir),
        "manifest_path": str(manifest_path),
        "summary_overall": str(result_dir / "summary_overall.csv"),
        "prediction_rows": len(prediction_rows),
        "query_count": query_count,
        "support_count": support_count,
    }


def plot_deepsense6g_residual_fusion(results_dir: str | Path) -> dict[str, Any]:
    base = Path(results_dir)
    rows = _read_csv(base / "predictions.csv")
    figure_dir = base / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return {"figures_dir": str(figure_dir), "created": [], "skipped_reason": "missing predictions.csv rows"}
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional plotting stack.
        return {"figures_dir": str(figure_dir), "created": [], "skipped_reason": str(exc)}

    created: list[str] = []
    main_rows = [row for row in rows if row.get("ablation") == "gps_plus_residual_gated_anchor"] or rows
    easting = [_float(row.get("E"), 0.0) for row in main_rows]
    northing = [_float(row.get("N"), 0.0) for row in main_rows]
    gps_error = [_float(row.get("gps_circular_error"), 0.0) for row in main_rows]
    final_error = [_float(row.get("final_circular_error"), 0.0) for row in main_rows]
    improvement = [_float(row.get("improvement"), 0.0) for row in main_rows]
    gate = [_float(row.get("correction_gate"), 0.0) for row in main_rows]
    target = [_int(row.get("target_label"), -1) for row in main_rows]
    signed_before = [_float(row.get("gps_signed_residual"), 0.0) for row in main_rows]
    signed_after = [_float(row.get("final_signed_residual"), 0.0) for row in main_rows]

    created.append(_scatter(plt, figure_dir / "enu_gps_error.png", easting, northing, gps_error, "GPS circular error"))
    created.append(_scatter(plt, figure_dir / "enu_final_error.png", easting, northing, final_error, "Final circular error"))
    created.append(_hist(plt, figure_dir / "before_after_residual_hist.png", [gps_error, final_error], ["GPS", "Final"], "Circular error"))
    created.append(_hist(plt, figure_dir / "signed_residual_before_after.png", [signed_before, signed_after], ["Before", "After"], "Signed residual"))
    created.append(_scatter(plt, figure_dir / "gate_vs_gps_error.png", gps_error, gate, improvement, "Gate vs GPS error"))
    created.append(_scatter(plt, figure_dir / "gate_vs_improvement.png", gate, improvement, final_error, "Gate vs improvement"))
    created.append(_hist(plt, figure_dir / "label_distribution.png", [target], ["target"], "Target label"))
    created.append(_good_bad_bar(plt, figure_dir / "good_bad_bar.png", main_rows))
    notes = {"modality_visualizations": "skipped when image/lidar/radar feature arrays are unavailable in residual manifest"}
    (figure_dir / "plot_metadata.json").write_text(json.dumps(notes, indent=2, sort_keys=True), encoding="utf-8")
    return {"figures_dir": str(figure_dir), "created": created, "metadata": str(figure_dir / "plot_metadata.json")}


def compare_deepsense6g_residual_with_gps_v2(
    *,
    gps_v2_root: str | Path,
    residual_root: str | Path,
    support_ratio: float = 0.15,
    label_space: str = "mapping_disabled",
) -> dict[str, Any]:
    gps_root = Path(gps_v2_root)
    residual_dir = Path(residual_root)
    tag = ratio_tag(support_ratio)
    r15 = _best_gps_row(gps_root / tag / label_space / "summary_overall.csv")
    r20 = _best_gps_row(gps_root / "r20" / label_space / "summary_overall.csv")
    residual_rows = _read_csv(residual_dir / "summary_overall.csv")
    candidates = [row for row in residual_rows if str(row.get("protocol")) == "target_adapt_beambench_residual" and not row.get("skipped_reason")]
    recommended = _recommend(candidates)
    report = _comparison_markdown(r15, r20, candidates, recommended, support_ratio=support_ratio, label_space=label_space)
    report_path = residual_dir / "comparison_report.md"
    report_path.write_text(report, encoding="utf-8")
    return {"report_path": str(report_path), "recommended_ablation": recommended.get("ablation", "") if recommended else ""}


def _prediction_rows_for_ablation(
    rows: Sequence[Mapping[str, Any]],
    *,
    ablation: str,
    modalities: Sequence[str],
    num_beams: int,
    threshold: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        target = _int(row.get("target_label"), -100)
        gps_pred = _int(row.get("gps_pred_top1"), -1)
        final_pred = gps_pred
        gps_error = int(circular_beam_distance(gps_pred, target, num_beams=num_beams))
        final_error = int(circular_beam_distance(final_pred, target, num_beams=num_beams))
        gps_good = gps_error < float(threshold)
        topk = _parse_topk(row.get("gps_topk_predictions"))
        result.append(
            {
                "scene": row.get("scene", ""),
                "sample_id": row.get("sample_id", ""),
                "support_query_role": row.get("support_query_role", ""),
                "protocol": "target_adapt_beambench_residual",
                "train_mode": "support_only",
                "ablation": ablation,
                "modalities": ",".join(modalities),
                "target_label": target,
                "gps_pred_top1": gps_pred,
                "gps_topk_predictions": json.dumps(topk),
                "final_predicted_beam": final_pred,
                "gps_circular_error": gps_error,
                "final_circular_error": final_error,
                "gps_signed_residual": int(signed_circular_residual(target, gps_pred, num_beams=num_beams)),
                "final_signed_residual": int(signed_circular_residual(target, final_pred, num_beams=num_beams)),
                "improvement": int(gps_error - final_error),
                "delta": int(final_pred - gps_pred),
                "correction_gate": 1.0 if not gps_good and ablation != "gps_prior_only" else 0.0,
                "correction_strength": 0.0,
                "gps_is_good_error_lt4": gps_good,
                "gps_prior_peak_prob": row.get("gps_prior_peak_prob", ""),
                "gps_prior_entropy": row.get("gps_prior_entropy", ""),
                "gps_prior_source": row.get("gps_prior_source", ""),
                "theta_degrees": row.get("theta_degrees", ""),
                "E": row.get("E", ""),
                "N": row.get("N", ""),
            }
        )
    return result


def _summary_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: str,
    ablation: str,
    modalities: Sequence[str],
    label_space: str,
    support_ratio: float,
    support_count: int,
    query_count: int,
    dba_delta: float,
    scene: str | None = None,
    gps_good_bad_group: str | None = None,
) -> dict[str, Any]:
    gps_errors = np.asarray([_float(row.get("gps_circular_error"), 0.0) for row in rows], dtype=np.float64)
    final_errors = np.asarray([_float(row.get("final_circular_error"), 0.0) for row in rows], dtype=np.float64)
    sample_count = int(len(rows))
    good_mask = gps_errors < 4.0
    bad_mask = ~good_mask
    good_degrade = float(np.mean((final_errors > gps_errors)[good_mask])) if np.any(good_mask) else 0.0
    bad_correct = float(np.mean((final_errors < gps_errors)[bad_mask])) if np.any(bad_mask) else 0.0
    result = {
        "protocol": protocol,
        "support_ratio": float(support_ratio),
        "label_space": label_space,
        "train_mode": "support_only",
        "ablation": ablation,
        "modalities": ",".join(modalities),
        "sample_count": sample_count,
        "valid_label_count": sample_count,
        "support_count": int(support_count),
        "query_count": int(query_count),
        "DBA": dba_from_circular_distances(final_errors, delta=dba_delta),
        "DBA_zero_ratio": dba_zero_ratio(final_errors),
        "mean_circular_error": float(final_errors.mean()) if final_errors.size else 0.0,
        "median_circular_error": float(np.median(final_errors)) if final_errors.size else 0.0,
        "exact_acc": float(np.mean(final_errors == 0)) if final_errors.size else 0.0,
        "pm1_acc": float(np.mean(final_errors <= 1)) if final_errors.size else 0.0,
        "pm2_acc": float(np.mean(final_errors <= 2)) if final_errors.size else 0.0,
        "pm4_acc": float(np.mean(final_errors <= 4)) if final_errors.size else 0.0,
        "top1": float(np.mean(final_errors == 0)) if final_errors.size else 0.0,
        "top3": _topk_acc(rows, k=3, num_beams=64),
        "top5": _topk_acc(rows, k=5, num_beams=64),
        "gps_DBA": dba_from_circular_distances(gps_errors, delta=dba_delta),
        "gps_DBA_zero_ratio": dba_zero_ratio(gps_errors),
        "gps_mean_circular_error": float(gps_errors.mean()) if gps_errors.size else 0.0,
        "gps_median_circular_error": float(np.median(gps_errors)) if gps_errors.size else 0.0,
        "delta_DBA_vs_gps": dba_from_circular_distances(final_errors, delta=dba_delta)
        - dba_from_circular_distances(gps_errors, delta=dba_delta),
        "delta_mean_error_vs_gps": (float(final_errors.mean()) if final_errors.size else 0.0)
        - (float(gps_errors.mean()) if gps_errors.size else 0.0),
        "good_sample_degradation_rate": good_degrade,
        "bad_sample_correction_rate": bad_correct,
        "upper_bound_protocol": False,
        "query_label_used_for_training": False,
        "skipped_reason": "",
    }
    if scene is not None:
        result["scene"] = scene
    if gps_good_bad_group is not None:
        result["gps_good_bad_group"] = gps_good_bad_group
    return result


def _skipped_summary(
    *,
    ablation: str,
    modalities: Sequence[str],
    label_space: str,
    support_ratio: float,
    reason: str,
) -> dict[str, Any]:
    row = _summary_row(
        [],
        protocol="target_adapt_beambench_residual",
        ablation=ablation,
        modalities=modalities,
        label_space=label_space,
        support_ratio=support_ratio,
        support_count=0,
        query_count=0,
        dba_delta=5.0,
    )
    row["skipped_reason"] = reason
    return row


def _candidate_recall_rows(rows: Sequence[Mapping[str, Any]], *, num_beams: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        target = _int(row.get("target_label"), -100)
        gps_top = _parse_topk(row.get("gps_topk_predictions"))
        gps_top1 = _int(row.get("gps_pred_top1"), -1)
        local = set(circular_window(gps_top1, radius=8, num_beams=num_beams)) if gps_top1 >= 0 else set()
        union = set(gps_top[:16]) | local
        out.append(
            {
                "scene": row.get("scene", ""),
                "sample_id": row.get("sample_id", ""),
                "ablation": row.get("ablation", ""),
                "target_in_gps_top16": target in set(gps_top[:16]),
                "target_in_local_radius8": target in local,
                "target_in_union_candidates": target in union,
                "rerank_top1": row.get("final_predicted_beam", ""),
                "rerank_top3_hit": target in set(gps_top[:3]),
            }
        )
    return out


def _comparison_markdown(
    r15: Mapping[str, Any],
    r20: Mapping[str, Any],
    residual_rows: Sequence[Mapping[str, Any]],
    recommended: Mapping[str, Any] | None,
    *,
    support_ratio: float,
    label_space: str,
) -> str:
    best = recommended or {}
    r15_dba = _float(r15.get("DBA"), 0.0)
    r20_dba = _float(r20.get("DBA"), 0.0)
    best_dba = _float(best.get("DBA"), 0.0)
    best_ablation = str(best.get("ablation") or "none")
    lines = [
        "# DeepSense6G GPS residual comparison",
        "",
        f"- support_ratio: {support_ratio:.2f}",
        f"- label_space: {label_space}",
        f"- GPS v2 r15 summary best DBA: {r15_dba:.6f}",
        f"- GPS v2 r20 best DBA: {r20_dba:.6f}",
        f"- best residual ablation: {best_ablation} (DBA={best_dba:.6f})",
        "",
        "## Diagnostic answers",
        "",
        f"- Exceeds r15 summary row: {best_dba > r15_dba}",
        f"- Exceeds direct gps_prior_only replay: {best_ablation != 'gps_prior_only' and best_dba > _float(best.get('gps_DBA'), 0.0)}",
        f"- Close to r20: {abs(best_dba - r20_dba) <= 0.02}",
        "- Scene contribution: see summary_by_scene.csv.",
        f"- Hard correction rate: {_float(best.get('bad_sample_correction_rate'), 0.0):.6f}",
        f"- Good degradation rate: {_float(best.get('good_sample_degradation_rate'), 0.0):.6f}",
        "- Gate behavior: correction_gate is diagnostic in predictions.csv and stays tied to GPS hard samples in this conservative run.",
        "- Multimodal benefit: optional modality ablations run when resources are discoverable; this conservative replay reports no positive delta.",
        "",
        "## Notes",
        "",
        "gps_prior_only is expected to match the selected GPS v2 prediction rows exactly. A difference between GPS v2 summary rows and gps_prior_only usually indicates an aggregate summary口径 difference; direct replay is the residual workflow sanity check.",
    ]
    if residual_rows:
        lines.extend(["", "## Residual rows", ""])
        for row in residual_rows:
            lines.append(
                f"- {row.get('ablation')}: DBA={_float(row.get('DBA'), 0.0):.6f}, "
                f"delta={_float(row.get('delta_DBA_vs_gps'), 0.0):.6f}, "
                f"good_degradation={_float(row.get('good_sample_degradation_rate'), 0.0):.6f}"
            )
    return "\n".join(lines) + "\n"


def _recommend(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [
        row
        for row in rows
        if _float(row.get("good_sample_degradation_rate"), 0.0) <= 0.10
        and not str(row.get("skipped_reason") or "")
        and str(row.get("protocol")) == "target_adapt_beambench_residual"
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda row: _float(row.get("DBA"), -1.0))


def _best_gps_row(path: Path) -> dict[str, Any]:
    rows = [row for row in _read_csv(path) if str(row.get("protocol")) == "target_adapt_beambench"]
    if not rows:
        return {}
    return max(rows, key=lambda row: _float(row.get("DBA"), -1.0))


def _modalities_for_ablation(ablation: str, availability: Mapping[str, Any]) -> tuple[str, ...]:
    if ablation == "gps_prior_only":
        return ("gps_prior",)
    if ablation in OPTIONAL_ABLATIONS:
        return ("gps_context", *OPTIONAL_ABLATIONS[ablation])
    return ("gps_context",)


def _modality_availability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for modality in ("image", "lidar", "radar"):
        path_key = f"{modality}_path"
        feature_key = f"{modality}_feature_path"
        path_count = sum(1 for row in rows if str(row.get(path_key) or ""))
        feature_count = sum(1 for row in rows if str(row.get(feature_key) or ""))
        result[modality] = {
            "path_count": path_count,
            "feature_count": feature_count,
            "available": path_count > 0 or feature_count > 0,
        }
    return result


def _topk_acc(rows: Sequence[Mapping[str, Any]], *, k: int, num_beams: int) -> float:
    hits = []
    for row in rows:
        target = _int(row.get("target_label"), -100)
        topk = _parse_topk(row.get("gps_topk_predictions"))[: int(k)]
        hits.append(target in set(topk))
    return float(np.mean(hits)) if hits else 0.0


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
    good = [row for row in rows if str(row.get("gps_is_good_error_lt4")) in {"True", "true", "1"}]
    bad = [row for row in rows if row not in good]
    labels = ["gps_good", "gps_bad"]
    gps = [np.mean([_float(row.get("gps_circular_error"), 0.0) for row in group]) if group else 0.0 for group in (good, bad)]
    final = [np.mean([_float(row.get("final_circular_error"), 0.0) for row in group]) if group else 0.0 for group in (good, bad)]
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


def _dominant(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return ""
    return max(counts, key=counts.get)


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
    if not isinstance(parsed, list):
        return []
    return [int(item) for item in parsed]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        resolved_fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    resolved_fieldnames.append(str(key))
                    seen.add(str(key))
    else:
        resolved_fieldnames = [str(item) for item in fieldnames]
    if not resolved_fieldnames:
        target.write_text("", encoding="utf-8")
        return
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=resolved_fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in resolved_fieldnames} for row in rows])


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
    "compare_deepsense6g_residual_with_gps_v2",
    "inspect_residual_inputs",
    "plot_deepsense6g_residual_fusion",
    "run_deepsense6g_residual_fusion",
]
