from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kd_sensing.config.io import dump_config
from kd_sensing.data.deepsense6g_topk_candidate_manifest import (
    MANIFEST_NAME,
    build_topk_candidate_manifest,
    circular_distance,
    ratio_tag,
    signed_circular_residual,
)
from kd_sensing.evaluation.metrics import dba_from_circular_distances, dba_zero_ratio


DEFAULT_ABLATIONS = (
    "gps_top1_baseline",
    "gps_top8_oracle",
    "gps_candidate_prob",
    "gps_context_only_selector",
    "camera_ae_only_selector",
    "camera_ae_gps_selector",
    "camera_ae_gps_selector_anchor",
    "top8_selector_no_gps_prior_fusion",
)
CAMERA_ABLATIONS = {"camera_ae_only_selector", "camera_ae_gps_selector", "camera_ae_gps_selector_anchor"}
TRAINABLE_SELECTOR_ABLATIONS = {
    "gps_context_only_selector",
    "camera_ae_only_selector",
    "camera_ae_gps_selector",
    "camera_ae_gps_selector_anchor",
    "top8_selector_no_gps_prior_fusion",
}
LEARNED_SELECTOR_ABLATIONS = TRAINABLE_SELECTOR_ABLATIONS | {"candidate_attention_selector"}


def run_deepsense6g_top8_selector(
    cfg: Mapping[str, Any],
    *,
    support_ratio: float | None = None,
    label_space: str | None = None,
    topk: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    data_cfg = _mapping(cfg.get("data"))
    candidate_cfg = _mapping(cfg.get("candidate"))
    outputs_cfg = _mapping(cfg.get("outputs"))
    metrics_cfg = _mapping(cfg.get("metrics"))
    train_cfg = _mapping(cfg.get("train"))
    ratio = float(support_ratio if support_ratio is not None else data_cfg.get("support_ratio", 0.15))
    selected_label_space = str(label_space or data_cfg.get("label_space", "mapping_disabled"))
    selected_topk = int(topk if topk is not None else candidate_cfg.get("topk", 8))
    tag = ratio_tag(ratio)
    out_root = Path(output_dir or outputs_cfg.get("root", "outputs/analysis/deepsense6g_top8_selector"))
    result_dir = out_root / tag / selected_label_space
    manifest_path = result_dir / str(outputs_cfg.get("manifest_dir", "manifest")) / MANIFEST_NAME
    if not manifest_path.exists():
        build_topk_candidate_manifest(
            cfg,
            support_ratio=ratio,
            label_space=selected_label_space,
            topk=selected_topk,
            output_dir=out_root,
        )
    rows = _read_csv(manifest_path)
    query_rows = [
        row
        for row in rows
        if str(row.get("support_query_role") or "").startswith("query")
        and _int(row.get("target_label"), -100) >= 0
    ]
    support_rows = [row for row in rows if str(row.get("support_query_role")) == "support"]
    result_dir.mkdir(parents=True, exist_ok=True)
    availability = _modality_availability(rows)
    ablations = list(_mapping(cfg.get("ablation")).get("enabled") or DEFAULT_ABLATIONS)
    if bool(_mapping(cfg.get("attention")).get("enabled", False)):
        ablations.append("candidate_attention_selector")
    ablations = _dedupe(ablations)
    train_mode = str(train_cfg.get("mode", "source_pretrain_target_finetune"))
    if train_mode not in {"support_only", "source_pretrain_target_finetune"}:
        train_mode = "source_pretrain_target_finetune"
    protocol = str(_mapping(cfg.get("experiment")).get("protocol", "target_adapt_beambench_top8_selector"))
    dba_delta = float(metrics_cfg.get("dba_delta", 5.0))
    summary_rows: list[dict[str, Any]] = []
    by_scene_rows: list[dict[str, Any]] = []
    by_hit_miss_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selection_events: list[dict[str, Any]] = []
    training_history: list[dict[str, Any]] = []
    warnings: list[str] = []
    skipped: dict[str, str] = {}

    for ablation in ablations:
        if ablation in CAMERA_ABLATIONS and not availability["camera_ae"]["available"]:
            reason = "missing_or_unstable_camera_ae_feature"
            skipped[ablation] = reason
            summary_rows.append(
                _skipped_summary(
                    protocol=protocol,
                    ablation=ablation,
                    train_mode=train_mode,
                    support_ratio=ratio,
                    label_space=selected_label_space,
                    topk=selected_topk,
                    reason=reason,
                )
            )
            continue
        if ablation in TRAINABLE_SELECTOR_ABLATIONS:
            trained = _trained_prediction_rows_for_ablation(
                manifest_path,
                ablation=ablation,
                cfg=cfg,
                train_mode=train_mode,
                protocol=protocol,
                support_ratio=ratio,
                label_space=selected_label_space,
                topk=selected_topk,
                num_beams=int(data_cfg.get("num_beams", 64)),
                result_dir=result_dir,
                dba_delta=dba_delta,
            )
            run_rows = trained["prediction_rows"]
            training_history.extend(trained["history"])
            warnings.extend(trained["warnings"])
        else:
            run_rows = _prediction_rows_for_ablation(
                query_rows,
                ablation=ablation,
                train_mode=train_mode,
                protocol=protocol,
                support_ratio=ratio,
                label_space=selected_label_space,
                topk=selected_topk,
                num_beams=int(data_cfg.get("num_beams", 64)),
            )
        prediction_rows.extend(run_rows)
        selection_events.extend([row for row in run_rows if int(row["final_top1"]) != int(row["gps_top1"])])
        summary_rows.append(
            _summary_row(
                run_rows,
                protocol=protocol,
                ablation=ablation,
                train_mode=train_mode,
                support_ratio=ratio,
                label_space=selected_label_space,
                topk=selected_topk,
                support_count=len(support_rows),
                query_count=len(query_rows),
                dba_delta=dba_delta,
            )
        )
        for scene in sorted({row["scene"] for row in run_rows}):
            scene_rows = [row for row in run_rows if row["scene"] == scene]
            by_scene_rows.append(
                _summary_row(
                    scene_rows,
                    protocol=protocol,
                    ablation=ablation,
                    train_mode=train_mode,
                    support_ratio=ratio,
                    label_space=selected_label_space,
                    topk=selected_topk,
                    support_count=len([row for row in support_rows if str(row.get("scene")) == scene]),
                    query_count=len(scene_rows),
                    dba_delta=dba_delta,
                    scene=scene,
                )
            )
        for hit_value in (True, False):
            group_rows = [row for row in run_rows if _bool(row.get("target_in_top8")) is hit_value]
            by_hit_miss_rows.append(
                _summary_row(
                    group_rows,
                    protocol=protocol,
                    ablation=ablation,
                    train_mode=train_mode,
                    support_ratio=ratio,
                    label_space=selected_label_space,
                    topk=selected_topk,
                    support_count=len(support_rows),
                    query_count=len(group_rows),
                    dba_delta=dba_delta,
                    top8_hit_miss_group="target_in_top8=1" if hit_value else "target_in_top8=0",
                )
            )

    if _gps_candidate_prob_mismatch(prediction_rows):
        warnings.append("gps_candidate_prob baseline does not match gps_top1_baseline on at least one row.")

    _write_csv(result_dir / "summary_overall.csv", summary_rows)
    _write_csv(result_dir / "summary_by_scene.csv", by_scene_rows)
    _write_csv(result_dir / "summary_by_top8_hit_miss.csv", by_hit_miss_rows)
    _write_csv(result_dir / "predictions.csv", prediction_rows)
    _write_csv(result_dir / "selection_events.csv", selection_events)
    _write_csv(result_dir / "candidate_rank_distribution.csv", _candidate_rank_distribution(prediction_rows))
    _write_csv(result_dir / "metrics.csv", summary_rows)
    if training_history:
        _write_csv(result_dir / "training_history.csv", training_history)
    if bool(outputs_cfg.get("write_config_snapshot", True)):
        dump_config(dict(cfg), result_dir / "resolved_config.yaml")
    metadata = {
        "workflow": "deepsense6g_gps_top8_candidate_selector",
        "result_dir": str(result_dir),
        "manifest_path": str(manifest_path),
        "support_ratio": ratio,
        "ratio_tag": tag,
        "label_space": selected_label_space,
        "topk": selected_topk,
        "protocol": protocol,
        "train_mode": train_mode,
        "supported_train_modes": ["support_only", "source_pretrain_target_finetune"],
        "support_count": len(support_rows),
        "query_count": len(query_rows),
        "source_missing_fallback": "support_only",
        "degraded_reason": (
            "source rows are not exported in the strict GPS v2 r15 manifest; conservative replay uses support_only behavior"
            if train_mode == "source_pretrain_target_finetune"
            else ""
        ),
        "enabled_modalities": availability,
        "skipped_ablations": skipped,
        "query_label_used_for_training": False,
        "standard_artifacts": [
            "summary_overall.csv",
            "summary_by_scene.csv",
            "summary_by_top8_hit_miss.csv",
            "predictions.csv",
            "selection_events.csv",
            "candidate_rank_distribution.csv",
            "metrics.csv",
            "training_history.csv",
            "run_metadata.json",
            "resolved_config.yaml",
        ],
        "trained_ablations": sorted({str(row.get("ablation")) for row in training_history if str(row.get("event")) == "epoch"}),
        "warnings": warnings,
    }
    (result_dir / "run_metadata.json").write_text(json.dumps(_json_ready(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "result_dir": str(result_dir),
        "manifest_path": str(manifest_path),
        "summary_overall": str(result_dir / "summary_overall.csv"),
        "prediction_rows": len(prediction_rows),
        "query_count": len(query_rows),
        "support_count": len(support_rows),
        "training_history_rows": len(training_history),
        "warnings": warnings,
    }


def plot_deepsense6g_top8_selector(results_dir: str | Path) -> dict[str, Any]:
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
    created: list[str] = []
    main = [row for row in rows if row.get("ablation") == "camera_ae_gps_selector_anchor"] or [
        row for row in rows if row.get("ablation") == "gps_context_only_selector"
    ] or rows
    groups = {"overall": main}
    for scene in sorted({row["scene"] for row in main}):
        groups[scene] = [row for row in main if row["scene"] == scene]
    for name, group in groups.items():
        prefix = f"{name}_"
        easting = [_float(row.get("E"), 0.0) for row in group]
        northing = [_float(row.get("N"), 0.0) for row in group]
        gps_error = [_float(row.get("gps_error"), 0.0) for row in group]
        final_error = [_float(row.get("final_error"), 0.0) for row in group]
        improvement = [_float(row.get("improvement"), 0.0) for row in group]
        target_rank = [_rank_value(row.get("target_candidate_rank")) for row in group]
        selected_rank = [_rank_value(row.get("selected_candidate_rank")) for row in group]
        miss_prob = [_float(row.get("miss_probability"), 0.0) for row in group]
        labels = [_int(row.get("target_label"), 0) for row in group]
        created.append(_scatter(plt, figure_dir / f"{prefix}enu_scatter.png", easting, northing, final_error, "Final error"))
        created.append(_scatter(plt, figure_dir / f"{prefix}improvement.png", easting, northing, improvement, "Improvement"))
        created.append(_scatter(plt, figure_dir / f"{prefix}hit_miss_spatial_map.png", easting, northing, [_bool(row.get("target_in_top8")) for row in group], "Top8 hit"))
        created.append(_hist(plt, figure_dir / f"{prefix}target_rank_distribution.png", [target_rank], ["target rank"], "Target candidate rank"))
        created.append(_hist(plt, figure_dir / f"{prefix}selected_candidate_rank_distribution.png", [selected_rank], ["selected rank"], "Selected rank"))
        created.append(_hist(plt, figure_dir / f"{prefix}residual_histogram.png", [gps_error, final_error], ["GPS", "Final"], "Circular error"))
        created.append(_hist(plt, figure_dir / f"{prefix}signed_residual.png", [[_float(row.get("gps_signed_residual"), 0.0) for row in group], [_float(row.get("final_signed_residual"), 0.0) for row in group]], ["GPS", "Final"], "Signed residual"))
        created.append(_hist(plt, figure_dir / f"{prefix}label_distribution.png", [labels], ["target"], "Target label"))
        created.append(_scatter(plt, figure_dir / f"{prefix}calibration.png", [_float(row.get("selected_candidate_prob"), 0.0) for row in group], final_error, miss_prob, "Candidate calibration"))
        created.append(_scatter(plt, figure_dir / f"{prefix}miss_diagnostics.png", gps_error, miss_prob, [_bool(row.get("target_in_top8")) for row in group], "Miss probability"))
    montage_notes = _write_montage_if_available(plt, figure_dir, main)
    notes = {"created": created, "montage": montage_notes}
    (figure_dir / "plot_metadata.json").write_text(json.dumps(_json_ready(notes), indent=2, sort_keys=True), encoding="utf-8")
    return {"figures_dir": str(figure_dir), "created": created, "metadata": str(figure_dir / "plot_metadata.json")}


def compare_deepsense6g_top8_selector_with_gps_v2(
    *,
    gps_v2_root: str | Path,
    selector_root: str | Path,
    support_ratio: float = 0.15,
    label_space: str = "mapping_disabled",
) -> dict[str, Any]:
    gps_root = Path(gps_v2_root)
    selector_dir = Path(selector_root)
    tag = ratio_tag(support_ratio)
    gps_row = _best_gps_row(gps_root / tag / label_space / "summary_overall.csv")
    selector_rows = [row for row in _read_csv(selector_dir / "summary_overall.csv") if not str(row.get("skipped_reason") or "")]
    comparison_rows = []
    for row in selector_rows:
        comparison_rows.append(
            {
                "scene": "overall",
                "ablation": row.get("ablation", ""),
                "gps_v2_DBA": _float(gps_row.get("DBA"), 0.0),
                "selector_DBA": _float(row.get("DBA"), 0.0),
                "delta_DBA": _float(row.get("DBA"), 0.0) - _float(gps_row.get("DBA"), 0.0),
                "gps_v2_mean_error": _float(gps_row.get("mean_circular_error"), 0.0),
                "selector_mean_error": _float(row.get("mean_circular_error"), 0.0),
                "delta_mean_error": _float(row.get("mean_circular_error"), 0.0) - _float(gps_row.get("mean_circular_error"), 0.0),
                "gps_v2_Top8": _float(row.get("top8_recall"), 0.0),
                "top8_oracle_DBA": _float(row.get("top8_oracle_DBA"), 0.0),
                "selector_accuracy_when_target_in_top8": _float(row.get("selector_accuracy_when_target_in_top8"), 0.0),
                "top8_hit_count": _int(row.get("top8_hit_count"), 0),
                "top8_miss_count": _int(row.get("top8_miss_count"), 0),
                "miss_AUC": row.get("miss_auc", ""),
            }
        )
    _write_csv(selector_dir / "comparison_with_gps_v2.csv", comparison_rows)
    report = _comparison_report(gps_row, comparison_rows, selector_rows, support_ratio=support_ratio, label_space=label_space)
    report_path = selector_dir / "comparison_report.md"
    report_path.write_text(report, encoding="utf-8")
    return {
        "comparison_path": str(selector_dir / "comparison_with_gps_v2.csv"),
        "report_path": str(report_path),
        "row_count": len(comparison_rows),
    }


def _trained_prediction_rows_for_ablation(
    manifest_path: str | Path,
    *,
    ablation: str,
    cfg: Mapping[str, Any],
    train_mode: str,
    protocol: str,
    support_ratio: float,
    label_space: str,
    topk: int,
    num_beams: int,
    result_dir: str | Path,
    dba_delta: float,
) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, Subset

        from kd_sensing.data.deepsense6g_topk_candidate_dataset import (
            CANDIDATE_FEATURE_NAMES,
            GPS_CONTEXT_FEATURE_NAMES,
            TopKCandidateManifestDataset,
            collate_topk_candidate_batch,
        )
        from kd_sensing.losses.topk_candidate_losses import TopKCandidateSelectorLoss
        from kd_sensing.models.topk_candidate_selector import TopKCandidateSelector
    except ImportError as exc:  # pragma: no cover - torch is a required project dependency in normal runs.
        return _training_fallback_result(
            manifest_path,
            ablation=ablation,
            train_mode=train_mode,
            protocol=protocol,
            support_ratio=support_ratio,
            label_space=label_space,
            topk=topk,
            num_beams=num_beams,
            reason=f"training_dependency_unavailable:{exc}",
        )

    train_cfg = _mapping(cfg.get("train"))
    model_cfg = _mapping(cfg.get("model"))
    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_request = str(train_cfg.get("device", "auto")).lower()
    if device_request == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_request)

    normalizer_path = Path(result_dir) / "normalizers" / f"{ablation}.json"
    dataset = TopKCandidateManifestDataset(
        manifest_path,
        topk=topk,
        num_beams=num_beams,
        enabled_modalities=_enabled_modalities_for_ablation(ablation),
        normalizer_path=normalizer_path,
        fit_normalizer=True,
        save_normalizer=True,
        include_query_labels=True,
        image_size=int(_mapping(cfg.get("image")).get("size", 64)),
    )
    train_indices, val_indices = _train_val_indices(
        dataset.rows,
        train_mode=train_mode,
        seed=seed,
        validation_fraction=float(train_cfg.get("validation_fraction", 0.2)),
    )
    query_indices = [
        idx
        for idx, row in enumerate(dataset.rows)
        if _is_query_role(row.get("support_query_role") or row.get("split_role")) and _int(row.get("target_label"), -100) >= 0
    ]
    if not train_indices or not query_indices:
        reason = "no_support_rows_for_training" if not train_indices else "no_query_rows_for_evaluation"
        return _training_fallback_result(
            manifest_path,
            ablation=ablation,
            train_mode=train_mode,
            protocol=protocol,
            support_ratio=support_ratio,
            label_space=label_space,
            topk=topk,
            num_beams=num_beams,
            reason=reason,
        )

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        collate_fn=collate_topk_candidate_batch,
    )
    model = TopKCandidateSelector(
        topk=topk,
        num_beams=num_beams,
        candidate_feature_dim=int(model_cfg.get("candidate_feature_dim", len(CANDIDATE_FEATURE_NAMES))),
        gps_context_dim=int(model_cfg.get("gps_context_dim", len(GPS_CONTEXT_FEATURE_NAMES))),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        lambda_init=float(model_cfg.get("lambda_init", 0.5)),
        lambda_max=float(model_cfg.get("lambda_max", 3.0)),
        use_gps_prior_fusion=_use_gps_prior_fusion(ablation),
    ).to(device)
    loss_cfg = dict(_mapping(cfg.get("loss")))
    loss_cfg["num_beams"] = int(num_beams)
    if not _use_prior_anchor(ablation):
        loss_cfg["prior_anchor_kl_weight"] = 0.0
    criterion = TopKCandidateSelectorLoss(loss_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = max(1, int(train_cfg.get("epochs", 20)))
    patience = max(1, int(train_cfg.get("patience", max(3, epochs // 4))))
    grad_clip = float(train_cfg.get("grad_clip_norm", 5.0))
    history: list[dict[str, Any]] = []
    best_metric = -float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        sample_total = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**_selector_inputs_from_batch(batch, ablation=ablation, device=device))
            losses = criterion(outputs, batch)
            loss = losses["loss"]
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            count = int(losses["train_sample_count"].detach().cpu().item())
            loss_total += float(loss.detach().cpu().item()) * max(count, 1)
            sample_total += max(count, 1)

        val_rows = _prediction_rows_from_trained_model(
            model,
            dataset,
            val_indices,
            ablation=ablation,
            train_mode=train_mode,
            protocol=protocol,
            support_ratio=support_ratio,
            label_space=label_space,
            topk=topk,
            num_beams=num_beams,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            model_metadata={"trained_model_used": True, "model_selection_split": "target_support_internal_validation"},
        )
        final_errors = np.asarray([_float(row.get("final_error"), 0.0) for row in val_rows], dtype=np.float64)
        gps_errors = np.asarray([_float(row.get("gps_error"), 0.0) for row in val_rows], dtype=np.float64)
        val_dba = dba_from_circular_distances(final_errors, delta=dba_delta)
        gps_val_dba = dba_from_circular_distances(gps_errors, delta=dba_delta)
        history.append(
            {
                "ablation": ablation,
                "event": "epoch",
                "epoch": epoch,
                "train_loss": loss_total / max(sample_total, 1),
                "train_sample_count": len(train_indices),
                "validation_sample_count": len(val_indices),
                "validation_DBA": val_dba,
                "validation_gps_DBA": gps_val_dba,
                "validation_delta_DBA_vs_gps": val_dba - gps_val_dba,
                "lambda_value": _model_lambda_value(model),
                "device": str(device),
                "query_label_used_for_training": False,
            }
        )
        if val_dba > best_metric:
            best_metric = float(val_dba)
            best_epoch = int(epoch)
            best_state = _clone_initialized_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    model.to(device)
    history.append(
        {
            "ablation": ablation,
            "event": "selected_model",
            "epoch": best_epoch,
            "validation_DBA": best_metric if math.isfinite(best_metric) else "",
            "train_mode": train_mode,
            "train_sample_count": len(train_indices),
            "validation_sample_count": len(val_indices),
            "query_sample_count": len(query_indices),
            "normalizer_path": str(normalizer_path),
            "use_gps_prior_fusion": _use_gps_prior_fusion(ablation),
            "use_prior_anchor": _use_prior_anchor(ablation),
            "query_label_used_for_training": False,
        }
    )
    query_rows = _prediction_rows_from_trained_model(
        model,
        dataset,
        query_indices,
        ablation=ablation,
        train_mode=train_mode,
        protocol=protocol,
        support_ratio=support_ratio,
        label_space=label_space,
        topk=topk,
        num_beams=num_beams,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        model_metadata={
            "trained_model_used": True,
            "model_selection_split": "target_support_internal_validation",
            "model_selection_metric": "validation_DBA",
            "model_selection_value": best_metric if math.isfinite(best_metric) else "",
            "model_selection_epoch": best_epoch,
        },
    )
    return {"prediction_rows": query_rows, "history": history, "warnings": []}


def _training_fallback_result(
    manifest_path: str | Path,
    *,
    ablation: str,
    train_mode: str,
    protocol: str,
    support_ratio: float,
    label_space: str,
    topk: int,
    num_beams: int,
    reason: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in _read_csv(manifest_path)
        if _is_query_role(row.get("support_query_role") or row.get("split_role")) and _int(row.get("target_label"), -100) >= 0
    ]
    prediction_rows = _prediction_rows_for_ablation(
        rows,
        ablation=ablation,
        train_mode=train_mode,
        protocol=protocol,
        support_ratio=support_ratio,
        label_space=label_space,
        topk=topk,
        num_beams=num_beams,
        model_metadata={"trained_model_used": False, "training_fallback_reason": reason},
    )
    return {
        "prediction_rows": prediction_rows,
        "history": [
            {
                "ablation": ablation,
                "event": "fallback",
                "reason": reason,
                "query_label_used_for_training": False,
            }
        ],
        "warnings": [f"{ablation} fell back to GPS candidate replay: {reason}"],
    }


def _prediction_rows_for_ablation(
    rows: Sequence[Mapping[str, Any]],
    *,
    ablation: str,
    train_mode: str,
    protocol: str,
    support_ratio: float,
    label_space: str,
    topk: int,
    num_beams: int,
    model_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        candidate_probs = [_float(row.get(f"cand{idx}_prob"), 0.0) for idx in range(topk)]
        candidate_scores = [math.log(max(prob, 1e-12)) for prob in candidate_probs]
        if ablation == "gps_top8_oracle":
            selected_idx = _int(row.get("nearest_candidate_index"), 0)
        else:
            selected_idx = int(np.argmax(np.asarray(candidate_probs, dtype=np.float64))) if candidate_probs else 0
        miss_probability = min(1.0, max(0.0, _float(row.get("gps_entropy"), 0.0) / max(math.log(float(num_beams)), 1e-8)))
        out.append(
            _prediction_item_from_row(
                row,
                selected_idx=selected_idx,
                candidate_probs=candidate_probs,
                candidate_scores=candidate_scores,
                miss_probability=miss_probability,
                ablation=ablation,
                train_mode=train_mode,
                protocol=protocol,
                support_ratio=support_ratio,
                label_space=label_space,
                topk=topk,
                num_beams=num_beams,
                model_metadata=model_metadata,
            )
        )
    return out


def _prediction_rows_from_trained_model(
    model: Any,
    dataset: Any,
    indices: Sequence[int],
    *,
    ablation: str,
    train_mode: str,
    protocol: str,
    support_ratio: float,
    label_space: str,
    topk: int,
    num_beams: int,
    device: Any,
    batch_size: int,
    num_workers: int,
    model_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader, Subset

    from kd_sensing.data.deepsense6g_topk_candidate_dataset import collate_topk_candidate_batch

    if not indices:
        return []
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_topk_candidate_batch,
    )
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(**_selector_inputs_from_batch(batch, ablation=ablation, device=device))
            probs = outputs["candidate_probs"].detach().cpu()
            scores = outputs["final_candidate_scores"].detach().cpu()
            miss_probability = torch.sigmoid(outputs["miss_logit"].reshape(-1)).detach().cpu()
            selected = probs.argmax(dim=-1)
            row_indices = batch["row_index"].detach().cpu().tolist()
            for offset, row_index in enumerate(row_indices):
                source_row = dataset.rows[int(row_index)]
                rows.append(
                    _prediction_item_from_row(
                        source_row,
                        selected_idx=int(selected[offset].item()),
                        candidate_probs=[float(item) for item in probs[offset].tolist()],
                        candidate_scores=[float(item) for item in scores[offset].tolist()],
                        miss_probability=float(miss_probability[offset].item()),
                        ablation=ablation,
                        train_mode=train_mode,
                        protocol=protocol,
                        support_ratio=support_ratio,
                        label_space=label_space,
                        topk=topk,
                        num_beams=num_beams,
                        model_metadata=model_metadata,
                    )
                )
    return rows


def _prediction_item_from_row(
    row: Mapping[str, Any],
    *,
    selected_idx: int,
    candidate_probs: Sequence[float],
    candidate_scores: Sequence[float],
    miss_probability: float,
    ablation: str,
    train_mode: str,
    protocol: str,
    support_ratio: float,
    label_space: str,
    topk: int,
    num_beams: int,
    model_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = _int(row.get("target_label"), -100)
    gps_top1 = _int(row.get("gps_top1"), _int(row.get("gps_pred_top1"), -1))
    candidate_beams = [_int(row.get(f"cand{idx}_beam"), -1) for idx in range(topk)]
    selected_idx = max(0, min(int(selected_idx), len(candidate_beams) - 1))
    final_top1 = int(candidate_beams[selected_idx])
    gps_error = circular_distance(gps_top1, target, num_beams=num_beams) if gps_top1 >= 0 and target >= 0 else 0
    final_error = circular_distance(final_top1, target, num_beams=num_beams) if final_top1 >= 0 and target >= 0 else 0
    oracle_beam = _int(row.get("top8_oracle_beam"), final_top1)
    oracle_error = _float(row.get("top8_oracle_error"), final_error)
    target_idx = _int(row.get("target_candidate_index"), -1)
    target_rank = "miss" if target_idx < 0 else target_idx + 1
    item = {
        "scene": row.get("scene", ""),
        "sample_id": row.get("sample_id", ""),
        "support_query_role": row.get("support_query_role", ""),
        "split_role": row.get("split_role", row.get("support_query_role", "")),
        "target_label": target,
        "gps_top1": gps_top1,
        "final_top1": final_top1,
        "gps_error": gps_error,
        "final_error": final_error,
        "gps_circular_error": gps_error,
        "final_circular_error": final_error,
        "gps_signed_residual": signed_circular_residual(target, gps_top1, num_beams=num_beams) if target >= 0 and gps_top1 >= 0 else "",
        "final_signed_residual": signed_circular_residual(target, final_top1, num_beams=num_beams) if target >= 0 and final_top1 >= 0 else "",
        "improvement": gps_error - final_error,
        "top8_oracle_beam": oracle_beam,
        "top8_oracle_error": oracle_error,
        "target_in_top8": _bool(row.get("target_in_top8")),
        "target_candidate_index": target_idx,
        "target_candidate_rank": target_rank,
        "nearest_candidate_index": row.get("nearest_candidate_index", ""),
        "selected_candidate_index": selected_idx,
        "selected_candidate_rank": selected_idx + 1,
        "selected_candidate_prob": float(candidate_probs[selected_idx]) if candidate_probs else 0.0,
        "miss_label": 1 if _bool(row.get("top8_miss")) else 0,
        "miss_probability": float(miss_probability),
        "candidate_beams_json": json.dumps(candidate_beams),
        "candidate_probs_json": json.dumps([float(item) for item in candidate_probs]),
        "candidate_scores_json": json.dumps([float(item) for item in candidate_scores]),
        "ablation": ablation,
        "train_mode": train_mode,
        "protocol": protocol,
        "support_ratio": support_ratio,
        "label_space": label_space,
        "topk": topk,
        "image_path": row.get("image_path", ""),
        "image_exists": row.get("image_exists", ""),
        "E": row.get("E", ""),
        "N": row.get("N", ""),
        "theta_degrees": row.get("theta_degrees", ""),
    }
    if model_metadata:
        item.update(dict(model_metadata))
    return item


def _selector_inputs_from_batch(batch: Mapping[str, Any], *, ablation: str, device: Any) -> dict[str, Any]:
    inputs = {
        "candidate_features": batch["candidate_features"].to(device),
        "gps_context": batch["gps_context"].to(device),
        "candidate_log_probs": batch["candidate_log_probs"].to(device),
        "candidate_probs": batch["candidate_probs"].to(device),
    }
    if "camera_ae" in _enabled_modalities_for_ablation(ablation) and "camera_ae_feature" in batch:
        inputs["camera_ae_feature"] = batch["camera_ae_feature"].to(device)
    if "lidar" in _enabled_modalities_for_ablation(ablation) and "lidar_feature" in batch:
        inputs["lidar_feature"] = batch["lidar_feature"].to(device)
    if "radar" in _enabled_modalities_for_ablation(ablation) and "radar_feature" in batch:
        inputs["radar_feature"] = batch["radar_feature"].to(device)
    return inputs


def _train_val_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_mode: str,
    seed: int,
    validation_fraction: float,
) -> tuple[list[int], list[int]]:
    trainable = [
        idx
        for idx, row in enumerate(rows)
        if _is_train_role(row.get("support_query_role") or row.get("split_role"))
        and _int(row.get("target_label"), -100) >= 0
    ]
    support = [
        idx
        for idx, row in enumerate(rows)
        if str(row.get("support_query_role") or row.get("split_role") or "") in {"support", "target_support"}
        and _int(row.get("target_label"), -100) >= 0
    ]
    source = [
        idx
        for idx, row in enumerate(rows)
        if str(row.get("support_query_role") or row.get("split_role") or "").startswith("source")
        and _int(row.get("target_label"), -100) >= 0
    ]
    pool = support if train_mode == "support_only" or not source else trainable
    val_pool = support or pool
    if not pool:
        return [], []
    rng = np.random.default_rng(int(seed))
    if len(val_pool) >= 5:
        val_count = max(1, min(len(val_pool) - 1, int(round(len(val_pool) * float(validation_fraction)))))
        val_set = set(int(item) for item in rng.choice(np.asarray(val_pool, dtype=np.int64), size=val_count, replace=False))
    else:
        val_set = set(val_pool)
    train_indices = [idx for idx in pool if idx not in val_set]
    if not train_indices:
        train_indices = list(pool)
    val_indices = sorted(val_set) if val_set else list(train_indices)
    return train_indices, val_indices


def _enabled_modalities_for_ablation(ablation: str) -> tuple[str, ...]:
    if ablation in CAMERA_ABLATIONS:
        return ("camera_ae",)
    return ()


def _use_gps_prior_fusion(ablation: str) -> bool:
    return ablation not in {"camera_ae_only_selector", "top8_selector_no_gps_prior_fusion"}


def _use_prior_anchor(ablation: str) -> bool:
    return ablation in {"gps_context_only_selector", "camera_ae_gps_selector_anchor"}


def _model_lambda_value(model: Any) -> float:
    value = getattr(model, "lambda_value", None)
    if value is None:
        return 0.0
    try:
        return float(value.detach().cpu().item())
    except Exception:
        return 0.0


def _clone_initialized_state_dict(model: Any) -> dict[str, Any]:
    from torch.nn.parameter import UninitializedParameter

    state: dict[str, Any] = {}
    for key, value in model.state_dict().items():
        if isinstance(value, UninitializedParameter):
            continue
        state[key] = value.detach().cpu().clone()
    return state


def _is_query_role(value: Any) -> bool:
    return str(value or "").startswith("query")


def _is_train_role(value: Any) -> bool:
    role = str(value or "")
    return bool(role) and not role.startswith("query")


def _summary_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: str,
    ablation: str,
    train_mode: str,
    support_ratio: float,
    label_space: str,
    topk: int,
    support_count: int,
    query_count: int,
    dba_delta: float,
    scene: str | None = None,
    top8_hit_miss_group: str | None = None,
) -> dict[str, Any]:
    gps_errors = np.asarray([_float(row.get("gps_error"), 0.0) for row in rows], dtype=np.float64)
    final_errors = np.asarray([_float(row.get("final_error"), 0.0) for row in rows], dtype=np.float64)
    oracle_errors = np.asarray([_float(row.get("top8_oracle_error"), 0.0) for row in rows], dtype=np.float64)
    target_hit = np.asarray([_bool(row.get("target_in_top8")) for row in rows], dtype=bool)
    selected_matches_target = np.asarray([_int(row.get("final_top1"), -1) == _int(row.get("target_label"), -100) for row in rows], dtype=bool)
    nearest_match = np.asarray([
        _int(row.get("selected_candidate_index"), -1) == _int(row.get("nearest_candidate_index"), -2)
        for row in rows
    ], dtype=bool)
    result = {
        "protocol": protocol,
        "support_ratio": support_ratio,
        "label_space": label_space,
        "topk": topk,
        "train_mode": train_mode,
        "ablation": ablation,
        "sample_count": len(rows),
        "valid_label_count": len(rows),
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
        "top3": float(np.mean([_rank_value(row.get("target_candidate_rank")) <= 3 for row in rows])) if rows else 0.0,
        "top5": float(np.mean([_rank_value(row.get("target_candidate_rank")) <= 5 for row in rows])) if rows else 0.0,
        "P_error_lt4": float(np.mean(final_errors < 4)) if final_errors.size else 0.0,
        "gps_DBA": dba_from_circular_distances(gps_errors, delta=dba_delta),
        "gps_DBA_zero_ratio": dba_zero_ratio(gps_errors),
        "gps_mean_circular_error": float(gps_errors.mean()) if gps_errors.size else 0.0,
        "gps_median_circular_error": float(np.median(gps_errors)) if gps_errors.size else 0.0,
        "gps_exact_acc": float(np.mean(gps_errors == 0)) if gps_errors.size else 0.0,
        "gps_P_error_lt4": float(np.mean(gps_errors < 4)) if gps_errors.size else 0.0,
        "delta_DBA_vs_gps": dba_from_circular_distances(final_errors, delta=dba_delta)
        - dba_from_circular_distances(gps_errors, delta=dba_delta),
        "delta_mean_error_vs_gps": (float(final_errors.mean()) if final_errors.size else 0.0)
        - (float(gps_errors.mean()) if gps_errors.size else 0.0),
        "top8_recall": float(np.mean(target_hit)) if target_hit.size else 0.0,
        "top8_hit_count": int(np.sum(target_hit)) if target_hit.size else 0,
        "top8_miss_count": int(np.sum(~target_hit)) if target_hit.size else 0,
        "top8_oracle_DBA": dba_from_circular_distances(oracle_errors, delta=dba_delta),
        "top8_oracle_mean_error": float(oracle_errors.mean()) if oracle_errors.size else 0.0,
        "top8_oracle_exact_acc": float(np.mean(oracle_errors == 0)) if oracle_errors.size else 0.0,
        "selector_accuracy_when_target_in_top8": float(np.mean(selected_matches_target[target_hit])) if np.any(target_hit) else 0.0,
        "nearest_candidate_selection_accuracy": float(np.mean(nearest_match)) if nearest_match.size else 0.0,
        "miss_probability_mean": float(np.mean([_float(row.get("miss_probability"), 0.0) for row in rows])) if rows else 0.0,
        "miss_auc": "",
        "upper_bound_protocol": ablation == "gps_top8_oracle",
        "query_label_used_for_training": False,
        "skipped_reason": "",
    }
    if scene is not None:
        result["scene"] = scene
    if top8_hit_miss_group is not None:
        result["top8_hit_miss_group"] = top8_hit_miss_group
    return result


def _skipped_summary(
    *,
    protocol: str,
    ablation: str,
    train_mode: str,
    support_ratio: float,
    label_space: str,
    topk: int,
    reason: str,
) -> dict[str, Any]:
    row = _summary_row(
        [],
        protocol=protocol,
        ablation=ablation,
        train_mode=train_mode,
        support_ratio=support_ratio,
        label_space=label_space,
        topk=topk,
        support_count=0,
        query_count=0,
        dba_delta=5.0,
    )
    row["skipped_reason"] = reason
    return row


def _modality_availability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "camera_ae": {
            "available_count": sum(1 for row in rows if _bool(row.get("camera_ae_feature_available"))),
            "available": any(_bool(row.get("camera_ae_feature_available")) for row in rows),
        },
        "image": {
            "available_count": sum(1 for row in rows if _bool(row.get("image_exists"))),
            "available": any(_bool(row.get("image_exists")) for row in rows),
        },
        "lidar": {
            "available_count": sum(1 for row in rows if _bool(row.get("lidar_feature_available"))),
            "available": any(_bool(row.get("lidar_feature_available")) for row in rows),
        },
        "radar": {
            "available_count": sum(1 for row in rows if _bool(row.get("radar_feature_available"))),
            "available": any(_bool(row.get("radar_feature_available")) for row in rows),
        },
    }


def _candidate_rank_distribution(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        target_rank = str(row.get("target_candidate_rank") or "miss")
        selected_rank = str(row.get("selected_candidate_rank") or "")
        key = (str(row.get("ablation") or ""), str(row.get("scene") or ""), target_rank, selected_rank)
        counts[key] = counts.get(key, 0) + 1
    return [
        {
            "ablation": ablation,
            "scene": scene,
            "target_rank": target_rank,
            "selected_rank": selected_rank,
            "count": count,
        }
        for (ablation, scene, target_rank, selected_rank), count in sorted(counts.items())
    ]


def _gps_candidate_prob_mismatch(rows: Sequence[Mapping[str, Any]]) -> bool:
    gps = {
        (row.get("scene"), row.get("sample_id")): _int(row.get("final_top1"), -1)
        for row in rows
        if str(row.get("ablation")) == "gps_top1_baseline"
    }
    for row in rows:
        if str(row.get("ablation")) != "gps_candidate_prob":
            continue
        key = (row.get("scene"), row.get("sample_id"))
        if gps.get(key, _int(row.get("gps_top1"), -1)) != _int(row.get("final_top1"), -1):
            return True
    return False


def _best_gps_row(path: Path) -> dict[str, Any]:
    rows = [row for row in _read_csv(path) if str(row.get("protocol")) == "target_adapt_beambench"]
    if not rows:
        return {}
    return max(rows, key=lambda row: _float(row.get("DBA"), -1.0))


def _comparison_report(
    gps_row: Mapping[str, Any],
    comparison_rows: Sequence[Mapping[str, Any]],
    selector_rows: Sequence[Mapping[str, Any]],
    *,
    support_ratio: float,
    label_space: str,
) -> str:
    learned_rows = [row for row in comparison_rows if str(row.get("ablation") or "") in LEARNED_SELECTOR_ABLATIONS]
    oracle_rows = [row for row in comparison_rows if str(row.get("ablation") or "") == "gps_top8_oracle"]
    selector_gps_rows = [row for row in comparison_rows if str(row.get("ablation") or "") == "gps_top1_baseline"]
    best_learned = max(learned_rows, key=lambda row: _float(row.get("selector_DBA"), -1.0)) if learned_rows else {}
    oracle = max(oracle_rows, key=lambda row: _float(row.get("selector_DBA"), -1.0)) if oracle_rows else {}
    selector_gps = selector_gps_rows[0] if selector_gps_rows else {}
    best_learned_ablation = str(best_learned.get("ablation") or "none")
    selector_gps_dba = _float(selector_gps.get("selector_DBA"), _float(gps_row.get("DBA"), 0.0))
    lines = [
        "# DeepSense6G GPS Top8 Candidate Selector comparison",
        "",
        f"- support_ratio: {support_ratio:.2f}",
        f"- label_space: {label_space}",
        f"- GPS v2 DBA: {_float(gps_row.get('DBA'), 0.0):.6f}",
        f"- selector GPS top1 baseline DBA: {selector_gps_dba:.6f}",
        f"- Top8 oracle DBA: {_float(oracle.get('selector_DBA'), 0.0):.6f}",
        f"- best learned selector: {best_learned_ablation} (DBA={_float(best_learned.get('selector_DBA'), 0.0):.6f})",
        "",
        "## Diagnostic answers",
        "",
        f"- Learned selector exceeds GPS v2 r15: {_float(best_learned.get('delta_DBA'), 0.0) > 0.0}",
        f"- Learned selector exceeds selector GPS top1 baseline: {_float(best_learned.get('selector_DBA'), 0.0) > selector_gps_dba}",
        f"- Top8 oracle exceeds GPS v2 r15: {_float(oracle.get('delta_DBA'), 0.0) > 0.0}",
        f"- Best learned selector close to Top8 oracle: {abs(_float(best_learned.get('selector_DBA'), 0.0) - _float(oracle.get('selector_DBA'), 0.0)) <= 0.05}",
        "- Scene contribution: inspect summary_by_scene.csv.",
        "- Scenario32/34 Top8 ceiling: inspect summary_by_scene.csv top8_recall and top8_oracle_DBA.",
        "- Camera AE benefit: compare camera_ae_gps_selector rows with gps_context_only_selector; skipped rows record skipped_reason.",
        "- GPS prior fusion stability: compare camera_ae_gps_selector_anchor with top8_selector_no_gps_prior_fusion.",
        "- Miss head: inspect summary_by_top8_hit_miss.csv miss_probability_mean and predictions.csv miss_probability.",
        "",
        "## Rows",
        "",
    ]
    for row in comparison_rows:
        lines.append(
            f"- {row.get('ablation')}: selector_DBA={_float(row.get('selector_DBA'), 0.0):.6f}, "
            f"delta_DBA={_float(row.get('delta_DBA'), 0.0):.6f}, "
            f"oracle_DBA={_float(row.get('top8_oracle_DBA'), 0.0):.6f}"
        )
    return "\n".join(lines) + "\n"


def _write_montage_if_available(plt: Any, figure_dir: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    image_rows = [row for row in rows if _bool(row.get("image_exists")) and str(row.get("image_path") or "")]
    if not image_rows:
        return {"created": [], "skipped_reason": "image paths unavailable"}
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        return {"created": [], "skipped_reason": str(exc)}
    created = []
    groups = {
        "successful_corrections_montage.png": [row for row in image_rows if _float(row.get("improvement"), 0.0) > 0],
        "degraded_samples_montage.png": [row for row in image_rows if _float(row.get("improvement"), 0.0) < 0],
        "top8_miss_montage.png": [row for row in image_rows if not _bool(row.get("target_in_top8"))],
    }
    for name, group in groups.items():
        if not group:
            continue
        subset = group[: min(6, len(group))]
        plt.figure(figsize=(2 * len(subset), 2))
        for idx, row in enumerate(subset, start=1):
            plt.subplot(1, len(subset), idx)
            plt.imshow(Image.open(str(row.get("image_path"))).convert("RGB"))
            plt.axis("off")
        path = figure_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        created.append(str(path))
    return {"created": created, "skipped_reason": "" if created else "no matching image montage samples"}


def _scatter(plt: Any, path: Path, x: Sequence[Any], y: Sequence[Any], color: Sequence[Any], title: str) -> str:
    plt.figure(figsize=(5, 4))
    plt.scatter([float(v) for v in x], [float(v) for v in y], c=[float(v) for v in color], s=8, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def _hist(plt: Any, path: Path, values: Sequence[Sequence[Any]], labels: Sequence[str], xlabel: str) -> str:
    plt.figure(figsize=(5, 4))
    for series, label in zip(values, labels):
        plt.hist([float(item) for item in series], bins=32, alpha=0.55, label=label)
    plt.xlabel(xlabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def _rank_value(value: Any) -> int:
    if str(value) == "miss":
        return 99
    return _int(value, 99)


def _dedupe(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value)
        if item not in result:
            result.append(item)
    return result


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    candidate = Path(path)
    if not candidate.exists():
        return []
    with candidate.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    names = _fieldnames(rows)
    with candidate.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in names})


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(str(key))
    return names


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in {None, ""}:
            return int(default)
        return int(float(str(value)))
    except (TypeError, ValueError):
        return int(default)


def _float(value: Any, default: float = 0.0) -> float:
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


__all__ = [
    "compare_deepsense6g_top8_selector_with_gps_v2",
    "plot_deepsense6g_top8_selector",
    "run_deepsense6g_top8_selector",
]
