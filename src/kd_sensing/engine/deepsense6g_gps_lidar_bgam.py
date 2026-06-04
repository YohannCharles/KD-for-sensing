from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kd_sensing.config.io import dump_config
from kd_sensing.data.deepsense6g_gps_lidar_bgam_manifest import (
    MANIFEST_NAME,
    PSEUDO_HISTORY_SUMMARY_NAME,
    build_gps_lidar_bgam_manifest,
)
from kd_sensing.data.deepsense6g_topk_candidate_manifest import circular_distance, ratio_tag, signed_circular_residual
from kd_sensing.evaluation.metrics import dba_from_circular_distances
from kd_sensing.utils.geometry import load_beam_angle_table


DEFAULT_ABLATIONS = (
    "gps_only",
    "lidar_only_no_bgam",
    "gps_lidar_no_bgam",
    "gps_lidar_topk_union_bgam",
    "gps_pseudo_history_soft_bgam",
    "gps_pseudo_history_topk_union_bgam",
    "gps_pseudo_history_per_candidate_rerank",
)


def run_deepsense6g_gps_lidar_bgam(
    cfg: Mapping[str, Any],
    *,
    support_ratio: float | None = None,
    label_space: str | None = None,
    topk: int | None = None,
    bgam_mode: str | None = None,
    output_dir: str | Path | None = None,
    ckpt: str | Path | None = None,
    evaluate_only: bool = False,
    debug_masks: bool | None = None,
) -> dict[str, Any]:
    data_cfg = _mapping(cfg.get("data"))
    candidate_cfg = _mapping(cfg.get("candidate"))
    outputs_cfg = _mapping(cfg.get("outputs"))
    train_cfg = _mapping(cfg.get("train"))
    eval_cfg = _mapping(cfg.get("eval"))
    metrics_cfg = _mapping(cfg.get("metrics"))
    workflow = str(_mapping(cfg.get("experiment")).get("name") or "deepsense6g_gps_lidar_bgam")
    ratio = float(support_ratio if support_ratio is not None else data_cfg.get("support_ratio", 0.15))
    selected_label_space = str(label_space or data_cfg.get("label_space", "mapping_disabled"))
    selected_topk = int(topk if topk is not None else candidate_cfg.get("topk", data_cfg.get("topk", 8)))
    num_beams = int(data_cfg.get("num_beams", candidate_cfg.get("num_beams", 64)))
    tag = ratio_tag(ratio)
    use_ratio_subdir = bool(outputs_cfg.get("use_support_ratio_subdir", workflow.startswith("deepsense6g")))
    out_root = Path(output_dir or outputs_cfg.get("root", "outputs/analysis/deepsense6g_gps_lidar_bgam"))
    result_dir = out_root / tag / selected_label_space if use_ratio_subdir else out_root / selected_label_space
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = result_dir / str(outputs_cfg.get("manifest_dir", "manifest"))
    manifest_path = manifest_dir / str(outputs_cfg.get("manifest_name", MANIFEST_NAME))
    if not manifest_path.exists():
        if workflow.startswith("mmw_town"):
            from kd_sensing.data.mmw_town_gps_lidar_bgam_manifest import build_mmw_town_gps_lidar_bgam_manifest

            build_mmw_town_gps_lidar_bgam_manifest(
                cfg,
                label_space=selected_label_space,
                topk=selected_topk,
                output_dir=out_root,
            )
        else:
            build_gps_lidar_bgam_manifest(
                cfg,
                support_ratio=ratio,
                label_space=selected_label_space,
                topk=selected_topk,
                output_dir=out_root,
            )
    pseudo_history_summary_path = manifest_dir / PSEUDO_HISTORY_SUMMARY_NAME
    rows = _read_csv(manifest_path)
    query_rows = [row for row in rows if _is_query_role(row.get("support_query_role") or row.get("split_role")) and _int(row.get("target_label"), -100) >= 0]
    support_rows = [row for row in rows if _is_support_role(row.get("support_query_role") or row.get("split_role")) and _int(row.get("target_label"), -100) >= 0]
    max_eval = int(eval_cfg.get("max_samples", train_cfg.get("max_eval_samples", 0)) or 0)
    if max_eval > 0:
        query_rows = query_rows[:max_eval]
    requested_ablations = [str(bgam_mode)] if bgam_mode else list(_mapping(cfg.get("ablation")).get("enabled") or DEFAULT_ABLATIONS)
    ablations = _dedupe(requested_ablations)
    beam_table = load_beam_angle_table(_mapping(cfg.get("geometry")), num_beams=num_beams)
    dba_delta = float(metrics_cfg.get("dba_delta", 5.0))
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    by_scene_rows: list[dict[str, Any]] = []
    by_mode_rows: list[dict[str, Any]] = []
    training_history: list[dict[str, Any]] = []
    checkpoints: dict[str, str] = {}
    skipped: dict[str, str] = {}
    warnings: list[str] = []
    lidar_quality: dict[str, Any] = {}

    for ablation in ablations:
        mode = _bgam_mode_for_ablation(ablation)
        uses_lidar = _ablation_uses_lidar(ablation)
        uses_pseudo_history = _ablation_uses_pseudo_history(ablation)
        if not query_rows:
            skipped[ablation] = "no_query_rows"
            run_rows: list[dict[str, Any]] = []
        elif uses_lidar and not any(_bool(row.get("lidar_available")) for row in rows):
            reason = _missing_lidar_reason(rows)
            skipped[ablation] = reason
            run_rows = []
        elif _uses_oracle_history(ablation) and not _rows_have_oracle_history(query_rows):
            skipped[ablation] = "missing_history_oracle_beams; oracle upper bound requires explicit oracle history artifact"
            run_rows = []
        elif uses_pseudo_history and not _rows_have_pseudo_history(query_rows):
            skipped[ablation] = _missing_pseudo_history_reason(query_rows)
            run_rows = []
        elif not uses_lidar or ablation not in _trainable_ablations(cfg) or evaluate_only and ckpt is None:
            run_rows = _prediction_rows_for_replay(
                query_rows,
                ablation=ablation,
                bgam_mode=mode,
                protocol=str(_mapping(cfg.get("experiment")).get("protocol", "target_adapt_beambench_gps_lidar_bgam")),
                support_ratio=ratio,
                label_space=selected_label_space,
                topk=selected_topk,
                num_beams=num_beams,
                beam_angle_source=beam_table.beam_angle_source,
                model_metadata={"trained_model_used": False},
            )
        else:
            trained = _train_and_predict(
                cfg,
                manifest_path,
                ablation=ablation,
                bgam_mode=mode,
                support_ratio=ratio,
                label_space=selected_label_space,
                topk=selected_topk,
                num_beams=num_beams,
                result_dir=result_dir,
                beam_angles=beam_table.angles,
                beam_angle_source=beam_table.beam_angle_source,
                ckpt=ckpt,
                evaluate_only=evaluate_only,
                debug_masks=bool(debug_masks if debug_masks is not None else _mapping(_mapping(cfg.get("bgam")).get("debug_masks")).get("enabled", False)),
            )
            run_rows = trained["prediction_rows"]
            training_history.extend(trained["history"])
            warnings.extend(trained["warnings"])
            if trained.get("checkpoint_path"):
                checkpoints[ablation] = str(trained["checkpoint_path"])
            if trained.get("lidar_quality_summary"):
                lidar_quality[ablation] = trained["lidar_quality_summary"]
        if ablation in skipped:
            summary_rows.append(_skipped_summary(ablation, mode, reason=skipped[ablation], support_ratio=ratio, label_space=selected_label_space, topk=selected_topk))
            by_mode_rows.append(dict(summary_rows[-1]))
            continue
        prediction_rows.extend(run_rows)
        summary = _summary_row(
            run_rows,
            ablation=ablation,
            bgam_mode=mode,
            support_ratio=ratio,
            label_space=selected_label_space,
            topk=selected_topk,
            support_count=len(support_rows),
            query_count=len(run_rows),
            dba_delta=dba_delta,
        )
        summary_rows.append(summary)
        by_mode_rows.append(dict(summary))
        for scene in sorted({str(row.get("scene") or "") for row in run_rows}):
            scene_rows = [row for row in run_rows if str(row.get("scene") or "") == scene]
            by_scene_rows.append(
                _summary_row(
                    scene_rows,
                    ablation=ablation,
                    bgam_mode=mode,
                    support_ratio=ratio,
                    label_space=selected_label_space,
                    topk=selected_topk,
                    support_count=len([row for row in support_rows if str(row.get("scene") or "") == scene]),
                    query_count=len(scene_rows),
                    dba_delta=dba_delta,
                    scene=scene,
                )
            )

    _write_csv(result_dir / "summary_overall.csv", summary_rows)
    _write_csv(result_dir / "summary_by_scene.csv", by_scene_rows)
    _write_csv(result_dir / "summary_by_bgam_mode.csv", by_mode_rows)
    _write_csv(result_dir / "predictions.csv", prediction_rows)
    if training_history:
        _write_csv(result_dir / "training_history.csv", training_history)
    metrics = {
        "workflow": workflow,
        "result_dir": str(result_dir),
        "summary_overall": summary_rows,
        "best_ablation": _best_ablation(summary_rows),
        "pseudo_history_summary": str(pseudo_history_summary_path) if pseudo_history_summary_path.exists() else "",
        "query_label_used_for_training": False,
    }
    (result_dir / "metrics.json").write_text(json.dumps(_json_ready(metrics), indent=2, sort_keys=True), encoding="utf-8")
    if bool(outputs_cfg.get("write_config_snapshot", True)):
        dump_config(dict(cfg), result_dir / "resolved_config.yaml")
    metadata = {
        "workflow": workflow,
        "result_dir": str(result_dir),
        "manifest_path": str(manifest_path),
        "support_ratio": ratio,
        "ratio_tag": tag,
        "label_space": selected_label_space,
        "beam_label_space": _dominant(rows, "beam_label_space"),
        "beam_label_mapping_fingerprint": _dominant(rows, "beam_label_mapping_fingerprint"),
        "topk": selected_topk,
        "num_beams": num_beams,
        "support_count": len(support_rows),
        "query_count": len(query_rows),
        "enabled_ablations": ablations,
        "skipped_ablations": skipped,
        "checkpoints": checkpoints,
        "beam_angle": beam_table.metadata,
        "gps_v2_artifact_path": str(data_cfg.get("gps_v2_artifact_root", "")),
        "top8_manifest_path": str(data_cfg.get("top8_manifest_path", "")),
        "pseudo_history_summary_path": str(pseudo_history_summary_path) if pseudo_history_summary_path.exists() else "",
        "lidar_profile": str(_mapping(cfg.get("lidar")).get("profile", "bev_cache")),
        "lidar_quality_summary": lidar_quality,
        "runtime_resource_controls": {
            "cpu_threads": train_cfg.get("cpu_threads", ""),
            "interop_threads": train_cfg.get("interop_threads", ""),
            "train_num_workers": train_cfg.get("num_workers", 0),
            "eval_num_workers": eval_cfg.get("num_workers", train_cfg.get("num_workers", 0)),
            "pin_memory": train_cfg.get("pin_memory", "auto"),
            "prefetch_factor": train_cfg.get("prefetch_factor", ""),
            "persistent_workers": train_cfg.get("persistent_workers", False),
        },
        "query_label_used_for_training": False,
        "standard_artifacts": [
            "metrics.json",
            "summary_overall.csv",
            "summary_by_scene.csv",
            "summary_by_bgam_mode.csv",
            "predictions.csv",
            f"manifest/{PSEUDO_HISTORY_SUMMARY_NAME}",
            "run_metadata.json",
            "resolved_config.yaml",
        ],
        "warnings": warnings,
    }
    (result_dir / "run_metadata.json").write_text(json.dumps(_json_ready(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "result_dir": str(result_dir),
        "manifest_path": str(manifest_path),
        "pseudo_history_summary_path": str(pseudo_history_summary_path) if pseudo_history_summary_path.exists() else "",
        "metrics_path": str(result_dir / "metrics.json"),
        "predictions_path": str(result_dir / "predictions.csv"),
        "summary_overall": str(result_dir / "summary_overall.csv"),
        "checkpoint_paths": checkpoints,
        "prediction_rows": len(prediction_rows),
        "support_count": len(support_rows),
        "query_count": len(query_rows),
        "warnings": warnings,
        "skipped_ablations": skipped,
    }


def evaluate_deepsense6g_gps_lidar_bgam(
    cfg: Mapping[str, Any],
    *,
    ckpt: str | Path | None = None,
    output_dir: str | Path | None = None,
    support_ratio: float | None = None,
    label_space: str | None = None,
    topk: int | None = None,
    bgam_mode: str | None = None,
    debug_masks: bool | None = None,
) -> dict[str, Any]:
    return run_deepsense6g_gps_lidar_bgam(
        cfg,
        support_ratio=support_ratio,
        label_space=label_space,
        topk=topk,
        bgam_mode=bgam_mode,
        output_dir=output_dir,
        ckpt=ckpt,
        evaluate_only=True,
        debug_masks=debug_masks,
    )


def _train_and_predict(
    cfg: Mapping[str, Any],
    manifest_path: Path,
    *,
    ablation: str,
    bgam_mode: str,
    support_ratio: float,
    label_space: str,
    topk: int,
    num_beams: int,
    result_dir: Path,
    beam_angles: np.ndarray,
    beam_angle_source: str,
    ckpt: str | Path | None,
    evaluate_only: bool,
    debug_masks: bool,
) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, Subset

        from kd_sensing.data.deepsense6g_gps_lidar_bgam_dataset import GPSLidarBGAMDataset, collate_gps_lidar_bgam_batch
        from kd_sensing.losses.gps_lidar_bgam_losses import GPSLidarBGAMLoss
        from kd_sensing.models.gps_lidar_bgam import save_debug_masks
        from kd_sensing.models.gps_lidar_bgam_model import GPSLidarBGAMBeamPredictor
        from kd_sensing.models.lidar_pillar_encoder import lidar_quality_summary
    except ImportError as exc:  # pragma: no cover
        rows = [row for row in _read_csv(manifest_path) if _is_query_role(row.get("support_query_role") or row.get("split_role"))]
        return {
            "prediction_rows": _prediction_rows_for_replay(
                rows,
                ablation=ablation,
                bgam_mode=bgam_mode,
                protocol=str(_mapping(cfg.get("experiment")).get("protocol", "")),
                support_ratio=support_ratio,
                label_space=label_space,
                topk=topk,
                num_beams=num_beams,
                beam_angle_source=beam_angle_source,
                model_metadata={"trained_model_used": False, "training_fallback_reason": str(exc)},
            ),
            "history": [],
            "warnings": [f"{ablation} fell back to GPS replay: {exc}"],
        }

    train_cfg = _mapping(cfg.get("train"))
    eval_cfg = _mapping(cfg.get("eval"))
    model_cfg = _mapping(cfg.get("model"))
    lidar_cfg = _mapping(cfg.get("lidar"))
    bgam_cfg = _mapping(cfg.get("bgam"))
    seed = int(train_cfg.get("seed", 42))
    _configure_torch_runtime(torch, train_cfg)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_request = str(train_cfg.get("device", "auto")).lower()
    device = torch.device("cuda" if device_request == "auto" and torch.cuda.is_available() else ("cpu" if device_request == "auto" else device_request))
    dataset = GPSLidarBGAMDataset(
        manifest_path,
        topk=topk,
        num_beams=num_beams,
        data_root=str(lidar_cfg.get("data_root", _mapping(cfg.get("data")).get("data_root", ""))),
        lidar_cfg=lidar_cfg,
        load_lidar=True,
        lidar_profile=str(lidar_cfg.get("profile", "bev_cache")),
        missing_lidar_policy=str(lidar_cfg.get("missing_policy", "zeros")),
        fit_normalizer=True,
        include_query_labels=True,
    )
    train_indices, val_indices = _train_val_indices(dataset.rows, seed=seed, validation_fraction=float(train_cfg.get("validation_fraction", 0.25)))
    query_indices = [idx for idx, row in enumerate(dataset.rows) if _is_query_role(row.get("support_query_role") or row.get("split_role")) and _int(row.get("target_label"), -100) >= 0]
    max_train = int(train_cfg.get("max_train_samples", 0) or 0)
    max_eval = int(eval_cfg.get("max_samples", train_cfg.get("max_eval_samples", 0)) or 0)
    if max_train > 0:
        train_indices = train_indices[:max_train]
        val_indices = val_indices[: max(1, min(len(val_indices), max_train))]
    if max_eval > 0:
        query_indices = query_indices[:max_eval]
    model = _build_model(cfg, topk=topk, num_beams=num_beams, bgam_mode=bgam_mode).to(device)
    checkpoint_path = Path(ckpt) if ckpt else result_dir / "checkpoints" / f"{ablation}.pt"
    history: list[dict[str, Any]] = []
    lidar_batches: list[torch.Tensor] = []
    if ckpt and Path(ckpt).exists():
        payload = torch.load(ckpt, map_location=device)
        state = payload.get("model_state_dict", payload) if isinstance(payload, Mapping) else payload
        model.load_state_dict(state, strict=False)
    elif not evaluate_only and train_indices:
        criterion = GPSLidarBGAMLoss(_mapping(cfg.get("loss")))
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=float(train_cfg.get("lr", 1e-3)), weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=int(train_cfg.get("batch_size", 8)),
            shuffle=True,
            num_workers=int(train_cfg.get("num_workers", 0)),
            generator=generator,
            collate_fn=collate_gps_lidar_bgam_batch,
            **_dataloader_runtime_kwargs(train_cfg, device=device),
        )
        best_state: dict[str, Any] | None = None
        best_metric = -float("inf")
        epochs = max(1, int(train_cfg.get("epochs", 2)))
        for epoch in range(1, epochs + 1):
            model.train()
            loss_total = 0.0
            count_total = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                inputs = _model_inputs_from_batch(batch, device=device, beam_angles=beam_angles, ablation=ablation)
                outputs = model(**inputs, bgam_mode=bgam_mode)
                losses = criterion(outputs, _batch_to_device(batch, device))
                losses["loss"].backward()
                optimizer.step()
                sample_count = max(float(losses["train_sample_count"].detach().cpu().item()), 1.0)
                loss_total += float(losses["loss"].detach().cpu().item()) * sample_count
                count_total += int(sample_count)
                if "lidar_bev" in batch and len(lidar_batches) < 3:
                    lidar_batches.append(batch["lidar_bev"].detach().cpu())
            val_rows = _prediction_rows_from_model(
                model,
                dataset,
                val_indices,
                ablation=ablation,
                bgam_mode=bgam_mode,
                support_ratio=support_ratio,
                label_space=label_space,
                topk=topk,
                num_beams=num_beams,
                beam_angles=beam_angles,
                beam_angle_source=beam_angle_source,
                device=device,
                batch_size=int(eval_cfg.get("batch_size", train_cfg.get("batch_size", 8))),
                num_workers=int(eval_cfg.get("num_workers", train_cfg.get("num_workers", 0))),
                loader_cfg=eval_cfg or train_cfg,
                debug_masks=False,
                result_dir=result_dir,
            )
            metric = dba_from_circular_distances([_float(row.get("final_error"), 0.0) for row in val_rows], delta=float(_mapping(cfg.get("metrics")).get("dba_delta", 5.0)))
            history.append(
                {
                    "ablation": ablation,
                    "event": "epoch",
                    "epoch": epoch,
                    "train_loss": loss_total / max(count_total, 1),
                    "train_sample_count": len(train_indices),
                    "validation_sample_count": len(val_indices),
                    "validation_DBA": metric,
                    "query_label_used_for_training": False,
                    "device": str(device),
                }
            )
            if metric >= best_metric:
                best_metric = float(metric)
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state, strict=False)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "ablation": ablation,
                "bgam_mode": bgam_mode,
                "topk": topk,
                "num_beams": num_beams,
                "query_label_used_for_training": False,
            },
            checkpoint_path,
        )
    query_rows = _prediction_rows_from_model(
        model,
        dataset,
        query_indices,
        ablation=ablation,
        bgam_mode=bgam_mode,
        support_ratio=support_ratio,
        label_space=label_space,
        topk=topk,
        num_beams=num_beams,
        beam_angles=beam_angles,
        beam_angle_source=beam_angle_source,
        device=device,
        batch_size=int(eval_cfg.get("batch_size", train_cfg.get("batch_size", 8))),
        num_workers=int(eval_cfg.get("num_workers", train_cfg.get("num_workers", 0))),
        loader_cfg=eval_cfg or train_cfg,
        debug_masks=debug_masks,
        result_dir=result_dir,
    )
    quality = lidar_quality_summary(
        lidar_batches,
        roi=tuple(lidar_cfg.get("roi", (-30.0, 30.0, -30.0, 30.0, -3.0, 5.0))),
        bev_size=tuple(lidar_cfg.get("bev_size", (64, 64))),
        cache_path=str(lidar_cfg.get("cache_dir", "")),
    )
    return {
        "prediction_rows": query_rows,
        "history": history,
        "warnings": [],
        "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else "",
        "lidar_quality_summary": quality,
    }


def _configure_torch_runtime(torch_module: Any, train_cfg: Mapping[str, Any]) -> None:
    cpu_threads = int(train_cfg.get("cpu_threads", 0) or 0)
    interop_threads = int(train_cfg.get("interop_threads", 0) or 0)
    if cpu_threads > 0:
        torch_module.set_num_threads(cpu_threads)
    if interop_threads > 0:
        try:
            torch_module.set_num_interop_threads(interop_threads)
        except RuntimeError:
            pass
    if bool(train_cfg.get("cudnn_benchmark", True)) and hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.benchmark = True


def _dataloader_runtime_kwargs(loader_cfg: Mapping[str, Any], *, device: Any) -> dict[str, Any]:
    workers = int(loader_cfg.get("num_workers", 0) or 0)
    pin_request = loader_cfg.get("pin_memory", "auto")
    pin_memory = bool(str(device).startswith("cuda")) if str(pin_request).lower() == "auto" else _bool(pin_request)
    kwargs: dict[str, Any] = {"pin_memory": pin_memory}
    if workers > 0:
        kwargs["persistent_workers"] = bool(loader_cfg.get("persistent_workers", True))
        prefetch = int(loader_cfg.get("prefetch_factor", 2) or 2)
        if prefetch > 0:
            kwargs["prefetch_factor"] = prefetch
    return kwargs


def _build_model(cfg: Mapping[str, Any], *, topk: int, num_beams: int, bgam_mode: str) -> Any:
    from kd_sensing.models.gps_lidar_bgam_model import GPSLidarBGAMBeamPredictor

    model_cfg = _mapping(cfg.get("model"))
    lidar_model_cfg = _mapping(model_cfg.get("lidar"))
    lidar_cfg = _mapping(cfg.get("lidar"))
    bgam_cfg = _mapping(cfg.get("bgam"))
    attention_cfg = _mapping(model_cfg.get("cross_attention"))
    return GPSLidarBGAMBeamPredictor(
        topk=topk,
        num_beams=num_beams,
        d_model=int(model_cfg.get("d_model", 64)),
        hidden_dim=int(model_cfg.get("hidden_dim", 96)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        fusion=str(model_cfg.get("fusion", "concat_mlp")),
        lambda_lidar_init=float(model_cfg.get("lambda_lidar_init", 0.5)),
        lambda_lidar_max=float(model_cfg.get("lambda_lidar_max", 3.0)),
        lidar_in_channels=int(lidar_model_cfg.get("in_channels", lidar_cfg.get("input_channels", 3))),
        lidar_channels=tuple(lidar_model_cfg.get("channels", (32, 64))),
        freeze_lidar_encoder=bool(lidar_model_cfg.get("freeze_lidar_encoder", False)),
        roi=tuple(lidar_cfg.get("roi", (-30.0, 30.0, -30.0, 30.0, -3.0, 5.0))),
        bev_size=tuple(lidar_cfg.get("bev_size", (64, 64))),
        bgam_mode=bgam_mode,
        bgam_sigma=float(bgam_cfg.get("sigma", 0.35)),
        bgam_hard_half_width=float(bgam_cfg.get("hard_half_width", 0.28)),
        adaptive_sigma=_mapping(bgam_cfg.get("adaptive_sigma")),
        attention_heads=int(attention_cfg.get("num_heads", 4)),
        attention_queries=int(attention_cfg.get("num_queries", 1)),
        full64_head_enabled=bool(_mapping(model_cfg.get("full64_head")).get("enabled", False)),
        lidar_profile=str(lidar_cfg.get("profile", "bev_cache")),
    )


def _prediction_rows_from_model(
    model: Any,
    dataset: Any,
    indices: Sequence[int],
    *,
    ablation: str,
    bgam_mode: str,
    support_ratio: float,
    label_space: str,
    topk: int,
    num_beams: int,
    beam_angles: np.ndarray,
    beam_angle_source: str,
    device: Any,
    batch_size: int,
    debug_masks: bool,
    result_dir: Path,
    num_workers: int = 0,
    loader_cfg: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not indices:
        return []
    import torch
    from torch.utils.data import DataLoader, Subset

    from kd_sensing.data.deepsense6g_gps_lidar_bgam_dataset import collate_gps_lidar_bgam_batch
    from kd_sensing.models.gps_lidar_bgam import save_debug_masks

    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_gps_lidar_bgam_batch,
        **_dataloader_runtime_kwargs(loader_cfg or {}, device=device),
    )
    rows: list[dict[str, Any]] = []
    model.eval()
    debug_written = False
    with torch.no_grad():
        for batch in loader:
            outputs = model(**_model_inputs_from_batch(batch, device=device, beam_angles=beam_angles, ablation=ablation), bgam_mode=bgam_mode)
            if debug_masks and not debug_written and outputs.get("bgam_mask") is not None:
                save_debug_masks(
                    outputs["bgam_mask"],
                    output_dir=result_dir / "debug_masks" / ablation,
                    sample_ids=batch.get("sample_id", []),
                    theta_gps=batch["theta_gps"],
                    mode=bgam_mode,
                    sigma=None,
                    half_width=None,
                    beam_angle_source=beam_angle_source,
                    max_samples=8,
                )
                debug_written = True
            probs = outputs["candidate_probs"].detach().cpu()
            scores = outputs["final_candidate_scores"].detach().cpu()
            selected = outputs["selected_beam"].detach().cpu()
            row_indices = batch["row_index"].detach().cpu().tolist()
            for offset, row_index in enumerate(row_indices):
                source_row = dataset.rows[int(row_index)]
                rows.append(
                    _prediction_item_from_row(
                        source_row,
                        candidate_scores=[float(value) for value in scores[offset].tolist()],
                        candidate_probs=[float(value) for value in probs[offset].tolist()],
                        selected_beam=int(selected[offset].item()),
                        ablation=ablation,
                        bgam_mode=bgam_mode,
                        support_ratio=support_ratio,
                        label_space=label_space,
                        topk=topk,
                        num_beams=num_beams,
                        beam_angle_source=beam_angle_source,
                        model_metadata={"trained_model_used": True},
                    )
                )
    return rows


def _prediction_rows_for_replay(
    rows: Sequence[Mapping[str, Any]],
    *,
    ablation: str,
    bgam_mode: str,
    protocol: str,
    support_ratio: float,
    label_space: str,
    topk: int,
    num_beams: int,
    beam_angle_source: str,
    model_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        candidate_probs = [_float(row.get(f"cand{idx}_prob"), 0.0) for idx in range(topk)]
        if sum(candidate_probs) <= 0:
            candidate_probs = [1.0 / max(topk, 1) for _ in range(topk)]
        scores = [math.log(max(prob, 1e-12)) for prob in candidate_probs]
        selected_idx = int(np.argmax(np.asarray(candidate_probs, dtype=np.float64)))
        selected_beam = _int(row.get(f"cand{selected_idx}_beam"), _int(row.get("gps_top1"), -1))
        out.append(
            _prediction_item_from_row(
                row,
                candidate_scores=scores,
                candidate_probs=candidate_probs,
                selected_beam=selected_beam,
                ablation=ablation,
                bgam_mode=bgam_mode,
                support_ratio=support_ratio,
                label_space=label_space,
                topk=topk,
                num_beams=num_beams,
                beam_angle_source=beam_angle_source,
                model_metadata={"protocol": protocol, **dict(model_metadata or {})},
            )
        )
    return out


def _prediction_item_from_row(
    row: Mapping[str, Any],
    *,
    candidate_scores: Sequence[float],
    candidate_probs: Sequence[float],
    selected_beam: int,
    ablation: str,
    bgam_mode: str,
    support_ratio: float,
    label_space: str,
    topk: int,
    num_beams: int,
    beam_angle_source: str,
    model_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = _int(row.get("gt_beam"), _int(row.get("target_label"), -100))
    gps_top1 = _int(row.get("gps_top1"), _int(row.get("gps_pred_top1"), _int(row.get("cand0_beam"), -1)))
    candidate_beams = [_int(row.get(f"cand{idx}_beam"), -1) for idx in range(topk)]
    ordered = [candidate_beams[idx] for idx in np.argsort(np.asarray(candidate_scores, dtype=np.float64))[::-1].tolist()]
    gps_error = circular_distance(gps_top1, target, num_beams=num_beams) if gps_top1 >= 0 and target >= 0 else 0
    final_error = circular_distance(selected_beam, target, num_beams=num_beams) if selected_beam >= 0 and target >= 0 else 0
    target_in_topk = target in set(candidate_beams) if target >= 0 else False
    selected_candidate_index = -1
    for idx, beam in enumerate(candidate_beams):
        if int(beam) == int(selected_beam):
            selected_candidate_index = idx
            break
    gps_gain = _optional_float(row.get("gps_normalized_gain"))
    final_gain = (
        _optional_float(row.get(f"cand{selected_candidate_index}_normalized_gain"))
        if selected_candidate_index >= 0
        else None
    )
    item = {
        "sample_id": row.get("sample_id", ""),
        "scene": row.get("scene", ""),
        "scenario_id": row.get("scenario_id", row.get("scene", "")),
        "support_query_role": row.get("support_query_role", ""),
        "gt_beam": target,
        "target_label": target,
        "label_space": label_space,
        "beam_label_space": row.get("beam_label_space", ""),
        "beam_label_mapping_fingerprint": row.get("beam_label_mapping_fingerprint", ""),
        "gps_top1": gps_top1,
        "history_pseudo_top1": _history_top1(row),
        "history_pseudo_entropy_mean": _history_entropy_mean(row),
        "history_pseudo_coverage": _history_coverage(row),
        "history_source": row.get("history_pseudo_label_source", ""),
        "pred_beam": int(selected_beam),
        "final_top1": int(selected_beam),
        "gps_topk": json.dumps(candidate_beams),
        "model_topk": json.dumps(ordered),
        "correct": int(selected_beam) == int(target),
        "target_in_topk": target_in_topk,
        "target_in_top8": target_in_topk,
        "gps_error": gps_error,
        "final_error": final_error,
        "gps_circular_error": gps_error,
        "final_circular_error": final_error,
        "gps_normalized_gain": "" if gps_gain is None else gps_gain,
        "final_normalized_gain": "" if final_gain is None else final_gain,
        "delta_normalized_gain_vs_GPS": ""
        if gps_gain is None or final_gain is None
        else float(final_gain - gps_gain),
        "gps_signed_residual": signed_circular_residual(target, gps_top1, num_beams=num_beams) if target >= 0 and gps_top1 >= 0 else "",
        "final_signed_residual": signed_circular_residual(target, selected_beam, num_beams=num_beams) if target >= 0 and selected_beam >= 0 else "",
        "improvement": gps_error - final_error,
        "candidate_scores_json": json.dumps([float(value) for value in candidate_scores]),
        "candidate_probs_json": json.dumps([float(value) for value in candidate_probs]),
        "ablation": ablation,
        "bgam_mode": bgam_mode,
        "uses_oracle_history_label": _uses_oracle_history(ablation),
        "beam_angle_source": beam_angle_source or row.get("beam_angle_source", ""),
        "support_ratio": support_ratio,
        "topk": topk,
        "query_label_used_for_training": False,
    }
    if model_metadata:
        item.update(dict(model_metadata))
    return item


def _model_inputs_from_batch(batch: Mapping[str, Any], *, device: Any, beam_angles: np.ndarray, ablation: str) -> dict[str, Any]:
    import torch

    candidate_log_probs = batch["candidate_log_probs"].to(device)
    candidate_probs = batch["candidate_probs"].to(device)
    if ablation == "lidar_only_no_bgam":
        candidate_log_probs = torch.zeros_like(candidate_log_probs)
        candidate_probs = torch.full_like(candidate_probs, 1.0 / max(candidate_probs.shape[-1], 1))
    inputs = {
        "candidate_beams": batch["candidate_beams"].to(device),
        "candidate_log_probs": candidate_log_probs,
        "candidate_probs": candidate_probs,
        "theta_gps": batch["theta_gps"].to(device),
        "distance_to_rsu": batch["distance_to_rsu"].to(device),
        "gps_entropy": batch.get("gps_entropy", torch.zeros_like(batch["theta_gps"])).to(device),
        "beam_angles": torch.as_tensor(beam_angles, device=device, dtype=torch.float32),
        "beam_angle_source": batch.get("beam_angle_source", ""),
    }
    if "lidar_bev" in batch:
        inputs["lidar_bev"] = batch["lidar_bev"].to(device)
    if "raw_points" in batch:
        inputs["raw_points"] = [points.to(device) for points in batch["raw_points"]]
    if "gps_probs" in batch:
        inputs["gps_probs"] = batch["gps_probs"].to(device)
    if "gps_logits" in batch:
        inputs["gps_logits"] = batch["gps_logits"].to(device)
    for key in ("history_pseudo_beams", "history_pseudo_probs", "history_pseudo_entropy", "history_valid_mask"):
        if key in batch and hasattr(batch[key], "numel") and int(batch[key].numel()) > 0:
            inputs[key] = batch[key].to(device)
    return inputs


def _batch_to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _train_val_indices(rows: Sequence[Mapping[str, Any]], *, seed: int, validation_fraction: float) -> tuple[list[int], list[int]]:
    support = [idx for idx, row in enumerate(rows) if _is_support_role(row.get("support_query_role") or row.get("split_role")) and _int(row.get("target_label"), -100) >= 0]
    if not support:
        return [], []
    rng = np.random.default_rng(int(seed))
    if len(support) >= 4:
        val_count = max(1, min(len(support) - 1, int(round(len(support) * float(validation_fraction)))))
        val = set(int(item) for item in rng.choice(np.asarray(support, dtype=np.int64), size=val_count, replace=False))
    else:
        val = set(support)
    train = [idx for idx in support if idx not in val] or list(support)
    return train, sorted(val)


def _summary_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    ablation: str,
    bgam_mode: str,
    support_ratio: float,
    label_space: str,
    topk: int,
    support_count: int,
    query_count: int,
    dba_delta: float,
    scene: str | None = None,
) -> dict[str, Any]:
    final_errors = np.asarray([_float(row.get("final_error"), 0.0) for row in rows], dtype=np.float64)
    gps_errors = np.asarray([_float(row.get("gps_error"), 0.0) for row in rows], dtype=np.float64)
    final_gains = np.asarray(
        [value for value in (_optional_float(row.get("final_normalized_gain")) for row in rows) if value is not None],
        dtype=np.float64,
    )
    gps_gains = np.asarray(
        [value for value in (_optional_float(row.get("gps_normalized_gain")) for row in rows) if value is not None],
        dtype=np.float64,
    )
    target = [_int(row.get("gt_beam"), -100) for row in rows]
    model_topk = [_json_list(row.get("model_topk")) for row in rows]
    result = {
        "ablation": ablation,
        "bgam_mode": bgam_mode,
        "sample_count": len(rows),
        "support_count": int(support_count),
        "query_count": int(query_count),
        "support_ratio": support_ratio,
        "label_space": label_space,
        "beam_label_space": str(rows[0].get("beam_label_space", "")) if rows else "",
        "beam_label_mapping_fingerprint": str(rows[0].get("beam_label_mapping_fingerprint", "")) if rows else "",
        "topk": topk,
        "Top1": float(np.mean(final_errors == 0)) if final_errors.size else 0.0,
        "Top3": _topk_acc(model_topk, target, 3),
        "Top5": _topk_acc(model_topk, target, 5),
        "Top8": _topk_acc(model_topk, target, 8),
        "DBA": dba_from_circular_distances(final_errors, delta=dba_delta),
        "mean_circular_error": float(final_errors.mean()) if final_errors.size else 0.0,
        "median_circular_error": float(np.median(final_errors)) if final_errors.size else 0.0,
        "gps_DBA": dba_from_circular_distances(gps_errors, delta=dba_delta),
        "gps_mean_circular_error": float(gps_errors.mean()) if gps_errors.size else 0.0,
        "delta_vs_GPS": dba_from_circular_distances(final_errors, delta=dba_delta) - dba_from_circular_distances(gps_errors, delta=dba_delta),
        "delta_mean_error_vs_GPS": (float(final_errors.mean()) if final_errors.size else 0.0) - (float(gps_errors.mean()) if gps_errors.size else 0.0),
        "mean_normalized_gain": float(final_gains.mean()) if final_gains.size else "",
        "gps_mean_normalized_gain": float(gps_gains.mean()) if gps_gains.size else "",
        "delta_normalized_gain_vs_GPS": ""
        if not final_gains.size or not gps_gains.size
        else float(final_gains.mean() - gps_gains.mean()),
        "target_in_topk_rate": float(np.mean([_bool(row.get("target_in_topk")) for row in rows])) if rows else 0.0,
        "pseudo_history_coverage": float(np.mean([_float(row.get("history_pseudo_coverage"), 0.0) for row in rows])) if rows else 0.0,
        "pseudo_history_entropy_mean": float(np.mean([_float(row.get("history_pseudo_entropy_mean"), 0.0) for row in rows])) if rows else 0.0,
        "evaluation_only_pseudo_accuracy": _pseudo_accuracy(rows),
        "beam_angle_source": str(rows[0].get("beam_angle_source", "")) if rows else "",
        "history_source": str(rows[0].get("history_source", "")) if rows else "",
        "uses_oracle_history_label": _uses_oracle_history(ablation),
        "query_label_used_for_training": False,
        "skipped_reason": "",
    }
    if scene is not None:
        result["scene"] = scene
    return result


def _skipped_summary(ablation: str, bgam_mode: str, *, reason: str, support_ratio: float, label_space: str, topk: int) -> dict[str, Any]:
    row = _summary_row([], ablation=ablation, bgam_mode=bgam_mode, support_ratio=support_ratio, label_space=label_space, topk=topk, support_count=0, query_count=0, dba_delta=5.0)
    row["skipped_reason"] = reason
    return row


def _topk_acc(model_topk: Sequence[Sequence[int]], target: Sequence[int], k: int) -> float:
    if not model_topk:
        return 0.0
    hits = []
    for beams, label in zip(model_topk, target):
        hits.append(int(label) in set(int(item) for item in beams[:k]))
    return float(np.mean(hits)) if hits else 0.0


def _json_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    try:
        return [int(item) for item in json.loads(str(value or "[]"))]
    except Exception:
        return []


def _best_ablation(rows: Sequence[Mapping[str, Any]]) -> str:
    usable = [
        row
        for row in rows
        if not str(row.get("skipped_reason") or "") and not _bool(row.get("uses_oracle_history_label"))
    ]
    if not usable:
        return ""
    return str(max(usable, key=lambda row: _float(row.get("DBA"), -1.0)).get("ablation") or "")


def _trainable_ablations(cfg: Mapping[str, Any]) -> set[str]:
    defaults = ("gps_pseudo_history_soft_bgam", "gps_pseudo_history_per_candidate_rerank")
    return {str(item) for item in _mapping(cfg.get("ablation")).get("trainable", defaults)}


def _ablation_uses_lidar(ablation: str) -> bool:
    return str(ablation) != "gps_only"


def _ablation_uses_pseudo_history(ablation: str) -> bool:
    value = str(ablation)
    return value.startswith("gps_pseudo_history") or value == "oracle_history_bgam_upper_bound"


def _uses_oracle_history(ablation: str) -> bool:
    return str(ablation) == "oracle_history_bgam_upper_bound"


def _bgam_mode_for_ablation(ablation: str) -> str:
    return {
        "gps_only": "none",
        "lidar_only_no_bgam": "none",
        "gps_lidar_no_bgam": "none",
        "gps_lidar_soft_bgam": "single_soft",
        "gps_lidar_hard_bgam": "single_hard",
        "gps_lidar_topk_union_bgam": "topk_union_soft",
        "gps_lidar_topk_per_candidate_rerank": "topk_per_candidate",
        "gps_pseudo_history_soft_bgam": "history_pseudo_soft",
        "gps_pseudo_history_topk_union_bgam": "history_pseudo_topk_union",
        "gps_pseudo_history_per_candidate_rerank": "history_pseudo_per_candidate",
        "oracle_history_bgam_upper_bound": "history_pseudo_soft",
    }.get(str(ablation), str(ablation))


def _rows_have_pseudo_history(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    required = ("history_pseudo_beams", "history_pseudo_probs", "history_pseudo_entropy", "history_valid_mask")
    return all(all(str(row.get(field) or "") for field in required) for row in rows)


def _rows_have_oracle_history(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(str(row.get("history_oracle_beams") or "") for row in rows)


def _missing_pseudo_history_reason(rows: Sequence[Mapping[str, Any]]) -> str:
    missing: dict[str, int] = {}
    for row in rows:
        for field in ("history_pseudo_beams", "history_pseudo_probs", "history_pseudo_entropy", "history_valid_mask"):
            if not str(row.get(field) or ""):
                missing[field] = missing.get(field, 0) + 1
    details = ",".join(f"{field}:{count}" for field, count in sorted(missing.items()))
    return f"missing_pseudo_history_fields({details}); pseudo_history_source=gps_v2_logits"


def _missing_lidar_reason(rows: Sequence[Mapping[str, Any]]) -> str:
    reasons: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("lidar_missing_reason") or "missing_lidar")
        reasons[reason] = reasons.get(reason, 0) + 1
    return max(reasons, key=reasons.get) if reasons else "missing_lidar"


def _history_top1(row: Mapping[str, Any]) -> int:
    beams = _json_list(row.get("history_pseudo_beams"))
    valid = _json_bool_list(row.get("history_valid_mask"))
    for idx in range(len(beams) - 1, -1, -1):
        if idx < len(valid) and valid[idx] and int(beams[idx]) >= 0:
            return int(beams[idx])
    return -1


def _history_entropy_mean(row: Mapping[str, Any]) -> float:
    entropy = _json_float_list(row.get("history_pseudo_entropy"))
    valid = _json_bool_list(row.get("history_valid_mask"))
    values = [float(value) for idx, value in enumerate(entropy) if idx < len(valid) and valid[idx]]
    return float(np.mean(values)) if values else 0.0


def _history_coverage(row: Mapping[str, Any]) -> float:
    valid = _json_bool_list(row.get("history_valid_mask"))
    return float(np.mean(valid)) if valid else 0.0


def _pseudo_accuracy(rows: Sequence[Mapping[str, Any]]) -> float | str:
    hits = []
    for row in rows:
        target = _int(row.get("gt_beam"), _int(row.get("target_label"), -100))
        pseudo = _int(row.get("history_pseudo_top1"), -1)
        if target >= 0 and pseudo >= 0:
            hits.append(int(target) == int(pseudo))
    return float(np.mean(hits)) if hits else ""


def _json_float_list(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    try:
        return [float(item) for item in json.loads(str(value or "[]"))]
    except Exception:
        return []


def _json_bool_list(value: Any) -> list[bool]:
    if isinstance(value, list):
        return [bool(item) for item in value]
    try:
        return [bool(item) for item in json.loads(str(value or "[]"))]
    except Exception:
        return []


def _dominant(rows: Sequence[Mapping[str, Any]], field: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "")
        if value:
            counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    names = _fieldnames(rows)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in names})


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    candidate = Path(path)
    if not candidate.exists():
        return []
    with candidate.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


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
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _dedupe(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value)
        if item and item not in result:
            result.append(item)
    return result


def _is_query_role(value: Any) -> bool:
    return str(value or "").startswith("query")


def _is_support_role(value: Any) -> bool:
    role = str(value or "")
    return bool(role) and not role.startswith("query")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _optional_float(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in {None, ""}:
            return int(default)
        return int(float(str(value)))
    except (TypeError, ValueError):
        return int(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


__all__ = [
    "DEFAULT_ABLATIONS",
    "evaluate_deepsense6g_gps_lidar_bgam",
    "run_deepsense6g_gps_lidar_bgam",
]
