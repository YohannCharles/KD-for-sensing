from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import ConcatDataset

from kd_sensing.baselines.beambench.image_ae_gps_config import (
    ImageAEGPSDirectTrainingConfig,
    TARGET_TABLE_III_ROW,
    _build_adamw,
    _configure_torch_runtime,
    _gps_calibration_metadata,
    _gps_scaler_from_metadata,
    _gps_scaler_metadata,
    _make_grad_scaler,
    _resolve_amp_dtype,
    _resolve_device,
    _scene_specific_cfg,
    _seed_everything,
    _torch_load,
    resolve_image_ae_gps_config,
)
from kd_sensing.baselines.beambench.image_ae_gps_datasets import (
    _build_loader,
    _build_paper_split_scene_datasets,
    _build_split_dataset,
)
from kd_sensing.baselines.beambench.image_ae_gps_ae import (
    prepare_ae_feature_sources_for_image_gps_baseline,
    resolve_camera_ae_checkpoint_for_image_gps_baseline,
)
from kd_sensing.baselines.beambench.image_ae_gps_evaluation import evaluate_image_ae_gps_model
from kd_sensing.baselines.beambench.image_ae_gps_models import BeamBenchImageAEGPSDirectModel
from kd_sensing.baselines.beambench.image_ae_gps_reports import (
    _json_ready,
    _paper_split_gps_calibration_metadata,
    _paper_split_summary,
    _performance_metadata,
    _write_csv_rows,
    _write_paper_split_summary_artifacts,
)
from kd_sensing.baselines.beambench.image_ae_gps_training import (
    _resolve_classifier_selection_sources,
    _train_classifier_epoch,
    run_image_ae_gps_training,
)

def run_image_ae_gps_paper_split_training(
    config: Mapping[str, Any] | ImageAEGPSDirectTrainingConfig,
    *,
    train_scenes: Sequence[int] = (32, 33, 34),
    eval_scenes: Sequence[int] = (31, 32, 33, 34),
    output_root: str | Path = "outputs/scenegroup_s32_s34/beambench_image_ae_gps_direct_tableiii/paper_split",
) -> dict[str, Any]:
    """Train once on scenes 32-34 and evaluate scene31-34, matching the paper split more closely."""

    base_cfg = config if isinstance(config, ImageAEGPSDirectTrainingConfig) else resolve_image_ae_gps_config(config)
    _seed_everything(base_cfg.seed)
    device = _resolve_device(base_cfg.device)
    runtime_report = _configure_torch_runtime(base_cfg, device)
    amp_enabled = bool(base_cfg.amp) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(base_cfg.amp_dtype)
    grad_scaler = _make_grad_scaler(base_cfg, amp_enabled)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    train_cfgs, eval_cfgs, train_datasets, eval_datasets, gps_scaler = _build_paper_split_scene_datasets(
        base_cfg,
        train_scenes=train_scenes,
        eval_scenes=eval_scenes,
    )
    combined_train_dataset = ConcatDataset(train_datasets)
    ae_checkpoint, ae_report = resolve_camera_ae_checkpoint_for_image_gps_baseline(
        base_cfg,
        combined_train_dataset,
        output_dir=root,
        device=device,
    )

    model = BeamBenchImageAEGPSDirectModel(
        num_beams=base_cfg.num_beams,
        gps_input_size=base_cfg.gps_input_size,
        ae_latent_dim=base_cfg.ae_latent_dim,
        image_channels=base_cfg.image_channels,
        image_size=base_cfg.image_size,
        hidden_dim=base_cfg.fusion_hidden_dim,
        dropout=base_cfg.fusion_dropout,
        fusion_architecture=base_cfg.fusion_architecture,
        fusion_dense_hidden_sizes=base_cfg.fusion_dense_hidden_sizes,
        fusion_activation=base_cfg.fusion_activation,
        fusion_last_activation=base_cfg.fusion_last_activation,
        ae_checkpoint_path=ae_checkpoint,
        freeze_ae_encoder=base_cfg.freeze_ae_encoder,
    ).to(device)

    train_sources, eval_sources, feature_cache_reports = prepare_ae_feature_sources_for_image_gps_baseline(
        model,
        train_sources=list(zip(train_cfgs, train_datasets, strict=True)),
        eval_sources=list(zip(eval_cfgs, eval_datasets, strict=True)),
        cfg=base_cfg,
        output_root=root,
        device=device,
        ae_checkpoint=ae_checkpoint,
    )

    train_source = ConcatDataset(train_sources)
    eval_source_by_scene = {scene: eval_sources[int(scene)] for scene in eval_scenes}
    combined_eval_source = ConcatDataset([eval_source_by_scene[int(scene)] for scene in eval_scenes])
    fit_source, selection_source, selection_metadata = _resolve_classifier_selection_sources(
        train_source,
        combined_eval_source,
        base_cfg,
    )

    train_loader = _build_loader(
        fit_source,
        batch_size=base_cfg.fusion_batch_size,
        shuffle=True,
        num_workers=base_cfg.num_workers,
        cfg=base_cfg,
    )
    selection_loader = _build_loader(
        selection_source,
        batch_size=base_cfg.fusion_batch_size,
        shuffle=False,
        num_workers=base_cfg.num_workers,
        cfg=base_cfg,
    )
    optimizer = _build_adamw(
        (param for param in model.parameters() if param.requires_grad),
        lr=float(base_cfg.fusion_lr),
        weight_decay=float(base_cfg.fusion_weight_decay),
        device=device,
        fused=base_cfg.fused_optimizer,
    )
    best_path = root / "checkpoints" / "best_image_ae_gps_direct_paper_split.pt"
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    stale = 0
    for epoch in range(int(base_cfg.fusion_epochs)):
        train_loss = _train_classifier_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
            non_blocking=base_cfg.non_blocking_transfer,
        )
        selection_result = evaluate_image_ae_gps_model(
            model,
            selection_loader,
            base_cfg,
            device=device,
            predictions_path=None,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        score = float(selection_result["metrics"].get("official_top3_dba", 0.0))
        if score > best_score:
            best_score = score
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(base_cfg),
                    "model_metadata": model.metadata(),
                    "gps_scaler": _gps_scaler_metadata(gps_scaler),
                    "ae_checkpoint_path": str(ae_checkpoint),
                    "epoch": int(epoch),
                    "best_official_top3_dba": float(best_score),
                    "target_table_iii_row": TARGET_TABLE_III_ROW,
                    "selection": selection_metadata,
                    "gps_calibration": _gps_calibration_metadata(base_cfg),
                    "paper_split": {
                        "train_scenes": [int(scene) for scene in train_scenes],
                        "eval_scenes": [int(scene) for scene in eval_scenes],
                    },
                    "performance": _performance_metadata(base_cfg, device, amp_enabled, runtime_report, feature_cache_reports),
                },
                best_path,
            )
        else:
            stale += 1
        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(train_loss),
                "selection_split": str(selection_metadata["mode"]),
                "selection_official_top3_dba": score,
                "selection_circular_top3_dba": float(selection_result["metrics"].get("circular_top3_dba", 0.0)),
                "selection_official_top1_acc": float(selection_result["metrics"].get("official_top1_acc", 0.0)),
                "best_official_top3_dba": float(best_score),
            }
        )
        _write_csv_rows(root / "history.csv", history)
        if stale >= int(base_cfg.fusion_patience):
            break

    checkpoint = _torch_load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    scene_reports = []
    for scene in eval_scenes:
        scene_dir = root / f"scene{int(scene)}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        loader = _build_loader(
            eval_source_by_scene[int(scene)],
            batch_size=base_cfg.fusion_batch_size,
            shuffle=False,
            num_workers=base_cfg.num_workers,
            cfg=base_cfg,
        )
        result = evaluate_image_ae_gps_model(
            model,
            loader,
            base_cfg,
            device=device,
            predictions_path=scene_dir / "predictions.csv" if base_cfg.save_predictions else None,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        scene_report = {
            "scene": int(scene),
            "metrics": result["metrics"],
            "dataset": eval_datasets[[int(cfg.scene) for cfg in eval_cfgs].index(int(scene))].metadata(),
            "predictions_path": str(scene_dir / "predictions.csv") if base_cfg.save_predictions else None,
        }
        scene_reports.append(scene_report)
        (scene_dir / "metrics.json").write_text(
            json.dumps(_json_ready(result["metrics"]), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (scene_dir / "run_report.json").write_text(
            json.dumps(_json_ready(scene_report), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    train_scene_ids = [int(scene) for scene in train_scenes]
    eval_scene_ids = [int(scene) for scene in eval_scenes]
    train_scene_text = ", ".join(str(scene) for scene in train_scene_ids)
    eval_scene_text = ", ".join(str(scene) for scene in eval_scene_ids)
    summary = _paper_split_summary(scene_reports)
    report = {
        "workflow": "beambench_image_ae_gps_direct_paper_split_train",
        "paper_target": "Arnold22 BeamBench Table III Camera=AE GPS=Direct Fusion=Yes",
        "target_table_iii_row": TARGET_TABLE_III_ROW,
        "status": "local_paper_split_training_complete",
        "output_root": str(root),
        "checkpoint_path": str(best_path),
        "ae_checkpoint_path": str(ae_checkpoint),
        "ae_report": ae_report,
        "config": asdict(base_cfg),
        "device": str(device),
        "gps_calibration": _paper_split_gps_calibration_metadata(train_cfgs, eval_cfgs),
        "paper_split": {
            "train_scenes": train_scene_ids,
            "eval_scenes": eval_scene_ids,
        },
        "selection": selection_metadata,
        "performance": _performance_metadata(base_cfg, device, amp_enabled, runtime_report, feature_cache_reports),
        "train_datasets": [dataset.metadata() for dataset in train_datasets],
        "eval_reports": scene_reports,
        "summary": summary,
        "history_path": str(root / "history.csv"),
        "official_comparability_note": (
            f"本地训练使用 scenes {train_scene_text}，评估 scenes {eval_scene_text}；"
            "但仍未使用官方预训练权重、官方完整 NNI/剪枝搜索和官方 unseen test packaging。"
        ),
    }
    _write_paper_split_summary_artifacts(report, root)
    (root / "run_report.json").write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True), encoding="utf-8")
    return report

def run_image_ae_gps_paper_split_evaluation(
    checkpoint_path: str | Path,
    *,
    eval_scenes: Sequence[int] = (31, 32, 33, 34),
    output_root: str | Path = "outputs/evaluations/beambench_image_ae_gps_direct_tableiii/eval_checkpoint",
    config: Mapping[str, Any] | ImageAEGPSDirectTrainingConfig | None = None,
    train_scenes: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Evaluate an existing paper-split checkpoint on requested scenes without retraining."""

    checkpoint = _torch_load(Path(checkpoint_path), map_location="cpu")
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(f"Not a BeamBench Image AE + GPS checkpoint: {checkpoint_path}")
    ckpt_cfg = ImageAEGPSDirectTrainingConfig(**dict(checkpoint["config"]))
    override_cfg = None
    if config is not None:
        override_cfg = config if isinstance(config, ImageAEGPSDirectTrainingConfig) else resolve_image_ae_gps_config(config)
    if override_cfg is not None:
        ckpt_cfg = replace(
            ckpt_cfg,
            output_dir=str(output_root),
            device=override_cfg.device,
            num_workers=override_cfg.num_workers,
            pin_memory=override_cfg.pin_memory,
            persistent_workers=override_cfg.persistent_workers,
            prefetch_factor=override_cfg.prefetch_factor,
            non_blocking_transfer=override_cfg.non_blocking_transfer,
            amp=override_cfg.amp,
            amp_dtype=override_cfg.amp_dtype,
            amp_grad_scaler=override_cfg.amp_grad_scaler,
            allow_tf32=override_cfg.allow_tf32,
            cudnn_benchmark=override_cfg.cudnn_benchmark,
            fused_optimizer=override_cfg.fused_optimizer,
            cache_frozen_ae_features=override_cfg.cache_frozen_ae_features,
            feature_cache_batch_size=override_cfg.feature_cache_batch_size,
            feature_cache_dir=override_cfg.feature_cache_dir,
            save_predictions=override_cfg.save_predictions,
        )
    _seed_everything(ckpt_cfg.seed)
    device = _resolve_device(ckpt_cfg.device)
    runtime_report = _configure_torch_runtime(ckpt_cfg, device)
    amp_enabled = bool(ckpt_cfg.amp) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(ckpt_cfg.amp_dtype)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    ae_checkpoint = Path(str(checkpoint.get("ae_checkpoint_path") or ckpt_cfg.ae_checkpoint_path or ""))
    if not ae_checkpoint.exists():
        raise FileNotFoundError(f"Camera AE checkpoint recorded by fusion checkpoint is missing: {ae_checkpoint}")
    model = BeamBenchImageAEGPSDirectModel(
        num_beams=ckpt_cfg.num_beams,
        gps_input_size=ckpt_cfg.gps_input_size,
        ae_latent_dim=ckpt_cfg.ae_latent_dim,
        image_channels=ckpt_cfg.image_channels,
        image_size=ckpt_cfg.image_size,
        hidden_dim=ckpt_cfg.fusion_hidden_dim,
        dropout=ckpt_cfg.fusion_dropout,
        fusion_architecture=ckpt_cfg.fusion_architecture,
        fusion_dense_hidden_sizes=ckpt_cfg.fusion_dense_hidden_sizes,
        fusion_activation=ckpt_cfg.fusion_activation,
        fusion_last_activation=ckpt_cfg.fusion_last_activation,
        ae_checkpoint_path=ae_checkpoint,
        freeze_ae_encoder=ckpt_cfg.freeze_ae_encoder,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    gps_scaler = _gps_scaler_from_metadata(checkpoint.get("gps_scaler")) if ckpt_cfg.gps_normalize else None
    eval_cfgs = [_scene_specific_cfg(ckpt_cfg, scene) for scene in eval_scenes]
    eval_datasets = [
        _build_split_dataset(cfg, split="test", gps_scaler=gps_scaler, gps_normalize=ckpt_cfg.gps_normalize)
        for cfg in eval_cfgs
    ]
    _, eval_source_by_scene, feature_cache_reports = prepare_ae_feature_sources_for_image_gps_baseline(
        model,
        train_sources=[],
        eval_sources=list(zip(eval_cfgs, eval_datasets, strict=True)),
        cfg=ckpt_cfg,
        output_root=root,
        device=device,
        ae_checkpoint=ae_checkpoint,
    )

    scene_reports = []
    eval_scene_ids = [int(scene) for scene in eval_scenes]
    for scene in eval_scene_ids:
        scene_dir = root / f"scene{scene}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        loader = _build_loader(
            eval_source_by_scene[int(scene)],
            batch_size=ckpt_cfg.fusion_batch_size,
            shuffle=False,
            num_workers=ckpt_cfg.num_workers,
            cfg=ckpt_cfg,
        )
        result = evaluate_image_ae_gps_model(
            model,
            loader,
            ckpt_cfg,
            device=device,
            predictions_path=scene_dir / "predictions.csv" if ckpt_cfg.save_predictions else None,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        dataset = eval_datasets[eval_scene_ids.index(int(scene))]
        scene_report = {
            "scene": int(scene),
            "metrics": result["metrics"],
            "dataset": dataset.metadata(),
            "predictions_path": str(scene_dir / "predictions.csv") if ckpt_cfg.save_predictions else None,
        }
        scene_reports.append(scene_report)
        (scene_dir / "metrics.json").write_text(
            json.dumps(_json_ready(result["metrics"]), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (scene_dir / "run_report.json").write_text(
            json.dumps(_json_ready(scene_report), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    checkpoint_split = dict(checkpoint.get("paper_split") or {})
    train_scene_ids = [int(scene) for scene in (train_scenes or checkpoint_split.get("train_scenes") or (32, 33, 34))]
    train_scene_text = ", ".join(str(scene) for scene in train_scene_ids)
    eval_scene_text = ", ".join(str(scene) for scene in eval_scene_ids)
    summary = _paper_split_summary(scene_reports)
    report = {
        "workflow": "beambench_image_ae_gps_direct_paper_split_eval",
        "paper_target": "Arnold22 BeamBench Table III Camera=AE GPS=Direct Fusion=Yes",
        "target_table_iii_row": TARGET_TABLE_III_ROW,
        "status": "local_paper_split_eval_complete",
        "output_root": str(root),
        "checkpoint_path": str(checkpoint_path),
        "ae_checkpoint_path": str(ae_checkpoint),
        "config": asdict(ckpt_cfg),
        "device": str(device),
        "gps_calibration": _paper_split_gps_calibration_metadata(
            [_scene_specific_cfg(ckpt_cfg, scene) for scene in train_scene_ids],
            eval_cfgs,
        ),
        "paper_split": {
            "train_scenes": train_scene_ids,
            "eval_scenes": eval_scene_ids,
        },
        "selection": dict(checkpoint.get("selection") or {}),
        "performance": _performance_metadata(ckpt_cfg, device, amp_enabled, runtime_report, feature_cache_reports),
        "eval_reports": scene_reports,
        "summary": summary,
        "official_comparability_note": (
            f"本地 eval-only 使用已训练 paper-split checkpoint；训练 scenes {train_scene_text}，评估 scenes {eval_scene_text}；"
            "未使用官方预训练权重、官方完整 NNI/剪枝搜索和官方 unseen test packaging。"
        ),
    }
    _write_paper_split_summary_artifacts(report, root)
    (root / "run_report.json").write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True), encoding="utf-8")
    return report

__all__ = [
    "run_image_ae_gps_paper_split_evaluation",
    "run_image_ae_gps_paper_split_training",
]
