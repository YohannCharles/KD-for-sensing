import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from kd_sensing.baselines.beambench.image_ae_gps_ae import (
    _load_or_build_ae_feature_dataset,
    train_camera_ae_for_image_gps_baseline,
)
from kd_sensing.baselines.beambench.image_ae_gps_config import (
    ImageAEGPSDirectTrainingConfig,
    TARGET_TABLE_III_ROW,
    _autocast_context,
    _build_adamw,
    _configure_torch_runtime,
    _gps_calibration_metadata,
    _gps_scaler_metadata,
    _make_grad_scaler,
    _normalize_selection_split,
    _resolve_amp_dtype,
    _resolve_device,
    _scaler_enabled,
    _seed_everything,
    _torch_load,
    resolve_image_ae_gps_config,
)
from kd_sensing.baselines.beambench.image_ae_gps_datasets import (
    BeamBenchImageAEGPSDataset,
    _build_loader,
    _build_split_dataset,
    _split_dataset,
)
from kd_sensing.baselines.beambench.image_ae_gps_evaluation import evaluate_image_ae_gps_model
from kd_sensing.baselines.beambench.image_ae_gps_models import (
    BeamBenchDenseModel,
    BeamBenchImageAEGPSDirectModel,
    _classifier_logits_from_batch,
)
from kd_sensing.baselines.beambench.image_ae_gps_reports import (
    _json_ready,
    _performance_metadata,
    _write_csv_rows,
)
from kd_sensing.data.transform_ops.gps import GPSStandardScaler


def run_image_ae_gps_training(config: Mapping[str, Any] | ImageAEGPSDirectTrainingConfig) -> dict[str, Any]:
    cfg = config if isinstance(config, ImageAEGPSDirectTrainingConfig) else resolve_image_ae_gps_config(config)
    _seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)
    runtime_report = _configure_torch_runtime(cfg, device)
    amp_enabled = bool(cfg.amp) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(cfg.amp_dtype)
    grad_scaler = _make_grad_scaler(cfg, amp_enabled)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_dataset = BeamBenchImageAEGPSDataset(
        data_root=cfg.data_root,
        csv_name=cfg.train_csv_name,
        split="train",
        seq_len=cfg.seq_len,
        gps_seq_len=cfg.gps_seq_len,
        gps_source_seq_len=cfg.gps_source_seq_len,
        gps_input_seq_len=cfg.gps_input_seq_len,
        num_pred=cfg.num_pred,
        image_size=cfg.image_size,
        num_beams=cfg.num_beams,
        target_beam_source=cfg.target_beam_source,
        portion=cfg.train_portion,
        portion_strategy=cfg.portion_strategy,
        portion_seed=cfg.portion_seed,
        gps_feature_mode=cfg.gps_feature_mode,
        gps_angle_offset_rad=cfg.gps_angle_offset_rad,
        max_samples=cfg.max_train_samples,
    )
    gps_scaler = GPSStandardScaler().fit(train_dataset.raw_gps_matrix()) if cfg.gps_normalize else None
    train_dataset.gps_scaler = gps_scaler
    test_dataset = BeamBenchImageAEGPSDataset(
        data_root=cfg.data_root,
        csv_name=cfg.test_csv_name,
        split="test",
        seq_len=cfg.seq_len,
        gps_seq_len=cfg.gps_seq_len,
        gps_source_seq_len=cfg.gps_source_seq_len,
        gps_input_seq_len=cfg.gps_input_seq_len,
        num_pred=cfg.num_pred,
        image_size=cfg.image_size,
        num_beams=cfg.num_beams,
        target_beam_source=cfg.target_beam_source,
        portion=cfg.test_portion,
        portion_strategy=cfg.portion_strategy,
        portion_seed=cfg.portion_seed,
        gps_scaler=gps_scaler,
        gps_feature_mode=cfg.gps_feature_mode,
        gps_angle_offset_rad=cfg.gps_angle_offset_rad,
        gps_normalize=cfg.gps_normalize,
        max_samples=cfg.max_test_samples,
    )
    default_ae_checkpoint = output_dir / "camera_ae" / "checkpoints" / "best.pt"
    configured_ae_checkpoint = Path(cfg.ae_checkpoint_path) if cfg.ae_checkpoint_path else None
    requested_ae_checkpoint = configured_ae_checkpoint or default_ae_checkpoint
    if configured_ae_checkpoint is not None and configured_ae_checkpoint.exists():
        ae_checkpoint = configured_ae_checkpoint
    else:
        ae_checkpoint = default_ae_checkpoint
    ae_report: dict[str, Any] | None = None
    if not ae_checkpoint.exists():
        if not cfg.auto_train_ae:
            raise FileNotFoundError(
                f"Camera AE checkpoint is missing: {requested_ae_checkpoint}. "
                "Enable beambench_paper.auto_train_ae or pass --ae-checkpoint."
            )
        ae_report = train_camera_ae_for_image_gps_baseline(cfg, train_dataset, output_dir=output_dir, device=device)
        ae_checkpoint = Path(str(ae_report["checkpoint_path"]))

    model = BeamBenchImageAEGPSDirectModel(
        num_beams=cfg.num_beams,
        gps_input_size=cfg.gps_input_size,
        ae_latent_dim=cfg.ae_latent_dim,
        image_channels=cfg.image_channels,
        image_size=cfg.image_size,
        hidden_dim=cfg.fusion_hidden_dim,
        dropout=cfg.fusion_dropout,
        fusion_architecture=cfg.fusion_architecture,
        fusion_dense_hidden_sizes=cfg.fusion_dense_hidden_sizes,
        fusion_activation=cfg.fusion_activation,
        fusion_last_activation=cfg.fusion_last_activation,
        ae_checkpoint_path=ae_checkpoint,
        freeze_ae_encoder=cfg.freeze_ae_encoder,
    ).to(device)
    feature_cache_reports: dict[str, Any] = {}
    train_source: Dataset = train_dataset
    test_source: Dataset = test_dataset
    if cfg.freeze_ae_encoder and cfg.cache_frozen_ae_features:
        train_source, feature_cache_reports["train"] = _load_or_build_ae_feature_dataset(
            model,
            train_dataset,
            cfg,
            output_dir=output_dir,
            split="train",
            device=device,
            ae_checkpoint=ae_checkpoint,
        )
        test_source, feature_cache_reports["test"] = _load_or_build_ae_feature_dataset(
            model,
            test_dataset,
            cfg,
            output_dir=output_dir,
            split="test",
            device=device,
            ae_checkpoint=ae_checkpoint,
        )
    fit_source, selection_source, selection_metadata = _resolve_classifier_selection_sources(
        train_source,
        test_source,
        cfg,
    )
    train_loader = _build_loader(
        fit_source,
        batch_size=cfg.fusion_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        cfg=cfg,
    )
    selection_loader = _build_loader(
        selection_source,
        batch_size=cfg.fusion_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        cfg=cfg,
    )
    test_loader = _build_loader(
        test_source,
        batch_size=cfg.fusion_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        cfg=cfg,
    )
    optimizer = _build_adamw(
        (param for param in model.parameters() if param.requires_grad),
        lr=float(cfg.fusion_lr),
        weight_decay=float(cfg.fusion_weight_decay),
        device=device,
        fused=cfg.fused_optimizer,
    )
    best_path = output_dir / "checkpoints" / "best_image_ae_gps_direct.pt"
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    stale = 0
    for epoch in range(int(cfg.fusion_epochs)):
        train_loss = _train_classifier_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
            non_blocking=cfg.non_blocking_transfer,
        )
        selection_result = evaluate_image_ae_gps_model(
            model,
            selection_loader,
            cfg,
            device=device,
            predictions_path=None,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        score = float(selection_result["metrics"].get("official_top3_dba", 0.0))
        improved = score > best_score
        if improved:
            best_score = score
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "model_metadata": model.metadata(),
            "gps_scaler": _gps_scaler_metadata(gps_scaler),
                    "ae_checkpoint_path": str(ae_checkpoint),
                    "epoch": int(epoch),
                    "best_official_top3_dba": float(best_score),
                    "target_table_iii_row": TARGET_TABLE_III_ROW,
                    "performance": _performance_metadata(cfg, device, amp_enabled, runtime_report, feature_cache_reports),
            "selection": selection_metadata,
            "gps_calibration": _gps_calibration_metadata(cfg),
        },
                best_path,
            )
        else:
            stale += 1
        row = {
            "epoch": int(epoch + 1),
            "train_loss": float(train_loss),
            "official_top3_dba": score,
            "selection_split": str(selection_metadata["mode"]),
            "selection_official_top3_dba": score,
            "selection_circular_top3_dba": float(selection_result["metrics"].get("circular_top3_dba", 0.0)),
            "selection_official_top1_acc": float(selection_result["metrics"].get("official_top1_acc", 0.0)),
            "circular_top3_dba": float(selection_result["metrics"].get("circular_top3_dba", 0.0)),
            "official_top1_acc": float(selection_result["metrics"].get("official_top1_acc", 0.0)),
            "best_official_top3_dba": float(best_score),
        }
        history.append(row)
        _write_csv_rows(output_dir / "history.csv", history)
        if stale >= int(cfg.fusion_patience):
            break

    checkpoint = _torch_load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_predictions_path = output_dir / "predictions.csv" if cfg.save_predictions else None
    final_result = evaluate_image_ae_gps_model(
        model,
        test_loader,
        cfg,
        device=device,
        predictions_path=final_predictions_path,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    report = {
        "workflow": "beambench_image_ae_gps_direct_train",
        "paper_target": "Arnold22 BeamBench Table III Camera=AE GPS=Direct Fusion=Yes",
        "target_table_iii_row": TARGET_TABLE_III_ROW,
        "status": "local_training_complete",
        "output_dir": str(output_dir),
        "checkpoint_path": str(best_path),
        "ae_checkpoint_path": str(ae_checkpoint),
        "ae_report": ae_report,
        "config": asdict(cfg),
        "device": str(device),
        "gps_calibration": _gps_calibration_metadata(cfg),
        "performance": _performance_metadata(cfg, device, amp_enabled, runtime_report, feature_cache_reports),
        "selection": selection_metadata,
        "train_dataset": train_dataset.metadata(),
        "test_dataset": test_dataset.metadata(),
        "metrics": final_result["metrics"],
        "predictions_path": str(final_predictions_path) if final_predictions_path is not None else None,
        "history_path": str(output_dir / "history.csv"),
        "official_comparability_note": (
            "本地训练实现贴合论文 Table III 的 Camera AE + GPS Direct fusion 结构；"
            "若未使用官方权重和官方完整训练搜索流程，则不能声称数值等同论文 DBA。"
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(_json_ready(final_result["metrics"]), indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "run_report.json").write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True), encoding="utf-8")
    return report

def _train_classifier_epoch(
    model: BeamBenchImageAEGPSDirectModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    grad_scaler: Any,
    non_blocking: bool,
) -> float:
    model.train()
    if model.freeze_ae_encoder:
        model.camera_ae.eval()
    total = 0.0
    count = 0
    for batch in loader:
        labels = batch["target"].to(device=device, dtype=torch.long, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(amp_enabled, device, amp_dtype):
            logits = _classifier_logits_from_batch(model, batch, device=device, non_blocking=non_blocking)
            loss = _fusion_classification_loss(logits, labels, model=model)
        if _scaler_enabled(grad_scaler):
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += float(loss.detach().cpu()) * int(labels.numel())
        count += int(labels.numel())
    return total / max(count, 1)

def _fusion_classification_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    model: BeamBenchImageAEGPSDirectModel,
) -> torch.Tensor:
    if getattr(model, "fusion_architecture", "") == "official_dense_model":
        targets = F.one_hot(labels, num_classes=int(model.num_beams)).to(dtype=outputs.dtype)
        if isinstance(model.fusion_head, BeamBenchDenseModel) and model.fusion_head.outputs_probabilities:
            return F.binary_cross_entropy(outputs.clamp(1e-7, 1.0 - 1e-7), targets)
        return F.binary_cross_entropy_with_logits(outputs, targets)
    return F.cross_entropy(outputs, labels)

def _resolve_classifier_selection_sources(
    train_source: Dataset,
    test_source: Dataset,
    cfg: ImageAEGPSDirectTrainingConfig,
) -> tuple[Dataset, Dataset, dict[str, Any]]:
    mode = _normalize_selection_split(cfg.selection_split)
    if mode == "test_as_validation":
        return train_source, test_source, {
            "mode": mode,
            "train_count": len(train_source),
            "selection_count": len(test_source),
            "test_count": len(test_source),
            "fusion_val_fraction": float(cfg.fusion_val_fraction),
            "comparability_note": (
                "Best checkpoint is selected on the local test CSV. "
                "This maximizes local reproduction metrics but is not equivalent to official unseen test evaluation."
            ),
        }
    val_fraction = float(cfg.fusion_val_fraction or 0.1)
    fit_source, validation_source = _split_dataset(train_source, val_fraction=val_fraction, seed=cfg.seed)
    return fit_source, validation_source, {
        "mode": mode,
        "train_count": len(fit_source),
        "selection_count": len(validation_source),
        "test_count": len(test_source),
        "fusion_val_fraction": val_fraction,
        "comparability_note": (
            "Best checkpoint is selected on a validation split carved from the local train CSV; "
            "the final metrics are computed on the local test CSV."
        ),
    }


__all__ = ["run_image_ae_gps_training"]
