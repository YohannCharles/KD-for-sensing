import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from kd_sensing.baselines.beambench.image_ae_gps_config import (
    ImageAEGPSDirectTrainingConfig,
    TARGET_TABLE_III_ROW,
    _autocast_context,
    _build_adamw,
    _gps_feature_version,
    _make_grad_scaler,
    _resolve_amp_dtype,
    _scaler_enabled,
    _torch_load,
)
from kd_sensing.baselines.beambench.image_ae_gps_datasets import (
    BeamBenchImageAEGPSDataset,
    BeamBenchImageAEGPSFeatureDataset,
    BeamBenchImageOnlyDataset,
    _build_loader,
    _metadata_rows,
    _split_dataset,
)
from kd_sensing.baselines.beambench.image_ae_gps_models import BeamBenchImageAEGPSDirectModel
from kd_sensing.baselines.beambench.image_ae_gps_reports import _json_ready
from kd_sensing.baselines.beambench.image_ae_gps_reports import _performance_metadata, _write_csv_rows
from kd_sensing.models.camera_autoencoder import CameraAutoEncoder


def train_camera_ae_for_image_gps_baseline(
    cfg: ImageAEGPSDirectTrainingConfig,
    dataset: Dataset,
    *,
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    ae_dir = Path(output_dir) / "camera_ae"
    checkpoint_dir = ae_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    model = CameraAutoEncoder(
        latent_dim=cfg.ae_latent_dim,
        image_channels=cfg.image_channels,
        image_size=cfg.image_size,
    ).to(device)
    ae_dataset = BeamBenchImageOnlyDataset(dataset)
    train_subset, val_subset = _split_dataset(ae_dataset, val_fraction=cfg.ae_val_fraction, seed=cfg.seed)
    train_loader = _build_loader(train_subset, batch_size=cfg.ae_batch_size, shuffle=True, num_workers=cfg.num_workers, cfg=cfg)
    val_loader = _build_loader(val_subset, batch_size=cfg.ae_batch_size, shuffle=False, num_workers=cfg.num_workers, cfg=cfg)
    optimizer = _build_adamw(
        model.parameters(),
        lr=float(cfg.ae_lr),
        weight_decay=float(cfg.ae_weight_decay),
        device=device,
        fused=cfg.fused_optimizer,
    )
    amp_enabled = bool(cfg.amp) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(cfg.amp_dtype)
    grad_scaler = _make_grad_scaler(cfg, amp_enabled)
    best_loss = float("inf")
    stale = 0
    rows: list[dict[str, Any]] = []
    for epoch in range(int(cfg.ae_epochs)):
        train_loss = _run_ae_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
            non_blocking=cfg.non_blocking_transfer,
        )
        val_loss = _run_ae_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            grad_scaler=None,
            non_blocking=cfg.non_blocking_transfer,
        )
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_metadata": model.metadata(),
                    "epoch": int(epoch),
                    "best_val_loss": float(best_loss),
                    "target_table_iii_row": TARGET_TABLE_III_ROW,
                    "performance": _performance_metadata(cfg, device, amp_enabled, {}, {}),
                },
                best_path,
            )
        else:
            stale += 1
        rows.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "best_val_loss": float(best_loss),
            }
        )
        _write_csv_rows(ae_dir / "history.csv", rows)
        if stale >= int(cfg.ae_patience):
            break
    if not best_path.exists():
        raise RuntimeError("Camera AE training did not produce a checkpoint.")
    report = {
        "workflow": "beambench_image_ae_pretrain",
        "checkpoint_path": str(best_path),
        "history_path": str(ae_dir / "history.csv"),
        "train_count": len(train_subset),
        "val_count": len(val_subset),
        "best_val_loss": float(best_loss),
    }
    (ae_dir / "training_metadata.json").write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True), encoding="utf-8")
    return report

def resolve_camera_ae_checkpoint_for_image_gps_baseline(
    cfg: ImageAEGPSDirectTrainingConfig,
    train_dataset: Dataset,
    *,
    output_dir: str | Path,
    device: torch.device,
) -> tuple[Path, dict[str, Any] | None]:
    default_checkpoint = Path(output_dir) / "camera_ae" / "checkpoints" / "best.pt"
    configured_checkpoint = Path(cfg.ae_checkpoint_path) if cfg.ae_checkpoint_path else None
    requested_checkpoint = configured_checkpoint or default_checkpoint
    if configured_checkpoint is not None and configured_checkpoint.exists():
        return configured_checkpoint, None
    if default_checkpoint.exists():
        return default_checkpoint, None
    if not cfg.auto_train_ae:
        raise FileNotFoundError(
            f"Camera AE checkpoint is missing: {requested_checkpoint}. "
            "Enable beambench_paper.auto_train_ae or pass --ae-checkpoint."
        )
    ae_report = train_camera_ae_for_image_gps_baseline(
        cfg,
        train_dataset,
        output_dir=output_dir,
        device=device,
    )
    return Path(str(ae_report["checkpoint_path"])), ae_report

def prepare_ae_feature_sources_for_image_gps_baseline(
    model: BeamBenchImageAEGPSDirectModel,
    *,
    train_sources: Sequence[tuple[ImageAEGPSDirectTrainingConfig, BeamBenchImageAEGPSDataset]],
    eval_sources: Sequence[tuple[ImageAEGPSDirectTrainingConfig, BeamBenchImageAEGPSDataset]],
    cfg: ImageAEGPSDirectTrainingConfig,
    output_root: str | Path,
    device: torch.device,
    ae_checkpoint: str | Path,
) -> tuple[list[Dataset], dict[int, Dataset], dict[str, Any]]:
    if not (cfg.freeze_ae_encoder and cfg.cache_frozen_ae_features):
        return (
            [dataset for _, dataset in train_sources],
            {int(scene_cfg.scene): dataset for scene_cfg, dataset in eval_sources},
            {},
        )

    root = Path(output_root)
    feature_cache_reports: dict[str, Any] = {}
    train_feature_sources: list[Dataset] = []
    eval_feature_sources: dict[int, Dataset] = {}
    for scene_cfg, dataset in train_sources:
        source, report = _load_or_build_ae_feature_dataset(
            model,
            dataset,
            scene_cfg,
            output_dir=root / "feature_cache_sources" / f"train_scene{scene_cfg.scene}",
            split="train",
            device=device,
            ae_checkpoint=ae_checkpoint,
        )
        train_feature_sources.append(source)
        feature_cache_reports[f"train_scene{scene_cfg.scene}"] = report
    for scene_cfg, dataset in eval_sources:
        source, report = _load_or_build_ae_feature_dataset(
            model,
            dataset,
            scene_cfg,
            output_dir=root / "feature_cache_sources" / f"test_scene{scene_cfg.scene}",
            split="test",
            device=device,
            ae_checkpoint=ae_checkpoint,
        )
        eval_feature_sources[int(scene_cfg.scene)] = source
        feature_cache_reports[f"test_scene{scene_cfg.scene}"] = report
    return train_feature_sources, eval_feature_sources, feature_cache_reports

def _run_ae_epoch(
    model: CameraAutoEncoder,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    grad_scaler: Any,
    non_blocking: bool,
) -> float:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    count = 0
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        frames = image.reshape(-1, int(image.shape[-3]), int(image.shape[-2]), int(image.shape[-1]))
        with torch.set_grad_enabled(train):
            with _autocast_context(amp_enabled, device, amp_dtype):
                output = model(frames)
                loss = F.mse_loss(output["reconstruction"], frames)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if _scaler_enabled(grad_scaler):
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total += float(loss.detach().cpu()) * int(frames.shape[0])
        count += int(frames.shape[0])
    return total / max(count, 1)

def _load_or_build_ae_feature_dataset(
    model: BeamBenchImageAEGPSDirectModel,
    dataset: BeamBenchImageAEGPSDataset,
    cfg: ImageAEGPSDirectTrainingConfig,
    *,
    output_dir: str | Path,
    split: str,
    device: torch.device,
    ae_checkpoint: str | Path,
) -> tuple[BeamBenchImageAEGPSFeatureDataset, dict[str, Any]]:
    cache_path = _feature_cache_path(cfg, output_dir=output_dir, split=split)
    expected_signature = _feature_cache_signature(cfg, dataset, split=split, ae_checkpoint=ae_checkpoint)
    if cache_path.exists():
        payload = _torch_load(cache_path, map_location="cpu")
        cached_signature = payload.get("cache_signature")
        if cached_signature == expected_signature:
            feature_dataset = BeamBenchImageAEGPSFeatureDataset(
                image_latent=payload["image_latent"],
                gps=payload["gps"],
                target=payload["target"],
                metadata=payload.get("metadata", []),
                split=split,
            )
            return feature_dataset, {
                "enabled": True,
                "split": split,
                "path": str(cache_path),
                "mode": "read_existing",
                "sample_count": len(feature_dataset),
                "latent_dim": int(feature_dataset.image_latent.shape[-1]),
                "signature": expected_signature,
            }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    loader = _build_loader(
        dataset,
        batch_size=cfg.feature_cache_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        cfg=cfg,
    )
    latents: list[torch.Tensor] = []
    gps_values: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    metadata_rows: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=cfg.non_blocking_transfer)
            latent = _encode_last_camera_ae_latent(model, image)
            latents.append(latent.detach().cpu().to(dtype=torch.float32))
            gps_values.append(batch["gps"].detach().cpu().to(dtype=torch.float32))
            targets.append(batch["target"].detach().cpu().to(dtype=torch.long))
            metadata_rows.extend(_metadata_rows(batch.get("metadata"), count=int(batch["target"].numel())))
    model.train(was_training)
    payload = {
        "image_latent": torch.cat(latents, dim=0).contiguous(),
        "gps": torch.cat(gps_values, dim=0).contiguous(),
        "target": torch.cat(targets, dim=0).contiguous(),
        "metadata": metadata_rows,
        "cache_metadata": {
            "split": split,
            "sample_count": int(sum(int(item.shape[0]) for item in targets)),
            "latent_dim": int(model.ae_latent_dim),
            "source_csv": str(dataset.csv_path),
            "target_table_iii_row": TARGET_TABLE_III_ROW,
        },
        "cache_signature": expected_signature,
    }
    torch.save(payload, cache_path)
    feature_dataset = BeamBenchImageAEGPSFeatureDataset(
        image_latent=payload["image_latent"],
        gps=payload["gps"],
        target=payload["target"],
        metadata=metadata_rows,
        split=split,
    )
    return feature_dataset, {
        "enabled": True,
        "split": split,
        "path": str(cache_path),
        "mode": "built",
        "sample_count": len(feature_dataset),
        "latent_dim": int(feature_dataset.image_latent.shape[-1]),
        "signature": expected_signature,
    }

def _encode_last_camera_ae_latent(model: BeamBenchImageAEGPSDirectModel, image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 5:
        raise ValueError(f"image must have shape [B, T, C, H, W], got {tuple(image.shape)}.")
    batch_size, seq_len, channels, height, width = image.shape
    frames = image.reshape(batch_size * seq_len, channels, height, width).to(dtype=torch.float32)
    if (int(height), int(width)) != (model.image_size, model.image_size):
        frames = F.interpolate(frames, size=(model.image_size, model.image_size), mode="bilinear", align_corners=False)
    latent = model.camera_ae.encode(frames)
    return latent.view(batch_size, seq_len, model.ae_latent_dim)[:, -1, :]

def _feature_cache_path(cfg: ImageAEGPSDirectTrainingConfig, *, output_dir: str | Path, split: str) -> Path:
    cache_dir = Path(cfg.feature_cache_dir) if cfg.feature_cache_dir else Path(output_dir) / "feature_cache"
    return cache_dir / f"{split}_camera_ae_latents.pt"

def _feature_cache_signature(
    cfg: ImageAEGPSDirectTrainingConfig,
    dataset: BeamBenchImageAEGPSDataset,
    *,
    split: str,
    ae_checkpoint: str | Path,
) -> dict[str, Any]:
    checkpoint_path = Path(ae_checkpoint)
    checkpoint_stat = checkpoint_path.stat() if checkpoint_path.exists() else None
    return {
        "split": str(split),
        "csv_path": str(dataset.csv_path),
        "sample_count": len(dataset),
        "seq_len": int(cfg.seq_len),
        "gps_seq_len": None if cfg.gps_seq_len is None else int(cfg.gps_seq_len),
        "gps_source_seq_len": None if cfg.gps_source_seq_len is None else int(cfg.gps_source_seq_len),
        "gps_input_seq_len": None if cfg.gps_input_seq_len is None else int(cfg.gps_input_seq_len),
        "num_pred": int(cfg.num_pred),
        "image_size": int(cfg.image_size),
        "num_beams": int(cfg.num_beams),
        "target_beam_source": str(cfg.target_beam_source),
        "gps_feature_mode": str(cfg.gps_feature_mode),
        "gps_feature_version": _gps_feature_version(cfg.gps_feature_mode),
        "gps_angle_offset_rad": None if cfg.gps_angle_offset_rad is None else float(cfg.gps_angle_offset_rad),
        "gps_normalize": bool(cfg.gps_normalize),
        "ae_latent_dim": int(cfg.ae_latent_dim),
        "ae_checkpoint_path": str(checkpoint_path),
        "ae_checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns) if checkpoint_stat is not None else None,
        "ae_checkpoint_size": int(checkpoint_stat.st_size) if checkpoint_stat is not None else None,
    }


__all__ = [
    "prepare_ae_feature_sources_for_image_gps_baseline",
    "resolve_camera_ae_checkpoint_for_image_gps_baseline",
    "train_camera_ae_for_image_gps_baseline",
]
