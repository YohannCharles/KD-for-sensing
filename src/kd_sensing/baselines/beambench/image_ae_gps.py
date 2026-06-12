from __future__ import annotations

import csv
import datetime as dt
import json
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split

from kd_sensing.baselines.beambench.metrics import beambench_metric_summary_from_logits
from kd_sensing.data.samples import create_samples
from kd_sensing.data.scenes import resolve_deepsense_scene
from kd_sensing.data.transform_ops.gps import (
    GPS_FEATURE_DIMS,
    GPSStandardScaler,
    PAPER_CALIBRATED_GPS_MODE,
    PAPER_DISTANCE_ANGLE_FEATURE_VERSION,
    PAPER_SCENE_CENTER_ANGLES_RAD,
    load_gps_feature_sequence,
)
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.models.camera_autoencoder import CameraAutoEncoder
from kd_sensing.utils.paths import resolve_path


TARGET_TABLE_III_ROW = {
    "camera": "AE",
    "radar": "none",
    "lidar": "none",
    "gps": "Direct",
    "fusion": True,
    "scene31": 0.6731,
    "scene32": 0.6173,
    "scene33": 0.8171,
    "scene34": 0.7313,
    "overall": 0.7127,
}

@dataclass(frozen=True)
class ImageAEGPSDirectTrainingConfig:
    data_root: str
    train_csv_name: str = "train_seqs_RA_GPS_LIDAR.csv"
    test_csv_name: str = "test_seqs_RA_GPS_LIDAR.csv"
    output_dir: str = "outputs/scene31/beambench_image_ae_gps_direct"
    scene: int = 31
    seq_len: int = 1
    num_pred: int = 1
    num_beams: int = 64
    target_beam_source: str = "current"
    image_size: int = 64
    image_channels: int = 3
    gps_input_size: int = 2
    gps_feature_mode: str = PAPER_CALIBRATED_GPS_MODE
    gps_angle_offset_rad: float | None = None
    gps_angle_offset_source: str = "paper_scene_default"
    gps_normalize: bool = True
    train_portion: float = 1.0
    test_portion: float = 1.0
    portion_strategy: str = "even"
    portion_seed: int = 42
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    ae_checkpoint_path: str | None = None
    auto_train_ae: bool = True
    ae_epochs: int = 20
    ae_batch_size: int = 64
    ae_lr: float = 1e-3
    ae_weight_decay: float = 1e-4
    ae_val_fraction: float = 0.1
    ae_patience: int = 5
    ae_latent_dim: int = 128
    fusion_epochs: int = 80
    fusion_batch_size: int = 64
    fusion_lr: float = 5e-4
    fusion_weight_decay: float = 1e-4
    fusion_patience: int = 15
    fusion_val_fraction: float = 0.0
    selection_split: str = "test_as_validation"
    fusion_hidden_dim: int = 256
    fusion_dropout: float = 0.2
    freeze_ae_encoder: bool = True
    dba_delta: float = 5.0
    topk: tuple[int, ...] = (1, 3, 5)
    seed: int = 42
    device: str = "auto"
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = 2
    non_blocking_transfer: bool = True
    amp: bool = True
    amp_dtype: str = "float16"
    amp_grad_scaler: bool = True
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    fused_optimizer: bool = True
    cache_frozen_ae_features: bool = True
    feature_cache_batch_size: int = 256
    feature_cache_dir: str | None = None
    save_predictions: bool = True
    dry_run: bool = False


class BeamBenchImageAEGPSDataset(Dataset):
    """DeepSense6G image/GPS/beam dataset for the BeamBench Table III row."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        csv_name: str,
        split: str,
        seq_len: int = 1,
        num_pred: int = 1,
        image_size: int = 64,
        num_beams: int = 64,
        target_beam_source: str = "current",
        portion: float = 1.0,
        portion_strategy: str = "even",
        portion_seed: int = 42,
        gps_scaler: GPSStandardScaler | None = None,
        gps_feature_mode: str = PAPER_CALIBRATED_GPS_MODE,
        gps_angle_offset_rad: float | None = None,
        gps_normalize: bool = True,
        max_samples: int | None = None,
        return_metadata: bool = True,
    ) -> None:
        self.data_root = resolve_path(data_root)
        self.csv_path = Path(csv_name)
        if not self.csv_path.is_absolute():
            self.csv_path = self.data_root / self.csv_path
        self.split = str(split)
        self.seq_len = int(seq_len)
        self.num_pred = int(num_pred)
        self.image_size = int(image_size)
        self.num_beams = int(num_beams)
        self.target_beam_source = _normalize_target_beam_source(target_beam_source)
        self.gps_scaler = gps_scaler
        self.gps_feature_mode = str(gps_feature_mode or "relative_polar")
        self.gps_angle_offset_rad = None if gps_angle_offset_rad is None else float(gps_angle_offset_rad)
        self.gps_normalize = bool(gps_normalize)
        self.return_metadata = bool(return_metadata)
        self.samples = create_samples(
            self.csv_path,
            portion=portion,
            enabled_modalities=("image", "gps"),
            seq_len=self.seq_len,
            num_pred=self.num_pred,
            portion_strategy=portion_strategy,
            portion_seed=portion_seed,
        )
        length = len(self.samples.future_beam_paths)
        if max_samples is not None:
            length = min(length, max(0, int(max_samples)))
        self.indices = list(range(length))
        if not self.indices:
            raise ValueError(f"No samples selected from {self.csv_path}.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw_index = self.indices[int(index)]
        image_paths = self.samples.rgb_paths[raw_index][-self.seq_len :]
        gps_paths = self.samples.gps_paths[raw_index] if self.samples.gps_paths is not None else None
        bs_gps_paths = self.samples.bs_gps_paths[raw_index] if self.samples.bs_gps_paths is not None else None
        if gps_paths is None or bs_gps_paths is None:
            raise ValueError(f"GPS/BS GPS paths are unavailable for sample {raw_index} in {self.csv_path}.")
        image = torch.stack([_load_ae_image(self.data_root, rel_path, self.image_size) for rel_path in image_paths], dim=0)
        gps = self.gps_features_for_index(raw_index)
        target = self.target_for_index(raw_index)
        item: dict[str, Any] = {
            "image": image,
            "gps": torch.tensor(gps, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.long),
        }
        if self.return_metadata:
            target_paths = self._target_beam_paths(raw_index)
            item["metadata"] = {
                "dataset_index": int(raw_index),
                "split": self.split,
                "csv_path": str(self.csv_path),
                "image_path": str(image_paths[-1]) if image_paths else "",
                "gps_path": str(gps_paths[-1]) if gps_paths else "",
                "target_beam_source": self.target_beam_source,
                "target_beam_path": str(target_paths[0] if self.target_beam_source == "future" else target_paths[-1]),
            }
        return item

    def gps_features_for_index(self, raw_index: int) -> np.ndarray:
        if self.samples.gps_paths is None or self.samples.bs_gps_paths is None:
            raise ValueError("GPS paths are unavailable.")
        features = load_gps_feature_sequence(
            self.data_root,
            self.samples.gps_paths[raw_index],
            self.samples.bs_gps_paths[raw_index],
            seq_len=self.seq_len,
            mode=self.gps_feature_mode,
            angle_offset_rad=self.gps_angle_offset_rad,
        ).astype(np.float32, copy=False)
        if self.gps_scaler is not None and self.gps_normalize:
            features = self.gps_scaler.transform(features)
        return features

    def target_for_index(self, raw_index: int) -> int:
        beam_paths = self._target_beam_paths(raw_index)
        if not beam_paths:
            return -100
        if self.target_beam_source == "future":
            rel_path = beam_paths[0]
        else:
            rel_path = beam_paths[-1]
        path = joined_resource(self.data_root, str(rel_path))
        values = np.loadtxt(path)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size <= 0:
            raise ValueError(f"Beam label file is empty: {path}")
        label = int(np.argmax(values))
        if label < 0 or label >= self.num_beams:
            raise ValueError(f"Beam label {label} from {path} is outside [0, {self.num_beams}).")
        return label

    def _target_beam_paths(self, raw_index: int) -> list[str]:
        if self.target_beam_source == "future":
            return self.samples.future_beam_paths[raw_index]
        return self.samples.input_beam_paths[raw_index][-self.seq_len :]

    def raw_gps_matrix(self) -> np.ndarray:
        values = []
        for raw_index in self.indices:
            if self.samples.gps_paths is None or self.samples.bs_gps_paths is None:
                raise ValueError("GPS paths are unavailable.")
            values.append(
                load_gps_feature_sequence(
                    self.data_root,
                    self.samples.gps_paths[raw_index],
                    self.samples.bs_gps_paths[raw_index],
                    seq_len=self.seq_len,
                    mode=self.gps_feature_mode,
                    angle_offset_rad=self.gps_angle_offset_rad,
                )
            )
        return np.concatenate(values, axis=0).astype(np.float32, copy=False)

    def metadata(self) -> dict[str, Any]:
        return {
            "data_root": str(self.data_root),
            "csv_path": str(self.csv_path),
            "split": self.split,
            "sample_count": len(self),
            "seq_len": self.seq_len,
            "num_pred": self.num_pred,
            "image_size": self.image_size,
            "num_beams": self.num_beams,
            "target_beam_source": self.target_beam_source,
            "gps_feature_mode": self.gps_feature_mode,
            "gps_angle_offset_rad": self.gps_angle_offset_rad,
            "create_samples": dict(self.samples.metadata or {}),
        }


class BeamBenchImageOnlyDataset(Dataset):
    """Image-only view used for Camera AE pretraining."""

    def __init__(self, source: Dataset) -> None:
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(self.source, BeamBenchImageAEGPSDataset):
            raw_index = self.source.indices[int(index)]
            image_paths = self.source.samples.rgb_paths[raw_index][-self.source.seq_len :]
            image = torch.stack(
                [_load_ae_image(self.source.data_root, rel_path, self.source.image_size) for rel_path in image_paths],
                dim=0,
            )
            return {
                "image": image,
                "metadata": {
                    "dataset_index": int(raw_index),
                    "split": self.source.split,
                    "csv_path": str(self.source.csv_path),
                    "image_path": str(image_paths[-1]) if image_paths else "",
                },
            }
        item = self.source[int(index)]
        return {"image": item["image"], "metadata": item.get("metadata", {})}


class BeamBenchImageAEGPSFeatureDataset(Dataset):
    """Precomputed Camera AE latent + GPS/label dataset for fast fusion training."""

    def __init__(
        self,
        *,
        image_latent: torch.Tensor,
        gps: torch.Tensor,
        target: torch.Tensor,
        metadata: Sequence[Mapping[str, Any]] | None = None,
        split: str = "",
    ) -> None:
        if image_latent.ndim != 2:
            raise ValueError(f"image_latent must have shape [N, D], got {tuple(image_latent.shape)}.")
        if gps.ndim != 3:
            raise ValueError(f"gps must have shape [N, T, D], got {tuple(gps.shape)}.")
        if target.ndim != 1:
            raise ValueError(f"target must have shape [N], got {tuple(target.shape)}.")
        length = int(image_latent.shape[0])
        if int(gps.shape[0]) != length or int(target.shape[0]) != length:
            raise ValueError("image_latent, gps, and target must have the same first dimension.")
        self.image_latent = image_latent.detach().cpu().to(dtype=torch.float32).contiguous()
        self.gps = gps.detach().cpu().to(dtype=torch.float32).contiguous()
        self.target = target.detach().cpu().to(dtype=torch.long).contiguous()
        rows = list(metadata or [{} for _ in range(length)])
        if len(rows) != length:
            raise ValueError("metadata length must match cached feature count.")
        self.rows = [dict(row) for row in rows]
        self.split = str(split)

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        idx = int(index)
        return {
            "image_latent": self.image_latent[idx],
            "gps": self.gps[idx],
            "target": self.target[idx],
            "metadata": dict(self.rows[idx]),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "sample_count": len(self),
            "latent_dim": int(self.image_latent.shape[-1]),
            "gps_shape": list(self.gps.shape),
        }


class BeamBenchImageAEGPSDirectModel(nn.Module):
    """Camera AE + GPS direct fusion classifier for Arnold22 BeamBench Table III."""

    def __init__(
        self,
        *,
        num_beams: int = 64,
        gps_input_size: int = 3,
        ae_latent_dim: int = 128,
        image_channels: int = 3,
        image_size: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        ae_checkpoint_path: str | Path | None = None,
        freeze_ae_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.num_beams = int(num_beams)
        self.gps_input_size = int(gps_input_size)
        self.ae_latent_dim = int(ae_latent_dim)
        self.image_size = int(image_size)
        self.freeze_ae_encoder = bool(freeze_ae_encoder)
        self.camera_ae = CameraAutoEncoder(
            latent_dim=self.ae_latent_dim,
            image_channels=int(image_channels),
            image_size=self.image_size,
        )
        if ae_checkpoint_path:
            payload = _torch_load(Path(ae_checkpoint_path), map_location="cpu")
            state_dict = payload.get("model_state_dict", payload)
            self.camera_ae.load_state_dict(state_dict)
        if self.freeze_ae_encoder:
            for param in self.camera_ae.parameters():
                param.requires_grad = False
        hidden = int(hidden_dim)
        self.image_projection = nn.Sequential(
            nn.LayerNorm(self.ae_latent_dim),
            nn.Linear(self.ae_latent_dim, hidden),
            nn.GELU(),
        )
        self.gps_encoder = nn.Sequential(
            nn.Linear(self.gps_input_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.num_beams),
        )

    def forward(self, image: torch.Tensor, gps: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5:
            raise ValueError(f"image must have shape [B, T, C, H, W], got {tuple(image.shape)}.")
        if gps.ndim != 3:
            raise ValueError(f"gps must have shape [B, T, D], got {tuple(gps.shape)}.")
        batch_size, seq_len, channels, height, width = image.shape
        frames = image.reshape(batch_size * seq_len, channels, height, width).to(dtype=torch.float32)
        if (int(height), int(width)) != (self.image_size, self.image_size):
            frames = F.interpolate(frames, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        with torch.set_grad_enabled(not self.freeze_ae_encoder):
            latent = self.camera_ae.encode(frames)
        latent = latent.view(batch_size, seq_len, self.ae_latent_dim)[:, -1, :]
        return self.forward_from_latent(latent, gps)

    def forward_from_latent(self, image_latent: torch.Tensor, gps: torch.Tensor) -> torch.Tensor:
        if image_latent.ndim == 3:
            image_latent = image_latent[:, -1, :]
        if image_latent.ndim != 2:
            raise ValueError(f"image_latent must have shape [B, D] or [B, T, D], got {tuple(image_latent.shape)}.")
        if gps.ndim != 3:
            raise ValueError(f"gps must have shape [B, T, D], got {tuple(gps.shape)}.")
        if int(image_latent.shape[-1]) != self.ae_latent_dim:
            raise ValueError(f"image latent dim must be {self.ae_latent_dim}, got {int(image_latent.shape[-1])}.")
        gps_last = gps.to(dtype=torch.float32)[:, -1, :]
        if int(gps_last.shape[-1]) != self.gps_input_size:
            raise ValueError(f"gps feature dim must be {self.gps_input_size}, got {int(gps_last.shape[-1])}.")
        fused = torch.cat([self.image_projection(image_latent.to(dtype=torch.float32)), self.gps_encoder(gps_last)], dim=-1)
        return self.fusion_head(fused)

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "BeamBenchImageAEGPSDirectModel",
            "paper_target": "Arnold22 BeamBench Table III Camera=AE GPS=Direct Fusion=Yes",
            "num_beams": self.num_beams,
            "gps_input_size": self.gps_input_size,
            "ae_latent_dim": self.ae_latent_dim,
            "image_size": self.image_size,
            "freeze_ae_encoder": self.freeze_ae_encoder,
        }


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

    train_cfgs = [_scene_specific_cfg(base_cfg, scene) for scene in train_scenes]
    eval_cfgs = [_scene_specific_cfg(base_cfg, scene) for scene in eval_scenes]
    train_datasets = [_build_split_dataset(cfg, split="train") for cfg in train_cfgs]
    raw_gps = np.concatenate([dataset.raw_gps_matrix() for dataset in train_datasets], axis=0)
    gps_scaler = GPSStandardScaler().fit(raw_gps) if base_cfg.gps_normalize else None
    for dataset in train_datasets:
        dataset.gps_scaler = gps_scaler
    eval_datasets = [
        _build_split_dataset(cfg, split="test", gps_scaler=gps_scaler, gps_normalize=base_cfg.gps_normalize)
        for cfg in eval_cfgs
    ]
    combined_train_dataset = ConcatDataset(train_datasets)

    ae_checkpoint = root / "camera_ae" / "checkpoints" / "best.pt"
    ae_report: dict[str, Any] | None = None
    if base_cfg.ae_checkpoint_path and Path(base_cfg.ae_checkpoint_path).exists():
        ae_checkpoint = Path(base_cfg.ae_checkpoint_path)
    elif not ae_checkpoint.exists():
        if not base_cfg.auto_train_ae:
            raise FileNotFoundError(
                f"Camera AE checkpoint is missing: {ae_checkpoint}. "
                "Enable beambench_paper.auto_train_ae or pass --ae-checkpoint."
            )
        ae_report = train_camera_ae_for_image_gps_baseline(
            base_cfg,
            combined_train_dataset,
            output_dir=root,
            device=device,
        )
        ae_checkpoint = Path(str(ae_report["checkpoint_path"]))

    model = BeamBenchImageAEGPSDirectModel(
        num_beams=base_cfg.num_beams,
        gps_input_size=base_cfg.gps_input_size,
        ae_latent_dim=base_cfg.ae_latent_dim,
        image_channels=base_cfg.image_channels,
        image_size=base_cfg.image_size,
        hidden_dim=base_cfg.fusion_hidden_dim,
        dropout=base_cfg.fusion_dropout,
        ae_checkpoint_path=ae_checkpoint,
        freeze_ae_encoder=base_cfg.freeze_ae_encoder,
    ).to(device)

    feature_cache_reports: dict[str, Any] = {}
    train_sources: list[Dataset] = []
    eval_sources: dict[int, Dataset] = {}
    if base_cfg.freeze_ae_encoder and base_cfg.cache_frozen_ae_features:
        for cfg, dataset in zip(train_cfgs, train_datasets, strict=True):
            source, report = _load_or_build_ae_feature_dataset(
                model,
                dataset,
                cfg,
                output_dir=root / "feature_cache_sources" / f"train_scene{cfg.scene}",
                split="train",
                device=device,
                ae_checkpoint=ae_checkpoint,
            )
            train_sources.append(source)
            feature_cache_reports[f"train_scene{cfg.scene}"] = report
        for cfg, dataset in zip(eval_cfgs, eval_datasets, strict=True):
            source, report = _load_or_build_ae_feature_dataset(
                model,
                dataset,
                cfg,
                output_dir=root / "feature_cache_sources" / f"test_scene{cfg.scene}",
                split="test",
                device=device,
                ae_checkpoint=ae_checkpoint,
            )
            eval_sources[int(cfg.scene)] = source
            feature_cache_reports[f"test_scene{cfg.scene}"] = report
    else:
        train_sources = list(train_datasets)
        eval_sources = {int(cfg.scene): dataset for cfg, dataset in zip(eval_cfgs, eval_datasets, strict=True)}

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
            "train_scenes": [int(scene) for scene in train_scenes],
            "eval_scenes": [int(scene) for scene in eval_scenes],
        },
        "selection": selection_metadata,
        "performance": _performance_metadata(base_cfg, device, amp_enabled, runtime_report, feature_cache_reports),
        "train_datasets": [dataset.metadata() for dataset in train_datasets],
        "eval_reports": scene_reports,
        "summary": summary,
        "history_path": str(root / "history.csv"),
        "official_comparability_note": (
            "本地训练按论文方向改为 scenes 32-34 联合训练、scenes 31-34 测试；"
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
    """Evaluate an existing paper-split checkpoint on scenes 31-34 without retraining."""

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
    eval_source_by_scene: dict[int, Dataset] = {}
    feature_cache_reports: dict[str, Any] = {}
    if ckpt_cfg.freeze_ae_encoder and ckpt_cfg.cache_frozen_ae_features:
        for cfg, dataset in zip(eval_cfgs, eval_datasets, strict=True):
            source, report = _load_or_build_ae_feature_dataset(
                model,
                dataset,
                cfg,
                output_dir=root / "feature_cache_sources" / f"test_scene{cfg.scene}",
                split="test",
                device=device,
                ae_checkpoint=ae_checkpoint,
            )
            eval_source_by_scene[int(cfg.scene)] = source
            feature_cache_reports[f"test_scene{cfg.scene}"] = report
    else:
        eval_source_by_scene = {int(cfg.scene): dataset for cfg, dataset in zip(eval_cfgs, eval_datasets, strict=True)}

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
            "本地 eval-only 使用已训练 paper-split checkpoint 评估 scenes 31-34；"
            "未使用官方预训练权重、官方完整 NNI/剪枝搜索和官方 unseen test packaging。"
        ),
    }
    _write_paper_split_summary_artifacts(report, root)
    (root / "run_report.json").write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True), encoding="utf-8")
    return report


def train_camera_ae_for_image_gps_baseline(
    cfg: ImageAEGPSDirectTrainingConfig,
    dataset: BeamBenchImageAEGPSDataset,
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


def evaluate_image_ae_gps_model(
    model: BeamBenchImageAEGPSDirectModel,
    loader: DataLoader,
    cfg: ImageAEGPSDirectTrainingConfig,
    *,
    device: torch.device,
    predictions_path: str | Path | None,
    amp_enabled: bool | None = None,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    use_amp = bool(cfg.amp) and device.type == "cuda" if amp_enabled is None else bool(amp_enabled)
    dtype = _resolve_amp_dtype(cfg.amp_dtype) if amp_dtype is None else amp_dtype
    with torch.no_grad():
        for batch in loader:
            labels = batch["target"].to(device=device, dtype=torch.long, non_blocking=cfg.non_blocking_transfer)
            with _autocast_context(use_amp, device, dtype):
                logits = _classifier_logits_from_batch(
                    model,
                    batch,
                    device=device,
                    non_blocking=cfg.non_blocking_transfer,
                )
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            if predictions_path is not None:
                probs = F.softmax(logits.detach().cpu(), dim=-1)
                topk = torch.topk(logits.detach().cpu(), k=min(5, int(logits.shape[-1])), dim=-1).indices
                metadata = _metadata_rows(batch.get("metadata"), count=int(labels.numel()))
                for idx in range(int(labels.numel())):
                    row = {
                        "row_index": len(rows),
                        "target_beam": int(labels.detach().cpu()[idx].item()),
                        "pred_top1": int(topk[idx, 0].item()),
                        "pred_top3": json.dumps([int(value) for value in topk[idx, : min(3, topk.shape[1])].tolist()]),
                        "top1_probability": float(probs[idx, int(topk[idx, 0].item())].item()),
                    }
                    row.update(metadata[idx] if idx < len(metadata) else {})
                    rows.append(row)
    logits_t = torch.cat(all_logits, dim=0)
    labels_t = torch.cat(all_labels, dim=0)
    metrics = beambench_metric_summary_from_logits(
        logits_t,
        labels_t,
        num_beams=cfg.num_beams,
        topk=cfg.topk,
        dba_delta=cfg.dba_delta,
        circular=True,
    )
    if predictions_path is not None:
        _write_csv_rows(Path(predictions_path), rows)
    return {"metrics": metrics, "sample_count": int(labels_t.numel())}


def resolve_image_ae_gps_config(raw: Mapping[str, Any]) -> ImageAEGPSDirectTrainingConfig:
    experiment = _mapping(raw.get("experiment"))
    data = _mapping(raw.get("data"))
    dataset = _mapping(data.get("dataset"))
    loader = _mapping(data.get("dataloader"))
    model = _mapping(raw.get("model"))
    primary = _mapping(model.get("primary"))
    encoders = _mapping(primary.get("encoders"))
    image_encoder = _mapping(encoders.get("image"))
    gps_encoder = _mapping(encoders.get("gps"))
    training = _mapping(raw.get("training"))
    output = _mapping(raw.get("output"))
    paper = _mapping(raw.get("beambench_paper"))

    scene_value = dataset.get("scene", 31)
    scene = resolve_deepsense_scene(scene_value)
    data_root = str(dataset.get("data_root") or scene.default_data_root)
    output_dir = paper.get("output_dir")
    if not output_dir:
        output_dir = Path(str(output.get("dir", "outputs"))) / scene.scene_slug / "beambench_image_ae_gps_direct"
        run_name = str(output.get("run_name") or "").strip()
        if run_name:
            output_dir = Path(str(output.get("dir", "outputs"))) / scene.scene_slug / run_name

    dry_run = bool(paper.get("dry_run", False))
    max_train_samples = _optional_int(paper.get("max_train_samples"))
    max_test_samples = _optional_int(paper.get("max_test_samples"))
    if dry_run:
        max_train_samples = max_train_samples or 4
        max_test_samples = max_test_samples or 4

    topk_raw = paper.get("topk", raw.get("metrics", {}).get("topk") if isinstance(raw.get("metrics"), Mapping) else None)
    topk = tuple(int(item) for item in (topk_raw or (1, 3, 5)))
    ae_epochs = int(paper.get("ae_epochs", 20))
    ae_patience = int(paper.get("ae_patience", 5))
    fusion_epochs = int(paper.get("fusion_epochs", training.get("epochs", 80)))
    fusion_patience = int(paper.get("fusion_patience", training.get("patience", 15)))
    num_workers = int(loader.get("num_workers", 0))
    prefetch_factor = _optional_int(loader.get("prefetch_factor", 2))
    pin_memory = _bool(loader.get("pin_memory", True), default=True)
    persistent_workers = _bool(loader.get("persistent_workers", True), default=True)
    transfer_cfg = _mapping(training.get("transfer"))
    amp_cfg = _mapping(training.get("amp"))
    if dry_run:
        ae_epochs = 1
        ae_patience = 1
        fusion_epochs = 1
        fusion_patience = 1
        num_workers = 0
        prefetch_factor = None
        persistent_workers = False
    gps_feature_mode = _normalize_gps_feature_mode(
        str(paper.get("gps_feature_mode", dataset.get("gps_feature_mode", PAPER_CALIBRATED_GPS_MODE)))
    )
    gps_angle_offset_rad, gps_angle_offset_source = _resolve_gps_angle_offset(
        scene=int(scene.scene_id),
        feature_mode=gps_feature_mode,
        explicit_value=paper.get("gps_angle_offset_rad"),
    )

    return ImageAEGPSDirectTrainingConfig(
        data_root=data_root,
        train_csv_name=str(dataset.get("train_csv_name", "train_seqs_RA_GPS_LIDAR.csv")),
        test_csv_name=str(dataset.get("test_csv_name", "test_seqs_RA_GPS_LIDAR.csv")),
        output_dir=str(output_dir),
        scene=int(scene.scene_id),
        seq_len=int(dataset.get("seq_len", 1)),
        num_pred=int(dataset.get("num_pred", 1)),
        num_beams=int(model.get("num_classes", primary.get("num_classes", 64))),
        target_beam_source=_normalize_target_beam_source(str(paper.get("target_beam_source", "current"))),
        image_size=int(paper.get("ae_image_size", image_encoder.get("image_size", 64))),
        image_channels=int(primary.get("image_channels", 3)),
        gps_input_size=_resolve_gps_input_size(
            paper=paper,
            primary=primary,
            gps_encoder=gps_encoder,
            gps_feature_mode=gps_feature_mode,
        ),
        gps_feature_mode=gps_feature_mode,
        gps_angle_offset_rad=gps_angle_offset_rad,
        gps_angle_offset_source=gps_angle_offset_source,
        gps_normalize=bool(dataset.get("gps_normalize", True)),
        train_portion=float(paper.get("train_portion", dataset.get("portion", 1.0))),
        test_portion=float(paper.get("test_portion", dataset.get("portion", 1.0))),
        portion_strategy=str(dataset.get("portion_strategy", "even")),
        portion_seed=int(dataset.get("portion_seed", experiment.get("seed", 42))),
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        ae_checkpoint_path=_optional_str(paper.get("ae_checkpoint_path", image_encoder.get("checkpoint_path"))),
        auto_train_ae=bool(paper.get("auto_train_ae", True)),
        ae_epochs=ae_epochs,
        ae_batch_size=int(paper.get("ae_batch_size", loader.get("train_batch_size", 64))),
        ae_lr=float(paper.get("ae_lr", 1e-3)),
        ae_weight_decay=float(paper.get("ae_weight_decay", training.get("weight_decay", 1e-4))),
        ae_val_fraction=float(paper.get("ae_val_fraction", 0.1)),
        ae_patience=ae_patience,
        ae_latent_dim=int(image_encoder.get("latent_dim", paper.get("ae_latent_dim", 128))),
        fusion_epochs=fusion_epochs,
        fusion_batch_size=int(paper.get("fusion_batch_size", loader.get("train_batch_size", 64))),
        fusion_lr=float(paper.get("fusion_lr", training.get("lr", 5e-4))),
        fusion_weight_decay=float(paper.get("fusion_weight_decay", training.get("weight_decay", 1e-4))),
        fusion_patience=fusion_patience,
        fusion_val_fraction=float(paper.get("fusion_val_fraction", 0.0)),
        selection_split=_normalize_selection_split(str(paper.get("selection_split", "test_as_validation"))),
        fusion_hidden_dim=int(paper.get("fusion_hidden_dim", primary.get("d_model", model.get("d_model", 256)))),
        fusion_dropout=float(paper.get("fusion_dropout", training.get("dropout", 0.2))),
        freeze_ae_encoder=bool(image_encoder.get("freeze_encoder", paper.get("freeze_ae_encoder", True))),
        dba_delta=float(paper.get("dba_delta", raw.get("evaluation", {}).get("dba_delta", 5.0) if isinstance(raw.get("evaluation"), Mapping) else 5.0)),
        topk=topk,
        seed=int(experiment.get("seed", 42)),
        device=str(experiment.get("device", paper.get("device", "auto"))),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        non_blocking_transfer=_bool(
            paper.get("non_blocking_transfer", transfer_cfg.get("non_blocking", True)),
            default=True,
        ),
        amp=_bool(paper.get("amp", amp_cfg.get("enabled", True)), default=True),
        amp_dtype=str(paper.get("amp_dtype", amp_cfg.get("dtype", "float16"))),
        amp_grad_scaler=_bool(paper.get("amp_grad_scaler", amp_cfg.get("grad_scaler", True)), default=True),
        allow_tf32=_bool(paper.get("allow_tf32", training.get("allow_tf32", True)), default=True),
        cudnn_benchmark=_bool(paper.get("cudnn_benchmark", training.get("cudnn_benchmark", True)), default=True),
        fused_optimizer=_bool(paper.get("fused_optimizer", training.get("fused_optimizer", True)), default=True),
        cache_frozen_ae_features=_bool(paper.get("cache_frozen_ae_features", True), default=True),
        feature_cache_batch_size=int(paper.get("feature_cache_batch_size", loader.get("test_batch_size", 256))),
        feature_cache_dir=_optional_str(paper.get("feature_cache_dir")),
        save_predictions=bool(paper.get("save_predictions", True)),
        dry_run=dry_run,
    )


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
            loss = F.cross_entropy(logits, labels)
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


def _normalize_selection_split(value: str) -> str:
    normalized = str(value or "test_as_validation").strip().lower().replace("-", "_")
    if normalized in {"test", "test_as_val", "test_as_validation"}:
        return "test_as_validation"
    if normalized in {"val", "valid", "validation", "train_validation"}:
        return "validation"
    raise ValueError("beambench_paper.selection_split must be 'test_as_validation' or 'validation'.")


def _normalize_target_beam_source(value: str) -> str:
    normalized = str(value or "current").strip().lower().replace("-", "_")
    if normalized in {"current", "current_beam", "beam", "beam_last", "last_beam"}:
        return "current"
    if normalized in {"future", "future_beam", "future_beam1", "next"}:
        return "future"
    raise ValueError("beambench_paper.target_beam_source must be 'current' or 'future'.")


def _normalize_gps_feature_mode(value: str) -> str:
    normalized = str(value or PAPER_CALIBRATED_GPS_MODE).strip().lower().replace("-", "_")
    if normalized in {"relative_polar", "raw_relative_polar"}:
        return "relative_polar"
    if normalized in {
        "paper_calibrated_relative_polar",
        "calibrated_relative_polar",
        "paper_calibrated_polar",
        "paper_centered_relative_polar",
    }:
        return "paper_calibrated_relative_polar"
    if normalized in {"paper_distance_angle", "distance_angle", "paper_gt_pos", "official_gps"}:
        return "paper_distance_angle"
    raise ValueError(
        "beambench_paper.gps_feature_mode must be 'relative_polar', "
        "'paper_calibrated_relative_polar', or 'paper_distance_angle'."
    )


def _resolve_gps_input_size(
    *,
    paper: Mapping[str, Any],
    primary: Mapping[str, Any],
    gps_encoder: Mapping[str, Any],
    gps_feature_mode: str,
) -> int:
    if "gps_input_size" in paper:
        return int(paper["gps_input_size"])
    mode = _normalize_gps_feature_mode(gps_feature_mode)
    configured = primary.get("gps_input_size", gps_encoder.get("gps_input_size"))
    if mode in GPS_FEATURE_DIMS:
        return int(GPS_FEATURE_DIMS[mode])
    return int(configured or 3)


def _resolve_gps_angle_offset(
    *,
    scene: int,
    feature_mode: str,
    explicit_value: Any,
) -> tuple[float | None, str]:
    mode = _normalize_gps_feature_mode(feature_mode)
    if explicit_value not in (None, ""):
        return float(explicit_value), "config"
    if mode in {PAPER_CALIBRATED_GPS_MODE, "paper_calibrated_relative_polar"}:
        try:
            return float(PAPER_SCENE_CENTER_ANGLES_RAD[int(scene)]), "paper_scene_default"
        except KeyError as exc:
            raise ValueError(f"Paper GPS calibration is only defined for scenes 31-34, got scene {scene}.") from exc
    return None, "none"


def _scene_specific_cfg(base_cfg: ImageAEGPSDirectTrainingConfig, scene: int) -> ImageAEGPSDirectTrainingConfig:
    scene_obj = resolve_deepsense_scene(scene)
    base_scene_obj = resolve_deepsense_scene(base_cfg.scene)
    try:
        base_root = resolve_path(base_cfg.data_root)
        base_default = resolve_path(base_scene_obj.default_data_root)
        data_root = str(scene_obj.default_data_root) if base_root == base_default else str(base_root)
    except Exception:
        data_root = str(scene_obj.default_data_root)
    gps_angle_offset_rad = base_cfg.gps_angle_offset_rad
    gps_angle_offset_source = base_cfg.gps_angle_offset_source
    if base_cfg.gps_angle_offset_source == "paper_scene_default":
        gps_angle_offset_rad, gps_angle_offset_source = _resolve_gps_angle_offset(
            scene=int(scene_obj.scene_id),
            feature_mode=base_cfg.gps_feature_mode,
            explicit_value=None,
        )
    return replace(
        base_cfg,
        scene=int(scene_obj.scene_id),
        data_root=data_root,
        output_dir=str(Path(base_cfg.output_dir).parent / scene_obj.scene_slug),
        gps_angle_offset_rad=gps_angle_offset_rad,
        gps_angle_offset_source=gps_angle_offset_source,
    )


def _build_split_dataset(
    cfg: ImageAEGPSDirectTrainingConfig,
    *,
    split: str,
    gps_scaler: GPSStandardScaler | None = None,
    gps_normalize: bool | None = None,
) -> BeamBenchImageAEGPSDataset:
    if split == "train":
        csv_name = cfg.train_csv_name
        portion = cfg.train_portion
        max_samples = cfg.max_train_samples
    elif split == "test":
        csv_name = cfg.test_csv_name
        portion = cfg.test_portion
        max_samples = cfg.max_test_samples
    else:
        raise ValueError("split must be 'train' or 'test'.")
    return BeamBenchImageAEGPSDataset(
        data_root=cfg.data_root,
        csv_name=csv_name,
        split=split,
        seq_len=cfg.seq_len,
        num_pred=cfg.num_pred,
        image_size=cfg.image_size,
        num_beams=cfg.num_beams,
        target_beam_source=cfg.target_beam_source,
        portion=portion,
        portion_strategy=cfg.portion_strategy,
        portion_seed=cfg.portion_seed,
        gps_scaler=gps_scaler,
        gps_feature_mode=cfg.gps_feature_mode,
        gps_angle_offset_rad=cfg.gps_angle_offset_rad,
        gps_normalize=cfg.gps_normalize if gps_normalize is None else gps_normalize,
        max_samples=max_samples,
    )


def _paper_split_summary(scene_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    weighted_numerator = 0.0
    weighted_count = 0
    for report in scene_reports:
        scene = int(report["scene"])
        metrics = dict(report["metrics"])
        metric = float(metrics.get("official_top3_dba", 0.0))
        sample_count = int(metrics.get("valid_label_count", metrics.get("sample_count", 0)))
        target = float(TARGET_TABLE_III_ROW[f"scene{scene}"])
        weighted_numerator += metric * sample_count
        weighted_count += sample_count
        rows.append(
            {
                "scene": scene,
                "local_official_top3_dba": metric,
                "paper_tableiii_dba": target,
                "delta_local_minus_paper": metric - target,
                "sample_count": sample_count,
                "official_top1_acc": float(metrics.get("official_top1_acc", 0.0)),
                "official_top3_acc": float(metrics.get("official_top3_acc", 0.0)),
                "official_top5_acc": float(metrics.get("official_top5_acc", 0.0)),
                "circular_top3_dba": float(metrics.get("circular_top3_dba", 0.0)),
            }
        )
    rows = sorted(rows, key=lambda item: int(item["scene"]))
    local_simple_mean = sum(float(row["local_official_top3_dba"]) for row in rows) / max(len(rows), 1)
    paper_simple_mean = sum(float(row["paper_tableiii_dba"]) for row in rows) / max(len(rows), 1)
    local_weighted_overall = weighted_numerator / max(weighted_count, 1)
    return {
        "rows": rows,
        "metric_field": "official_top3_dba",
        "local_simple_mean": local_simple_mean,
        "local_weighted_overall": local_weighted_overall,
        "paper_simple_mean": paper_simple_mean,
        "paper_tableiii_overall": float(TARGET_TABLE_III_ROW["overall"]),
        "delta_weighted_minus_paper_overall": local_weighted_overall - float(TARGET_TABLE_III_ROW["overall"]),
    }


def _write_paper_split_summary_artifacts(report: Mapping[str, Any], output_root: Path) -> None:
    summary = dict(report["summary"])
    rows = list(summary["rows"])
    csv_path = output_root / "tableiii_camera_ae_gps_summary.csv"
    md_path = output_root / "tableiii_camera_ae_gps_summary.md"
    json_path = output_root / "tableiii_camera_ae_gps_summary.json"
    _write_csv_rows(csv_path, rows)
    md_path.write_text(_paper_split_summary_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")


def _paper_split_summary_markdown(report: Mapping[str, Any]) -> str:
    summary = dict(report["summary"])
    lines = [
        "# Camera AE + GPS Direct Paper-Split Local Reproduction",
        "",
        f"- Train scenes: {', '.join(str(item) for item in report['paper_split']['train_scenes'])}",
        f"- Eval scenes: {', '.join(str(item) for item in report['paper_split']['eval_scenes'])}",
        f"- Selection split: {report['selection']['mode']}",
        f"- Target beam source: {report['config'].get('target_beam_source', 'current')}",
        f"- GPS feature mode: {report['config'].get('gps_feature_mode', 'relative_polar')}",
        f"- Metric field: {summary['metric_field']}",
        "",
        "| Scene | Local DBA | Paper DBA | Delta | Top1 | Top3 | Top5 | Samples |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {scene} | {local:.4f} | {paper:.4f} | {delta:+.4f} | {top1:.4f} | {top3:.4f} | {top5:.4f} | {samples} |".format(
                scene=int(row["scene"]),
                local=float(row["local_official_top3_dba"]),
                paper=float(row["paper_tableiii_dba"]),
                delta=float(row["delta_local_minus_paper"]),
                top1=float(row["official_top1_acc"]),
                top3=float(row["official_top3_acc"]),
                top5=float(row["official_top5_acc"]),
                samples=int(row["sample_count"]),
            )
        )
    lines.extend(
        [
            "",
            f"- Local simple mean: {float(summary['local_simple_mean']):.4f}",
            f"- Local weighted overall: {float(summary['local_weighted_overall']):.4f}",
            f"- Paper Table III overall: {float(summary['paper_tableiii_overall']):.4f}",
            f"- Delta weighted overall: {float(summary['delta_weighted_minus_paper_overall']):+.4f}",
            "",
            str(report["official_comparability_note"]),
            "",
        ]
    )
    return "\n".join(lines)


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


def _classifier_logits_from_batch(
    model: BeamBenchImageAEGPSDirectModel,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    non_blocking: bool,
) -> torch.Tensor:
    gps = batch["gps"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    if "image_latent" in batch:
        image_latent = batch["image_latent"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        return model.forward_from_latent(image_latent, gps)
    image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    return model(image, gps)


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


def _gps_feature_version(mode: str) -> str:
    normalized = _normalize_gps_feature_mode(mode)
    if normalized == PAPER_CALIBRATED_GPS_MODE:
        return PAPER_DISTANCE_ANGLE_FEATURE_VERSION
    return "default"


def _build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    cfg: ImageAEGPSDirectTrainingConfig,
) -> DataLoader:
    workers = max(0, int(num_workers))
    kwargs: dict[str, Any] = {
        "batch_size": max(1, int(batch_size)),
        "shuffle": bool(shuffle),
        "num_workers": workers,
        "pin_memory": bool(cfg.pin_memory),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        if cfg.prefetch_factor is not None and int(cfg.prefetch_factor) > 0:
            kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(
        dataset,
        **kwargs,
    )


def _split_dataset(dataset: Dataset, *, val_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    length = len(dataset)
    if length <= 1:
        return dataset, dataset
    val_count = max(1, int(round(length * max(float(val_fraction), 0.0))))
    val_count = min(val_count, length - 1)
    train_count = length - val_count
    generator = torch.Generator().manual_seed(int(seed))
    train_subset, val_subset = random_split(dataset, [train_count, val_count], generator=generator)
    return train_subset, val_subset


def _load_ae_image(data_root: str | Path, rel_path: Any, image_size: int) -> torch.Tensor:
    path = joined_resource(data_root, str(rel_path))
    if not path.exists():
        raise FileNotFoundError(f"Camera image is missing: {path}")
    image = Image.open(path).convert("RGB").resize((int(image_size), int(image_size)), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return (tensor - 0.5) / 0.5


def _metadata_rows(value: Any, *, count: int) -> list[dict[str, Any]]:
    if value is None:
        return [{} for _ in range(count)]
    if isinstance(value, list):
        return [dict(item) if isinstance(item, Mapping) else {} for item in value]
    if not isinstance(value, Mapping):
        return [{} for _ in range(count)]
    rows: list[dict[str, Any]] = []
    for index in range(count):
        row = {}
        for key, item in value.items():
            row[str(key)] = _collated_value_at(item, index)
        rows.append(row)
    return rows


def _collated_value_at(value: Any, index: int) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        item = value[index]
        return item.item() if hasattr(item, "item") and item.ndim == 0 else item.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value[index].item() if value.ndim == 1 else value[index].tolist()
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    return value


def _write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_ready(row.get(key, "")) for key in fieldnames})


def _csv_ready(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True)
    if torch.is_tensor(value):
        return _csv_ready(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _csv_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _gps_scaler_metadata(scaler: GPSStandardScaler | None) -> dict[str, list[float]] | None:
    if scaler is None:
        return None
    if scaler.mean_ is None or scaler.scale_ is None:
        return None
    return {
        "mean": np.asarray(scaler.mean_, dtype=float).tolist(),
        "scale": np.asarray(scaler.scale_, dtype=float).tolist(),
    }


def _gps_scaler_from_metadata(payload: Any) -> GPSStandardScaler | None:
    if not isinstance(payload, Mapping):
        return None
    if "mean" not in payload or "scale" not in payload:
        return None
    return GPSStandardScaler(
        mean_=np.asarray(payload["mean"], dtype=np.float64),
        scale_=np.asarray(payload["scale"], dtype=np.float64),
    )


def _gps_calibration_metadata(cfg: ImageAEGPSDirectTrainingConfig) -> dict[str, Any]:
    if cfg.gps_feature_mode == "paper_distance_angle":
        note = (
            "paper_distance_angle follows the official challenge.py GPS Direct input: "
            "distance plus scene-calibrated angle in degrees."
        )
    elif cfg.gps_feature_mode == "paper_calibrated_relative_polar":
        note = (
            "paper_calibrated_relative_polar subtracts the paper's scene-specific "
            "boresight angle before encoding GPS as distance/sin/cos."
        )
    else:
        note = "relative_polar encodes GPS as distance/sin/cos without paper boresight calibration."
    return {
        "gps_feature_mode": str(cfg.gps_feature_mode),
        "gps_angle_offset_rad": None if cfg.gps_angle_offset_rad is None else float(cfg.gps_angle_offset_rad),
        "gps_angle_offset_source": str(cfg.gps_angle_offset_source),
        "paper_scene_center_angles_rad": dict(PAPER_SCENE_CENTER_ANGLES_RAD),
        "note": note,
    }


def _paper_split_gps_calibration_metadata(
    train_cfgs: Sequence[ImageAEGPSDirectTrainingConfig],
    eval_cfgs: Sequence[ImageAEGPSDirectTrainingConfig],
) -> dict[str, Any]:
    return {
        "train_scenes": {str(cfg.scene): _gps_calibration_metadata(cfg) for cfg in train_cfgs},
        "eval_scenes": {str(cfg.scene): _gps_calibration_metadata(cfg) for cfg in eval_cfgs},
    }


def _configure_torch_runtime(cfg: ImageAEGPSDirectTrainingConfig, device: torch.device) -> dict[str, Any]:
    report: dict[str, Any] = {
        "device": str(device),
        "cudnn_benchmark": False,
        "allow_tf32": False,
        "float32_matmul_precision": None,
    }
    if device.type != "cuda":
        return report
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = bool(cfg.allow_tf32)
        report["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
        report["allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    if bool(cfg.allow_tf32) and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
        report["float32_matmul_precision"] = "high"
    return report


def _performance_metadata(
    cfg: ImageAEGPSDirectTrainingConfig,
    device: torch.device,
    amp_enabled: bool,
    runtime_report: Mapping[str, Any],
    feature_cache_reports: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "device": str(device),
        "amp": {
            "enabled": bool(amp_enabled),
            "dtype": str(cfg.amp_dtype),
            "grad_scaler": bool(cfg.amp_grad_scaler) and bool(amp_enabled),
        },
        "runtime": dict(runtime_report),
        "optimizer": {
            "type": "AdamW",
            "fused_requested": bool(cfg.fused_optimizer),
        },
        "dataloader": {
            "num_workers": int(cfg.num_workers),
            "pin_memory": bool(cfg.pin_memory),
            "persistent_workers": bool(cfg.persistent_workers) and int(cfg.num_workers) > 0,
            "prefetch_factor": int(cfg.prefetch_factor) if cfg.prefetch_factor is not None else None,
            "non_blocking_transfer": bool(cfg.non_blocking_transfer),
        },
        "batches": {
            "ae_batch_size": int(cfg.ae_batch_size),
            "fusion_batch_size": int(cfg.fusion_batch_size),
            "feature_cache_batch_size": int(cfg.feature_cache_batch_size),
        },
        "feature_cache": {
            "enabled_requested": bool(cfg.cache_frozen_ae_features),
            "active": bool(cfg.freeze_ae_encoder and cfg.cache_frozen_ae_features),
            "reports": _json_ready(dict(feature_cache_reports)),
        },
    }


def _resolve_amp_dtype(name: str) -> torch.dtype:
    normalized = str(name or "float16").lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError("beambench_paper.amp_dtype must be 'float16' or 'bfloat16'.")


def _autocast_context(enabled: bool, device: torch.device, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def _make_grad_scaler(cfg: ImageAEGPSDirectTrainingConfig, amp_enabled: bool):
    enabled = bool(amp_enabled) and bool(cfg.amp_grad_scaler)
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _scaler_enabled(grad_scaler: Any) -> bool:
    return grad_scaler is not None and bool(getattr(grad_scaler, "is_enabled", lambda: False)())


def _build_adamw(
    params: Any,
    *,
    lr: float,
    weight_decay: float,
    device: torch.device,
    fused: bool,
) -> torch.optim.Optimizer:
    materialized = list(params)
    kwargs: dict[str, Any] = {}
    if bool(fused) and device.type == "cuda":
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(materialized, lr=lr, weight_decay=weight_decay, **kwargs)
    except TypeError:
        return torch.optim.AdamW(materialized, lr=lr, weight_decay=weight_decay)


def _torch_load(path: str | Path, *, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    if normalized == "auto":
        return bool(default)
    return bool(value)


def timestamped_default_output(scene: int | str) -> str:
    scene_obj = resolve_deepsense_scene(scene)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"outputs/{scene_obj.scene_slug}/beambench_image_ae_gps_direct/{stamp}"


__all__ = [
    "BeamBenchImageAEGPSDataset",
    "BeamBenchImageAEGPSDirectModel",
    "BeamBenchImageAEGPSFeatureDataset",
    "BeamBenchImageOnlyDataset",
    "ImageAEGPSDirectTrainingConfig",
    "TARGET_TABLE_III_ROW",
    "evaluate_image_ae_gps_model",
    "resolve_image_ae_gps_config",
    "run_image_ae_gps_paper_split_evaluation",
    "run_image_ae_gps_paper_split_training",
    "run_image_ae_gps_training",
    "timestamped_default_output",
    "train_camera_ae_for_image_gps_baseline",
]
