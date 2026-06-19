from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from kd_sensing.baselines.beambench.image_ae_gps_config import (
    ImageAEGPSDirectTrainingConfig,
    _normalize_target_beam_source,
    _scene_specific_cfg,
)
from kd_sensing.data.samples import create_samples
from kd_sensing.data.transform_ops.gps import GPSStandardScaler, PAPER_CALIBRATED_GPS_MODE, load_gps_feature_sequence
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.utils.paths import resolve_path


class BeamBenchImageAEGPSDataset(Dataset):
    """DeepSense6G image/GPS/beam dataset for the BeamBench Table III row."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        csv_name: str,
        split: str,
        seq_len: int = 1,
        gps_seq_len: int | None = None,
        gps_source_seq_len: int | None = None,
        gps_input_seq_len: int | None = None,
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
        selected_gps_source_seq_len = gps_source_seq_len if gps_source_seq_len is not None else gps_seq_len
        self.gps_source_seq_len = (
            int(selected_gps_source_seq_len) if selected_gps_source_seq_len is not None else self.seq_len
        )
        if self.gps_source_seq_len <= 0:
            raise ValueError("gps_source_seq_len must be positive when provided.")
        self.gps_seq_len = self.gps_source_seq_len
        self.gps_input_seq_len = int(gps_input_seq_len) if gps_input_seq_len is not None else None
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
            gps_source_seq_len=self.gps_source_seq_len,
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
            seq_len=self.gps_source_seq_len,
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
                    seq_len=self.gps_source_seq_len,
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
            "gps_seq_len": self.gps_source_seq_len,
            "gps_source_seq_len": self.gps_source_seq_len,
            "gps_input_seq_len": self.gps_input_seq_len,
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
        gps_seq_len=cfg.gps_seq_len,
        gps_source_seq_len=cfg.gps_source_seq_len,
        gps_input_seq_len=cfg.gps_input_seq_len,
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

def _build_paper_split_scene_datasets(
    base_cfg: ImageAEGPSDirectTrainingConfig,
    *,
    train_scenes: Sequence[int],
    eval_scenes: Sequence[int],
) -> tuple[
    list[ImageAEGPSDirectTrainingConfig],
    list[ImageAEGPSDirectTrainingConfig],
    list[BeamBenchImageAEGPSDataset],
    list[BeamBenchImageAEGPSDataset],
    GPSStandardScaler | None,
]:
    train_cfgs = [_scene_specific_cfg(base_cfg, scene) for scene in train_scenes]
    eval_cfgs = [_scene_specific_cfg(base_cfg, scene) for scene in eval_scenes]
    train_datasets = [_build_split_dataset(cfg, split="train") for cfg in train_cfgs]
    gps_scaler = None
    if base_cfg.gps_normalize:
        raw_gps = np.concatenate([dataset.raw_gps_matrix() for dataset in train_datasets], axis=0)
        gps_scaler = GPSStandardScaler().fit(raw_gps)
    for dataset in train_datasets:
        dataset.gps_scaler = gps_scaler
    eval_datasets = [
        _build_split_dataset(cfg, split="test", gps_scaler=gps_scaler, gps_normalize=base_cfg.gps_normalize)
        for cfg in eval_cfgs
    ]
    return train_cfgs, eval_cfgs, train_datasets, eval_datasets, gps_scaler

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


__all__ = [
    "BeamBenchImageAEGPSDataset",
    "BeamBenchImageAEGPSFeatureDataset",
    "BeamBenchImageOnlyDataset",
]
