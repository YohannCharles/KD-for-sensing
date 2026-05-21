from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from kd_sensing.data.layouts import raymobtime_s008_layout
from kd_sensing.data.transform_ops.image import IMAGENET_RGB_MEAN, IMAGENET_RGB_STD
from kd_sensing.modalities import image_profile_spec, normalize_modalities, resolve_image_profile
from kd_sensing.registries import DATASETS
from kd_sensing.utils.paths import resolve_path


RAYMOBTIME_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "validation": "val",
    "test": "test",
}


@DATASETS.register("raymobtime_s008")
class RaymobtimeS008SnapshotDataset(Dataset):
    """Raymobtime s008 current snapshot dataset backed by generated cache files."""

    def __init__(
        self,
        data_root: str | None = None,
        cache_dir: str | None = None,
        split: str = "train",
        modalities: list[str] | tuple[str, ...] | None = None,
        enabled_modalities: list[str] | tuple[str, ...] | None = None,
        normalize: bool = True,
        coord_normalize: bool | None = None,
        ray_normalize: bool | None = None,
        image_normalize: bool | None = None,
        image_profile: str | None = None,
        image_size: list[int] | tuple[int, int] | None = None,
        image_resize_in_dataset: bool = True,
        link_normalize: bool = False,
        use_coord: bool = False,
        use_ray: bool = False,
        use_lidar: bool = False,
        return_metadata: bool = True,
        **_: Any,
    ) -> None:
        layout = raymobtime_s008_layout()
        self.data_root = resolve_path(data_root or layout.root)
        self.cache_dir = resolve_path(cache_dir or self.data_root / "cache")
        self.split = _normalize_split(split)
        selected = modalities or enabled_modalities or _modalities_from_flags(
            use_coord=use_coord,
            use_lidar=use_lidar,
            use_ray=use_ray,
        )
        self.enabled_modalities = normalize_modalities(tuple(selected or ("coord",)), context="Raymobtime modalities")
        self.normalize = bool(normalize)
        self.coord_normalize = bool(normalize if coord_normalize is None else coord_normalize)
        self.ray_normalize = bool(normalize if ray_normalize is None else ray_normalize)
        self.image_normalize = bool(normalize if image_normalize is None else image_normalize)
        self.image_profile = resolve_image_profile(image_profile)
        image_spec = image_profile_spec(self.image_profile)
        self.image_size = tuple(int(value) for value in (image_size or image_spec.default_size))
        self.image_resize_in_dataset = bool(image_resize_in_dataset)
        self.link_normalize = bool(link_normalize)
        self.return_metadata = bool(return_metadata)
        self.use_coord = "coord" in self.enabled_modalities
        self.use_lidar = "lidar" in self.enabled_modalities
        self.use_ray = "ray" in self.enabled_modalities
        self.use_gps = False
        self.use_mmwave = False
        self.use_csi = False
        self.root_csv = self.cache_dir / f"index_{self.split}.csv"
        self.index = _read_required_csv(self.root_csv, self.cache_dir)
        self.cache = _read_required_npz(self.cache_dir / f"cache_{self.split}.npz", self.cache_dir)
        self.labels = _read_required_npz(self.cache_dir / f"labels_{self.split}.npz", self.cache_dir)
        self.ray_features = _read_optional_npz(self.cache_dir / f"ray_features_{self.split}.npz")
        self.cache_arrays = _materialize_npz_arrays(
            self.cache,
            _required_cache_keys(self.enabled_modalities),
            source_path=self.cache_dir / f"cache_{self.split}.npz",
        )
        self.label_arrays = _materialize_npz_arrays(
            self.labels,
            ("num_beam_classes",) if "num_beam_classes" in self.labels.files else (),
            source_path=self.cache_dir / f"labels_{self.split}.npz",
        )
        self.cache_loaded_keys = tuple(self.cache_arrays)
        self.cache_loaded_bytes = int(sum(int(array.nbytes) for array in self.cache_arrays.values()))
        self.cache_metadata = _read_json(self.cache_dir / "cache_metadata.json")
        self.split_metadata = _read_json(self.cache_dir / "split_metadata.json")
        self.scene_id = "s008"
        self.scene_slug = "raymobtime_s008"
        self.task_semantics = "current_snapshot_beam_selection"
        self.num_beam_classes = _metadata_int(self.cache_metadata, ("beam", "num_beam_classes"), default=None)
        if self.num_beam_classes is None and "num_beam_classes" in self.label_arrays:
            self.num_beam_classes = int(np.asarray(self.label_arrays["num_beam_classes"]).item())
        self.num_tx_beams = _metadata_int(self.cache_metadata, ("beam", "num_tx_beams"), default=None)
        self.num_rx_beams = _metadata_int(self.cache_metadata, ("beam", "num_rx_beams"), default=None)
        self.link_target_name = str(self.cache_metadata.get("link_target_name", "link_power_max_dbm"))
        self.link_target_unit = str(self.cache_metadata.get("link_target_unit", "dBm"))

    def __len__(self) -> int:
        return int(len(self.index))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample: dict[str, Any] = {}
        if "coord" in self.enabled_modalities:
            sample["coord"] = self._vector("coord", idx, normalize=self.coord_normalize).unsqueeze(0)
        if "image" in self.enabled_modalities:
            sample["image"] = self._image(idx).unsqueeze(0)
        if "lidar" in self.enabled_modalities:
            sample["lidar"] = self._lidar_occupancy(idx).unsqueeze(0)
        if "ray" in self.enabled_modalities:
            sample["ray"] = self._vector("ray", idx, normalize=self.ray_normalize).unsqueeze(0)
        sample["target_beam"] = torch.tensor(
            [int(np.asarray(self.cache_arrays["target_beam"])[idx])],
            dtype=torch.int64,
        )
        sample["los_label"] = torch.tensor(
            [float(np.asarray(self.cache_arrays["los_label"])[idx])],
            dtype=torch.float32,
        )
        link = torch.tensor([float(np.asarray(self.cache_arrays["link_quality"])[idx])], dtype=torch.float32)
        if self.link_normalize:
            link = self._normalize_tensor("link_quality", link)
        sample["link_quality"] = link
        sample["meta"] = self._metadata(idx)
        return sample

    def auxiliary_target_metadata(self) -> dict[str, Any]:
        return {
            "beam": {
                "num_beam_classes": self.num_beam_classes,
                "num_tx_beams": self.num_tx_beams,
                "num_rx_beams": self.num_rx_beams,
            },
            "los": {"source": "CoordVehiclesRxPerScene_s008.csv:LOS"},
            "link_quality": {
                "target_name": self.link_target_name,
                "unit": self.link_target_unit,
                "aggregation": self.cache_metadata.get("link_target_aggregation"),
            },
        }

    def raymobtime_metadata(self) -> dict[str, Any]:
        return {
            "dataset": "raymobtime_s008",
            "task_semantics": self.task_semantics,
            "cache_dir": str(self.cache_dir),
            "split_metadata_path": str(self.cache_dir / "split_metadata.json"),
            "cache_metadata_path": str(self.cache_dir / "cache_metadata.json"),
            "num_beam_classes": self.num_beam_classes,
            "num_tx_beams": self.num_tx_beams,
            "num_rx_beams": self.num_rx_beams,
            "link_target_name": self.link_target_name,
            "link_target_unit": self.link_target_unit,
            "cache_loaded_keys": list(self.cache_loaded_keys),
            "cache_loaded_bytes": self.cache_loaded_bytes,
        }

    def _vector(self, key: str, idx: int, *, normalize: bool) -> torch.Tensor:
        values = np.asarray(self.cache_arrays[key], dtype=np.float32)
        vector = torch.from_numpy(values[idx].astype(np.float32))
        if normalize:
            vector = self._normalize_tensor(key, vector)
        return vector

    def _image_like(self, key: str, idx: int, *, default_channels: int) -> torch.Tensor:
        values = np.asarray(self.cache_arrays[key], dtype=np.float32)
        item = values[idx]
        if item.ndim == 1:
            item = item.reshape(1, 1, -1)
        if item.ndim == 2:
            item = np.broadcast_to(item.reshape(1, *item.shape), (default_channels, *item.shape)).copy()
        if item.ndim == 3 and item.shape[-1] in {1, 3} and item.shape[0] not in {1, 3}:
            item = np.moveaxis(item, -1, 0)
        if item.ndim == 3 and item.shape[0] == 1 and default_channels > 1:
            item = np.broadcast_to(item, (default_channels, *item.shape[1:])).copy()
        if item.ndim != 3:
            raise ValueError(f"Raymobtime {key} cache item must be [C,H,W], [H,W,C], [H,W], or [F], got {item.shape}.")
        return torch.from_numpy(item.astype(np.float32))

    def _image(self, idx: int) -> torch.Tensor:
        image = self._image_like("image", idx, default_channels=3)
        if int(image.shape[0]) != 3:
            raise ValueError(f"Raymobtime image cache item must resolve to 3 RGB channels, got {tuple(image.shape)}.")
        if self.image_resize_in_dataset and tuple(int(value) for value in image.shape[-2:]) != self.image_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if self.image_normalize:
            image = _normalize_rgb_imagenet(image)
        return image

    def _lidar_occupancy(self, idx: int) -> torch.Tensor:
        values = np.asarray(self.cache_arrays["lidar"], dtype=np.float32)
        item = np.asarray(values[idx], dtype=np.float32)
        if item.ndim == 1:
            item = item.reshape(1, item.shape[0], 1, 1)
        elif item.ndim == 2:
            item = item.reshape(1, 1, *item.shape)
        elif item.ndim == 3:
            item = item.reshape(1, *item.shape)
        elif item.ndim == 4 and item.shape[-1] in {1, 2, 3, 4} and item.shape[0] not in {1, 2, 3, 4}:
            item = np.moveaxis(item, -1, 0)
        elif item.ndim != 4:
            raise ValueError(
                "Raymobtime lidar cache item must be a 3D occupancy grid [D,H,W] or [C,D,H,W], "
                f"got {item.shape}."
            )
        return torch.from_numpy(np.ascontiguousarray(item, dtype=np.float32))

    def _normalize_tensor(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        stats_key = "link_quality" if key == "link_quality" else key
        stats = self.cache_metadata.get("normalization", {}).get(stats_key)
        if not isinstance(stats, dict):
            return tensor
        mean = torch.tensor(stats.get("mean", [0.0]), dtype=tensor.dtype, device=tensor.device).reshape_as(tensor)
        std = torch.tensor(stats.get("std", [1.0]), dtype=tensor.dtype, device=tensor.device).reshape_as(tensor)
        return (tensor - mean) / std.clamp_min(1e-6)

    def _metadata(self, idx: int) -> dict[str, Any]:
        row = self.index.iloc[int(idx)]
        return {
            "sample_id": str(row["sample_id"]),
            "EpisodeID": int(row["EpisodeID"]),
            "SceneID": int(row["SceneID"]),
            "VehicleArrayID": int(row["VehicleArrayID"]),
            "VehicleName": str(row.get("VehicleName", "")),
            "valid_index": int(row["valid_index"]),
            "split": str(row["split"]),
        }


def _normalize_split(split: str) -> str:
    key = str(split).strip().lower()
    if key not in RAYMOBTIME_SPLIT_ALIASES:
        raise ValueError("Raymobtime s008 split must be one of train, validation/val, or test.")
    return RAYMOBTIME_SPLIT_ALIASES[key]


def _modalities_from_flags(*, use_coord: bool, use_lidar: bool, use_ray: bool) -> tuple[str, ...]:
    selected = ["coord"]
    if use_lidar:
        selected.append("lidar")
    if use_ray:
        selected.append("ray")
    if use_coord and "coord" not in selected:
        selected.append("coord")
    return tuple(selected)


def _read_required_csv(path: Path, cache_dir: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(_missing_cache_message(cache_dir, path))
    return pd.read_csv(path)


def _read_required_npz(path: Path, cache_dir: Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(_missing_cache_message(cache_dir, path))
    return np.load(path, allow_pickle=True)


def _read_optional_npz(path: Path) -> np.lib.npyio.NpzFile | None:
    return np.load(path, allow_pickle=True) if path.exists() else None


def _required_cache_keys(enabled_modalities: tuple[str, ...]) -> tuple[str, ...]:
    keys = ["target_beam", "los_label", "link_quality"]
    keys.extend(str(modality) for modality in enabled_modalities)
    return tuple(dict.fromkeys(keys))


def _materialize_npz_arrays(
    npz_file: np.lib.npyio.NpzFile,
    keys: tuple[str, ...],
    *,
    source_path: Path,
) -> dict[str, np.ndarray]:
    missing = [key for key in keys if key not in npz_file.files]
    if missing:
        available = ", ".join(npz_file.files)
        raise KeyError(
            f"Raymobtime s008 cache {source_path} is missing keys {missing}. Available keys: {available}"
        )
    return {key: np.asarray(npz_file[key]) for key in keys}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _missing_cache_message(cache_dir: Path, missing_path: Path) -> str:
    return (
        f"Raymobtime s008 cache is missing required file: {missing_path}. "
        "Run the Raymobtime s008 audit, index, ray feature, and cache preprocessors first, for example: "
        "conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/raymobtime_s008_cache.yaml. "
        f"Configured cache_dir: {cache_dir}"
    )


def _metadata_int(metadata: dict[str, Any], path: tuple[str, ...], *, default: int | None) -> int | None:
    cursor: Any = metadata
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return int(cursor)


def _normalize_rgb_imagenet(image: torch.Tensor) -> torch.Tensor:
    values = image.to(dtype=torch.float32)
    if values.numel() and float(values.max().detach().cpu()) > 2.0:
        values = values / 255.0
    mean = torch.tensor(IMAGENET_RGB_MEAN, dtype=values.dtype, device=values.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_RGB_STD, dtype=values.dtype, device=values.device).view(3, 1, 1)
    return (values - mean) / std


__all__ = ["RaymobtimeS008SnapshotDataset"]
