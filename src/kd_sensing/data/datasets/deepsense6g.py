"""Minimal DeepSense6G Scene31--34 four-sensor dataset."""

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from kd_sensing.data.transform_ops.gps import GPSStandardScaler, load_gps_feature_sequence, normalize_gps_feature_mode
from kd_sensing.data.transform_ops.image import build_rgb_imagenet_transform, load_rgb_imagenet_frames
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.data.transform_ops.lidar import load_lidar_bev_sequence
from kd_sensing.data.transform_ops.radar import load_radar_maps
from kd_sensing.modalities import normalize_modalities, resolve_image_profile
from kd_sensing.registries import DATASETS


SUPPORTED_SCENES = frozenset((31, 32, 33, 34))
FOUR_MODALITIES = ("image", "radar", "gps", "lidar")
BEAM_POWER_SIZE = 64


@dataclass
class _DeepSense6GSamples:
    rows: list[dict[str, str]]
    rgb_paths: list[list[str]]
    radar_paths: list[list[str]]
    gps_paths: list[list[str]]
    bs_gps_paths: list[list[str]]
    lidar_paths: list[list[str]]
    future_beam_paths: list[list[str]]
    metadata: dict[str, Any]


@DATASETS.register("deepsense6g")
class DeepSense6GDataset(Dataset):
    """Scene31--34 DeepSense6G sequences on the retained four-sensor contract."""

    def __init__(
        self,
        *,
        scene: int,
        data_root: str | None = None,
        train_csv_name: str | None = None,
        test_csv_name: str | None = None,
        val_csv_name: str | None = None,
        split: str = "train",
        seq_len: int = 5,
        num_pred: int = 1,
        portion: float = 1.0,
        portion_strategy: str = "even",
        portion_seed: int = 42,
        image_profile: str | None = "rgb_imagenet",
        image_size: list[int] | tuple[int, int] = (224, 224),
        fft_tuple: list[int] | tuple[int, int, int] = (64, 256, 128),
        clipped_range: int = 128,
        use_gps: bool = True,
        gps_feature_mode: str = "relative_polar",
        gps_normalize: bool = True,
        gps_scaler: GPSStandardScaler | None = None,
        use_lidar: bool = True,
        lidar_bev_size: list[int] | tuple[int, int] = (224, 224),
        lidar_roi: list[float] | tuple[float, ...] = (-30.0, 30.0, -30.0, 30.0, -3.0, 5.0),
        lidar_fov_degrees: list[float] | tuple[float, float] | None = None,
        lidar_remove_ground: bool = False,
        lidar_ground_z_threshold: float = 0.1,
        lidar_background_distance_threshold: float = 0.2,
        lidar_augment: bool = False,
        lidar_point_dropout: float = 0.0,
        lidar_jitter_std: float = 0.0,
        enabled_modalities: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if type(scene) is not int or scene not in SUPPORTED_SCENES:
            raise ValueError(f"DeepSense6G scene must be one of {sorted(SUPPORTED_SCENES)}, got {scene!r}.")
        self.scene_id = scene
        self.scene_slug = f"scenario{scene}"
        self.data_root = Path(data_root) if data_root is not None else Path("dataset/DeepSense6G") / self.scene_slug
        self.split = str(split)
        csv_name = _split_csv_name(
            self.split,
            train_csv_name=train_csv_name,
            test_csv_name=test_csv_name,
            val_csv_name=val_csv_name,
        )
        csv_path = Path(csv_name)
        self.root_csv = csv_path if csv_path.is_absolute() else joined_resource(self.data_root, csv_name)
        if not self.root_csv.exists():
            raise FileNotFoundError(f"DeepSense6G split CSV is missing: {self.root_csv}")

        self.seq_len = int(seq_len)
        self.num_pred = int(num_pred)
        if self.seq_len <= 0 or self.num_pred <= 0:
            raise ValueError("DeepSense6G seq_len and num_pred must be positive.")
        requested = normalize_modalities(
            tuple(enabled_modalities or FOUR_MODALITIES),
            context="DeepSense6G enabled modalities",
        )
        if requested != FOUR_MODALITIES:
            raise ValueError("DeepSense6G requires exactly image/radar/gps/lidar.")
        if not use_gps or not use_lidar:
            raise ValueError("DeepSense6G requires enabled GPS and LiDAR inputs.")
        if not gps_normalize:
            raise ValueError("DeepSense6G requires train-fitted GPS normalization.")
        self.enabled_modalities = requested
        self.use_gps = True
        self.use_lidar = True

        self.image_profile = resolve_image_profile(image_profile)
        self.image_size = tuple(int(value) for value in image_size)
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("DeepSense6G image_size must contain two positive dimensions.")
        self.transform = build_rgb_imagenet_transform(self.image_size)
        self.fft_tuple = tuple(int(value) for value in fft_tuple)
        if len(self.fft_tuple) != 3 or min(self.fft_tuple) <= 0:
            raise ValueError("DeepSense6G fft_tuple must contain three positive dimensions.")
        self.clipped_range = int(clipped_range)
        if self.clipped_range <= 0:
            raise ValueError("DeepSense6G clipped_range must be positive.")

        self.gps_feature_mode = normalize_gps_feature_mode(gps_feature_mode)
        self.gps_normalize = True
        self.gps_scaler = gps_scaler
        self.gps_scaler_metadata: dict[str, Any] | None = None
        self._gps_feature_cache: dict[str, np.ndarray] = {}

        self.lidar_bev_size = tuple(int(value) for value in lidar_bev_size)
        self.lidar_roi = tuple(float(value) for value in lidar_roi)
        self.lidar_fov_degrees = tuple(float(value) for value in lidar_fov_degrees) if lidar_fov_degrees is not None else None
        self.lidar_remove_ground = bool(lidar_remove_ground)
        self.lidar_ground_z_threshold = float(lidar_ground_z_threshold)
        self.lidar_background_distance_threshold = float(lidar_background_distance_threshold)
        self.lidar_augment = bool(lidar_augment)
        self.lidar_point_dropout = float(lidar_point_dropout)
        self.lidar_jitter_std = float(lidar_jitter_std)
        self._lidar_augmentation_seed = int(portion_seed)
        self._lidar_epoch = 0

        self.samples = _load_samples(
            self.root_csv,
            seq_len=self.seq_len,
            num_pred=self.num_pred,
            portion=float(portion),
            portion_strategy=portion_strategy,
            portion_seed=int(portion_seed),
        )
        self._future_beam_labels = _load_future_beam_labels(
            self.data_root,
            self.samples.future_beam_paths,
        )
        self.schema_identity = {
            "dataset_family": "DeepSense6G",
            "scene": self.scene_id,
            "modalities": list(self.enabled_modalities),
            "seq_len": self.seq_len,
            "num_pred": self.num_pred,
            "target": "future_beam_power_argmax_64",
        }

    def __len__(self) -> int:
        return len(self.samples.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        radar_ra, radar_da = load_radar_maps(
            self.data_root,
            self.samples.radar_paths[idx],
            self.seq_len,
            self.fft_tuple,
            self.clipped_range,
        )
        gps = self._gps_features_for_index(idx)
        sample: dict[str, Any] = {
            "target_beam": torch.tensor(
                [self._target_beam_label(idx, horizon) for horizon in range(self.num_pred)],
                dtype=torch.long,
            ),
            "image": load_rgb_imagenet_frames(
                self.data_root,
                self.samples.rgb_paths[idx],
                self.seq_len,
                self.transform,
                image_size=self.image_size,
            ),
            "radar_ra": radar_ra,
            "radar_da": radar_da,
            "gps": torch.tensor(
                self.gps_scaler.transform(gps) if self.gps_scaler is not None else gps,
                dtype=torch.float32,
            ),
            "lidar": torch.tensor(
                self._lidar_bev_for_index(idx, augment=self.split == "train" and self.lidar_augment),
                dtype=torch.float32,
            ),
        }
        return self._with_metadata(idx, sample)

    def _target_beam_label(self, idx: int, horizon: int) -> int:
        return self._future_beam_labels[idx][horizon]

    def _gps_features_for_index(self, idx: int) -> np.ndarray:
        return load_gps_feature_sequence(
            self.data_root,
            self.samples.gps_paths[idx],
            self.samples.bs_gps_paths[idx],
            seq_len=self.seq_len,
            mode=self.gps_feature_mode,
            frame_feature_cache=self._gps_feature_cache,
        )

    def _lidar_bev_for_index(self, idx: int, *, augment: bool) -> np.ndarray:
        return load_lidar_bev_sequence(
            self.data_root,
            self.samples.lidar_paths[idx],
            seq_len=self.seq_len,
            bev_size=self.lidar_bev_size,
            roi=self.lidar_roi,
            fov_degrees=self.lidar_fov_degrees,
            remove_ground=self.lidar_remove_ground,
            ground_z_threshold=self.lidar_ground_z_threshold,
            background_distance_threshold=self.lidar_background_distance_threshold,
            augment=augment,
            point_dropout=self.lidar_point_dropout,
            jitter_std=self.lidar_jitter_std,
            rng=self._lidar_rng(idx) if augment else None,
        )

    def set_epoch(self, epoch: int) -> None:
        self._lidar_epoch = int(epoch)

    def _lidar_rng(self, idx: int) -> np.random.Generator:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        sample_id = str(self.samples.rows[idx].get("sample_id") or self.samples.rows[idx].get("seq_index") or idx)
        payload = f"{self._lidar_augmentation_seed}:{self._lidar_epoch}:{worker_id}:{self.split}:{sample_id}".encode("utf-8")
        return np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))

    def _with_metadata(self, idx: int, sample: dict[str, Any]) -> dict[str, Any]:
        row = self.samples.rows[idx]
        seq_index = _row_text(row.get("seq_index")) or str(idx)
        sample_id = f"deepsense6g:{self.scene_slug}:{self.split}:{seq_index}"
        sample["metadata"] = {
            "dataset_family": "DeepSense6G",
            "scene_id": self.scene_id,
            "scene_slug": self.scene_slug,
            "split": self.split,
            "dataset_index": int(idx),
            "sample_id": sample_id,
            "seq_index": seq_index,
            "root_csv": str(self.root_csv),
            "seq_len": self.seq_len,
            "prediction_window": self.num_pred,
            "future_beam_path": self.samples.future_beam_paths[idx][0],
            "target_label_source": "future_beam_power_argmax",
        }
        sample["sample_id"] = sample_id
        sample["domain_metadata"] = {
            "dataset_family": "DeepSense6G",
            "scene_id": self.scene_id,
            "scene_slug": self.scene_slug,
        }
        return sample


def _split_csv_name(
    split: str,
    *,
    train_csv_name: str | None,
    test_csv_name: str | None,
    val_csv_name: str | None,
) -> str:
    if split == "train":
        return train_csv_name or "train_seqs_RA_GPS_LIDAR.csv"
    if split == "test":
        return test_csv_name or "test_seqs_RA_GPS_LIDAR.csv"
    if split == "validation" and val_csv_name:
        return val_csv_name
    if split == "validation":
        raise ValueError("DeepSense6G validation requires an explicit val_csv_name.")
    raise ValueError(f"Unsupported DeepSense6G split: {split}.")


def _load_samples(
    csv_path: Path,
    *,
    seq_len: int,
    num_pred: int,
    portion: float,
    portion_strategy: str,
    portion_seed: int,
) -> _DeepSense6GSamples:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [str(name) for name in reader.fieldnames or ()]
        rows = [{str(key): str(value or "") for key, value in row.items()} for row in reader]
    if not fieldnames:
        raise ValueError(f"DeepSense6G CSV has no header: {csv_path}")

    columns = {
        prefix: _numbered_columns(fieldnames, prefix)
        for prefix in ("camera", "radar", "gps", "bs_gps", "lidar", "future_beam")
    }
    _validate_columns(csv_path, columns, seq_len=seq_len, num_pred=num_pred)
    rows, metadata = _select_rows(rows, portion=portion, strategy=portion_strategy, seed=portion_seed)

    return _DeepSense6GSamples(
        rows=rows,
        rgb_paths=_paths_for_rows(csv_path, rows, columns["camera"]),
        radar_paths=_paths_for_rows(csv_path, rows, columns["radar"]),
        gps_paths=_paths_for_rows(csv_path, rows, columns["gps"]),
        bs_gps_paths=_paths_for_rows(csv_path, rows, columns["bs_gps"]),
        lidar_paths=_paths_for_rows(csv_path, rows, columns["lidar"]),
        future_beam_paths=_paths_for_rows(csv_path, rows, columns["future_beam"]),
        metadata=metadata,
    )


def _numbered_columns(fieldnames: list[str], prefix: str) -> list[str]:
    indexed = sorted(
        (
            (int(name[len(prefix) :]), name)
            for name in fieldnames
            if name.startswith(prefix) and name[len(prefix) :].isdigit()
        ),
        key=lambda item: item[0],
    )
    if indexed and [number for number, _ in indexed] != list(range(1, len(indexed) + 1)):
        raise ValueError(f"DeepSense6G CSV has non-contiguous {prefix}1..{prefix}N columns.")
    return [name for _, name in indexed]


def _validate_columns(csv_path: Path, columns: dict[str, list[str]], *, seq_len: int, num_pred: int) -> None:
    required = {
        "camera": seq_len,
        "radar": seq_len,
        "gps": seq_len,
        "bs_gps": seq_len,
        "lidar": seq_len,
        "future_beam": num_pred,
    }
    for prefix, minimum in required.items():
        if len(columns[prefix]) < minimum:
            raise ValueError(
                f"DeepSense6G CSV {csv_path} needs at least {minimum} {prefix}1..{prefix}N columns, "
                f"found {len(columns[prefix])}."
            )


def _paths_for_rows(csv_path: Path, rows: list[dict[str, str]], columns: list[str]) -> list[list[str]]:
    paths: list[list[str]] = []
    for row_index, row in enumerate(rows):
        values = []
        for column in columns:
            value = _row_text(row.get(column))
            if value is None:
                raise ValueError(f"DeepSense6G CSV {csv_path} row {row_index} is missing {column}.")
            values.append(value)
        paths.append(values)
    return paths


def _load_future_beam_labels(data_root: Path, paths_by_sample: list[list[str]]) -> list[list[int]]:
    labels: list[list[int]] = []
    for sample_index, paths in enumerate(paths_by_sample):
        sample_labels = []
        for horizon, rel_path in enumerate(paths):
            path = joined_resource(data_root, rel_path)
            try:
                powers = np.asarray(np.loadtxt(path, dtype=np.float64)).reshape(-1)
            except Exception as exc:
                raise ValueError(
                    f"Failed to read DeepSense6G future beam power file for row {sample_index}, horizon {horizon + 1}: {path}: {exc}"
                ) from exc
            if powers.size != BEAM_POWER_SIZE or not np.isfinite(powers).all():
                raise ValueError(
                    f"DeepSense6G future beam power file for row {sample_index}, horizon {horizon + 1} "
                    f"must contain {BEAM_POWER_SIZE} finite values: {path}"
                )
            sample_labels.append(int(np.argmax(powers)))
        labels.append(sample_labels)
    return labels


def _select_rows(
    rows: list[dict[str, str]],
    *,
    portion: float,
    strategy: str,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not np.isfinite(portion) or portion <= 0:
        raise ValueError(f"portion must be positive, got {portion}.")
    total = len(rows)
    if portion >= 1.0 or total == 0:
        indices = list(range(total))
        effective_strategy = "all"
    else:
        count = min(max(1, int(total * portion)), total)
        if strategy == "head":
            indices = list(range(count))
        elif strategy == "random":
            indices = sorted(int(index) for index in np.random.default_rng(seed).choice(total, size=count, replace=False))
        elif strategy == "even":
            indices = [int(round(value)) for value in np.linspace(0, total - 1, count)]
        else:
            raise ValueError(f"Unsupported portion_strategy '{strategy}'.")
        effective_strategy = strategy
    selected = [rows[index] for index in indices]
    return selected, {
        "total_rows": total,
        "selected_rows": len(selected),
        "portion": float(portion),
        "portion_strategy": effective_strategy,
        "portion_seed": int(seed),
    }


def _row_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text if text and text.lower() not in {"nan", "none", "-99"} else None


__all__ = ["DeepSense6GDataset", "SUPPORTED_SCENES"]
