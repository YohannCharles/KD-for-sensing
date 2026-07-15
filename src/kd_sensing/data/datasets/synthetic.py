import hashlib
import json

import torch
from torch.utils.data import Dataset

from kd_sensing.data.transform_ops.mmwave import OcclusionTargetStats
from kd_sensing.registries import DATASETS


@DATASETS.register("synthetic_sequence")
@DATASETS.register("synthetic")
class SyntheticSequenceDataset(Dataset):
    """Small synthetic dataset for import and train/eval smoke checks."""

    def __init__(
        self,
        length: int = 4,
        seq_len: int = 8,
        num_pred: int = 3,
        num_classes: int = 64,
        image_size: list[int] | tuple[int, int] = (224, 224),
        image_channels: int = 3,
        radar_size: list[int] | tuple[int, int] = (128, 64),
        lidar_size: list[int] | tuple[int, int] = (224, 224),
        use_gps: bool = False,
        gps_input_size: int = 3,
        use_gps_bev_xy: bool = False,
        gps_bev_roi: list[float] | tuple[float, ...] = (-20.0, 20.0, -20.0, 20.0),
        use_lidar: bool = False,
        lidar_channels: int = 3,
        use_mmwave: bool = False,
        mmwave_input_size: int = 64,
        use_csi: bool = False,
        csi_shape: list[int] | tuple[int, int] = (16, 4),
        csi_train_rms: bool = True,
        csi_rms_normalizer: object | None = None,
        occlusion_target: bool | dict[str, object] | None = None,
        position_target: bool | dict[str, object] | None = None,
        seed: int = 0,
        split: str = "train",
        **_: object,
    ):
        self.length = length
        self.seq_len = seq_len
        self.num_pred = num_pred
        self.num_classes = num_classes
        self.image_size = tuple(image_size)
        self.image_channels = image_channels
        self.radar_size = tuple(radar_size)
        self.lidar_size = tuple(lidar_size)
        self.use_gps = use_gps
        self.gps_input_size = gps_input_size
        self.use_gps_bev_xy = bool(use_gps_bev_xy)
        self.gps_bev_roi = tuple(float(value) for value in gps_bev_roi)
        self.use_lidar = use_lidar
        self.lidar_channels = lidar_channels
        self.use_mmwave = use_mmwave
        self.mmwave_input_size = mmwave_input_size
        self.use_csi = use_csi
        self.csi_shape = tuple(int(value) for value in csi_shape)
        self.csi_train_rms = bool(csi_train_rms)
        self.csi_rms_normalizer = csi_rms_normalizer
        self.occlusion_target_enabled = _enabled(occlusion_target)
        self.position_target_enabled = _enabled(position_target)
        self.position_target_normalize = False
        self.occlusion_target_stats = (
            OcclusionTargetStats(
                threshold=0.5,
                threshold_percentile=50.0,
                sample_count=length,
                positive_count=length // 2,
                positive_ratio=(length // 2) / max(length, 1),
            )
            if self.occlusion_target_enabled
            else None
        )
        self.position_target_scaler = None
        self.seed = int(seed)
        self.split = str(split)
        self.schema_identity = hashlib.sha256(
            json.dumps(
                {
                    "schema_version": 1,
                    "seq_len": self.seq_len,
                    "num_pred": self.num_pred,
                    "num_classes": self.num_classes,
                    "image_size": self.image_size,
                    "image_channels": self.image_channels,
                    "radar_size": self.radar_size,
                    "lidar_size": self.lidar_size,
                    "use_gps": self.use_gps,
                    "gps_input_size": self.gps_input_size,
                    "use_gps_bev_xy": self.use_gps_bev_xy,
                    "use_lidar": self.use_lidar,
                    "lidar_channels": self.lidar_channels,
                    "use_mmwave": self.use_mmwave,
                    "mmwave_input_size": self.mmwave_input_size,
                    "use_csi": self.use_csi,
                    "csi_shape": self.csi_shape,
                    "occlusion_target": self.occlusion_target_enabled,
                    "position_target": self.position_target_enabled,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        generator = self._generator_for_index(idx)
        image = torch.rand((self.seq_len, self.image_channels, *self.image_size), generator=generator)
        radar_ra = torch.rand((self.seq_len, *self.radar_size), generator=generator)
        radar_da = torch.rand((self.seq_len, *self.radar_size), generator=generator)
        input_beam = torch.randint(0, self.num_classes, (self.seq_len,), generator=generator)
        target_beam = torch.randint(0, self.num_classes, (self.num_pred,), generator=generator)
        sample = {
            "image": image.float(),
            "radar_ra": radar_ra.float(),
            "radar_da": radar_da.float(),
            "input_beam": input_beam.long(),
            "target_beam": target_beam.long(),
            "history_indices": torch.arange(self.seq_len, dtype=torch.long),
            "target_index": torch.tensor(self.seq_len, dtype=torch.long),
        }
        if self.use_gps:
            sample["gps"] = torch.rand((self.seq_len, self.gps_input_size), generator=generator)
        if self.use_gps_bev_xy:
            x_min, x_max, y_min, y_max = self.gps_bev_roi
            xy = torch.rand((self.seq_len, 2), generator=generator)
            xy[:, 0] = xy[:, 0] * (x_max - x_min) + x_min
            xy[:, 1] = xy[:, 1] * (y_max - y_min) + y_min
            sample["gps_bev_xy"] = xy.float()
        if self.use_lidar:
            sample["lidar"] = torch.rand(
                (self.seq_len, self.lidar_channels, *self.lidar_size),
                generator=generator,
            )
        if self.use_mmwave:
            sample["mmwave"] = torch.rand((self.seq_len, self.mmwave_input_size), generator=generator)
        if self.use_csi:
            sample["csi"] = torch.randn((self.seq_len, *self.csi_shape, 2), generator=generator).float()
        if self.occlusion_target_enabled:
            sample["occlusion_label"] = torch.randint(
                0,
                2,
                (self.num_pred,),
                generator=generator,
            ).float()
            sample["occlusion_valid"] = torch.ones((self.num_pred,), dtype=torch.bool)
        if self.position_target_enabled:
            sample["position_target"] = torch.rand((self.num_pred, 2), generator=generator).float()
            sample["position_valid"] = torch.ones((self.num_pred,), dtype=torch.bool)
        return sample

    def _generator_for_index(self, index: int) -> torch.Generator:
        payload = f"{self.seed}:{self.split}:{self.schema_identity}:{int(index)}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
        return torch.Generator().manual_seed(seed)


def _enabled(value: bool | dict[str, object] | None) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value.get("enabled", value.get("enable", False)))
    return False
