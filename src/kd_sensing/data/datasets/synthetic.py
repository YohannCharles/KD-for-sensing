from __future__ import annotations

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
        use_lidar: bool = False,
        lidar_channels: int = 3,
        use_mmwave: bool = False,
        mmwave_input_size: int = 64,
        occlusion_target: bool | dict[str, object] | None = None,
        position_target: bool | dict[str, object] | None = None,
        seed: int = 0,
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
        self.use_lidar = use_lidar
        self.lidar_channels = lidar_channels
        self.use_mmwave = use_mmwave
        self.mmwave_input_size = mmwave_input_size
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
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = torch.rand((self.seq_len, self.image_channels, *self.image_size), generator=self.generator)
        radar_ra = torch.rand((self.seq_len, *self.radar_size), generator=self.generator)
        radar_da = torch.rand((self.seq_len, *self.radar_size), generator=self.generator)
        input_beam = torch.randint(0, self.num_classes, (self.seq_len,), generator=self.generator)
        target_beam = torch.randint(0, self.num_classes, (self.num_pred,), generator=self.generator)
        sample = {
            "image": image.float(),
            "radar_ra": radar_ra.float(),
            "radar_da": radar_da.float(),
            "input_beam": input_beam.long(),
            "target_beam": target_beam.long(),
        }
        if self.use_gps:
            sample["gps"] = torch.rand((self.seq_len, self.gps_input_size), generator=self.generator)
        if self.use_lidar:
            sample["lidar"] = torch.rand(
                (self.seq_len, self.lidar_channels, *self.lidar_size),
                generator=self.generator,
            )
        if self.use_mmwave:
            sample["mmwave"] = torch.rand((self.seq_len, self.mmwave_input_size), generator=self.generator)
        if self.occlusion_target_enabled:
            sample["occlusion_label"] = torch.randint(
                0,
                2,
                (self.num_pred,),
                generator=self.generator,
            ).float()
            sample["occlusion_valid"] = torch.ones((self.num_pred,), dtype=torch.bool)
        if self.position_target_enabled:
            sample["position_target"] = torch.rand((self.num_pred, 2), generator=self.generator).float()
            sample["position_valid"] = torch.ones((self.num_pred,), dtype=torch.bool)
        return sample


def _enabled(value: bool | dict[str, object] | None) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value.get("enabled", value.get("enable", False)))
    return False
