from __future__ import annotations

import torch
from torch.utils.data import Dataset

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
        radar_size: list[int] | tuple[int, int] = (128, 64),
        seed: int = 0,
        **_: object,
    ):
        self.length = length
        self.seq_len = seq_len
        self.num_pred = num_pred
        self.num_classes = num_classes
        self.image_size = tuple(image_size)
        self.radar_size = tuple(radar_size)
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = torch.rand((self.seq_len - 1, *self.image_size), generator=self.generator)
        radar_ra = torch.rand((self.seq_len, *self.radar_size), generator=self.generator)
        radar_da = torch.rand((self.seq_len, *self.radar_size), generator=self.generator)
        input_beam = torch.randint(0, self.num_classes, (self.seq_len,), generator=self.generator)
        target_beam = torch.randint(0, self.num_classes, (self.num_pred,), generator=self.generator)
        return {
            "image": image.float(),
            "radar_ra": radar_ra.float(),
            "radar_da": radar_da.float(),
            "input_beam": input_beam.long(),
            "target_beam": target_beam.long(),
        }

