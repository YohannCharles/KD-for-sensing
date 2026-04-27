from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from kd_sensing.data.samples import create_samples
from kd_sensing.data.transforms import build_image_transform, joined_resource, load_motion_masks, load_radar_maps
from kd_sensing.registries import DATASETS
from kd_sensing.utils.paths import resolve_path


@DATASETS.register("scenario9")
class Scenario9Dataset(Dataset):
    """Scenario 9 sequence dataset with standardized batch field names."""

    def __init__(
        self,
        data_root: str,
        csv_name: str | None = None,
        root_csv: str | None = None,
        split: str = "train",
        train_csv_name: str = "train_seqs_RA.csv",
        test_csv_name: str = "test_seqs_RA.csv",
        seq_len: int = 8,
        num_pred: int = 3,
        image_size: list[int] | tuple[int, int] = (224, 224),
        fft_tuple: list[int] | tuple[int, int, int] = (64, 256, 128),
        clipped_range: int = 128,
        portion: float = 1.0,
        **_: object,
    ):
        self.data_root = resolve_path(data_root)
        selected_csv = root_csv or csv_name
        if selected_csv is None:
            selected_csv = train_csv_name if split == "train" else test_csv_name
        self.root_csv = Path(selected_csv)
        if not self.root_csv.is_absolute():
            self.root_csv = self.data_root / self.root_csv
        self.seq_len = seq_len
        self.num_pred = num_pred
        self.fft_tuple = tuple(fft_tuple)
        self.clipped_range = clipped_range
        self.transform = build_image_transform(image_size)
        self.samples = create_samples(self.root_csv, portion=portion)

    def __len__(self) -> int:
        return len(self.samples.rgb_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        samples_rgb = self.samples.rgb_paths[idx][-self.seq_len :]
        samples_radar = self.samples.radar_paths[idx][-self.seq_len :]
        beam_paths = self.samples.input_beam_paths[idx][-self.seq_len :]
        future_beam_paths = self.samples.future_beam_paths[idx][: self.num_pred]

        image = load_motion_masks(self.data_root, samples_rgb, self.seq_len, self.transform)
        radar_ra, radar_da = load_radar_maps(
            self.data_root,
            samples_radar,
            self.seq_len,
            self.fft_tuple,
            self.clipped_range,
        )

        input_beam = [
            int(np.argmax(np.loadtxt(joined_resource(self.data_root, beam_path))))
            for beam_path in beam_paths
        ]
        target_beam = [
            int(np.argmax(np.loadtxt(joined_resource(self.data_root, beam_path))))
            for beam_path in future_beam_paths
        ]
        return {
            "image": image,
            "radar_ra": radar_ra,
            "radar_da": radar_da,
            "input_beam": torch.tensor(input_beam, dtype=torch.int64),
            "target_beam": torch.tensor(target_beam, dtype=torch.int64).squeeze(),
        }

