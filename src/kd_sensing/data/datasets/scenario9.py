from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from kd_sensing.data.samples import create_samples
from kd_sensing.data.transforms import (
    GPSStandardScaler,
    SUPPORTED_GPS_FEATURE_MODE,
    build_image_transform,
    joined_resource,
    load_gps_feature_sequence,
    load_motion_masks,
    load_radar_maps,
)
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
        use_gps: bool = False,
        gps_feature_mode: str = SUPPORTED_GPS_FEATURE_MODE,
        gps_normalize: bool = True,
        gps_smooth_window: int = 3,
        gps_scaler: GPSStandardScaler | None = None,
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
        self.split = split
        self.use_gps = use_gps
        self.gps_feature_mode = gps_feature_mode
        self.gps_normalize = gps_normalize
        self.gps_smooth_window = gps_smooth_window
        self.gps_scaler = gps_scaler
        self._gps_feature_cache: dict[int, np.ndarray] = {}
        self.transform = build_image_transform(image_size)
        self.samples = create_samples(self.root_csv, portion=portion)
        if self.use_gps:
            self._ensure_gps_columns()
            self._prepare_gps_scaler()

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
        sample = {
            "image": image,
            "radar_ra": radar_ra,
            "radar_da": radar_da,
            "input_beam": torch.tensor(input_beam, dtype=torch.int64),
            "target_beam": torch.tensor(target_beam, dtype=torch.int64).squeeze(),
        }
        if self.use_gps:
            gps_features = self._gps_features_for_index(idx)
            if self.gps_scaler is not None:
                gps_features = self.gps_scaler.transform(gps_features)
            sample["gps"] = torch.tensor(gps_features, dtype=torch.float32)
        return sample

    def _ensure_gps_columns(self) -> None:
        if self.samples.gps_paths is None:
            raise ValueError(
                f"GPS is enabled but {self.root_csv} does not contain gps1..gpsN columns. "
                "Regenerate sequence CSVs with include_gps: true."
            )
        if self.gps_feature_mode != SUPPORTED_GPS_FEATURE_MODE:
            raise ValueError(
                f"Unsupported gps_feature_mode '{self.gps_feature_mode}'. "
                f"This change only supports '{SUPPORTED_GPS_FEATURE_MODE}'."
            )
        if self.samples.bs_gps_paths is None:
            raise ValueError(
                f"gps_feature_mode '{self.gps_feature_mode}' requires bs_gps1..bs_gpsN columns in {self.root_csv}."
            )

    def _prepare_gps_scaler(self) -> None:
        if not self.gps_normalize:
            self.gps_scaler = None
            return
        if self.gps_scaler is not None:
            return
        if self.split != "train":
            raise ValueError(
                "GPS normalization for non-train split requires a train-fitted gps_scaler. "
                "Use build_dataloaders/evaluate so the train scaler can be reused."
            )
        all_features = [self._gps_features_for_index(idx) for idx in range(len(self))]
        stacked = np.concatenate(all_features, axis=0)
        self.gps_scaler = GPSStandardScaler().fit(stacked)

    def _gps_features_for_index(self, idx: int) -> np.ndarray:
        if idx not in self._gps_feature_cache:
            if self.samples.gps_paths is None:
                raise ValueError("GPS paths are unavailable for this dataset.")
            bs_paths = self.samples.bs_gps_paths[idx] if self.samples.bs_gps_paths is not None else None
            self._gps_feature_cache[idx] = load_gps_feature_sequence(
                self.data_root,
                self.samples.gps_paths[idx],
                bs_paths,
                seq_len=self.seq_len,
                mode=self.gps_feature_mode,
                smooth_window=self.gps_smooth_window,
            )
        return self._gps_feature_cache[idx]
