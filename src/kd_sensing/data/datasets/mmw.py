from typing import Any

import numpy as np
import torch

from kd_sensing.data.datasets.mmw_family_adapter import MMWFamilyAdapter, prepare_mmw_family_init
from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset
from kd_sensing.registries import DATASETS


@DATASETS.register("mmw")
class MMWDataset(DeepSense6GDataset):
    """Prepared MMW sequence dataset using the existing beam/mmWave sample contract."""

    def __init__(
        self,
        condition: str = "sunny",
        scene: str | None = "town10_skybridge_seed24",
        scene_id: str | None = None,
        scene_slug: str | None = None,
        data_root: str | None = None,
        train_csv_name: str | None = None,
        test_csv_name: str | None = None,
        val_csv_name: str | None = None,
        return_geometry: bool = False,
        geometry_fields: list[str] | tuple[str, ...] | None = None,
        return_modality_availability: bool = False,
        radio_semantic: bool | dict[str, Any] | None = None,
        path_semantic: bool | dict[str, Any] | None = None,
        physical_label: bool | dict[str, Any] | None = None,
        beam_label_calibration: bool | dict[str, Any] | None = None,
        physics_supervision: bool | dict[str, Any] | None = None,
        field_map: dict[str, Any] | None = None,
        return_beam_power: bool | None = None,
        **kwargs: Any,
    ) -> None:
        init = prepare_mmw_family_init(
            condition=condition,
            scene=scene,
            scene_id=scene_id,
            scene_slug=scene_slug,
            data_root=data_root,
            train_csv_name=train_csv_name,
            test_csv_name=test_csv_name,
            val_csv_name=val_csv_name,
            beam_label_calibration=beam_label_calibration,
            physics_supervision=physics_supervision,
            kwargs=kwargs,
        )
        super().__init__(
            data_root=init.root,
            train_csv_name=init.train_csv_name,
            test_csv_name=init.test_csv_name,
            val_csv_name=init.val_csv_name,
            scene=31,
            beam_label_mapping=init.beam_label_mapping,
            **init.kwargs,
        )
        self.family_adapter = MMWFamilyAdapter(
            self,
            condition=init.condition,
            scenario=init.scenario,
            return_geometry=return_geometry,
            geometry_fields=geometry_fields,
            return_modality_availability=return_modality_availability,
            radio_semantic=radio_semantic,
            path_semantic=path_semantic,
            physical_label=physical_label,
            physics_supervision_config=init.physics_supervision_config,
            field_map=field_map,
            return_beam_power=return_beam_power,
            kwargs=init.kwargs,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.sample_cache is not None:
            cached = self.sample_cache.get(self._sample_cache_key(idx))
            if cached is not None:
                return cached
        sample, _ = self._getitem_with_timing(idx, collect_timing=False)
        sample = self.family_adapter.augment_sample(idx, sample)
        if self.sample_cache is not None and self.sample_cache_write_on_miss:
            self.sample_cache.put(self._sample_cache_key(idx), sample)
        return sample

    def _target_raw_beam_label_for_index(self, idx: int, horizon: int, beam_path: str) -> int:
        return self.family_adapter.target_raw_beam_label_for_index(idx, horizon, beam_path)

    def _target_beam_label_source_for_index(self, idx: int, horizon: int, beam_path: str) -> str:
        return self.family_adapter.target_beam_label_source_for_index(idx, horizon, beam_path)

    def _explicit_target_raw_label(self, idx: int, horizon: int) -> int | None:
        return self.family_adapter.explicit_target_raw_label(idx, horizon)

    def _geometry_for_index(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.family_adapter.geometry_for_index(idx)

    def _availability_for_index(self, idx: int) -> dict[str, Any]:
        return self.family_adapter.availability_for_index(idx)

    def _radio_semantic_for_index(self, idx: int, sample: dict[str, Any]) -> dict[str, Any]:
        return self.family_adapter.radio_semantic_for_index(idx, sample)

    def _physical_label_for_index(self, idx: int, sample: dict[str, Any]) -> dict[str, Any]:
        return self.family_adapter.physical_label_for_index(idx, sample)

    def _path_semantic_for_index(self, idx: int, sample: dict[str, Any]) -> dict[str, Any]:
        return self.family_adapter.path_semantic_for_index(idx, sample)

    def _load_beam_power(self, rel_path: object) -> tuple[np.ndarray | None, str]:
        return self.family_adapter.load_beam_power(rel_path)

    def _load_path_params(self, rel_path: object) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
        return self.family_adapter.load_path_params(rel_path)

    def _future_path_files_for_index(self, idx: int) -> list[str]:
        return self.family_adapter.future_path_files_for_index(idx)

    def _load_beam_to_channel_map(self) -> dict[str, str]:
        return self.family_adapter.load_beam_to_channel_map()

    def _load_or_build_physical_label_cache(self) -> dict[str, Any]:
        return self.family_adapter.load_or_build_physical_label_cache()

    def _build_physical_labels_for_index(
        self,
        idx: int,
        sample: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[dict[str, Any]]]:
        return self.family_adapter.build_physical_labels_for_index(idx, sample)

    def _target_beam_for_index(self, idx: int) -> list[int]:
        return self.family_adapter.target_beam_for_index(idx)

    def _physical_cache_metadata(self, *, classes: int, stats: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.family_adapter.physical_cache_metadata(classes=classes, stats=stats)

    def _calibrate_distribution(self, distribution: np.ndarray) -> np.ndarray:
        return self.family_adapter.calibrate_distribution(distribution)

    def _with_label_mapping_diagnostics(self, diagnostics: dict[str, Any]) -> dict[str, Any]:
        return self.family_adapter.with_label_mapping_diagnostics(diagnostics)


__all__ = ["MMWDataset"]
