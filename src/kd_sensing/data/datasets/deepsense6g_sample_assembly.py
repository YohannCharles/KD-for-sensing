from typing import Any

import numpy as np
import torch


def build_beam_target_tensors(
    dataset: Any,
    idx: int,
    beam_paths: list[str],
    target_beam_paths: list[str],
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    raw_input_beam = [
        dataset._input_raw_beam_label_for_index(idx, horizon, beam_path)
        for horizon, beam_path in enumerate(beam_paths)
    ]
    raw_target_beam = [
        dataset._target_raw_beam_label_for_index(idx, horizon, beam_path)
        for horizon, beam_path in enumerate(target_beam_paths)
    ]
    input_beam = [dataset._map_beam_label(label) for label in raw_input_beam]
    target_beam = [dataset._map_beam_label(label) for label in raw_target_beam]
    return (
        {
            "input_beam": torch.tensor(input_beam, dtype=torch.int64),
            "target_beam": torch.tensor(target_beam, dtype=torch.int64),
        },
        {
            "input_beam": input_beam,
            "target_beam": target_beam,
            "raw_input_beam": raw_input_beam,
            "raw_target_beam": raw_target_beam,
        },
    )


def build_auxiliary_target_tensors(dataset: Any, idx: int, future_beam_paths: list[str]) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    if dataset.occlusion_target_enabled:
        occlusion_label, occlusion_valid = dataset.target_provider.occlusion_targets_for_paths(future_beam_paths)
        values["occlusion_label"] = torch.tensor(occlusion_label, dtype=torch.float32)
        values["occlusion_valid"] = torch.tensor(occlusion_valid, dtype=torch.bool)
    if dataset.position_target_enabled:
        position_target, position_valid = dataset.target_provider.position_targets_for_index(idx)
        if dataset.position_target_scaler is not None and dataset.position_target_normalize:
            scaled = position_target.copy()
            valid = position_valid.astype(bool)
            if np.any(valid):
                scaled[valid] = dataset.position_target_scaler.transform(scaled[valid])
            position_target = scaled
        values["position_target"] = torch.tensor(position_target, dtype=torch.float32)
        values["position_valid"] = torch.tensor(position_valid, dtype=torch.bool)
    return values


__all__ = ["build_auxiliary_target_tensors", "build_beam_target_tensors"]
