from types import SimpleNamespace

import torch
from torch.utils.data import ConcatDataset, Dataset

from kd_sensing.engine import validator


class _SplitDataset(Dataset):
    split = "validation"

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> int:
        return index


def test_validate_uses_pooled_dataset_metadata_split(monkeypatch) -> None:
    pooled = ConcatDataset([_SplitDataset(), _SplitDataset()])
    dataloader = SimpleNamespace(dataset=pooled)
    result = SimpleNamespace(metrics={"val_loss": 1.0})
    monkeypatch.setattr(validator, "run_evaluation_pass", lambda *_args, **_kwargs: result)

    metrics = validator.validate(None, dataloader, {}, None, torch.device("cpu"))

    assert metrics["prediction_setup"]["splits"] == {"validation": {"num_samples": 2}}


def test_validate_applies_the_configured_fixed_single_modality(monkeypatch) -> None:
    pooled = ConcatDataset([_SplitDataset(), _SplitDataset()])
    dataloader = SimpleNamespace(dataset=pooled)
    captured = {}

    def fake_evaluation(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(metrics={"val_loss": 1.0})

    monkeypatch.setattr(validator, "run_evaluation_pass", fake_evaluation)
    cfg = {
        "model": {
            "primary": {
                "modalities": ["image", "radar", "gps", "lidar"],
                "use_sparse_csi": True,
            }
        },
        "temporal_missing": {
            "enabled": True,
            "mode": "fixed_single_modality",
            "fixed_modality": "gps",
        },
    }

    metrics = validator.validate(None, dataloader, cfg, None, torch.device("cpu"))

    assert torch.equal(captured["force_modality_mask"], torch.tensor([False, False, True, False, False]))
    raw_batch = {
        "image": torch.full((2, 5, 1), torch.nan),
        "radar_ra": torch.full((2, 5, 1), torch.inf),
        "radar_da": torch.full((2, 5, 1), torch.inf),
        "gps": torch.ones(2, 5, 1),
        "lidar": torch.full((2, 5, 1), torch.nan),
        "csi": torch.full((2, 5, 2, 2), complex(float("nan"), 0.0), dtype=torch.complex64),
        "csi_pilot_mask": torch.ones(2, 5, 2, 2, dtype=torch.bool),
    }
    transformed = captured["batch_transform"](raw_batch)
    assert torch.equal(transformed["gps"], torch.ones_like(transformed["gps"]))
    for key in ("image", "radar_ra", "radar_da", "lidar", "csi"):
        assert torch.equal(transformed[key], torch.zeros_like(transformed[key]))
    expected = torch.tensor([False, False, True, False, False]).expand(2, -1)
    assert torch.equal(transformed["available_modalities"], expected)
    assert metrics["fixed_modality"] == "gps"
    assert metrics["prediction_setup"]["temporal_missing"]["fixed_modality"] == "gps"
