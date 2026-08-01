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
