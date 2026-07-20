from pathlib import Path

import pytest
import torch

from kd_sensing.engine.checkpointing import CheckpointManager
from kd_sensing.engine.trainer_runtime_helpers import _checkpoint_selection
from kd_sensing.engine.training_state import TrainingState
from kd_sensing.utils.checkpoint import load_torch_payload, validate_checkpoint_publication


def _manager(tmp_path: Path) -> CheckpointManager:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = {
        "experiment": {"task": "fusion"},
        "training": {"epochs": 2},
        "model": {
            "primary": {
                "type": "u_mask_beam_jepa",
                "modalities": ["image", "radar", "gps", "lidar"],
            }
        },
        "output": {"run_name": "test"},
    }
    return CheckpointManager(
        cfg=cfg,
        run_dir=tmp_path,
        primary_model=model,
        optimizer=optimizer,
        scheduler=None,
        split_metadata={},
        normalization_artifacts={},
        objective_metadata={"name": "beam"},
        dataloaders={},
        grad_scaler=None,
        extensions=[],
        extension_states=[],
    )


def test_validation_best_checkpoint_records_selection_and_integrity(tmp_path: Path) -> None:
    path = _manager(tmp_path).save_best_checkpoint(
        state=TrainingState(), epoch=2, val_loss=0.25
    )
    payload = load_torch_payload(path)
    metadata = validate_checkpoint_publication(path, payload=payload)
    assert path.name == "best.pth"
    assert payload["checkpoint_role"] == "validation_best"
    assert payload["selection"] == {
        "metric": "validation_loss",
        "mode": "min",
        "value": pytest.approx(0.25),
        "epoch": 3,
    }
    assert metadata["checkpoint_policy"] == "best_validation_loss"
    assert metadata["integrity_verified"] is True


def test_checkpoint_selection_defaults_to_last_and_fails_closed() -> None:
    assert _checkpoint_selection({}) == "last"
    assert _checkpoint_selection({"checkpoint_selection": "best_validation_loss"}) == "best_validation_loss"
    with pytest.raises(ValueError, match="checkpoint_selection"):
        _checkpoint_selection({"checkpoint_selection": "test_loss"})
