from pathlib import Path

import pytest
import torch

from kd_sensing.utils.checkpoint import CheckpointLoadError, load_model_state, load_torch_payload, publish_checkpoint, validate_checkpoint_publication


def test_checkpoint_loader_uses_weights_only(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"placeholder")
    calls = []

    def fake_load(path, **kwargs):
        calls.append((Path(path), kwargs))
        return {"state_dict": {"weight": torch.ones(1)}}

    monkeypatch.setattr(torch, "load", fake_load)

    assert load_torch_payload(checkpoint)["state_dict"]["weight"].shape == (1,)
    assert calls == [(checkpoint, {"map_location": "cpu", "weights_only": True})]


def test_current_checkpoint_requires_an_intact_publication_sidecar(tmp_path: Path) -> None:
    payload = {
        "checkpoint_schema_version": 1,
        "checkpoint_role": "last",
        "state_dict": {"weight": torch.ones(1, 1), "bias": torch.zeros(1)},
    }
    checkpoint, metadata = publish_checkpoint(payload, tmp_path, "last.pth", metadata={"selection": {}})

    assert metadata["publish_complete"] is True
    validate_checkpoint_publication(checkpoint)
    with checkpoint.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(CheckpointLoadError, match="digest|size"):
        validate_checkpoint_publication(checkpoint)
    with pytest.raises(CheckpointLoadError, match="digest|size"):
        load_model_state(checkpoint, torch.nn.Linear(1, 1))
