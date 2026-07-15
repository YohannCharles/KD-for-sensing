from pathlib import Path

import pytest
import torch

from kd_sensing.utils.checkpoint import (
    CheckpointLoadError,
    load_model_state,
    load_torch_payload,
    publish_checkpoint,
    validate_checkpoint_publication,
)


def test_safe_checkpoint_loader_passes_weights_only_true(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"placeholder")
    calls = []

    def fake_load(path, **kwargs):
        calls.append((Path(path), dict(kwargs)))
        return {"state_dict": {"weight": torch.ones(1)}}

    monkeypatch.setattr(torch, "load", fake_load)

    payload = load_torch_payload(checkpoint)

    assert payload["state_dict"]["weight"].shape == (1,)
    assert calls == [(checkpoint, {"map_location": "cpu", "weights_only": True})]


def test_trusted_local_checkpoint_requires_explicit_existing_file(tmp_path: Path, monkeypatch):
    missing = tmp_path / "missing.pth"
    with pytest.raises(CheckpointLoadError, match="does not exist"):
        load_torch_payload(missing, trusted_local=True)

    checkpoint = tmp_path / "legacy.pth"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: kwargs)

    with pytest.warns(RuntimeWarning, match="unsafe pickle"):
        kwargs = load_torch_payload(checkpoint, trusted_local=True)

    assert kwargs["weights_only"] is False


def test_current_checkpoint_requires_complete_sidecar_and_matching_digest(tmp_path: Path):
    payload = {
        "checkpoint_schema_version": 1,
        "checkpoint_role": "last",
        "state_dict": {"weight": torch.ones(1, 1), "bias": torch.zeros(1)},
    }
    checkpoint, metadata = publish_checkpoint(
        payload,
        tmp_path,
        "last.pth",
        metadata={"selection": {}},
    )

    assert metadata["publish_complete"] is True
    assert metadata["checkpoint_size_bytes"] == checkpoint.stat().st_size
    assert len(metadata["checkpoint_sha256"]) == 64
    validate_checkpoint_publication(checkpoint)

    with checkpoint.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(CheckpointLoadError, match="digest|size"):
        validate_checkpoint_publication(checkpoint)
    with pytest.raises(CheckpointLoadError, match="digest|size"):
        load_model_state(checkpoint, torch.nn.Linear(1, 1))


def test_current_checkpoint_without_sidecar_is_rejected_but_legacy_is_readable(tmp_path: Path):
    current = tmp_path / "current.pth"
    legacy = tmp_path / "legacy.pth"
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "checkpoint_role": "last",
            "state_dict": {"weight": torch.ones(1, 1), "bias": torch.zeros(1)},
        },
        current,
    )
    torch.save({"state_dict": {"weight": torch.ones(1, 1), "bias": torch.zeros(1)}}, legacy)

    with pytest.raises(CheckpointLoadError, match="sidecar"):
        load_model_state(current, torch.nn.Linear(1, 1))
    loaded = load_model_state(legacy, torch.nn.Linear(1, 1))
    assert loaded["checkpoint"]["state_dict"]["weight"].shape == (1, 1)
