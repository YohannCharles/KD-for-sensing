from pathlib import Path

import pytest
import torch

from kd_sensing.engine.checkpointing import resolve_resume_checkpoint
from kd_sensing.engine.training_resume import (
    CHECKPOINT_SCHEMA_VERSION,
    build_resume_contract,
    preflight_resume,
    validate_resume_contract,
    validate_resume_payload,
)
from kd_sensing.utils.checkpoint import CheckpointLoadError, publish_checkpoint


def _payload(contract: dict) -> dict:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": "last",
        "state_dict": {"weight": torch.ones(1)},
        "optimizer": {"state": {}, "param_groups": []},
        "scheduler": None,
        "epoch": 2,
        "runtime_state": {
            "runtime_state_schema_version": 1,
            "rng": {},
            "dataloaders": {},
            "grad_scaler": {"enabled": False, "state": {}},
            "extensions": [],
            "training_state": {},
        },
        "resume_contract": contract,
    }


def test_resume_requires_the_current_run_last_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"checkpoints/last\.pth"):
        resolve_resume_checkpoint({"training": {"resume": True}, "output": {"run_name": "U0"}}, tmp_path / "U0")


def test_current_resume_payload_and_publication_preflight(tmp_path: Path) -> None:
    cfg = {"training": {"epochs": 4}, "model": {"primary": {"type": "u_mask_beam_jepa"}}, "output": {"run_name": "U0"}}
    contract = build_resume_contract(cfg, {}, {})
    payload = _payload(contract)
    checkpoint, _ = publish_checkpoint(payload, tmp_path / "U0" / "checkpoints", "last.pth")

    assert validate_resume_payload(payload, path=checkpoint, scheduler_enabled=False)["epoch"] == 2
    plan = preflight_resume({**cfg, "training": {"epochs": 4, "resume": str(checkpoint)}}, tmp_path / "U0", scheduler_enabled=False)
    assert plan is not None and plan.path == checkpoint and plan.trajectory_equivalence is True


def test_resume_contract_rejects_model_drift() -> None:
    base = {"training": {"epochs": 2}, "model": {"primary": {"type": "u_mask_beam_jepa"}}}
    recorded = build_resume_contract(base, {"ids": [1]}, {"sha256": "a"})
    changed = build_resume_contract(
        {"training": {"epochs": 3}, "model": {"primary": {"type": "modular_sequence"}}},
        {"ids": [1]},
        {"sha256": "a"},
    )

    with pytest.raises(CheckpointLoadError, match=r"model\.primary\.type"):
        validate_resume_contract(recorded, changed, next_epoch=2)
