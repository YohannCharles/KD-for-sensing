from pathlib import Path

import pytest
import torch

from kd_sensing.engine import trainer
from kd_sensing.engine.checkpointing import resolve_resume_checkpoint
from kd_sensing.engine.training_resume import (
    CHECKPOINT_SCHEMA_VERSION,
    _fingerprint,
    build_resume_contract,
    preflight_resume,
    validate_resume_contract,
    validate_resume_payload,
)
from kd_sensing.utils.checkpoint import CheckpointLoadError, publish_checkpoint


class _NormalizationRestored(Exception):
    pass


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


def test_resume_prefers_checkpoint_normalization_over_resolved_config(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.pth"
    checkpoint.touch()
    resume_metadata = {"normalization_artifacts": {"gps_scaler": "checkpoint.npz"}}
    cfg = {
        "training": {"resume": True},
        "output": {"run_name": "run"},
        "data": {"normalization_artifacts": {"gps_scaler": "resolved.npz"}},
    }

    monkeypatch.setattr(trainer, "configure_torch_runtime_threads", lambda _cfg: None)
    monkeypatch.setattr(trainer, "_print_mmw_split_binding", lambda _cfg: None)
    monkeypatch.setattr(trainer, "set_seed", lambda _seed: None)
    monkeypatch.setattr(trainer, "create_run_dir", lambda _cfg: tmp_path / "run")
    monkeypatch.setattr(trainer, "write_running_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer, "ArtifactWriter", lambda **_kwargs: object())
    monkeypatch.setattr(trainer, "resolve_resume_checkpoint", lambda *_args: checkpoint)
    monkeypatch.setattr(trainer, "load_checkpoint_metadata", lambda _path: resume_metadata)
    monkeypatch.setattr(trainer, "validate_normalization_artifact_fingerprint", lambda _cfg, metadata: None)

    def restore(metadata):
        assert metadata is resume_metadata
        raise _NormalizationRestored

    monkeypatch.setattr(trainer, "load_normalization_artifacts", restore)

    with pytest.raises(_NormalizationRestored):
        trainer._prepare_training_run_context(cfg)


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


def test_resume_contract_ignores_cli_config_provenance_path() -> None:
    base = {
        "runtime": {"cli_config_path": "/tmp/timing.yaml"},
        "training": {"epochs": 1},
        "model": {"primary": {"type": "u_mask_beam_jepa"}},
    }
    resumed = {
        **base,
        "runtime": {"cli_config_path": "/tmp/resume.yaml"},
        "training": {"epochs": 4, "resume": True},
    }

    validate_resume_contract(
        build_resume_contract(base, {"ids": [1]}, {"sha256": "a"}),
        build_resume_contract(resumed, {"ids": [1]}, {"sha256": "a"}),
        next_epoch=1,
    )


def test_resume_contract_projects_cli_path_from_older_recorded_contract() -> None:
    base = {"training": {"epochs": 1}, "model": {"primary": {"type": "u_mask_beam_jepa"}}}
    recorded = build_resume_contract(base, {"ids": [1]}, {"sha256": "a"})
    recorded["config"]["runtime"] = {"cli_config_path": "/tmp/timing.yaml"}
    recorded["config_sha256"] = _fingerprint(recorded["config"])
    resumed = build_resume_contract(
        {**base, "training": {"epochs": 4, "resume": True}},
        {"ids": [1]},
        {"sha256": "a"},
    )

    validate_resume_contract(recorded, resumed, next_epoch=1)
