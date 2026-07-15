import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kd_sensing.engine.checkpointing import CheckpointManager, resolve_resume_checkpoint
from kd_sensing.engine.training_extensions import TrainingExtension
from kd_sensing.engine.training_resume import (
    CHECKPOINT_SCHEMA_VERSION,
    build_resume_contract,
    capture_runtime_state,
    migrate_legacy_resume_payload,
    preflight_resume,
    resolve_selected_checkpoint,
    restore_runtime_state,
    validate_resume_contract,
    validate_resume_payload,
)
from kd_sensing.engine.training_state import TrainingState
from kd_sensing.utils.checkpoint import (
    CheckpointLoadError,
    load_torch_payload,
    publish_checkpoint,
    validate_checkpoint_publication,
)


class _StatefulExtension(TrainingExtension):
    name = "fixture"
    stateless = False

    def state_dict(self, state):
        return {"counter": int(state["counter"])}

    def load_state_dict(self, state, payload):
        state["counter"] = int(payload["counter"])


class _Scaler:
    def __init__(self, scale: float = 1.0, *, enabled: bool = True):
        self.scale = scale
        self.enabled = enabled

    def is_enabled(self):
        return self.enabled

    def state_dict(self):
        return {"scale": self.scale}

    def load_state_dict(self, payload):
        self.scale = float(payload["scale"])


def _current_payload(contract: dict, runtime_state: dict | None = None) -> dict:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": "last",
        "state_dict": {"weight": torch.ones(1)},
        "optimizer": {"state": {}, "param_groups": []},
        "scheduler": None,
        "epoch": 2,
        "runtime_state": runtime_state or {
            "runtime_state_schema_version": 1,
            "rng": {},
            "dataloaders": {},
            "grad_scaler": {"enabled": False, "state": {}},
            "extensions": [],
            "training_state": {},
        },
        "resume_contract": contract,
        "selection": {
            "metric": None,
            "mode": None,
            "value": None,
            "selected_epoch": 2,
            "source_run": ".",
            "final_test_candidate": False,
        },
        "selection_catalog": {},
    }


def test_resume_true_requires_existing_last_checkpoint(tmp_path: Path):
    cfg = {"training": {"resume": True}, "output": {"run_name": "existing"}}

    with pytest.raises(FileNotFoundError, match=r"checkpoints/last\.pth"):
        resolve_resume_checkpoint(cfg, tmp_path / "existing")


@pytest.mark.parametrize("missing", ["optimizer", "scheduler", "epoch"])
def test_resume_payload_requires_training_role_fields(tmp_path: Path, missing: str):
    payload = _current_payload(build_resume_contract({}, {}, {}))
    payload.pop(missing)

    with pytest.raises(CheckpointLoadError, match=rf"resume.*{missing}"):
        validate_resume_payload(
            payload,
            path=tmp_path / "last.pth",
            scheduler_enabled=False,
        )


def test_resume_payload_rejects_scheduler_topology_mismatch(tmp_path: Path):
    payload = _current_payload(build_resume_contract({}, {}, {}))

    with pytest.raises(CheckpointLoadError, match="scheduler"):
        validate_resume_payload(payload, path=tmp_path / "last.pth", scheduler_enabled=True)


def test_legacy_migration_is_explicit_and_scopes_test_loss_alias(tmp_path: Path):
    legacy = {
        "state_dict": {"weight": torch.ones(1)},
        "optimizer": {},
        "scheduler": None,
        "epoch": 3,
        "test_loss": 0.25,
    }

    with pytest.warns(RuntimeWarning, match="legacy"):
        migrated = migrate_legacy_resume_payload(
            legacy,
            path=tmp_path / "legacy.pth",
            scheduler_enabled=False,
        )

    assert migrated["best_val_loss"] == pytest.approx(0.25)
    assert migrated["resume_migration"]["trajectory_equivalence"] is False
    assert migrated["resume_migration"]["source_schema"] == "legacy-unversioned"


def test_runtime_state_round_trip_restores_rng_loader_scaler_extension_and_history():
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    loader_generator = torch.Generator().manual_seed(29)
    loader = SimpleNamespace(generator=loader_generator, sampler=SimpleNamespace(generator=None))
    scaler = _Scaler(128.0)
    extension = _StatefulExtension()
    extension_state = {"counter": 7}
    training_state = TrainingState(
        start_epoch=2,
        history={"train_loss": [2.0, 1.0]},
        epoch_logs=[{"epoch": 1}, {"epoch": 2}],
    )

    runtime = capture_runtime_state(
        dataloaders={"train": loader},
        grad_scaler=scaler,
        extensions=[extension],
        extension_states=[extension_state],
        training_state=training_state,
    )
    expected = (
        random.random(),
        float(np.random.rand()),
        torch.rand(2),
        torch.rand(2, generator=loader_generator),
    )

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    loader_generator.manual_seed(99)
    scaler.scale = 1.0
    extension_state["counter"] = 0
    training_state.history.clear()
    training_state.epoch_logs.clear()

    restore_runtime_state(
        runtime,
        dataloaders={"train": loader},
        grad_scaler=scaler,
        extensions=[extension],
        extension_states=[extension_state],
        training_state=training_state,
    )

    actual = (
        random.random(),
        float(np.random.rand()),
        torch.rand(2),
        torch.rand(2, generator=loader_generator),
    )
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])
    assert scaler.scale == pytest.approx(128.0)
    assert extension_state == {"counter": 7}
    assert training_state.history == {"train_loss": [2.0, 1.0]}
    assert training_state.epoch_logs == [{"epoch": 1}, {"epoch": 2}]


def test_resume_contract_allows_only_runtime_controls_and_epoch_increase():
    recorded_cfg = {
        "model": {"primary": {"name": "fixture"}},
        "training": {"epochs": 2, "resume": False, "seed": 4, "timing": {"enabled": False}},
        "output": {"dir": "old", "run_name": "source", "tensorboard": {"enabled": True}},
    }
    current_cfg = {
        "model": {"primary": {"name": "fixture"}},
        "training": {"epochs": 4, "resume": "source/last.pth", "seed": 4, "timing": {"enabled": True}},
        "output": {"dir": "new", "run_name": "target", "tensorboard": {"enabled": False}},
    }
    recorded = build_resume_contract(recorded_cfg, {"ids": [1, 2]}, {"sha256": "abc"})
    current = build_resume_contract(current_cfg, {"ids": [1, 2]}, {"sha256": "abc"})

    validate_resume_contract(recorded, current, next_epoch=2)

    current_cfg["training"]["seed"] = 5
    incompatible = build_resume_contract(current_cfg, {"ids": [1, 2]}, {"sha256": "abc"})
    with pytest.raises(CheckpointLoadError, match=r"training\.seed"):
        validate_resume_contract(recorded, incompatible, next_epoch=2)


def test_preflight_current_checkpoint_validates_complete_marker(tmp_path: Path):
    run_dir = tmp_path / "source"
    contract = build_resume_contract(
        {"training": {"epochs": 2}, "output": {"run_name": "source"}},
        {},
        {},
    )
    payload = _current_payload(contract)
    checkpoint, _ = publish_checkpoint(
        payload,
        run_dir / "checkpoints",
        "last.pth",
        metadata={"selection": payload["selection"]},
    )
    cfg = {
        "training": {"resume": str(checkpoint), "epochs": 2},
        "output": {"run_name": "source"},
    }

    plan = preflight_resume(cfg, run_dir, scheduler_enabled=False)

    assert plan is not None
    assert plan.path == checkpoint
    assert plan.next_epoch == 2
    assert plan.schema == "current"
    assert plan.trajectory_equivalence is True


def test_checkpoint_manager_publishes_per_file_selection_provenance(tmp_path: Path):
    cfg = {
        "experiment": {"task": "beam_prediction"},
        "training": {"min_delta": 0.0, "epochs": 1},
        "checkpoint": {
            "selection_metric": "avg_missing_top1",
            "registry": {"enabled": False},
        },
        "output": {"run_name": "fixture"},
    }
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    state = TrainingState(history={"val_loss": [0.5]}, epoch_logs=[])
    epoch_log = {"epoch": 1, "val_loss": 0.5, "val_acc": 0.7, "avg_missing_top1": 0.6}
    state.epoch_logs.append(epoch_log)
    manager = CheckpointManager(
        cfg=cfg,
        run_dir=tmp_path,
        primary_model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        split_metadata={},
        normalization_artifacts={},
        objective_metadata={"name": "beam"},
        early_stopping_metric="val_loss",
        early_stopping_mode="min",
    )

    manager.update_best_checkpoints(
        state=state,
        epoch=0,
        epoch_log=epoch_log,
        val_loss=0.5,
        val_acc=0.7,
        train_dataset=SimpleNamespace(enabled_modalities=[]),
    )
    manager.save_last_checkpoint(state=state, epoch=0, val_loss=0.5)

    expected = {
        "best.pth": ("objective_best", "val_loss"),
        "best_top1.pth": ("top1_best", "val_acc_top1"),
        "best_avg_missing_top1.pth": ("selection_best", "avg_missing_top1"),
        "last.pth": ("last", None),
    }
    for filename, (role, metric) in expected.items():
        path = tmp_path / "checkpoints" / filename
        payload = load_torch_payload(path)
        metadata = validate_checkpoint_publication(path, payload=payload)
        assert payload["checkpoint_role"] == role
        assert payload["selection"]["metric"] == metric
        assert "validation_loss" in payload
        assert "test_loss" not in payload
        assert metadata["checkpoint_role"] == role
        assert metadata["selection"]["metric"] == metric
        assert payload["best_val_top1"] == pytest.approx(0.7)
        assert payload["best_selection_value"] == pytest.approx(0.6)


def test_selected_checkpoint_resolver_honors_default_custom_last_and_cross_run(tmp_path: Path):
    source_run = tmp_path / "source"
    contract = build_resume_contract({}, {}, {})
    catalog = {}
    for role, filename, metric in (
        ("objective_best", "best.pth", "val_loss"),
        ("top1_best", "best_top1.pth", "val_acc_top1"),
        ("selection_best", "best_avg_missing_top1.pth", "avg_missing_top1"),
        ("last", "last.pth", None),
    ):
        selection = {
            "metric": metric,
            "mode": "min" if metric == "val_loss" else ("max" if metric else None),
            "value": 0.5 if metric else None,
            "selected_epoch": 2,
            "source_run": str(source_run),
            "final_test_candidate": role != "last",
        }
        payload = _current_payload(contract)
        payload["checkpoint_role"] = role
        payload["selection"] = selection
        path, metadata = publish_checkpoint(
            payload,
            source_run / "checkpoints",
            filename,
            metadata={"selection": selection},
        )
        key = metric if role == "selection_best" else role
        catalog[key] = {
            "path": str(path),
            "checkpoint_role": role,
            "selection": selection,
            "checkpoint_sha256": metadata["checkpoint_sha256"],
        }

    default = resolve_selected_checkpoint({}, tmp_path / "target", catalog=catalog)
    custom = resolve_selected_checkpoint(
        {"checkpoint": {"selection_metric": "avg_missing_top1"}},
        tmp_path / "target",
        catalog=catalog,
    )
    top1 = resolve_selected_checkpoint(
        {"checkpoint": {"selection_metric": "val_acc"}},
        tmp_path / "target",
        catalog=catalog,
    )
    last = resolve_selected_checkpoint(
        {"training": {"model_selection": {"enabled": False}}},
        tmp_path / "target",
        catalog=catalog,
    )

    assert default.path.name == "best.pth"
    assert custom.path.name == "best_avg_missing_top1.pth"
    assert top1.path.name == "best_top1.pth"
    assert last.path.name == "last.pth"
    assert default.source_run == source_run
