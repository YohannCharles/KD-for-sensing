import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
from kd_sensing.config import load_config
from kd_sensing.config.io import dump_config, safe_load_yaml
from kd_sensing.cli.preprocess import _apply_scene_override_to_sequence_preprocess
from kd_sensing.data.beam_soft_targets import beam_power_to_distribution, gaussian_beam_distribution
import kd_sensing.data.datasets.deepsense6g as deepsense6g_module
import kd_sensing.data.datasets.deepsense6g_targets as deepsense6g_targets
import kd_sensing.data.transform_ops.io as io_transforms
import kd_sensing.data.transform_ops.lidar as lidar_transforms
import kd_sensing.preprocessing.lidar as lidar_preprocessing
from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset
from kd_sensing.data.layouts import (
    deepsense6g_scene_layout,
    mmw_condition_layout,
    physical_labels_cache_root,
    runtime_cache_root,
)
from kd_sensing.data.samples import create_samples
from kd_sensing.data.scenes import retarget_deepsense_dataset_config
from kd_sensing.losses.beam import FocalLoss, SoftTargetCrossEntropyLoss
from kd_sensing.engine.batch import prepare_fusion_inputs, prepare_labels, prepare_soft_beam_targets
from kd_sensing.engine.batch_step import BatchStepRunner
from kd_sensing.engine.cache_policy import apply_cache_policy
from kd_sensing.engine.data_factory import (
    build_dataloader,
    build_dataloaders,
    build_dataset,
    build_dataloader_kwargs,
    build_protocol_split_datasets,
    shutdown_dataloader_workers,
)
from kd_sensing.engine.epoch_subsampling import EpochSubsampleSampler
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.engine.training_extensions import ExtensionContext, NoOpTrainingExtension
from kd_sensing.engine.model_output import adapt_model_output, select_prediction_slots
from kd_sensing.engine.runtime import resolve_amp_settings, transfer_non_blocking
from kd_sensing.engine.evaluator import _evaluation_split_protocol_report
from kd_sensing.engine.run_metadata import dataset_run_metadata, prediction_setup_metadata, throughput_run_metadata
from kd_sensing.engine.training_metrics import training_outputs_payload
from kd_sensing.engine.trainer import (
    _configure_early_stopping,
    _early_stopping_improved,
    _early_stopping_min_epoch,
    _early_stopping_metric_value,
    _validate_early_stopping_source_available,
    _write_tensorboard_scalars,
    _write_tensorboard_startup_scalars,
    create_eval_run_dir,
    create_run_dir,
    train,
)
from kd_sensing.preprocessing.sequences import generate_sequence_data
from kd_sensing.utils.artifact_registry import (
    archive_best_checkpoint,
    find_registry_checkpoint,
    load_checkpoint_metadata,
    resolve_evaluation_checkpoint,
    write_sidecar,
)
from kd_sensing.utils.checkpoint import save_checkpoint

def test_save_checkpoint_replaces_existing_file_without_temp_leftover(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"

    save_checkpoint({"epoch": 1}, checkpoint_dir, "last.pth")
    save_checkpoint({"epoch": 2}, checkpoint_dir, "last.pth")

    assert torch.load(checkpoint_dir / "last.pth", map_location="cpu", weights_only=False)["epoch"] == 2
    assert not list(checkpoint_dir.glob(".last.pth.tmp-*"))


@pytest.mark.parametrize(
    ("raw_metric", "expected_metric", "expected_mode"),
    [
        ("val_adba", "val_adba", "max"),
        ("dba", "val_adba", "max"),
        ("dba/val_adba", "val_adba", "max"),
        ("beam/val_adba", "val_adba", "max"),
        ("top1_val_acc", "val_acc", "max"),
        ("accuracy/val", "val_acc", "max"),
        ("beam/accuracy_val", "val_acc", "max"),
        ("beam/val_top1", "val_acc", "max"),
        ("val/acc_top1", "val_acc", "max"),
        ("val_loss", "val_loss", "min"),
    ],
)
def test_early_stopping_metric_aliases_and_default_direction(raw_metric: str, expected_metric: str, expected_mode: str):
    training_cfg = {"early_stopping_metric": raw_metric}

    metric, mode = _configure_early_stopping(training_cfg)

    assert metric == expected_metric
    assert mode == expected_mode
    assert training_cfg["early_stopping_metric"] == expected_metric
    assert training_cfg["early_stopping_mode"] == expected_mode


def test_early_stopping_improvement_direction_and_missing_dba_error():
    assert _early_stopping_improved(0.91, 0.90, mode="max", min_delta=0.001) is True
    assert _early_stopping_improved(0.9005, 0.90, mode="max", min_delta=0.001) is False
    assert _early_stopping_improved(0.49, 0.50, mode="min", min_delta=0.001) is True
    assert _early_stopping_improved(0.4995, 0.50, mode="min", min_delta=0.001) is False
    assert _early_stopping_metric_value({"val_acc": 0.25}, "val_acc") == 0.25

    with pytest.raises(ValueError, match="val_adba.*not available"):
        _early_stopping_metric_value({"val_acc": 0.25}, "val_adba")
    with pytest.raises(ValueError, match="DBA/ADBA"):
        _validate_early_stopping_source_available({"loss": 1.0, "topk": {}}, "val_adba")
    with pytest.raises(ValueError, match="val_position_rmse.*not available.*Available metrics"):
        _validate_early_stopping_source_available(
            {"loss": 1.0, "topk": {"1": [0.25]}, "dba": [0.7], "objective": {"name": "beam"}},
            "val_position_rmse",
        )


@pytest.mark.parametrize(("total_epochs", "expected_min_epoch"), [(0, 0), (1, 1), (2, 1), (5, 3), (100, 50)])
def test_early_stopping_min_epoch_uses_half_target_epochs(total_epochs: int, expected_min_epoch: int):
    assert _early_stopping_min_epoch(total_epochs) == expected_min_epoch


def test_train_early_stopping_waits_until_half_target_epochs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = load_config(
        ROOT / "configs/gps/lightweight.yaml",
        [
            "experiment.device=cpu",
            "data.dataset.type=synthetic",
            "data.dataset.length=2",
            "data.dataset.seed=19",
            "data.validation_from_train.enabled=true",
            "data.validation_from_train.fraction=0.5",
            "data.validation_from_train.seed=19",
            "data.dataloader.train_batch_size=1",
            "data.dataloader.test_batch_size=1",
            "data.dataloader.num_workers=0",
            "training.epochs=5",
            "training.patience=1",
            "training.use_early_stopping=true",
            "training.early_stopping_metric=val_loss",
            "training.early_stopping_mode=min",
            "scheduler.type=none",
            "output.run_name=early_stop_min_epoch",
            "output.progress.enabled=false",
            "output.tensorboard.enabled=false",
            f"output.dir={tmp_path}",
            "output.overwrite=true",
            "checkpoint.registry.enabled=false",
        ],
    )

    def constant_validation_metrics(*_args, **_kwargs) -> dict:
        return {
            "loss": 1.0,
            "topk": {"1": [0.0], "3": [0.0], "5": [0.0]},
            "total": [1],
            "dba": [0.0],
            "available_metrics": ["val_loss", "val_acc", "val_adba"],
            "objective": {"name": "beam"},
        }

    monkeypatch.setattr("kd_sensing.engine.trainer.validate", constant_validation_metrics)

    result = train(cfg)

    assert [epoch_log["epoch"] for epoch_log in result["epoch_logs"]] == [1, 2, 3]
    assert result["epoch_logs"][-1]["epochs_without_improvement"] >= cfg["training"]["patience"]
    assert result["final_test_metrics"]["evaluation_split"] == "test"
    assert result["final_test_metrics"]["model_selection_split"] == "validation"
    assert result["split_metadata"]["validation"]["selection_split_source"] == "train"


def test_training_validation_interval_skips_intermediate_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = load_config(
        ROOT / "configs/gps/lightweight.yaml",
        [
            "experiment.device=cpu",
            "data.dataset.type=synthetic",
            "data.dataset.length=2",
            "data.dataloader.train_batch_size=1",
            "data.dataloader.test_batch_size=1",
            "data.dataloader.num_workers=0",
            "training.epochs=5",
            "training.use_early_stopping=false",
            "training.validation.interval_epochs=3",
            "scheduler.type=none",
            "output.run_name=validation_interval",
            "output.progress.enabled=false",
            "output.tensorboard.enabled=false",
            f"output.dir={tmp_path}",
            "output.overwrite=true",
            "checkpoint.registry.enabled=false",
        ],
    )
    calls = 0

    def constant_validation_metrics(*_args, **_kwargs) -> dict:
        nonlocal calls
        calls += 1
        return {
            "loss": float(calls),
            "topk": {"1": [0.0], "3": [0.0], "5": [0.0]},
            "total": [1],
            "dba": [0.0],
            "available_metrics": ["val_loss", "val_acc", "val_adba"],
            "objective": {"name": "beam"},
        }

    monkeypatch.setattr("kd_sensing.engine.trainer.validate", constant_validation_metrics)

    result = train(cfg)

    assert calls == 3
    assert [log["validation_ran"] for log in result["epoch_logs"]] == [True, False, True, False, True]
    assert [log["val_loss"] for log in result["epoch_logs"]] == [1.0, 1.0, 2.0, 2.0, 3.0]


def test_train_io_characterization_history_checkpoint_and_final_config(tmp_path: Path):
    cfg = load_config(
        ROOT / "configs/gps/lightweight.yaml",
        [
            "experiment.device=cpu",
            "data.dataset.type=synthetic",
            "data.dataset.length=2",
            "data.dataset.seed=13",
            "data.dataloader.train_batch_size=1",
            "data.dataloader.test_batch_size=1",
            "data.dataloader.num_workers=0",
            "training.epochs=1",
            "scheduler.type=none",
            "output.run_name=trainer_characterization",
            "output.progress.enabled=false",
            "output.tensorboard.enabled=false",
            f"output.dir={tmp_path}",
            "output.overwrite=true",
            "checkpoint.registry.enabled=false",
        ],
    )

    result = train(cfg)
    run_dir = Path(result["run_dir"])
    history = result["history"]
    epoch_log = result["epoch_logs"][0]
    checkpoint = torch.load(run_dir / "checkpoints" / "last.pth", map_location="cpu")
    final_cfg = safe_load_yaml((run_dir / "final_config.yaml").read_text(encoding="utf-8"))
    train_log = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))
    outputs = np.load(run_dir / "training_outputs.npz")

    assert set(history) == {
        "train_loss",
        "train_task_loss",
        "train_objective_loss",
        "train_beam_soft_loss",
        "train_unimodal_loss",
        "train_occlusion_loss",
        "train_position_loss",
        "train_multitask_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_atop3",
        "val_atop5",
        "val_adba",
        "val_occlusion_accuracy",
        "val_occlusion_blocked_f1",
        "val_position_rmse",
        "val_position_mae",
        "val_multitask_loss",
        "val_primary_metric",
        "learning_rates",
    }
    assert {
        "epoch",
        "total_epochs",
        "train_batches",
        "train_loss",
        "train_task_loss",
        "train_objective_loss",
        "train_beam_soft_loss",
        "train_unimodal_loss",
        "train_occlusion_loss",
        "train_position_loss",
        "train_multitask_loss",
        "loss/occlusion",
        "loss/position",
        "loss/multitask_total",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_atop3",
        "val_atop5",
        "val_adba",
        "val_occlusion_accuracy",
        "val_occlusion_blocked_f1",
        "val_position_rmse",
        "val_position_mae",
        "val_multitask_loss",
        "val_primary_metric",
        "early_stopping_metric",
        "early_stopping_mode",
        "early_stopping_value",
        "early_stopping_improved",
        "best_early_stopping_value",
        "best_early_stopping_epoch",
        "epochs_without_improvement",
        "learning_rate",
        "optimizer/lr/main",
        "optimizer/params/main",
    } <= set(epoch_log)
    assert "train_distill_loss" not in history
    assert "train_distill_loss" not in epoch_log
    assert {
        "epoch",
        "state_dict",
        "optimizer",
        "scheduler",
        "test_loss",
        "best_val_loss",
        "best_val_top1",
        "best_top1_epoch",
        "early_stopping_metric",
        "early_stopping_mode",
        "best_early_stopping_value",
        "best_early_stopping_epoch",
        "epochs_without_improvement",
        "normalization_artifacts",
        "checkpoint_registry",
    } <= set(checkpoint)
    assert checkpoint["early_stopping_metric"] == "val_adba"
    assert checkpoint["early_stopping_mode"] == "max"
    assert checkpoint["best_early_stopping_epoch"] == 1
    assert result["early_stopping"]["metric"] == "val_adba"
    assert result["early_stopping"]["mode"] == "max"
    assert final_cfg["runtime"]["run_dir"] == str(run_dir)
    assert final_cfg["runtime"]["output_overwrite"] is True
    assert result["final_test_metrics"]["evaluation_split"] == "test"
    assert result["final_test_metrics"]["model_selection_split"] == "test"
    assert final_cfg["runtime"]["final_test_metrics"]["evaluation_split"] == "test"
    assert train_log["final_test_metrics"]["evaluation_split"] == "test"
    assert final_cfg["training"]["early_stopping_metric"] == "val_adba"
    assert final_cfg["training"]["early_stopping_mode"] == "max"
    assert final_cfg["runtime"]["early_stopping"]["metric"] == "val_adba"
    assert final_cfg["runtime"]["prediction_objective"]["loss_weights"] == {
        "beam": 1.0,
        "occlusion": 1.0,
        "position": 0.01,
    }
    objective_meta = final_cfg["runtime"]["prediction_objective"]
    assert {
        "name",
        "primary_loss",
        "primary_metric",
        "primary_metric_mode",
        "available_metrics",
        "metric_aliases",
        "metric_modes",
        "history_fields",
        "tensorboard_scalars",
        "enabled_targets",
        "enabled_heads",
    } <= set(objective_meta)
    assert objective_meta["name"] == "beam"
    assert objective_meta["primary_metric"] == "val_adba"
    assert "adba" in objective_meta["metric_aliases"]
    assert "val_primary_metric" in objective_meta["history_fields"]
    assert {"tag": "objective/val_primary_metric", "history_key": "val_primary_metric"} in objective_meta[
        "tensorboard_scalars"
    ]
    assert train_log["prediction_objective"] == objective_meta
    assert epoch_log["val_occlusion_accuracy"] is None
    assert epoch_log["val_occlusion_blocked_f1"] is None
    assert epoch_log["val_position_rmse"] is None
    assert epoch_log["val_position_mae"] is None
    assert epoch_log["val_multitask_loss"] is None
    assert "val_occlusion_blocked_f1" not in epoch_log["validation_metrics"]
    assert "val_position_rmse" not in epoch_log["validation_metrics"]
    assert np.isnan(outputs["val_occlusion_blocked_f1"][0])
    assert np.isnan(outputs["val_position_rmse"][0])
    assert np.isnan(outputs["val_multitask_loss"][0])
    assert "splits" in final_cfg["runtime"]
    assert "normalization_artifacts" in final_cfg["runtime"]
    assert "throughput" in final_cfg["runtime"]
    assert final_cfg["runtime"]["throughput"]["progress"]["enabled"] is False
    assert (run_dir / "training_outputs.npz").exists()
    assert (run_dir / "checkpoints" / "last.pth").exists()


def test_train_epoch_subsampling_smoke_logs_metadata(tmp_path: Path):
    cfg = load_config(
        ROOT / "configs/gps/lightweight.yaml",
        [
            "experiment.device=cpu",
            "data.dataset.type=synthetic",
            "data.dataset.length=4",
            "data.dataset.seed=23",
            "data.dataloader.train_batch_size=1",
            "data.dataloader.test_batch_size=1",
            "data.dataloader.num_workers=0",
            "training.epochs=1",
            "training.epoch_subsampling.enabled=true",
            "training.epoch_subsampling.num_samples=2",
            "training.epoch_subsampling.rotate_each_epoch=true",
            "scheduler.type=none",
            "output.run_name=trainer_epoch_subsampling",
            "output.progress.enabled=false",
            "output.tensorboard.enabled=false",
            f"output.dir={tmp_path}",
            "output.overwrite=true",
            "checkpoint.registry.enabled=false",
        ],
    )

    result = train(cfg)
    run_dir = Path(result["run_dir"])
    epoch_log = result["epoch_logs"][0]
    final_cfg = safe_load_yaml((run_dir / "final_config.yaml").read_text(encoding="utf-8"))
    train_log = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))

    assert epoch_log["train_batches"] == 2
    assert epoch_log["train_epoch_subsampling_enabled"] is True
    assert epoch_log["train_full_samples"] == 4
    assert epoch_log["train_effective_samples"] == 2
    assert epoch_log["train_sampler_epoch"] == 0
    assert epoch_log["train_epoch_subsampling"]["seed"] == cfg["experiment"]["seed"]
    assert result["split_metadata"]["train"]["num_samples"] == 4
    assert result["split_metadata"]["test"]["num_samples"] == 4
    assert result["split_metadata"]["train"]["epoch_subsampling"]["effective_train_samples"] == 2
    assert final_cfg["runtime"]["splits"]["train"]["epoch_subsampling"]["num_samples"] == 2
    assert final_cfg["runtime"]["throughput"]["epoch_subsampling"]["train"]["effective_train_samples"] == 2
    assert train_log["epoch_logs"][0]["train_effective_samples"] == 2
    assert (run_dir / "training_outputs.npz").exists()
    assert (run_dir / "checkpoints" / "last.pth").exists()


def test_csi_debug_training_writes_resolved_diff_startup_and_health_artifacts(tmp_path: Path):
    cfg = load_config(
        ROOT / "configs/csi/hardening_matrix/debug/A0_clone_generated.yaml",
        [
            "experiment.device=cpu",
            "data.dataset.type=synthetic",
            "data.dataset.length=2",
            "data.dataset.seed=31",
            "data.dataset.csi_shape=[8,4]",
            "data.dataloader.train_batch_size=1",
            "data.dataloader.test_batch_size=1",
            "data.dataloader.num_workers=0",
            "training.epochs=1",
            "training.use_early_stopping=false",
            "scheduler.type=none",
            "output.run_name=csi_debug_smoke",
            "output.progress.enabled=false",
            "output.tensorboard.enabled=false",
            f"output.dir={tmp_path}",
            "output.overwrite=true",
            "checkpoint.registry.enabled=false",
        ],
    )
    reference_path = tmp_path / "a0_reference.yaml"
    reference_cfg = deepcopy(cfg)
    reference_cfg["output"]["run_name"] = "different_identity"
    reference_cfg["experiment"]["seed"] = 999
    dump_config(reference_cfg, reference_path)
    cfg["debug"]["config_diff"]["reference"] = str(reference_path)

    result = train(cfg)
    run_dir = Path(result["run_dir"])
    train_log = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))
    config_diff = json.loads((run_dir / "config_diff.json").read_text(encoding="utf-8"))
    startup = json.loads((run_dir / "startup_summary.json").read_text(encoding="utf-8"))
    csi_records = json.loads((run_dir / "csi_first_batch_diagnostics.json").read_text(encoding="utf-8"))
    epoch_log = train_log["epoch_logs"][0]

    assert (run_dir / "resolved_config.yaml").exists()
    assert config_diff["parity_passed"] is True
    assert any(item["path"] == "experiment.seed" for item in config_diff["allowed_identity_differences"])
    assert startup["model"]["csi_encoder_type"] == "pilot_dual_view_csi"
    assert startup["model"]["use_internal_gru"] is True
    assert startup["parameters"]["modules"]["csi_encoder"]["trainable_params"] > 0
    assert train_log["startup_summary"]["parameters"]["total_params"] == startup["parameters"]["total_params"]
    assert train_log["pilot_noise_validity"]["valid"] is True
    assert train_log["pilot_noise_validity"]["reason"] == "pilot_noise_disabled"
    assert {record["source"] for record in csi_records} == {"train", "val"}
    assert csi_records[0]["complex"]["before_hardening"]["shape"] == [1, 10, 8, 4]
    assert epoch_log["grad_norm_csi_encoder"] > 0.0
    assert epoch_log["grad_norm_representation_core"] > 0.0
    assert epoch_log["grad_norm_beam_head"] > 0.0
    assert epoch_log["param_delta_csi_encoder"] > 0.0
    assert epoch_log["param_delta_representation_core"] > 0.0
    assert epoch_log["param_delta_beam_head"] > 0.0


class _FakeTensorboardWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        pass


def _tensorboard_history() -> dict:
    return {
        "train_loss": [1.0],
        "train_task_loss": [0.9],
        "train_objective_loss": [0.9],
        "train_distill_loss": [0.0],
        "val_loss": [0.8],
        "val_primary_metric": [0.8],
        "train_acc": [0.25],
        "val_acc": [0.2],
        "val_atop3": [0.3],
        "val_atop5": [0.4],
        "val_adba": [0.5],
        "learning_rates": [0.001],
        "train_beam_soft_loss": [],
        "train_unimodal_loss": [],
        "train_multitask_loss": [None],
        "val_multitask_loss": [None],
        "train_occlusion_loss": [0.7],
        "train_position_loss": [None],
        "val_occlusion_accuracy": [0.6],
        "val_occlusion_blocked_f1": [float("nan")],
        "val_position_rmse": [None],
        "val_position_mae": [None],
    }


def test_optional_objective_tensorboard_scalars_skip_inactive_values():
    writer = _FakeTensorboardWriter()
    history = _tensorboard_history()

    _write_tensorboard_scalars(writer, history, 1)

    tags = {tag for tag, _, _ in writer.scalars}
    assert "beam/accuracy_train" in tags
    assert "beam/accuracy_val" in tags
    assert "beam/val_atop3" in tags
    assert "beam/val_atop5" in tags
    assert "beam/val_adba" in tags
    assert "accuracy/train" not in tags
    assert "accuracy/val" not in tags
    assert "accuracy/val_atop3" not in tags
    assert "accuracy/val_atop5" not in tags
    assert "dba/val_adba" not in tags
    assert "occlusion/accuracy" in tags
    assert "occlusion/blocked_f1" not in tags
    assert "position/rmse" not in tags
    assert "position/mae" not in tags
    assert "loss/val_multitask_total" not in tags


def test_tensorboard_startup_scalars_make_new_event_non_empty():
    writer = _FakeTensorboardWriter()

    _write_tensorboard_startup_scalars(
        writer,
        {
            "data": {"batch_size": {"train": 16, "test": 16}},
            "optimization": {"max_epochs": 20},
            "parameters": {
                "total_params": 123,
                "trainable_params": 120,
                "modules": {"image_encoder": {"total_params": 10, "trainable_params": 10}},
            },
        },
    )

    tags = {tag for tag, _, _ in writer.scalars}
    assert "run/start" in tags
    assert "model/total_params" in tags
    assert "model/modules/image_encoder/total_params" in tags


@pytest.mark.parametrize("objective", ["occlusion", "position"])
def test_non_beam_objectives_do_not_write_beam_tensorboard_scalars(objective: str):
    writer = _FakeTensorboardWriter()
    history = _tensorboard_history()

    _write_tensorboard_scalars(writer, history, 1, objective=objective)

    tags = {tag for tag, _, _ in writer.scalars}
    assert "beam/accuracy_train" not in tags
    assert "beam/accuracy_val" not in tags
    assert "beam/val_atop3" not in tags
    assert "beam/val_atop5" not in tags
    assert "beam/val_adba" not in tags
    assert "accuracy/val" not in tags
    assert "dba/val_adba" not in tags


def test_current_selection_objectives_tensorboard_write_only_formal_tags():
    base = {
        "train_loss": [1.0],
        "train_task_loss": [0.9],
        "train_objective_loss": [0.9],
        "train_distill_loss": [0.0],
        "val_loss": [0.8],
        "val_primary_metric": [0.7],
        "learning_rates": [0.001],
        "train_beam_soft_loss": [],
        "train_unimodal_loss": [],
        "train_acc": [0.25],
        "val_beam_top1": [0.4],
        "val_beam_top3": [0.6],
        "val_beam_top5": [0.8],
        "val_beam_dba": [0.5],
        "train_los_loss": [0.3],
        "val_los_accuracy": [0.9],
        "val_los_f1": [0.85],
        "val_los_auc": [0.95],
        "train_link_quality_loss": [0.2],
        "val_link_mae": [1.0],
        "val_link_rmse": [1.2],
        "val_link_r2": [0.1],
    }

    cases = {
        "current_beam_selection": {"beam/val_top1", "beam/val_top3", "beam/val_top5", "beam/val_dba_current"},
        "current_los_classification": {"loss/los", "los/accuracy", "los/f1", "los/auc"},
        "current_link_quality": {"loss/link_quality", "link/mae", "link/rmse", "link/r2"},
    }
    forbidden = {
        "beam/val_top1",
        "beam/val_dba_current",
        "los/accuracy",
        "link/mae",
        "beam/val_adba",
        "dba/val_adba",
    }
    for objective, expected in cases.items():
        writer = _FakeTensorboardWriter()
        _write_tensorboard_scalars(writer, base, 1, objective=objective)
        tags = {tag for tag, _, _ in writer.scalars}
        assert expected <= tags
        assert not ((forbidden - expected) & tags)


def test_tensorboard_legacy_accuracy_tags_restore_historical_scalars():
    writer = _FakeTensorboardWriter()
    history = _tensorboard_history()

    _write_tensorboard_scalars(
        writer,
        history,
        1,
        objective="occlusion",
        tensorboard_cfg={"legacy_accuracy_tags": True},
    )

    tags = {tag for tag, _, _ in writer.scalars}
    assert "beam/accuracy_val" not in tags
    assert "accuracy/train" in tags
    assert "accuracy/val" in tags
    assert "accuracy/val_atop3" in tags
    assert "accuracy/val_atop5" in tags
    assert "dba/val_adba" in tags


def test_training_outputs_payload_converts_inactive_optional_metrics_to_nan():
    payload = training_outputs_payload(
        {
            "train_loss": [1.0],
            "train_occlusion_loss": [None],
            "val_position_rmse": [None],
            "val_occlusion_blocked_f1": [0.5],
        },
        {
            "name": "beam",
            "primary_loss": "beam",
            "enabled_targets": ["beam"],
            "enabled_heads": ["beam"],
            "loss_weights": {"beam": 1.0, "occlusion": 1.0, "position": 0.01},
        },
        "val_adba",
        "max",
    )

    assert np.isnan(payload["train_occlusion_loss"][0])
    assert np.isnan(payload["val_position_rmse"][0])
    assert payload["val_occlusion_blocked_f1"][0] == pytest.approx(0.5)
    assert payload["loss_weights"].tolist() == [1.0, 1.0, 0.01]


def test_non_blocking_transfer_and_amp_cpu_fallback():
    cfg = {
        "training": {
            "transfer": {"non_blocking": True},
            "amp": {"enabled": True, "dtype": "float16", "grad_scaler": True},
        }
    }
    batch = {"gps": torch.randn(2, 8, 3)}

    enabled, dtype = resolve_amp_settings(cfg, torch.device("cpu"))
    fusion_inputs = prepare_fusion_inputs(
        batch,
        seq_length=8,
        num_pred=3,
        device=torch.device("cpu"),
        modalities=["gps"],
        non_blocking=transfer_non_blocking(cfg),
    )

    assert enabled is False
    assert dtype is torch.float16
    assert fusion_inputs["gps_batch"].shape == (2, 10, 3)


def test_builder_rejects_conflicting_dataset_flags():
    with pytest.raises(ValueError, match="use_lidar=true conflicts"):
        resolve_enabled_modalities(
            {
                "experiment": {"task": "gps"},
                "data": {"dataset": {"use_lidar": True}},
                "model": {"primary": {}},
            }
        )
    with pytest.raises(ValueError, match="must match"):
        resolve_enabled_modalities(
            {
                "experiment": {"task": "fusion"},
                "data": {"dataset": {}},
                "model": {"modalities": ["image"], "primary": {"modalities": ["radar"]}},
            }
        )
