from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.engine.artifacts import final_config_with_runtime  # noqa: E402
from kd_sensing.engine.debug_diagnostics import build_startup_summary  # noqa: E402
from kd_sensing.engine.model_output import ModelOutput  # noqa: E402
from kd_sensing.engine.prediction_objectives import (  # noqa: E402
    PredictionTargets,
    compute_prediction_loss,
    multitask_loss_weights,
    normalize_objective_metric,
    objective_available_metrics,
    objective_history_fields,
    objective_metric_mode,
    objective_runtime_metadata,
    objective_spec,
    objective_tensorboard_scalars,
    resolve_prediction_objective,
)
from kd_sensing.engine.trainer import train  # noqa: E402


def test_objective_config_defaults_validation_and_legacy_compatibility():
    beam_cfg = load_config(ROOT / "configs/fusion/all_modalities_no_kd.yaml")
    occlusion_cfg = load_config(ROOT / "configs/fusion/strong_only_occlusion_no_kd.yaml")
    position_cfg = load_config(ROOT / "configs/fusion/weak_only_position_no_kd.yaml")

    assert beam_cfg["experiment"]["objective"] == "beam"
    assert beam_cfg["training"]["early_stopping_metric"] == "val_adba"
    assert occlusion_cfg["experiment"]["objective"] == "occlusion"
    assert occlusion_cfg["training"]["early_stopping_metric"] == "val_occlusion_blocked_f1"
    assert occlusion_cfg["data"]["dataset"]["occlusion_target"]["enabled"] is True
    assert occlusion_cfg["model"]["student"]["auxiliary_heads"]["occlusion"] is True
    assert position_cfg["experiment"]["objective"] == "position"
    assert position_cfg["training"]["early_stopping_metric"] == "val_position_rmse"
    assert position_cfg["training"]["early_stopping_mode"] == "min"

    with pytest.raises(ValueError, match="experiment.objective.*current_beam_selection.*selection_multitask"):
        load_config(ROOT / "configs/fusion/all_modalities_no_kd.yaml", ["experiment.objective=bad"])
    auto_occlusion = load_config(ROOT / "configs/fusion/all_modalities_no_kd.yaml", ["experiment.objective=occlusion"])
    auto_position = load_config(ROOT / "configs/fusion/all_modalities_no_kd.yaml", ["experiment.objective=position"])
    assert auto_occlusion["data"]["dataset"]["occlusion_target"]["enabled"] is True
    assert auto_occlusion["model"]["student"]["auxiliary_heads"]["occlusion"] is True
    assert auto_position["data"]["dataset"]["position_target"]["enabled"] is True
    assert auto_position["model"]["student"]["auxiliary_heads"]["position"] is True
    assert auto_position["data"]["dataset"]["train_csv_name"] == "train_seqs_RA_GPS_LIDAR_POS.csv"


@pytest.mark.parametrize(
    ("objective", "default_metric", "mode", "aliases", "available"),
    [
        ("beam", "val_adba", "max", ["adba", "beam/val_adba"], ["val_adba", "val_acc"]),
        (
            "occlusion",
            "val_occlusion_blocked_f1",
            "max",
            ["occlusion", "blocked_f1"],
            ["val_occlusion_blocked_f1"],
        ),
        ("position", "val_position_rmse", "min", ["position", "position_rmse"], ["val_position_rmse"]),
        ("multitask", "val_multitask_loss", "min", ["multitask", "multitask_loss"], ["val_multitask_loss"]),
        (
            "current_beam_selection",
            "val_beam_top1",
            "max",
            ["current_beam_selection", "beam_top1"],
            ["val_beam_top1", "val_beam_dba"],
        ),
        (
            "current_los_classification",
            "val_los_f1",
            "max",
            ["current_los_classification", "los"],
            ["val_los_f1", "val_los_accuracy"],
        ),
        (
            "current_link_quality",
            "val_link_mae",
            "min",
            ["current_link_quality", "link_quality"],
            ["val_link_mae", "val_link_rmse"],
        ),
        (
            "selection_multitask",
            "val_selection_multitask_loss",
            "min",
            ["selection_multitask", "selection_multitask_loss"],
            ["val_selection_multitask_loss", "val_los_f1", "val_link_mae"],
        ),
    ],
)
def test_objective_metadata_contract_covers_metrics_aliases_history_and_logging(
    objective: str,
    default_metric: str,
    mode: str,
    aliases: list[str],
    available: list[str],
):
    cfg = {"experiment": {"objective": objective}, "loss": {}}
    spec = objective_spec(objective)
    runtime = objective_runtime_metadata(cfg)

    assert spec.default_metric == default_metric
    assert spec.default_metric_mode == mode
    assert runtime["primary_metric"] == default_metric
    assert runtime["primary_metric_mode"] == mode
    assert runtime["available_metrics"] == list(spec.available_metrics)
    for alias in aliases:
        assert normalize_objective_metric(alias, objective=objective) == default_metric
    assert objective_metric_mode(default_metric) == mode
    for metric in available:
        assert metric in objective_available_metrics(objective)
    assert "train_objective_loss" in objective_history_fields(objective)
    assert "val_primary_metric" in objective_history_fields(objective)
    assert ("objective/val_primary_metric", "val_primary_metric") in objective_tensorboard_scalars(objective)
    if objective in {"beam", "multitask"}:
        assert ("beam/val_adba", "val_adba") in objective_tensorboard_scalars(objective)
    elif objective in {"current_beam_selection", "selection_multitask"}:
        assert ("beam/val_top1", "val_beam_top1") in objective_tensorboard_scalars(objective)
        assert ("beam/val_dba_current", "val_beam_dba") in objective_tensorboard_scalars(objective)
    else:
        assert ("beam/val_adba", "val_adba") not in objective_tensorboard_scalars(objective)
        assert ("beam/val_top1", "val_beam_top1") not in objective_tensorboard_scalars(objective)


def test_raymobtime_single_task_tensorboard_scalars_are_isolated():
    beam_tags = {tag for tag, _ in objective_tensorboard_scalars("current_beam_selection")}
    los_tags = {tag for tag, _ in objective_tensorboard_scalars("current_los_classification")}
    link_tags = {tag for tag, _ in objective_tensorboard_scalars("current_link_quality")}
    multitask_tags = {tag for tag, _ in objective_tensorboard_scalars("selection_multitask")}

    assert {"beam/val_top1", "beam/val_top3", "beam/val_top5", "beam/val_dba_current"} <= beam_tags
    assert not {"los/accuracy", "link/mae"} & beam_tags
    assert {"los/accuracy", "los/f1", "los/auc"} <= los_tags
    assert not {"beam/val_top1", "beam/val_dba_current", "link/mae"} & los_tags
    assert {"link/mae", "link/rmse", "link/r2"} <= link_tags
    assert not {"beam/val_top1", "beam/val_dba_current", "los/accuracy"} & link_tags
    assert {"beam/val_dba_current", "los/accuracy", "link/mae", "loss/val_selection_multitask_total"} <= multitask_tags


def test_multimodal_nf_config_rejects_codebook_num_class_mismatch():
    with pytest.raises(ValueError, match="num_beam_classes=4000.*model.num_classes=8"):
        load_config(
            ROOT / "configs/multimodal_nf/gps_beam.yaml",
            ["model.num_classes=8", "model.student.num_classes=8"],
        )


def test_multimodal_nf_los_override_requires_los_head():
    with pytest.raises(ValueError, match="model.student.auxiliary_heads.los=true"):
        load_config(
            ROOT / "configs/multimodal_nf/gps_beam.yaml",
            ["experiment.objective=current_los_classification"],
        )


def test_multimodal_nf_runtime_artifact_metadata_is_consistent(tmp_path: Path):
    cfg = {
        "experiment": {"name": "metadata_consistency", "task": "gps", "objective": "near_field_beam_selection"},
        "data": {"dataset": {"type": "multimodal_nf", "codebook_shape": [2, 3, 4], "seq_len": 1, "num_pred": 1}},
        "model": {
            "modalities": ["gps"],
            "num_classes": 24,
            "student": {"type": "modular_sequence", "modalities": ["gps"], "num_classes": 24},
        },
    }
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    final_cfg = final_config_with_runtime(cfg, run_dir=tmp_path)
    startup = build_startup_summary(cfg, model, optimizer, None, device=torch.device("cpu"))

    final_objective = final_cfg["runtime"]["prediction_objective"]
    assert startup["experiment"]["objective"] == final_objective["name"] == "near_field_beam_selection"
    assert startup["objective"]["name"] == final_objective["name"]
    assert startup["objective"]["target_schema"] == final_objective["target_schema"]
    assert startup["data"]["num_beam_classes"] == final_cfg["runtime"]["prediction_setup"]["num_beam_classes"] == 24
    assert final_cfg["runtime"]["prediction_setup"]["target_schema"] == "near_field_3d_codebook_flattened_beam_class"


@pytest.mark.parametrize("modality", ["image", "radar", "gps", "lidar", "mmwave"])
@pytest.mark.parametrize(
    ("objective", "expected_metric", "expected_mode"),
    [
        ("occlusion", "val_occlusion_blocked_f1", "max"),
        ("position", "val_position_rmse", "min"),
        ("multitask", "val_multitask_loss", "min"),
    ],
)
def test_single_modality_objective_overrides_autoconfigure_targets_and_heads(
    modality: str,
    objective: str,
    expected_metric: str,
    expected_mode: str,
):
    cfg = load_config(ROOT / f"configs/{modality}/teacher_no_kd.yaml", [f"experiment.objective={objective}"])
    needs_occlusion = objective in {"occlusion", "multitask"}
    needs_position = objective in {"position", "multitask"}

    assert cfg["experiment"]["task"] == modality
    assert cfg["experiment"]["objective"] == objective
    assert cfg["training"]["early_stopping_metric"] == expected_metric
    assert cfg["training"]["early_stopping_mode"] == expected_mode
    assert cfg["model"]["student"]["auxiliary_heads"]["enabled"] is True
    assert cfg["model"]["student"]["auxiliary_heads"].get("occlusion", False) is needs_occlusion
    assert cfg["model"]["student"]["auxiliary_heads"].get("position", False) is needs_position
    assert cfg["model"]["student"]["num_pred"] == cfg["model"]["num_pred"]
    if needs_occlusion:
        assert cfg["data"]["dataset"]["occlusion_target"]["enabled"] is True
    if needs_position:
        assert cfg["data"]["dataset"]["position_target"]["enabled"] is True
        assert cfg["data"]["dataset"]["train_csv_name"] == "train_seqs_RA_GPS_LIDAR_POS.csv"
        assert cfg["data"]["dataset"]["test_csv_name"] == "test_seqs_RA_GPS_LIDAR_POS.csv"
    if objective == "multitask":
        assert multitask_loss_weights(cfg) == {"beam": 1.0, "occlusion": 1.0, "position": 1.0}


@pytest.mark.parametrize(
    ("config_path", "modalities"),
    [
        (
            "configs/fusion/image_radar_gps_lidar_mmwave_multitask_no_kd.yaml",
            ["image", "radar", "gps", "lidar", "mmwave"],
        ),
        ("configs/fusion/strong_only_multitask_no_kd.yaml", ["gps", "mmwave"]),
        ("configs/fusion/weak_only_multitask_no_kd.yaml", ["image", "radar", "lidar"]),
    ],
)
def test_objective_multitask_virtual_configs_default_to_equal_weights(config_path: str, modalities: list[str]):
    cfg = load_config(ROOT / config_path)

    assert cfg["experiment"]["objective"] == "multitask"
    assert cfg["model"]["student"]["modalities"] == modalities
    assert cfg["data"]["dataset"]["occlusion_target"]["enabled"] is True
    assert cfg["data"]["dataset"]["position_target"]["enabled"] is True
    assert cfg["model"]["student"]["auxiliary_heads"]["occlusion"] is True
    assert cfg["model"]["student"]["auxiliary_heads"]["position"] is True
    assert multitask_loss_weights(cfg) == {"beam": 1.0, "occlusion": 1.0, "position": 1.0}
    assert cfg["training"]["early_stopping_metric"] == "val_multitask_loss"
    assert cfg["training"]["early_stopping_mode"] == "min"


def test_multitask_weight_and_early_stopping_overrides_are_scoped():
    cfg = load_config(
        ROOT / "configs/fusion/image_radar_gps_lidar_mmwave_multitask_no_kd.yaml",
        [
            "loss.objective.weights.position=0.25",
            "training.early_stopping_metric=val_loss",
            "training.early_stopping_mode=min",
        ],
    )

    assert multitask_loss_weights(cfg) == {"beam": 1.0, "occlusion": 1.0, "position": 0.25}
    assert cfg["training"]["early_stopping_metric"] == "val_loss"
    assert cfg["training"]["early_stopping_mode"] == "min"


@pytest.mark.parametrize(
    ("objective", "expected_metric", "expected_mode"),
    [
        ("beam", "val_adba", "max"),
        ("occlusion", "val_occlusion_blocked_f1", "max"),
        ("position", "val_position_rmse", "min"),
        ("multitask", "val_multitask_loss", "min"),
    ],
)
def test_objective_virtual_configs_select_default_early_stopping(
    objective: str,
    expected_metric: str,
    expected_mode: str,
):
    cfg = load_config(ROOT / f"configs/fusion/strong_only_{objective}_no_kd.yaml")

    assert cfg["experiment"]["objective"] == objective
    assert cfg["training"]["early_stopping_metric"] == expected_metric
    assert cfg["training"]["early_stopping_mode"] == expected_mode


def test_objective_loss_masks_and_multitask_weights():
    logits = torch.tensor([[[1.0, 0.0], [0.2, 0.8]]])
    occlusion_logits = torch.tensor([[0.0, 2.0]])
    position = torch.tensor([[[1.0, 1.0], [3.0, 5.0]]])
    output = ModelOutput(
        logits=logits,
        input_features=None,
        output_features=None,
        diagnostics={"occlusion_logits": occlusion_logits, "position": position},
    )
    targets = PredictionTargets(
        labels=torch.tensor([[0, 1]]),
        occlusion_label=torch.tensor([[1.0, 0.0]]),
        occlusion_valid=torch.tensor([[True, False]]),
        position_target=torch.tensor([[[0.0, 1.0], [1.0, 1.0]]]),
        position_valid=torch.tensor([[True, False]]),
    )
    beam = logits.sum() * 0.0 + 3.0

    occlusion = compute_prediction_loss(
        output,
        targets,
        {"experiment": {"objective": "occlusion"}, "loss": {"objective": {"occlusion": {"pos_weight": None}}}},
        reference=logits,
        beam_total_loss=beam,
        beam_task_loss=beam,
    )
    expected_bce = F.binary_cross_entropy_with_logits(
        occlusion_logits[:, :1],
        targets.occlusion_label[:, :1],
    )
    assert torch.allclose(occlusion.total, expected_bce)
    assert torch.allclose(occlusion.primary, expected_bce)

    position_loss = compute_prediction_loss(
        output,
        targets,
        {"experiment": {"objective": "position"}, "loss": {"objective": {"position": {"type": "mse"}}}},
        reference=logits,
        beam_total_loss=beam,
        beam_task_loss=beam,
    )
    assert torch.allclose(position_loss.total, torch.tensor(0.5))

    multitask = compute_prediction_loss(
        output,
        targets,
        {
            "experiment": {"objective": "multitask"},
            "loss": {"objective": {"weights": {"beam": 0.5, "occlusion": 2.0, "position": 0.25}}},
        },
        reference=logits,
        beam_total_loss=beam,
        beam_task_loss=beam,
    )
    expected_total = 0.5 * beam + 2.0 * expected_bce + 0.25 * torch.tensor(0.5)
    assert torch.allclose(multitask.total, expected_total)


def test_beam_objective_keeps_old_auxiliary_loss_compatibility():
    logits = torch.zeros(1, 2, 2)
    output = ModelOutput(
        logits=logits,
        input_features=None,
        output_features=None,
        diagnostics={
            "occlusion_logits": torch.zeros(1, 2),
            "position": torch.zeros(1, 2, 2),
        },
    )
    targets = PredictionTargets(
        labels=torch.tensor([[0, 1]]),
        occlusion_label=torch.ones(1, 2),
        occlusion_valid=torch.tensor([[True, True]]),
        position_target=torch.ones(1, 2, 2),
        position_valid=torch.tensor([[True, True]]),
    )
    beam = logits.sum() * 0.0 + 1.0
    cfg = {
        "experiment": {"objective": "beam"},
        "data": {
            "dataset": {
                "occlusion_target": {"enabled": True},
                "position_target": {"enabled": True},
            }
        },
        "model": {"student": {"auxiliary_heads": {"enabled": True, "occlusion": True, "position": True}}},
        "loss": {
            "auxiliary": {
                "enabled": True,
                "occlusion": {"enabled": True, "weight": 1.0},
                "position": {"enabled": True, "weight": 0.5},
            }
        },
    }

    result = compute_prediction_loss(output, targets, cfg, reference=logits, beam_total_loss=beam, beam_task_loss=beam)

    assert result.total.item() > beam.item()
    assert result.multitask_total.item() == pytest.approx(result.occlusion.item() + 0.5 * result.position.item())


@pytest.mark.parametrize("objective", ["beam", "occlusion", "position", "multitask"])
def test_tiny_training_smoke_for_first_class_objectives(tmp_path: Path, objective: str):
    cfg = _tiny_objective_cfg(objective, tmp_path)

    result = train(cfg)

    history = result["history"]
    assert resolve_prediction_objective(cfg) == objective
    assert len(history["train_task_loss"]) == 1
    if objective == "occlusion":
        assert history["train_task_loss"][0] == pytest.approx(history["train_occlusion_loss"][0])
        assert history["val_occlusion_blocked_f1"][0] is not None
        assert history["val_position_rmse"][0] is None
        assert "val_position_rmse" not in result["epoch_logs"][0]["validation_metrics"]
    if objective == "position":
        assert history["train_task_loss"][0] == pytest.approx(history["train_position_loss"][0])
        assert history["val_occlusion_blocked_f1"][0] is None
        assert history["val_position_rmse"][0] is not None
        assert "val_occlusion_blocked_f1" not in result["epoch_logs"][0]["validation_metrics"]
    if objective == "beam":
        assert history["val_occlusion_blocked_f1"][0] is None
        assert history["val_position_rmse"][0] is None
        assert history["val_multitask_loss"][0] is None
        assert "val_position_rmse" not in result["epoch_logs"][0]["validation_metrics"]
    if objective == "multitask":
        assert history["val_occlusion_blocked_f1"][0] is not None
        assert history["val_position_rmse"][0] is not None
        assert history["val_multitask_loss"][0] is not None
        assert result["prediction_objective"]["loss_weights"] == {"beam": 1.0, "occlusion": 1.0, "position": 1.0}
        assert result["epoch_logs"][0]["loss_weights"] == {"beam": 1.0, "occlusion": 1.0, "position": 1.0}


def _tiny_objective_cfg(objective: str, tmp_path: Path) -> dict:
    needs_occlusion = objective in {"occlusion", "multitask"}
    needs_position = objective in {"position", "multitask"}
    return {
        "experiment": {"name": f"tiny_{objective}", "task": "fusion", "objective": objective, "seed": 3, "device": "cpu"},
        "data": {
            "dataset": {
                "type": "synthetic",
                "scene": 31,
                "length": 2,
                "seq_len": 2,
                "num_pred": 2,
                "num_classes": 8,
                "use_gps": True,
                "gps_input_size": 3,
                "occlusion_target": {"enabled": needs_occlusion},
                "position_target": {"enabled": needs_position, "normalize": False},
            },
            "dataloader": {"train_batch_size": 1, "test_batch_size": 1, "num_workers": 0, "pin_memory": False},
        },
        "model": {
            "modalities": ["gps"],
            "feature_size": 8,
            "num_classes": 8,
            "seq_length_teacher": 2,
            "seq_length_student": 2,
            "num_pred": 2,
            "downsample_ratio": 1,
            "teacher": {
                "type": "cls_token_transformer_fusion",
                "modalities": ["gps"],
                "feature_size": 8,
                "d_model": 8,
                "num_classes": 8,
                "num_pred": 2,
                "num_heads": 2,
                "num_layers": 1,
                "max_seq_len": 4,
                "gps_input_size": 3,
            },
            "student": {
                "type": "cls_token_transformer_fusion",
                "modalities": ["gps"],
                "feature_size": 8,
                "d_model": 8,
                "num_classes": 8,
                "num_pred": 2,
                "num_heads": 2,
                "num_layers": 1,
                "max_seq_len": 4,
                "gps_input_size": 3,
                "auxiliary_heads": {
                    "enabled": needs_occlusion or needs_position,
                    "occlusion": needs_occlusion,
                    "position": needs_position,
                },
            },
        },
        "loss": {
            "type": "cross_entropy",
            "objective": {
                "weights": {"beam": 1.0, "occlusion": 1.0, "position": 1.0},
                "position": {"type": "mse"},
            },
        },
        "distillation": {"type": "no_kd", "teacher_model_name": None},
        "training": {
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "grad_clip": None,
            "patience": 2,
            "use_early_stopping": False,
            "early_stopping_metric": {
                "beam": "val_adba",
                "occlusion": "val_occlusion_blocked_f1",
                "position": "val_position_rmse",
                "multitask": "val_multitask_loss",
            }[objective],
            "early_stopping_mode": "min" if objective in {"position", "multitask"} else "max",
            "min_delta": 0.0,
            "transfer": {"non_blocking": False},
            "amp": {"enabled": False, "dtype": "float16", "grad_scaler": True},
        },
        "scheduler": {"type": "none"},
        "evaluation": {"k_values": [1, 3], "dba_delta": 5},
        "output": {
            "dir": str(tmp_path),
            "run_name": f"tiny_{objective}",
            "overwrite": True,
            "progress": {"enabled": False},
            "tensorboard": {"enabled": False},
        },
        "checkpoint": {"registry": {"enabled": False}, "strict_load": True},
    }
