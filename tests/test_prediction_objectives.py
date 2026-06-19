from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.engine.model_output import ModelOutput  # noqa: E402
from kd_sensing.engine.prediction_objectives import (  # noqa: E402
    PredictionTargets,
    build_dba_aware_soft_targets,
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
from kd_sensing.engine.teacher_guidance import TeacherGuidanceTrainingExtension  # noqa: E402
from kd_sensing.engine.training_extensions import BatchState, ExtensionContext, ForwardControls  # noqa: E402
from kd_sensing.engine.trainer import train  # noqa: E402


def test_objective_config_defaults_validation_and_primary_model_autoconfiguration():
    beam_cfg = load_config(ROOT / "configs/fusion/all_modalities_beam_supervised.yaml")
    occlusion_cfg = load_config(ROOT / "configs/fusion/strong_only_occlusion_supervised.yaml")
    position_cfg = load_config(ROOT / "configs/fusion/weak_only_position_supervised.yaml")

    assert beam_cfg["experiment"]["objective"] == "beam"
    assert beam_cfg["training"]["early_stopping_metric"] == "val_adba"
    assert occlusion_cfg["experiment"]["objective"] == "occlusion"
    assert occlusion_cfg["training"]["early_stopping_metric"] == "val_occlusion_blocked_f1"
    assert occlusion_cfg["data"]["dataset"]["occlusion_target"]["enabled"] is True
    assert occlusion_cfg["model"]["primary"]["auxiliary_heads"]["occlusion"] is True
    assert position_cfg["experiment"]["objective"] == "position"
    assert position_cfg["training"]["early_stopping_metric"] == "val_position_rmse"
    assert position_cfg["training"]["early_stopping_mode"] == "min"

    with pytest.raises(ValueError, match="experiment.objective.*current_beam_selection.*selection_multitask"):
        load_config(ROOT / "configs/fusion/all_modalities_beam_supervised.yaml", ["experiment.objective=bad"])
    auto_occlusion = load_config(
        ROOT / "configs/fusion/all_modalities_beam_supervised.yaml",
        ["experiment.objective=occlusion"],
    )
    auto_position = load_config(
        ROOT / "configs/fusion/all_modalities_beam_supervised.yaml",
        ["experiment.objective=position"],
    )
    assert auto_occlusion["data"]["dataset"]["occlusion_target"]["enabled"] is True
    assert auto_occlusion["model"]["primary"]["auxiliary_heads"]["occlusion"] is True
    assert auto_position["data"]["dataset"]["position_target"]["enabled"] is True
    assert auto_position["model"]["primary"]["auxiliary_heads"]["position"] is True
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


def test_current_selection_objective_tensorboard_scalars_are_isolated():
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
    cfg = load_config(ROOT / f"configs/{modality}/strong.yaml", [f"experiment.objective={objective}"])
    needs_occlusion = objective in {"occlusion", "multitask"}
    needs_position = objective in {"position", "multitask"}

    assert cfg["experiment"]["task"] == modality
    assert cfg["experiment"]["objective"] == objective
    assert cfg["training"]["early_stopping_metric"] == expected_metric
    assert cfg["training"]["early_stopping_mode"] == expected_mode
    assert cfg["model"]["primary"]["auxiliary_heads"]["enabled"] is True
    assert cfg["model"]["primary"]["auxiliary_heads"].get("occlusion", False) is needs_occlusion
    assert cfg["model"]["primary"]["auxiliary_heads"].get("position", False) is needs_position
    assert cfg["model"]["primary"]["num_pred"] == cfg["model"]["num_pred"]
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
            "configs/fusion/image_radar_gps_lidar_mmwave_multitask_supervised.yaml",
            ["image", "radar", "gps", "lidar", "mmwave"],
        ),
        ("configs/fusion/strong_only_multitask_supervised.yaml", ["gps", "mmwave"]),
        ("configs/fusion/weak_only_multitask_supervised.yaml", ["image", "radar", "lidar"]),
    ],
)
def test_objective_multitask_virtual_configs_default_to_equal_weights(config_path: str, modalities: list[str]):
    cfg = load_config(ROOT / config_path)

    assert cfg["experiment"]["objective"] == "multitask"
    assert cfg["model"]["primary"]["modalities"] == modalities
    assert cfg["data"]["dataset"]["occlusion_target"]["enabled"] is True
    assert cfg["data"]["dataset"]["position_target"]["enabled"] is True
    assert cfg["model"]["primary"]["auxiliary_heads"]["occlusion"] is True
    assert cfg["model"]["primary"]["auxiliary_heads"]["position"] is True
    assert multitask_loss_weights(cfg) == {"beam": 1.0, "occlusion": 1.0, "position": 1.0}
    assert cfg["training"]["early_stopping_metric"] == "val_multitask_loss"
    assert cfg["training"]["early_stopping_mode"] == "min"


def test_multitask_weight_and_early_stopping_overrides_are_scoped():
    cfg = load_config(
        ROOT / "configs/fusion/image_radar_gps_lidar_mmwave_multitask_supervised.yaml",
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
    cfg = load_config(ROOT / f"configs/fusion/strong_only_{objective}_supervised.yaml")

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
        "model": {"primary": {"auxiliary_heads": {"enabled": True, "occlusion": True, "position": True}}},
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


def test_predictive_latent_auxiliary_loss_is_opt_in_and_detaches_target():
    logits = torch.randn(1, 2, 3)
    predicted = torch.zeros(1, 2, 4, requires_grad=True)
    current = torch.ones(1, 2, 4, requires_grad=True)
    output = ModelOutput(
        logits=logits,
        input_features=None,
        output_features=None,
        diagnostics={
            "encoder_auxiliary_features": {
                "image": {
                    "temporal_predicted_latent": predicted,
                    "current_latent": current,
                }
            }
        },
    )
    targets = PredictionTargets(labels=torch.tensor([[0, 1]]))
    beam = logits.sum() * 0.0 + 1.0

    disabled = compute_prediction_loss(
        output,
        targets,
        {"experiment": {"objective": "beam"}},
        reference=logits,
        beam_total_loss=beam,
        beam_task_loss=beam,
    )
    enabled = compute_prediction_loss(
        output,
        targets,
        {
            "experiment": {"objective": "beam"},
            "loss": {"auxiliary": {"predictive_latent": {"enabled": True, "weight": 0.5}}},
        },
        reference=logits,
        beam_total_loss=beam,
        beam_task_loss=beam,
    )

    assert disabled.total.item() == pytest.approx(beam.item())
    assert enabled.total.item() > beam.item()
    assert enabled.diagnostics["loss/predictive_latent_auxiliary"] == pytest.approx(0.5)
    assert enabled.diagnostics["predictive_latent_auxiliary/sample_count"] == 2
    assert enabled.diagnostics["predictive_latent_auxiliary/target_detached"] == 1.0


def test_safe_rerank_auxiliary_loss_uses_candidate_coverage_and_ignores_invalid_labels():
    logits = torch.tensor([[[0.1, 2.0, 0.0, -0.5]], [[1.0, 0.0, -0.1, -0.2]]], dtype=torch.float32)
    anchor_logits = torch.tensor([[[2.0, 0.1, 0.0, -0.5]], [[1.0, 0.0, -0.1, -0.2]]], dtype=torch.float32)
    output = ModelOutput(
        logits=logits,
        input_features=None,
        output_features=None,
        diagnostics={
            "rerank_logits": logits,
            "anchor_logits": anchor_logits,
            "candidate_ids": torch.tensor([[[0, 1, -1]], [[0, 2, -1]]]),
            "residual_scale": torch.ones(2, 1) * 0.2,
        },
    )
    targets = PredictionTargets(labels=torch.tensor([[1], [-100]]))
    beam = logits.sum() * 0.0 + 0.5

    result = compute_prediction_loss(
        output,
        targets,
        {
            "experiment": {"objective": "beam"},
            "loss": {
                "safe_rerank": {
                    "enabled": True,
                    "weight": 1.0,
                    "pairwise_margin_weight": 0.25,
                    "no_regret_weight": 0.5,
                }
            },
        },
        reference=logits,
        beam_total_loss=beam,
        beam_task_loss=beam,
    )

    assert result.total.item() > beam.item()
    assert result.diagnostics["rerank_loss/candidate_coverage"] == pytest.approx(1.0)
    assert result.diagnostics["rerank_loss/skipped_samples"] == pytest.approx(0.0)
    assert result.diagnostics["rerank_loss/anchor_correct_count"] == pytest.approx(0.0)
    assert result.diagnostics["rerank_loss/residual_scale_mean"] == pytest.approx(0.2)


def test_safe_rerank_auxiliary_loss_accepts_half_precision_logits():
    logits = torch.tensor([[[0.1, 2.0, 0.0, -0.5]]], dtype=torch.float16)
    anchor_logits = torch.tensor([[[2.0, 0.1, 0.0, -0.5]]], dtype=torch.float16)
    output = ModelOutput(
        logits=logits,
        input_features=None,
        output_features=None,
        diagnostics={
            "rerank_logits": logits,
            "anchor_logits": anchor_logits,
            "candidate_ids": torch.tensor([[[0, 1, -1]]]),
        },
    )
    beam = logits.float().sum() * 0.0 + 0.5

    result = compute_prediction_loss(
        output,
        PredictionTargets(labels=torch.tensor([[1]])),
        {"experiment": {"objective": "beam"}, "loss": {"safe_rerank": {"enabled": True, "weight": 1.0}}},
        reference=logits,
        beam_total_loss=beam,
        beam_task_loss=beam,
    )

    assert torch.isfinite(result.total)
    assert result.diagnostics["rerank_loss/candidate_coverage"] == pytest.approx(1.0)


def test_dba_aware_beam_loss_builds_circular_soft_targets_from_hard_labels():
    labels = torch.tensor([[0, 7, -100]])
    targets, diagnostics = build_dba_aware_soft_targets(
        labels,
        num_classes=8,
        cfg={
            "loss": {
                "dba_aware": {
                    "enabled": True,
                    "mode": "circular_gaussian",
                    "distance_mode": "circular",
                    "sigma": 1.0,
                }
            }
        },
    )

    assert targets is not None
    torch.testing.assert_close(targets[0, 0].sum(), torch.tensor(1.0))
    torch.testing.assert_close(targets[0, 1].sum(), torch.tensor(1.0))
    assert targets[0, 0, 0] > targets[0, 0, 4]
    assert targets[0, 0, 7] > targets[0, 0, 4]
    torch.testing.assert_close(targets[0, 2], torch.zeros(8))
    assert diagnostics["loss/beam_dba_aware_sample_count"] == 2.0

    disabled, disabled_diag = build_dba_aware_soft_targets(labels, num_classes=8, cfg={"loss": {}})
    assert disabled is None
    assert disabled_diag == {}


def test_teacher_guidance_extension_uses_non_retired_loss_names_and_detaches_teacher(tmp_path: Path):
    extension = TeacherGuidanceTrainingExtension()
    cfg = {
        "loss": {
            "teacher_guidance": {
                "enabled": True,
                "weight": 0.5,
                "temperature": 2.0,
                "checkpoint_path": "outputs/teacher/image_resnet_gps/best.pth",
                "checkpoint_provenance": "unit_test_teacher",
                "enabled_splits": ["train"],
            }
        }
    }
    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]]], requires_grad=True)
    teacher_logits = torch.tensor([[[3.0, 0.0, -1.0], [0.0, 2.0, 3.0]]])
    context = ExtensionContext(
        cfg=cfg,
        task="fusion",
        model_cfg={"num_pred": 2, "num_classes": 3, "seq_length": 2, "primary": {}},
        training_cfg={},
        primary_model=torch.nn.Linear(1, 1),
        task_criterion=torch.nn.CrossEntropyLoss(),
        run_dir=tmp_path,
        device=torch.device("cpu"),
        num_pred=2,
        num_classes=3,
        seq_length=2,
        non_blocking=False,
    )
    state = extension.setup(context)
    batch_state = BatchState(
        epoch=0,
        step=0,
        batch={"teacher_logits": teacher_logits},
        labels=torch.tensor([[0, 2]]),
        soft_beam_targets=None,
        primary_output=ModelOutput(logits=logits, input_features=None, output_features=None, diagnostics={}),
        primary_logits=logits,
        controls=ForwardControls(),
    )

    bundle = extension.after_forward(context, state, batch_state)
    loads = extension.checkpoint_loads(state)
    epoch_meta = extension.after_epoch(context, state, epoch=0)

    assert bundle is not None
    assert bundle.total.item() > 0
    assert "loss/teacher_guidance" in bundle.diagnostics
    assert "loss/geometry_teacher_kl" in bundle.diagnostics
    assert all("distillation" not in key and "kd_soft_label" not in key for key in bundle.diagnostics)
    assert loads[0]["role"] == "teacher_guidance_stabilization"
    assert loads[0]["provenance"] == "unit_test_teacher"
    assert epoch_meta["teacher_guidance"]["mode"] == "opt_in_stabilization"


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
            "seq_length": 2,
            "num_pred": 2,
            "downsample_ratio": 1,
            "primary": {
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
