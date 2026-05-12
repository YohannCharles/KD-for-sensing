from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.distillation.craf_losses import prior_regularization_loss  # noqa: E402
from kd_sensing.engine.teacher_loader import (  # noqa: E402
    apply_selective_finetune,
    load_teacher_encoders,
)
from kd_sensing.models.fusion.craf import PriorResidualGate  # noqa: E402
from kd_sensing.registries import MODELS, import_default_components  # noqa: E402
from kd_sensing.utils.checkpoint import CheckpointLoadError  # noqa: E402
from kd_sensing.utils.teacher_registry import build_teacher_registry  # noqa: E402


def test_teacher_registry_manual_and_metric_prior(tmp_path: Path):
    root = tmp_path / "teachers"
    for idx, modality in enumerate(["image", "radar", "gps", "lidar", "mmwave"], start=1):
        run_dir = root / f"{modality}_teacher_no_kd"
        (run_dir / "checkpoints").mkdir(parents=True)
        torch.save({"value": torch.tensor([idx])}, run_dir / "checkpoints" / "best_top1.pth")
        (run_dir / "teacher_metrics.json").write_text(
            json.dumps(
                {
                    "modality": modality,
                    "best_epoch": idx,
                    "val_acc_top1": 0.1 * idx,
                    "val_acc_top3": 0.2 * idx,
                    "val_acc_top5": 0.3 * idx,
                    "val_adba": 0.05 * idx,
                    "train_acc_top1": 0.4 * idx,
                }
            ),
            encoding="utf-8",
        )

    manual = build_teacher_registry(
        teacher_root=root,
        output_path=tmp_path / "manual.json",
        scene=32,
        prior_mode="manual",
    )
    assert manual["teachers"]["gps"]["prior"] == pytest.approx(0.85)
    assert manual["teachers"]["mmwave"]["ckpt"].endswith("best_top1.pth")

    metric = build_teacher_registry(
        teacher_root=root,
        output_path=tmp_path / "metric.json",
        scene=32,
        prior_mode="metric",
        prior_min=0.2,
        prior_max=0.9,
    )
    assert metric["teachers"]["mmwave"]["prior"] == pytest.approx(0.9)
    assert metric["teachers"]["image"]["prior"] == pytest.approx(0.2)


def test_teacher_registry_rejects_missing_metrics_and_modality_mismatch(tmp_path: Path):
    run_dir = tmp_path / "teachers" / "gps_teacher_no_kd"
    (run_dir / "checkpoints").mkdir(parents=True)
    torch.save({"value": torch.tensor([1])}, run_dir / "checkpoints" / "best.pth")
    (run_dir / "teacher_metrics.json").write_text(
        json.dumps(
            {
                "modality": "mmwave",
                "best_epoch": 1,
                "val_acc_top1": 0.3,
                "val_acc_top3": 0.4,
                "val_acc_top5": 0.5,
                "val_adba": 0.2,
                "train_acc_top1": 0.6,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="modality mismatch"):
        build_teacher_registry(
            teacher_root=tmp_path / "teachers",
            output_path=tmp_path / "registry.json",
            modalities=["gps"],
        )
    missing_dir = tmp_path / "teachers" / "radar_teacher_no_kd"
    (missing_dir / "checkpoints").mkdir(parents=True)
    torch.save({"value": torch.tensor([1])}, missing_dir / "checkpoints" / "best.pth")
    (missing_dir / "teacher_metrics.json").write_text(
        json.dumps({"modality": "radar", "best_epoch": 1, "val_acc_top1": 0.2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="val_acc_top3"):
        build_teacher_registry(
            teacher_root=tmp_path / "teachers",
            output_path=tmp_path / "registry.json",
            modalities=["radar"],
            prior_mode="metric",
        )


def test_prior_residual_gate_initializes_to_prior_and_masks_unavailable():
    prior = {"image": 0.20, "radar": 0.20, "gps": 0.85, "lidar": 0.15, "mmwave": 0.90}
    gate = PriorResidualGate(
        8,
        5,
        dataset_prior=prior,
        modalities=("image", "radar", "gps", "lidar", "mmwave"),
    )
    output = gate(
        torch.zeros(2, 5, 8),
        torch.zeros(2, 5, 2),
        torch.tensor([[True, True, True, True, False], [True, True, True, True, True]]),
    )
    assert output["gate"].shape == (2, 5)
    means = output["gate"].mean(dim=0)
    assert means[2].item() == pytest.approx(0.85, abs=0.03)
    assert means[4].item() < 0.90
    assert output["gate"][0, 4].item() == 0.0
    loss = prior_regularization_loss(output["gate"], output["prior"], torch.tensor([[False, False, False, False, False], [True, True, True, True, True]]))
    assert loss.ndim == 0
    assert prior_regularization_loss(output["gate"], output["prior"], torch.zeros(2, 5, dtype=torch.bool)).item() == 0.0
    assert prior_regularization_loss(output["gate"], output["prior"], loss_type="l1").item() >= 0.0


def test_craf_prior_residual_diagnostics_and_none_gate():
    import_default_components()
    model = MODELS.build(
        {
            "type": "craf_fusion",
            "modalities": ["gps", "mmwave"],
            "feature_size": 16,
            "d_model": 16,
            "num_classes": 8,
            "num_pred": 1,
            "num_heads": 4,
            "num_layers": 1,
            "gps_input_size": 3,
            "mmwave_input_size": 64,
            "reliability": {
                "gate_type": "prior_residual_sigmoid",
                "dataset_prior": {"gps": 0.85, "mmwave": 0.9},
            },
        }
    )
    model.eval()
    with torch.no_grad():
        output = model(
            gps_batch=torch.randn(1, 3, 3),
            mmwave_batch=torch.randn(1, 3, 64),
            force_modality_mask=torch.tensor([[True, False]]),
        )
    assert output["gate"][0, 0].item() == pytest.approx(0.85, abs=0.03)
    assert output["gate"][0, 1].item() == 0.0
    assert output["gate_logits"].shape == (1, 2)
    assert output["prior"].shape == (1, 2)
    assert output["residual_logits"].abs().max().item() == pytest.approx(0.0)

    none_gate = MODELS.build(
        {
            "type": "craf_fusion",
            "modalities": ["gps", "mmwave"],
            "feature_size": 16,
            "d_model": 16,
            "num_classes": 8,
            "num_pred": 1,
            "num_heads": 4,
            "num_layers": 1,
            "gps_input_size": 3,
            "mmwave_input_size": 64,
            "reliability": {"gate_type": "none"},
        }
    )
    none_gate.eval()
    with torch.no_grad():
        none_output = none_gate(
            gps_batch=torch.randn(1, 3, 3),
            mmwave_batch=torch.randn(1, 3, 64),
            force_modality_mask=torch.tensor([[True, False]]),
        )
    assert none_output["gate"].tolist() == [[1.0, 0.0]]


def test_teacher_loader_maps_feature_extraction_and_freezes(tmp_path: Path):
    import_default_components()
    model = _gps_mmwave_model()
    original = {key: value.clone() for key, value in model.encoders["gps"].state_dict().items()}
    shifted = {f"feature_extraction.{key}": value + 1.0 for key, value in original.items()}
    ckpt = tmp_path / "gps_teacher.pth"
    torch.save({"state_dict": shifted}, ckpt)
    registry = {"teachers": {"gps": {"ckpt": str(ckpt), "prior": 0.8}}}

    summaries = load_teacher_encoders(model, registry, ["gps"], strict=True, freeze_loaded=True)
    assert summaries["gps"]["success"] is True
    assert summaries["gps"]["frozen"] is True
    assert all(not param.requires_grad for param in model.encoders["gps"].parameters())
    for key, value in model.encoders["gps"].state_dict().items():
        assert torch.allclose(value, original[key] + 1.0)


def test_teacher_loader_strict_shape_mismatch_and_selective_finetune(tmp_path: Path):
    import_default_components()
    model = _gps_mmwave_model()
    state = {}
    for key, value in model.encoders["gps"].state_dict().items():
        state[f"feature_extraction.{key}"] = value.clone()
    first_key = next(iter(state))
    state[first_key] = torch.randn(1)
    ckpt = tmp_path / "bad_gps_teacher.pth"
    torch.save({"state_dict": state}, ckpt)
    registry = {"teachers": {"gps": {"ckpt": str(ckpt), "prior": 0.8}}}

    with pytest.raises(CheckpointLoadError, match="gps"):
        load_teacher_encoders(model, registry, ["gps"], strict=True)

    summary = load_teacher_encoders(model, registry, ["gps"], strict=False)
    assert summary["gps"]["shape_mismatches"]

    finetune = apply_selective_finetune(
        model,
        unfreeze_modalities=["mmwave"],
        freeze_modalities=["gps"],
    )
    assert finetune["gps"]["frozen"] is True
    assert finetune["mmwave"]["trainable_params"] > 0


def test_teacher_prior_configs_load_and_legacy_configs_remain_stable():
    stage2 = load_config(ROOT / "configs/fusion/stage2_teacher_init_prior_residual.yaml")
    stage3 = load_config(ROOT / "configs/fusion/stage3_selective_ft_gps_mmwave.yaml")
    no_prior = load_config(ROOT / "configs/fusion/teacher_init_no_prior_ablation.yaml")
    random_encoder = load_config(ROOT / "configs/fusion/prior_gate_random_encoder_ablation.yaml")
    fixed_prior = load_config(ROOT / "configs/fusion/teacher_init_fixed_prior_ablation.yaml")
    prior_residual = load_config(ROOT / "configs/fusion/teacher_init_prior_residual_ablation.yaml")
    legacy = load_config(ROOT / "configs/fusion/craf_all_modalities_fixed_prior_sanity.yaml")

    assert stage2["experiment"]["task"] == "fusion"
    assert stage2["model"]["student"]["type"] == "craf_fusion"
    assert stage2["model"]["student"]["reliability"]["gate_type"] == "prior_residual_sigmoid"
    assert stage2["teacher"]["load_encoders"] is True
    assert stage2["teacher"]["freeze_encoders"] is True
    assert stage3["finetune"]["unfreeze_modalities"] == ["gps", "mmwave"]
    assert stage3["finetune"]["param_groups"]["enabled"] is True
    assert no_prior["model"]["student"]["reliability"]["gate_type"] == "none"
    assert random_encoder["teacher"]["load_encoders"] is False
    assert fixed_prior["model"]["student"]["reliability"]["gate_type"] == "fixed_prior"
    assert prior_residual["teacher"]["load_encoders"] is True
    assert legacy["teacher"]["load_encoders"] is False


def _gps_mmwave_model():
    return MODELS.build(
        {
            "type": "craf_fusion",
            "modalities": ["gps", "mmwave"],
            "feature_size": 8,
            "d_model": 8,
            "num_classes": 4,
            "num_pred": 1,
            "num_heads": 2,
            "num_layers": 1,
            "gps_input_size": 3,
            "mmwave_input_size": 64,
            "reliability": {"gate_type": "prior_residual_sigmoid", "dataset_prior": {"gps": 0.8, "mmwave": 0.9}},
        }
    )
