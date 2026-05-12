from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.engine.teacher_loader import apply_teacher_priors, load_teacher_encoders  # noqa: E402
from kd_sensing.models.fusion.marf import ModalityRouter  # noqa: E402
from kd_sensing.registries import MODELS  # noqa: E402


def test_marf_forward_shapes_and_output_contract():
    model = _gps_mmwave_marf()
    model.eval()
    with torch.no_grad():
        output = model(
            gps_batch=torch.randn(2, 4, 3),
            mmwave_batch=torch.randn(2, 4, 64),
        )

    assert "marf_fusion" in MODELS.list()
    assert model.supports_force_modality_mask is True
    assert model.supports_marf_routing is True
    assert model.horizon == model.num_pred == 2
    assert model.router.horizon == 2
    assert model.anchor_fusion.horizon == 2
    assert model.residual_adapter.horizon == 2
    assert model.unimodal_head.horizon == 2
    assert set(model.encoders.keys()) == {"gps", "mmwave"}
    assert output["logits"].shape == (2, 2, 8)
    assert output["anchor_weights"].shape == (2, 2, 2)
    assert output["residual_weights"].shape == (2, 2, 2)
    assert output["h_anchor"].shape == (2, 2, 16)
    assert output["h_final"].shape == (2, 2, 16)
    assert output["residual_delta"].shape == (2, 2, 2, 16)
    assert output["token_features"].shape[:3] == (2, 2, 4)
    assert output["modalities"] == ("gps", "mmwave")


def test_marf_anchor_softmax_and_mask_zero_unavailable_modalities():
    model = _gps_mmwave_marf()
    model.eval()
    mask = torch.tensor([[True, False], [False, True]])
    with torch.no_grad():
        output = model(
            gps_batch=torch.randn(2, 4, 3),
            mmwave_batch=torch.randn(2, 4, 64),
            force_modality_mask=mask,
        )

    anchor = output["anchor_weights"]
    residual = output["residual_weights"]
    assert torch.allclose(anchor.sum(dim=-1), torch.ones(2, 2), atol=1e-6)
    assert torch.all(anchor[0, :, 1] == 0.0)
    assert torch.all(anchor[1, :, 0] == 0.0)
    assert torch.all(residual[0, :, 1] == 0.0)
    assert torch.all(residual[1, :, 0] == 0.0)
    assert output["effective_modality_mask"].tolist() == [[True, False], [False, True]]


def test_marf_router_prior_bias_can_be_disabled():
    router = ModalityRouter(
        4,
        2,
        3,
        modalities=("gps", "mmwave"),
        dataset_prior={"gps": 0.9, "mmwave": 0.1},
        use_prior_bias=False,
    )
    result = router(torch.zeros(1, 2, 4), torch.zeros(1, 2, 2), torch.ones(1, 2, dtype=torch.bool))
    assert torch.allclose(result["anchor_weights"], torch.full((1, 3, 2), 0.5), atol=1e-6)
    assert torch.allclose(result["prior"], torch.tensor([[0.9, 0.1]]), atol=1e-6)


def test_marf_teacher_prior_encoder_loading_and_freeze_boundary(tmp_path: Path):
    model = _gps_mmwave_marf(feature_size=8, d_model=16, num_classes=4)
    original = {key: value.clone() for key, value in model.encoders["gps"].state_dict().items()}
    shifted = {f"feature_extraction.{key}": value + 1.0 for key, value in original.items()}
    ckpt = tmp_path / "gps_teacher.pth"
    torch.save({"state_dict": shifted}, ckpt)
    registry = {
        "teachers": {
            "gps": {"ckpt": str(ckpt), "prior": 0.8},
            "mmwave": {"ckpt": str(ckpt), "prior": 0.4},
        }
    }

    priors = apply_teacher_priors(model, registry, ["gps", "mmwave"])
    summaries = load_teacher_encoders(model, registry, ["gps"], strict=True, freeze_loaded=True)

    assert priors == {"gps": 0.8, "mmwave": 0.4}
    assert model.router.prior.tolist() == pytest.approx([0.8, 0.4])
    assert summaries["gps"]["success"] is True
    assert summaries["gps"]["frozen"] is True
    assert all(not param.requires_grad for param in model.encoders["gps"].parameters())
    assert any(param.requires_grad for param in model.router.parameters())
    assert any(param.requires_grad for param in model.anchor_fusion.parameters())
    assert any(param.requires_grad for param in model.residual_adapter.parameters())
    assert any(param.requires_grad for param in model.feature_projections.parameters())
    assert any(param.requires_grad for param in model.prediction_head.parameters())


def test_marf_no_residual_ablation_keeps_anchor_path():
    model = _gps_mmwave_marf(residual_enabled=False)
    model.eval()
    with torch.no_grad():
        output = model(gps_batch=torch.randn(1, 3, 3), mmwave_batch=torch.randn(1, 3, 64))
    assert torch.allclose(output["h_anchor"], output["h_final"])
    assert output["residual_delta"].abs().max().item() == 0.0


def test_marf_configs_load_and_change_only_target_ablation_fields():
    main = load_config(ROOT / "configs/fusion/marf.yaml")
    subset = load_config(ROOT / "configs/fusion/marf_subset_training.yaml")
    no_residual = load_config(ROOT / "configs/fusion/marf_no_residual_ablation.yaml")
    no_prior = load_config(ROOT / "configs/fusion/marf_no_prior_bias_ablation.yaml")
    no_subset = load_config(ROOT / "configs/fusion/marf_no_subset_training_ablation.yaml")

    assert main["experiment"]["task"] == "fusion"
    assert main["model"]["student"]["type"] == "marf_fusion"
    assert main["loss"]["type"] == "cross_entropy"
    assert main["loss"]["label_smoothing"] == pytest.approx(0.03)
    assert main["teacher"]["load_encoders"] is True
    assert main["teacher"]["freeze_encoders"] is True
    assert subset["training"]["subset_training"]["enabled"] is True
    assert subset["training"]["subset_training"]["modes"] == ["top_prior", "random_with_top_prior"]
    assert no_residual["model"]["student"]["residual_adapter"]["enabled"] is False
    assert no_prior["model"]["student"]["router"]["use_prior_bias"] is False
    assert no_subset["training"]["subset_training"]["enabled"] is False
    for cfg in (no_residual, no_prior, no_subset):
        assert cfg["model"]["student"]["modalities"] == subset["model"]["student"]["modalities"]
        assert cfg["data"]["dataset"]["train_csv_name"] == subset["data"]["dataset"]["train_csv_name"]
        assert cfg["training"]["lr"] == subset["training"]["lr"]


def test_marf_top_level_modalities_override_syncs_roles_and_dataset_flags():
    cfg = load_config(
        ROOT / "configs/fusion/marf_subset_training.yaml",
        ["model.modalities=[\"image\",\"radar\",\"gps\",\"lidar\"]"],
    )

    expected = ["image", "radar", "gps", "lidar"]
    assert cfg["model"]["modalities"] == expected
    assert cfg["model"]["teacher"]["modalities"] == expected
    assert cfg["model"]["student"]["modalities"] == expected
    assert cfg["data"]["dataset"]["use_gps"] is True
    assert cfg["data"]["dataset"]["use_lidar"] is True
    assert cfg["data"]["dataset"]["use_mmwave"] is False


def _gps_mmwave_marf(
    *,
    feature_size: int = 16,
    d_model: int = 16,
    num_classes: int = 8,
    residual_enabled: bool = True,
):
    return MODELS.build(
        {
            "type": "marf_fusion",
            "modalities": ["gps", "mmwave"],
            "feature_size": feature_size,
            "d_model": d_model,
            "num_classes": num_classes,
            "num_pred": 2,
            "num_heads": 4 if d_model % 4 == 0 else 2,
            "gps_input_size": 3,
            "mmwave_input_size": 64,
            "router": {
                "dataset_prior": {"gps": 0.7, "mmwave": 0.3},
                "use_prior_bias": True,
                "prior_anchor_scale": 0.0,
                "prior_residual_scale": 0.0,
            },
            "residual_adapter": {"enabled": residual_enabled, "residual_scale": 0.2},
        }
    )
