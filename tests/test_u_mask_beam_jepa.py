from pathlib import Path

import pytest
import torch
import torch.nn as nn

from kd_sensing.config import load_config
from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank
from kd_sensing.losses.u_mask_beam_jepa import u_mask_beam_jepa_loss
from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config
from kd_sensing.registries import ENCODERS, MODELS

import kd_sensing.models.u_mask_beam_jepa  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]


@ENCODERS.register("u_mask_test_sequence", force=True)
class _SequenceEncoder(nn.Module):
    def __init__(self, output_dim: int = 4, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, steps = values.shape[:2]
        scalar = values.float().reshape(batch, steps, -1).mean(dim=-1, keepdim=True)
        return scalar.expand(-1, -1, self.output_dim)


def _model_config(*, head_type: str = "prototype") -> dict[str, object]:
    return {
        "type": "u_mask_beam_jepa",
        "modalities": ["image", "radar"],
        "d_model": 4,
        "num_classes": 4,
        "num_pred": 1,
        "dropout": 0.0,
        "fusion_type": "supervised_router",
        "head_type": head_type,
        "temporal_pooling": {"enabled": True, "type": "masked_mean"},
        "encoders": {
            "image": {"type": "u_mask_test_sequence", "output_dim": 4},
            "radar": {"type": "u_mask_test_sequence", "output_dim": 4},
        },
    }


def _loss_output(classes: int = 4, features: int = 5) -> dict[str, torch.Tensor]:
    return {
        "logits": torch.randn(3, 1, classes, requires_grad=True),
        "output_features": torch.randn(3, features, requires_grad=True),
        "modality_features": torch.randn(3, 2, features, requires_grad=True),
        "missing_mask": torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.bool),
    }


def test_t2_and_s1_resolve_to_the_retained_surface() -> None:
    t2 = u_mask_beam_jepa_config(load_config(ROOT / "configs/mmw/t2.yaml"))
    s1 = u_mask_beam_jepa_config(load_config(ROOT / "configs/mmw/s1.yaml"))

    assert t2["enabled"] is True
    assert t2["use_beam_prototype_alignment"] is True
    assert t2["superset_consistency"]["enabled"] is True
    assert t2["superset_consistency"]["confidence_gated_kl"] is True
    assert t2["router_supervision"] == "oracle"
    assert t2["router_oracle_weight"] == pytest.approx(0.1)
    assert s1["superset_consistency"]["enabled"] is False


def test_masked_mean_supervised_router_masks_missing_cells() -> None:
    model = MODELS.build(_model_config())
    output = model(
        image_batch=torch.tensor([[[1.0], [3.0], [5.0]]]),
        radar_batch=torch.tensor([[[2.0], [4.0], [6.0]]]),
        modality_temporal_mask=torch.tensor([[[1, 1], [1, 0], [0, 0]]], dtype=torch.bool),
        missing_mask=torch.tensor([[True, True]]),
    )

    assert torch.equal(output["input_features"][0, 0], torch.full((4,), 2.0))
    assert torch.equal(output["input_features"][0, 1], torch.full((4,), 2.0))
    assert output["logits"].shape == (1, 1, 4)
    assert output["unimodal_logits"].shape == (1, 2, 4)
    assert torch.allclose(output["supervised_router_gate_weights"].sum(dim=1), torch.ones(1))
    assert output["metadata"]["temporal_pooling_type"] == "masked_mean"
    assert output["metadata"]["fusion_type"] == "supervised_router"


def test_bpa_and_cma_are_separate_t2_ablation_terms() -> None:
    labels = torch.tensor([[0], [1], [2]])
    bpa_output = _loss_output()
    bpa = u_mask_beam_jepa_loss(
        bpa_output,
        labels,
        prototype_bank=BeamPrototypeBank(5, 4),
        use_beam_prototype_alignment=True,
        lambda_proto=0.2,
        lambda_modality_proto=0.1,
        prototype_target_circular=True,
    )
    bpa["loss"].backward()
    assert torch.isfinite(bpa["loss"])
    assert bpa["diagnostics"]["loss/prototype_total"] > 0.0

    cma_output = _loss_output()
    cma = u_mask_beam_jepa_loss(
        cma_output,
        labels,
        use_amber_cma_analogue=True,
        lambda_amber_cma=0.2,
        sample_ids=["sunny:a", "rainy:a", "foggy:b"],
    )
    cma["loss"].backward()
    assert torch.isfinite(cma["loss_amber_cma"])
    assert cma["diagnostics"]["loss/amber_cma_weighted"] > 0.0

    with pytest.raises(ValueError, match="mutually exclusive"):
        u_mask_beam_jepa_loss(
            _loss_output(),
            labels,
            prototype_bank=BeamPrototypeBank(5, 4),
            use_beam_prototype_alignment=True,
            use_amber_cma_analogue=True,
        )


def test_superset_kl_uses_a_stop_gradient_reference() -> None:
    student = torch.tensor([[[2.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]]], requires_grad=True)
    reference = torch.tensor([[[3.0, 0.0, 0.0]], [[0.0, 3.0, 0.0]]], requires_grad=True)
    output = {
        "logits": student,
        "output_features": torch.randn(2, 3, requires_grad=True),
        "modality_features": torch.randn(2, 1, 3, requires_grad=True),
        "missing_mask": torch.ones(2, 1, dtype=torch.bool),
    }
    result = u_mask_beam_jepa_loss(
        output,
        torch.tensor([[0], [1]]),
        superset_output={"logits": reference},
        use_superset_confidence_gated_kl=True,
        lambda_superset_consistency=0.2,
        superset_temperature=2.0,
    )
    result["loss"].backward()

    assert result["diagnostics"]["superset_consistency/gate_active_ratio"] == pytest.approx(1.0)
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert reference.grad is None
