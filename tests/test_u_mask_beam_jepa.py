from pathlib import Path

import pytest
import torch
import torch.nn as nn

from kd_sensing.config import load_config
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
        scalar = values.float().reshape(*values.shape[:2], -1).mean(dim=-1, keepdim=True)
        return scalar.expand(-1, -1, self.output_dim)


def _model_config(*, head_type: str = "prototype") -> dict[str, object]:
    return {
        "type": "u_mask_beam_jepa",
        "modalities": ["image", "radar"],
        "d_model": 4,
        "num_classes": 4,
        "num_pred": 1,
        "seq_length": 3,
        "dropout": 0.0,
        "fusion_type": "supervised_router",
        "head_type": head_type,
        "temporal_pooling": {"enabled": True, "type": "masked_mean"},
        "encoders": {
            "image": {"type": "u_mask_test_sequence", "output_dim": 4},
            "radar": {"type": "u_mask_test_sequence", "output_dim": 4},
        },
    }


def test_u0_and_deepsense_recipes_resolve_to_the_retained_surface() -> None:
    u0 = u_mask_beam_jepa_config(load_config(ROOT / "configs/mmw/u0.yaml"))
    deep = u_mask_beam_jepa_config(load_config(ROOT / "configs/deepsense6g/t2.yaml"))

    assert u0["use_beam_prototype_alignment"] is True
    assert u0["superset_consistency"]["enabled"] is True
    assert u0["missing_mask"] == {"mode": "external"}
    assert deep["enabled"] is True
    with pytest.raises(ValueError, match="retired training sections"):
        u_mask_beam_jepa_config({"bcacl": {"enabled": True}})


def test_model_masks_temporal_cells_and_routes_only_available_modalities() -> None:
    model = MODELS.build(_model_config())
    output = model(
        image_batch=torch.tensor([[[1.0], [3.0], [5.0]]]),
        radar_batch=torch.tensor([[[2.0], [4.0], [6.0]]]),
        modality_temporal_mask=torch.tensor([[[1, 1], [1, 0], [0, 0]]], dtype=torch.bool),
        missing_mask=torch.tensor([[True, True]]),
    )

    assert torch.equal(output["input_features"], torch.full((1, 2, 4), 2.0))
    assert output["logits"].shape == (1, 1, 4)
    assert torch.allclose(output["supervised_router_gate_weights"].sum(dim=1), torch.ones(1))
    assert output["metadata"]["architecture_category"] == "u0_temporal_supervised_router"


def test_u0_loss_keeps_only_fusion_bpa_and_router_supervision() -> None:
    model = MODELS.build(_model_config())
    output = model(
        image_batch=torch.randn(2, 3, 1),
        radar_batch=torch.randn(2, 3, 1),
        missing_mask=torch.tensor([[True, True], [True, False]]),
    )
    result = u_mask_beam_jepa_loss(
        output,
        torch.tensor([[0], [1]]),
        prototype_bank=model.prototype_bank,
        use_beam_prototype_alignment=True,
        lambda_proto=0.2,
        lambda_modality_proto=0.1,
        router_oracle_weight=0.1,
    )
    result["loss"].backward()

    assert torch.isfinite(result["loss"])
    assert result["diagnostics"]["router_oracle_enabled"] == 1.0
    assert any(parameter.grad is not None for parameter in model.supervised_router.parameters())


def test_non_u0_fusion_and_pooling_are_rejected() -> None:
    with pytest.raises(ValueError, match="supervised_router only"):
        MODELS.build({**_model_config(), "fusion_type": "uniform_mean"})
    with pytest.raises(ValueError, match="masked_mean"):
        MODELS.build({**_model_config(), "temporal_pooling": {"enabled": True, "type": "masked_attention"}})
