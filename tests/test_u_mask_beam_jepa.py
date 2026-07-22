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


def test_t2_and_s1_resolve_to_the_current_surface() -> None:
    t2 = u_mask_beam_jepa_config(load_config(ROOT / "configs/mmw/t2.yaml"))
    s1 = u_mask_beam_jepa_config(load_config(ROOT / "configs/mmw/s1.yaml"))
    deep = u_mask_beam_jepa_config(load_config(ROOT / "configs/deepsense6g/t2.yaml"))

    assert t2["use_beam_prototype_alignment"] is True
    assert t2["superset_consistency"]["enabled"] is True
    assert t2["missing_mask"] == {"mode": "external"}
    assert t2["bcacl"] == {"enabled": False}
    assert t2["cmsbl"] == {"enabled": False}
    assert s1["superset_consistency"]["enabled"] is False
    assert deep["missing_mask"]["p_missing"] == [0.25, 0.25, 0.25, 0.1]


@pytest.mark.parametrize("field", ["pcer", "pgcd", "dynamic_router", "router_quality_pairing"])
def test_retired_loss_fields_fail_closed(field: str) -> None:
    with pytest.raises(ValueError, match="does not support fields"):
        u_mask_beam_jepa_config({"loss": {"u_mask_beam_jepa": {field: {"enabled": True}}}})


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
    assert output["supervised_router_gate_weights"][0, 1] > 0
    assert output["metadata"]["temporal_pooling_type"] == "masked_mean"

    dropped = model(
        image_batch=torch.ones(1, 3, 1),
        radar_batch=torch.ones(1, 3, 1),
        missing_mask=torch.tensor([[True, False]]),
    )
    assert dropped["supervised_router_gate_weights"].tolist() == [[1.0, 0.0]]


def test_only_the_active_head_is_trainable() -> None:
    prototype = MODELS.build(_model_config(head_type="prototype"))
    classifier = MODELS.build(_model_config(head_type="classifier"))

    assert all(not parameter.requires_grad for parameter in prototype.classifier.parameters())
    assert all(parameter.requires_grad for parameter in prototype.prototype_bank.parameters())
    assert all(parameter.requires_grad for parameter in classifier.classifier.parameters())
    assert all(not parameter.requires_grad for parameter in classifier.prototype_bank.parameters())


def test_loss_combines_fusion_bpa_and_router_supervision() -> None:
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
    assert result["diagnostics"]["prototype/modality_sample_count"] == 3.0
    assert any(parameter.grad is not None for parameter in model.supervised_router.parameters())


def test_bpa_and_cma_remain_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        u_mask_beam_jepa_loss(
            {"logits": torch.zeros(1, 1, 2)},
            torch.zeros(1, 1, dtype=torch.long),
            use_beam_prototype_alignment=True,
            use_amber_cma_analogue=True,
        )


def test_non_current_fusion_and_pooling_are_rejected() -> None:
    with pytest.raises(ValueError, match="supervised_router only"):
        MODELS.build({**_model_config(), "fusion_type": "uniform_mean"})
    with pytest.raises(ValueError, match="masked_mean"):
        MODELS.build(
            {**_model_config(), "temporal_pooling": {"enabled": True, "type": "masked_attention"}}
        )
