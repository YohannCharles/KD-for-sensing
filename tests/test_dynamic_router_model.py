import pytest
import torch
import torch.nn as nn

from kd_sensing.registries import ENCODERS, MODELS

import kd_sensing.models.u_mask_beam_jepa  # noqa: F401


@ENCODERS.register("dynamic_router_test_sequence", force=True)
class _SequenceEncoder(nn.Module):
    def __init__(self, output_dim: int = 4, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, steps = values.shape[:2]
        scalar = values.float().reshape(batch, steps, -1).mean(dim=-1, keepdim=True)
        return scalar.expand(-1, -1, self.output_dim)


def _model_config() -> dict[str, object]:
    return {
        "type": "u_mask_beam_jepa",
        "modalities": ["image", "radar"],
        "d_model": 4,
        "num_classes": 4,
        "num_pred": 1,
        "dropout": 0.0,
        "fusion_type": "supervised_router",
        "head_type": "prototype",
        "temporal_pooling": {"enabled": True, "type": "masked_mean"},
        "encoders": {
            "image": {"type": "dynamic_router_test_sequence", "output_dim": 4},
            "radar": {"type": "dynamic_router_test_sequence", "output_dim": 4},
        },
    }


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "image_batch": torch.tensor([[[1.0], [3.0], [5.0]], [[2.0], [4.0], [6.0]]]),
        "radar_batch": torch.tensor([[[2.0], [4.0], [7.0]], [[1.0], [5.0], [8.0]]]),
        "modality_temporal_mask": torch.tensor(
            [
                [[1, 1], [1, 1], [1, 0]],
                [[1, 1], [0, 1], [1, 1]],
            ],
            dtype=torch.bool,
        ),
        "missing_mask": torch.ones(2, 2, dtype=torch.bool),
    }


def _candidate_config(variant: str) -> dict[str, object]:
    config = _model_config()
    config.update(
        {
            "router_variant": variant,
            "router_calibration_only": True,
            "router_variant_config": {
                "topology_id": "cyclic_index_v1",
                "circular": True,
                "dropout": 0.0,
            },
        }
    )
    return config


def test_current_router_default_and_explicit_state_dict_are_identical() -> None:
    torch.manual_seed(17)
    implicit = MODELS.build(_model_config()).eval()
    torch.manual_seed(17)
    explicit_config = _model_config()
    explicit_config["router_variant"] = "current"
    explicit = MODELS.build(explicit_config).eval()
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    for name, value in implicit.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name]), name
    with torch.no_grad():
        first = implicit(**_inputs())
        second = explicit(**_inputs())
    assert torch.equal(first["logits"], second["logits"])


@pytest.mark.parametrize("variant", ["patr", "h2r", "core", "unified_hpr"])
def test_dynamic_router_variants_preserve_output_and_reroute_contract(variant: str) -> None:
    model = MODELS.build(_candidate_config(variant)).train()
    output = model(**_inputs(), return_router_state=True)
    weights = output["router_gate_weights"]
    assert output["router_variant"] == variant
    assert weights.shape == (2, 2)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))
    assert torch.allclose(output["logits"][:, 0], (weights.unsqueeze(-1) * output["unimodal_logits"]).sum(dim=1))
    assert output["router_temporal_weights"].shape == (2, 3, 2)
    assert torch.allclose(output["router_temporal_weights"].sum(dim=1), torch.ones(2, 2))
    assert output["reference_router_gate_weights"].shape == weights.shape
    assert all(not parameter.requires_grad for parameter in model.encoders.parameters())
    assert model.prototype_reliability_router is not None
    assert all(module.training is False for module in model.encoders.modules())

    rerouted = model.route_from_candidate_state(output["candidate_router_state"])
    assert rerouted["fused_logits"].shape == (2, 4)
    assert torch.allclose(rerouted["router_gate_weights"].sum(dim=1), torch.ones(2))
    rerouted["fused_logits"].sum().backward()
    gradients = [
        parameter.grad
        for parameter in model.prototype_reliability_router.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_h2r_temporal_gate_receives_gradient_before_pooling() -> None:
    model = MODELS.build(_candidate_config("h2r"))
    output = model(**_inputs(), return_router_state=True)
    rerouted = model.route_from_candidate_state(output["candidate_router_state"])
    rerouted["fused_logits"].square().mean().backward()
    health = model.prototype_reliability_router.frame_health_head
    assert health is not None
    assert health[-1].weight.grad is not None
    assert health[-1].weight.grad.abs().sum().item() > 0.0


def test_dynamic_router_rejects_independent_incompatible_modes() -> None:
    config = _candidate_config("patr")
    config["head_type"] = "classifier"
    with pytest.raises(ValueError, match="require supervised_router fusion and prototype head"):
        MODELS.build(config)
