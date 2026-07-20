from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from kd_sensing.losses.pcer_temporal_fusion import (
    counterfactual_router_loss,
    counterfactual_router_targets,
    prototype_evidence_consistency_loss,
)
from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config
from kd_sensing.losses.u_mask_beam_jepa import UMaskBeamJEPATrainingExtension
from kd_sensing.engine.model_output import adapt_model_output
from kd_sensing.engine.training_extensions import BatchState, ForwardControls
from kd_sensing.registries import ENCODERS, MODELS

import kd_sensing.models.u_mask_beam_jepa  # noqa: F401


@ENCODERS.register("pcer_test_sequence", force=True)
class _SequenceEncoder(nn.Module):
    def __init__(self, output_dim: int = 4, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.calls = 0

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        scalar = values.float().mean(dim=-1, keepdim=True)
        return scalar.expand(-1, -1, self.output_dim)


def _model_config(mode: str | None = None) -> dict:
    config = {
        "type": "u_mask_beam_jepa",
        "modalities": ["image", "radar"],
        "seq_length": 3,
        "d_model": 4,
        "num_classes": 4,
        "num_pred": 1,
        "dropout": 0.0,
        "fusion_type": "supervised_router" if mode is None else "uniform_mean",
        "head_type": "prototype",
        "temporal_pooling": {"enabled": True, "type": "masked_mean"},
        "encoders": {
            "image": {"type": "pcer_test_sequence", "output_dim": 4},
            "radar": {"type": "pcer_test_sequence", "output_dim": 4},
        },
    }
    if mode is not None:
        config["pcer"] = {"mode": mode, "hidden_dim": 8, "embedding_dim": 2, "dropout": 0.0}
    return config


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "image_batch": torch.tensor([[[1.0], [2.0], [3.0]], [[3.0], [2.0], [1.0]]]),
        "radar_batch": torch.tensor([[[4.0], [5.0], [6.0]], [[6.0], [5.0], [4.0]]]),
        "modality_temporal_mask": torch.tensor(
            [[[1, 1], [1, 0], [0, 1]], [[1, 0], [1, 1], [1, 1]]], dtype=torch.bool
        ),
        "missing_mask": torch.ones(2, 2, dtype=torch.bool),
    }


def test_default_path_has_no_pcer_parameters_and_is_exactly_compatible() -> None:
    torch.manual_seed(11)
    implicit = MODELS.build(_model_config()).eval()
    torch.manual_seed(11)
    explicit_config = _model_config()
    explicit_config["pcer"] = None
    explicit = MODELS.build(explicit_config).eval()
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    assert not any(name.startswith("pcer_router") for name in implicit.state_dict())
    with torch.no_grad():
        first = implicit(**_inputs())
        second = explicit(**_inputs())
    assert torch.equal(first["logits"], second["logits"])


@pytest.mark.parametrize("mode", ("evidence_static", "counterfactual_router"))
def test_pcer_forward_masks_blocks_and_reuses_each_encoder_once(mode: str) -> None:
    model = MODELS.build(_model_config(mode))
    output = model(**_inputs())
    availability = output["pcer_block_availability"]
    weights = output["pcer_block_router_weights"]
    assert output["pcer_block_features"].shape == (2, 6, 4)
    assert output["pcer_block_evidence_logits"].shape == (2, 6, 4)
    assert weights.shape == (2, 6)
    assert torch.equal(weights.masked_select(~availability), torch.zeros_like(weights.masked_select(~availability)))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))
    assert torch.allclose(
        output["logits"][:, 0],
        (weights.unsqueeze(-1) * output["pcer_block_evidence_logits"]).sum(dim=1),
    )
    assert model.encoders["image"].calls == model.encoders["radar"].calls == 1


def test_consistency_and_counterfactual_route_are_finite_with_router_gradient() -> None:
    model = MODELS.build(_model_config("counterfactual_router"))
    masked = model(**_inputs())
    full_inputs = _inputs()
    full_inputs["modality_temporal_mask"] = torch.ones(2, 3, 2, dtype=torch.bool)
    full = model(**full_inputs)
    consistency, diagnostics = prototype_evidence_consistency_loss(
        masked["logits"],
        full["logits"].detach(),
        masked["modality_temporal_mask"],
        full["modality_temporal_mask"],
        temperature=2.0,
    )
    target, contribution = counterfactual_router_targets(
        masked["pcer_block_evidence_logits"],
        masked["pcer_block_availability"],
        torch.tensor([[0], [1]]),
        beam_label_sigma=1.0,
        circular=True,
        topology_id="cyclic_index_v1",
        topology_permutation=None,
        contribution_temperature=0.5,
        contribution_clip=5.0,
    )
    route, route_diagnostics = counterfactual_router_loss(
        masked["pcer_block_router_weights"], target, masked["pcer_block_availability"]
    )
    total = consistency + 0.2 * route
    total.backward()
    gradients = [parameter.grad for parameter in model.pcer_router.parameters() if parameter.grad is not None]
    assert torch.isfinite(total) and torch.isfinite(contribution.masked_select(masked["pcer_block_availability"])).all()
    assert diagnostics["pcer_mask_consistency_active_ratio"] == pytest.approx(1.0)
    assert route_diagnostics["pcer_router_top1_agreement"] >= 0.0
    assert gradients and sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_full_consistency_is_exactly_zero() -> None:
    logits = torch.randn(2, 1, 4, requires_grad=True)
    mask = torch.ones(2, 3, 2, dtype=torch.bool)
    loss, diagnostics = prototype_evidence_consistency_loss(logits, logits.detach(), mask, mask, temperature=2.0)
    assert loss.item() == pytest.approx(0.0)
    assert diagnostics["pcer_mask_consistency_active_ratio"] == pytest.approx(0.0)


def _resolved_config(mode: str) -> dict:
    route_weight = 0.2 if mode == "counterfactual_router" else 0.0
    return {
        "model": {"primary": {"fusion_type": "uniform_mean", "head_type": "prototype", "pcer": {"mode": mode}}},
        "temporal_missing": {
            "enabled": True,
            "mode": "pcer_curriculum",
            "preserve_unmasked_for_superset": True,
        },
        "loss": {
            "u_mask_beam_jepa": {
                "enabled": True,
                "use_beam_prototype_alignment": True,
                "router_oracle_weight": 0.0,
                "superset_consistency": {"enabled": False, "confidence_gated_kl": False, "kl_weight": 0.0},
                "pcer": {"lambda_mask": 0.5, "lambda_route": route_weight},
            }
        },
    }


def test_pcer_config_is_strict_and_modes_are_mutually_exclusive() -> None:
    static = u_mask_beam_jepa_config(_resolved_config("evidence_static"))["pcer"]
    full = u_mask_beam_jepa_config(_resolved_config("counterfactual_router"))["pcer"]
    assert static["lambda_route"] == 0.0
    assert full["lambda_route"] == pytest.approx(0.2)
    invalid = deepcopy(_resolved_config("counterfactual_router"))
    invalid["loss"]["u_mask_beam_jepa"]["router_oracle_weight"] = 0.1
    with pytest.raises(ValueError, match="uniform_mean fusion requires router_oracle_weight=0"):
        u_mask_beam_jepa_config(invalid)


def test_training_extension_adds_consistency_and_counterfactual_losses() -> None:
    cfg = _resolved_config("counterfactual_router")
    model = MODELS.build(_model_config("counterfactual_router"))
    masked = model(**_inputs())
    full_inputs = _inputs()
    full_inputs["modality_temporal_mask"] = torch.ones(2, 3, 2, dtype=torch.bool)
    with torch.no_grad():
        full = model(**full_inputs)
    extension = UMaskBeamJEPATrainingExtension()
    state = {
        "config": u_mask_beam_jepa_config(cfg),
        "online_superset": {
            "logits": full["logits"].detach(),
            "modality_temporal_mask": full["modality_temporal_mask"],
        },
    }
    labels = torch.tensor([[0], [1]])
    batch_state = BatchState(
        epoch=0,
        step=0,
        batch={"future_beam_power": None},
        labels=labels,
        primary_output=adapt_model_output(masked),
        primary_logits=masked["logits"],
        controls=ForwardControls(),
    )
    result = extension.compute_base_loss(SimpleNamespace(primary_model=model), state, batch_state)
    assert result is not None and torch.isfinite(result.total_loss)
    assert result.diagnostics["loss/pcer_mask_consistency_weighted"] >= 0.0
    assert result.diagnostics["loss/pcer_route_weighted"] >= 0.0
    result.total_loss.backward()
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum().gt(0))
        for parameter in model.pcer_router.parameters()
    )
