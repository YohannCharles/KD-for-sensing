from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from kd_sensing.engine.model_output import adapt_model_output
from kd_sensing.engine.training_extensions import BatchState, ForwardControls
from kd_sensing.losses.pcer_temporal_fusion import (
    onpolicy_block_router_targets,
    onpolicy_modality_router_targets,
    standalone_quality_router_targets,
)
from kd_sensing.losses.u_mask_beam_jepa import UMaskBeamJEPATrainingExtension, _balanced_lomo_modality
from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config
from kd_sensing.registries import ENCODERS, MODELS

import kd_sensing.models.u_mask_beam_jepa  # noqa: F401


@ENCODERS.register("direction_test_sequence", force=True)
class _SequenceEncoder(nn.Module):
    def __init__(self, output_dim: int = 4, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.projection = nn.Linear(1, self.output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(values.float().mean(dim=-1, keepdim=True))


def _model_config(mode: str) -> dict:
    return {
        "type": "u_mask_beam_jepa",
        "modalities": ["image", "radar"],
        "seq_length": 3,
        "d_model": 4,
        "num_classes": 8,
        "num_pred": 1,
        "dropout": 0.0,
        "fusion_type": "supervised_router" if mode == "evidence_only" else "uniform_mean",
        "head_type": "prototype",
        "temporal_pooling": {"enabled": True, "type": "masked_mean"},
        "encoders": {
            "image": {"type": "direction_test_sequence", "output_dim": 4},
            "radar": {"type": "direction_test_sequence", "output_dim": 4},
        },
        "pcer": {"mode": mode, "hidden_dim": 8, "embedding_dim": 2, "dropout": 0.0},
    }


def _inputs(full: bool = False) -> dict[str, torch.Tensor]:
    mask = torch.ones(2, 3, 2, dtype=torch.bool)
    if not full:
        mask[0, 1, 0] = False
        mask[1, 2, 1] = False
    return {
        "image_batch": torch.randn(2, 3, 2),
        "radar_batch": torch.randn(2, 3, 2),
        "modality_temporal_mask": mask,
        "missing_mask": mask.any(dim=1),
    }


@pytest.mark.parametrize("mode", ("evidence_only", "block_router", "hierarchical_router", "mask_residual_router"))
def test_direction_router_modes_preserve_availability_and_normalization(mode: str) -> None:
    model = MODELS.build(_model_config(mode))
    output = model(**_inputs())
    available = output["pcer_block_availability"]
    weights = output["pcer_block_router_weights"]
    assert weights.shape == (2, 6)
    assert torch.equal(weights.masked_select(~available), torch.zeros_like(weights.masked_select(~available)))
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)
    if mode == "evidence_only":
        assert output["router_gate_weights"].shape == (2, 2)
    if mode == "hierarchical_router":
        alpha, beta = output["pcer_alpha"], output["pcer_beta"]
        assert torch.allclose(alpha.sum(dim=1), torch.ones(2), atol=1e-6)
        assert torch.allclose(beta.sum(dim=2), torch.ones(2, 2), atol=1e-6)
    if mode == "mask_residual_router":
        residual = output["pcer_dynamic_residual"]
        valid = available.float()
        assert torch.allclose((residual * valid).sum(dim=1), torch.zeros(2), atol=1e-5)
        output["logits"].sum().backward()
        assert model.pcer_router.residual_logit_scale.grad is not None


def _target_kwargs() -> dict:
    return {
        "beam_label_sigma": 0.2,
        "circular": True,
        "topology_id": "cyclic_index_v1",
        "topology_permutation": None,
    }


def test_standalone_quality_target_prefers_unique_correct_block() -> None:
    evidence = torch.zeros(1, 4, 8)
    evidence[0, 0, 0] = 12.0
    evidence[0, 1:, 4] = 12.0
    target, quality = standalone_quality_router_targets(
        evidence, torch.ones(1, 4, dtype=torch.bool), torch.tensor([0]),
        quality_temperature=0.5, **_target_kwargs()
    )
    assert target.argmax(dim=1).item() == 0
    assert quality.argmax(dim=1).item() == 0
    assert not target.requires_grad and torch.isfinite(target).all()


def test_onpolicy_targets_rerun_only_cached_router_and_keep_order() -> None:
    batch, timesteps, modalities, classes, width = 1, 3, 2, 8, 4
    blocks = timesteps * modalities
    evidence = torch.zeros(batch, blocks, classes)
    evidence[0, 0, 0] = 12.0
    evidence[0, 1:, 4] = 8.0
    features = torch.randn(batch, blocks, width)
    available = torch.ones(batch, blocks, dtype=torch.bool)
    calls = []

    def route_fn(cached_features, cached_evidence, cached_available):
        calls.append((cached_features.shape, cached_evidence.shape, cached_available.shape))
        weights = cached_available.float() / cached_available.sum(dim=1, keepdim=True)
        alpha = weights.reshape(-1, timesteps, modalities).sum(dim=1)
        return {"weights": weights, "alpha": alpha}

    full = available.float() / blocks
    block_target, block_contribution = onpolicy_block_router_targets(
        features, evidence, available, full, torch.tensor([0]), route_fn=route_fn,
        contribution_temperature=0.5, contribution_clip=None, **_target_kwargs()
    )
    modality_target, _ = onpolicy_modality_router_targets(
        features, evidence, available, full, torch.tensor([0]), route_fn=route_fn,
        num_timesteps=timesteps, num_modalities=modalities,
        contribution_temperature=0.5, contribution_clip=None, **_target_kwargs()
    )
    assert calls == [
        ((blocks, blocks, width), (blocks, blocks, classes), (blocks, blocks)),
        ((modalities, blocks, width), (modalities, blocks, classes), (modalities, blocks)),
    ]
    assert block_target.argmax(dim=1).item() == 0
    assert block_contribution.argmax(dim=1).item() == 0
    assert modality_target.argmax(dim=1).item() == 0
    assert not block_target.requires_grad and not modality_target.requires_grad


def _resolved_evidence_config() -> dict:
    return {
        "model": {"primary": _model_config("evidence_only")},
        "temporal_missing": {
            "enabled": True,
            "mode": "pcer_curriculum",
            "preserve_unmasked_for_superset": True,
        },
        "loss": {
            "u_mask_beam_jepa": {
                "enabled": True,
                "use_beam_prototype_alignment": True,
                "lambda_proto": 0.2,
                "lambda_modality_proto": 0.1,
                "beam_label_sigma": 0.5,
                "prototype_target_circular": True,
                "router_oracle_weight": 0.0,
                "superset_consistency": {"enabled": False},
                "pcer": {
                    "lambda_mask": 0.5,
                    "lambda_route": 0.0,
                    "route_target": "none",
                    "distill_temperature": 2.0,
                    "contribution_temperature": 0.5,
                    "evidence_learning": {
                        "enabled": True,
                        "lambda_lomo": 0.5,
                        "lambda_unimodal": 0.1,
                        "distill_temperature": 2.0,
                    },
                },
            }
        },
    }


def test_b7_evidence_losses_are_finite_and_balanced_selector_cycles() -> None:
    cfg = _resolved_evidence_config()
    model = MODELS.build(cfg["model"]["primary"])
    masked = model(**_inputs())
    full = model(**_inputs(full=True))
    lomo_inputs = _inputs(full=True)
    lomo_inputs["modality_temporal_mask"][:, :, 0] = False
    lomo_inputs["missing_mask"] = lomo_inputs["modality_temporal_mask"].any(dim=1)
    lomo = model(**lomo_inputs)
    extension = UMaskBeamJEPATrainingExtension()
    state = {
        "config": u_mask_beam_jepa_config(cfg),
        "online_superset": {
            "logits": full["logits"].detach(),
            "modality_temporal_mask": full["modality_temporal_mask"],
        },
        "online_lomo": {
            "logits": lomo["logits"],
            "modality_temporal_mask": lomo["modality_temporal_mask"],
            "modality_index": 0,
        },
    }
    labels = torch.tensor([[0], [1]])
    batch_state = BatchState(
        epoch=0, step=0, batch={}, labels=labels,
        primary_output=adapt_model_output(masked), primary_logits=masked["logits"], controls=ForwardControls()
    )
    result = extension.compute_base_loss(SimpleNamespace(primary_model=model), state, batch_state)
    assert result is not None and torch.isfinite(result.total_loss)
    assert result.diagnostics["loss/pcer_lomo_weighted"] >= 0
    assert result.diagnostics["loss/pcer_unimodal_aux_weighted"] > 0
    result.total_loss.backward()
    assert model.prototype_bank.prototypes.grad is not None
    assert [_balanced_lomo_modality(0, step, 4) for step in range(8)] == [0, 1, 2, 3, 0, 1, 2, 3]
