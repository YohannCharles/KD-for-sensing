import pytest
import torch
import torch.nn as nn

from kd_sensing.losses.pcpf_temporal_risk import pcpf_temporal_risk_loss
from kd_sensing.losses.pcpf_temporal_risk_config import pcpf_temporal_risk_config
from kd_sensing.models.pcpf_temporal_risk import PCPFTemporalRiskFusion
from kd_sensing.registries import ENCODERS


@ENCODERS.register("pcpf_loss_test_sequence", force=True)
class _LossTestSequenceEncoder(nn.Module):
    def __init__(self, output_dim: int = 64, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.projection = nn.Linear(64, self.output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


def _model(stage: str) -> PCPFTemporalRiskFusion:
    encoders = {
        name: {"type": "pcpf_loss_test_sequence", "output_dim": 64}
        for name in ("image", "radar", "gps", "lidar")
    }
    return PCPFTemporalRiskFusion(
        encoders=encoders,
        training_stage=stage,
        fusion_mode="pcpf_analytic",
        temporal_transformer={"dropout": 0.0},
    )


def _inputs() -> dict[str, torch.Tensor]:
    payload = {f"{name}_batch": torch.randn(4, 5, 64) for name in ("image", "radar", "gps", "lidar")}
    mask = torch.ones(4, 5, 4, dtype=torch.bool)
    mask[0, :, 3] = False
    mask[1, :3, 0] = False
    payload["modality_temporal_mask"] = mask
    return payload


def _config(stage: str) -> dict:
    cfg = {
        "model": {"primary": {"training_stage": stage}},
        "loss": {
            "pcpf_temporal_risk": {
                "enabled": True,
                "prototype_topology": "cyclic_index_v1",
                "stage_preparation": {"enabled": True},
            }
        },
    }
    return pcpf_temporal_risk_config(cfg)


def test_stage1_loss_updates_expert_and_shared_prototype_only() -> None:
    model = _model("stage1_expert").train()
    output = model(**_inputs())

    result = pcpf_temporal_risk_loss(
        output,
        torch.tensor([[0], [1], [63], [7]]),
        prototype_bank=model.prototype_bank,
        config=_config("stage1_expert"),
    )
    result["loss"].backward()

    assert torch.isfinite(result["loss"])
    assert model.prototype_bank.prototypes.grad is not None
    assert torch.isfinite(model.prototype_bank.prototypes.grad).all()
    assert all(parameter.grad is None for parameter in model.probability_head.parameters())
    assert model.risk_coefficient_raw.grad is None
    assert model.temperature_raw.grad is None


def test_stage2_loss_updates_evidence_and_risk_without_fused_ce() -> None:
    model = _model("stage2_risk").train()
    output = model(**_inputs())

    result = pcpf_temporal_risk_loss(
        output,
        torch.tensor([[0], [1], [63], [7]]),
        prototype_bank=model.prototype_bank,
        config=_config("stage2_risk"),
    )
    result["loss"].backward()

    assert not result["risk_target"].requires_grad
    assert model.risk_coefficient_raw.grad is not None
    assert any(parameter.grad is not None for parameter in model.probability_head.parameters())
    assert model.prototype_bank.prototypes.grad is None
    assert all(parameter.grad is None for parameter in model.encoders.parameters())
    assert "loss/pcpf_fusion_nll" not in result["diagnostics"]
    assert "loss/pcpf_concentration" in result["diagnostics"]
    assert "loss/pcpf_kl" not in result["diagnostics"]


def test_retired_gaussian_loss_fields_are_rejected() -> None:
    cfg = {
        "model": {"primary": {"training_stage": "stage2_risk"}},
        "loss": {"pcpf_temporal_risk": {"enabled": True, "beta_kl": 1e-4}},
    }

    with pytest.raises(ValueError, match="unsupported fields"):
        pcpf_temporal_risk_config(cfg)
