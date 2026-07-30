import pytest
import torch
import torch.nn as nn

from kd_sensing.models.pcpf_temporal_risk import PCPFTemporalRiskFusion
from kd_sensing.registries import ENCODERS


@ENCODERS.register("pcpf_test_sequence", force=True)
class _TestSequenceEncoder(nn.Module):
    def __init__(self, output_dim: int = 64, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.projection = nn.Linear(64, self.output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


def _model(stage: str, *, fusion_mode: str = "pcpf_analytic") -> PCPFTemporalRiskFusion:
    encoders = {name: {"type": "pcpf_test_sequence", "output_dim": 64} for name in ("image", "radar", "gps", "lidar")}
    return PCPFTemporalRiskFusion(
        encoders=encoders,
        training_stage=stage,
        fusion_mode=fusion_mode,
        temporal_transformer={"dropout": 0.0},
    )


def _inputs() -> dict[str, torch.Tensor]:
    values = {f"{name}_batch": torch.randn(3, 5, 64) for name in ("image", "radar", "gps", "lidar")}
    mask = torch.ones(3, 5, 4, dtype=torch.bool)
    mask[0, :, 2] = False
    mask[1, :3, 1] = False
    values["modality_temporal_mask"] = mask
    return values


@pytest.mark.parametrize(
    ("stage", "allowed_prefixes", "required_names"),
    [
        (
            "stage1_expert",
            ("encoders.", "encoder_projections.", "temporal_transformer.", "prototype_bank."),
            ("prototype_bank.prototypes",),
        ),
        (
            "stage2_risk",
            ("probability_head.", "risk_coefficient_raw", "risk_bias"),
            ("risk_coefficient_raw", "risk_bias"),
        ),
        (
            "stage3_fusion",
            ("temperature_raw", "tau_raw"),
            ("temperature_raw", "tau_raw"),
        ),
    ],
)
def test_stage_trainable_parameter_contract(
    stage: str,
    allowed_prefixes: tuple[str, ...],
    required_names: tuple[str, ...],
) -> None:
    model = _model(stage)
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert names
    assert all(name.startswith(allowed_prefixes) for name in names)
    assert all(any(name == required or name.startswith(f"{required}.") for name in names) for required in required_names)
    model.assert_trainable_parameters()


def test_full_model_masks_evidence_and_has_no_router_in_main_mode() -> None:
    model = _model("stage3_fusion").eval()

    output = model(**_inputs())

    assert output["unimodal_logits"].shape == (3, 4, 64)
    assert output["unimodal_probabilities"].shape == (3, 4, 64)
    assert output["risk_components"].shape == (3, 4, 4)
    assert output["fusion_weights"].shape == (3, 4)
    assert output["unimodal_probabilities"][0, 2].eq(0).all()
    assert output["raw_risk"][0, 2].item() == 0.0
    assert output["fusion_weights"][0, 2].item() == 0.0
    torch.testing.assert_close(output["fusion_weights"].sum(dim=1), torch.ones(3))
    assert not any("router" in name for name, _ in model.named_parameters())
    assert model.risk_coefficients.ge(0).all()


def test_probability_head_is_identity_initially_and_eval_is_deterministic() -> None:
    model = _model("stage2_risk").eval()
    inputs = _inputs()
    first = model(**inputs)
    second = model(**inputs)

    torch.testing.assert_close(first["probability_mu"], first["temporal_cls_features"])
    torch.testing.assert_close(first["probability_logvar"].mean(), torch.tensor(-4.0))
    torch.testing.assert_close(first["unimodal_probabilities"], second["unimodal_probabilities"])
    torch.testing.assert_close(first["raw_risk"], second["raw_risk"])
    torch.testing.assert_close(first["fusion_weights"], second["fusion_weights"])


def test_stage3_backward_reaches_only_temperature_and_tau() -> None:
    model = _model("stage3_fusion").train()
    output = model(**_inputs())
    labels = torch.tensor([0, 3, 9])
    loss = torch.nn.functional.nll_loss(output["logits"][:, 0], labels)

    loss.backward()

    gradients = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
    assert gradients == {"temperature_raw", "tau_raw"}
    assert torch.isfinite(model.temperature_raw.grad).all()
    assert torch.isfinite(model.tau_raw.grad).all()
