from pathlib import Path

import pytest
import torch
import torch.nn as nn

from kd_sensing.config import load_config
from kd_sensing.engine.model_output import adapt_model_output
from kd_sensing.engine.optim import build_model
from kd_sensing.engine.prediction_objectives import PredictionTargets, compute_prediction_loss
from kd_sensing.models.modular import ModularSequenceModel
from kd_sensing.registries import ENCODERS


ROOT = Path(__file__).resolve().parents[1]


@ENCODERS.register("twc_baseline_test_identity", force=True)
class _IdentityEncoder(nn.Module):
    def __init__(self, output_dim: int = 8, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[..., : self.output_dim]


@pytest.mark.parametrize(
    ("config_name", "core_type"),
    (("masktrain_cls.yaml", "masktrain_mean_fusion"), ("amr_net_4m.yaml", "amr_gaussian_uncertainty_fusion")),
)
def test_new_twc_baseline_configs_build(config_name: str, core_type: str) -> None:
    cfg = load_config(ROOT / "configs" / "mmw" / config_name)
    model = build_model(cfg["model"]["primary"])

    assert isinstance(model, ModularSequenceModel)
    assert model.representation_core_config["type"] == core_type
    assert model.modalities == ("image", "radar", "gps", "lidar")


def test_masktrain_mean_ignores_unavailable_features() -> None:
    model = _model("masktrain_mean_fusion")
    batch = _batch()
    availability = torch.ones(2, 4, 3, dtype=torch.bool)
    availability[:, 1] = False

    first = model(**batch, modality_temporal_mask=availability.permute(0, 2, 1))
    batch["radar_batch"] = torch.full_like(batch["radar_batch"], 1.0e6)
    second = model(**batch, modality_temporal_mask=availability.permute(0, 2, 1))

    assert torch.allclose(first["logits"], second["logits"])
    assert torch.equal(first["fusion_weights"][:, 1], torch.zeros_like(first["fusion_weights"][:, 1]))
    assert torch.allclose(first["fusion_weights"].sum(dim=1), torch.ones(2, 3))


def test_amr_adapted_masks_uncertainty_weights_and_backpropagates_auxiliary() -> None:
    model = _model("amr_gaussian_uncertainty_fusion")
    model.train()
    availability = torch.ones(2, 4, 3, dtype=torch.bool)
    availability[:, 2] = False
    output = adapt_model_output(model(**_batch(), modality_temporal_mask=availability.permute(0, 2, 1)))
    beam = output.logits.square().mean()
    cfg = {"loss": {"auxiliary": {"amr_adapted": {"enabled": True, "kl_weight": 0.01, "consistency_weight": 0.05}}}}

    bundle = compute_prediction_loss(
        output,
        PredictionTargets(labels=torch.zeros(2, 3, dtype=torch.long)),
        cfg,
        reference=output.logits,
        beam_total_loss=beam,
    )
    bundle.total.backward()

    weights = output.diagnostics["fusion_weights"]
    assert torch.equal(weights[:, 2], torch.zeros_like(weights[:, 2]))
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 3))
    assert bundle.diagnostics["loss/amr_adapted_total"] > 0.0
    assert model.representation_core.mu_heads[0].weight.grad is not None


@pytest.mark.parametrize("core_type", ("masktrain_mean_fusion", "amr_gaussian_uncertainty_fusion"))
def test_baseline_core_emits_zero_fusion_weights_for_an_empty_time_step(core_type: str) -> None:
    model = _model(core_type)
    model.eval()
    availability = torch.ones(2, 4, 3, dtype=torch.bool)
    availability[:, :, 1] = False

    output = adapt_model_output(model(**_batch(), modality_temporal_mask=availability.permute(0, 2, 1)))

    weights = output.diagnostics["fusion_weights"]
    assert torch.equal(weights[:, :, 1], torch.zeros_like(weights[:, :, 1]))
    assert torch.isfinite(output.logits).all()


def _model(core_type: str) -> ModularSequenceModel:
    modalities = ["image", "radar", "gps", "lidar"]
    core = {"type": core_type, "d_model": 8, "modality_count": 4}
    if core_type == "amr_gaussian_uncertainty_fusion":
        core.update({"latent_dim": 8, "dropout": 0.0})
    return ModularSequenceModel(
        modalities=modalities,
        encoders={name: {"type": "twc_baseline_test_identity", "output_dim": 8} for name in modalities},
        projectors={name: {"type": "identity"} for name in modalities},
        representation_core=core,
        heads={"beam": {"type": "beam_head", "dropout": 0.0}},
        feature_size=8,
        d_model=8,
        num_classes=6,
    )


def _batch() -> dict[str, torch.Tensor]:
    return {f"{name}_batch": torch.randn(2, 3, 8) for name in ("image", "radar", "gps", "lidar")}
