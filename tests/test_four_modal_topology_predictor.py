from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from kd_sensing.eval.topology_predictor import resolve_topology_missing_patterns
from kd_sensing.losses.four_modal_topology import four_modal_topology_config, four_modal_topology_loss
from kd_sensing.models.four_modal_topology_predictor import FourModalTopologyPredictor
from kd_sensing.registries import ENCODERS


@ENCODERS.register("test_topology_sequence", force=True)
class _TestSequenceEncoder(nn.Module):
    def __init__(self, *, output_dim: int = 64, **_kwargs) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.scale = nn.Parameter(torch.ones(self.output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        reduced = value.float().flatten(2).mean(dim=2, keepdim=True)
        return reduced * self.scale.view(1, 1, -1)


def _model(*, fusion_mode: str = "mean", topology_id: str = "cyclic_index_v1") -> FourModalTopologyPredictor:
    encoders = {
        name: {"type": "test_topology_sequence", "output_dim": 64}
        for name in ("image", "radar", "gps", "lidar")
    }
    return FourModalTopologyPredictor(
        modalities=("image", "radar", "gps", "lidar"),
        encoders=encoders,
        fusion_mode=fusion_mode,
        prototype_topology_id=topology_id,
        dropout=0.0,
        temporal_transformer={"num_layers": 1, "num_heads": 4, "dropout": 0.0},
    )


def _inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "image_batch": torch.randn(batch_size, 5, 3, 4, 4),
        "radar_batch": torch.randn(batch_size, 5, 2, 4, 4),
        "gps_batch": torch.randn(batch_size, 5, 3),
        "lidar_batch": torch.randn(batch_size, 5, 3, 4, 4),
    }


def _loss_config(*, topology: bool = True) -> dict:
    return four_modal_topology_config(
        {
            "loss": {
                "four_modal_topology": {
                    "enabled": True,
                    "unimodal_soft_weight": 0.5 if topology else 0.0,
                    "use_beam_prototype_alignment": topology,
                    "lambda_proto": 0.2 if topology else 0.0,
                    "lambda_modality_proto": 0.1 if topology else 0.0,
                    "prototype_topology": {"id": "cyclic_index_v1"},
                }
            }
        }
    )


def test_native_predictor_has_one_stage_four_modalities_and_stateless_uncertainty() -> None:
    model = _model().eval()
    output = model(**_inputs(), missing_mask=torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]]))

    assert output["logits"].shape == (2, 1, 64)
    assert output["unimodal_logits"].shape == (2, 4, 64)
    assert output["fused_probability"].shape == (2, 64)
    assert torch.allclose(output["fused_probability"].sum(dim=-1), torch.ones(2), atol=1e-6)
    assert output["metadata"]["architecture_category"] == "single_stage_temporal_shared_prototype"
    assert "training_stage" not in output["metadata"]
    assert "risk" not in " ".join(model.state_dict()).lower()
    assert all("csi" not in key.lower() for key in model.state_dict())
    assert "fusion_logits" not in model.state_dict()
    assert output["beam_variance"].shape == (2,)


def test_static_reliability_fusion_is_four_global_trainable_weights() -> None:
    torch.manual_seed(5)
    mean_model = _model().eval()
    static_model = _model(fusion_mode="trainable_static_reliability").eval()
    static_model.load_state_dict(mean_model.state_dict(), strict=False)
    inputs = _inputs()

    mean_output = mean_model(**inputs)
    static_output = static_model(**inputs)
    assert static_model.fusion_logits.shape == (4,)
    assert torch.allclose(static_output["fused_probability"], mean_output["fused_probability"], atol=1e-6)
    assert static_output["metadata"]["fusion_mode"] == "trainable_static_reliability"
    assert static_output["metadata"]["fusion_logit_constraint"] == "none"
    assert static_output["metadata"]["global_fusion_weights"] == pytest.approx([0.25] * 4)

    single = static_model(**inputs, missing_mask=torch.tensor([[0, 1, 0, 0], [0, 0, 0, 1]]))
    assert torch.allclose(single["fused_probability"][0], single["unimodal_probabilities"][0, 1], atol=1e-6)
    assert torch.allclose(single["fused_probability"][1], single["unimodal_probabilities"][1, 3], atol=1e-6)
    assert torch.equal(single["fusion_weights"], torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]))

    with torch.no_grad():
        static_model.fusion_logits.copy_(torch.tensor([100.0, -100.0, -100.0, -100.0]))
    assert float(static_model(**inputs)["fusion_weights"].max().detach()) > 0.999
    with torch.no_grad():
        static_model.fusion_logits.zero_()

    static_model.train()
    trained = static_model(**inputs)
    loss = -trained["fused_probability"][:, 3].clamp_min(1e-12).log().mean()
    loss.backward()
    assert static_model.fusion_logits.grad is not None
    assert torch.isfinite(static_model.fusion_logits.grad).all()


def test_static_reliability_rejects_unknown_fusion_mode() -> None:
    with pytest.raises(ValueError, match="fusion_mode"):
        _model(fusion_mode="dynamic_router")


def test_bounded_static_reliability_prevents_full_mask_weight_collapse() -> None:
    torch.manual_seed(7)
    mean_model = _model().eval()
    bounded_model = _model(fusion_mode="bounded_static_reliability").eval()
    bounded_model.load_state_dict(mean_model.state_dict(), strict=False)
    inputs = _inputs()

    initial = bounded_model(**inputs)
    assert torch.allclose(initial["fused_probability"], mean_model(**inputs)["fused_probability"], atol=1e-6)
    assert bounded_model.fusion_logits.shape == (4,)

    with torch.no_grad():
        bounded_model.fusion_logits.copy_(torch.tensor([100.0, -100.0, -100.0, -100.0]))
    bounded = bounded_model(**inputs)
    maximum_weight = math.exp(2.0) / (math.exp(2.0) + 3.0)
    assert float(bounded["fusion_weights"].max().detach()) <= maximum_weight + 1e-6
    assert bounded["metadata"]["fusion_mode"] == "bounded_static_reliability"
    assert bounded["metadata"]["fusion_logit_constraint"] == "tanh_unit_interval"
    assert bounded["metadata"]["probability_parameterization"] == (
        "prototype_probability_bounded_static_reliability_v1"
    )

    single = bounded_model(**inputs, missing_mask=torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]]))
    assert torch.equal(single["fusion_weights"], torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]))
    assert torch.allclose(single["fused_probability"][0], single["unimodal_probabilities"][0, 0], atol=1e-6)
    assert torch.allclose(single["fused_probability"][1], single["unimodal_probabilities"][1, 2], atol=1e-6)


def test_bounded_static_reliability_has_finite_training_gradient() -> None:
    model = _model(fusion_mode="bounded_static_reliability").train()
    output = model(**_inputs())
    loss = -output["fused_probability"][:, 3].clamp_min(1e-12).log().mean()
    loss.backward()

    assert model.fusion_logits.grad is not None
    assert torch.isfinite(model.fusion_logits.grad).all()


def test_masked_feature_mlp_uses_one_shared_bank_for_unimodal_and_fused_features() -> None:
    torch.manual_seed(11)
    model = _model(fusion_mode="masked_feature_mlp").eval()
    inputs = _inputs()
    mask = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1]])
    output = model(**inputs, missing_mask=mask)

    expected_unimodal = model.prototype_bank(output["modality_features"].reshape(-1, 64)).reshape(2, 4, 64)
    expected_fused = model.prototype_bank(output["output_features"])
    assert torch.allclose(output["unimodal_logits"], expected_unimodal, atol=1e-6)
    assert torch.allclose(output["logits"][:, 0], expected_fused, atol=1e-6)
    assert output["fusion_weights"] is None
    assert output["metadata"]["prototype_bank_count"] == 1
    assert output["metadata"]["prototype_feature_sources"] == ["image", "radar", "gps", "lidar", "fused"]
    assert output["metadata"]["fusion_has_explicit_modality_weights"] is False
    assert output["metadata"]["probability_parameterization"] == "prototype_probability_masked_feature_mlp_v1"
    assert sum(key == "prototype_bank.prototypes" for key in model.state_dict()) == 1
    assert "fusion_logits" not in model.state_dict()


def test_masked_feature_mlp_cannot_observe_unavailable_modality_features() -> None:
    torch.manual_seed(13)
    model = _model(fusion_mode="masked_feature_mlp").eval()
    inputs = _inputs()
    changed = dict(inputs)
    changed["radar_batch"] = torch.full_like(inputs["radar_batch"], 1e6)
    mask = torch.tensor([[1, 0, 1, 1], [1, 0, 1, 1]])

    original = model(**inputs, missing_mask=mask)
    perturbed = model(**changed, missing_mask=mask)
    assert torch.allclose(original["fused_probability"], perturbed["fused_probability"], atol=1e-6)


def test_masked_feature_mlp_and_shared_prototype_receive_finite_gradients() -> None:
    model = _model(fusion_mode="masked_feature_mlp").train()
    output = model(**_inputs(), missing_mask=torch.tensor([[1, 1, 1, 1], [1, 0, 1, 0]]))
    loss = -output["fused_probability"][:, 3].clamp_min(1e-12).log().mean()
    loss.backward()

    gradients = [parameter.grad for parameter in model.feature_fusion.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert model.prototype_bank.prototypes.grad is not None
    assert torch.isfinite(model.prototype_bank.prototypes.grad).all()


def test_topology_loss_backpropagates_and_ablation_removes_only_topology_terms() -> None:
    model = _model().train()
    output = model(**_inputs())
    labels = torch.tensor([[3], [9]])

    enabled = four_modal_topology_loss(output, labels, prototype_bank=model.prototype_bank, config=_loss_config())
    disabled = four_modal_topology_loss(
        output,
        labels,
        prototype_bank=model.prototype_bank,
        config=_loss_config(topology=False),
    )
    enabled["loss"].backward()

    assert torch.isfinite(enabled["loss"])
    assert torch.isfinite(disabled["loss"])
    assert model.prototype_bank.prototypes.grad is not None
    assert disabled["diagnostics"]["loss/unimodal_soft"] >= 0.0


def test_joint_topology_is_one_weighted_average_of_fused_and_available_unimodal_losses() -> None:
    model = _model(fusion_mode="masked_feature_mlp").train()
    output = model(**_inputs(), missing_mask=torch.tensor([[1, 1, 1, 1], [1, 0, 1, 0]]))
    labels = torch.tensor([[3], [9]])
    base = _loss_config(topology=False)
    joint = dict(base, joint_topology_weight=0.1)

    disabled = four_modal_topology_loss(output, labels, prototype_bank=model.prototype_bank, config=base)
    enabled = four_modal_topology_loss(output, labels, prototype_bank=model.prototype_bank, config=joint)
    diagnostics = enabled["diagnostics"]
    expected_joint = 0.5 * (
        diagnostics["loss/joint_topology_fused"] + diagnostics["loss/joint_topology_unimodal"]
    )

    assert diagnostics["loss/joint_topology"] == pytest.approx(expected_joint)
    assert diagnostics["loss/joint_topology_weight"] == 0.1
    assert float((enabled["loss"] - disabled["loss"]).detach()) == pytest.approx(
        0.1 * expected_joint, rel=1e-5
    )
    enabled["loss"].backward()
    assert model.prototype_bank.prototypes.grad is not None
    assert torch.isfinite(model.prototype_bank.prototypes.grad).all()


def test_joint_topology_weight_is_nonnegative_and_defaults_to_zero() -> None:
    assert _loss_config(topology=False)["joint_topology_weight"] == 0.0
    with pytest.raises(ValueError, match="joint_topology_weight"):
        four_modal_topology_config(
            {
                "loss": {
                    "four_modal_topology": {
                        "enabled": True,
                        "joint_topology_weight": -0.1,
                        "prototype_topology": {"id": "cyclic_index_v1"},
                    }
                }
            }
        )


def test_hard_label_smoothing_is_zero_by_default_and_range_checked() -> None:
    default = _loss_config()
    assert default["hard_label_smoothing"] == 0.0

    smoothed = four_modal_topology_config(
        {
            "loss": {
                "four_modal_topology": {
                    "enabled": True,
                    "hard_label_smoothing": 0.1,
                    "prototype_topology": {"id": "cyclic_index_v1"},
                }
            }
        }
    )
    assert smoothed["hard_label_smoothing"] == 0.1

    for value in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="hard_label_smoothing"):
            four_modal_topology_config(
                {
                    "loss": {
                        "four_modal_topology": {
                            "enabled": True,
                            "hard_label_smoothing": value,
                            "prototype_topology": {"id": "cyclic_index_v1"},
                        }
                    }
                }
            )


def test_hard_label_smoothing_applies_to_fused_and_unimodal_hard_ce() -> None:
    model = _model().eval()
    output = model(**_inputs())
    labels = torch.tensor([[3], [9]])
    unsmoothed = four_modal_topology_loss(
        output,
        labels,
        prototype_bank=model.prototype_bank,
        config=_loss_config(topology=False),
    )
    smoothed_cfg = four_modal_topology_config(
        {
            "loss": {
                "four_modal_topology": {
                    "enabled": True,
                    "hard_label_smoothing": 0.1,
                    "unimodal_soft_weight": 0.0,
                    "use_beam_prototype_alignment": False,
                    "lambda_proto": 0.0,
                    "lambda_modality_proto": 0.0,
                    "prototype_topology": {"id": "cyclic_index_v1"},
                }
            }
        }
    )
    smoothed = four_modal_topology_loss(
        output,
        labels,
        prototype_bank=model.prototype_bank,
        config=smoothed_cfg,
    )

    assert smoothed["diagnostics"]["loss/hard_label_smoothing"] == 0.1
    assert not torch.equal(unsmoothed["loss"], smoothed["loss"])


def test_linear_topology_does_not_wrap_endpoint_neighbor_target() -> None:
    model = _model(fusion_mode="masked_feature_mlp", topology_id="linear_index_v1").eval()
    output = model(**_inputs())
    config = four_modal_topology_config(
        {
            "loss": {
                "four_modal_topology": {
                    "enabled": True,
                    "unimodal_soft_weight": 0.0,
                    "use_beam_prototype_alignment": True,
                    "lambda_proto": 0.1,
                    "lambda_modality_proto": 0.0,
                    "prototype_topology": {"id": "linear_index_v1"},
                }
            }
        }
    )

    result = four_modal_topology_loss(
        output,
        torch.tensor([[0], [63]]),
        prototype_bank=model.prototype_bank,
        config=config,
    )

    assert torch.isfinite(result["loss"])
    assert result["diagnostics"]["prototype/topology_is_cyclic"] == 0.0


def test_missing_matrix_is_exactly_the_15_nonempty_native_masks() -> None:
    configured = [
        "full", "missing_image", "missing_radar", "missing_gps", "missing_lidar",
        "image_only", "radar_only", "gps_only", "lidar_only",
        "missing_image_radar", "missing_image_gps", "missing_image_lidar",
        "missing_radar_gps", "missing_radar_lidar", "missing_gps_lidar",
    ]
    patterns = resolve_topology_missing_patterns(configured)

    assert len(patterns) == 15
    assert {tuple(value) for value in patterns.values()} == {
        tuple(int(bool(bits & (1 << index))) for index in range(4)) for bits in range(1, 16)
    }
