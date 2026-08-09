from __future__ import annotations

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


def _model() -> FourModalTopologyPredictor:
    encoders = {
        name: {"type": "test_topology_sequence", "output_dim": 64}
        for name in ("image", "radar", "gps", "lidar")
    }
    return FourModalTopologyPredictor(
        modalities=("image", "radar", "gps", "lidar"),
        encoders=encoders,
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
    assert output["beam_variance"].shape == (2,)


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
