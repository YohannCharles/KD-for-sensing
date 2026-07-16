import torch
import torch.nn as nn

from kd_sensing.registries import ENCODERS, MODELS, import_default_components


@ENCODERS.register("masked_mean_test_encoder", force=True)
class MaskedMeanTestEncoder(nn.Module):
    def __init__(self, output_dim: int = 4, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        values = batch.float().reshape(*batch.shape[:2], -1).mean(dim=-1, keepdim=True)
        return values.expand(-1, -1, self.output_dim)


def test_t2_masked_mean_excludes_missing_temporal_cells() -> None:
    import_default_components()
    model = MODELS.build(
        {
            "type": "u_mask_beam_jepa",
            "modalities": ["image", "radar"],
            "d_model": 4,
            "num_classes": 4,
            "num_pred": 1,
            "dropout": 0.0,
            "fusion_type": "supervised_router",
            "head_type": "classifier",
            "temporal_pooling": {"enabled": True, "type": "masked_mean"},
            "encoders": {
                "image": {"type": "masked_mean_test_encoder", "output_dim": 4},
                "radar": {"type": "masked_mean_test_encoder", "output_dim": 4},
            },
        }
    )

    output = model(
        image_batch=torch.tensor([[[1.0], [3.0], [5.0]]]),
        radar_batch=torch.tensor([[[2.0], [4.0], [6.0]]]),
        modality_temporal_mask=torch.tensor([[[1, 1], [1, 0], [0, 0]]], dtype=torch.bool),
    )

    assert torch.equal(output["input_features"], torch.full((1, 2, 4), 2.0))
    assert output["temporal_pooling_type"] == "masked_mean"
    assert output["temporal_pooling_param_count"] == 0
    assert torch.allclose(output["supervised_router_gate_weights"].sum(dim=1), torch.ones(1))
