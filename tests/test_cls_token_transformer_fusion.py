from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.engine.model_output import adapt_model_output, select_prediction_slots  # noqa: E402
from kd_sensing.models.fusion import CLSTokenTransformerFusionNet  # noqa: E402
from kd_sensing.registries import MODELS  # noqa: E402


def test_cls_token_transformer_five_modality_forward_shapes_and_diagnostics():
    model = _build_model(["image", "radar", "gps", "lidar", "mmwave"])
    model.eval()

    with torch.no_grad():
        output = model(
            image_batch=torch.randn(2, 3, 3, 224, 224),
            radar_batch=torch.randn(2, 3, 2, 128, 64),
            gps_batch=torch.randn(2, 3, 3),
            lidar_batch=torch.randn(2, 3, 3, 224, 224),
            mmwave_batch=torch.randn(2, 3, 64),
        )

    assert output["logits"].shape == (2, 2, 8)
    assert output["input_features"].shape == (2, 3, 16)
    assert output["output_features"].shape == (2, 2, 16)
    assert output["token_features"].shape == (2, 5, 3, 16)
    assert output["fusion_memory"].shape == (2, 1 + 3 * 5, 16)
    assert output["cls_features"].shape == (2, 16)
    assert output["effective_modality_mask"].tolist() == [[True] * 5, [True] * 5]
    assert output["modalities"] == ("image", "radar", "gps", "lidar", "mmwave")
    assert select_prediction_slots(output["logits"], num_pred=2).shape == (2, 2, 8)

    adapted = adapt_model_output(output)
    assert adapted.logits.shape == (2, 2, 8)
    assert adapted.input_features.shape == (2, 3, 16)
    assert adapted.output_features.shape == (2, 2, 16)
    gps_index = output["modalities"].index("gps")
    gps_last_token = output["token_features"][:, gps_index, -1, :]
    assert gps_last_token.shape == (2, 16)


def test_cls_token_transformer_auxiliary_heads_are_optional_and_mask_compatible():
    default_model = _build_model(["gps", "mmwave"])
    aux_model = MODELS.build(
        {
            "type": "cls_token_transformer_fusion",
            "modalities": ["gps", "mmwave"],
            "feature_size": 16,
            "d_model": 16,
            "num_classes": 8,
            "num_pred": 2,
            "num_heads": 4,
            "num_layers": 1,
            "max_seq_len": 4,
            "gps_input_size": 3,
            "mmwave_input_size": 64,
            "auxiliary_heads": {"enabled": True, "occlusion": True, "position": True},
        }
    )
    inputs = _inputs_for(["gps", "mmwave"], batch_size=2, seq_len=3)

    with torch.no_grad():
        default_output = default_model(**inputs)
        aux_output = aux_model(**inputs, force_modality_mask=torch.tensor([[True, False], [False, True]]))

    assert "occlusion_logits" not in default_output
    assert "position" not in default_output
    assert aux_output["logits"].shape == (2, 2, 8)
    assert aux_output["occlusion_logits"].shape == (2, 2)
    assert aux_output["position"].shape == (2, 2, 2)
    adapted = adapt_model_output(aux_output)
    assert adapted.diagnostics["occlusion_logits"].shape == (2, 2)
    assert adapted.diagnostics["position"].shape == (2, 2, 2)
    assert adapted.diagnostics["effective_modality_mask"].tolist() == [[True, False], [False, True]]


@pytest.mark.parametrize(
    "modalities",
    [
        ["image", "gps"],
        ["radar", "gps"],
        ["radar"],
    ],
)
def test_cls_token_transformer_accepts_legal_modality_subsets(modalities: list[str]):
    model = _build_model(modalities)
    model.eval()
    inputs = _inputs_for(modalities, batch_size=1, seq_len=2)

    with torch.no_grad():
        output = model(**inputs)

    assert output["logits"].shape == (1, 2, 8)
    assert output["token_features"].shape[:3] == (1, len(modalities), 2)
    assert output["modalities"] == tuple(modalities)


def test_cls_token_transformer_force_modality_mask_excludes_tokens_time_first():
    model = _build_model(["gps", "mmwave"])
    model.eval()
    mask = torch.tensor([[True, False], [False, True]])

    with torch.no_grad():
        output = model(
            gps_batch=torch.randn(2, 3, 3),
            mmwave_batch=torch.randn(2, 3, 64),
            force_modality_mask=mask,
        )

    assert output["effective_modality_mask"].tolist() == [[True, False], [False, True]]
    assert output["token_padding_mask"].shape == (2, 2, 3)
    assert torch.all(output["token_padding_mask"][0, 1])
    assert torch.all(output["token_padding_mask"][1, 0])
    assert torch.all(output["token_features"][0, 1] == 0.0)
    assert torch.all(output["token_features"][1, 0] == 0.0)
    assert output["serialized_token_padding_mask"][0].tolist() == [False, True, False, True, False, True]
    assert output["fusion_memory"].shape == (2, 1 + 3 * 2, 16)

    with pytest.raises(ValueError, match="no available modalities"):
        model(
            gps_batch=torch.randn(1, 3, 3),
            mmwave_batch=torch.randn(1, 3, 64),
            force_modality_mask=torch.tensor([[False, False]]),
        )


def test_cls_token_transformer_rejects_mismatched_sequence_dimensions():
    model = _build_model(["gps", "mmwave"])

    with pytest.raises(ValueError, match="share batch and sequence"):
        model(
            gps_batch=torch.randn(1, 3, 3),
            mmwave_batch=torch.randn(1, 2, 64),
        )


def test_cls_token_transformer_registry_returns_public_class():
    model = MODELS.build(
        {
            "type": "cls_token_transformer_fusion",
            "modalities": ["mmwave", "gps"],
            "feature_size": 16,
            "d_model": 16,
            "num_classes": 8,
            "num_pred": 2,
            "num_heads": 4,
            "num_layers": 1,
            "gps_input_size": 3,
            "mmwave_input_size": 64,
        }
    )

    assert type(model) is CLSTokenTransformerFusionNet
    assert model.modalities == ("gps", "mmwave")


def _build_model(modalities: list[str]) -> CLSTokenTransformerFusionNet:
    return MODELS.build(
        {
            "type": "cls_token_transformer_fusion",
            "modalities": modalities,
            "feature_size": 16,
            "d_model": 16,
            "num_classes": 8,
            "num_pred": 2,
            "num_heads": 4,
            "num_layers": 1,
            "max_seq_len": 4,
            "image_channels": 3,
            "radar_channels": 2,
            "gps_input_size": 3,
            "lidar_channels": 3,
            "mmwave_input_size": 64,
        }
    )


def _inputs_for(modalities: list[str], *, batch_size: int, seq_len: int) -> dict[str, torch.Tensor]:
    inputs: dict[str, torch.Tensor] = {}
    if "image" in modalities:
        inputs["image_batch"] = torch.randn(batch_size, seq_len, 3, 224, 224)
    if "radar" in modalities:
        inputs["radar_batch"] = torch.randn(batch_size, seq_len, 2, 128, 64)
    if "gps" in modalities:
        inputs["gps_batch"] = torch.randn(batch_size, seq_len, 3)
    if "lidar" in modalities:
        inputs["lidar_batch"] = torch.randn(batch_size, seq_len, 3, 224, 224)
    if "mmwave" in modalities:
        inputs["mmwave_batch"] = torch.randn(batch_size, seq_len, 64)
    return inputs
