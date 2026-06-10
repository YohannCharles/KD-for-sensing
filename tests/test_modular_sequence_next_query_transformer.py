from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.models.modular import ModularSequenceModel, NextBeamQueryTransformerCore  # noqa: E402
from kd_sensing.registries import ENCODERS, REPRESENTATION_CORES  # noqa: E402


@ENCODERS.register("next_query_test_identity", force=True)
class NextQueryTestIdentityEncoder(nn.Module):
    def __init__(self, output_dim: int = 8, **_: object):
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f"next_query_test_identity expects [B, T, D], got {tuple(batch.shape)}.")
        if int(batch.shape[-1]) != self.output_dim:
            raise ValueError(f"expected D={self.output_dim}, got {tuple(batch.shape)}.")
        return batch


def test_next_beam_query_transformer_core_forward_shape_and_registry_build():
    core = REPRESENTATION_CORES.build(
        {
            "type": "next_beam_query_transformer",
            "d_model": 8,
            "modality_count": 2,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.0,
            "max_seq_len": 4,
            "output_dim": 6,
        }
    )

    output = core(torch.randn(3, 2, 4, 8))

    assert isinstance(core, NextBeamQueryTransformerCore)
    assert output.shape == (3, 1, 6)


def test_next_beam_query_transformer_core_rejects_bad_inputs():
    core = NextBeamQueryTransformerCore(
        d_model=8,
        modality_count=2,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        max_seq_len=4,
    )

    with pytest.raises(ValueError, match=r"requires multimodal \[B, K, T, D\] input"):
        core(torch.randn(3, 4, 8))
    with pytest.raises(ValueError, match=r"expected K=2, got K=3"):
        core(torch.randn(3, 3, 4, 8))
    with pytest.raises(ValueError, match=r"expected D=8, got D=7"):
        core(torch.randn(3, 2, 4, 7))
    with pytest.raises(ValueError, match=r"T=5 exceeds max_seq_len=4"):
        core(torch.randn(3, 2, 5, 8))


def test_modular_sequence_next_query_transformer_beam_head_and_diagnostics():
    model = _modular_model(
        ["image", "gps"],
        {
            "type": "next_beam_query_transformer",
            "d_model": 8,
            "output_dim": 10,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.0,
            "max_seq_len": 3,
        },
        num_classes=5,
    )

    output = model(image_batch=torch.randn(2, 3, 8), gps_batch=torch.randn(2, 3, 8))

    assert output["logits"].shape == (2, 1, 5)
    assert output["input_features"].shape == (2, 3, 16)
    assert output["output_features"].shape == (2, 1, 10)
    assert output["modalities"] == ("image", "gps")
    assert set(output["modality_features"]) == {"image", "gps"}
    assert set(output["encoder_features"]) == {"image", "gps"}


@pytest.mark.parametrize(
    ("core_cfg", "seq_len", "expected_time"),
    [
        ({"type": "early_concat_gru", "d_model": 8, "hidden_size": 8, "num_layers": 1}, 3, 3),
        ({"type": "snapshot_frame", "d_model": 8, "output_dim": 8}, 1, 1),
        ({"type": "token_transformer", "d_model": 8, "num_heads": 2, "num_layers": 1}, 3, 3),
    ],
)
def test_existing_modular_sequence_cores_keep_forward_shapes(core_cfg: dict, seq_len: int, expected_time: int):
    model = _modular_model(["image", "gps"], core_cfg, num_classes=7)

    output = model(image_batch=torch.randn(2, seq_len, 8), gps_batch=torch.randn(2, seq_len, 8))

    assert output["logits"].shape == (2, expected_time, 7)
    assert output["output_features"].shape[:2] == (2, expected_time)


def _modular_model(modalities: list[str], core_cfg: dict, *, num_classes: int) -> ModularSequenceModel:
    encoders = {modality: {"type": "next_query_test_identity", "output_dim": 8} for modality in modalities}
    projectors = {modality: {"type": "identity", "input_dim": 8, "d_model": 8} for modality in modalities}
    return ModularSequenceModel(
        modalities=modalities,
        encoders=encoders,
        projectors=projectors,
        representation_core=core_cfg,
        feature_size=8,
        d_model=8,
        num_classes=num_classes,
        num_pred=1,
    )
