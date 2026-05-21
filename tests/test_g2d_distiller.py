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

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.distillation.g2d import (  # noqa: E402
    G2DDistiller,
    extract_modality_feature,
    teacher_confidence_from_logits,
)
from kd_sensing.distillation.teacher_ensemble import normalize_teacher_logits  # noqa: E402
from kd_sensing.engine.g2d_training import build_g2d_teacher_ensemble  # noqa: E402
from kd_sensing.engine.model_output import ModelOutput  # noqa: E402


def test_teacher_confidence_matches_label_probability():
    logits = torch.zeros(2, 3, 4)
    labels = torch.tensor([[0, 1, 2], [0, 1, 3]])
    logits[:, 0, 0] = 4.0
    logits[:, 1, 1] = 2.0
    logits[:, 2, 2] = 1.0

    confidence = teacher_confidence_from_logits(logits, labels, num_pred=3, modality="image")

    probs = torch.softmax(logits, dim=-1)
    expected = torch.stack(
        [
            probs[:, 0, 0].mean(),
            probs[:, 1, 1].mean(),
            torch.stack([probs[0, 2, 2], probs[1, 2, 3]]).mean(),
        ]
    )
    assert torch.allclose(confidence, expected)


def test_feature_extractor_supports_modality_features_and_token_features():
    modality_features = {"radar": torch.randn(2, 5, 7)}
    output = ModelOutput(
        logits=torch.randn(2, 3, 64),
        input_features=None,
        output_features=None,
        diagnostics={"modality_features": modality_features},
    )

    extracted = extract_modality_feature(output, "radar", pool="mean", source="modality")

    assert extracted.shape == (2, 7)
    assert torch.allclose(extracted, modality_features["radar"].mean(dim=1))

    tokens = torch.randn(2, 3, 5, 11)
    token_output = ModelOutput(
        logits=torch.randn(2, 3, 64),
        input_features=None,
        output_features=None,
        diagnostics={"token_features": tokens, "modalities": ("image", "gps", "mmwave")},
    )

    token_feature = extract_modality_feature(token_output, "gps", pool="last", source="modality")

    assert token_feature.shape == (2, 11)
    assert torch.allclose(token_feature, tokens[:, 1, -1, :])


def test_g2d_auto_projection_handles_feature_dim_mismatch():
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
    student = ModelOutput(
        logits=torch.randn(2, 3, 64, requires_grad=True),
        input_features=None,
        output_features=None,
        diagnostics={
            "modality_features": {"image": torch.randn(2, 8, 5, requires_grad=True)},
            "modalities": ("image",),
        },
    )
    teacher = {
        "image": ModelOutput(
            logits=torch.randn(2, 3, 64),
            input_features=torch.randn(2, 8, 7),
            output_features=torch.randn(2, 8, 7),
            diagnostics={},
        )
    }
    distiller = G2DDistiller(
        nn.CrossEntropyLoss(),
        g2d={"modalities": ["image"], "loss": {"feature_align": {"projection": "auto"}}},
        modalities=["image"],
    )

    result = distiller.compute(student, teacher, labels)

    assert result.feature_kd_loss.ndim == 0
    assert "image" in distiller.projections


def test_teacher_ensemble_logit_normalization_rejects_legacy_four_slots():
    with pytest.raises(ValueError, match="legacy"):
        normalize_teacher_logits(
            torch.randn(2, 4, 64),
            num_pred=3,
            num_classes=64,
            modality="image",
            allowed_slots=10,
        )


def test_g2d_configs_load_as_five_modality_g2d():
    for path, mode in [
        ("configs/fusion/image_radar_gps_lidar_mmwave_g2d_lite.yaml", "lite"),
        ("configs/fusion/image_radar_gps_lidar_mmwave_g2d_global.yaml", "global"),
        ("configs/fusion/image_radar_gps_lidar_mmwave_g2d_horizon.yaml", "horizon_diagnostic"),
    ]:
        cfg = load_config(ROOT / path)
        assert cfg["experiment"]["task"] == "fusion"
        assert cfg["model"]["num_pred"] == 3
        assert cfg["model"]["student"]["modalities"] == ["image", "radar", "gps", "lidar", "mmwave"]
        assert cfg["distillation"]["type"] == "g2d"
        assert cfg["distillation"]["g2d"]["mode"] == mode


def test_g2d_teacher_confidence_and_ranking_include_csi():
    labels = torch.tensor([[0, 1, 2], [1, 2, 3]])
    student = ModelOutput(
        logits=torch.randn(2, 3, 64, requires_grad=True),
        input_features=None,
        output_features=torch.randn(2, 8, 64, requires_grad=True),
        diagnostics={
            "modality_features": {
                "gps": torch.randn(2, 8, 64, requires_grad=True),
                "csi": torch.randn(2, 8, 64, requires_grad=True),
            },
            "modalities": ("gps", "csi"),
        },
    )
    teacher = {
        "gps": ModelOutput(
            logits=torch.randn(2, 3, 64),
            input_features=torch.randn(2, 8, 64),
            output_features=torch.randn(2, 8, 64),
            diagnostics={},
        ),
        "csi": ModelOutput(
            logits=torch.randn(2, 3, 64),
            input_features=torch.randn(2, 8, 64),
            output_features=torch.randn(2, 8, 64),
            diagnostics={},
        ),
    }
    distiller = G2DDistiller(
        nn.CrossEntropyLoss(),
        g2d={
            "modalities": ["gps", "csi"],
            "loss": {
                "feature_weight": 0.0,
                "logit_weight": 0.1,
                "feature_align": {"enabled": False},
            },
            "smp": {"enabled": True, "tau": {"per_modality": 1}},
        },
        modalities=["gps", "csi"],
    )

    result = distiller.compute(student, teacher, labels, epoch=0)

    assert set(result.teacher_confidence) == {"gps", "csi"}
    assert set(result.diagnostics["teacher_confidence"]) == {"gps", "csi"}
    assert "csi" in result.diagnostics["modality_ranking_weak_to_strong"]["avg"]
    assert result.active_modalities[0] in {"gps", "csi"}


def test_g2d_csi_teacher_checkpoint_error_names_modality(tmp_path: Path):
    pytest.importorskip("pandas")
    cfg = {
        "model": {
            "feature_size": 8,
            "num_classes": 64,
            "num_pred": 3,
            "student": {"modalities": ["csi"]},
            "teacher": {"csi_train_rms": 1.0},
        },
        "distillation": {
            "type": "g2d",
            "g2d": {
                "modalities": ["csi"],
                "teachers": {"csi": {"checkpoint": str(tmp_path / "missing.pth")}},
            },
        },
        "checkpoint": {"strict_load": True},
    }

    with pytest.raises(FileNotFoundError, match="modality 'csi'"):
        build_g2d_teacher_ensemble(cfg, torch.device("cpu"))
