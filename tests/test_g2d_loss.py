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

from kd_sensing.distillation.g2d import G2DDistiller  # noqa: E402
from kd_sensing.engine.model_output import ModelOutput  # noqa: E402


MODALITIES = ["image", "radar", "gps", "lidar", "mmwave"]


def _outputs(*, teacher_requires_grad: bool = False):
    batch_size, horizon, classes = 2, 3, 64
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    student_logits = torch.randn(batch_size, horizon, classes, requires_grad=True)
    student = ModelOutput(
        logits=student_logits,
        input_features=None,
        output_features=torch.randn(batch_size, 8, 64, requires_grad=True),
        diagnostics={
            "modality_features": {
                modality: torch.randn(batch_size, 8, 64, requires_grad=True)
                for modality in MODALITIES
            },
            "modalities": tuple(MODALITIES),
        },
    )
    teachers = {}
    teacher_logits = {}
    teacher_features = {}
    for modality in MODALITIES:
        logits = torch.randn(batch_size, horizon, classes, requires_grad=teacher_requires_grad)
        features = torch.randn(batch_size, 8, 64, requires_grad=teacher_requires_grad)
        teacher_logits[modality] = logits
        teacher_features[modality] = features
        teachers[modality] = ModelOutput(
            logits=logits,
            input_features=features,
            output_features=features,
            diagnostics={},
        )
    return student, teachers, labels, teacher_logits, teacher_features


def test_g2d_loss_returns_scalar_and_uses_three_horizons():
    student, teachers, labels, _, _ = _outputs()
    distiller = G2DDistiller(nn.CrossEntropyLoss(), g2d={"modalities": MODALITIES}, modalities=MODALITIES)

    result = distiller.compute(student, teachers, labels)

    assert result.total_loss.ndim == 0
    assert result.supervised_loss.ndim == 0
    assert result.diagnostics["num_pred"] == 3
    assert result.diagnostics["horizon_names"] == ["t+1", "t+2", "t+3"]


def test_g2d_loss_component_weights_can_disable_feature_or_logit_terms():
    student, teachers, labels, _, _ = _outputs()
    distiller = G2DDistiller(
        nn.CrossEntropyLoss(),
        g2d={
            "modalities": MODALITIES,
            "loss": {
                "supervised_weight": 1.0,
                "feature_weight": 0.0,
                "logit_weight": 0.0,
                "feature_align": {"enabled": False},
                "logit_align": {"enabled": False},
            },
        },
        modalities=MODALITIES,
    )

    result = distiller.compute(student, teachers, labels)

    assert torch.allclose(result.total_loss, result.supervised_loss)
    assert result.feature_kd_loss.item() == pytest.approx(0.0)
    assert result.logit_kd_loss.item() == pytest.approx(0.0)


def test_g2d_teacher_tensors_do_not_receive_gradients():
    student, teachers, labels, teacher_logits, teacher_features = _outputs(teacher_requires_grad=True)
    distiller = G2DDistiller(nn.CrossEntropyLoss(), g2d={"modalities": MODALITIES}, modalities=MODALITIES)

    result = distiller.compute(student, teachers, labels)
    result.total_loss.backward()

    assert student.logits.grad is not None
    for modality in MODALITIES:
        assert teacher_logits[modality].grad is None
        assert teacher_features[modality].grad is None


def test_g2d_rejects_legacy_four_slot_logits():
    student, teachers, labels, _, _ = _outputs()
    teachers["image"] = ModelOutput(
        logits=torch.randn(2, 4, 64),
        input_features=torch.randn(2, 8, 64),
        output_features=torch.randn(2, 8, 64),
        diagnostics={},
    )
    distiller = G2DDistiller(nn.CrossEntropyLoss(), g2d={"modalities": MODALITIES}, modalities=MODALITIES)

    with pytest.raises(ValueError, match="horizon=3"):
        distiller.compute(student, teachers, labels)
