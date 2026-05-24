from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.engine.evaluation_pass import _metrics_from_outputs  # noqa: E402
from kd_sensing.engine.model_output import ModelOutput  # noqa: E402
from kd_sensing.engine.objective_metadata import (  # noqa: E402
    default_primary_metric,
    normalize_objective_metric,
    objective_available_metrics,
    objective_enabled_heads,
    objective_spec,
)
from kd_sensing.engine.prediction_objectives import (  # noqa: E402
    PredictionTargets,
    compute_prediction_loss,
)


def test_near_field_objective_metadata_and_aliases():
    spec = objective_spec("near_field_beam_selection")
    assert spec.default_metric == "val_beam_top1"
    assert spec.default_metric_mode == "max"
    assert "val_beam_top1" in spec.available_metrics
    assert "val_beam_top3" in spec.available_metrics
    assert "val_beam_top5" in spec.available_metrics
    assert "val_adba" not in spec.available_metrics
    assert "val_beam_dba" not in spec.available_metrics
    assert default_primary_metric("near_field_beam_selection") == ("val_beam_top1", "max")
    assert normalize_objective_metric("near_field", objective="near_field_beam_selection") == "val_beam_top1"
    assert objective_enabled_heads({"experiment": {"objective": "near_field_beam_selection"}}) == ["beam_selection"]


def test_near_field_loss_checks_codebook_class_count():
    cfg = {
        "experiment": {"objective": "near_field_beam_selection"},
        "data": {"dataset": {"codebook_shape": [2, 3, 4]}},
    }
    logits = torch.randn(2, 1, 24)
    labels = torch.tensor([[1], [2]], dtype=torch.int64)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 24), labels.flatten())
    bundle = compute_prediction_loss(
        ModelOutput(logits=logits, input_features=None, output_features=None, diagnostics={}),
        PredictionTargets(labels=labels),
        cfg,
        reference=logits,
        beam_total_loss=ce,
        beam_task_loss=ce,
    )
    assert bundle.primary is ce
    assert "loss/near_field_beam_selection" in bundle.diagnostics

    bad_logits = torch.randn(2, 1, 8)
    with pytest.raises(ValueError, match="model_output_classes=8"):
        compute_prediction_loss(
            ModelOutput(logits=bad_logits, input_features=None, output_features=None, diagnostics={}),
            PredictionTargets(labels=labels),
            cfg,
            reference=bad_logits,
            beam_total_loss=bad_logits.sum(),
            beam_task_loss=bad_logits.sum(),
        )


def test_near_field_metrics_do_not_emit_adba():
    cfg = {"evaluation": {"k_values": [1, 3, 5]}}
    outputs = torch.tensor(
        [
            [[0.1, 2.0, 0.0, -1.0, 0.5, 0.2]],
            [[0.1, 0.0, 0.2, 3.0, 0.4, 0.5]],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[1], [3]], dtype=torch.int64)
    metrics = _metrics_from_outputs(0.5, outputs, labels, cfg, objective="near_field_beam_selection")
    available = objective_available_metrics("near_field_beam_selection", metrics)
    assert metrics["val_beam_top1"] == pytest.approx(1.0)
    assert "dba" not in metrics
    assert "val_adba" not in available
    assert "val_beam_dba" not in available
