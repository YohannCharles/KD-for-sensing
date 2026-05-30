from __future__ import annotations

import pytest
import torch

from kd_sensing.engine.evaluation_pass import _metrics_from_outputs
from kd_sensing.engine.training_metrics import aggregate_validation_metrics, mean_valid_slots, validation_subset_epoch_scalars
from kd_sensing.evaluation.horizon_selection import normalize_metric_horizons


def test_aggregate_validation_metrics_uses_selected_horizons() -> None:
    val_metrics = {
        "topk": {
            "3": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "5": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        },
        "dba": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "total": [10, 10, 10, 10, 10, 10],
        "metric_horizons": [2, 4, 6],
    }

    metrics = aggregate_validation_metrics(val_metrics)

    assert metrics["val_atop3"] == pytest.approx((0.2 + 0.4 + 0.6) / 3)
    assert metrics["val_atop5"] == pytest.approx((0.3 + 0.5 + 0.7) / 3)
    assert metrics["val_adba"] == pytest.approx((2.0 + 4.0 + 6.0) / 3)


def test_flat_future_topk_avg_uses_selected_horizons() -> None:
    labels = torch.tensor([[0, 1, 2, 3, 4, 5]])
    outputs = torch.zeros((1, 6, 10))
    outputs[:, :, 9] = 10.0
    outputs[0, 1, 1] = 20.0
    outputs[0, 5, 5] = 20.0

    metrics = _metrics_from_outputs(
        0.0,
        outputs,
        labels,
        {"evaluation": {"k_values": [1, 3, 5], "metric_horizons": [2, 4, 6]}},
        objective="beam",
    )

    assert metrics["metric_horizons"] == [2, 4, 6]
    assert metrics["val_top1_avg"] == pytest.approx(2 / 3)
    assert metrics["val_top1_t1"] == pytest.approx(0.0)
    assert metrics["val_top1_t2"] == pytest.approx(1.0)


def test_metric_horizons_validate_against_num_pred() -> None:
    with pytest.raises(ValueError, match="outside 1..3"):
        normalize_metric_horizons([2, 4, 6], num_pred=3, field_name="evaluation.metric_horizons")


def test_mean_valid_slots_ignores_unselected_horizons() -> None:
    assert mean_valid_slots([1, 2, 100, 4], [1, 1, 1, 1], horizons=[2, 4]) == pytest.approx(3.0)


def test_validation_subset_scalars_inherit_selected_horizons() -> None:
    val_metrics = {
        "metric_horizons": [2, 4, 6],
        "metric_horizon_source": "evaluation.metric_horizons",
        "modality_subsets": {
            "radar_only": {
                "loss": 0.5,
                "topk": {
                    "1": [0.9, 0.2, 0.9, 0.4, 0.9, 0.6],
                    "3": [0.9, 0.3, 0.9, 0.5, 0.9, 0.7],
                    "5": [0.9, 0.4, 0.9, 0.6, 0.9, 0.8],
                },
                "dba": [9.0, 2.0, 9.0, 4.0, 9.0, 6.0],
                "total": [10, 10, 10, 10, 10, 10],
            }
        },
    }

    scalars = validation_subset_epoch_scalars(val_metrics)

    assert scalars["val/subset/radar_only/top1"] == pytest.approx((0.2 + 0.4 + 0.6) / 3)
    assert scalars["val/subset/radar_only/atop3"] == pytest.approx((0.3 + 0.5 + 0.7) / 3)
    assert scalars["val/subset/radar_only/atop5"] == pytest.approx((0.4 + 0.6 + 0.8) / 3)
    assert scalars["val/subset/radar_only/adba"] == pytest.approx((2.0 + 4.0 + 6.0) / 3)
