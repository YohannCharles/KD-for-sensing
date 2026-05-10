from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.diagnostics.communication_state_features import (  # noqa: E402
    assign_buckets,
    communication_state_feature_records,
    compute_beam_transition_features,
    compute_gps_state_features,
    compute_mmwave_state_features,
    fit_bucket_thresholds,
)


def test_mmwave_gps_and_beam_transition_features():
    mmwave = torch.tensor([[[0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]]])
    gps = torch.tensor([[[8.0, 1.0, 0.0], [10.0, 0.0, 1.0]]])
    input_beam = torch.tensor([[1, 2]])
    target_beam = torch.tensor([[2, 3]])

    mmwave_features = compute_mmwave_state_features(mmwave)
    gps_features = compute_gps_state_features(gps)
    transition = compute_beam_transition_features(input_beam, target_beam, horizon_names=["t+1", "t+2"])

    assert mmwave_features.shape[0] == 1
    assert mmwave_features.loc[0, "mmwave_peak_drift"] == 1
    assert mmwave_features.loc[0, "mmwave_top1_prob"] > 0.9
    assert gps_features.loc[0, "range_to_bs"] == 10.0
    assert gps_features.loc[0, "bearing"] == pytest.approx(0.0)
    assert gps_features.loc[0, "delta_bearing"] == pytest.approx(-math.pi / 2)
    assert transition["beam_transition"].tolist() == [0, 1]


def test_communication_records_and_bucket_assignment():
    batch = {
        "mmwave": torch.tensor(
            [
                [[0.0, 2.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]],
            ]
        ),
        "gps": torch.tensor(
            [
                [[5.0, 0.0, 1.0], [6.0, 0.0, 1.0]],
                [[5.0, 1.0, 0.0], [7.0, 0.0, 1.0]],
            ]
        ),
        "input_beam": torch.tensor([[1, 1], [2, 3]]),
    }
    labels = torch.tensor([[1, 2], [4, 4]])
    metadata = {"dataset_index": torch.tensor([10, 11]), "sample_id": ["a", "b"]}

    records = communication_state_feature_records(
        batch,
        labels=labels,
        metadata=metadata,
        horizon_names=["t+1", "t+2"],
    )
    assert len(records) == 4
    assert records[0]["sample_id"] == "a"
    assert records[0]["beam_transition"] == 0
    assert records[1]["beam_transition"] == 1

    import pandas as pd

    frame = pd.DataFrame(records)
    thresholds = fit_bucket_thresholds(frame, ["mmwave_entropy", "beam_transition"], quantiles=(0.5,), bucket_names=("low", "high"))
    bucketed = assign_buckets(frame, thresholds)
    assert set(bucketed["beam_transition_bucket"]) == {"stable", "transition"}
    assert set(bucketed["mmwave_entropy_bucket"]).issubset({"low", "high"})
