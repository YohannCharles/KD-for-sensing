import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_radio_guided_hierarchical_prototypes import (  # noqa: E402
    spherical_kmeans,
    validate_no_future_csi,
)


def test_historical_five_frame_csi_is_accepted() -> None:
    validate_no_future_csi({"candidate_history": torch.zeros(2, 5, 32, 16, dtype=torch.complex64)})


def test_future_csi_or_channel_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_no_future_csi(
            {
                "candidate_history": torch.zeros(1, 5, 32, 16, dtype=torch.complex64),
                "future_channel_ref": "not-allowed",
            }
        )


def test_cluster_assignment_api_has_no_target_or_label_input() -> None:
    features = torch.randn(20, 4)
    assignment, centers, _ = spherical_kmeans(features, 2, seed=1)
    assert assignment.shape == (20,)
    assert centers.shape == (2, 4)
