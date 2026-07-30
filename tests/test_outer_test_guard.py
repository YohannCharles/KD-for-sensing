import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_radio_guided_hierarchical_prototypes import (  # noqa: E402
    MASK_NAMES,
    _attach_reference_probabilities,
    analyze_subclusters,
    require_inner_split,
    validate_trajectory_disjointness,
)


def _reference_records(sample_ids: list[str]) -> dict[str, object]:
    count = len(sample_ids)
    records: dict[str, object] = {
        "sample_ids": sample_ids,
        "labels_future": torch.arange(count),
        "p0_full": torch.softmax(torch.randn(count, 64), dim=-1),
    }
    records.update(
        {f"p0_{name}": torch.softmax(torch.randn(count, 64), dim=-1) for name in MASK_NAMES}
    )
    return records


def test_outer_test_access_guard_fails_closed() -> None:
    with pytest.raises(ValueError, match="outer test remains sealed"):
        require_inner_split("test")


def test_cluster_analysis_rejects_validation_before_loading_data() -> None:
    with pytest.raises(ValueError, match="only use the train split"):
        analyze_subclusters({}, role="validation")


def test_train_and_validation_trajectories_must_be_disjoint() -> None:
    validate_trajectory_disjointness(["train-a", "train-b"], ["val-a"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_trajectory_disjointness(["shared"], ["shared"])


def test_published_m4_reference_probabilities_are_attached_exactly() -> None:
    records = _reference_records(["a", "b"])
    cache = {"sample_ids": ["a", "b"], "target": torch.tensor([0, 1])}
    attached = _attach_reference_probabilities(cache, records)
    assert torch.equal(attached["reference_full_probability"], records["p0_full"])
    assert torch.equal(attached["reference_probability_all_masks"][:, 0], records[f"p0_{MASK_NAMES[0]}"])


@pytest.mark.parametrize("field", ["sample_ids", "target"])
def test_published_m4_reference_alignment_fails_closed(field: str) -> None:
    records = _reference_records(["a", "b"])
    cache = {"sample_ids": ["a", "b"], "target": torch.tensor([0, 1])}
    if field == "sample_ids":
        cache[field] = ["b", "a"]
    else:
        cache[field] = torch.tensor([1, 0])
    with pytest.raises(ValueError, match="not aligned"):
        _attach_reference_probabilities(cache, records)
