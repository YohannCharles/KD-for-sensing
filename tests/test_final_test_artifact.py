import json
from pathlib import Path

import pytest

from kd_sensing.engine.artifacts import write_final_test_metrics


def test_final_test_metrics_publish_only_after_provenance_is_complete(tmp_path: Path):
    metrics = {
        "evaluation_split": "test",
        "top1": 0.5,
        "selected_checkpoint": {
            "path": str(tmp_path / "checkpoints" / "last.pth"),
            "checkpoint_role": "last",
        },
    }

    path = write_final_test_metrics(tmp_path, metrics)

    assert json.loads(path.read_text(encoding="utf-8")) == metrics
    assert list(tmp_path.glob(".final_test_metrics.json.*.tmp")) == []


@pytest.mark.parametrize(
    "metrics",
    [
        {"evaluation_split": "validation", "selected_checkpoint": {"path": "x", "checkpoint_role": "last"}},
        {"evaluation_split": "test"},
        {"evaluation_split": "test", "selected_checkpoint": {"path": "x"}},
    ],
)
def test_final_test_metrics_reject_incomplete_provenance(tmp_path: Path, metrics: dict):
    with pytest.raises(ValueError):
        write_final_test_metrics(tmp_path, metrics)

    assert not (tmp_path / "final_test_metrics.json").exists()
