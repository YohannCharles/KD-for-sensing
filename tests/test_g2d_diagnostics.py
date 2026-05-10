from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.diagnostics.g2d_diagnostics import G2DDiagnosticsAccumulator  # noqa: E402


def test_g2d_diagnostics_accumulator_writes_expected_schema(tmp_path: Path):
    accumulator = G2DDiagnosticsAccumulator(num_pred=3, horizon_names=["t+1", "t+2", "t+3"])
    accumulator.update(
        {
            "num_pred": 3,
            "horizon_names": ["t+1", "t+2", "t+3"],
            "teacher_confidence": {
                "image": {"t+1": 0.1, "t+2": 0.2, "t+3": 0.3, "avg": 0.2},
            },
            "modality_ranking_weak_to_strong": {
                "avg": ["image", "radar"],
                "t+1": ["image", "radar"],
                "t+2": ["radar", "image"],
                "t+3": ["image", "radar"],
            },
            "active_modalities": ["image"],
            "loss": {
                "supervised": 1.0,
                "feature_kd": 0.2,
                "logit_kd": 0.3,
                "total": 1.5,
            },
        }
    )

    path = accumulator.write_epoch(tmp_path, epoch=3)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "g2d_epoch_3.json"
    assert payload["num_pred"] == 3
    assert payload["horizon_names"] == ["t+1", "t+2", "t+3"]
    assert "teacher_confidence" in payload
    assert "modality_ranking_weak_to_strong" in payload
    assert payload["active_modalities"] == ["image"]
    assert payload["loss"]["supervised"] == 1.0
    assert payload["loss"]["feature_kd"] == 0.2
    assert payload["loss"]["logit_kd"] == 0.3
    assert payload["loss"]["total"] == 1.5
