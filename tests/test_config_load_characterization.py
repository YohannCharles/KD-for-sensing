from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
from kd_sensing.config import load_config  # noqa: E402


def test_config_load_pipeline_characterization_covers_sources_and_overrides():
    entity = load_config(ROOT / "configs/gps/lightweight.yaml")
    virtual_fusion = load_config(ROOT / "configs/fusion/gps_mmwave_lightweight.yaml")
    snapshot = load_config(ROOT / "configs/fusion/all_modalities_snapshot_next_frame_supervised.yaml")
    overridden = load_config(
        ROOT / "configs/fusion/gps_mmwave_lightweight.yaml",
        [
            "experiment.objective=beam",
            "training.early_stopping_metric=val_loss",
            "training.early_stopping_mode=min",
            "data.dataset.scene=32",
        ],
    )

    assert entity["data"]["dataset"]["scene_slug"] == "scene31"
    assert entity["training"]["early_stopping_metric"] == "val_adba"
    assert virtual_fusion["model"]["modalities"] == ["gps", "mmwave"]
    assert virtual_fusion["model"]["primary"]["modalities"] == ["gps", "mmwave"]
    assert "distillation" not in virtual_fusion
    assert snapshot["experiment"]["variant"] == "snapshot_next_frame"
    assert snapshot["experiment"]["uses_history_window"] is False
    assert overridden["data"]["dataset"]["scene_slug"] == "scene32"
    assert overridden["training"]["early_stopping_metric"] == "val_loss"
    assert overridden["training"]["early_stopping_mode"] == "min"


def test_retired_raymobtime_configs_fail_fast(tmp_path: Path):
    with pytest.raises(ValueError, match="Raymobtime s008 has been retired"):
        load_config(ROOT / "configs/raymobtime/s008_multitask_selection.yaml")
    with pytest.raises(ValueError, match="Raymobtime s008 has been retired"):
        load_config(ROOT / "configs/preprocess/raymobtime_s008_cache.yaml")

    config_path = tmp_path / "retired_raymobtime.yaml"
    config_path.write_text(
        """
experiment:
  task: fusion
  objective: current_beam_selection
data:
  dataset:
    type: raymobtime_s008
model:
  primary:
    type: simple_concat_multitask_selection
    modalities: [coord]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Raymobtime s008 has been retired"):
        load_config(config_path)
