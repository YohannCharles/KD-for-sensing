from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402


def test_config_load_pipeline_characterization_covers_sources_and_overrides():
    entity = load_config(ROOT / "configs/gps/student_no_kd.yaml")
    virtual_fusion = load_config(ROOT / "configs/fusion/gps_mmwave_logits_kd.yaml")
    snapshot = load_config(ROOT / "configs/fusion/all_modalities_snapshot_next_frame_no_kd.yaml")
    raymobtime = load_config(ROOT / "configs/raymobtime/s008_multitask_selection.yaml")
    overridden = load_config(
        ROOT / "configs/fusion/gps_mmwave_logits_kd.yaml",
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
    assert virtual_fusion["distillation"]["type"] == "logits_kd"
    assert snapshot["experiment"]["variant"] == "snapshot_next_frame"
    assert snapshot["experiment"]["uses_history_window"] is False
    assert raymobtime["experiment"]["task_semantics"] == "current_snapshot_beam_selection"
    assert raymobtime["model"]["student"]["encoders"]["lidar"]["type"] == "raymobtime_lidar_3d_cnn"
    assert overridden["data"]["dataset"]["scene_slug"] == "scene32"
    assert overridden["training"]["early_stopping_metric"] == "val_loss"
    assert overridden["training"]["early_stopping_mode"] == "min"
