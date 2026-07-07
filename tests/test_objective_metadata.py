from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
from kd_sensing.config import load_config
from kd_sensing.engine.objectives.metadata import (
    normalize_objective_metric,
    objective_available_metrics,
    objective_runtime_metadata,
    objective_spec,
    objective_tensorboard_scalars,
)


JEPA_CONFIG = ROOT / "configs/pretraining/deepsense6g_gps_conditioned_jepa_smoke.yaml"


def test_jepa_objective_metadata_and_tensorboard_are_isolated():
    spec = objective_spec("gps_conditioned_jepa")
    runtime = objective_runtime_metadata({"experiment": {"objective": "gps_conditioned_jepa"}, "loss": {}})
    tags = {tag for tag, _ in objective_tensorboard_scalars("gps_conditioned_jepa")}

    assert spec.primary_loss_name == "jepa"
    assert spec.default_metric == "val_jepa_loss"
    assert spec.default_metric_mode == "min"
    assert set(objective_available_metrics("gps_conditioned_jepa")) == {
        "val_loss",
        "val_jepa_loss",
        "val_jepa_mask_context_ratio",
        "val_jepa_mask_target_ratio",
        "val_jepa_ema_decay",
    }
    assert normalize_objective_metric("jepa", objective="gps_conditioned_jepa") == "val_jepa_loss"
    assert runtime["pretraining_kind"] == "gps_conditioned_jepa"
    assert "jepa/mask_target_ratio" in tags
    assert "beam/val_adba" not in tags
    assert "beam/val_top1" not in tags


def test_jepa_config_loads_and_rejects_beam_early_stopping_metric():
    cfg = load_config(JEPA_CONFIG)

    assert cfg["experiment"]["objective"] == "gps_conditioned_jepa"
    assert cfg["training"]["early_stopping_metric"] == "val_jepa_loss"
    assert cfg["training"]["early_stopping_mode"] == "min"
    assert cfg["model"]["primary"]["type"] == "gps_conditioned_jepa"
    assert cfg["model"]["primary"]["modalities"] == ["image", "gps"]
    assert cfg["data"]["dataset"]["image_profile"] == "rgb_imagenet"
    assert cfg["data"]["dataset"]["gps_feature_mode"] == "relative_polar"
    assert "distillation" not in cfg
    assert "teacher" not in cfg["model"]
    assert "student" not in cfg["model"]

    with pytest.raises(ValueError, match="gps_conditioned_jepa.*Available metrics"):
        load_config(JEPA_CONFIG, ["training.early_stopping_metric=val_adba"])
