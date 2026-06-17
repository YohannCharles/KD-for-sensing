from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.engine.artifacts import final_config_with_runtime  # noqa: E402


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


def test_predictive_jepa_hybrid_config_loads_and_preserves_existing_jepa_baselines(tmp_path: Path):
    predictive = load_config(
        ROOT
        / "configs/fusion/experiments/jepa_image_gps/"
        "image_gps_jepa_predictive_hybrid_beambench_fair_lowmem.yaml"
    )
    gps_query = load_config(
        ROOT
        / "configs/fusion/experiments/jepa_image_gps/"
        "image_gps_jepa_gps_query_pool_best_beambench_fair_lowmem.yaml"
    )
    resnet_image_gps = load_config(
        ROOT
        / "configs/fusion/experiments/jepa_image_gps/"
        "image_gps_supervised_beambench_fair_lowmem.yaml"
    )

    image_encoder = predictive["model"]["primary"]["encoders"]["image"]
    assert predictive["experiment"]["model_group"] == "jepa_predictive_hybrid"
    assert predictive["data"]["dataset"]["seq_len"] == 4
    assert image_encoder["pooling"] == "hybrid_residual_query"
    assert image_encoder["pooler"]["type"] == "hybrid_residual_query"
    assert image_encoder["pooler"]["residual_alpha_init"] == 0.1
    assert image_encoder["temporal_auxiliary"]["enabled"] is True
    assert predictive["model"]["primary"]["representation_core"]["type"] == "feature_consistency_gate"
    assert predictive["difficulty"]["profiles"][0]["condition"] == "P4_joint_predictive_recovery"
    assert predictive["difficulty"]["profiles"][0]["operators"][0]["affected_modalities"] == ["image", "gps"]
    assert predictive["output"]["dir"].startswith("outputs/analysis/predictive_jepa_robustness")

    metadata = final_config_with_runtime(predictive, run_dir=tmp_path / "predictive")["runtime"]["jepa_downstream"]
    assert metadata["pooler_type"] == "hybrid_residual_query"
    assert metadata["hybrid_residual_query_enabled"] is True
    assert metadata["hybrid_content_queries"] == 2
    assert metadata["hybrid_gps_queries"] == 2
    assert metadata["temporal_auxiliary_enabled"] is True
    assert metadata["representation_core_type"] == "feature_consistency_gate"
    assert metadata["representation_core"]["history_window"] == 3

    assert gps_query["model"]["primary"]["encoders"]["image"]["pooling"] == "gps_query_attention"
    assert resnet_image_gps["model"]["primary"]["encoders"]["image"]["type"] == "resnet18_imagenet_rgb"


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
