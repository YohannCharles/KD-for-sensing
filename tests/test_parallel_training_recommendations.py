from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.engine.throughput_recommendations import recommend_parallel_training  # noqa: E402


def test_multimodal_nf_recommendations_cover_representative_runs():
    cases = [
        ("configs/multimodal_nf/image_beam.yaml", True),
        ("configs/multimodal_nf/lidar_beam.yaml", True),
        ("configs/multimodal_nf/fusion_beam.yaml", True),
        ("configs/multimodal_nf/gps_beam.yaml", False),
        ("configs/multimodal_nf/gps_los.yaml", False),
    ]
    for config_path, expect_heavy_io in cases:
        cfg = load_config(ROOT / config_path)
        result = recommend_parallel_training(cfg, config_path=config_path, parallel_runs=3, cpu_count=16, check_cache=False)
        heavy_modalities = result["recommendations"]["multimodal_nf_io"]["heavy_io_modalities"]
        if expect_heavy_io:
            assert heavy_modalities
            assert "training.epoch_subsampling.order=locality" in result["overrides"]
            assert any("validation_mode=lightweight" in item for item in result["overrides"])
        else:
            assert heavy_modalities == []
            assert not any("validation_mode=lightweight" in item for item in result["overrides"])


def test_multimodal_nf_image_lidar_recommendation_is_io_aware():
    cfg = load_config(ROOT / "configs/multimodal_nf/image_lidar.yaml")

    result = recommend_parallel_training(
        cfg,
        config_path="configs/multimodal_nf/image_lidar.yaml",
        parallel_runs=4,
        cpu_count=24,
        check_cache=False,
    )

    assert "data.cache.multimodal_nf.image.validation_mode=lightweight" in result["overrides"]
    assert "data.cache.multimodal_nf.lidar.validation_mode=lightweight" in result["overrides"]
    assert "training.epoch_subsampling.shuffle=false" in result["overrides"]
    assert "training.epoch_subsampling.order=locality" in result["overrides"]
    assert result["recommendations"]["multimodal_nf_io"]["gpu_assignment"] == (
        "spread_heavy_image_lidar_fusion_runs_evenly_across_gpus"
    )
    assert result["recommendations"]["multimodal_nf_io"]["avoid_repeated_strong_validation"] is True


def test_multimodal_nf_gps_only_recommendation_skips_heavy_cache_advice():
    cfg = load_config(ROOT / "configs/multimodal_nf/gps_beam.yaml")

    result = recommend_parallel_training(cfg, parallel_runs=2, cpu_count=8, check_cache=False)

    assert result["recommendations"]["multimodal_nf_io"]["heavy_io_modalities"] == []
    assert not any("data.cache.multimodal_nf.image" in item for item in result["overrides"])
    assert not any("training.epoch_subsampling.order=locality" == item for item in result["overrides"])
