from __future__ import annotations

import json
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


def test_multimodal_nf_recommendation_distinguishes_migration_pending(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_path = cache_dir / "derived" / "image" / "rgb_imagenet" / "train" / "city_seq8_pred3.npy"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"placeholder")
    cache_path.with_suffix(".npy.json").write_text(
        json.dumps(
            {
                "version": "multimodal_nf_derived_v1",
                "modality": "image",
                "profile": "rgb_imagenet",
                "split": "train",
                "seq_len": 8,
                "num_pred": 3,
                "source_path": "fixture.h5",
                "source_fingerprint": "abc",
                "shape": [8, 5, 6, 3],
                "dtype": "uint8",
                "sample_count": 8,
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(
        ROOT / "configs/multimodal_nf/image_lidar.yaml",
        [f"data.dataset.cache_dir={cache_dir}"],
    )

    result = recommend_parallel_training(
        cfg,
        config_path="configs/multimodal_nf/image_lidar.yaml",
        parallel_runs=2,
        cpu_count=16,
        check_cache=True,
    )

    image_cache = result["cache"]["multimodal_nf"]["image"]
    assert image_cache["status"] == "migration_pending"
    assert image_cache["recommended_policy"] == "auto"
    assert image_cache["migration_pending"] == 1
    assert image_cache["maintenance_recommendation"] == "run_metadata_only_derived_cache_upgrade_before_training"
    assert result["recommendations"]["multimodal_nf_io"]["migration_pending"] == 1
    assert result["recommendations"]["multimodal_nf_io"]["metadata_upgrade_recommendation"] == (
        "run_metadata_only_derived_cache_upgrade_before_training"
    )
    assert result["cache"]["prewarm_command"].endswith("multimodal_nf_derived_cache.yaml")


def test_multimodal_nf_gps_only_recommendation_skips_heavy_cache_advice():
    cfg = load_config(ROOT / "configs/multimodal_nf/gps_beam.yaml")

    result = recommend_parallel_training(cfg, parallel_runs=2, cpu_count=8, check_cache=False)

    assert result["recommendations"]["multimodal_nf_io"]["heavy_io_modalities"] == []
    assert not any("data.cache.multimodal_nf.image" in item for item in result["overrides"])
    assert not any("training.epoch_subsampling.order=locality" == item for item in result["overrides"])


def test_mmw_image_heavy_recommendation_is_memory_aware():
    cfg = {
        "experiment": {"task": "fusion"},
        "data": {
            "dataset": {"type": "mmw", "seq_len": 8},
            "dataloader": {"batch_size": 8, "num_workers": 4, "prefetch_factor": 2, "persistent_workers": True},
            "cache": {"policy": "off"},
        },
        "model": {"student": {"modalities": ["image", "gps", "mmwave"]}},
        "training": {"amp": {}, "transfer": {}},
    }

    result = recommend_parallel_training(
        cfg,
        config_path="configs/hist_beam/mmw_scenario_loso.yaml",
        parallel_runs=4,
        cpu_count=16,
        check_cache=False,
        profile={"io_risk": {"loader_wait_dominates_step": True}, "message": "exit code 137 killed"},
    )

    mmw = result["recommendations"]["mmw_image_heavy"]
    assert mmw["enabled"] is True
    assert mmw["train_num_workers_upper_bound"] <= 1
    assert mmw["recommended_batch_size"] <= 2
    assert mmw["recommended_parallel_runs"] <= 2
    assert "data.cache.image.policy=auto" in result["overrides"]
    assert "data.dataloader.train_persistent_workers=false" in result["overrides"]
    assert "training.amp.enabled=true" in result["optional_overrides"]
    assert "AMP does not reduce PNG decode" in mmw["amp_limit"]
    assert "cap_train_num_workers" in mmw["actions"]
