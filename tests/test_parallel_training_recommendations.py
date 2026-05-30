from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.engine.throughput_recommendations import recommend_parallel_training  # noqa: E402


def test_lidar_parallel_recommendation_uses_generic_cache_policy():
    cfg = {
        "experiment": {"task": "lidar"},
        "data": {
            "dataset": {"type": "deepsense6g", "use_lidar": True},
            "dataloader": {"batch_size": 8, "num_workers": 4},
            "cache": {"policy": "off"},
        },
        "model": {"student": {"modalities": ["lidar"]}},
        "training": {"amp": {}},
    }

    result = recommend_parallel_training(
        cfg,
        config_path="configs/lidar/student_no_kd.yaml",
        parallel_runs=3,
        cpu_count=12,
        check_cache=False,
    )

    assert result["modalities"] == ["lidar"]
    assert "data.cache.policy=auto" in result["overrides"]
    assert result["cache"]["lidar"] == {"status": "skipped", "enabled": True, "coverage": 0.0}
    assert result["recommendations"]["cache_policy"] == "auto"


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
