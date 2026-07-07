import json

from kd_sensing.cli import training_throughput as throughput_cli
from kd_sensing.engine.throughput_recommendations import recommend_parallel_training


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
        config_path="configs/fusion/image_gps_supervised.yaml",
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


def test_training_throughput_cli_routes_profile_and_recommend_modes(tmp_path, monkeypatch):
    profile_calls = {}
    recommend_calls = {}

    def fake_profile_training_io(**kwargs):
        profile_calls.update(kwargs)
        return {"mode": "profile", "config": str(kwargs["config_path"])}

    def fake_load_config(path, overrides):
        return {"path": str(path), "overrides": list(overrides)}

    def fake_recommend_parallel_training(cfg, **kwargs):
        recommend_calls["cfg"] = cfg
        recommend_calls.update(kwargs)
        return {"mode": "recommend", "overrides": ["output.progress.enabled=false"]}

    monkeypatch.setattr(throughput_cli, "profile_training_io", fake_profile_training_io)
    monkeypatch.setattr(throughput_cli, "load_config", fake_load_config)
    monkeypatch.setattr(throughput_cli, "recommend_parallel_training", fake_recommend_parallel_training)

    profile = throughput_cli.run(
        [
            "--mode",
            "profile",
            "--config",
            "configs/image/lightweight.yaml",
            "--split",
            "test",
            "--samples",
            "2",
            "--warmup",
            "0",
            "--device",
            "cpu",
            "--csv-output",
            str(tmp_path / "profile.csv"),
            "-o",
            "data.dataset.length=1",
        ]
    )
    profile_json = tmp_path / "profile.json"
    profile_json.write_text(json.dumps({"io_risk": {"loader_wait_dominates_step": True}}), encoding="utf-8")
    output_json = tmp_path / "recommend.json"
    recommend = throughput_cli.run(
        [
            "--mode",
            "recommend",
            "--config",
            "configs/fusion/image_gps_supervised.yaml",
            "--parallel-runs",
            "2",
            "--cpu-count",
            "8",
            "--profile-json",
            str(profile_json),
            "--output",
            str(output_json),
            "-o",
            "data.cache.policy=auto",
        ]
    )

    assert profile == {"mode": "profile", "config": "configs/image/lightweight.yaml"}
    assert profile_calls["split"] == "test"
    assert profile_calls["device_override"] == "cpu"
    assert profile_calls["overrides"] == ["data.dataset.length=1"]
    assert recommend == {"mode": "recommend", "overrides": ["output.progress.enabled=false"]}
    assert recommend_calls["cfg"]["overrides"] == ["data.cache.policy=auto"]
    assert recommend_calls["parallel_runs"] == 2
    assert recommend_calls["profile"]["io_risk"]["loader_wait_dominates_step"] is True
    assert json.loads(output_json.read_text(encoding="utf-8"))["mode"] == "recommend"
