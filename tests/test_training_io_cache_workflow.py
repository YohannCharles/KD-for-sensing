import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from kd_sensing.config import load_config
from kd_sensing.config.io import dump_config, safe_load_yaml
from kd_sensing.cli.preprocess import _apply_scene_override_to_sequence_preprocess
from kd_sensing.data.beam_soft_targets import beam_power_to_distribution, gaussian_beam_distribution
import kd_sensing.data.datasets.deepsense6g as deepsense6g_module
import kd_sensing.data.datasets.deepsense6g_targets as deepsense6g_targets
import kd_sensing.data.transform_ops.io as io_transforms
import kd_sensing.data.transform_ops.lidar as lidar_transforms
import kd_sensing.preprocessing.lidar as lidar_preprocessing
from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset
from kd_sensing.data.layouts import (
    deepsense6g_scene_layout,
    mmw_condition_layout,
    physical_labels_cache_root,
    runtime_cache_root,
)
from kd_sensing.data.samples import create_samples
from kd_sensing.data.scenes import retarget_deepsense_dataset_config
from kd_sensing.losses.beam import FocalLoss, SoftTargetCrossEntropyLoss
from kd_sensing.engine.batch import prepare_fusion_inputs, prepare_labels, prepare_soft_beam_targets
from kd_sensing.engine.batch_step import BatchStepRunner
from kd_sensing.engine.cache_policy import apply_cache_policy
from kd_sensing.engine.data_factory import (
    build_dataloader,
    build_dataloaders,
    build_dataset,
    build_dataloader_kwargs,
    build_protocol_split_datasets,
    shutdown_dataloader_workers,
)
from kd_sensing.engine.epoch_subsampling import EpochSubsampleSampler
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.engine.training_extensions import ExtensionContext, NoOpTrainingExtension
from kd_sensing.engine.model_output import adapt_model_output, select_prediction_slots
from kd_sensing.engine.runtime import resolve_amp_settings, transfer_non_blocking
from kd_sensing.engine.evaluator import _evaluation_split_protocol_report
from kd_sensing.engine.run_metadata import dataset_run_metadata, prediction_setup_metadata, throughput_run_metadata
from kd_sensing.engine.training_metrics import training_outputs_payload
from kd_sensing.engine.throughput_recommendations import (
    lidar_cache_coverage,
    recommend_parallel_training,
)
import kd_sensing.engine.training_io_profile as profile_training_io
from kd_sensing.engine.trainer import (
    _configure_early_stopping,
    _early_stopping_improved,
    _early_stopping_min_epoch,
    _early_stopping_metric_value,
    _validate_early_stopping_source_available,
    _write_tensorboard_scalars,
    _write_tensorboard_startup_scalars,
    create_eval_run_dir,
    create_run_dir,
    train,
)
from kd_sensing.preprocessing.sequences import generate_sequence_data
from kd_sensing.utils.artifact_registry import (
    archive_best_checkpoint,
    find_registry_checkpoint,
    load_checkpoint_metadata,
    resolve_evaluation_checkpoint,
    write_sidecar,
)
from kd_sensing.utils.checkpoint import save_checkpoint
from tests.training_io_helpers import (
    _DisabledGradScaler,
    _TinyImageBatchModel,
    _removed_encoder_name,
    _removed_image_option,
    _removed_image_profile,
    _seq_index_keys_for_dataset,
    _write_aux_training_csv,
    _write_camera_files,
    _write_full_sequence_fixture,
    _write_minimal_csv,
    _write_multirow_gps_sequence_fixture,
    _write_stratified_gps_sequence_fixture,
)

ROOT = Path(__file__).resolve().parents[1]

def test_cache_policy_resolves_lidar_and_supported_image_policy():
    cfg = {
        "data": {"cache": {"policy": "read_only", "lidar": {"policy": "auto"}}, "dataset": {}},
        "experiment": {"task": "fusion"},
        "model": {"primary": {"modalities": ["image", "lidar"]}},
    }
    dataset_cfg = {
        "lidar_use_cache": None,
        "lidar_write_cache": None,
    }

    resolved = apply_cache_policy(dataset_cfg, cfg, ("image", "lidar"))

    assert resolved["lidar_use_cache"] is True
    assert resolved["lidar_write_cache"] is True
    assert resolved["lidar_cache_policy"] == "auto"
    assert resolved["image_cache_policy"] == "read_only"

    image_cfg = {}
    apply_cache_policy(
        image_cfg,
        {"data": {"cache": {"policy": "off", "image": {"policy": "auto", "cache_dir": "image_cache"}}}},
        ("image",),
    )
    assert image_cfg["image_cache_policy"] == "auto"
    assert image_cfg["image_use_cache"] is True
    assert image_cfg["image_write_cache"] is True
    assert image_cfg["image_cache_dir"] == "image_cache"
    with pytest.raises(ValueError, match="Removed image motion cache"):
        apply_cache_policy({}, {"data": {"cache": {"policy": "auto", "image_motion_policy": "auto"}}}, ("image",))

def test_load_config_accepts_rgb_image_cache_policy_and_rejects_motion_cache():
    cfg = load_config(
        ROOT / "configs/fusion/image_gps_supervised.yaml",
        ["data.cache.image.policy=read_only"],
    )

    assert cfg["data"]["cache"]["image"]["policy"] == "read_only"

    with pytest.raises(ValueError, match="Removed image motion cache option"):
        load_config(
            ROOT / "configs/fusion/image_gps_supervised.yaml",
            ["data.cache.image_motion_policy=auto"],
        )

def test_parallel_training_recommendation_outputs_background_overrides():
    cfg = load_config(ROOT / "configs/fusion/image_radar_gps_lidar_mmwave_beam_supervised.yaml")

    result = recommend_parallel_training(
        cfg,
        config_path="configs/fusion/image_radar_gps_lidar_mmwave_beam_supervised.yaml",
        parallel_runs=4,
        cpu_count=32,
        check_cache=False,
    )

    assert result["modalities"] == ["image", "radar", "gps", "lidar", "mmwave"]
    assert "output.progress.enabled=false" in result["overrides"]
    assert "data.dataloader.test_persistent_workers=false" in result["overrides"]
    assert "data.cache.policy=auto" in result["overrides"]
    assert result["recommendations"]["prefetch_factor"] == 1
    assert "training.amp.enabled=true" in result["optional_overrides"]
    assert result["commands"]["train"].startswith("conda run -n kd_mm_beam kd-sensing-train")

def test_parallel_training_recommendation_warns_when_lidar_cache_is_cold(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    _write_minimal_csv(train_csv, camera=False, radar=False, gps=False, lidar=True)
    _write_minimal_csv(test_csv, camera=False, radar=False, gps=False, lidar=True)
    cfg = load_config(
        ROOT / "configs/lidar/lightweight.yaml",
        [
            f"data.dataset.data_root={tmp_path}",
            f"data.dataset.train_csv_name={train_csv.name}",
            f"data.dataset.test_csv_name={test_csv.name}",
            "data.dataset.seq_len=1",
            "data.dataset.num_pred=1",
            "data.dataset.lidar_cache_dir=lidar_cache",
        ],
    )

    coverage = lidar_cache_coverage(cfg)
    result = recommend_parallel_training(cfg, parallel_runs=4, cpu_count=32)

    assert coverage["coverage"] == 0.0
    assert coverage["missing"] == 1
    assert "data.cache.policy=auto" in result["overrides"]
    assert result["cache"]["lidar"]["status"] == "cold"
    assert result["cache"]["prewarm_command"] is not None

def test_cache_policy_non_relevant_modalities_are_disabled():
    dataset_cfg = {
        "lidar_use_cache": None,
        "lidar_write_cache": None,
    }

    apply_cache_policy(dataset_cfg, {"data": {"cache": {"policy": "auto"}}}, ("radar", "mmwave"))

    assert dataset_cfg["lidar_use_cache"] is False
    assert dataset_cfg["lidar_write_cache"] is False
    assert dataset_cfg["lidar_cache_policy"] == "off"

def test_build_dataset_auto_policy_uses_rgb_image_without_cache_metadata(tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=2, num_pred=1)
    _write_camera_files(tmp_path, count=2)
    cfg = {
        "experiment": {"task": "image"},
        "data": {
            "cache": {"policy": "auto"},
            "dataset": {
                "type": "deepsense6g",
                "scene": 9,
                "data_root": str(tmp_path),
                "train_csv_name": csv_path.name,
                "test_csv_name": csv_path.name,
                "seq_len": 2,
                "num_pred": 1,
                "image_profile": "rgb_imagenet",
                "image_size": [8, 8],
                "beam_label_cache": "lazy",
            },
        },
        "model": {"primary": {}},
    }

    dataset = build_dataset(cfg, "train")
    sample = dataset[0]
    metadata = dataset_run_metadata(dataset)

    assert sample["image"].shape == (2, 3, 8, 8)
    assert not hasattr(dataset, _removed_image_option("cache_dir"))
    assert metadata["scene_id"] == 9
    assert metadata["scene_slug"] == "scene9"
    assert _removed_image_option("cache_policy") not in metadata

def test_image_derived_cache_hit_miss_and_read_only_policy(tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=2, num_pred=1)
    _write_camera_files(tmp_path, count=2)
    cfg = {
        "experiment": {"task": "image"},
        "data": {
            "cache": {"policy": "off", "image": {"policy": "auto", "cache_dir": "image_cache"}},
            "dataset": {
                "type": "deepsense6g",
                "scene": 9,
                "data_root": str(tmp_path),
                "train_csv_name": csv_path.name,
                "test_csv_name": csv_path.name,
                "seq_len": 2,
                "num_pred": 1,
                "image_profile": "rgb_imagenet",
                "image_size": [8, 8],
            },
        },
        "model": {"primary": {"modalities": ["image"]}},
    }

    dataset = build_dataset(cfg, "train")
    uncached = dataset[0]["image"]
    generated = sorted((tmp_path / "image_cache").rglob("*.npy"))
    assert len(generated) == 2
    assert dataset.image_cache_metadata()["generated"] == 2

    read_only_cfg = deepcopy(cfg)
    read_only_cfg["data"]["cache"]["image"]["policy"] = "read_only"
    read_only = build_dataset(read_only_cfg, "train")
    cached = read_only[0]["image"]

    torch.testing.assert_close(cached, uncached)
    assert read_only.image_cache_metadata()["hits"] == 2

def test_disabled_image_modality_does_not_initialize_image_cache(tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=1, num_pred=1)
    cfg = {
        "experiment": {"task": "fusion"},
        "data": {
            "cache": {"policy": "off", "image": {"policy": "auto", "cache_dir": "image_cache"}},
            "dataset": {
                "type": "deepsense6g",
                "scene": 9,
                "data_root": str(tmp_path),
                "train_csv_name": csv_path.name,
                "test_csv_name": csv_path.name,
                "seq_len": 1,
                "num_pred": 1,
                "gps_normalize": False,
                "mmwave_normalize": False,
            },
        },
        "model": {"primary": {"modalities": ["gps"]}},
    }

    dataset = build_dataset(cfg, "train")

    assert "image" not in dataset.enabled_modalities
    assert dataset.image_cache is None
    assert not (tmp_path / "image_cache").exists()

def test_gps_mmwave_scaler_fit_does_not_retain_per_sample_sequence_cache(tmp_path: Path):
    csv_path = tmp_path / "train.csv"
    rows = []
    for row_idx in range(2):
        gps_name = f"gps_{row_idx}.txt"
        bs_name = f"bs_{row_idx}.txt"
        mmwave_name = f"mmwave_{row_idx}.txt"
        beam_name = f"beam_{row_idx}.txt"
        future_name = f"future_{row_idx}.txt"
        np.savetxt(tmp_path / gps_name, np.asarray([42.0 + row_idx * 1e-5, -71.0], dtype=np.float32))
        np.savetxt(tmp_path / bs_name, np.asarray([42.0, -71.0], dtype=np.float32))
        np.savetxt(tmp_path / mmwave_name, np.linspace(1.0 + row_idx, 64.0 + row_idx, 64, dtype=np.float32))
        beam = np.zeros(64, dtype=np.float32)
        beam[row_idx] = 1.0
        np.savetxt(tmp_path / beam_name, beam)
        np.savetxt(tmp_path / future_name, beam)
        rows.append([gps_name, bs_name, mmwave_name, beam_name, future_name, str(row_idx)])
    csv_path.write_text(
        "gps1,bs_gps1,mmwave1,beam1,future_beam1,seq_index\n"
        + "\n".join(",".join(row) for row in rows)
        + "\n",
        encoding="utf-8",
    )

    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=csv_path.name,
        split="train",
        seq_len=1,
        num_pred=1,
        enabled_modalities=["gps", "mmwave"],
        gps_normalize=True,
        mmwave_normalize=True,
    )
    metadata = dataset_run_metadata(dataset)

    assert dataset._gps_feature_cache == {}
    assert dataset._mmwave_feature_cache == {}
    assert metadata["gps_scaler"]["streaming"] is True
    assert metadata["mmwave_scaler"]["streaming"] is True
    assert metadata["gps_scaler"]["retains_per_sample_sequence_cache"] is False
    assert metadata["mmwave_scaler"]["retains_per_sample_sequence_cache"] is False

def test_dataset_does_not_resolve_unenabled_cache_dirs(tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=1, num_pred=1)

    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=1,
        num_pred=1,
        enabled_modalities=["radar"],
        lidar_cache_dir="lidar_cache",
        lidar_use_cache=True,
        lidar_write_cache=True,
    )

    assert dataset.lidar_cache_dir is None

def test_atomic_save_npy_overwrites_without_visible_temp_files(tmp_path: Path):
    target = tmp_path / "cache.npy"

    io_transforms.atomic_save_npy(target, np.ones((2, 2), dtype=np.float32))
    io_transforms.atomic_save_npy(target, np.zeros((2, 2), dtype=np.float32))

    assert np.load(target).sum() == 0.0
    assert list(tmp_path.glob("*.tmp")) == []

def test_throughput_metadata_includes_cache_policy():
    metadata = throughput_run_metadata(
        {
            "experiment": {"task": "fusion"},
            "data": {"cache": {"policy": "read_only", "lidar": {"policy": "auto"}}, "dataloader": {}},
            "model": {"primary": {"modalities": ["image", "lidar"]}},
            "training": {"transfer": {}, "amp": {}},
        }
    )

    assert metadata["cache"]["policy"] == "read_only"
    assert metadata["cache"]["image"]["input"] == "rgb_imagenet"
    assert metadata["cache"]["image"]["policy"] == "read_only"
    assert metadata["cache"]["lidar"]["policy"] == "auto"
    assert metadata["cache"]["enabled_modalities"] == ["image", "lidar"]
    assert metadata["dataloader_splits"]["train"]["batch_size"] == 3
    assert metadata["dataloader_splits"]["test"]["persistent_workers"] is False
    assert metadata["progress"]["enabled"] is True

def test_mmw_profile_helpers_mark_image_heavy_loader_wait():
    cfg = {
        "experiment": {"task": "fusion"},
        "data": {
            "dataset": {"type": "mmw", "seq_len": 8, "image_profile": "rgb_imagenet"},
            "dataloader": {"batch_size": 4, "num_workers": 2, "prefetch_factor": 2, "persistent_workers": True},
            "cache": {"policy": "off", "image": {"policy": "auto"}},
        },
        "model": {"primary": {"modalities": ["image", "gps", "mmwave"]}},
        "training": {"transfer": {}, "amp": {}},
    }
    runtime = throughput_run_metadata(cfg)
    mmw = profile_training_io._mmw_sensor_profile_summary(cfg, runtime)
    risk = profile_training_io._io_risk_summary(
        wait_breakdown={"p95_spikes": {"wait_gt_gpu_step": True}},
        mmw_sensor_profile=mmw,
    )

    assert mmw["image_heavy"] is True
    assert mmw["worker_memory_risk"] is True
    assert risk["loader_wait_dominates_step"] is True
    assert "enable_or_prewarm_image_derived_cache" in risk["primary_actions"]

def test_lidar_cache_hit_miss_write_and_parameter_isolation(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=1, num_pred=1)
    build_calls = {"count": 0}

    def fake_build(*args, **kwargs):  # noqa: ARG001
        build_calls["count"] += 1
        return np.ones((3, 4, 4), dtype=np.float32)

    monkeypatch.setattr(lidar_transforms, "build_lidar_bev", fake_build)
    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=1,
        num_pred=1,
        enabled_modalities=["lidar"],
        lidar_bev_size=[4, 4],
        lidar_roi=[0.0, 2.0, -1.0, 1.0, -1.0, 1.0],
        lidar_cache_dir="lidar_cache",
        lidar_use_cache=True,
        lidar_write_cache=True,
    )
    sample = dataset[0]
    cache_files = list(dataset.lidar_cache_dir.glob("*.npy"))

    assert sample["lidar"].shape == (1, 3, 4, 4)
    assert build_calls["count"] == 1
    assert len(cache_files) == 1

    monkeypatch.setattr(lidar_transforms, "build_lidar_bev", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    cached_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=1,
        num_pred=1,
        enabled_modalities=["lidar"],
        lidar_bev_size=[4, 4],
        lidar_roi=[0.0, 2.0, -1.0, 1.0, -1.0, 1.0],
        lidar_cache_dir="lidar_cache",
        lidar_use_cache=True,
        lidar_write_cache=False,
    )
    cached = cached_dataset[0]
    different_roi_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=1,
        num_pred=1,
        enabled_modalities=["lidar"],
        lidar_bev_size=[4, 4],
        lidar_roi=[0.0, 3.0, -1.0, 1.0, -1.0, 1.0],
        lidar_cache_dir="lidar_cache",
        lidar_use_cache=True,
    )

    assert cached["lidar"].shape == (1, 3, 4, 4)
    assert cached_dataset.lidar_cache_dir == dataset.lidar_cache_dir
    assert different_roi_dataset.lidar_cache_dir != dataset.lidar_cache_dir

def test_lidar_cache_initialization_does_not_load_cache(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=1, num_pred=1)
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=1,
        num_pred=1,
        enabled_modalities=["lidar"],
        lidar_cache_dir="lidar_cache",
        lidar_use_cache=True,
    )

    assert dataset.lidar_cache_dir is not None

def test_rgb_image_path_reads_frames_without_cache(tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=2, num_pred=1)
    _write_camera_files(tmp_path, count=2)

    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=2,
        num_pred=1,
        enabled_modalities=["image"],
        image_profile="rgb_imagenet",
        image_size=[8, 8],
    )
    sample = dataset[0]

    assert sample["image"].shape == (2, 3, 8, 8)
    assert not hasattr(dataset, _removed_image_option("cache_dir"))

def test_removed_image_path_config_is_rejected():
    removed_use_key = _removed_image_option("use_cache")
    removed_profile = _removed_image_profile()
    removed_encoder = _removed_encoder_name()
    removed_legacy_encoder = _removed_encoder_name(prefix="legacy_")

    with pytest.raises(ValueError, match="Removed image motion"):
        load_config(ROOT / "configs/image/strong.yaml", [f"data.dataset.{removed_use_key}=true"])
    with pytest.raises(ValueError, match="has been removed"):
        load_config(ROOT / "configs/image/strong.yaml", [f"data.dataset.image_profile={removed_profile}"])
    with pytest.raises(ValueError, match="Removed image encoder"):
        load_config(
            ROOT / "configs/image/resnet18_strong.yaml",
            [f"model.primary.encoders.image.type={removed_encoder}"],
        )
    with pytest.raises(ValueError, match="Removed image encoder"):
        load_config(
            ROOT / "configs/image/resnet18_strong.yaml",
            [f"model.primary.encoders.image.type={removed_legacy_encoder}"],
        )

def test_lidar_cache_preprocessor_skips_existing_and_writes_metadata(monkeypatch, tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    train_csv.write_text("lidar1,lidar2\nlidar_0.txt,lidar_1.txt\n", encoding="utf-8")
    test_csv.write_text("lidar1,lidar2\nlidar_1.txt,lidar_0.txt\n", encoding="utf-8")
    build_calls = {"count": 0}

    def fake_build(*args, **kwargs):  # noqa: ARG001
        build_calls["count"] += 1
        return np.ones((3, 4, 4), dtype=np.float32)

    monkeypatch.setattr(lidar_preprocessing, "build_lidar_bev", fake_build)
    first = lidar_preprocessing.generate_lidar_bev_cache(
        csv_paths=[train_csv, test_csv],
        data_root=tmp_path,
        cache_dir=tmp_path / "lidar_cache",
        bev_size=[4, 4],
        roi=[0.0, 2.0, -1.0, 1.0, -1.0, 1.0],
        progress=False,
    )
    second = lidar_preprocessing.generate_lidar_bev_cache(
        csv_paths=[train_csv, test_csv],
        data_root=tmp_path,
        cache_dir=tmp_path / "lidar_cache",
        bev_size=[4, 4],
        roi=[0.0, 2.0, -1.0, 1.0, -1.0, 1.0],
        progress=False,
    )

    assert first["count"] == 2
    assert first["generated"] == 2
    assert second["skipped"] == 2
    assert build_calls["count"] == 2
    assert (Path(first["cache_dir"]) / "metadata.json").exists()

def test_beam_label_cache_reuses_repeated_paths(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    beam = np.zeros(64, dtype=np.float32)
    beam[7] = 1.0
    np.savetxt(tmp_path / "beam.txt", beam)
    csv_path.write_text(
        "radar1,radar2,beam1,beam2,future_beam1,seq_index\n"
        "radar_0_RA.npy,radar_1_RA.npy,beam.txt,beam.txt,beam.txt,1\n",
        encoding="utf-8",
    )
    calls = {"count": 0}
    real_loadtxt = np.loadtxt

    def counting_loadtxt(*args, **kwargs):
        calls["count"] += 1
        return real_loadtxt(*args, **kwargs)

    monkeypatch.setattr(np, "loadtxt", counting_loadtxt)
    monkeypatch.setattr(
        deepsense6g_module,
        "load_radar_maps",
        lambda *args, **kwargs: (torch.zeros(2, 4, 4), torch.zeros(2, 6, 4)),  # noqa: ARG005
    )
    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=2,
        num_pred=1,
        enabled_modalities=["radar"],
        beam_label_cache="lazy",
        fft_tuple=[4, 8, 6],
        clipped_range=4,
    )
    sample = dataset[0]

    assert sample["input_beam"].tolist() == [7, 7]
    assert sample["target_beam"].tolist() == [7]
    assert calls["count"] == 1
