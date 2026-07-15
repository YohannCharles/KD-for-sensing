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
from kd_sensing.data.datasets.synthetic import SyntheticSequenceDataset
import kd_sensing.data.transform_ops.io as io_transforms
import kd_sensing.data.transform_ops.lidar as lidar_transforms
from kd_sensing.data.transform_ops.gps import GPSStandardScaler
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
import kd_sensing.engine.data_factory as data_factory
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
from kd_sensing.engine.run_metadata import (
    dataloaders_run_metadata,
    dataset_run_metadata,
    prediction_setup_metadata,
    throughput_run_metadata,
)
from kd_sensing.engine.training_metrics import training_outputs_payload
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

def test_deepsense_scene_defaults_and_aliases():
    default_cfg = load_config(ROOT / "configs/mmwave/strong.yaml")
    scene9_cfg = load_config(ROOT / "configs/mmwave/strong.yaml", ["data.dataset.scene=scene9"])
    scene31_cfg = load_config(ROOT / "configs/mmwave/strong.yaml", ["data.dataset.scene=scenario31"])
    scene32_cfg = load_config(ROOT / "configs/mmwave/strong.yaml", ["data.dataset.scene=scene32"])

    assert default_cfg["data"]["dataset"]["scene_id"] == 31
    assert default_cfg["data"]["dataset"]["scene_slug"] == "scene31"
    assert default_cfg["data"]["dataset"]["data_root"] == "dataset/DeepSense6G/scenario31"
    assert scene9_cfg["data"]["dataset"]["scene_id"] == 9
    assert scene9_cfg["data"]["dataset"]["scene_slug"] == "scene9"
    assert scene9_cfg["data"]["dataset"]["data_root"] == "dataset/DeepSense6G/scenario9"
    assert scene31_cfg["data"]["dataset"]["scene_id"] == 31
    assert scene31_cfg["data"]["dataset"]["scene_slug"] == "scene31"
    assert scene31_cfg["data"]["dataset"]["data_root"] == "dataset/DeepSense6G/scenario31"
    assert scene32_cfg["data"]["dataset"]["scene_id"] == 32
    assert scene32_cfg["data"]["dataset"]["scene_slug"] == "scene32"
    assert scene32_cfg["data"]["dataset"]["data_root"] == "dataset/DeepSense6G/scenario32"

def test_dataset_layout_helpers_define_supported_roots():
    scene31 = deepsense6g_scene_layout(31)
    scene9 = deepsense6g_scene_layout("scenario9")
    sunny = mmw_condition_layout("sunny")
    rainy = mmw_condition_layout("rainy")
    foggy = mmw_condition_layout("foggy")

    assert scene31.canonical_root == "dataset/DeepSense6G/scenario31"
    assert scene31.legacy_root == "dataset/scenario31"
    assert scene31.radar_csv_path == "dataset/DeepSense6G/scenario31/scenario31_RA.csv"
    assert scene31.image_cache_root == "outputs/cache/DeepSense6G/scenario31/image_derived"
    assert scene31.lidar_bev_cache_root == "outputs/cache/DeepSense6G/scenario31/lidar_bev"
    assert scene9.canonical_root == "dataset/DeepSense6G/scenario9"
    assert sunny.sensor_data_root == "dataset/MMW/sunny/Sensor_Data"
    assert sunny.channel_data_root == "dataset/MMW/sunny/Channel_Data"
    assert sunny.image_cache_root == "outputs/cache/MMW/sunny/image_derived"
    assert sunny.lidar_bev_cache_root == "outputs/cache/MMW/sunny/lidar_bev"
    assert sunny.prepared_scenario_root("Town10_skybridge_seed24") == (
        "dataset/MMW/sunny/Prepared/Town10_skybridge_seed24"
    )
    assert rainy.required_subdirs == ("Sensor_Data", "Channel_Data")
    assert foggy.root == "dataset/MMW/foggy"
    assert runtime_cache_root() == "outputs/cache"
    assert physical_labels_cache_root() == "outputs/cache/physical_labels"

def test_deepsense_explicit_legacy_root_is_preserved_by_normalize_and_retarget():
    cfg = load_config(
        ROOT / "configs/mmwave/strong.yaml",
        ["data.dataset.scene=31", "data.dataset.data_root=dataset/scenario31"],
    )
    dataset_cfg = cfg["data"]["dataset"]

    assert dataset_cfg["scene_id"] == 31
    assert dataset_cfg["data_root"] == "dataset/scenario31"

    retarget_deepsense_dataset_config(dataset_cfg, 32)

    assert dataset_cfg["scene_id"] == 32
    assert dataset_cfg["scene_slug"] == "scene32"
    assert dataset_cfg["data_root"] == "dataset/scenario31"

@pytest.mark.parametrize(("removed_type", "scene"), [("scenario9", 9), ("scenario31", 31), ("scenario32", 32)])
def test_deepsense_scene_specific_dataset_types_are_rejected(removed_type: str, scene: int):
    with pytest.raises(ValueError, match=f"deepsense6g.*scene: {scene}"):
        load_config(
            ROOT / "configs/mmwave/strong.yaml",
            [f"data.dataset.type={removed_type}", "data.dataset.scene=null"],
        )

def test_deepsense_unknown_scene_is_rejected():
    with pytest.raises(ValueError, match="Supported scenes"):
        load_config(ROOT / "configs/mmwave/strong.yaml", ["data.dataset.scene=99"])

def test_sequence_preprocess_scene_override_updates_root_and_csv():
    pre_cfg = {
        "type": "sequence_csv",
        "csv_path": "dataset/scenario32/scenario32_RA.csv",
        "data_root": "dataset/scenario32",
    }
    cfg = {"data": {"dataset": {"type": "deepsense6g", "scene": 9}}}

    _apply_scene_override_to_sequence_preprocess(pre_cfg, cfg)

    assert pre_cfg["data_root"] == "dataset/DeepSense6G/scenario9"
    assert pre_cfg["csv_path"] == "dataset/DeepSense6G/scenario9/scenario9_RA.csv"

    cfg["data"]["dataset"]["scene"] = 31
    _apply_scene_override_to_sequence_preprocess(pre_cfg, cfg)

    assert pre_cfg["data_root"] == "dataset/DeepSense6G/scenario31"
    assert pre_cfg["csv_path"] == "dataset/DeepSense6G/scenario31/scenario31_RA.csv"

def test_sequence_preprocess_scene_override_keeps_absolute_custom_paths(tmp_path: Path):
    pre_cfg = {
        "type": "sequence_csv",
        "csv_path": str(tmp_path / "custom_RA.csv"),
        "data_root": str(tmp_path),
    }
    cfg = {"data": {"dataset": {"type": "deepsense6g", "scene": 9}}}

    _apply_scene_override_to_sequence_preprocess(pre_cfg, cfg)

    assert pre_cfg["data_root"] == str(tmp_path)
    assert pre_cfg["csv_path"] == str(tmp_path / "custom_RA.csv")

def test_deepsense_csv_relative_paths_resolve_from_scene_root():
    assert io_transforms.joined_resource(
        "dataset/DeepSense6G/scenario31",
        "/unit1/radar_data_RA/sample.npy",
    ) == Path("dataset/DeepSense6G/scenario31/unit1/radar_data_RA/sample.npy")
    assert io_transforms.joined_resource(
        "dataset/scenario31",
        "/unit1/pwr/sample.txt",
    ) == Path("dataset/scenario31/unit1/pwr/sample.txt")

@pytest.mark.parametrize(
    ("enabled", "expected_calls", "expected_fields"),
    [
        (["image"], {"image"}, {"image"}),
        (["radar"], {"radar"}, {"radar_ra", "radar_da"}),
        (["gps"], {"gps"}, {"gps"}),
        (["lidar"], {"lidar"}, {"lidar", "lidar_raw"}),
        (["radar", "gps"], {"radar", "gps"}, {"radar_ra", "radar_da", "gps"}),
    ],
)
def test_deepsense_scene9_loads_only_enabled_modalities(
    monkeypatch,
    tmp_path: Path,
    enabled,
    expected_calls,
    expected_fields,
):
    csv_path = tmp_path / "seq.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=3, num_pred=1)
    calls: set[str] = set()

    def fake_image(*args, **kwargs):  # noqa: ARG001
        calls.add("image")
        return torch.zeros(3, 3, 8, 8)

    def fake_radar(*args, **kwargs):  # noqa: ARG001
        calls.add("radar")
        return torch.zeros(3, 4, 4), torch.zeros(3, 6, 4)

    def fake_gps(*args, **kwargs):  # noqa: ARG001
        calls.add("gps")
        return np.zeros((3, 3), dtype=np.float32)

    def fake_lidar(self, idx: int, *, augment: bool):  # noqa: ARG001
        calls.add("lidar")
        return np.zeros((3, 3, 4, 4), dtype=np.float32)

    monkeypatch.setattr(deepsense6g_module, "load_rgb_imagenet_frames", fake_image)
    monkeypatch.setattr(deepsense6g_module, "load_radar_maps", fake_radar)
    monkeypatch.setattr(deepsense6g_module, "load_gps_feature_sequence", fake_gps)
    monkeypatch.setattr(DeepSense6GDataset, "_lidar_bev_for_index", fake_lidar)

    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=3,
        num_pred=1,
        enabled_modalities=enabled,
        image_profile="rgb_imagenet",
        gps_normalize=False,
        lidar_normalize=False,
    )
    sample = dataset[0]
    timings = dataset.profile_getitem_components(0)

    assert calls == expected_calls
    assert set(sample) == expected_fields | {"input_beam", "target_beam", "history_indices", "target_index"}
    for key in ("image", "radar", "gps", "lidar", "mmwave", "auxiliary_targets"):
        assert key in timings
        assert timings[key] >= 0.0

def test_deepsense_target_provider_skips_disabled_target_resources(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "train_aux.csv"
    _write_aux_training_csv(tmp_path, csv_path, prefix="train", future_max=[1.0, 5.0])

    def fail_occlusion(*args, **kwargs):  # noqa: ANN001, ARG001
        raise AssertionError("disabled occlusion target should not read mmWave power")

    def fail_position(*args, **kwargs):  # noqa: ANN001, ARG001
        raise AssertionError("disabled position target should not read future GPS resources")

    monkeypatch.setattr(deepsense6g_targets, "finite_max_mmwave_power", fail_occlusion)
    monkeypatch.setattr(deepsense6g_targets, "load_relative_xy_target_sequence", fail_position)

    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=2,
        num_pred=2,
        enabled_modalities=["gps"],
        gps_normalize=False,
        occlusion_target=False,
        position_target=False,
    )
    sample = dataset[0]

    assert dataset.target_provider.occlusion_target_stats is None
    assert "occlusion_label" not in sample
    assert "position_target" not in sample
    assert set(sample) == {"input_beam", "target_beam", "gps", "history_indices", "target_index"}

def test_deepsense_target_provider_outputs_target_shapes_and_dtypes(tmp_path: Path):
    csv_path = tmp_path / "train_aux.csv"
    _write_aux_training_csv(tmp_path, csv_path, prefix="train", future_max=[1.0, 5.0])

    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=2,
        num_pred=2,
        enabled_modalities=["gps"],
        gps_normalize=False,
        occlusion_target={"enabled": True, "threshold_percentile": 50.0},
        position_target={"enabled": True, "source": "future_gps_local_xy", "normalize": False},
    )
    sample = dataset[0]

    assert dataset.target_provider.occlusion_target_stats is not None
    assert sample["occlusion_label"].shape == (2,)
    assert sample["occlusion_label"].dtype == torch.float32
    assert sample["occlusion_valid"].shape == (2,)
    assert sample["occlusion_valid"].dtype == torch.bool
    assert sample["position_target"].shape == (2, 2)
    assert sample["position_target"].dtype == torch.float32
    assert sample["position_valid"].shape == (2,)
    assert sample["position_valid"].dtype == torch.bool

def test_create_samples_validates_only_enabled_modality_columns(tmp_path: Path):
    csv_path = tmp_path / "image_only.csv"
    _write_minimal_csv(csv_path, camera=True, radar=False, gps=False, lidar=False)

    samples = create_samples(csv_path, enabled_modalities=["image"], seq_len=1, num_pred=1)

    assert len(samples.rgb_paths) == 1
    assert samples.radar_paths == []
    with pytest.raises(ValueError, match="gps is enabled"):
        create_samples(csv_path, enabled_modalities=["gps"], seq_len=1, num_pred=1)
    with pytest.raises(ValueError, match="lidar is enabled"):
        create_samples(csv_path, enabled_modalities=["lidar"], seq_len=1, num_pred=1)

def test_create_samples_portion_uses_even_global_sampling(tmp_path: Path):
    csv_path = tmp_path / "many.csv"
    columns = ["camera1", "beam1", "future_beam1", "seq_index"]
    rows = [
        [f"camera_{idx}.jpg", f"beam_{idx}.txt", f"future_{idx}.txt", str(idx)]
        for idx in range(100)
    ]
    csv_path.write_text(
        ",".join(columns) + "\n" + "\n".join(",".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    samples = create_samples(csv_path, portion=0.05, enabled_modalities=["image"], seq_len=1, num_pred=1)

    assert len(samples.input_beam_paths) == 5
    assert [item[0] for item in samples.input_beam_paths] != [f"beam_{idx}.txt" for idx in range(5)]
    assert samples.metadata["portion_strategy"] == "even"
    assert samples.metadata["seq_index_min"] == 0
    assert samples.metadata["seq_index_max"] == 99

def test_generate_sequence_data_includes_last_legal_window(tmp_path: Path):
    source = tmp_path / "scenario9.csv"
    rows = []
    for idx in range(5):
        rows.append([f"camera_{idx}.jpg", f"radar_{idx}.npy", f"beam_{idx}.txt", "1"])
    source.write_text(
        "unit1_rgb,unit1_radar,unit1_pwr_60ghz,seq_index\n"
        + "\n".join(",".join(row) for row in rows)
        + "\n",
        encoding="utf-8",
    )

    train_path, _ = generate_sequence_data(
        source,
        tmp_path,
        "_test",
        in_len=3,
        out_len=2,
        training_set_pct=1.0,
    )

    frame = pd.read_csv(train_path)
    assert len(frame) == 1
    assert frame.loc[0, "camera1"] == "camera_0.jpg"
    assert frame.loc[0, "future_beam2"] == "beam_4.txt"

def test_num_pred_one_target_shape_and_prepare_labels(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=2, num_pred=1)

    monkeypatch.setattr(
        deepsense6g_module,
        "load_rgb_imagenet_frames",
        lambda *args, **kwargs: torch.zeros(2, 3, 8, 8),  # noqa: ARG005
    )
    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=2,
        num_pred=1,
        enabled_modalities=["image"],
        image_profile="rgb_imagenet",
    )

    sample = dataset[0]
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    labels = prepare_labels(batch, num_pred=1, downsample_ratio=1, device=torch.device("cpu"))

    assert sample["target_beam"].shape == (1,)
    assert batch["target_beam"].shape == (1, 1)
    assert sample["input_beam"].tolist()[-1] == 1
    assert sample["target_beam"].tolist() == [10]
    assert labels.shape == (1, 1)
    assert labels.tolist() == [[10]]

def test_deepsense_dataset_can_use_current_beam_target(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=2, num_pred=1)
    monkeypatch.setattr(
        deepsense6g_module,
        "load_rgb_imagenet_frames",
        lambda *args, **kwargs: torch.zeros(2, 3, 8, 8),  # noqa: ARG005
    )

    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=2,
        num_pred=1,
        enabled_modalities=["image"],
        image_profile="rgb_imagenet",
        beam_target_source="current",
        return_metadata=True,
    )

    sample = dataset[0]

    assert sample["input_beam"].tolist() == [0, 1]
    assert sample["target_beam"].tolist() == [1]
    assert sample["metadata"]["beam_target_source"] == "current"
    assert sample["metadata"]["target_beam_path"] == "beam_1.txt"
    assert sample["metadata"]["future_beam_path"] == "future_0.txt"
    assert sample["metadata"]["target_beam_label_source"] == ["beam_power_argmax"]

def test_future_slot_selection_and_missing_features_supervised_contract():
    labels = prepare_labels(
        {"target_beam": torch.tensor([[0, 1, 2], [1, 2, 3]])},
        num_pred=2,
        downsample_ratio=1,
        device=torch.device("cpu"),
    )
    logits = torch.randn(2, 5, 4)
    selected = select_prediction_slots(logits, num_pred=2)
    model_output = adapt_model_output({"logits": selected})

    assert labels.tolist() == [[0, 1], [1, 2]]
    assert torch.equal(selected, logits[:, -2:, :])
    assert model_output.input_features is None
    assert model_output.output_features is None

def test_dataloader_kwargs_filter_worker_only_options():
    zero_worker = build_dataloader_kwargs(
        {
            "train_batch_size": 4,
            "num_workers": 0,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "train_drop_last": True,
        },
        split="train",
    )
    multi_worker = build_dataloader_kwargs(
        {
            "test_batch_size": 2,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 3,
        },
        split="test",
    )

    assert zero_worker["batch_size"] == 4
    assert zero_worker["drop_last"] is True
    assert "persistent_workers" not in zero_worker
    assert "prefetch_factor" not in zero_worker
    assert multi_worker["batch_size"] == 2
    assert multi_worker["shuffle"] is False
    assert multi_worker["persistent_workers"] is True
    assert multi_worker["prefetch_factor"] == 3

def test_loaded_dataloader_batch_size_alias_overrides_default_split_sizes(tmp_path: Path):
    config_path = tmp_path / "batch_alias.yaml"
    dump_config({"data": {"dataloader": {"batch_size": 8, "num_workers": 0}}}, config_path)

    cfg = load_config(config_path)
    loader_cfg = cfg["data"]["dataloader"]

    assert loader_cfg["train_batch_size"] == 8
    assert loader_cfg["test_batch_size"] == 8
    assert build_dataloader_kwargs(loader_cfg, split="train")["batch_size"] == 8
    assert build_dataloader_kwargs(loader_cfg, split="test")["batch_size"] == 8

def test_loaded_dataloader_batch_size_alias_keeps_explicit_split_size(tmp_path: Path):
    config_path = tmp_path / "batch_alias_split.yaml"
    dump_config(
        {
            "data": {
                "dataloader": {
                    "batch_size": 8,
                    "train_batch_size": 4,
                    "num_workers": 0,
                }
            }
        },
        config_path,
    )

    cfg = load_config(config_path)
    loader_cfg = cfg["data"]["dataloader"]

    assert loader_cfg["train_batch_size"] == 4
    assert loader_cfg["test_batch_size"] == 8

def test_dataloader_kwargs_support_split_specific_worker_options():
    loader_cfg = {
        "train_batch_size": 8,
        "test_batch_size": 2,
        "num_workers": 4,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "test_num_workers": 1,
        "test_persistent_workers": False,
        "test_prefetch_factor": 1,
        "train": {"num_workers": 3, "persistent_workers": True, "prefetch_factor": 4},
    }

    train_kwargs = build_dataloader_kwargs(loader_cfg, split="train")
    test_kwargs = build_dataloader_kwargs(loader_cfg, split="test")

    assert train_kwargs["batch_size"] == 8
    assert train_kwargs["num_workers"] == 3
    assert train_kwargs["persistent_workers"] is True
    assert train_kwargs["prefetch_factor"] == 4
    assert test_kwargs["batch_size"] == 2
    assert test_kwargs["num_workers"] == 1
    assert test_kwargs["persistent_workers"] is False
    assert test_kwargs["prefetch_factor"] == 1

def test_epoch_subsampling_config_validation_defaults_and_limits():
    default_cfg = load_config(ROOT / "configs/gps/lightweight.yaml")
    fraction_cfg = load_config(
        ROOT / "configs/gps/lightweight.yaml",
        [
            "training.epoch_subsampling.enabled=true",
            "training.epoch_subsampling.fraction=0.25",
        ],
    )
    count_cfg = load_config(
        ROOT / "configs/gps/lightweight.yaml",
        [
            "training.epoch_subsampling.enabled=true",
            "training.epoch_subsampling.num_samples=8",
        ],
    )

    assert default_cfg["training"]["epoch_subsampling"] == {
        "enabled": False,
        "fraction": None,
        "num_samples": None,
        "seed": None,
        "rotate_each_epoch": True,
        "shuffle": True,
        "order": None,
        "block_size": None,
    }
    assert fraction_cfg["training"]["epoch_subsampling"]["fraction"] == pytest.approx(0.25)
    assert count_cfg["training"]["epoch_subsampling"]["num_samples"] == 8

@pytest.mark.parametrize(
    "overrides",
    [
        ["training.epoch_subsampling.enabled=true", "training.epoch_subsampling.fraction=0"],
        ["training.epoch_subsampling.enabled=true", "training.epoch_subsampling.fraction=1.5"],
        ["training.epoch_subsampling.enabled=true", "training.epoch_subsampling.num_samples=0"],
        ["training.epoch_subsampling.enabled=true", "training.epoch_subsampling.num_samples=2.5"],
        [
            "training.epoch_subsampling.enabled=true",
            "training.epoch_subsampling.fraction=0.5",
            "training.epoch_subsampling.num_samples=4",
        ],
        ["training.epoch_subsampling.enabled=true"],
    ],
)
def test_epoch_subsampling_config_validation_rejects_invalid_limits(overrides):
    with pytest.raises(ValueError, match="training\\.epoch_subsampling"):
        load_config(ROOT / "configs/gps/lightweight.yaml", overrides)

def test_epoch_subsample_sampler_reproducible_rotation_and_fixed_subset():
    sampler = EpochSubsampleSampler(
        dataset_length=10,
        effective_num_samples=4,
        seed=123,
        rotate_each_epoch=True,
    )
    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)
    resumed = EpochSubsampleSampler(
        dataset_length=10,
        effective_num_samples=4,
        seed=123,
        rotate_each_epoch=True,
    )
    resumed.set_epoch(1)

    assert len(epoch0) == 4
    assert len(set(epoch0)) == 4
    assert epoch0 != epoch1
    assert epoch1 == list(resumed)

    fixed = EpochSubsampleSampler(
        dataset_length=10,
        effective_num_samples=4,
        seed=123,
        rotate_each_epoch=False,
    )
    fixed_epoch0 = list(fixed)
    fixed.set_epoch(9)

    assert fixed_epoch0 == list(fixed)

def test_epoch_subsample_sampler_locality_order_preserves_selected_set():
    random_sampler = EpochSubsampleSampler(
        dataset_length=10,
        effective_num_samples=5,
        seed=321,
        rotate_each_epoch=False,
        shuffle=True,
    )
    locality_keys = [("b", idx) if idx % 2 else ("a", idx) for idx in range(10)]
    locality_sampler = EpochSubsampleSampler(
        dataset_length=10,
        effective_num_samples=5,
        seed=321,
        rotate_each_epoch=False,
        shuffle=True,
        order="locality",
        locality_keys=locality_keys,
    )

    random_epoch = list(random_sampler)
    locality_epoch = list(locality_sampler)

    assert set(locality_epoch) == set(random_epoch)
    assert locality_epoch == sorted(random_epoch, key=lambda index: locality_keys[index])
    assert locality_sampler.metadata()["order"] == "locality"

def test_epoch_subsampling_dataloader_only_affects_train_split():
    dataset = torch.utils.data.TensorDataset(torch.arange(10))
    loader_cfg = {
        "train_batch_size": 2,
        "test_batch_size": 2,
        "num_workers": 0,
        "train_drop_last": False,
    }
    subsampling_cfg = {
        "enabled": True,
        "num_samples": 5,
        "seed": None,
        "rotate_each_epoch": True,
        "shuffle": True,
    }

    train_loader = build_dataloader(
        dataset,
        loader_cfg,
        split="train",
        epoch_subsampling_cfg=subsampling_cfg,
        experiment_seed=7,
    )
    test_loader = build_dataloader(
        dataset,
        loader_cfg,
        split="test",
        epoch_subsampling_cfg=subsampling_cfg,
        experiment_seed=7,
    )
    drop_last_loader = build_dataloader(
        dataset,
        {**loader_cfg, "train_drop_last": True},
        split="train",
        epoch_subsampling_cfg=subsampling_cfg,
        experiment_seed=7,
    )

    assert isinstance(train_loader.sampler, EpochSubsampleSampler)
    assert train_loader.sampler.seed == 7
    assert len(train_loader) == 3
    assert len(drop_last_loader) == 2
    assert not isinstance(test_loader.sampler, EpochSubsampleSampler)
    assert len(test_loader) == 5

def test_dataloader_uses_experiment_seed_for_generator_and_workers():
    dataset = torch.utils.data.TensorDataset(torch.arange(4))
    loader = build_dataloader(
        dataset,
        {"train_batch_size": 2, "num_workers": 0},
        split="train",
        experiment_seed=7,
    )

    repeated = build_dataloader(
        dataset,
        {"train_batch_size": 2, "num_workers": 0},
        split="train",
        experiment_seed=7,
    )
    test_loader = build_dataloader(
        dataset,
        {"test_batch_size": 2, "num_workers": 0},
        split="test",
        experiment_seed=7,
    )

    assert loader.generator.initial_seed() == repeated.generator.initial_seed()
    assert loader.generator.initial_seed() != test_loader.generator.initial_seed()
    assert loader.generator_metadata["algorithm"] == "sha256-v1"
    assert loader.generator_metadata["split"] == "train"
    assert loader.generator_metadata["dataset_fingerprint"]
    assert loader.worker_init_fn is not None


def test_dataloader_random_state_round_trip_restores_next_shuffle():
    dataset = torch.utils.data.TensorDataset(torch.arange(8))
    loader = build_dataloader(
        dataset,
        {"train_batch_size": 2, "num_workers": 0},
        split="train",
        experiment_seed=11,
    )
    state = data_factory.capture_dataloaders_random_state({"train": loader})
    expected = torch.cat([batch[0] for batch in loader]).tolist()

    data_factory.restore_dataloaders_random_state({"train": loader}, state)
    restored = torch.cat([batch[0] for batch in loader]).tolist()

    assert restored == expected
    assert state["train"]["identity"] == loader.generator_metadata
    assert dataloaders_run_metadata({"train": loader})["train"]["dataloader_generator"] == loader.generator_metadata


def test_synthetic_samples_are_index_stable_and_split_separated():
    kwargs = {
        "length": 3,
        "seq_len": 2,
        "num_pred": 1,
        "image_size": (2, 2),
        "radar_size": (2, 2),
        "lidar_size": (2, 2),
        "use_gps": True,
        "use_lidar": True,
        "seed": 17,
    }
    train = SyntheticSequenceDataset(**kwargs, split="train")
    repeated = train[1]
    _ = train[0]
    after_other_index = train[1]
    validation = SyntheticSequenceDataset(**kwargs, split="validation")

    assert repeated.keys() == after_other_index.keys()
    assert all(torch.equal(repeated[key], after_other_index[key]) for key in repeated)
    assert not torch.equal(repeated["image"], validation[1]["image"])


def test_synthetic_index_content_is_independent_of_worker_count():
    dataset = SyntheticSequenceDataset(
        length=4,
        seq_len=2,
        num_pred=1,
        image_size=(2, 2),
        radar_size=(2, 2),
        seed=23,
        split="test",
    )
    single = DataLoader(dataset, batch_size=1, num_workers=0)
    multi = DataLoader(dataset, batch_size=1, num_workers=2)

    single_images = torch.cat([batch["image"] for batch in single])
    multi_images = torch.cat([batch["image"] for batch in multi])

    assert torch.equal(single_images, multi_images)

def test_shutdown_dataloader_workers_clears_persistent_iterator():
    class FakeIterator:
        def __init__(self):
            self.closed = False

        def _shutdown_workers(self):
            self.closed = True

    class FakeLoader:
        def __init__(self):
            self._iterator = FakeIterator()

    loader = FakeLoader()
    iterator = loader._iterator

    shutdown_dataloader_workers(loader)

    assert iterator.closed is True
    assert loader._iterator is None

def test_build_dataset_deepsense_scene_32_records_metadata(tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=1, num_pred=1)
    cfg = {
        "experiment": {"task": "radar"},
        "data": {
            "cache": {"policy": "off"},
            "dataset": {
                "type": "deepsense6g",
                "scene": 32,
                "data_root": str(tmp_path),
                "train_csv_name": csv_path.name,
                "test_csv_name": csv_path.name,
                "seq_len": 1,
                "num_pred": 1,
            },
        },
        "model": {"primary": {}},
    }

    dataset = build_dataset(cfg, "train")
    metadata = dataset_run_metadata(dataset)

    assert isinstance(dataset, DeepSense6GDataset)
    assert metadata["scene_id"] == 32
    assert metadata["scene_slug"] == "scene32"
    assert metadata["csv_name"] == csv_path.name

def test_build_dataloaders_deepsense_train_scenes_concat_records_metadata(tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=1, num_pred=1)
    cfg = {
        "experiment": {"task": "fusion", "seed": 7},
        "data": {
            "cache": {"policy": "off"},
            "dataset": {
                "type": "deepsense6g",
                "scene": 31,
                "train_scenes": [31, 32],
                "test_scenes": [31],
                "data_root": str(tmp_path),
                "train_csv_name": csv_path.name,
                "test_csv_name": csv_path.name,
                "seq_len": 1,
                "num_pred": 1,
                "use_gps": True,
                "gps_normalize": True,
            },
            "dataloader": {"train_batch_size": 2, "test_batch_size": 2, "num_workers": 0},
        },
        "model": {"modalities": ["gps"], "primary": {"modalities": ["gps"]}},
    }

    loaders = build_dataloaders(cfg)
    train_dataset = loaders["train"].dataset
    metadata = dataset_run_metadata(train_dataset)

    assert isinstance(train_dataset, ConcatDataset)
    assert [dataset.scene_id for dataset in train_dataset.datasets] == [31, 32]
    assert metadata["multi_scene"] is True
    assert metadata["scene_slugs"] == ["scene31", "scene32"]
    assert metadata["component_num_samples"] == [len(train_dataset.datasets[0]), len(train_dataset.datasets[1])]
    assert train_dataset.datasets[0].gps_scaler is train_dataset.datasets[1].gps_scaler
    assert metadata["components"][0]["gps_scaler"]["source"] == "multi_scene_train_split_streaming_fit"

def test_build_dataloaders_internal_validation_uses_train_subset_scaler(tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_multirow_gps_sequence_fixture(tmp_path, csv_path, rows=4, seq_len=1, num_pred=1)
    cfg = {
        "experiment": {"task": "fusion", "seed": 7},
        "data": {
            "cache": {"policy": "off"},
            "validation_from_train": {"enabled": True, "fraction": 0.5, "seed": 3},
            "dataset": {
                "type": "deepsense6g",
                "scene": 31,
                "train_scenes": [31, 32],
                "test_scenes": [31],
                "data_root": str(tmp_path),
                "train_csv_name": csv_path.name,
                "test_csv_name": csv_path.name,
                "seq_len": 1,
                "num_pred": 1,
                "use_gps": True,
                "gps_normalize": True,
            },
            "dataloader": {"train_batch_size": 2, "test_batch_size": 2, "num_workers": 0},
        },
        "model": {"modalities": ["gps"], "primary": {"modalities": ["gps"]}},
    }

    loaders = build_dataloaders(cfg)
    train_dataset = loaders["train"].dataset
    validation_dataset = loaders["validation"].dataset
    metadata = dataset_run_metadata(validation_dataset)

    assert isinstance(train_dataset, ConcatDataset)
    assert isinstance(validation_dataset, ConcatDataset)
    assert all(isinstance(dataset, Subset) for dataset in train_dataset.datasets)
    assert all(isinstance(dataset, Subset) for dataset in validation_dataset.datasets)
    assert len(train_dataset) == 4
    assert len(validation_dataset) == 4
    assert metadata["multi_scene"] is True
    assert metadata["components"][0]["internal_validation_split"]["source_split"] == "train"
    assert metadata["components"][0]["selection_split_role"] == "validation"
    first_train_base = train_dataset.datasets[0].dataset
    first_val_base = validation_dataset.datasets[0].dataset
    assert first_train_base.gps_scaler is first_val_base.gps_scaler
    assert first_train_base.gps_scaler_metadata["source"] == "internal_train_subset_streaming_fit"


def test_build_dataloaders_internal_validation_reuses_provided_normalizer(tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_multirow_gps_sequence_fixture(tmp_path, csv_path, rows=4, seq_len=1, num_pred=1)
    cfg = {
        "experiment": {"task": "fusion", "seed": 7},
        "data": {
            "cache": {"policy": "off"},
            "validation_from_train": {"enabled": True, "fraction": 0.5, "seed": 3},
            "dataset": {
                "type": "deepsense6g",
                "scene": 31,
                "data_root": str(tmp_path),
                "train_csv_name": csv_path.name,
                "test_csv_name": csv_path.name,
                "seq_len": 1,
                "num_pred": 1,
                "use_gps": True,
                "gps_normalize": True,
            },
            "dataloader": {"train_batch_size": 2, "test_batch_size": 2, "num_workers": 0},
        },
        "model": {"modalities": ["gps"], "primary": {"modalities": ["gps"]}},
    }
    provided = GPSStandardScaler(
        mean_=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        scale_=np.array([4.0, 5.0, 6.0], dtype=np.float32),
        feature_mode_="relative_polar",
    )

    loaders = build_dataloaders(cfg, normalization_overrides={"gps_scaler": provided})
    train_leaf = loaders["train"].dataset.dataset
    validation_leaf = loaders["validation"].dataset.dataset
    test_leaf = loaders["test"].dataset

    assert train_leaf.gps_scaler is provided
    assert validation_leaf.gps_scaler is provided
    assert test_leaf.gps_scaler is provided


def test_direct_dataloaders_reuse_provided_normalizer(tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_multirow_gps_sequence_fixture(tmp_path, csv_path, rows=4, seq_len=1, num_pred=1)
    cfg = {
        "experiment": {"task": "fusion", "seed": 7},
        "data": {
            "cache": {"policy": "off"},
            "dataset": {
                "type": "deepsense6g",
                "scene": 31,
                "data_root": str(tmp_path),
                "train_csv_name": csv_path.name,
                "val_csv_name": csv_path.name,
                "test_csv_name": csv_path.name,
                "seq_len": 1,
                "num_pred": 1,
                "use_gps": True,
                "gps_normalize": True,
            },
            "dataloader": {"train_batch_size": 2, "test_batch_size": 2, "num_workers": 0},
        },
        "model": {"modalities": ["gps"], "primary": {"modalities": ["gps"]}},
    }
    provided = GPSStandardScaler(
        mean_=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        scale_=np.array([4.0, 5.0, 6.0], dtype=np.float32),
        feature_mode_="relative_polar",
    )

    loaders = build_dataloaders(cfg, normalization_overrides={"gps_scaler": provided})

    assert all(loaders[split].dataset.gps_scaler is provided for split in ("train", "validation", "test"))


def test_domain_dataloaders_reuse_provided_normalizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    marker = object()
    for split in ("train", "val", "test"):
        (tmp_path / f"{split}.csv").write_text("fixture\n", encoding="utf-8")

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, split: str, scaler: object):
            self.split = split
            self.mmwave_scaler = scaler
            self.mmwave_normalize = True
            self.use_mmwave = True
            self.enabled_modalities = ["mmwave"]

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return torch.tensor(index)

    def fake_build_dataset(_cfg, split, **kwargs):
        return FakeDataset(split, kwargs["mmwave_scaler"])

    monkeypatch.setattr(data_factory, "build_dataset", fake_build_dataset)
    domain = {
        "id": "domain-a",
        "condition": "sunny",
        "scene": "scene1",
        "data_root": str(tmp_path),
        "train_csv_name": "train.csv",
        "val_csv_name": "val.csv",
        "test_csv_name": "test.csv",
    }
    cfg = {
        "experiment": {"task": "mmwave", "seed": 7},
        "data": {
            "dataset": {"type": "mmw", "domains": [domain]},
            "dataloader": {"train_batch_size": 2, "test_batch_size": 2, "num_workers": 0},
        },
        "model": {"modalities": ["mmwave"], "primary": {"modalities": ["mmwave"]}},
    }

    loaders = build_dataloaders(cfg, normalization_overrides={"mmwave_scaler": marker})

    assert all(loaders[split].dataset.datasets[0].mmwave_scaler is marker for split in ("train", "validation", "test"))

def test_build_dataloaders_deepsense_2604_stratified_split_uses_union_and_train_scaler(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    _write_stratified_gps_sequence_fixture(tmp_path, train_csv, rows=10, offset=0)
    _write_stratified_gps_sequence_fixture(tmp_path, test_csv, rows=10, offset=10)
    cfg = {
        "experiment": {"task": "fusion", "seed": 7},
        "data": {
            "cache": {"policy": "off"},
            "dataset": {
                "type": "deepsense6g",
                "scene": 31,
                "train_scenes": [31, 32],
                "validation_scenes": [31, 32],
                "test_scenes": [31, 32],
                "data_root": str(tmp_path),
                "train_csv_name": train_csv.name,
                "test_csv_name": test_csv.name,
                "split_protocol": "stratified_80_10_10",
                "split_seed": 11,
                "split_fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
                "seq_len": 5,
                "num_pred": 1,
                "use_gps": True,
                "gps_normalize": True,
            },
            "dataloader": {"train_batch_size": 4, "test_batch_size": 4, "num_workers": 0},
        },
        "model": {"modalities": ["gps"], "primary": {"modalities": ["gps"]}},
    }

    split_datasets = build_protocol_split_datasets(cfg)
    loaders = build_dataloaders(cfg)

    assert split_datasets is not None
    assert set(loaders) == {"train", "test", "validation"}
    assert len(loaders["train"].dataset) == 32
    assert len(loaders["validation"].dataset) == 4
    assert len(loaders["test"].dataset) == 4
    metadata = dataset_run_metadata(loaders["validation"].dataset)
    assert metadata["multi_scene"] is True
    assert metadata["components"][0]["stratified_split"]["source_splits"] == ["train", "test"]
    assert metadata["components"][0]["split_protocol"] == "stratified_80_10_10"
    first_train_leaf = loaders["train"].dataset.datasets[0].dataset.datasets[0]
    first_validation_leaf = loaders["validation"].dataset.datasets[0].dataset.datasets[0]
    first_test_leaf = loaders["test"].dataset.datasets[0].dataset.datasets[0]
    assert first_train_leaf.gps_scaler is first_validation_leaf.gps_scaler
    assert first_train_leaf.gps_scaler is first_test_leaf.gps_scaler
    assert first_train_leaf.gps_scaler_metadata["source"] == "stratified_train_subset_streaming_fit"


def test_protocol_dataloaders_reuse_provided_normalizer(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    _write_stratified_gps_sequence_fixture(tmp_path, train_csv, rows=10, offset=0)
    _write_stratified_gps_sequence_fixture(tmp_path, test_csv, rows=10, offset=10)
    cfg = {
        "experiment": {"task": "fusion", "seed": 7},
        "data": {
            "cache": {"policy": "off"},
            "dataset": {
                "type": "deepsense6g",
                "scene": 31,
                "data_root": str(tmp_path),
                "train_csv_name": train_csv.name,
                "test_csv_name": test_csv.name,
                "split_protocol": "stratified_80_10_10",
                "split_seed": 11,
                "split_fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
                "seq_len": 5,
                "num_pred": 1,
                "use_gps": True,
                "gps_normalize": True,
            },
            "dataloader": {"train_batch_size": 4, "test_batch_size": 4, "num_workers": 0},
        },
        "model": {"modalities": ["gps"], "primary": {"modalities": ["gps"]}},
    }
    provided = GPSStandardScaler(
        mean_=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        scale_=np.array([4.0, 5.0, 6.0], dtype=np.float32),
        feature_mode_="relative_polar",
    )

    loaders = build_dataloaders(cfg, normalization_overrides={"gps_scaler": provided})
    leaves = [
        loaders["train"].dataset.dataset.datasets[0],
        loaders["validation"].dataset.dataset.datasets[0],
        loaders["test"].dataset.dataset.datasets[0],
    ]

    assert all(leaf.gps_scaler is provided for leaf in leaves)

def test_deepsense_2604_sequence_group_split_keeps_seq_index_exclusive(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    _write_stratified_gps_sequence_fixture(tmp_path, train_csv, rows=20, offset=0, seq_block_size=2)
    _write_stratified_gps_sequence_fixture(tmp_path, test_csv, rows=20, offset=20, seq_block_size=2)
    cfg = {
        "experiment": {"task": "fusion", "seed": 7},
        "data": {
            "cache": {"policy": "off"},
            "dataset": {
                "type": "deepsense6g",
                "scene": 31,
                "data_root": str(tmp_path),
                "train_csv_name": train_csv.name,
                "test_csv_name": test_csv.name,
                "split_protocol": "stratified_80_10_10",
                "split_strategy": "stratified_by_target_beam_per_scene_sequence_group",
                "split_seed": 11,
                "split_fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
                "seq_len": 5,
                "num_pred": 1,
                "use_gps": True,
                "gps_normalize": True,
            },
            "dataloader": {"train_batch_size": 4, "test_batch_size": 4, "num_workers": 0},
        },
        "model": {"modalities": ["gps"], "primary": {"modalities": ["gps"]}},
    }

    split_datasets = build_protocol_split_datasets(cfg)

    assert split_datasets is not None
    train_seq = _seq_index_keys_for_dataset(split_datasets["train"])
    validation_seq = _seq_index_keys_for_dataset(split_datasets["validation"])
    test_seq = _seq_index_keys_for_dataset(split_datasets["test"])
    assert train_seq.isdisjoint(validation_seq)
    assert train_seq.isdisjoint(test_seq)
    assert validation_seq.isdisjoint(test_seq)
    assert len(split_datasets["train"]) == 32
    assert len(split_datasets["validation"]) == 4
    assert len(split_datasets["test"]) == 4
