import json
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.config.io import dump_config, safe_load_yaml  # noqa: E402
from kd_sensing.cli.preprocess import _apply_scene_override_to_sequence_preprocess  # noqa: E402
from kd_sensing.data.beam_soft_targets import beam_power_to_distribution, gaussian_beam_distribution  # noqa: E402
import kd_sensing.data.datasets.deepsense6g as deepsense6g_module  # noqa: E402
import kd_sensing.data.datasets.deepsense6g_targets as deepsense6g_targets  # noqa: E402
import kd_sensing.data.transform_ops.io as io_transforms  # noqa: E402
import kd_sensing.data.transform_ops.lidar as lidar_transforms  # noqa: E402
import kd_sensing.preprocessing.lidar as lidar_preprocessing  # noqa: E402
from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset  # noqa: E402
from kd_sensing.data.layouts import (  # noqa: E402
    deepsense6g_scene_layout,
    mmw_condition_layout,
    physical_labels_cache_root,
    runtime_cache_root,
)
from kd_sensing.data.samples import create_samples  # noqa: E402
from kd_sensing.data.scenes import retarget_deepsense_dataset_config  # noqa: E402
from kd_sensing.losses import FocalLoss, SoftTargetCrossEntropyLoss  # noqa: E402
from kd_sensing.engine.batch import prepare_fusion_inputs, prepare_labels, prepare_soft_beam_targets  # noqa: E402
from kd_sensing.engine.batch_step import BatchStepRunner  # noqa: E402
from kd_sensing.engine.cache_policy import apply_cache_policy  # noqa: E402
from kd_sensing.engine.data_factory import (  # noqa: E402
    build_dataloader,
    build_dataloaders,
    build_dataset,
    build_dataloader_kwargs,
    build_protocol_split_datasets,
    shutdown_dataloader_workers,
)
from kd_sensing.engine.epoch_subsampling import EpochSubsampleSampler  # noqa: E402
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities  # noqa: E402
from kd_sensing.engine.training_extensions import ExtensionContext, NoOpTrainingExtension  # noqa: E402
from kd_sensing.engine.model_output import adapt_model_output, select_prediction_slots  # noqa: E402
from kd_sensing.engine.runtime import resolve_amp_settings, transfer_non_blocking  # noqa: E402
from kd_sensing.engine.evaluator import _evaluation_split_protocol_report  # noqa: E402
from kd_sensing.engine.run_metadata import dataset_run_metadata, prediction_setup_metadata, throughput_run_metadata  # noqa: E402
from kd_sensing.engine.training_metrics import training_outputs_payload  # noqa: E402
from kd_sensing.engine.throughput_recommendations import (  # noqa: E402
    lidar_cache_coverage,
    recommend_parallel_training,
)
from kd_sensing.engine.trainer import (  # noqa: E402
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
from kd_sensing.preprocessing.sequences import generate_sequence_data  # noqa: E402
from kd_sensing.utils.artifact_registry import (  # noqa: E402
    archive_best_checkpoint,
    find_registry_checkpoint,
    load_checkpoint_metadata,
    resolve_evaluation_checkpoint,
    write_sidecar,
)

_PROFILE_SPEC = importlib.util.spec_from_file_location("profile_training_io", ROOT / "scripts/profile_training_io.py")
profile_training_io = importlib.util.module_from_spec(_PROFILE_SPEC)
assert _PROFILE_SPEC.loader is not None
_PROFILE_SPEC.loader.exec_module(profile_training_io)


class _TinyImageBatchModel(torch.nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.3, 0.0], dtype=torch.float32))
        self.num_classes = num_classes
        self.calls = 0

    def forward(self, image_batch=None, **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        batch_size = image_batch.shape[0]
        horizon = 1
        logits = self.weight.view(1, 1, self.num_classes).expand(batch_size, horizon, self.num_classes)
        features = logits.detach().clone()
        return {"logits": logits, "input_features": features, "output_features": features}


class _DisabledGradScaler:
    def is_enabled(self) -> bool:
        return False


def _removed_image_option(suffix: str) -> str:
    return "image_" + "motion_" + suffix


def _removed_image_profile() -> str:
    return "motion" + "_mask"


def _removed_encoder_name(prefix: str = "") -> str:
    return prefix + "motion" + "_cnn"


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
    assert set(sample) == expected_fields | {"input_beam", "target_beam"}
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
    assert set(sample) == {"input_beam", "target_beam", "gps"}


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


def test_soft_beam_distribution_generation_handles_power_and_circular_fallback():
    distribution = beam_power_to_distribution([-1.0, 0.0, 3.0, 1.0], num_classes=4)
    circular = gaussian_beam_distribution(0, num_classes=4, sigma=1.0, circular=True)
    linear = gaussian_beam_distribution(0, num_classes=4, sigma=1.0, circular=False)

    assert distribution is not None
    assert distribution.tolist() == pytest.approx([0.0, 0.0, 0.75, 0.25])
    assert beam_power_to_distribution([0.0, 0.0, 0.0, 0.0], num_classes=4) is None
    assert circular.sum() == pytest.approx(1.0)
    assert circular[1] == pytest.approx(circular[-1])
    assert circular[-1] > linear[-1]


def test_deepsense_dataset_outputs_soft_beam_distribution(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=2, num_pred=2)
    power = np.zeros(64, dtype=np.float32)
    power[7] = 2.0
    power[8] = 1.0
    np.savetxt(tmp_path / "future_0.txt", power)
    np.savetxt(tmp_path / "future_1.txt", np.zeros(64, dtype=np.float32))

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
        num_pred=2,
        enabled_modalities=["image"],
        image_profile="rgb_imagenet",
        soft_beam_labels={
            "enabled": True,
            "source": "power_or_gaussian",
            "num_classes": 64,
            "sigma": 1.0,
            "circular": True,
        },
    )

    sample = dataset[0]

    assert sample["target_beam"].tolist() == [7, 0]
    assert sample["target_beam_distribution"].shape == (2, 64)
    assert sample["target_beam_distribution_mask"].tolist() == [True, True]
    assert sample["target_beam_distribution"].sum(dim=-1).tolist() == pytest.approx([1.0, 1.0])
    assert sample["target_beam_distribution"][0, 7].item() == pytest.approx(2.0 / 3.0)
    assert sample["target_beam_distribution"][0, 8].item() == pytest.approx(1.0 / 3.0)
    assert sample["target_beam_distribution"][1].argmax().item() == 0
    assert sample["target_beam_distribution"][1, -1].item() == pytest.approx(
        sample["target_beam_distribution"][1, 1].item()
    )


def test_target_domain_soft_beam_labels_ignore_power_and_use_circular_gaussian(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=2, num_pred=1)
    power = np.zeros(64, dtype=np.float32)
    power[7] = 2.0
    power[8] = 1.0
    np.savetxt(tmp_path / "future_0.txt", power)

    monkeypatch.setattr(
        deepsense6g_module,
        "load_rgb_imagenet_frames",
        lambda *args, **kwargs: torch.zeros(2, 3, 8, 8),  # noqa: ARG005
    )
    common_kwargs = {
        "data_root": str(tmp_path),
        "csv_name": str(csv_path),
        "seq_len": 2,
        "num_pred": 1,
        "enabled_modalities": ["image"],
        "image_profile": "rgb_imagenet",
    }
    source_dataset = DeepSense6GDataset(
        **common_kwargs,
        split="train",
        soft_beam_labels={
            "enabled": True,
            "domain": "source",
            "source": "power_or_gaussian",
            "num_classes": 64,
            "sigma": 1.0,
        },
    )
    target_dataset = DeepSense6GDataset(
        **common_kwargs,
        split="target_adapt",
        soft_beam_labels={
            "enabled": True,
            "domain": "target",
            "source": "power_or_gaussian",
            "target_source": "gaussian",
            "num_classes": 64,
            "sigma": 1.0,
        },
    )

    source_dist = source_dataset[0]["target_beam_distribution"][0]
    target_dist = target_dataset[0]["target_beam_distribution"][0]
    gaussian = torch.tensor(gaussian_beam_distribution(7, num_classes=64, sigma=1.0, circular=True))

    assert source_dataset[0]["target_beam"].tolist() == [7]
    assert source_dist[7].item() == pytest.approx(2.0 / 3.0)
    assert source_dist[8].item() == pytest.approx(1.0 / 3.0)
    assert target_dist.tolist() == pytest.approx(gaussian.tolist())
    assert target_dist[6].item() == pytest.approx(target_dist[8].item())
    assert target_dist[0].item() < target_dist[6].item()


def test_prepare_soft_beam_targets_crops_downsamples_and_masks_rows():
    batch = {
        "target_beam_distribution": torch.tensor(
            [[[1.0, 1.0, 2.0, 0.0], [0.0, 5.0, 0.0, 5.0], [9.0, 0.0, 0.0, 0.0]]]
        ),
        "target_beam_distribution_mask": torch.tensor([[True, False, True]]),
    }

    targets = prepare_soft_beam_targets(
        batch,
        num_pred=2,
        num_classes=2,
        downsample_ratio=2,
        device=torch.device("cpu"),
    )

    assert targets is not None
    assert targets.shape == (1, 2, 2)
    assert targets[0, 0].tolist() == pytest.approx([0.5, 0.5])
    assert targets[0, 1].tolist() == pytest.approx([0.0, 0.0])
    assert prepare_soft_beam_targets(
        batch,
        num_pred=2,
        num_classes=2,
        downsample_ratio=2,
        device=torch.device("cpu"),
        enabled=False,
    ) is None


def test_prepare_soft_beam_targets_falls_back_to_hard_labels_for_invalid_soft_rows():
    batch = {
        "target_beam": torch.tensor([[1, 3, -100]]),
        "target_beam_distribution": torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
        ),
        "target_beam_distribution_mask": torch.tensor([[False, True, False]]),
    }

    targets = prepare_soft_beam_targets(
        batch,
        num_pred=3,
        num_classes=4,
        downsample_ratio=1,
        device=torch.device("cpu"),
    )

    assert targets is not None
    assert targets[0, 0].tolist() == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert targets[0, 1].tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert targets[0, 2].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_prepare_soft_beam_targets_ignores_removed_kd_soft_label_alias():
    batch = {
        "target_beam": torch.tensor([[1, 3]]),
        "kd_soft_label": torch.tensor([[[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]]),
        "kd_soft_label_mask": torch.tensor([[True, True]]),
    }

    targets = prepare_soft_beam_targets(
        batch,
        num_pred=2,
        num_classes=4,
        downsample_ratio=1,
        device=torch.device("cpu"),
    )

    assert targets is None


def test_soft_focal_and_supervised_ce_loss_consume_soft_targets():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 3.0]])
    hard_targets = torch.tensor([0, 1])
    soft_targets = torch.tensor([[0.0, 1.0, 0.0], [0.25, 0.25, 0.5]])
    expected_soft_ce = -(soft_targets * torch.nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    focal_loss = FocalLoss(alpha=1.0, gamma=0.0)(logits, soft_targets)
    soft_ce = SoftTargetCrossEntropyLoss()(logits, soft_targets)

    assert focal_loss == pytest.approx(expected_soft_ce.item())
    assert soft_ce == pytest.approx(expected_soft_ce.item())
    assert not torch.isclose(soft_ce, torch.nn.functional.cross_entropy(logits, hard_targets))


def test_supervised_batch_step_uses_beam_soft_target_without_distillation_runtime(tmp_path: Path):
    cfg = {
        "experiment": {"task": "image", "objective": "beam"},
        "model": {
            "num_pred": 1,
            "downsample_ratio": 1,
            "seq_length": 1,
            "num_classes": 4,
            "primary": {"image_profile": "rgb_imagenet"},
        },
        "loss": {"soft_targets": {"enabled": True}},
        "training": {},
    }
    primary = _TinyImageBatchModel(num_classes=4)
    task_criterion = SoftTargetCrossEntropyLoss()
    optimizer = torch.optim.SGD(primary.parameters(), lr=0.1)
    extension_context = ExtensionContext(
        cfg=cfg,
        task="image",
        model_cfg=cfg["model"],
        training_cfg=cfg["training"],
        primary_model=primary,
        task_criterion=task_criterion,
        run_dir=tmp_path,
        device=torch.device("cpu"),
        num_pred=1,
        num_classes=4,
        seq_length=1,
        non_blocking=False,
    )
    extensions = [NoOpTrainingExtension()]
    runner = BatchStepRunner(
        cfg=cfg,
        task="image",
        model_cfg=cfg["model"],
        training_cfg=cfg["training"],
        optimizer=optimizer,
        grad_scaler=_DisabledGradScaler(),
        amp_enabled=False,
        amp_dtype=torch.float32,
        extension_context=extension_context,
        extensions=extensions,
        extension_states=[extension.setup(extension_context) for extension in extensions],
    )
    raw_batch = {
        "image": torch.zeros(2, 1, 3, 8, 8),
        "target_beam": torch.tensor([[0], [2]]),
        "target_beam_distribution": torch.tensor(
            [
                [[0.0, 0.0, 0.0, 1.0]],
                [[0.0, 1.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "target_beam_distribution_mask": torch.tensor([[True], [True]]),
    }

    result = runner.run(raw_batch, epoch=0, step=0, current_alpha=0.0)

    hard_loss = torch.nn.functional.cross_entropy(
        result.primary_logits.reshape(-1, 4),
        result.labels.flatten(),
    )
    assert primary.calls == 1
    assert result.scalar_diagnostics["loss/beam_soft_target"] == pytest.approx(result.task_loss.item())
    assert "loss/distillation" not in result.scalar_diagnostics
    assert result.extra_loss_values["beam_soft"].item() == pytest.approx(result.task_loss.item())
    assert not torch.isclose(result.task_loss.detach(), hard_loss.detach())


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


def _seq_index_keys_for_dataset(dataset) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for leaf, indices in _leaf_indices_for_test(dataset):
        frame = pd.read_csv(leaf.root_csv, na_values="").fillna(-99)
        for idx in indices:
            keys.add((str(getattr(leaf, "scene_id", "")), str(frame.iloc[int(idx)]["seq_index"])))
    return keys


def _leaf_indices_for_test(dataset) -> list[tuple[object, list[int]]]:
    if isinstance(dataset, ConcatDataset):
        result: list[tuple[object, list[int]]] = []
        for component in dataset.datasets:
            result.extend(_leaf_indices_for_test(component))
        return result
    if isinstance(dataset, Subset):
        parent = dataset.dataset
        if isinstance(parent, ConcatDataset):
            grouped: dict[int, list[int]] = {}
            cumulative = list(parent.cumulative_sizes)
            for raw_index in dataset.indices:
                global_index = int(raw_index)
                component_idx = int(np.searchsorted(cumulative, global_index, side="right"))
                previous = cumulative[component_idx - 1] if component_idx > 0 else 0
                grouped.setdefault(component_idx, []).append(global_index - previous)
            result: list[tuple[object, list[int]]] = []
            for component_idx, local_indices in sorted(grouped.items()):
                component = parent.datasets[component_idx]
                if isinstance(component, Subset):
                    base_pairs = _leaf_indices_for_test(component)
                    if len(base_pairs) == 1:
                        base_dataset, base_indices = base_pairs[0]
                        result.append((base_dataset, [base_indices[int(index)] for index in local_indices]))
                    else:
                        result.extend(base_pairs)
                else:
                    result.append((component, [int(index) for index in local_indices]))
            return result
        if isinstance(parent, Subset):
            base_pairs = _leaf_indices_for_test(parent)
            if len(base_pairs) == 1:
                base_dataset, base_indices = base_pairs[0]
                return [(base_dataset, [base_indices[int(index)] for index in dataset.indices])]
        return [(parent, [int(index) for index in dataset.indices])]
    return [(dataset, list(range(len(dataset))))]


def test_dataset_run_metadata_records_balanced_split_sidecar(tmp_path: Path):
    csv_path = tmp_path / "train_seqs_RA_GPS_LIDAR.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=1, num_pred=1)
    sidecar = tmp_path / "split_metadata_RA_GPS_LIDAR.json"
    sidecar.write_text(
        json.dumps(
            {
                "split_protocol": "balanced_seq",
                "split_seed": 42,
                "training_set_pct": 0.8,
                "sequence_counts": {"total": 3, "train": 2, "test": 1},
                "window_counts": {"total": 9, "train": 6, "test": 3},
                "splits": {
                    "train": {
                        "csv_path": str(csv_path),
                        "num_samples": 1,
                        "sequence_count": 2,
                        "seq_index": [1, 2],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=csv_path.name,
        split="train",
        seq_len=1,
        num_pred=1,
        enabled_modalities=["radar"],
    )

    metadata = dataset_run_metadata(dataset)

    assert metadata["split_protocol"] == "balanced_seq"
    assert metadata["split_seed"] == 42
    assert metadata["split_metadata_path"] == str(sidecar)
    assert metadata["split_sequence_count"] == 2
    assert metadata["split_num_samples"] == 1


def test_dataset_run_metadata_records_mmw_split_eligibility_sidecar(tmp_path: Path):
    split_dir = tmp_path / "Prepared" / "Town10_skybridge_seed24" / "splits" / "l5p6_group_safe"
    split_dir.mkdir(parents=True)
    csv_path = split_dir / "train.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=1, num_pred=1)
    sidecar = split_dir / "split_metadata.json"
    sidecar.write_text(
        json.dumps(
            {
                "split_protocol": "mmw_sequence_split_v2",
                "split_protocol_version": "mmw_sequence_split_v2",
                "split_strategy": "group_safe_time_block",
                "split_seed": 42,
                "strict_validation_eligible": True,
                "eligibility_reasons": [],
                "guard_band_frames": 10,
                "block_size_frames": 12,
                "group_key_fields": [
                    "condition",
                    "town",
                    "sensor_scenario",
                    "agent",
                    "contiguous_segment_id",
                    "time_block_id",
                ],
                "train_window_count": 1,
                "test_window_count": 1,
                "train_seq_indices": [1],
                "test_seq_indices": [2],
                "leakage_diagnostics": {
                    "train_test_frame_overlap_count": 0,
                    "guard_band_violations": 0,
                },
                "label_distribution": {"train": {"10": 1}, "test": {"11": 1}},
            }
        ),
        encoding="utf-8",
    )
    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=1,
        num_pred=1,
        enabled_modalities=["radar"],
    )

    metadata = dataset_run_metadata(dataset)
    setup = prediction_setup_metadata(
        {
            "experiment": {"task": "radar"},
            "data": {
                "dataset": {
                    "type": "mmw",
                    "seq_len": 1,
                    "num_pred": 1,
                    "train_csv_name": str(csv_path),
                    "enabled_modalities": ["radar"],
                }
            },
            "model": {"primary": {"modalities": ["radar"]}},
        },
        split_metadata={"train": metadata},
    )
    report = _evaluation_split_protocol_report({"test": metadata})

    assert metadata["split_protocol"] == "mmw_sequence_split_v2"
    assert metadata["split_strategy"] == "group_safe_time_block"
    assert metadata["split_protocol_version"] == "mmw_sequence_split_v2"
    assert metadata["strict_validation_eligible"] is True
    assert metadata["leakage_diagnostics"]["train_test_frame_overlap_count"] == 0
    assert metadata["split_metadata_path"] == str(sidecar)
    assert metadata["split_sequence_count"] == 1
    assert metadata["split_num_samples"] == 1
    assert setup["split_strategy"] == "group_safe_time_block"
    assert setup["strict_validation_eligible"] is True
    assert setup["splits"]["train"]["split_metadata_path"] == str(sidecar)
    assert report["test_csv"] == str(csv_path)
    assert report["split_metadata_path"] == str(sidecar)
    assert report["strict_validation_eligible"] is True
    assert report["warnings"] == []


def test_evaluation_split_protocol_report_warns_for_ineligible_and_missing_metadata(tmp_path: Path):
    missing_csv = tmp_path / "missing_split.csv"
    missing_report = _evaluation_split_protocol_report(
        {
            "test": {
                "csv_path": str(missing_csv),
                "csv_name": missing_csv.name,
                "split_metadata": {"available": False, "expected_path": str(tmp_path / "split_metadata.json")},
            }
        }
    )
    ineligible_report = _evaluation_split_protocol_report(
        {
            "test": {
                "csv_path": str(tmp_path / "ineligible.csv"),
                "csv_name": "ineligible.csv",
                "split_protocol": "mmw_sequence_split_v2",
                "split_strategy": "group_safe_time_block",
                "strict_validation_eligible": False,
                "eligibility_reasons": ["guard_band_violation"],
                "split_metadata_path": str(tmp_path / "split_metadata.json"),
                "split_metadata": {"available": True},
            }
        }
    )

    assert missing_report["split_metadata_available"] is False
    assert missing_report["warnings"][0]["code"] == "split_metadata_missing"
    warning_codes = {item["code"] for item in ineligible_report["warnings"]}
    assert "split_not_strict_validation_eligible" in warning_codes


def test_default_unified_split_missing_sidecar_warns(tmp_path: Path):
    csv_path = tmp_path / "train_seqs_RA_GPS_LIDAR.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=1, num_pred=1)
    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=csv_path.name,
        split="train",
        seq_len=1,
        num_pred=1,
        enabled_modalities=["radar"],
    )

    with pytest.warns(UserWarning, match="balanced_seq split metadata sidecar is missing"):
        metadata = dataset_run_metadata(dataset)

    assert metadata["split_metadata"]["available"] is False
    assert "warning" in metadata["split_metadata"]


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


def test_fixed_run_name_defaults_to_unique_directory_and_resume_reuses(tmp_path: Path):
    cfg = {"experiment": {"name": "exp"}, "output": {"dir": str(tmp_path), "run_name": "fixed"}, "training": {}}

    first = create_run_dir(cfg)
    (first / "final_config.yaml").write_text("old", encoding="utf-8")
    second = create_run_dir(cfg)
    resume = create_run_dir({**cfg, "training": {"resume": True}})
    overwrite = create_run_dir({**cfg, "output": {"dir": str(tmp_path), "run_name": "fixed", "overwrite": True}})

    assert first == tmp_path / "fixed"
    assert second != first
    assert (first / "final_config.yaml").read_text(encoding="utf-8") == "old"
    assert resume == first
    assert overwrite == first


def test_scene_grouped_run_dir_defaults_and_resume_reuses(tmp_path: Path):
    cfg = {
        "experiment": {"name": "exp"},
        "data": {"dataset": {"type": "deepsense6g"}},
        "output": {"dir": str(tmp_path), "run_name": "fixed"},
        "training": {},
    }
    scene32_cfg = {
        **cfg,
        "data": {"dataset": {"type": "deepsense6g", "scene": 32}},
        "output": {"dir": str(tmp_path), "run_name": "scene32_fixed"},
    }

    first = create_run_dir(cfg)
    (first / "final_config.yaml").write_text("old", encoding="utf-8")
    second = create_run_dir(cfg)
    resume = create_run_dir({**cfg, "training": {"resume": True}})
    overwrite = create_run_dir({**cfg, "output": {"dir": str(tmp_path), "run_name": "fixed", "overwrite": True}})
    scene32 = create_run_dir(scene32_cfg)

    assert first == tmp_path / "scene31" / "fixed"
    assert second != first
    assert resume == first
    assert overwrite == first
    assert scene32 == tmp_path / "scene32" / "scene32_fixed"


def test_multiscene_run_dir_defaults_to_scenegroup_and_explicit_output_can_opt_out(tmp_path: Path):
    cfg = {
        "experiment": {"name": "exp"},
        "data": {"dataset": {"type": "deepsense6g", "train_scenes": [32, 33, 34]}},
        "output": {"dir": str(tmp_path), "run_name": "fixed"},
        "training": {},
    }
    explicit = {
        **cfg,
        "output": {"dir": str(tmp_path / "manual_root"), "run_name": "fixed", "group_by_scene": False},
    }

    grouped = create_run_dir(cfg)
    manual = create_run_dir(explicit)
    resume = create_run_dir({**cfg, "training": {"resume": True}})

    assert grouped == tmp_path / "scenegroup_s32_s34" / "fixed"
    assert manual == tmp_path / "manual_root" / "fixed"
    assert resume == grouped


def test_evaluation_run_dir_defaults_to_unique_directory(tmp_path: Path):
    cfg = {"experiment": {"name": "eval"}, "output": {"dir": str(tmp_path), "run_name": "fixed"}}

    first = create_eval_run_dir(cfg)
    second = create_eval_run_dir(cfg)

    assert first != second
    assert first.exists()
    assert second.exists()
    assert first.parent == tmp_path / "evaluations" / "eval"
    assert second.parent == tmp_path / "evaluations" / "eval"
    assert first.name.startswith("evaluation_fixed")
    assert second.name.startswith("evaluation_fixed")


def test_scene_grouped_eval_dir_and_explicit_eval_output(tmp_path: Path):
    cfg = {
        "experiment": {"name": "eval"},
        "data": {"dataset": {"type": "deepsense6g", "scene": 9}},
        "output": {"dir": str(tmp_path), "run_name": "fixed"},
    }

    grouped = create_eval_run_dir(cfg)
    explicit = create_eval_run_dir(cfg, output_dir=str(tmp_path / "manual_eval"))

    assert grouped.parent == tmp_path / "evaluations" / "eval"
    assert grouped.name.startswith("evaluation_fixed")
    assert explicit == tmp_path / "manual_eval"


@pytest.mark.parametrize(
    ("raw_metric", "expected_metric", "expected_mode"),
    [
        ("val_adba", "val_adba", "max"),
        ("dba", "val_adba", "max"),
        ("dba/val_adba", "val_adba", "max"),
        ("beam/val_adba", "val_adba", "max"),
        ("top1_val_acc", "val_acc", "max"),
        ("accuracy/val", "val_acc", "max"),
        ("beam/accuracy_val", "val_acc", "max"),
        ("beam/val_top1", "val_acc", "max"),
        ("val/acc_top1", "val_acc", "max"),
        ("val_loss", "val_loss", "min"),
    ],
)
def test_early_stopping_metric_aliases_and_default_direction(raw_metric: str, expected_metric: str, expected_mode: str):
    training_cfg = {"early_stopping_metric": raw_metric}

    metric, mode = _configure_early_stopping(training_cfg)

    assert metric == expected_metric
    assert mode == expected_mode
    assert training_cfg["early_stopping_metric"] == expected_metric
    assert training_cfg["early_stopping_mode"] == expected_mode


def test_early_stopping_improvement_direction_and_missing_dba_error():
    assert _early_stopping_improved(0.91, 0.90, mode="max", min_delta=0.001) is True
    assert _early_stopping_improved(0.9005, 0.90, mode="max", min_delta=0.001) is False
    assert _early_stopping_improved(0.49, 0.50, mode="min", min_delta=0.001) is True
    assert _early_stopping_improved(0.4995, 0.50, mode="min", min_delta=0.001) is False
    assert _early_stopping_metric_value({"val_acc": 0.25}, "val_acc") == 0.25

    with pytest.raises(ValueError, match="val_adba.*not available"):
        _early_stopping_metric_value({"val_acc": 0.25}, "val_adba")
    with pytest.raises(ValueError, match="DBA/ADBA"):
        _validate_early_stopping_source_available({"loss": 1.0, "topk": {}}, "val_adba")
    with pytest.raises(ValueError, match="val_position_rmse.*not available.*Available metrics"):
        _validate_early_stopping_source_available(
            {"loss": 1.0, "topk": {"1": [0.25]}, "dba": [0.7], "objective": {"name": "beam"}},
            "val_position_rmse",
        )


@pytest.mark.parametrize(("total_epochs", "expected_min_epoch"), [(0, 0), (1, 1), (2, 1), (5, 3), (100, 50)])
def test_early_stopping_min_epoch_uses_half_target_epochs(total_epochs: int, expected_min_epoch: int):
    assert _early_stopping_min_epoch(total_epochs) == expected_min_epoch


def test_train_early_stopping_waits_until_half_target_epochs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = load_config(
        ROOT / "configs/gps/lightweight.yaml",
        [
            "experiment.device=cpu",
            "data.dataset.type=synthetic",
            "data.dataset.length=2",
            "data.dataset.seed=19",
            "data.validation_from_train.enabled=true",
            "data.validation_from_train.fraction=0.5",
            "data.validation_from_train.seed=19",
            "data.dataloader.train_batch_size=1",
            "data.dataloader.test_batch_size=1",
            "data.dataloader.num_workers=0",
            "training.epochs=5",
            "training.patience=1",
            "training.use_early_stopping=true",
            "training.early_stopping_metric=val_loss",
            "training.early_stopping_mode=min",
            "scheduler.type=none",
            "output.run_name=early_stop_min_epoch",
            "output.progress.enabled=false",
            "output.tensorboard.enabled=false",
            f"output.dir={tmp_path}",
            "output.overwrite=true",
            "checkpoint.registry.enabled=false",
        ],
    )

    def constant_validation_metrics(*_args, **_kwargs) -> dict:
        return {
            "loss": 1.0,
            "topk": {"1": [0.0], "3": [0.0], "5": [0.0]},
            "total": [1],
            "dba": [0.0],
            "available_metrics": ["val_loss", "val_acc", "val_adba"],
            "objective": {"name": "beam"},
        }

    monkeypatch.setattr("kd_sensing.engine.trainer.validate", constant_validation_metrics)

    result = train(cfg)

    assert [epoch_log["epoch"] for epoch_log in result["epoch_logs"]] == [1, 2, 3]
    assert result["epoch_logs"][-1]["epochs_without_improvement"] >= cfg["training"]["patience"]
    assert result["final_test_metrics"]["evaluation_split"] == "test"
    assert result["final_test_metrics"]["model_selection_split"] == "validation"
    assert result["split_metadata"]["validation"]["selection_split_source"] == "train"


def test_training_validation_interval_skips_intermediate_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = load_config(
        ROOT / "configs/gps/lightweight.yaml",
        [
            "experiment.device=cpu",
            "data.dataset.type=synthetic",
            "data.dataset.length=2",
            "data.dataloader.train_batch_size=1",
            "data.dataloader.test_batch_size=1",
            "data.dataloader.num_workers=0",
            "training.epochs=5",
            "training.use_early_stopping=false",
            "training.validation.interval_epochs=3",
            "scheduler.type=none",
            "output.run_name=validation_interval",
            "output.progress.enabled=false",
            "output.tensorboard.enabled=false",
            f"output.dir={tmp_path}",
            "output.overwrite=true",
            "checkpoint.registry.enabled=false",
        ],
    )
    calls = 0

    def constant_validation_metrics(*_args, **_kwargs) -> dict:
        nonlocal calls
        calls += 1
        return {
            "loss": float(calls),
            "topk": {"1": [0.0], "3": [0.0], "5": [0.0]},
            "total": [1],
            "dba": [0.0],
            "available_metrics": ["val_loss", "val_acc", "val_adba"],
            "objective": {"name": "beam"},
        }

    monkeypatch.setattr("kd_sensing.engine.trainer.validate", constant_validation_metrics)

    result = train(cfg)

    assert calls == 3
    assert [log["validation_ran"] for log in result["epoch_logs"]] == [True, False, True, False, True]
    assert [log["val_loss"] for log in result["epoch_logs"]] == [1.0, 1.0, 2.0, 2.0, 3.0]


def test_train_io_characterization_history_checkpoint_and_final_config(tmp_path: Path):
    cfg = load_config(
        ROOT / "configs/gps/lightweight.yaml",
        [
            "experiment.device=cpu",
            "data.dataset.type=synthetic",
            "data.dataset.length=2",
            "data.dataset.seed=13",
            "data.dataloader.train_batch_size=1",
            "data.dataloader.test_batch_size=1",
            "data.dataloader.num_workers=0",
            "training.epochs=1",
            "scheduler.type=none",
            "output.run_name=trainer_characterization",
            "output.progress.enabled=false",
            "output.tensorboard.enabled=false",
            f"output.dir={tmp_path}",
            "output.overwrite=true",
            "checkpoint.registry.enabled=false",
        ],
    )

    result = train(cfg)
    run_dir = Path(result["run_dir"])
    history = result["history"]
    epoch_log = result["epoch_logs"][0]
    checkpoint = torch.load(run_dir / "checkpoints" / "last.pth", map_location="cpu")
    final_cfg = safe_load_yaml((run_dir / "final_config.yaml").read_text(encoding="utf-8"))
    train_log = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))
    outputs = np.load(run_dir / "training_outputs.npz")

    assert set(history) == {
        "train_loss",
        "train_task_loss",
        "train_objective_loss",
        "train_beam_soft_loss",
        "train_unimodal_loss",
        "train_occlusion_loss",
        "train_position_loss",
        "train_multitask_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_atop3",
        "val_atop5",
        "val_adba",
        "val_occlusion_accuracy",
        "val_occlusion_blocked_f1",
        "val_position_rmse",
        "val_position_mae",
        "val_multitask_loss",
        "val_primary_metric",
        "learning_rates",
    }
    assert {
        "epoch",
        "total_epochs",
        "train_batches",
        "train_loss",
        "train_task_loss",
        "train_objective_loss",
        "train_beam_soft_loss",
        "train_unimodal_loss",
        "train_occlusion_loss",
        "train_position_loss",
        "train_multitask_loss",
        "loss/occlusion",
        "loss/position",
        "loss/multitask_total",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_atop3",
        "val_atop5",
        "val_adba",
        "val_occlusion_accuracy",
        "val_occlusion_blocked_f1",
        "val_position_rmse",
        "val_position_mae",
        "val_multitask_loss",
        "val_primary_metric",
        "early_stopping_metric",
        "early_stopping_mode",
        "early_stopping_value",
        "early_stopping_improved",
        "best_early_stopping_value",
        "best_early_stopping_epoch",
        "epochs_without_improvement",
        "learning_rate",
        "optimizer/lr/main",
        "optimizer/params/main",
    } <= set(epoch_log)
    assert "train_distill_loss" not in history
    assert "train_distill_loss" not in epoch_log
    assert {
        "epoch",
        "state_dict",
        "optimizer",
        "scheduler",
        "test_loss",
        "best_val_loss",
        "best_val_top1",
        "best_top1_epoch",
        "early_stopping_metric",
        "early_stopping_mode",
        "best_early_stopping_value",
        "best_early_stopping_epoch",
        "epochs_without_improvement",
        "normalization_artifacts",
        "checkpoint_registry",
    } <= set(checkpoint)
    assert checkpoint["early_stopping_metric"] == "val_adba"
    assert checkpoint["early_stopping_mode"] == "max"
    assert checkpoint["best_early_stopping_epoch"] == 1
    assert result["early_stopping"]["metric"] == "val_adba"
    assert result["early_stopping"]["mode"] == "max"
    assert final_cfg["runtime"]["run_dir"] == str(run_dir)
    assert final_cfg["runtime"]["output_overwrite"] is True
    assert result["final_test_metrics"]["evaluation_split"] == "test"
    assert result["final_test_metrics"]["model_selection_split"] == "test"
    assert final_cfg["runtime"]["final_test_metrics"]["evaluation_split"] == "test"
    assert train_log["final_test_metrics"]["evaluation_split"] == "test"
    assert final_cfg["training"]["early_stopping_metric"] == "val_adba"
    assert final_cfg["training"]["early_stopping_mode"] == "max"
    assert final_cfg["runtime"]["early_stopping"]["metric"] == "val_adba"
    assert final_cfg["runtime"]["prediction_objective"]["loss_weights"] == {
        "beam": 1.0,
        "occlusion": 1.0,
        "position": 0.01,
    }
    objective_meta = final_cfg["runtime"]["prediction_objective"]
    assert {
        "name",
        "primary_loss",
        "primary_metric",
        "primary_metric_mode",
        "available_metrics",
        "metric_aliases",
        "metric_modes",
        "history_fields",
        "tensorboard_scalars",
        "enabled_targets",
        "enabled_heads",
    } <= set(objective_meta)
    assert objective_meta["name"] == "beam"
    assert objective_meta["primary_metric"] == "val_adba"
    assert "adba" in objective_meta["metric_aliases"]
    assert "val_primary_metric" in objective_meta["history_fields"]
    assert {"tag": "objective/val_primary_metric", "history_key": "val_primary_metric"} in objective_meta[
        "tensorboard_scalars"
    ]
    assert train_log["prediction_objective"] == objective_meta
    assert epoch_log["val_occlusion_accuracy"] is None
    assert epoch_log["val_occlusion_blocked_f1"] is None
    assert epoch_log["val_position_rmse"] is None
    assert epoch_log["val_position_mae"] is None
    assert epoch_log["val_multitask_loss"] is None
    assert "val_occlusion_blocked_f1" not in epoch_log["validation_metrics"]
    assert "val_position_rmse" not in epoch_log["validation_metrics"]
    assert np.isnan(outputs["val_occlusion_blocked_f1"][0])
    assert np.isnan(outputs["val_position_rmse"][0])
    assert np.isnan(outputs["val_multitask_loss"][0])
    assert "splits" in final_cfg["runtime"]
    assert "normalization_artifacts" in final_cfg["runtime"]
    assert "throughput" in final_cfg["runtime"]
    assert final_cfg["runtime"]["throughput"]["progress"]["enabled"] is False
    assert (run_dir / "training_outputs.npz").exists()
    assert (run_dir / "checkpoints" / "last.pth").exists()


def test_train_epoch_subsampling_smoke_logs_metadata(tmp_path: Path):
    cfg = load_config(
        ROOT / "configs/gps/lightweight.yaml",
        [
            "experiment.device=cpu",
            "data.dataset.type=synthetic",
            "data.dataset.length=4",
            "data.dataset.seed=23",
            "data.dataloader.train_batch_size=1",
            "data.dataloader.test_batch_size=1",
            "data.dataloader.num_workers=0",
            "training.epochs=1",
            "training.epoch_subsampling.enabled=true",
            "training.epoch_subsampling.num_samples=2",
            "training.epoch_subsampling.rotate_each_epoch=true",
            "scheduler.type=none",
            "output.run_name=trainer_epoch_subsampling",
            "output.progress.enabled=false",
            "output.tensorboard.enabled=false",
            f"output.dir={tmp_path}",
            "output.overwrite=true",
            "checkpoint.registry.enabled=false",
        ],
    )

    result = train(cfg)
    run_dir = Path(result["run_dir"])
    epoch_log = result["epoch_logs"][0]
    final_cfg = safe_load_yaml((run_dir / "final_config.yaml").read_text(encoding="utf-8"))
    train_log = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))

    assert epoch_log["train_batches"] == 2
    assert epoch_log["train_epoch_subsampling_enabled"] is True
    assert epoch_log["train_full_samples"] == 4
    assert epoch_log["train_effective_samples"] == 2
    assert epoch_log["train_sampler_epoch"] == 0
    assert epoch_log["train_epoch_subsampling"]["seed"] == cfg["experiment"]["seed"]
    assert result["split_metadata"]["train"]["num_samples"] == 4
    assert result["split_metadata"]["test"]["num_samples"] == 4
    assert result["split_metadata"]["train"]["epoch_subsampling"]["effective_train_samples"] == 2
    assert final_cfg["runtime"]["splits"]["train"]["epoch_subsampling"]["num_samples"] == 2
    assert final_cfg["runtime"]["throughput"]["epoch_subsampling"]["train"]["effective_train_samples"] == 2
    assert train_log["epoch_logs"][0]["train_effective_samples"] == 2
    assert (run_dir / "training_outputs.npz").exists()
    assert (run_dir / "checkpoints" / "last.pth").exists()


def test_csi_debug_training_writes_resolved_diff_startup_and_health_artifacts(tmp_path: Path):
    cfg = load_config(
        ROOT / "configs/csi/hardening_matrix/debug/A0_clone_generated.yaml",
        [
            "experiment.device=cpu",
            "data.dataset.type=synthetic",
            "data.dataset.length=2",
            "data.dataset.seed=31",
            "data.dataset.csi_shape=[8,4]",
            "data.dataloader.train_batch_size=1",
            "data.dataloader.test_batch_size=1",
            "data.dataloader.num_workers=0",
            "training.epochs=1",
            "training.use_early_stopping=false",
            "scheduler.type=none",
            "output.run_name=csi_debug_smoke",
            "output.progress.enabled=false",
            "output.tensorboard.enabled=false",
            f"output.dir={tmp_path}",
            "output.overwrite=true",
            "checkpoint.registry.enabled=false",
        ],
    )
    reference_path = tmp_path / "a0_reference.yaml"
    reference_cfg = deepcopy(cfg)
    reference_cfg["output"]["run_name"] = "different_identity"
    reference_cfg["experiment"]["seed"] = 999
    dump_config(reference_cfg, reference_path)
    cfg["debug"]["config_diff"]["reference"] = str(reference_path)

    result = train(cfg)
    run_dir = Path(result["run_dir"])
    train_log = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))
    config_diff = json.loads((run_dir / "config_diff.json").read_text(encoding="utf-8"))
    startup = json.loads((run_dir / "startup_summary.json").read_text(encoding="utf-8"))
    csi_records = json.loads((run_dir / "csi_first_batch_diagnostics.json").read_text(encoding="utf-8"))
    epoch_log = train_log["epoch_logs"][0]

    assert (run_dir / "resolved_config.yaml").exists()
    assert config_diff["parity_passed"] is True
    assert any(item["path"] == "experiment.seed" for item in config_diff["allowed_identity_differences"])
    assert startup["model"]["csi_encoder_type"] == "pilot_dual_view_csi"
    assert startup["model"]["use_internal_gru"] is True
    assert startup["parameters"]["modules"]["csi_encoder"]["trainable_params"] > 0
    assert train_log["startup_summary"]["parameters"]["total_params"] == startup["parameters"]["total_params"]
    assert train_log["pilot_noise_validity"]["valid"] is True
    assert train_log["pilot_noise_validity"]["reason"] == "pilot_noise_disabled"
    assert {record["source"] for record in csi_records} == {"train", "val"}
    assert csi_records[0]["complex"]["before_hardening"]["shape"] == [1, 10, 8, 4]
    assert epoch_log["grad_norm_csi_encoder"] > 0.0
    assert epoch_log["grad_norm_representation_core"] > 0.0
    assert epoch_log["grad_norm_beam_head"] > 0.0
    assert epoch_log["param_delta_csi_encoder"] > 0.0
    assert epoch_log["param_delta_representation_core"] > 0.0
    assert epoch_log["param_delta_beam_head"] > 0.0


class _FakeTensorboardWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        pass


def _tensorboard_history() -> dict:
    return {
        "train_loss": [1.0],
        "train_task_loss": [0.9],
        "train_objective_loss": [0.9],
        "train_distill_loss": [0.0],
        "val_loss": [0.8],
        "val_primary_metric": [0.8],
        "train_acc": [0.25],
        "val_acc": [0.2],
        "val_atop3": [0.3],
        "val_atop5": [0.4],
        "val_adba": [0.5],
        "learning_rates": [0.001],
        "train_beam_soft_loss": [],
        "train_unimodal_loss": [],
        "train_multitask_loss": [None],
        "val_multitask_loss": [None],
        "train_occlusion_loss": [0.7],
        "train_position_loss": [None],
        "val_occlusion_accuracy": [0.6],
        "val_occlusion_blocked_f1": [float("nan")],
        "val_position_rmse": [None],
        "val_position_mae": [None],
    }


def test_optional_objective_tensorboard_scalars_skip_inactive_values():
    writer = _FakeTensorboardWriter()
    history = _tensorboard_history()

    _write_tensorboard_scalars(writer, history, 1)

    tags = {tag for tag, _, _ in writer.scalars}
    assert "beam/accuracy_train" in tags
    assert "beam/accuracy_val" in tags
    assert "beam/val_atop3" in tags
    assert "beam/val_atop5" in tags
    assert "beam/val_adba" in tags
    assert "accuracy/train" not in tags
    assert "accuracy/val" not in tags
    assert "accuracy/val_atop3" not in tags
    assert "accuracy/val_atop5" not in tags
    assert "dba/val_adba" not in tags
    assert "occlusion/accuracy" in tags
    assert "occlusion/blocked_f1" not in tags
    assert "position/rmse" not in tags
    assert "position/mae" not in tags
    assert "loss/val_multitask_total" not in tags


def test_tensorboard_startup_scalars_make_new_event_non_empty():
    writer = _FakeTensorboardWriter()

    _write_tensorboard_startup_scalars(
        writer,
        {
            "data": {"batch_size": {"train": 16, "test": 16}},
            "optimization": {"max_epochs": 20},
            "parameters": {
                "total_params": 123,
                "trainable_params": 120,
                "modules": {"image_encoder": {"total_params": 10, "trainable_params": 10}},
            },
        },
    )

    tags = {tag for tag, _, _ in writer.scalars}
    assert "run/start" in tags
    assert "model/total_params" in tags
    assert "model/modules/image_encoder/total_params" in tags


@pytest.mark.parametrize("objective", ["occlusion", "position"])
def test_non_beam_objectives_do_not_write_beam_tensorboard_scalars(objective: str):
    writer = _FakeTensorboardWriter()
    history = _tensorboard_history()

    _write_tensorboard_scalars(writer, history, 1, objective=objective)

    tags = {tag for tag, _, _ in writer.scalars}
    assert "beam/accuracy_train" not in tags
    assert "beam/accuracy_val" not in tags
    assert "beam/val_atop3" not in tags
    assert "beam/val_atop5" not in tags
    assert "beam/val_adba" not in tags
    assert "accuracy/val" not in tags
    assert "dba/val_adba" not in tags


def test_current_selection_objectives_tensorboard_write_only_formal_tags():
    base = {
        "train_loss": [1.0],
        "train_task_loss": [0.9],
        "train_objective_loss": [0.9],
        "train_distill_loss": [0.0],
        "val_loss": [0.8],
        "val_primary_metric": [0.7],
        "learning_rates": [0.001],
        "train_beam_soft_loss": [],
        "train_unimodal_loss": [],
        "train_acc": [0.25],
        "val_beam_top1": [0.4],
        "val_beam_top3": [0.6],
        "val_beam_top5": [0.8],
        "val_beam_dba": [0.5],
        "train_los_loss": [0.3],
        "val_los_accuracy": [0.9],
        "val_los_f1": [0.85],
        "val_los_auc": [0.95],
        "train_link_quality_loss": [0.2],
        "val_link_mae": [1.0],
        "val_link_rmse": [1.2],
        "val_link_r2": [0.1],
    }

    cases = {
        "current_beam_selection": {"beam/val_top1", "beam/val_top3", "beam/val_top5", "beam/val_dba_current"},
        "current_los_classification": {"loss/los", "los/accuracy", "los/f1", "los/auc"},
        "current_link_quality": {"loss/link_quality", "link/mae", "link/rmse", "link/r2"},
    }
    forbidden = {
        "beam/val_top1",
        "beam/val_dba_current",
        "los/accuracy",
        "link/mae",
        "beam/val_adba",
        "dba/val_adba",
    }
    for objective, expected in cases.items():
        writer = _FakeTensorboardWriter()
        _write_tensorboard_scalars(writer, base, 1, objective=objective)
        tags = {tag for tag, _, _ in writer.scalars}
        assert expected <= tags
        assert not ((forbidden - expected) & tags)


def test_tensorboard_legacy_accuracy_tags_restore_historical_scalars():
    writer = _FakeTensorboardWriter()
    history = _tensorboard_history()

    _write_tensorboard_scalars(
        writer,
        history,
        1,
        objective="occlusion",
        tensorboard_cfg={"legacy_accuracy_tags": True},
    )

    tags = {tag for tag, _, _ in writer.scalars}
    assert "beam/accuracy_val" not in tags
    assert "accuracy/train" in tags
    assert "accuracy/val" in tags
    assert "accuracy/val_atop3" in tags
    assert "accuracy/val_atop5" in tags
    assert "dba/val_adba" in tags


def test_training_outputs_payload_converts_inactive_optional_metrics_to_nan():
    payload = training_outputs_payload(
        {
            "train_loss": [1.0],
            "train_occlusion_loss": [None],
            "val_position_rmse": [None],
            "val_occlusion_blocked_f1": [0.5],
        },
        {
            "name": "beam",
            "primary_loss": "beam",
            "enabled_targets": ["beam"],
            "enabled_heads": ["beam"],
            "loss_weights": {"beam": 1.0, "occlusion": 1.0, "position": 0.01},
        },
        "val_adba",
        "max",
    )

    assert np.isnan(payload["train_occlusion_loss"][0])
    assert np.isnan(payload["val_position_rmse"][0])
    assert payload["val_occlusion_blocked_f1"][0] == pytest.approx(0.5)
    assert payload["loss_weights"].tolist() == [1.0, 1.0, 0.01]


def test_multitask_supervised_training_logs_auxiliary_losses_and_metrics(tmp_path: Path):
    train_csv = tmp_path / "train_aux.csv"
    test_csv = tmp_path / "test_aux.csv"
    _write_aux_training_csv(tmp_path, train_csv, prefix="train", future_max=[1.0, 5.0])
    _write_aux_training_csv(tmp_path, test_csv, prefix="test", future_max=[0.5, 8.0])
    cfg = {
        "experiment": {"name": "aux_smoke", "task": "fusion", "seed": 3, "device": "cpu"},
        "data": {
            "dataset": {
                "type": "deepsense6g",
                "scene": 31,
                "data_root": str(tmp_path),
                "train_csv_name": train_csv.name,
                "test_csv_name": test_csv.name,
                "seq_len": 2,
                "num_pred": 2,
                "portion": 1.0,
                "use_gps": True,
                "gps_normalize": False,
                "occlusion_target": {"enabled": True, "threshold_percentile": 50.0},
                "position_target": {"enabled": True, "source": "future_gps_local_xy", "normalize": True},
            },
            "dataloader": {
                "train_batch_size": 1,
                "test_batch_size": 1,
                "num_workers": 0,
                "pin_memory": False,
            },
        },
        "model": {
            "modalities": ["gps"],
            "feature_size": 8,
            "num_classes": 64,
            "seq_length": 2,
            "num_pred": 2,
            "downsample_ratio": 1,
            "primary": {
                "type": "cls_token_transformer_fusion",
                "modalities": ["gps"],
                "feature_size": 8,
                "d_model": 8,
                "num_classes": 64,
                "num_pred": 2,
                "num_heads": 2,
                "num_layers": 1,
                "max_seq_len": 4,
                "gps_input_size": 3,
                "auxiliary_heads": {"enabled": True, "occlusion": True, "position": True},
            },
        },
        "loss": {
            "type": "cross_entropy",
            "auxiliary": {
                "enabled": True,
                "occlusion": {"enabled": True, "weight": 1.0, "pos_weight": "auto"},
                "position": {"enabled": True, "weight": 0.01},
            },
        },
        "training": {
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "grad_clip": None,
            "patience": 2,
            "use_early_stopping": False,
            "early_stopping_metric": "val_adba",
            "early_stopping_mode": "max",
            "min_delta": 0.0,
            "transfer": {"non_blocking": False},
            "amp": {"enabled": False, "dtype": "float16", "grad_scaler": True},
        },
        "scheduler": {"type": "none"},
        "evaluation": {"k_values": [1, 3, 5], "dba_delta": 5},
        "output": {
            "dir": str(tmp_path),
            "run_name": "aux_smoke",
            "overwrite": True,
            "progress": {"enabled": False},
            "tensorboard": {"enabled": False},
        },
        "checkpoint": {"registry": {"enabled": False}, "strict_load": True},
    }

    result = train(cfg)
    run_dir = Path(result["run_dir"])
    train_log = json.loads((run_dir / "train_log.json").read_text(encoding="utf-8"))
    final_cfg = safe_load_yaml((run_dir / "final_config.yaml").read_text(encoding="utf-8"))
    outputs = np.load(run_dir / "training_outputs.npz")

    assert train_log["train_multitask_loss"][0] >= 0.0
    assert "auxiliary" in train_log["epoch_logs"][0]["validation_metrics"]
    assert "train_occlusion_loss" in outputs
    assert "occlusion_target_stats" in final_cfg["runtime"]["normalization_artifacts"]
    assert "position_target_scaler" in final_cfg["runtime"]["normalization_artifacts"]


def test_artifact_registry_archives_highest_metric_and_resolves_evaluation_checkpoint(tmp_path: Path):
    registry_dir = tmp_path / "registry"
    cfg = {
        "checkpoint": {"registry": {"enabled": True, "prefer": True, "dir": str(registry_dir)}},
        "experiment": {"name": "gps_strong", "task": "gps"},
        "model": {"capacity": "strong", "primary": {"type": "modular_sequence", "modalities": ["gps"]}},
        "output": {"run_name": "gps_strong"},
    }
    low = tmp_path / "low.pth"
    high = tmp_path / "high.pth"
    torch.save({"value": torch.tensor([1])}, low)
    torch.save({"value": torch.tensor([2])}, high)

    first = archive_best_checkpoint(
        cfg,
        source_checkpoint=high,
        val_top1=0.75,
        epoch=2,
        run_dir=tmp_path / "run_high",
    )
    second = archive_best_checkpoint(
        cfg,
        source_checkpoint=low,
        val_top1=0.25,
        epoch=3,
        run_dir=tmp_path / "run_low",
    )
    found = find_registry_checkpoint(cfg, target_slug="gps_strong", role="strong")
    resolved = resolve_evaluation_checkpoint(cfg)

    assert first["updated"] is True
    assert second["updated"] is False
    assert found.path == resolved.path
    assert resolved.source == "registry"
    assert "acc_0.7500" in resolved.path.name


def test_artifact_registry_tolerates_malformed_sidecar_during_parallel_archival(tmp_path: Path):
    registry_dir = tmp_path / "registry"
    cfg = {
        "checkpoint": {"registry": {"enabled": True, "prefer": True, "dir": str(registry_dir)}},
        "experiment": {"name": "gps_strong", "task": "gps"},
        "model": {"capacity": "strong", "primary": {"type": "modular_sequence", "modalities": ["gps"]}},
        "output": {"run_name": "gps_strong"},
    }
    metadata = {
        "config_slug": "gps_strong",
        "artifact_role": "strong",
        "metric_value": 0.90,
        "path": str(registry_dir / "gps_strong_strong_acc_0.9000.pth"),
    }
    embedded = Path(metadata["path"])
    embedded.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"checkpoint_registry": metadata}, embedded)
    embedded.with_suffix(embedded.suffix + ".json").write_text("", encoding="utf-8")

    loaded = load_checkpoint_metadata(embedded)
    found = find_registry_checkpoint(cfg, target_slug="gps_strong", role="strong")

    assert loaded["metric_value"] == 0.90
    assert found.path == embedded

    candidate = tmp_path / "candidate.pth"
    torch.save({"value": torch.tensor([3])}, candidate)
    archived = archive_best_checkpoint(
        cfg,
        source_checkpoint=candidate,
        val_top1=0.95,
        epoch=4,
        run_dir=tmp_path / "run",
    )

    assert archived["updated"] is True
    assert load_checkpoint_metadata(archived["path"])["metric_value"] == 0.95


def test_write_sidecar_uses_complete_json_file(tmp_path: Path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"weights")

    sidecar = write_sidecar(checkpoint, {"metric_value": 0.5, "path": checkpoint})

    assert json.loads(sidecar.read_text(encoding="utf-8"))["metric_value"] == 0.5
    assert list(tmp_path.glob("*.tmp")) == []


def test_default_registry_is_scene_scoped(tmp_path: Path):
    scene9_cfg = {
        "checkpoint": {"registry": {"enabled": True, "prefer": True}},
        "data": {"dataset": {"type": "deepsense6g", "scene": 9}},
        "experiment": {"name": "gps_strong", "task": "gps"},
        "model": {"capacity": "strong", "primary": {"type": "modular_sequence", "modalities": ["gps"]}},
        "output": {"dir": str(tmp_path), "run_name": "gps_strong"},
    }
    scene31_cfg = {
        "checkpoint": {"registry": {"enabled": True, "prefer": True}},
        "data": {"dataset": {"type": "deepsense6g"}},
        "experiment": {"name": "gps_strong", "task": "gps"},
        "model": {"capacity": "strong", "primary": {"type": "modular_sequence", "modalities": ["gps"]}},
        "output": {"dir": str(tmp_path), "run_name": "gps_strong"},
    }
    scene_32_cfg = {
        "checkpoint": {"registry": {"enabled": True, "prefer": True}},
        "data": {"dataset": {"type": "deepsense6g", "scene": 32}},
        "experiment": {"name": "gps_strong", "task": "gps"},
        "model": {"capacity": "strong", "primary": {"type": "modular_sequence", "modalities": ["gps"]}},
        "output": {"dir": str(tmp_path), "run_name": "gps_strong"},
    }
    checkpoint = tmp_path / "primary.pth"
    torch.save({"value": torch.tensor([1])}, checkpoint)

    scene9_archive = archive_best_checkpoint(
        scene9_cfg,
        source_checkpoint=checkpoint,
        val_top1=0.75,
        epoch=1,
        run_dir=tmp_path / "scene9_run",
    )
    missing_scene31 = find_registry_checkpoint(scene31_cfg)
    scene31_archive = archive_best_checkpoint(
        scene31_cfg,
        source_checkpoint=checkpoint,
        val_top1=0.78,
        epoch=1,
        run_dir=tmp_path / "scene31_run",
    )
    resolved_scene31 = find_registry_checkpoint(scene31_cfg)
    missing_scene_32 = find_registry_checkpoint(scene_32_cfg)
    scene_32_archive = archive_best_checkpoint(
        scene_32_cfg,
        source_checkpoint=checkpoint,
        val_top1=0.80,
        epoch=1,
        run_dir=tmp_path / "scene32_run",
    )
    resolved_scene_32 = find_registry_checkpoint(scene_32_cfg)

    assert Path(scene9_archive["path"]).parent == tmp_path / "scene9" / "best_checkpoints"
    assert scene9_archive["scene_slug"] == "scene9"
    assert missing_scene31.source == "missing"
    assert missing_scene31.registry_dir == tmp_path / "scene31" / "best_checkpoints"
    assert Path(scene31_archive["path"]).parent == tmp_path / "scene31" / "best_checkpoints"
    assert scene31_archive["scene_slug"] == "scene31"
    assert resolved_scene31.path == Path(scene31_archive["path"])
    assert resolved_scene31.metadata["scene_slug"] == "scene31"
    assert missing_scene_32.source == "missing"
    assert missing_scene_32.registry_dir == tmp_path / "scene32" / "best_checkpoints"
    assert Path(scene_32_archive["path"]).parent == tmp_path / "scene32" / "best_checkpoints"
    assert scene_32_archive["scene_slug"] == "scene32"
    assert resolved_scene_32.path == Path(scene_32_archive["path"])
    assert resolved_scene_32.metadata["scene_slug"] == "scene32"


def test_default_registry_is_scenegroup_scoped_for_multiscene_configs(tmp_path: Path):
    single_cfg = {
        "checkpoint": {"registry": {"enabled": True, "prefer": True}},
        "data": {"dataset": {"type": "deepsense6g"}},
        "experiment": {"name": "gps_strong", "task": "gps"},
        "model": {"capacity": "strong", "primary": {"type": "modular_sequence", "modalities": ["gps"]}},
        "output": {"dir": str(tmp_path), "run_name": "gps_strong"},
    }
    group_cfg = {
        **single_cfg,
        "data": {"dataset": {"type": "deepsense6g", "train_scenes": [32, 33, 34], "test_scenes": [31, 32, 33, 34]}},
    }
    checkpoint = tmp_path / "primary.pth"
    torch.save({"value": torch.tensor([1])}, checkpoint)

    group_archive = archive_best_checkpoint(
        group_cfg,
        source_checkpoint=checkpoint,
        val_top1=0.82,
        epoch=1,
        run_dir=tmp_path / "scenegroup_run",
    )
    missing_single = find_registry_checkpoint(single_cfg)
    resolved_group = find_registry_checkpoint(group_cfg)

    assert Path(group_archive["path"]).parent == tmp_path / "scenegroup_s32_s34" / "best_checkpoints"
    assert group_archive["scene_scope"] == "scenegroup"
    assert group_archive["scene_slug"] == "scenegroup_s32_s34"
    assert group_archive["train_scenes"] == [32, 33, 34]
    assert group_archive["test_scenes"] == [31, 32, 33, 34]
    assert missing_single.source == "missing"
    assert missing_single.registry_dir == tmp_path / "scene31" / "best_checkpoints"
    assert resolved_group.path == Path(group_archive["path"])
    assert resolved_group.metadata["scene_slug"] == "scenegroup_s32_s34"


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


def test_non_blocking_transfer_and_amp_cpu_fallback():
    cfg = {
        "training": {
            "transfer": {"non_blocking": True},
            "amp": {"enabled": True, "dtype": "float16", "grad_scaler": True},
        }
    }
    batch = {"gps": torch.randn(2, 8, 3)}

    enabled, dtype = resolve_amp_settings(cfg, torch.device("cpu"))
    fusion_inputs = prepare_fusion_inputs(
        batch,
        seq_length=8,
        num_pred=3,
        device=torch.device("cpu"),
        modalities=["gps"],
        non_blocking=transfer_non_blocking(cfg),
    )

    assert enabled is False
    assert dtype is torch.float16
    assert fusion_inputs["gps_batch"].shape == (2, 10, 3)


def test_builder_rejects_conflicting_dataset_flags():
    with pytest.raises(ValueError, match="use_lidar=true conflicts"):
        resolve_enabled_modalities(
            {
                "experiment": {"task": "gps"},
                "data": {"dataset": {"use_lidar": True}},
                "model": {"primary": {}},
            }
        )
    with pytest.raises(ValueError, match="must match"):
        resolve_enabled_modalities(
            {
                "experiment": {"task": "fusion"},
                "data": {"dataset": {}},
                "model": {"modalities": ["image"], "primary": {"modalities": ["radar"]}},
            }
        )


def _write_full_sequence_fixture(root: Path, csv_path: Path, *, seq_len: int, num_pred: int) -> None:
    for idx in range(seq_len):
        beam = np.zeros(64, dtype=np.float32)
        beam[idx] = 1.0
        np.savetxt(root / f"beam_{idx}.txt", beam)
        np.savetxt(root / f"gps_{idx}.txt", np.asarray([42.0 + idx * 1e-5, -71.0], dtype=np.float32))
        np.savetxt(root / f"bs_gps_{idx}.txt", np.asarray([42.0, -71.0], dtype=np.float32))
    for idx in range(num_pred):
        future = np.zeros(64, dtype=np.float32)
        future[idx + 10] = 1.0
        np.savetxt(root / f"future_{idx}.txt", future)
    columns = (
        [f"camera{i}" for i in range(1, seq_len + 1)]
        + [f"radar{i}" for i in range(1, seq_len + 1)]
        + [f"gps{i}" for i in range(1, seq_len + 1)]
        + [f"bs_gps{i}" for i in range(1, seq_len + 1)]
        + [f"lidar{i}" for i in range(1, seq_len + 1)]
        + [f"beam{i}" for i in range(1, seq_len + 1)]
        + [f"future_beam{i}" for i in range(1, num_pred + 1)]
        + ["seq_index"]
    )
    values = (
        [f"camera_{idx}.jpg" for idx in range(seq_len)]
        + [f"radar_{idx}_RA.npy" for idx in range(seq_len)]
        + [f"gps_{idx}.txt" for idx in range(seq_len)]
        + [f"bs_gps_{idx}.txt" for idx in range(seq_len)]
        + [f"lidar_{idx}.txt" for idx in range(seq_len)]
        + [f"beam_{idx}.txt" for idx in range(seq_len)]
        + [f"future_{idx}.txt" for idx in range(num_pred)]
        + ["1"]
    )
    csv_path.write_text(",".join(columns) + "\n" + ",".join(values) + "\n", encoding="utf-8")


def _write_multirow_gps_sequence_fixture(root: Path, csv_path: Path, *, rows: int, seq_len: int, num_pred: int) -> None:
    columns = (
        [f"gps{i}" for i in range(1, seq_len + 1)]
        + [f"bs_gps{i}" for i in range(1, seq_len + 1)]
        + [f"beam{i}" for i in range(1, seq_len + 1)]
        + [f"future_beam{i}" for i in range(1, num_pred + 1)]
        + ["seq_index"]
    )
    lines = [",".join(columns)]
    for row_idx in range(rows):
        gps_paths = []
        bs_paths = []
        beam_paths = []
        future_paths = []
        for idx in range(seq_len):
            gps_name = f"row{row_idx}_gps_{idx}.txt"
            bs_name = f"row{row_idx}_bs_{idx}.txt"
            beam_name = f"row{row_idx}_beam_{idx}.txt"
            np.savetxt(root / gps_name, np.asarray([42.0 + row_idx * 1e-4 + idx * 1e-5, -71.0], dtype=np.float32))
            np.savetxt(root / bs_name, np.asarray([42.0, -71.0], dtype=np.float32))
            beam = np.zeros(64, dtype=np.float32)
            beam[(row_idx + idx) % 64] = 1.0
            np.savetxt(root / beam_name, beam)
            gps_paths.append(gps_name)
            bs_paths.append(bs_name)
            beam_paths.append(beam_name)
        for idx in range(num_pred):
            future_name = f"row{row_idx}_future_{idx}.txt"
            future = np.zeros(64, dtype=np.float32)
            future[(row_idx + idx + 10) % 64] = 1.0
            np.savetxt(root / future_name, future)
            future_paths.append(future_name)
        values = gps_paths + bs_paths + beam_paths + future_paths + [str(row_idx)]
        lines.append(",".join(values))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stratified_gps_sequence_fixture(
    root: Path,
    csv_path: Path,
    *,
    rows: int,
    offset: int,
    seq_block_size: int = 1,
) -> None:
    seq_len = 5
    columns = (
        [f"gps{i}" for i in range(1, seq_len + 1)]
        + [f"bs_gps{i}" for i in range(1, seq_len + 1)]
        + [f"beam{i}" for i in range(1, seq_len + 1)]
        + ["future_beam1", "seq_index"]
    )
    lines = [",".join(columns)]
    for row_idx in range(rows):
        global_idx = int(offset + row_idx)
        label = 10 if row_idx % 2 == 0 else 20
        gps_paths = []
        bs_paths = []
        beam_paths = []
        for frame_idx in range(seq_len):
            gps_name = f"s{offset}_row{row_idx}_gps_{frame_idx}.txt"
            bs_name = f"s{offset}_row{row_idx}_bs_{frame_idx}.txt"
            beam_name = f"s{offset}_row{row_idx}_beam_{frame_idx}.txt"
            np.savetxt(root / gps_name, np.asarray([42.0 + global_idx * 1e-4 + frame_idx * 1e-5, -71.0]))
            np.savetxt(root / bs_name, np.asarray([42.0, -71.0]))
            beam = np.zeros(64, dtype=np.float32)
            beam[(label + frame_idx) % 64] = 1.0
            np.savetxt(root / beam_name, beam)
            gps_paths.append(gps_name)
            bs_paths.append(bs_name)
            beam_paths.append(beam_name)
        future_name = f"s{offset}_row{row_idx}_future.txt"
        future = np.zeros(64, dtype=np.float32)
        future[label] = 1.0
        np.savetxt(root / future_name, future)
        seq_index = int(offset + (row_idx // max(int(seq_block_size), 1)))
        lines.append(",".join(gps_paths + bs_paths + beam_paths + [future_name, str(seq_index)]))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_aux_training_csv(root: Path, csv_path: Path, *, prefix: str, future_max: list[float]) -> None:
    seq_len = 2
    num_pred = len(future_max)
    gps_paths = []
    bs_paths = []
    beam_paths = []
    future_paths = []
    future_gps_paths = []
    future_bs_paths = []
    for idx in range(seq_len):
        gps_name = f"{prefix}_gps_{idx}.txt"
        bs_name = f"{prefix}_bs_{idx}.txt"
        beam_name = f"{prefix}_beam_{idx}.txt"
        np.savetxt(root / gps_name, np.asarray([42.0 + idx * 1e-5, -71.0], dtype=np.float32))
        np.savetxt(root / bs_name, np.asarray([42.0, -71.0], dtype=np.float32))
        beam = np.zeros(64, dtype=np.float32)
        beam[idx] = 1.0
        np.savetxt(root / beam_name, beam)
        gps_paths.append(gps_name)
        bs_paths.append(bs_name)
        beam_paths.append(beam_name)
    for idx, max_power in enumerate(future_max):
        future_name = f"{prefix}_future_{idx}.txt"
        gps_name = f"{prefix}_future_gps_{idx}.txt"
        bs_name = f"{prefix}_future_bs_{idx}.txt"
        future = np.linspace(0.1, float(max_power), 64, dtype=np.float32)
        np.savetxt(root / future_name, future)
        np.savetxt(root / gps_name, np.asarray([42.0001 + idx * 1e-5, -71.0], dtype=np.float32))
        np.savetxt(root / bs_name, np.asarray([42.0, -71.0], dtype=np.float32))
        future_paths.append(future_name)
        future_gps_paths.append(gps_name)
        future_bs_paths.append(bs_name)
    columns = (
        [f"gps{i}" for i in range(1, seq_len + 1)]
        + [f"bs_gps{i}" for i in range(1, seq_len + 1)]
        + [f"beam{i}" for i in range(1, seq_len + 1)]
        + [f"future_beam{i}" for i in range(1, num_pred + 1)]
        + [f"future_gps{i}" for i in range(1, num_pred + 1)]
        + [f"future_bs_gps{i}" for i in range(1, num_pred + 1)]
        + ["seq_index"]
    )
    values = gps_paths + bs_paths + beam_paths + future_paths + future_gps_paths + future_bs_paths + ["1"]
    csv_path.write_text(",".join(columns) + "\n" + ",".join(values) + "\n", encoding="utf-8")


def _write_camera_files(root: Path, *, count: int) -> None:
    from PIL import Image

    for idx in range(count):
        Image.fromarray(np.full((8, 8, 3), idx * 40, dtype=np.uint8)).save(root / f"camera_{idx}.jpg")


def _write_minimal_csv(path: Path, *, camera: bool, radar: bool, gps: bool, lidar: bool) -> None:
    columns: list[str] = []
    values: list[str] = []
    if camera:
        columns.append("camera1")
        values.append("camera.jpg")
    if radar:
        columns.append("radar1")
        values.append("radar_RA.npy")
    if gps:
        columns.extend(["gps1", "bs_gps1"])
        values.extend(["gps.txt", "bs_gps.txt"])
    if lidar:
        columns.append("lidar1")
        values.append("lidar.txt")
    columns.extend(["beam1", "future_beam1", "seq_index"])
    values.extend(["beam.txt", "future.txt", "1"])
    path.write_text(",".join(columns) + "\n" + ",".join(values) + "\n", encoding="utf-8")
