from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.config.io import safe_load_yaml  # noqa: E402
from kd_sensing.cli.preprocess import _apply_scene_override_to_sequence_preprocess  # noqa: E402
import kd_sensing.data.datasets.deepsense6g as deepsense6g_module  # noqa: E402
import kd_sensing.data.datasets.deepsense6g_targets as deepsense6g_targets  # noqa: E402
import kd_sensing.data.transform_ops.io as io_transforms  # noqa: E402
import kd_sensing.data.transform_ops.lidar as lidar_transforms  # noqa: E402
import kd_sensing.preprocessing.lidar as lidar_preprocessing  # noqa: E402
from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset  # noqa: E402
from kd_sensing.data.samples import create_samples  # noqa: E402
from kd_sensing.distillation.distillers import KnowledgeDistillationLoss  # noqa: E402
from kd_sensing.engine.batch import prepare_fusion_inputs, prepare_labels  # noqa: E402
from kd_sensing.engine.cache_policy import apply_cache_policy  # noqa: E402
from kd_sensing.engine.data_factory import (  # noqa: E402
    build_dataset,
    build_dataloader_kwargs,
    shutdown_dataloader_workers,
)
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities  # noqa: E402
from kd_sensing.engine.model_output import adapt_model_output, select_prediction_slots  # noqa: E402
from kd_sensing.engine.runtime import resolve_amp_settings, transfer_non_blocking  # noqa: E402
from kd_sensing.engine.run_metadata import dataset_run_metadata, throughput_run_metadata  # noqa: E402
from kd_sensing.engine.throughput_recommendations import (  # noqa: E402
    lidar_cache_coverage,
    recommend_parallel_training,
)
from kd_sensing.engine.trainer import (  # noqa: E402
    _configure_early_stopping,
    _early_stopping_improved,
    _early_stopping_metric_value,
    _training_outputs_payload,
    _validate_early_stopping_source_available,
    _write_tensorboard_scalars,
    create_eval_run_dir,
    create_run_dir,
    train,
)
from kd_sensing.preprocessing.sequences import generate_sequence_data  # noqa: E402
from kd_sensing.utils.artifact_registry import (  # noqa: E402
    archive_best_checkpoint,
    find_registry_checkpoint,
    resolve_teacher_checkpoint,
)


def _removed_image_option(suffix: str) -> str:
    return "image_" + "motion_" + suffix


def _removed_image_profile() -> str:
    return "motion" + "_mask"


def _removed_encoder_name(prefix: str = "") -> str:
    return prefix + "motion" + "_cnn"


def test_deepsense_scene_defaults_and_aliases():
    default_cfg = load_config(ROOT / "configs/mmwave/teacher_no_kd.yaml")
    scene9_cfg = load_config(ROOT / "configs/mmwave/teacher_no_kd.yaml", ["data.dataset.scene=scene9"])
    scene31_cfg = load_config(ROOT / "configs/mmwave/teacher_no_kd.yaml", ["data.dataset.scene=scenario31"])
    scene32_cfg = load_config(ROOT / "configs/mmwave/teacher_no_kd.yaml", ["data.dataset.scene=scene32"])

    assert default_cfg["data"]["dataset"]["scene_id"] == 31
    assert default_cfg["data"]["dataset"]["scene_slug"] == "scene31"
    assert default_cfg["data"]["dataset"]["data_root"] == "dataset/scenario31"
    assert scene9_cfg["data"]["dataset"]["scene_id"] == 9
    assert scene9_cfg["data"]["dataset"]["scene_slug"] == "scene9"
    assert scene9_cfg["data"]["dataset"]["data_root"] == "dataset/scenario9"
    assert scene31_cfg["data"]["dataset"]["scene_id"] == 31
    assert scene31_cfg["data"]["dataset"]["scene_slug"] == "scene31"
    assert scene31_cfg["data"]["dataset"]["data_root"] == "dataset/scenario31"
    assert scene32_cfg["data"]["dataset"]["scene_id"] == 32
    assert scene32_cfg["data"]["dataset"]["scene_slug"] == "scene32"
    assert scene32_cfg["data"]["dataset"]["data_root"] == "dataset/scenario32"


@pytest.mark.parametrize(("removed_type", "scene"), [("scenario9", 9), ("scenario31", 31), ("scenario32", 32)])
def test_deepsense_scene_specific_dataset_types_are_rejected(removed_type: str, scene: int):
    with pytest.raises(ValueError, match=f"deepsense6g.*scene: {scene}"):
        load_config(
            ROOT / "configs/mmwave/teacher_no_kd.yaml",
            [f"data.dataset.type={removed_type}", "data.dataset.scene=null"],
        )


def test_deepsense_unknown_scene_is_rejected():
    with pytest.raises(ValueError, match="Supported scenes"):
        load_config(ROOT / "configs/mmwave/teacher_no_kd.yaml", ["data.dataset.scene=99"])


def test_sequence_preprocess_scene_override_updates_root_and_csv():
    pre_cfg = {
        "type": "sequence_csv",
        "csv_path": "dataset/scenario32/scenario32_RA.csv",
        "data_root": "dataset/scenario32",
    }
    cfg = {"data": {"dataset": {"type": "deepsense6g", "scene": 9}}}

    _apply_scene_override_to_sequence_preprocess(pre_cfg, cfg)

    assert pre_cfg["data_root"] == "dataset/scenario9"
    assert pre_cfg["csv_path"] == "dataset/scenario9/scenario9_RA.csv"

    cfg["data"]["dataset"]["scene"] = 31
    _apply_scene_override_to_sequence_preprocess(pre_cfg, cfg)

    assert pre_cfg["data_root"] == "dataset/scenario31"
    assert pre_cfg["csv_path"] == "dataset/scenario31/scenario31_RA.csv"


@pytest.mark.parametrize(
    ("enabled", "expected_calls", "expected_fields"),
    [
        (["image"], {"image"}, {"image"}),
        (["radar"], {"radar"}, {"radar_ra", "radar_da"}),
        (["gps"], {"gps"}, {"gps"}),
        (["lidar"], {"lidar"}, {"lidar"}),
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


def test_future_slot_selection_and_missing_features_kd_contract():
    labels = prepare_labels(
        {"target_beam": torch.tensor([[0, 1, 2], [1, 2, 3]])},
        num_pred=2,
        downsample_ratio=1,
        device=torch.device("cpu"),
    )
    logits = torch.randn(2, 5, 4)
    selected = select_prediction_slots(logits, num_pred=2)
    model_output = adapt_model_output({"logits": selected})
    targets = labels.flatten()
    student_logits = selected.reshape(-1, 4)
    teacher_logits = selected.detach().reshape(-1, 4)
    criterion = torch.nn.CrossEntropyLoss()

    assert labels.tolist() == [[0, 1], [1, 2]]
    assert torch.equal(selected, logits[:, -2:, :])
    assert model_output.input_features is None
    assert model_output.output_features is None
    assert KnowledgeDistillationLoss(criterion, kd_mode=0)(
        student_logits,
        teacher_logits,
        targets,
        None,
        None,
        None,
        None,
    )[0].ndim == 0
    assert KnowledgeDistillationLoss(criterion, kd_mode=1)(
        student_logits,
        teacher_logits,
        targets,
        None,
        None,
        None,
        None,
    )[0].ndim == 0
    with pytest.raises(ValueError, match="Relational KD requires real"):
        KnowledgeDistillationLoss(criterion, kd_mode=2)(
            student_logits,
            teacher_logits,
            targets,
            None,
            None,
            None,
            None,
        )


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


def test_cache_policy_resolves_lidar_policy_and_rejects_removed_image_override():
    cfg = {
        "data": {"cache": {"policy": "read_only", "lidar": {"policy": "auto"}}, "dataset": {}},
        "experiment": {"task": "fusion"},
        "model": {"teacher": {"modalities": ["image", "lidar"]}, "student": {"modalities": ["image", "lidar"]}},
    }
    dataset_cfg = {
        "lidar_use_cache": None,
        "lidar_write_cache": None,
    }

    resolved = apply_cache_policy(dataset_cfg, cfg, ("image", "lidar"))

    assert resolved["lidar_use_cache"] is True
    assert resolved["lidar_write_cache"] is True
    assert resolved["lidar_cache_policy"] == "auto"
    with pytest.raises(ValueError, match="Image cache policy has been removed"):
        apply_cache_policy({}, {"data": {"cache": {"policy": "auto", "image": {"policy": "auto"}}}}, ("image",))


def test_parallel_training_recommendation_outputs_background_overrides():
    cfg = load_config(ROOT / "configs/fusion/image_radar_gps_lidar_mmwave_beam_no_kd.yaml")

    result = recommend_parallel_training(
        cfg,
        config_path="configs/fusion/image_radar_gps_lidar_mmwave_beam_no_kd.yaml",
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
    assert result["commands"]["train"].startswith("conda run -n kd_mm_beam python scripts/train.py")


def test_parallel_training_recommendation_warns_when_lidar_cache_is_cold(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    _write_minimal_csv(train_csv, camera=False, radar=False, gps=False, lidar=True)
    _write_minimal_csv(test_csv, camera=False, radar=False, gps=False, lidar=True)
    cfg = load_config(
        ROOT / "configs/lidar/student_no_kd.yaml",
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
        "model": {"teacher": {}, "student": {}},
    }

    dataset = build_dataset(cfg, "train")
    sample = dataset[0]
    metadata = dataset_run_metadata(dataset)

    assert sample["image"].shape == (2, 3, 8, 8)
    assert not hasattr(dataset, _removed_image_option("cache_dir"))
    assert metadata["scene_id"] == 9
    assert metadata["scene_slug"] == "scene9"
    assert _removed_image_option("cache_policy") not in metadata


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
        "model": {"teacher": {}, "student": {}},
    }

    dataset = build_dataset(cfg, "train")
    metadata = dataset_run_metadata(dataset)

    assert isinstance(dataset, DeepSense6GDataset)
    assert metadata["scene_id"] == 32
    assert metadata["scene_slug"] == "scene32"
    assert metadata["csv_name"] == csv_path.name


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
            "model": {"teacher": {"modalities": ["image", "lidar"]}, "student": {"modalities": ["image", "lidar"]}},
            "training": {"transfer": {}, "amp": {}},
        }
    )

    assert metadata["cache"]["policy"] == "read_only"
    assert metadata["cache"]["image"]["input"] == "rgb_imagenet"
    assert metadata["cache"]["lidar"]["policy"] == "auto"
    assert metadata["cache"]["enabled_modalities"] == ["image", "lidar"]
    assert metadata["dataloader_splits"]["train"]["batch_size"] == 3
    assert metadata["dataloader_splits"]["test"]["persistent_workers"] is False
    assert metadata["progress"]["enabled"] is True


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


def test_evaluation_run_dir_defaults_to_unique_directory(tmp_path: Path):
    cfg = {"experiment": {"name": "eval"}, "output": {"dir": str(tmp_path), "run_name": "fixed"}}

    first = create_eval_run_dir(cfg)
    second = create_eval_run_dir(cfg)

    assert first != second
    assert first.exists()
    assert second.exists()
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

    assert grouped.parent == tmp_path / "scene9"
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


def test_train_io_characterization_history_checkpoint_and_final_config(tmp_path: Path):
    cfg = load_config(
        ROOT / "configs/gps/student_no_kd.yaml",
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
    outputs = np.load(run_dir / "training_outputs.npz")

    assert set(history) == {
        "train_loss",
        "train_task_loss",
        "train_objective_loss",
        "train_distill_loss",
        "train_beam_soft_loss",
        "train_unimodal_loss",
        "train_counterfactual_loss",
        "train_prior_regularization_loss",
        "train_reliability_kd_loss",
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
        "train_distill_loss",
        "train_beam_soft_loss",
        "train_unimodal_loss",
        "train_counterfactual_loss",
        "train_prior_regularization_loss",
        "train_reliability_kd_loss",
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
    assert final_cfg["training"]["early_stopping_metric"] == "val_adba"
    assert final_cfg["training"]["early_stopping_mode"] == "max"
    assert final_cfg["runtime"]["early_stopping"]["metric"] == "val_adba"
    assert final_cfg["runtime"]["prediction_objective"]["loss_weights"] == {
        "beam": 1.0,
        "occlusion": 1.0,
        "position": 0.01,
    }
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
    assert (run_dir / "train_log.json").exists()
    assert (run_dir / "training_outputs.npz").exists()
    assert (run_dir / "checkpoints" / "last.pth").exists()


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
        "train_counterfactual_loss": [],
        "train_prior_regularization_loss": [],
        "train_reliability_kd_loss": [],
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
    payload = _training_outputs_payload(
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


def test_multitask_no_kd_training_logs_auxiliary_losses_and_metrics(tmp_path: Path):
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
            "seq_length_teacher": 2,
            "seq_length_student": 2,
            "num_pred": 2,
            "downsample_ratio": 1,
            "teacher": {
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
            },
            "student": {
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
        "distillation": {"type": "no_kd", "teacher_model_name": None},
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


def test_artifact_registry_archives_highest_metric_and_resolves_teacher(tmp_path: Path):
    registry_dir = tmp_path / "registry"
    teacher_cfg = {
        "checkpoint": {"registry": {"enabled": True, "prefer": True, "dir": str(registry_dir)}},
        "experiment": {"name": "gps_teacher_no_kd", "task": "gps"},
        "model": {"teacher": {"type": "gps_teacher"}, "student": {"type": "gps_teacher"}},
        "distillation": {"type": "no_kd"},
        "output": {"run_name": "gps_teacher_no_kd"},
    }
    low = tmp_path / "low.pth"
    high = tmp_path / "high.pth"
    torch.save({"value": torch.tensor([1])}, low)
    torch.save({"value": torch.tensor([2])}, high)

    first = archive_best_checkpoint(
        teacher_cfg,
        source_checkpoint=high,
        val_top1=0.75,
        epoch=2,
        run_dir=tmp_path / "run_high",
    )
    second = archive_best_checkpoint(
        teacher_cfg,
        source_checkpoint=low,
        val_top1=0.25,
        epoch=3,
        run_dir=tmp_path / "run_low",
    )
    found = find_registry_checkpoint(teacher_cfg, target_slug="gps_teacher_no_kd", role="teacher_no_kd")
    kd_cfg = {
        "checkpoint": {"registry": {"enabled": True, "prefer": True, "dir": str(registry_dir)}},
        "paths": {"weights_dir": str(tmp_path / "missing")},
        "experiment": {"name": "gps_logits_kd", "task": "gps"},
        "model": {"teacher": {"type": "gps_teacher"}, "student": {"type": "gps_student"}},
        "distillation": {"type": "logits_kd", "teacher_model_name": "best.pth"},
    }
    resolved = resolve_teacher_checkpoint(kd_cfg, "best.pth")

    assert first["updated"] is True
    assert second["updated"] is False
    assert found.path == resolved.path
    assert resolved.source == "registry"
    assert "acc_0.7500" in resolved.path.name


def test_default_registry_is_scene_scoped(tmp_path: Path):
    scene9_cfg = {
        "checkpoint": {"registry": {"enabled": True, "prefer": True}},
        "data": {"dataset": {"type": "deepsense6g", "scene": 9}},
        "experiment": {"name": "gps_teacher_no_kd", "task": "gps"},
        "model": {"teacher": {"type": "gps_teacher"}, "student": {"type": "gps_teacher"}},
        "distillation": {"type": "no_kd"},
        "output": {"dir": str(tmp_path), "run_name": "gps_teacher_no_kd"},
    }
    scene31_cfg = {
        "checkpoint": {"registry": {"enabled": True, "prefer": True}},
        "data": {"dataset": {"type": "deepsense6g"}},
        "experiment": {"name": "gps_teacher_no_kd", "task": "gps"},
        "model": {"teacher": {"type": "gps_teacher"}, "student": {"type": "gps_teacher"}},
        "distillation": {"type": "no_kd"},
        "output": {"dir": str(tmp_path), "run_name": "gps_teacher_no_kd"},
    }
    scene_32_cfg = {
        "checkpoint": {"registry": {"enabled": True, "prefer": True}},
        "data": {"dataset": {"type": "deepsense6g", "scene": 32}},
        "experiment": {"name": "gps_teacher_no_kd", "task": "gps"},
        "model": {"teacher": {"type": "gps_teacher"}, "student": {"type": "gps_teacher"}},
        "distillation": {"type": "no_kd"},
        "output": {"dir": str(tmp_path), "run_name": "gps_teacher_no_kd"},
    }
    kd_scene31_cfg = {
        **scene31_cfg,
        "paths": {"weights_dir": str(tmp_path / "missing")},
        "experiment": {"name": "gps_logits_kd", "task": "gps"},
        "model": {"teacher": {"type": "gps_teacher"}, "student": {"type": "gps_student"}},
        "distillation": {"type": "logits_kd", "teacher_model_name": "best.pth"},
    }
    kd_scene_32_cfg = {
        **scene_32_cfg,
        "paths": {"weights_dir": str(tmp_path / "missing")},
        "experiment": {"name": "gps_logits_kd", "task": "gps"},
        "model": {"teacher": {"type": "gps_teacher"}, "student": {"type": "gps_student"}},
        "distillation": {"type": "logits_kd", "teacher_model_name": "best.pth"},
    }
    checkpoint = tmp_path / "teacher.pth"
    torch.save({"value": torch.tensor([1])}, checkpoint)

    scene9_archive = archive_best_checkpoint(
        scene9_cfg,
        source_checkpoint=checkpoint,
        val_top1=0.75,
        epoch=1,
        run_dir=tmp_path / "scene9_run",
    )
    missing_scene31 = resolve_teacher_checkpoint(kd_scene31_cfg, "best.pth")
    scene31_archive = archive_best_checkpoint(
        scene31_cfg,
        source_checkpoint=checkpoint,
        val_top1=0.78,
        epoch=1,
        run_dir=tmp_path / "scene31_run",
    )
    resolved_scene31 = resolve_teacher_checkpoint(kd_scene31_cfg, "best.pth")
    missing_scene_32 = resolve_teacher_checkpoint(kd_scene_32_cfg, "best.pth")
    scene_32_archive = archive_best_checkpoint(
        scene_32_cfg,
        source_checkpoint=checkpoint,
        val_top1=0.80,
        epoch=1,
        run_dir=tmp_path / "scene32_run",
    )
    resolved_scene_32 = resolve_teacher_checkpoint(kd_scene_32_cfg, "best.pth")

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
        load_config(ROOT / "configs/image/teacher_no_kd.yaml", [f"data.dataset.{removed_use_key}=true"])
    with pytest.raises(ValueError, match="has been removed"):
        load_config(ROOT / "configs/image/teacher_no_kd.yaml", [f"data.dataset.image_profile={removed_profile}"])
    with pytest.raises(ValueError, match="Removed image encoder"):
        load_config(
            ROOT / "configs/image/resnet18_teacher_no_kd.yaml",
            [f"model.student.encoders.image.type={removed_encoder}"],
        )
    with pytest.raises(ValueError, match="Removed image encoder"):
        load_config(
            ROOT / "configs/image/resnet18_teacher_no_kd.yaml",
            [f"model.student.encoders.image.type={removed_legacy_encoder}"],
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
                "model": {"teacher": {}, "student": {}},
            }
        )
    with pytest.raises(ValueError, match="must match"):
        resolve_enabled_modalities(
            {
                "experiment": {"task": "fusion"},
                "data": {"dataset": {}},
                "model": {"teacher": {"modalities": ["image"]}, "student": {"modalities": ["radar"]}},
            }
        )


def _write_full_sequence_fixture(root: Path, csv_path: Path, *, seq_len: int, num_pred: int) -> None:
    for idx in range(seq_len):
        beam = np.zeros(64, dtype=np.float32)
        beam[idx] = 1.0
        np.savetxt(root / f"beam_{idx}.txt", beam)
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
