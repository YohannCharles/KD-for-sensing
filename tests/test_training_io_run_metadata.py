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
