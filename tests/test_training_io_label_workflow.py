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
