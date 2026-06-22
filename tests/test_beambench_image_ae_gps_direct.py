import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import kd_sensing.baselines.beambench as beambench_package
from kd_sensing.baselines.beambench.image_ae_gps_config import resolve_image_ae_gps_config
from kd_sensing.baselines.beambench.image_ae_gps_datasets import BeamBenchImageAEGPSDataset
from kd_sensing.baselines.beambench.image_ae_gps_models import BeamBenchImageAEGPSDirectModel
from kd_sensing.baselines.beambench.image_ae_gps_training import run_image_ae_gps_training
from kd_sensing.data.transform_ops.gps import (
    PAPER_CALIBRATED_GPS_MODE,
    PAPER_SCENE_CENTER_ANGLES_RAD,
)
from kd_sensing.cli.run_beambench_image_ae_gps_tableiii import run_main as run_tableiii_main


def test_beambench_package_import_is_lightweight():
    assert "__all__" not in vars(beambench_package)
    assert "run_image_ae_gps_training" not in vars(beambench_package)
    assert "BeamBenchImageAEGPSDirectModel" not in vars(beambench_package)


def test_beambench_image_ae_gps_direct_model_forward():
    model = BeamBenchImageAEGPSDirectModel(
        num_beams=64,
        gps_input_size=3,
        ae_latent_dim=16,
        image_size=32,
        hidden_dim=32,
        freeze_ae_encoder=True,
    )

    logits = model(torch.randn(2, 1, 3, 32, 32), torch.randn(2, 1, 3))

    assert tuple(logits.shape) == (2, 64)
    assert torch.all(logits >= 0.0)
    assert torch.all(logits <= 1.0)
    assert all(not param.requires_grad for param in model.camera_ae.parameters())
    assert model.metadata()["fusion_architecture"] == "official_dense_model"

    latent_logits = model.forward_from_latent(torch.randn(2, 16), torch.randn(2, 1, 3))

    assert tuple(latent_logits.shape) == (2, 64)
    assert torch.all(latent_logits >= 0.0)
    assert torch.all(latent_logits <= 1.0)


def test_beambench_image_ae_gps_direct_dry_run_trains(tmp_path: Path):
    data_root = _write_tiny_deepsense_scene(tmp_path / "scenario31")
    output_dir = tmp_path / "outputs" / "beambench_image_ae_gps"

    report = run_image_ae_gps_training(
        {
            "experiment": {"seed": 7, "device": "cpu"},
            "data": {
                "dataset": {
                    "type": "deepsense6g",
                    "scene": 31,
                    "data_root": str(data_root),
                    "train_csv_name": "train_seqs_RA_GPS_LIDAR.csv",
                    "test_csv_name": "test_seqs_RA_GPS_LIDAR.csv",
                    "seq_len": 1,
                    "num_pred": 1,
                    "gps_normalize": True,
                },
                "dataloader": {"train_batch_size": 2, "num_workers": 0},
            },
            "model": {"num_classes": 64, "d_model": 32, "primary": {"image_channels": 3, "gps_input_size": 3}},
            "beambench_paper": {
                "output_dir": str(output_dir),
                "dry_run": True,
                "max_train_samples": 4,
                "max_test_samples": 4,
                "ae_image_size": 32,
                "ae_epochs": 1,
                "ae_batch_size": 2,
                "ae_latent_dim": 16,
                "fusion_epochs": 1,
                "fusion_batch_size": 2,
                "fusion_hidden_dim": 32,
                "fusion_patience": 1,
                "save_predictions": True,
            },
        }
    )

    assert report["workflow"] == "beambench_image_ae_gps_direct_train"
    assert Path(report["checkpoint_path"]).exists()
    assert Path(report["ae_checkpoint_path"]).exists()
    assert Path(report["predictions_path"]).exists()
    assert report["performance"]["feature_cache"]["active"] is True
    assert report["performance"]["amp"]["enabled"] is False
    assert report["selection"]["mode"] == "test_as_validation"
    assert Path(report["performance"]["feature_cache"]["reports"]["train"]["path"]).exists()
    assert Path(report["performance"]["feature_cache"]["reports"]["test"]["path"]).exists()
    assert report["metrics"]["valid_label_count"] == 4
    assert "official_top3_dba" in report["metrics"]


def test_beambench_image_ae_gps_dataset_reads_local_scene(tmp_path: Path):
    data_root = _write_tiny_deepsense_scene(tmp_path / "scenario31")

    dataset = BeamBenchImageAEGPSDataset(
        data_root=data_root,
        csv_name="train_seqs_RA_GPS_LIDAR.csv",
        split="train",
        seq_len=1,
        num_pred=1,
        image_size=32,
        max_samples=2,
    )
    item = dataset[0]

    assert tuple(item["image"].shape) == (1, 3, 32, 32)
    assert tuple(item["gps"].shape) == (1, 2)
    assert int(item["target"]) == 0

    future_dataset = BeamBenchImageAEGPSDataset(
        data_root=data_root,
        csv_name="train_seqs_RA_GPS_LIDAR.csv",
        split="train",
        seq_len=1,
        num_pred=1,
        image_size=32,
        max_samples=2,
        target_beam_source="future",
    )

    assert int(future_dataset[0]["target"]) == 4


def test_beambench_image_ae_gps_dataset_supports_paper_calibrated_gps(tmp_path: Path):
    data_root = _write_tiny_deepsense_scene(tmp_path / "scenario31")

    raw_dataset = BeamBenchImageAEGPSDataset(
        data_root=data_root,
        csv_name="train_seqs_RA_GPS_LIDAR.csv",
        split="train",
        seq_len=1,
        num_pred=1,
        image_size=32,
        max_samples=1,
        gps_feature_mode="relative_polar",
        gps_normalize=False,
    )
    calibrated_dataset = BeamBenchImageAEGPSDataset(
        data_root=data_root,
        csv_name="train_seqs_RA_GPS_LIDAR.csv",
        split="train",
        seq_len=1,
        num_pred=1,
        image_size=32,
        max_samples=1,
        gps_feature_mode="paper_calibrated_relative_polar",
        gps_angle_offset_rad=PAPER_SCENE_CENTER_ANGLES_RAD[31],
        gps_normalize=False,
    )

    raw = raw_dataset[0]["gps"]
    calibrated = calibrated_dataset[0]["gps"]

    assert raw.shape == calibrated.shape == (1, 3)
    assert torch.isclose(raw[0, 0], calibrated[0, 0])
    assert not torch.allclose(raw[0, 1:], calibrated[0, 1:])


def test_beambench_image_ae_gps_dataset_supports_official_distance_angle_gps(tmp_path: Path):
    data_root = _write_tiny_deepsense_scene(tmp_path / "scenario31")

    dataset = BeamBenchImageAEGPSDataset(
        data_root=data_root,
        csv_name="train_seqs_RA_GPS_LIDAR.csv",
        split="train",
        seq_len=1,
        num_pred=1,
        image_size=32,
        max_samples=1,
        gps_feature_mode=PAPER_CALIBRATED_GPS_MODE,
        gps_angle_offset_rad=PAPER_SCENE_CENTER_ANGLES_RAD[31],
        gps_normalize=False,
    )

    item = dataset[0]

    assert tuple(item["gps"].shape) == (1, 2)
    assert abs(float(item["gps"][0, 1])) <= 90.0


def test_beambench_image_ae_gps_config_resolves_throughput_defaults(tmp_path: Path):
    cfg = resolve_image_ae_gps_config(
        {
            "experiment": {"device": "cuda"},
            "data": {
                "dataset": {
                    "scene": 31,
                    "data_root": str(tmp_path),
                    "seq_len": 5,
                    "gps_source_seq_len": 6,
                    "gps_input_seq_len": 2,
                },
                "dataloader": {
                    "num_workers": 6,
                    "pin_memory": True,
                    "persistent_workers": True,
                    "prefetch_factor": 3,
                },
            },
            "training": {
                "amp": {"enabled": True, "dtype": "float16", "grad_scaler": True},
                "transfer": {"non_blocking": True},
                "allow_tf32": True,
                "fused_optimizer": True,
            },
            "beambench_paper": {
                "cache_frozen_ae_features": True,
                "feature_cache_batch_size": 128,
            },
        }
    )

    assert cfg.num_workers == 6
    assert cfg.seq_len == 5
    assert cfg.gps_source_seq_len == 6
    assert cfg.gps_input_seq_len == 2
    assert cfg.pin_memory is True
    assert cfg.persistent_workers is True
    assert cfg.prefetch_factor == 3
    assert cfg.non_blocking_transfer is True
    assert cfg.amp is True
    assert cfg.allow_tf32 is True
    assert cfg.fused_optimizer is True
    assert cfg.cache_frozen_ae_features is True
    assert cfg.feature_cache_batch_size == 128
    assert cfg.gps_feature_mode == PAPER_CALIBRATED_GPS_MODE
    assert cfg.gps_angle_offset_rad == PAPER_SCENE_CENTER_ANGLES_RAD[31]
    assert cfg.gps_input_size == 2
    assert cfg.target_beam_source == "current"
    assert cfg.fusion_architecture == "official_dense_model"
    assert cfg.fusion_loss == "bce"
    assert cfg.fusion_dense_hidden_sizes == (128, 256, 512, 128)
    assert cfg.fusion_activation == "LeakyReLU"
    assert cfg.fusion_last_activation == "Sigmoid"


def test_beambench_image_ae_gps_tableiii_runner_dry_run(tmp_path: Path):
    data_root = _write_tiny_deepsense_scene(tmp_path / "scenario31")
    output_root = tmp_path / "outputs" / "tableiii"

    summary = run_tableiii_main(
        [
            "--config",
            "configs/fusion/beambench_image_ae_gps_direct.yaml",
            "--train-scenes",
            "31",
            "--eval-scenes",
            "31",
            "--output-root",
            str(output_root),
            "--dry-run",
            "--override",
            f"data.dataset.data_root={data_root}",
            "--override",
            "experiment.device=cpu",
            "--override",
            "data.dataloader.num_workers=0",
            "--override",
            "beambench_paper.ae_image_size=32",
            "--override",
            "beambench_paper.ae_latent_dim=16",
            "--override",
            "beambench_paper.fusion_hidden_dim=32",
            "--override",
            "beambench_paper.ae_batch_size=2",
            "--override",
            "beambench_paper.fusion_batch_size=2",
        ]
    )

    assert summary["workflow"] == "beambench_image_ae_gps_direct_paper_split_train"
    assert summary["paper_split"]["train_scenes"] == [31]
    assert summary["paper_split"]["eval_scenes"] == [31]
    assert len(summary["summary"]["rows"]) == 1
    assert summary["summary"]["metric_field"] == "official_top3_dba"
    assert "local_weighted_overall" in summary["summary"]
    assert "gps_calibration" in summary
    assert "31" in summary["gps_calibration"]["train_scenes"]
    assert summary["performance"]["feature_cache"]["active"] is True
    assert "train_scene31" in summary["performance"]["feature_cache"]["reports"]
    assert "test_scene31" in summary["performance"]["feature_cache"]["reports"]
    assert summary["summary"]["rows"][0]["scene"] == 31
    assert (output_root / "tableiii_camera_ae_gps_summary.csv").exists()
    assert (output_root / "tableiii_camera_ae_gps_summary.md").exists()
    assert (output_root / "tableiii_camera_ae_gps_summary.json").exists()
    assert (output_root / "scene31" / "run_report.json").exists()
    checkpoint = torch.load(summary["checkpoint_path"], map_location="cpu", weights_only=False)
    assert checkpoint["paper_split"]["train_scenes"] == [31]
    assert checkpoint["paper_split"]["eval_scenes"] == [31]
    assert checkpoint["ae_checkpoint_path"] == summary["ae_checkpoint_path"]
    assert "gps_calibration" in checkpoint
    assert "feature_cache" in checkpoint["performance"]

    eval_root = tmp_path / "outputs" / "tableiii_eval"
    eval_summary = run_tableiii_main(
        [
            "--config",
            "configs/fusion/beambench_image_ae_gps_direct.yaml",
            "--train-scenes",
            "31",
            "--eval-scenes",
            "31",
            "--output-root",
            str(eval_root),
            "--fusion-checkpoint",
            str(summary["checkpoint_path"]),
            "--dry-run",
            "--override",
            "experiment.device=cpu",
            "--override",
            "data.dataloader.num_workers=0",
        ]
    )

    assert eval_summary["workflow"] == "beambench_image_ae_gps_direct_paper_split_eval"
    assert eval_summary["paper_split"]["eval_scenes"] == [31]
    assert len(eval_summary["summary"]["rows"]) == 1
    assert eval_summary["selection"] == summary["selection"]
    assert eval_summary["performance"]["feature_cache"]["active"] is True
    assert "test_scene31" in eval_summary["performance"]["feature_cache"]["reports"]
    assert (eval_root / "tableiii_camera_ae_gps_summary.csv").exists()
    assert (eval_root / "scene31" / "run_report.json").exists()


def _write_tiny_deepsense_scene(root: Path) -> Path:
    root.mkdir(parents=True)
    for subdir in ("unit1/camera_data", "unit1/mmWave_data", "unit2/GPS_data", "unit1/GPS_data"):
        (root / subdir).mkdir(parents=True)
    for idx in range(1, 9):
        _write_image(root / "unit1" / "camera_data" / f"image_{idx}.jpg", value=idx * 20)
        _write_gps(root / "unit2" / "GPS_data" / f"GPS_location_{idx}.txt", lat=33.0 + idx * 0.0001, lon=-111.0)
        _write_beam(root / "unit1" / "mmWave_data" / f"mmWave_power_{idx}.txt", label=(idx - 1) % 64)
    _write_gps(root / "unit1" / "GPS_data" / "gps_location.txt", lat=33.0, lon=-111.001)

    rows = []
    for row_idx in range(4):
        frame = row_idx + 1
        rows.append(
            {
                "camera1": f"./unit1/camera_data/image_{frame}.jpg",
                "gps1": f"./unit2/GPS_data/GPS_location_{frame}.txt",
                "bs_gps1": "./unit1/GPS_data/gps_location.txt",
                "beam1": f"./unit1/mmWave_data/mmWave_power_{frame}.txt",
                "future_beam1": f"./unit1/mmWave_data/mmWave_power_{frame + 4}.txt",
                "seq_index": row_idx,
            }
        )
    _write_csv(root / "train_seqs_RA_GPS_LIDAR.csv", rows)
    _write_csv(root / "test_seqs_RA_GPS_LIDAR.csv", rows)
    return root


def _write_image(path: Path, *, value: int) -> None:
    array = np.full((12, 12, 3), int(value), dtype=np.uint8)
    Image.fromarray(array).save(path)


def _write_gps(path: Path, *, lat: float, lon: float) -> None:
    path.write_text(f"{lat:.8f} {lon:.8f}\n", encoding="utf-8")


def _write_beam(path: Path, *, label: int) -> None:
    values = np.zeros((64,), dtype=np.float32)
    values[int(label)] = 1.0
    np.savetxt(path, values)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
