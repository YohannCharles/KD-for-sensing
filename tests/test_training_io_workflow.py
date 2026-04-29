from __future__ import annotations

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

import kd_sensing.data.transforms as transforms  # noqa: E402
import kd_sensing.data.datasets.scenario9 as scenario9_module  # noqa: E402
from kd_sensing.data.datasets.scenario9 import Scenario9Dataset  # noqa: E402
from kd_sensing.data.samples import create_samples  # noqa: E402
from kd_sensing.engine.batch import prepare_labels  # noqa: E402
from kd_sensing.engine.builders import build_dataloader_kwargs, resolve_enabled_modalities  # noqa: E402
from kd_sensing.engine.trainer import create_eval_run_dir, create_run_dir  # noqa: E402
from kd_sensing.preprocessing.sequences import generate_sequence_data  # noqa: E402
from kd_sensing.utils.artifact_registry import (  # noqa: E402
    archive_best_checkpoint,
    find_registry_checkpoint,
    resolve_teacher_checkpoint,
)


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
def test_scenario9_loads_only_enabled_modalities(monkeypatch, tmp_path: Path, enabled, expected_calls, expected_fields):
    csv_path = tmp_path / "seq.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=3, num_pred=1)
    calls: set[str] = set()

    def fake_image(*args, **kwargs):  # noqa: ARG001
        calls.add("image")
        return torch.zeros(2, 8, 8)

    def fake_radar(*args, **kwargs):  # noqa: ARG001
        calls.add("radar")
        return torch.zeros(3, 4, 4), torch.zeros(3, 6, 4)

    def fake_gps(*args, **kwargs):  # noqa: ARG001
        calls.add("gps")
        return np.zeros((3, 3), dtype=np.float32)

    def fake_lidar(self, idx: int, *, augment: bool):  # noqa: ARG001
        calls.add("lidar")
        return np.zeros((3, 3, 4, 4), dtype=np.float32)

    monkeypatch.setattr(scenario9_module, "load_motion_masks", fake_image)
    monkeypatch.setattr(scenario9_module, "load_radar_maps", fake_radar)
    monkeypatch.setattr(scenario9_module, "load_gps_feature_sequence", fake_gps)
    monkeypatch.setattr(Scenario9Dataset, "_lidar_bev_for_index", fake_lidar)

    dataset = Scenario9Dataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=3,
        num_pred=1,
        enabled_modalities=enabled,
        gps_normalize=False,
        lidar_normalize=False,
    )
    sample = dataset[0]

    assert calls == expected_calls
    assert set(sample) == expected_fields | {"input_beam", "target_beam"}


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
        scenario9_module,
        "load_motion_masks",
        lambda *args, **kwargs: torch.zeros(1, 8, 8),  # noqa: ARG005
    )
    dataset = Scenario9Dataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=2,
        num_pred=1,
        enabled_modalities=["image"],
    )

    sample = dataset[0]
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    labels = prepare_labels(batch, num_pred=1, downsample_ratio=1, device=torch.device("cpu"))

    assert sample["target_beam"].shape == (1,)
    assert batch["target_beam"].shape == (1, 1)
    assert labels.shape == (1, 2)


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


def test_evaluation_run_dir_defaults_to_unique_directory(tmp_path: Path):
    cfg = {"experiment": {"name": "eval"}, "output": {"dir": str(tmp_path), "run_name": "fixed"}}

    first = create_eval_run_dir(cfg)
    second = create_eval_run_dir(cfg)

    assert first != second
    assert first.exists()
    assert second.exists()
    assert first.name.startswith("evaluation_fixed")
    assert second.name.startswith("evaluation_fixed")


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


def test_lidar_cache_hit_miss_write_and_parameter_isolation(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_full_sequence_fixture(tmp_path, csv_path, seq_len=1, num_pred=1)
    build_calls = {"count": 0}

    def fake_build(*args, **kwargs):  # noqa: ARG001
        build_calls["count"] += 1
        return np.ones((3, 4, 4), dtype=np.float32)

    monkeypatch.setattr(transforms, "build_lidar_bev", fake_build)
    dataset = Scenario9Dataset(
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

    monkeypatch.setattr(transforms, "build_lidar_bev", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    cached_dataset = Scenario9Dataset(
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
    different_roi_dataset = Scenario9Dataset(
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

    dataset = Scenario9Dataset(
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
