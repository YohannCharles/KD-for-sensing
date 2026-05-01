from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import kd_sensing.data.datasets.scenario9 as scenario9_module  # noqa: E402
from kd_sensing.data.datasets.scenario9 import DeepSense6GDataset  # noqa: E402
from kd_sensing.diagnostics.modality_visualization import (  # noqa: E402
    SampleCandidate,
    modality_statistics,
    select_sample_candidates,
    tensor_stats,
    visualize_modalities,
)


def test_visual_diagnostic_sampling_is_seeded_and_handles_shortage():
    candidates = [
        SampleCandidate(dataset_index=0, csv_row_index=10, seq_index=1, future_label=2),
        SampleCandidate(dataset_index=1, csv_row_index=11, seq_index=1, future_label=2),
        SampleCandidate(dataset_index=2, csv_row_index=12, seq_index=1, future_label=3),
        SampleCandidate(dataset_index=3, csv_row_index=13, seq_index=2, future_label=2),
    ]

    first, first_summary = select_sample_candidates(
        candidates,
        sample_count=1,
        seed=123,
        seq_index=(1,),
        labels=(2,),
    )
    repeat, repeat_summary = select_sample_candidates(
        candidates,
        sample_count=1,
        seed=123,
        seq_index=("1",),
        labels=(2,),
    )
    shortage, shortage_summary = select_sample_candidates(
        candidates,
        sample_count=10,
        seed=999,
        seq_index=(1,),
        labels=(2,),
    )

    assert [item.dataset_index for item in first] == [item.dataset_index for item in repeat]
    assert first_summary["selected_dataset_indices"] == repeat_summary["selected_dataset_indices"]
    assert [item.dataset_index for item in shortage] == [0, 1]
    assert shortage_summary["requested_count"] == 10
    assert shortage_summary["actual_count"] == 2


def test_modality_statistics_include_stable_tensor_fields():
    sample = {
        "image": torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),
        "radar_ra": torch.ones(2, 3, 4),
        "radar_da": torch.zeros(2, 5, 4),
        "lidar": torch.zeros(2, 3, 4, 4),
        "gps": torch.tensor([[1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]),
        "mmwave": torch.ones(2, 64),
    }
    sample["lidar"][0, 1, 0, 0] = 3.0

    stats = modality_statistics(sample)
    image_stats = tensor_stats(sample["image"])

    assert image_stats["shape"] == [1, 2, 2]
    assert image_stats["dtype"] == "float32"
    assert image_stats["nonzero_fraction"] == 0.25
    assert stats["image"]["mask_density"] == 0.25
    assert stats["radar"]["radar_ra"]["shape"] == [2, 3, 4]
    assert stats["radar"]["radar_da"]["max"] == 0.0
    assert stats["lidar"]["channel_nonzero_fraction"][1] > 0.0
    assert stats["gps"]["per_dimension_min"] == [1.0, 0.0, 0.0]
    assert len(stats["mmwave"]["per_time_mean"]) == 2


def test_visualize_modalities_smoke_writes_png_summary_and_samples(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    _write_multimodal_csv(tmp_path, train_csv, rows=2, seq_len=3)
    cfg = _diagnostic_cfg(
        tmp_path,
        train_csv_name="train.csv",
        modalities=["image", "radar", "gps"],
        extra_dataset={
            "image_size": [8, 8],
            "fft_tuple": [4, 8, 6],
            "clipped_range": 4,
            "use_gps": True,
            "gps_feature_mode": "relative_polar",
            "gps_normalize": False,
        },
        visualization={
            "splits": ["train"],
            "sample_count": 1,
            "seed": 7,
            "max_frames_per_sample": 2,
            "include_raw_image_preview": True,
        },
    )

    result = visualize_modalities(cfg)

    summary_path = Path(result["summary_path"])
    samples_path = Path(result["samples_jsonl"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]

    assert summary_path.exists()
    assert Path(result["samples_csv"]).exists()
    assert Path(result["final_config_path"]).exists()
    assert summary["actual_sample_count"] == 1
    assert records[0]["statistics"]["image"]["shape"] == [2, 8, 8]
    assert Path(records[0]["png_path"]).exists()


def test_diagnostics_skip_unenabled_image_and_lidar_cache_access(monkeypatch, tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    _write_radar_csv(tmp_path, train_csv, seq_len=2)

    def fail_cache(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("disabled modality cache should not be touched")

    monkeypatch.setattr(scenario9_module, "parameterized_image_motion_cache_dir", fail_cache)
    monkeypatch.setattr(scenario9_module, "parameterized_lidar_cache_dir", fail_cache)
    monkeypatch.setattr(scenario9_module, "load_motion_masks", fail_cache)
    monkeypatch.setattr(DeepSense6GDataset, "_lidar_bev_for_index", fail_cache)
    monkeypatch.setattr(
        scenario9_module,
        "load_radar_maps",
        lambda *args, **kwargs: (torch.zeros(2, 4, 4), torch.zeros(2, 6, 4)),  # noqa: ARG005
    )
    cfg = _diagnostic_cfg(
        tmp_path,
        train_csv_name="train.csv",
        modalities=["radar"],
        extra_dataset={
            "seq_len": 2,
            "num_pred": 1,
            "fft_tuple": [4, 8, 6],
            "clipped_range": 4,
            "image_motion_cache_dir": "image_cache",
            "lidar_cache_dir": "lidar_cache",
        },
        visualization={
            "splits": ["train"],
            "sample_count": 1,
            "modalities": ["radar"],
        },
        cache_policy="auto",
    )

    result = visualize_modalities(cfg)

    assert Path(result["summary_path"]).exists()


def _diagnostic_cfg(
    root: Path,
    *,
    train_csv_name: str,
    modalities: list[str],
    extra_dataset: dict,
    visualization: dict,
    cache_policy: str = "off",
) -> dict:
    dataset = {
        "type": "deepsense6g",
        "scene": 9,
        "data_root": str(root),
        "train_csv_name": train_csv_name,
        "test_csv_name": train_csv_name,
        "seq_len": extra_dataset.pop("seq_len", 3),
        "num_pred": extra_dataset.pop("num_pred", 1),
        "portion": 1.0,
        **extra_dataset,
    }
    return {
        "experiment": {"task": "fusion", "seed": 0},
        "data": {"cache": {"policy": cache_policy}, "dataset": dataset, "dataloader": {}},
        "model": {"teacher": {"modalities": modalities}, "student": {"modalities": modalities}},
        "diagnostics": {
            "visualization": {
                "output_dir": str(root / "diagnostics"),
                "sample_count": 1,
                "seed": 0,
                **visualization,
            }
        },
        "output": {"dir": str(root), "run_name": "diagnostics"},
    }


def _write_multimodal_csv(root: Path, csv_path: Path, *, rows: int, seq_len: int) -> None:
    columns = (
        [f"camera{i}" for i in range(1, seq_len + 1)]
        + [f"radar{i}" for i in range(1, seq_len + 1)]
        + [f"gps{i}" for i in range(1, seq_len + 1)]
        + [f"bs_gps{i}" for i in range(1, seq_len + 1)]
        + [f"beam{i}" for i in range(1, seq_len + 1)]
        + ["future_beam1", "seq_index"]
    )
    rows_out = []
    for row_idx in range(rows):
        prefix = f"row{row_idx}"
        for frame_idx in range(seq_len):
            Image.fromarray(np.full((8, 8, 3), frame_idx * 60 + row_idx, dtype=np.uint8)).save(
                root / f"{prefix}_camera_{frame_idx}.jpg"
            )
            np.save(root / f"{prefix}_radar_{frame_idx}_RA.npy", np.full((4, 4), frame_idx + 1, dtype=np.float32))
            np.save(root / f"{prefix}_radar_{frame_idx}_DA.npy", np.full((6, 4), frame_idx + 2, dtype=np.float32))
            np.savetxt(root / f"{prefix}_gps_{frame_idx}.txt", np.array([33.0 + frame_idx * 1e-5, -111.0]))
            np.savetxt(root / f"{prefix}_bs_gps_{frame_idx}.txt", np.array([33.0, -111.0]))
            _write_beam(root / f"{prefix}_beam_{frame_idx}.txt", frame_idx)
        _write_beam(root / f"{prefix}_future.txt", 5 + row_idx)
        rows_out.append(
            [f"{prefix}_camera_{idx}.jpg" for idx in range(seq_len)]
            + [f"{prefix}_radar_{idx}_RA.npy" for idx in range(seq_len)]
            + [f"{prefix}_gps_{idx}.txt" for idx in range(seq_len)]
            + [f"{prefix}_bs_gps_{idx}.txt" for idx in range(seq_len)]
            + [f"{prefix}_beam_{idx}.txt" for idx in range(seq_len)]
            + [f"{prefix}_future.txt", str(row_idx + 1)]
        )
    csv_path.write_text(
        ",".join(columns) + "\n" + "\n".join(",".join(row) for row in rows_out) + "\n",
        encoding="utf-8",
    )


def _write_radar_csv(root: Path, csv_path: Path, *, seq_len: int) -> None:
    for idx in range(seq_len):
        _write_beam(root / f"beam_{idx}.txt", idx)
    _write_beam(root / "future.txt", 4)
    columns = [f"radar{i}" for i in range(1, seq_len + 1)] + [f"beam{i}" for i in range(1, seq_len + 1)]
    columns += ["future_beam1", "seq_index"]
    values = [f"radar_{idx}_RA.npy" for idx in range(seq_len)] + [f"beam_{idx}.txt" for idx in range(seq_len)]
    values += ["future.txt", "1"]
    csv_path.write_text(",".join(columns) + "\n" + ",".join(values) + "\n", encoding="utf-8")


def _write_beam(path: Path, label: int) -> None:
    beam = np.zeros(64, dtype=np.float32)
    beam[label] = 1.0
    np.savetxt(path, beam)
