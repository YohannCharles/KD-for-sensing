from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "tools" / "benchmark_mmw_frozen_methods.py"
    spec = importlib.util.spec_from_file_location("benchmark_mmw_frozen_methods", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_defaults_bind_the_four_frozen_seed1_validation_best_paths():
    benchmark = _module()
    specs = benchmark.resolve_method_specs(None)

    assert [item.name for item in specs] == [
        "Prototype-only",
        "Hard",
        "RMBP-MM-local",
        "AMBER-Full-local",
    ]
    assert all("seed1" in item.config or "train_seed1" in item.config for item in specs)
    assert all(item.checkpoint.endswith("checkpoints/best.pth") for item in specs)
    assert "masked_feature_fusion_off_seed1" in specs[1].config


def test_rf_table_records_2_plus_1_and_95_percent_count_reduction():
    benchmark = _module()
    rows = {item["policy"]: item for item in benchmark.RF_TABLE}

    assert rows["TBCP-3"]["beam_measurements"] == 3
    assert rows["TBCP-3"]["measurement_rounds"] == 2
    assert rows["TBCP-3"]["feedback_updates"] == 1
    assert rows["Batch-TBCP-2+1"]["measurement_rounds"] == 2
    assert rows["Batch-TBCP-2+1"]["feedback_updates"] == 1
    assert rows["TBCP-3"]["measurement_reduction_vs_full64"] == pytest.approx(61 / 64)


def test_cli_smoke_exposes_device_output_warmup_and_repeats():
    benchmark = _module()
    args = benchmark.build_parser().parse_args(
        [
            "--device",
            "cpu",
            "--output",
            "tmp/complexity",
            "--warmup",
            "2",
            "--repeats",
            "3",
            "--method",
            "Hard",
            "hard.yaml",
            "hard.pth",
        ]
    )

    assert args.device == "cpu"
    assert args.output == "tmp/complexity"
    assert args.warmup == 2
    assert args.repeats == 3
    assert args.method == [["Hard", "hard.yaml", "hard.pth"]]


def test_full_model_input_helper_drops_loader_masks_and_sets_explicit_all_available_mask():
    benchmark = _module()
    batch = {
        "image": torch.zeros(1, 5, 3, 224, 224),
        "radar_ra": torch.zeros(1, 5, 1, 128, 64),
        "radar_da": torch.zeros(1, 5, 1, 128, 64),
        "gps": torch.zeros(1, 5, 3),
        "lidar": torch.zeros(1, 5, 3, 224, 224),
        "available_modalities": torch.tensor([[False, True, False, True]]),
        "modality_temporal_mask": torch.zeros(1, 5, 4, dtype=torch.bool),
    }

    inputs = benchmark._prepare_model_inputs(
        batch,
        {"seq_length": 5, "modalities": ["image", "radar", "gps", "lidar"]},
        torch.device("cpu"),
    )

    assert set(inputs) == {"image_batch", "radar_batch", "gps_batch", "lidar_batch", "force_modality_mask"}
    assert inputs["force_modality_mask"].tolist() == [[True, True, True, True]]
    assert inputs["radar_batch"].shape == (1, 5, 2, 128, 64)


def test_report_refuses_nonempty_output_and_serializes_config_device_metadata(tmp_path):
    benchmark = _module()
    row = benchmark.BenchmarkRow(
        method="Hard",
        config="hard.yaml",
        config_sha256="a" * 64,
        checkpoint="hard.pth",
        checkpoint_sha256="b" * 64,
        checkpoint_role="validation_best",
        seed=1,
        parameter_count=10,
        trainable_parameter_count=10,
        profiler_covered_flops=20,
        flops_status="profiler_covered",
        flops_error=None,
        flops_unsupported_note="unsupported note",
        forward_median_ms=1.0,
        forward_p95_ms=2.0,
        warmup=20,
        repeats=100,
        batch_size=1,
        sequence_length=5,
        modalities=("image", "radar", "gps", "lidar"),
        device="cuda:0",
        device_name="NVIDIA A40",
        device_capability=(8, 6),
        dtype="float32",
        split="validation",
        outer_test_accessed=False,
    )
    output = tmp_path / "reports"
    paths = benchmark._write_reports(output, [row], cli={"device": "cuda:0"})
    assert Path(paths["json"]).is_file()
    payload = __import__("json").loads(Path(paths["json"]).read_text())
    saved = payload["methods"][0]
    assert saved["config_sha256"] == "a" * 64
    assert saved["device_name"] == "NVIDIA A40"
    assert saved["device_capability"] == [8, 6]
    with pytest.raises(FileExistsError, match="non-empty"):
        benchmark._write_reports(output, [row], cli={})
