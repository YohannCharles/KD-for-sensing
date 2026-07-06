import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from kd_sensing.engine.pcpg_radar_balance import bprr_gate_regularization
from kd_sensing.eval.u_mask_beam_jepa_eval_matrix import _oracle_logits_from_diagnostics
from kd_sensing.models.u_mask_beam_jepa import (
    BPRRTemperatureCalibration,
    BeamPrototypeReliabilityRouter,
)


ROOT = Path(__file__).resolve().parents[1]


def test_bprr_masked_softmax_masks_unavailable_without_nan():
    router = BeamPrototypeReliabilityRouter(num_modalities=4, feature_dim=8, pattern_dim=6, hidden_dim=8, dropout=0.0)
    for param in router.parameters():
        torch.nn.init.zeros_(param)
    features = torch.randn(3, 4, 8)
    pattern = torch.zeros(3, 6)
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]], dtype=torch.bool)

    gate = router(features, mask, pattern)

    assert torch.isfinite(gate).all()
    assert torch.all(gate[~mask] == 0)
    assert gate[0, 0].item() == pytest.approx(1.0)
    assert gate[2, 3].item() == pytest.approx(1.0)
    assert gate[1].sum().item() == pytest.approx(1.0)


def test_bprr_temperature_calibration_positive_independent_and_serializable(tmp_path):
    calibration = BPRRTemperatureCalibration(num_modalities=4, init_temperature=1.0)
    with torch.no_grad():
        calibration.raw_temperature[0] += 0.5
    logits = torch.randn(2, 4, 8)

    calibrated = calibration(logits)
    temperatures = calibration.temperatures()
    payload = {name: float(value.detach()) for name, value in zip(["image", "radar", "gps", "lidar"], temperatures)}
    path = tmp_path / "modality_temperatures.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert torch.isfinite(calibrated).all()
    assert torch.all(temperatures > 0)
    assert temperatures[0].item() != pytest.approx(temperatures[1].item())
    assert json.loads(path.read_text(encoding="utf-8"))["image"] == pytest.approx(payload["image"])


def test_bprr_radar_gate_regularization_cases():
    modalities = ["image", "radar", "gps", "lidar"]
    gate = torch.tensor([[0.8, 0.05, 0.15, 0.0], [0.0, 1.0, 0.0, 0.0], [0.2, 0.2, 0.6, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0], [0, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)

    total, parts = bprr_gate_regularization(
        gate,
        mask,
        modalities,
        balance_weight=0.0,
        radar_weight=1.0,
        radar_floor=0.10,
        radar_patterns=["missing_lidar", "missing_image", "miss3"],
    )
    unavailable_total, unavailable_parts = bprr_gate_regularization(
        gate[2:],
        mask[2:],
        modalities,
        balance_weight=0.0,
        radar_weight=1.0,
        radar_floor=0.10,
        radar_patterns=["missing_lidar", "missing_image", "miss3"],
    )
    high_total, high_parts = bprr_gate_regularization(
        torch.tensor([[0.4, 0.2, 0.4, 0.0]]),
        torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
        modalities,
        balance_weight=0.0,
        radar_weight=1.0,
        radar_floor=0.10,
        radar_patterns=["missing_lidar"],
    )

    assert total.item() > 0
    assert parts["radar"].item() > 0
    assert unavailable_total.item() == pytest.approx(0.0)
    assert unavailable_parts["radar"].item() == pytest.approx(0.0)
    assert high_total.item() == pytest.approx(0.0)
    assert high_parts["radar"].item() == pytest.approx(0.0)


def test_oracle_gate_chooses_closest_available_branch_and_distribution():
    logits = torch.full((3, 3, 6), -5.0)
    logits[0, 0, 0] = 5.0
    logits[0, 1, 3] = 5.0
    logits[0, 2, 2] = 5.0
    logits[1, 0, 1] = 5.0
    logits[1, 1, 4] = 5.0
    logits[1, 2, 2] = 5.0
    logits[2, 0, 5] = 5.0
    logits[2, 1, 1] = 5.0
    logits[2, 2, 3] = 5.0
    target = torch.tensor([2, 4, 0])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0], [0, 1, 1]], dtype=torch.bool)

    oracle, chosen = _oracle_logits_from_diagnostics(
        {"unimodal_logits": logits},
        torch.zeros(3, 6),
        target,
        mask,
        ["image", "radar", "lidar"],
    )

    assert oracle.argmax(dim=-1).tolist() == [2, 4, 1]
    assert chosen == ["lidar", "radar", "radar"]
    assert Counter(chosen) == {"radar": 2, "lidar": 1}


def test_bprr_launcher_dry_run_gpu_plan_and_manifest(tmp_path):
    launcher = _load_script("launch_bprr_reliability_router_v1.py")
    experiments = ["e3", "e7", "e8", "e9", "e10", "e11", "e12"]
    jobs = launcher.plan_jobs(
        experiments=experiments,
        seeds=[1],
        gpus=[str(index) for index in range(8)],
        per_gpu=1,
        output_root=str(tmp_path),
        baseline_root=str(tmp_path / "missing_baseline"),
        base_config="missing_seed{seed}.yaml",
    )
    launcher.write_generated_configs(jobs, dry_run=True)
    manifest = launcher.write_manifest(jobs, str(tmp_path))

    assert {job["experiment"] for job in jobs} == {
        "e3_oracle_gate_eval",
        "e7_raw_confidence_gate",
        "e8_bprr_calibrated_router",
        "e9_bprr_radar_gate_reg",
        "e10_bprr_hard_subset_no_jepa",
        "e11_bprr_jepa_no_hard_subset",
        "e12_bprr_full_combo",
    }
    assert max(Counter(job["gpu"] for job in jobs).values()) == 1
    assert len(jobs) <= 8
    assert all(job["command"][:4] == ["conda", "run", "-n", "kd_mm_beam"] for job in jobs)
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", newline="")))
    assert rows[0]["gpu"] == "0"
    assert "e3_oracle_gate_eval" in {row["experiment"] for row in rows}


def test_bprr_summary_parser_reads_fake_metrics_and_baseline(tmp_path):
    summary = _load_script("summarize_bprr_reliability_router_v1.py")
    root = tmp_path / "bprr"
    baseline = tmp_path / "baseline"
    _write_eval(root / "e7_raw_confidence_gate" / "seed1" / "eval_matrix.csv", "e7_raw_confidence_gate", 1)
    _write_eval(
        root / "e8_bprr_calibrated_router" / "eval" / "e8_bprr_calibrated_router_seed2_missing_patterns.csv",
        "e8_bprr_calibrated_router",
        2,
        flat_run_name=True,
    )
    _write_gate(root / "e7_raw_confidence_gate" / "seed1" / "reliability_weights_epoch.csv")
    _write_eval(baseline / "e5_pcpg_low_encoder_lr_seed1" / "eval_matrix.csv", "e5_pcpg_low_encoder_lr", 1, top1=0.4)

    assert summary.main(["--root", str(root), "--baseline_root", str(baseline)]) == 0

    summary_rows = list(csv.DictReader((root / "summary.csv").open("r", encoding="utf-8", newline="")))
    assert summary_rows[0]["experiment"] == "e7_raw_confidence_gate"
    assert {row["seed"] for row in summary_rows} == {"1", "2"}
    assert float(summary_rows[0]["avg_missing"]) > 0
    assert (root / "summary.md").exists()
    assert (root / "drop_count_summary.csv").exists()
    assert (root / "gate_diagnostics.csv").exists()


def _write_eval(
    path: Path,
    experiment: str,
    seed: int,
    *,
    top1: float = 0.5,
    flat_run_name: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run_name = experiment if flat_run_name else f"{experiment}/seed{seed}"
    rows = [
        {"run_name": run_name, "seed": seed, "pattern": "full", "mask": "1,1,1,1", "top1": top1, "within_3": 0.7, "mae": 3.0},
        {"run_name": run_name, "seed": seed, "pattern": "missing_image", "mask": "0,1,1,1", "top1": top1 - 0.1, "within_3": 0.6, "mae": 4.0},
        {"run_name": run_name, "seed": seed, "pattern": "radar_only", "mask": "0,1,0,0", "top1": top1 - 0.2, "within_3": 0.5, "mae": 5.0},
        {"run_name": run_name, "seed": seed, "pattern": "avg_missing", "mask": "aggregate", "top1": top1 - 0.15, "within_3": 0.55, "mae": 4.5},
    ]
    _write_csv(path, rows)


def _write_gate(path: Path) -> None:
    rows = [
        {"epoch": 0, "pattern": "missing_image", "modality": "image", "mean_weight": 0.0, "available_rate": 0.0},
        {"epoch": 0, "pattern": "missing_image", "modality": "radar", "mean_weight": 0.2, "available_rate": 1.0},
        {"epoch": 0, "pattern": "missing_image", "modality": "gps", "mean_weight": 0.5, "available_rate": 1.0},
        {"epoch": 0, "pattern": "missing_image", "modality": "lidar", "mean_weight": 0.3, "available_rate": 1.0},
        {"epoch": 0, "pattern": "radar_only", "modality": "radar", "mean_weight": 1.0, "available_rate": 1.0},
    ]
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
