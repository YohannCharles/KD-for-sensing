import csv
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import torch

from kd_sensing.engine.pcpg_radar_balance import (
    hard_subset_sample_weight,
    router_focus_pattern_enabled,
    soft_static_hard_subset_weight,
    supervised_router_masked_softmax,
    supervised_router_oracle_targets,
)


ROOT = Path(__file__).resolve().parents[1]


def test_soft_static_hard_subset_weighting_order_unknown_and_finite():
    radar_only = soft_static_hard_subset_weight("radar_only")
    drop3 = soft_static_hard_subset_weight("image_only")
    drop2 = soft_static_hard_subset_weight("drop2")
    full = soft_static_hard_subset_weight("full")

    assert radar_only > drop3 > drop2 > full
    assert soft_static_hard_subset_weight("missing_image") == pytest.approx(1.35)
    assert soft_static_hard_subset_weight("unknown_pattern") == pytest.approx(1.0)
    assert hard_subset_sample_weight("missing_radar_gps", mode="soft_static") == pytest.approx(1.15)
    assert torch.isfinite(torch.tensor([radar_only, drop3, drop2, full])).all()


def test_supervised_router_oracle_target_available_min_error_and_tie():
    logits = torch.full((4, 3, 8), -6.0)
    logits[0, 0, 1] = 6.0
    logits[0, 1, 4] = 6.0
    logits[0, 2, 2] = 6.0
    logits[1, 0, 0] = 6.0
    logits[1, 1, 3] = 6.0
    logits[1, 2, 4] = 6.0
    logits[2, 0, 1] = 6.0
    logits[2, 1, 3] = 6.0
    logits[2, 2, 6] = 6.0
    logits[3, 0, 7] = 6.0
    logits[3, 1, 0] = 6.0
    logits[3, 2, 2] = 6.0
    labels = torch.tensor([2, 4, 2, 7])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.bool)

    targets = supervised_router_oracle_targets(logits, labels, mask)

    assert targets.tolist() == [2, 1, 0, 0]
    assert torch.all(mask[torch.arange(mask.shape[0]), targets])
    assert Counter(targets.tolist())[0] == 2


def test_supervised_router_masked_softmax_single_multi_no_nan():
    logits = torch.tensor([[1.0, 99.0, -3.0], [0.0, 2.0, -1.0], [-4.0, 0.0, 7.0]])
    mask = torch.tensor([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=torch.bool)

    gate = supervised_router_masked_softmax(logits, mask)

    assert torch.isfinite(gate).all()
    assert torch.all(gate[~mask] == 0)
    assert gate[0, 0].item() == pytest.approx(1.0)
    assert gate[2, 2].item() == pytest.approx(1.0)
    assert gate[1].sum().item() == pytest.approx(1.0)


def test_router_distill_focus_pattern_aliases():
    focus = {"missing_image", "miss2"}

    assert router_focus_pattern_enabled("missing_image", focus)
    assert router_focus_pattern_enabled("drop2", focus)
    assert router_focus_pattern_enabled("miss2", focus)
    assert not router_focus_pattern_enabled("full", focus)
    assert not router_focus_pattern_enabled("radar_only", focus)


def test_overnight_launcher_dry_run_gpu_plan_and_manifest(tmp_path: Path):
    launcher = load_script("launch_overnight_branch_router_v2.py")
    jobs = launcher.plan_jobs(
        experiments=list(launcher.ANCHOR_EXPERIMENTS + launcher.EXPLORE_EXPERIMENTS),
        anchor_seeds=[1, 2, 3, 4, 5],
        explore_seeds=[1, 2, 3],
        gpus=["1", "2"],
        per_gpu=2,
        output_root=str(tmp_path),
        baseline_root=str(tmp_path / "baseline"),
        base_config="missing_seed{seed}.yaml",
    )
    launcher.write_generated_configs(jobs, dry_run=True)
    manifest = launcher.write_manifest(jobs, str(tmp_path))

    assert len(jobs) == 40
    assert {job["gpu"] for job in jobs} <= {"1", "2"}
    assert [job["seed"] for job in jobs if job["experiment"].startswith("a1_")] == [1, 2, 3, 4, 5]
    assert [job["seed"] for job in jobs if job["experiment"].startswith("b3_")] == [1, 2, 3]
    for start in range(0, len(jobs), 4):
        wave = jobs[start : start + 4]
        per_gpu = Counter(job["gpu"] for job in wave)
        assert len(wave) <= 4
        assert max(per_gpu.values()) <= 2
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 40
    assert {"experiment", "seed", "gpu", "cmd", "status", "log_path", "output_dir"} <= set(rows[0])


def test_overnight_launcher_reuses_seed1_base_config_when_seed_specific_missing(tmp_path: Path, monkeypatch):
    launcher = load_script("launch_overnight_branch_router_v2.py")
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    (tmp_path / "base_seed1.yaml").write_text(
        "experiment:\n  seed: 99\ntraining:\n  epochs: 40\n",
        encoding="utf-8",
    )
    jobs = launcher.plan_jobs(
        experiments=["a1"],
        anchor_seeds=[2],
        explore_seeds=[2],
        gpus=["4"],
        per_gpu=1,
        output_root=str(tmp_path / "out"),
        baseline_root=str(tmp_path / "baseline"),
        base_config="base_seed{seed}.yaml",
    )

    launcher.write_generated_configs(jobs, dry_run=False)

    generated = (tmp_path / "out" / "generated_configs" / "a1_e5_low_encoder_lr_anchor_seed2.yaml").read_text(
        encoding="utf-8"
    )
    assert jobs[0]["base_config_resolved"] == "base_seed1.yaml"
    assert "seed: 2" in generated
    assert "epochs: 40" in generated


def test_overnight_summary_parser_reads_fake_metrics_and_baselines(tmp_path: Path):
    summary = load_script("summarize_overnight_branch_router_v2.py")
    root = tmp_path / "overnight"
    baseline = tmp_path / "baseline"
    write_eval(root / "b3_hard_soft_no_jepa" / "eval" / "b3_hard_soft_no_jepa_seed1_missing_patterns.csv", "b3_hard_soft_no_jepa", 1)
    write_eval(root / "c1_supervised_router_e5" / "eval" / "c1_supervised_router_e5_seed1_missing_patterns.csv", "c1_supervised_router_e5", 1, router=True)
    write_gate(root / "c1_supervised_router_e5" / "seed1" / "reliability_weights_epoch.csv")
    write_eval(baseline / "e5_pcpg_low_encoder_lr_seed1" / "eval_matrix.csv", "e5_pcpg_low_encoder_lr", 1, top1=0.4)

    assert summary.main(["--root", str(root), "--baseline_roots", str(baseline)]) == 0

    rows = list(csv.DictReader((root / "summary.csv").open("r", encoding="utf-8", newline="")))
    experiments = {row["experiment"] for row in rows}
    assert {"b3_hard_soft_no_jepa", "c1_supervised_router_e5", "e5_pcpg_low_encoder_lr"} <= experiments
    assert (root / "summary.md").exists()
    assert (root / "drop_count_summary.csv").exists()
    assert (root / "pattern_metrics.csv").exists()
    router_rows = list(csv.DictReader((root / "router_diagnostics.csv").open("r", encoding="utf-8", newline="")))
    assert any(row.get("experiment") == "c1_supervised_router_e5" for row in router_rows)


def write_eval(path: Path, experiment: str, seed: int, *, top1: float = 0.5, router: bool = False) -> None:
    run_name = f"{experiment}/seed{seed}"
    rows = [
        {"run_name": run_name, "seed": seed, "pattern": "full", "mask": "1,1,1,1", "top1": top1, "within_3": 0.8, "mae": 2.0},
        {"run_name": run_name, "seed": seed, "pattern": "missing_image", "mask": "0,1,1,1", "top1": top1 - 0.1, "within_3": 0.7, "mae": 3.0},
        {"run_name": run_name, "seed": seed, "pattern": "missing_image_lidar", "mask": "0,1,0,1", "top1": top1 - 0.15, "within_3": 0.6, "mae": 3.5},
        {"run_name": run_name, "seed": seed, "pattern": "radar_only", "mask": "0,1,0,0", "top1": top1 - 0.2, "within_3": 0.5, "mae": 4.0},
        {"run_name": run_name, "seed": seed, "pattern": "avg_missing", "mask": "aggregate", "top1": top1 - 0.12, "within_3": 0.65, "mae": 3.2},
    ]
    if router:
        for row in rows:
            row.update(
                {
                    "mean_gate_image": 0.2,
                    "mean_gate_lidar": 0.2,
                    "mean_gate_radar": 0.4,
                    "mean_gate_gps": 0.2,
                    "gate_entropy": 1.1,
                    "router_oracle_acc": 0.7,
                    "router_oracle_acc_missing_image": 0.75 if row["pattern"] == "missing_image" else "",
                    "router_oracle_acc_drop2": 0.6 if row["pattern"] == "missing_image_lidar" else "",
                    "oracle_target_radar_rate": 0.5,
                    "radar_gate_missing_image": 0.45 if row["pattern"] == "missing_image" else "",
                    "radar_gate_drop2": 0.35 if row["pattern"] == "missing_image_lidar" else "",
                }
            )
    write_csv(path, rows)


def write_gate(path: Path) -> None:
    rows = [
        {"epoch": 0, "pattern": "missing_image", "modality": "radar", "mean_weight": 0.45, "available_rate": 1.0, "gate_entropy": 1.0},
        {"epoch": 0, "pattern": "missing_image_lidar", "modality": "radar", "mean_weight": 0.35, "available_rate": 1.0, "gate_entropy": 1.1},
        {"epoch": 0, "pattern": "missing_image", "modality": "image", "mean_weight": 0.0, "available_rate": 0.0, "gate_entropy": 1.0},
    ]
    write_csv(path, rows)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
