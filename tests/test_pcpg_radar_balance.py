import csv
import importlib.util
from collections import Counter
from pathlib import Path

import pytest
import torch

from kd_sensing.engine.checkpoint_selection import select_best_checkpoint_epoch
from kd_sensing.engine.pcpg_radar_balance import static_hard_subset_weight
from kd_sensing.models.u_mask_beam_jepa import PatternConditionedPrototypeGate


ROOT = Path(__file__).resolve().parents[1]


def test_pcpg_gate_masks_unavailable_modalities_without_nan():
    gate = PatternConditionedPrototypeGate(num_modalities=4, reliability_dim=6, pattern_dim=6, hidden_dim=8)
    for param in gate.parameters():
        torch.nn.init.zeros_(param)
    features = torch.randn(3, 4, 6)
    pattern = torch.zeros(3, 6)
    mask = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    weights = gate(features, pattern, mask)

    assert torch.isfinite(weights).all()
    assert torch.all(weights[~mask] == 0)
    assert float(weights[0, 0].detach()) == pytest.approx(1.0)
    assert float(weights[1].sum().detach()) == pytest.approx(1.0)
    assert float(weights[2].sum().detach()) == pytest.approx(0.0)


def test_static_hard_subset_weight_prioritizes_hard_patterns():
    full = static_hard_subset_weight("full")
    assert static_hard_subset_weight("image_only") > full
    assert static_hard_subset_weight("lidar_only") > full
    assert static_hard_subset_weight("radar_only") > full
    assert static_hard_subset_weight("missing_image") > full
    assert static_hard_subset_weight("miss3") > full
    assert static_hard_subset_weight("unknown_pattern") == pytest.approx(1.0)


def test_checkpoint_selection_uses_avg_missing_top1_epoch():
    logs = [
        {"epoch": 1, "val/subset/missing_image/top1": 0.20, "val/subset/radar/top1": 0.30},
        {"epoch": 2, "val/subset/missing_image/top1": 0.35, "val/subset/radar/top1": 0.45},
        {"epoch": 3, "val/subset/missing_image/top1": 0.25, "val/subset/radar/top1": 0.30},
    ]

    best = select_best_checkpoint_epoch(logs, "avg_missing_top1")

    assert best["epoch"] == 2
    assert best["score"] == pytest.approx(0.40)
    assert best["metric"] == "avg_missing_top1"


def test_pcpg_launcher_dry_run_gpu_plan_and_manifest(tmp_path):
    launcher = _load_script("launch_pcpg_radar_balance_v1.py")
    jobs = launcher.plan_jobs(
        experiments=["e1_tinyvit_valacc_ckpt", "e2_tinyvit_avgmissing_ckpt"],
        seeds=[1, 2, 3, 4],
        gpus=["1", "2", "3", "4"],
        slots_per_gpu=2,
        output_root=str(tmp_path),
        base_config="base_seed{seed}.yaml",
    )
    manifest = launcher.write_manifest(jobs, str(tmp_path))

    assert len(jobs) == 8
    assert Counter(job["gpu"] for job in jobs) == {"1": 2, "2": 2, "3": 2, "4": 2}
    assert all(job["command"][:4] == ["conda", "run", "-n", "kd_mm_beam"] for job in jobs)
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["experiment"] == "e1_tinyvit_valacc_ckpt"
    assert rows[0]["config_path"].endswith("e1_tinyvit_valacc_ckpt_seed1.yaml")
    assert "kd-sensing-train" in rows[0]["command"]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
