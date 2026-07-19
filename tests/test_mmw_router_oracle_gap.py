import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "eval_mmw_router_oracle_gap", ROOT / "scripts/eval_mmw_router_oracle_gap.py"
)
assert SPEC is not None and SPEC.loader is not None
ORACLE = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault(SPEC.name, ORACLE)
SPEC.loader.exec_module(ORACLE)


def test_oracle_gap_branches_use_same_logits_and_beam_power_utility() -> None:
    unimodal = torch.zeros(2, 4, 4)
    unimodal[0, 0, 0] = 4
    unimodal[0, 1, 1] = 4
    unimodal[0, 2, 2] = 4
    unimodal[0, 3, 3] = 4
    unimodal[1] = unimodal[0]
    router = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]])
    powers = torch.tensor([[1.0, 2.0, 8.0, 4.0], [8.0, 4.0, 2.0, 1.0]])

    result = ORACLE.oracle_gap_branches(unimodal, router, powers)

    assert torch.equal(result["uniform_logits"], unimodal.mean(dim=1))
    assert result["oracle_modality"].tolist() == [2, 0]
    assert torch.equal(result["oracle_logits"][0], unimodal[0, 2])
    assert result["unimodal_normalized_gain"][0].tolist() == pytest.approx([0.125, 0.25, 1.0, 0.5])
    assert bool((result["router_soft_oracle_regret"] >= 0).all())
    assert result["router_selection_oracle_regret"].tolist() == pytest.approx([0.875, 0.875])


def test_oracle_gap_branches_ignore_unavailable_extreme_modality() -> None:
    unimodal = torch.zeros(1, 3, 3)
    unimodal[0, 0, 0] = 4
    unimodal[0, 1, 1] = 4
    unimodal[0, 2, 2] = 1_000_000
    router = torch.tensor([[0.1, 0.3, 0.6]])
    powers = torch.tensor([[4.0, 8.0, 16.0]])
    available = torch.tensor([[True, True, False]])

    result = ORACLE.oracle_gap_branches(unimodal, router, powers, available)

    assert torch.equal(result["uniform_logits"], unimodal[:, :2].mean(dim=1))
    assert result["oracle_modality"].tolist() == [1]
    assert torch.equal(result["oracle_logits"], unimodal[:, 1])
    assert result["router_soft_oracle_regret"].tolist() == pytest.approx([0.0625])
    assert result["router_selection_oracle_regret"].tolist() == pytest.approx([0.0])


def test_condition_grid_is_clean_plus_four_paired_three_level_corruptions() -> None:
    assert len(ORACLE.CONDITIONS) == 13
    assert ORACLE.CONDITIONS[0] == "clean"
    assert {ORACLE.parse_condition(value).name for value in ORACLE.CONDITIONS[1:]} == {
        "image_occlusion",
        "radar_noise",
        "lidar_sparsify",
        "gps_noise",
    }
