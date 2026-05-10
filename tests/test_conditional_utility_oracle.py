from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.diagnostics.conditional_utility import (  # noqa: E402
    compute_subset_oracle,
    compute_teacher_complementarity,
)


def test_subset_oracle_selects_min_ce_and_reports_distribution():
    rows = [
        _prediction("s1", 0, "strong_only", ce=1.0, top1=0, dba=0.5),
        _prediction("s1", 0, "strong_plus_image", ce=0.2, top1=1, dba=1.0),
        _prediction("s1", 0, "all", ce=0.3, top1=1, dba=0.9),
        _prediction("s2", 1, "strong_only", ce=0.4, top1=1, dba=0.8),
        _prediction("s2", 1, "strong_plus_image", ce=0.8, top1=0, dba=0.4),
        _prediction("s2", 1, "all", ce=0.7, top1=0, dba=0.5),
    ]
    summary, oracle_rows = compute_subset_oracle(pd.DataFrame(rows), candidates=("strong_only", "strong_plus_image", "all"))

    assert oracle_rows.sort_values("sample_id")["oracle_subset"].tolist() == ["strong_plus_image", "strong_only"]
    assert summary["oracle_choice_distribution"]["strong_plus_image"] == 0.5
    assert summary["oracle_gain_vs_strong_only"]["delta_ce"] > 0


def test_teacher_complementarity_counts_rescue_and_advantage():
    subset = pd.DataFrame(
        [
            _prediction("s1", 0, "strong_only", ce=1.0, top1=0, dba=0.4, gt_prob=0.2),
            _prediction("s2", 1, "strong_only", ce=0.1, top1=1, dba=1.0, gt_prob=0.9),
        ]
    )
    teacher = pd.DataFrame(
        [
            _teacher("s1", 0, "image", ce=0.2, top1=1, gt_prob=0.8),
            _teacher("s2", 1, "image", ce=0.3, top1=1, gt_prob=0.7),
        ]
    )
    summary, rescue = compute_teacher_complementarity(teacher, subset, weak_modalities=("image",))

    assert summary["image"]["rescue_rate_given_strong_top1_wrong"] == 1.0
    assert summary["image"]["teacher_gt_prob_advantage_rate"] == 0.5
    assert rescue["teacher_rescue_top1"].sum() == 1


def _prediction(sample_id: str, dataset_index: int, subset: str, *, ce: float, top1: int, dba: float, gt_prob: float = 0.5):
    return {
        "sample_id": sample_id,
        "dataset_index": dataset_index,
        "horizon_idx": 0,
        "horizon_name": "t+1",
        "subset_name": subset,
        "gt_beam": 1,
        "valid": True,
        "ce": ce,
        "top1_hit": top1,
        "top3_hit": 1,
        "top5_hit": 1,
        "dba_score": dba,
        "gt_prob": gt_prob,
    }


def _teacher(sample_id: str, dataset_index: int, modality: str, *, ce: float, top1: int, gt_prob: float):
    row = _prediction(sample_id, dataset_index, modality, ce=ce, top1=top1, dba=1.0, gt_prob=gt_prob)
    row["teacher_modality"] = modality
    return row
