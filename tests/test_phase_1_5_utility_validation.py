from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.diagnostics.conditional_utility import diagnose_modalities, write_table  # noqa: E402
from kd_sensing.diagnostics.phase_1_5_utility_validation import (  # noqa: E402
    build_phase_1_5_summary,
    compute_bootstrap_confidence,
    compute_paired_delta_frame,
    run_phase_1_5_utility_validation,
)


def test_bootstrap_confidence_covers_weak_and_all_modal_comparisons():
    subset = pd.DataFrame(
        [
            _prediction("seq-a:s1", 0, "strong_only", ce=1.0, top1=0, top3=1, dba=0.5, seq_id="seq-a"),
            _prediction("seq-a:s1", 0, "strong_plus_image", ce=0.7, top1=1, top3=1, dba=1.0, seq_id="seq-a"),
            _prediction("seq-a:s1", 0, "all", ce=1.1, top1=0, top3=1, dba=0.4, seq_id="seq-a"),
            _prediction("seq-b:s2", 1, "strong_only", ce=0.4, top1=1, top3=1, dba=1.0, seq_id="seq-b"),
            _prediction("seq-b:s2", 1, "strong_plus_image", ce=0.5, top1=1, top3=1, dba=0.8, seq_id="seq-b"),
            _prediction("seq-b:s2", 1, "all", ce=0.3, top1=1, top3=1, dba=1.0, seq_id="seq-b"),
        ]
    )
    delta = pd.DataFrame(
        [
            _delta("seq-a:s1", 0, "image", delta_ce=0.3, delta_top1=1, delta_top3=0, delta_dba=0.5, seq_id="seq-a"),
            _delta("seq-b:s2", 1, "image", delta_ce=-0.1, delta_top1=0, delta_top3=0, delta_dba=-0.2, seq_id="seq-b"),
        ]
    )

    paired = compute_paired_delta_frame(subset, delta)
    ci = compute_bootstrap_confidence(
        subset,
        delta,
        bootstrap_cfg={"num_bootstrap": 25, "random_seed": 0, "cluster_key_preference": ["seq_id", "sample_id"]},
    )

    assert set(paired["comparison"]) == {"strong_plus_image_vs_strong_only", "all_vs_strong_only"}
    assert set(ci["comparison"]) == {"strong_plus_image_vs_strong_only", "all_vs_strong_only"}
    assert set(ci["metric"]) >= {"delta_dba", "delta_ce", "delta_top1", "delta_top3"}
    assert ci["cluster_key"].unique().tolist() == ["seq_id"]
    assert ci["num_clusters"].min() == 2


def test_phase_1_5_runner_writes_manifest_outputs_and_pending_status(tmp_path: Path):
    audit_dir = tmp_path / "audit"
    subset = pd.DataFrame(
        [
            _prediction("s1", 0, "strong_only", ce=1.0, top1=0, top3=1, dba=0.5),
            _prediction("s1", 0, "strong_plus_image", ce=0.9, top1=0, top3=1, dba=0.6),
            _prediction("s1", 0, "all", ce=1.1, top1=0, top3=1, dba=0.4),
        ]
    )
    delta = pd.DataFrame([_delta("s1", 0, "image", delta_ce=0.1, delta_top1=0, delta_top3=0, delta_dba=0.1)])
    write_table(subset, audit_dir, "subset_predictions")
    write_table(delta, audit_dir, "conditional_utility_per_sample_delta")
    (audit_dir / "conditional_utility_by_bucket.csv").write_text(
        "bucket_feature,bucket_name,weak_modality,horizon_name,num_samples,delta_dba,mean_delta_ce\n"
        "range,low,image,t+1,1,0.1,0.1\n",
        encoding="utf-8",
    )
    (audit_dir / "teacher_complementarity_summary.json").write_text("{}", encoding="utf-8")
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "best_top1.pth").write_text("placeholder", encoding="utf-8")

    manifest = {
        "output_dir": str(tmp_path / "phase15"),
        "conditional_utility_input": str(audit_dir),
        "bootstrap": {"num_bootstrap": 10, "random_seed": 1},
        "thresholds": {"global_delta_dba": 0.001, "min_bucket_samples": 1},
        "checkpoint_matrix": {
            "config": "configs/analysis/scene32_marf_conditional_utility_audit.yaml",
            "checkpoints_dir": str(ckpt_dir),
            "roles": {
                "best_top1": {"checkpoint": "best_top1.pth", "audit_dir": str(audit_dir)},
                "best": {"checkpoint": "best.pth", "audit_dir": str(tmp_path / "missing_audit")},
            },
        },
        "baseline_matrix": {
            "seeds": [0, 1, 2],
            "primary_baseline": "strong_only",
            "run_name_template": f"pytest_{tmp_path.name}_{{slug}}_seed{{seed}}",
            "subsets": {
                "strong_only": {
                    "slug": "gps_mmwave",
                    "modalities": ["gps", "mmwave"],
                    "config": "configs/fusion/gps_mmwave_teacher_no_kd.yaml",
                }
            },
        },
    }

    result = run_phase_1_5_utility_validation(manifest)
    summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
    metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))

    assert Path(result["bootstrap_ci"]).exists()
    assert summary["baseline_matrix"]["primary_baseline"] == "strong_only"
    assert summary["decision"]["status"] == "pending"
    assert metadata["input_status"]["subset_predictions"]["status"] == "complete"
    assert "conda run -n kd_mm_beam" in Path(summary["outputs"]["baseline_training_commands"]).read_text(encoding="utf-8")


def test_ci_aware_diagnosis_rejects_tiny_positive_gain():
    ci = pd.DataFrame(
        [
            {
                "comparison": "strong_plus_image_vs_strong_only",
                "weak_modality": "image",
                "metric": "delta_dba",
                "horizon_name": "overall",
                "mean_delta": 0.0005,
                "ci_lower": -0.001,
                "ci_upper": 0.002,
            }
        ]
    )
    diagnosis = diagnose_modalities(
        {"image": {"delta_dba": 0.0005, "delta_ce": 0.0}},
        pd.DataFrame(),
        {},
        thresholds={"global_delta_dba": 0.0, "global_delta_ce": 0.0, "teacher_rescue_rate": 0.10},
        bootstrap_confidence=ci,
    )

    assert diagnosis["image"]["label"] == "not_significant"
    assert diagnosis["image"]["evidence"]["bootstrap_ci"]["ci_lower"] < 0


def test_complete_summary_marks_low_weak_utility_when_no_evidence():
    summary = build_phase_1_5_summary(
        manifest={"baseline_matrix": {"primary_baseline": "strong_only"}, "thresholds": {"global_delta_dba": 0.001}},
        bootstrap_ci=_bootstrap_ci_without_gain(),
        checkpoint_frame=_checkpoint_frame("complete"),
        baseline_frame=pd.DataFrame(),
        baseline_summary=_baseline_summary("complete"),
        bucket_summary=pd.DataFrame(),
        teacher_summary={},
        outputs={},
    )

    assert summary["decision"]["status"] == "complete"
    assert summary["decision"]["label"] == "low_weak_utility"
    assert summary["decision"]["evidence_level"] == "final"
    assert summary["baseline_matrix"]["primary_baseline"] == "strong_only"


@pytest.mark.parametrize(
    ("checkpoint_status", "baseline_status"),
    [
        ("missing", "complete"),
        ("pending", "complete"),
        ("complete", "pending"),
    ],
)
def test_phase_1_5_summary_keeps_final_decision_pending_when_matrices_are_incomplete(
    checkpoint_status: str,
    baseline_status: str,
):
    summary = build_phase_1_5_summary(
        manifest={"baseline_matrix": {"primary_baseline": "strong_only"}, "thresholds": {"global_delta_dba": 0.001}},
        bootstrap_ci=_bootstrap_ci_without_gain(),
        checkpoint_frame=_checkpoint_frame(checkpoint_status),
        baseline_frame=pd.DataFrame(),
        baseline_summary=_baseline_summary(baseline_status),
        bucket_summary=pd.DataFrame(),
        teacher_summary={},
        outputs={},
    )

    assert summary["decision"]["status"] == "pending"
    assert summary["decision"]["label"] == "pending"
    assert summary["decision"]["evidence_level"] == "exploratory"
    assert summary["bootstrap"]["status"] == "complete"
    assert summary["baseline_matrix"]["status"] == baseline_status
    expected_checkpoint_status = "complete" if checkpoint_status == "complete" else "pending"
    assert summary["checkpoint_matrix"]["status"] == expected_checkpoint_status
    assert summary["baseline_matrix"]["primary_baseline"] == "strong_only"


def _bootstrap_ci_without_gain() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "comparison": "strong_plus_image_vs_strong_only",
                "weak_modality": "image",
                "metric": "delta_dba",
                "horizon_name": "overall",
                "mean_delta": 0.0002,
                "ci_lower": -0.0001,
                "ci_upper": 0.0005,
                "cluster_key": "sample_id",
                "cluster_key_status": "fallback",
            }
        ]
    )


def _baseline_summary(status: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"subset": "strong_only", "status": status, "dba_avg_mean": 0.90, "complete_seeds": 3, "num_seeds": 3},
            {
                "subset": "strong_plus_image",
                "status": status,
                "dba_avg_mean": 0.9002,
                "complete_seeds": 3 if status == "complete" else 2,
                "num_seeds": 3,
            },
        ]
    )


def _checkpoint_frame(status: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "role": "best_top1",
                "status": status,
                "diagnosis_labels": "image:not_significant",
            }
        ]
    )


def _prediction(
    sample_id: str,
    dataset_index: int,
    subset: str,
    *,
    ce: float,
    top1: int,
    top3: int,
    dba: float,
    seq_id: str | None = None,
) -> dict:
    row = {
        "sample_id": sample_id,
        "dataset_index": dataset_index,
        "horizon_idx": 0,
        "horizon_name": "t+1",
        "subset_name": subset,
        "gt_beam": 1,
        "valid": True,
        "ce": ce,
        "top1_hit": top1,
        "top3_hit": top3,
        "top5_hit": 1,
        "dba_score": dba,
    }
    if seq_id is not None:
        row["seq_id"] = seq_id
    return row


def _delta(
    sample_id: str,
    dataset_index: int,
    weak: str,
    *,
    delta_ce: float,
    delta_top1: float,
    delta_top3: float,
    delta_dba: float,
    seq_id: str | None = None,
) -> dict:
    row = {
        "sample_id": sample_id,
        "dataset_index": dataset_index,
        "horizon_idx": 0,
        "horizon_name": "t+1",
        "weak_modality": weak,
        "strong_plus_subset": f"strong_plus_{weak}",
        "gt_beam": 1,
        "ce_strong_only": 1.0,
        "ce_strong_plus": 1.0 - delta_ce,
        "delta_ce": delta_ce,
        "strong_only_top1": 0,
        "strong_plus_top1": delta_top1,
        "delta_top1": delta_top1,
        "strong_only_top3": 1,
        "strong_plus_top3": 1 + delta_top3,
        "delta_top3": delta_top3,
        "strong_only_dba": 0.5,
        "strong_plus_dba": 0.5 + delta_dba,
        "delta_dba": delta_dba,
    }
    if seq_id is not None:
        row["seq_id"] = seq_id
    return row
