from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.diagnostics.complementarity import (  # noqa: E402
    CASE_ALL_CORRECT,
    CASE_ALL_WRONG,
    CASE_NEGATIVE_TRANSFER,
    CASE_OTHER,
    CASE_RESCUE,
    CASE_STRONG_WRONG_FUSION_CORRECT,
    CASE_UNUSED_COMPLEMENTARY,
    build_case_table,
    canonical_subset_name,
    compute_bucket_summary,
    compute_summary,
    load_subset_predictions,
)


def test_case_table_assigns_main_case_type_and_research_tags():
    subset, teacher, delta, features = _analysis_frames()

    cases, metadata = build_case_table(
        subset,
        teacher_predictions=teacher,
        per_sample_delta=delta,
        communication_state_features=features,
        strong_subset="gps+mmwave",
        weak_modalities=("image",),
        fusion_subsets={"image": "gps+mmwave+image"},
        scene="scene32",
    )

    by_id = cases.set_index("sample_id")
    assert by_id.loc["rescue", "case_type"] == CASE_RESCUE
    assert by_id.loc["unused", "case_type"] == CASE_UNUSED_COMPLEMENTARY
    assert by_id.loc["negative", "case_type"] == CASE_NEGATIVE_TRANSFER
    assert by_id.loc["fusion-rescue", "case_type"] == CASE_STRONG_WRONG_FUSION_CORRECT
    assert by_id.loc["all-correct", "case_type"] == CASE_ALL_CORRECT
    assert by_id.loc["all-wrong", "case_type"] == CASE_ALL_WRONG
    assert by_id.loc["other", "case_type"] == CASE_OTHER
    assert "strong_wrong_weak_correct" in by_id.loc["rescue", "research_tags"]
    assert "rescue" in by_id.loc["rescue", "research_tags"]
    assert "unused_complementary" in by_id.loc["unused", "research_tags"]
    assert "negative_transfer" in by_id.loc["negative", "research_tags"]
    assert by_id.loc["rescue", "weak_prediction_source"] == "teacher_predictions"
    assert by_id.loc["rescue", "weak_gt_gain"] == pytest.approx(0.5)
    assert by_id.loc["rescue", "fusion_gt_gain"] == pytest.approx(0.4)
    assert by_id.loc["rescue", "delta_top1"] == 1
    assert "mmwave_entropy_bucket" in cases.columns
    assert metadata["weak_prediction_sources"]["image"]["source"] == "teacher_predictions"
    assert metadata["probability_metrics_available"] is True


def test_summary_metrics_use_explicit_denominators_and_net_gain_count():
    subset, teacher, _, _ = _analysis_frames()
    cases, metadata = build_case_table(
        subset,
        teacher_predictions=teacher,
        strong_subset="gps+mmwave",
        weak_modalities=("image",),
        fusion_subsets={"image": "gps+mmwave+image"},
    )

    summary = compute_summary(cases, metadata=metadata, scene="scene32")
    global_metrics = summary["global"]

    assert global_metrics["complementarity_rate"] == {
        "numerator": 2,
        "denominator": 4,
        "value": 0.5,
    }
    assert global_metrics["rescue_rate_given_complementary"]["value"] == 0.5
    assert global_metrics["unused_complementary_rate"]["value"] == 0.5
    assert global_metrics["negative_transfer_rate"]["numerator"] == 1
    assert global_metrics["negative_transfer_rate"]["denominator"] == 3
    assert global_metrics["net_fusion_gain_count"] == 1
    assert summary["by_weak_modality"]["image"]["count"] == 7
    assert summary["by_case_type"][CASE_RESCUE]["mean_fusion_gt_gain"] == pytest.approx(0.4)


def test_schema_alias_load_and_missing_probability_degrade_safely(tmp_path: Path):
    subset, teacher, _, _ = _analysis_frames(include_probability=False)
    subset_path = tmp_path / "subset_predictions.csv.gz"
    teacher_path = tmp_path / "teacher_predictions.csv.gz"
    subset.to_csv(subset_path, index=False, compression="gzip")
    teacher.to_csv(teacher_path, index=False, compression="gzip")

    tables = load_subset_predictions(tmp_path)
    cases, metadata = build_case_table(
        tables.subset_predictions,
        teacher_predictions=tables.teacher_predictions,
        strong_subset="gps+mmwave",
        weak_modalities=("image",),
        fusion_subsets={"image": "gps+mmwave+image"},
    )

    assert canonical_subset_name("gps+mmwave") == "strong_only"
    assert canonical_subset_name("gps+mmwave+image") == "strong_plus_image"
    assert cases["p_true_strong"].isna().all()
    assert cases["weak_gt_gain"].isna().all()
    assert metadata["probability_metrics_available"] is False
    assert "Missing probability" in metadata["probability_metrics_unavailable_reason"]


def test_bucket_summary_uses_sample_buckets_and_reports_unavailable_path():
    subset, teacher, _, features = _analysis_frames()
    cases, _ = build_case_table(
        subset,
        teacher_predictions=teacher,
        strong_subset="gps+mmwave",
        weak_modalities=("image",),
        fusion_subsets={"image": "gps+mmwave+image"},
        communication_state_features=features,
    )

    bucket_summary, bucket_metadata = compute_bucket_summary(cases, return_metadata=True)
    empty_summary, empty_metadata = compute_bucket_summary(
        cases.drop(columns=[column for column in cases.columns if column.endswith("_bucket")]),
        return_metadata=True,
    )

    assert bucket_metadata["bucket_statistics_available"] is True
    assert set(bucket_summary["bucket_feature"]) >= {"mmwave_entropy"}
    assert "strong_wrong_weak_correct_count" in bucket_summary.columns
    assert empty_summary.empty
    assert empty_metadata["bucket_statistics_available"] is False


def test_strong_modality_pair_mode_uses_teacher_predictions_and_optional_fusion():
    subset, teacher = _strong_pair_frames()

    cases, metadata = build_case_table(
        subset,
        teacher_predictions=teacher,
        strong_modalities=("mmwave",),
        weak_modalities=("image", "radar"),
        pair_fusion_subsets={"mmwave+image": "mmwave+image"},
        scene="scene32",
    )
    summary = compute_summary(cases, metadata=metadata, scene="scene32")

    by_pair = cases.set_index(["strong_weak_pair", "sample_id"])
    assert canonical_subset_name("single_best_mmwave") == "mmwave"
    assert metadata["analysis_mode"] == "strong_modality_pair"
    assert metadata["strong_prediction_sources"]["mmwave"]["source"] == "teacher_predictions"
    assert metadata["fusion_subset_availability"]["mmwave+image"]["fusion_prediction_available"] is True
    assert metadata["fusion_subset_availability"]["mmwave+radar"]["fusion_prediction_available"] is False
    assert set(cases["strong_weak_pair"]) == {"mmwave+image", "mmwave+radar"}
    assert by_pair.loc[("mmwave+image", "pair-rescue"), "case_type"] == CASE_RESCUE
    assert by_pair.loc[("mmwave+image", "pair-negative"), "case_type"] == CASE_NEGATIVE_TRANSFER
    assert by_pair.loc[("mmwave+radar", "pair-rescue"), "case_type"] == "strong_wrong_weak_correct"
    assert bool(by_pair.loc[("mmwave+radar", "pair-rescue"), "fusion_prediction_available"]) is False
    assert "strong_wrong_weak_correct" in by_pair.loc[("mmwave+radar", "pair-rescue"), "research_tags"]
    assert summary["by_strong_modality"]["mmwave"]["count"] == 4
    assert summary["by_strong_weak_pair"]["mmwave+image"]["fusion_metrics_available"] is True
    assert summary["by_strong_weak_pair"]["mmwave+radar"]["fusion_metrics_available"] is False
    assert summary["by_strong_weak_pair"]["mmwave+radar"]["rescue_rate_given_complementary"]["value"] is None


def _analysis_frames(include_probability: bool = True):
    cases = [
        ("rescue", 1, 0, 1, 1, 0.2, 0.7, 0.6),
        ("unused", 1, 0, 1, 2, 0.2, 0.8, 0.1),
        ("negative", 1, 1, 2, 2, 0.7, 0.2, 0.1),
        ("fusion-rescue", 1, 0, 2, 1, 0.1, 0.2, 0.6),
        ("all-correct", 1, 1, 1, 1, 0.7, 0.8, 0.9),
        ("all-wrong", 1, 0, 2, 2, 0.1, 0.2, 0.2),
        ("other", 1, 1, 2, 1, 0.7, 0.1, 0.8),
    ]
    subset_rows = []
    teacher_rows = []
    delta_rows = []
    feature_rows = []
    for idx, (sample_id, y_true, strong_pred, weak_pred, fusion_pred, p_strong, p_weak, p_fusion) in enumerate(cases):
        base = {
            "sample_id": sample_id,
            "dataset_index": idx,
            "scene_slug": "scene32",
            "split": "test",
            "horizon_idx": 0,
            "horizon_name": "t+1",
            "gt_beam": y_true,
            "valid": True,
        }
        subset_rows.append(_prediction_row(base, "gps+mmwave", strong_pred, p_strong, include_probability))
        subset_rows.append(_prediction_row(base, "gps+mmwave+image", fusion_pred, p_fusion, include_probability))
        teacher_row = _prediction_row(base, "image", weak_pred, p_weak, include_probability)
        teacher_row["teacher_modality"] = "image"
        teacher_rows.append(teacher_row)
        delta_rows.append(
            {
                "sample_id": sample_id,
                "dataset_index": idx,
                "horizon_idx": 0,
                "horizon_name": "t+1",
                "weak_modality": "image",
                "delta_ce": 0.1,
                "delta_top1": int(fusion_pred == y_true) - int(strong_pred == y_true),
                "delta_top3": 0,
                "delta_dba": 0.2,
            }
        )
        feature_rows.append(
            {
                "sample_id": sample_id,
                "dataset_index": idx,
                "horizon_idx": 0,
                "horizon_name": "t+1",
                "mmwave_entropy": float(idx),
                "beam_transition": idx % 2,
            }
        )
    return pd.DataFrame(subset_rows), pd.DataFrame(teacher_rows), pd.DataFrame(delta_rows), pd.DataFrame(feature_rows)


def _strong_pair_frames():
    rows = [
        ("pair-rescue", 1, 0, 1, 1, 1, 0.2, 0.7, 0.8, 0.6),
        ("pair-negative", 1, 1, 2, 2, 2, 0.8, 0.2, 0.1, 0.1),
    ]
    subset_rows = []
    teacher_rows = []
    for idx, (
        sample_id,
        y_true,
        strong_pred,
        image_pred,
        radar_pred,
        fusion_pred,
        p_strong,
        p_image,
        p_radar,
        p_fusion,
    ) in enumerate(rows):
        base = {
            "sample_id": sample_id,
            "dataset_index": idx,
            "scene_slug": "scene32",
            "split": "test",
            "horizon_idx": 0,
            "horizon_name": "t+1",
            "gt_beam": y_true,
            "valid": True,
        }
        subset_rows.append(_prediction_row(base, "mmwave+image", fusion_pred, p_fusion, True))
        for modality, pred, p_true in [
            ("mmwave", strong_pred, p_strong),
            ("gps", strong_pred, p_strong),
            ("image", image_pred, p_image),
            ("radar", radar_pred, p_radar),
            ("lidar", image_pred, p_image),
        ]:
            row = _prediction_row(base, modality, pred, p_true, True)
            row["teacher_modality"] = modality
            teacher_rows.append(row)
    return pd.DataFrame(subset_rows), pd.DataFrame(teacher_rows)


def _prediction_row(base: dict, subset_name: str, pred: int, p_true: float, include_probability: bool) -> dict:
    row = {
        **base,
        "subset_name": subset_name,
        "pred_top1": pred,
    }
    if include_probability:
        row.update(
            {
                "gt_prob": p_true,
                "top1_prob": max(p_true, 0.55),
                "top2_prob": 0.2,
            }
        )
    return row
