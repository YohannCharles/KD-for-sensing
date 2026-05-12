from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.visualization.complementarity_explorer import (  # noqa: E402
    case_detail_payload,
    export_filtered_cases,
    filter_complementarity_cases,
    find_sample_index_for_case,
    load_complementarity_explorer,
    selected_event_row,
)


def test_explorer_loads_choices_and_filters_cases(tmp_path: Path):
    cases = _explorer_cases()
    output_dir = tmp_path / "complementarity"
    output_dir.mkdir()
    cases.to_csv(output_dir / "complementarity_cases.csv.gz", index=False, compression="gzip")
    (output_dir / "complementarity_summary.json").write_text(json.dumps({"total_cases": len(cases)}), encoding="utf-8")
    pd.DataFrame({"bucket_feature": ["mmwave_entropy"], "bucket_name": ["high"]}).to_csv(
        output_dir / "complementarity_by_bucket.csv",
        index=False,
    )

    data = load_complementarity_explorer(output_dir)
    result = filter_complementarity_cases(
        data["cases"],
        scene="scene32",
        horizon="t+1",
        weak_modality="image",
        strong_modality="mmwave",
        case_types=["strong_wrong_weak_correct"],
        bucket="mmwave_entropy=high",
        min_gain=0.1,
        sort_by="weak_gt_gain desc",
        max_rows=1,
    )

    assert data["available"] is True
    assert data["choices"]["defaults"]["scene"] == "scene32"
    assert data["choices"]["defaults"]["horizon"] == "t+1"
    assert data["choices"]["defaults"]["strong_modality"] == "mmwave"
    assert data["choices"]["defaults"]["weak_modality"] == "image"
    assert result["stats"]["filtered_rows"] == 2
    assert result["stats"]["displayed_rows"] == 1
    assert result["table"]["sample_id"].tolist() == ["b"]
    assert result["records"][0]["sample_id"] == "b"


def test_explorer_filter_fallback_selection_detail_and_export(tmp_path: Path):
    cases = _explorer_cases()
    result = filter_complementarity_cases(
        cases,
        scene="scene32",
        horizon="t+1",
        strong_modality="mmwave",
        weak_modality="image",
        case_types=["strong_correct_fusion_wrong"],
        sort_by="missing_metric desc",
        max_rows=10,
    )
    row = selected_event_row(SimpleNamespace(index=(0, 0)), result["table"])
    samples = [
        {"sample_id": "different", "_manifest_index": 0},
        {"sample_id": "c", "_manifest_index": 1, "label": {"future_beams": [1]}},
    ]
    sample_index = find_sample_index_for_case(samples, row)
    detail = case_detail_payload(row, samples[sample_index])
    export_path = export_filtered_cases(result["records"], output_dir=tmp_path)

    assert "falling back to sample_id" in result["warnings"][0]
    assert row["sample_id"] == "c"
    assert sample_index == 1
    assert detail["manifest"]["matched"] is True
    assert detail["manifest"]["distribution_message"] == "probability distribution unavailable"
    assert export_path is not None
    assert Path(export_path).exists()


def test_explorer_filters_strong_modality_all_weaks_and_exports_fields(tmp_path: Path):
    cases = _explorer_cases()
    all_result = filter_complementarity_cases(
        cases,
        scene="all",
        horizon="all",
        strong_modality="all",
        weak_modality="all",
        case_types=None,
        sort_by="sample_id asc",
        max_rows=10,
    )
    gps_result = filter_complementarity_cases(
        cases,
        scene="scene9",
        horizon="t+2",
        strong_modality="gps",
        weak_modality="all",
        case_types=None,
        sort_by="sample_id asc",
        max_rows=10,
    )
    detail = case_detail_payload(gps_result["records"][0], None)
    export_path = export_filtered_cases(gps_result["records"], output_dir=tmp_path)
    exported = pd.read_csv(export_path)

    assert all_result["stats"]["filtered_rows"] == 4
    assert gps_result["table"]["sample_id"].tolist() == ["d"]
    assert detail["case"]["strong_modality"] == "gps"
    assert detail["case"]["strong_weak_pair"] == "gps+radar"
    assert {"strong_modality", "strong_prediction_source", "strong_weak_pair"}.issubset(exported.columns)


def test_explorer_empty_state_is_safe():
    data = load_complementarity_explorer(None)
    result = filter_complementarity_cases(data["cases"])

    assert data["available"] is False
    assert data["choices"]["scenes"] == ["all"]
    assert result["table"].empty
    assert result["stats"]["filtered_rows"] == 0


def _explorer_cases() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "a",
                "dataset_index": 0,
                "scene": "scene32",
                "horizon_name": "t+1",
                "strong_modality": "mmwave",
                "weak_modality": "image",
                "strong_weak_pair": "mmwave+image",
                "strong_prediction_source": "teacher_predictions",
                "fusion_prediction_available": True,
                "strong_correct": False,
                "case_type": "strong_wrong_weak_correct_fusion_wrong",
                "research_tags": "strong_wrong_weak_correct|unused_complementary",
                "weak_gt_gain": 0.2,
                "fusion_gt_gain": -0.1,
                "mmwave_entropy_bucket": "high",
            },
            {
                "sample_id": "b",
                "dataset_index": 1,
                "scene": "scene32",
                "horizon_name": "t+1",
                "strong_modality": "mmwave",
                "weak_modality": "image",
                "strong_weak_pair": "mmwave+image",
                "strong_prediction_source": "teacher_predictions",
                "fusion_prediction_available": True,
                "strong_correct": False,
                "case_type": "strong_wrong_weak_correct_fusion_correct",
                "research_tags": "strong_wrong_weak_correct|rescue|strong_wrong_fusion_correct",
                "weak_gt_gain": 0.4,
                "fusion_gt_gain": 0.3,
                "mmwave_entropy_bucket": "high",
            },
            {
                "sample_id": "c",
                "dataset_index": 2,
                "scene": "scene32",
                "horizon_name": "t+1",
                "strong_modality": "mmwave",
                "weak_modality": "image",
                "strong_weak_pair": "mmwave+image",
                "strong_prediction_source": "teacher_predictions",
                "fusion_prediction_available": True,
                "strong_correct": True,
                "case_type": "strong_correct_fusion_wrong",
                "research_tags": "negative_transfer",
                "weak_gt_gain": -0.2,
                "fusion_gt_gain": -0.4,
                "mmwave_entropy_bucket": "low",
            },
            {
                "sample_id": "d",
                "dataset_index": 3,
                "scene": "scene9",
                "horizon_name": "t+2",
                "strong_modality": "gps",
                "weak_modality": "radar",
                "strong_weak_pair": "gps+radar",
                "strong_prediction_source": "teacher_predictions",
                "fusion_prediction_available": False,
                "strong_correct": False,
                "case_type": "all_wrong",
                "research_tags": "none",
                "weak_gt_gain": 0.8,
                "fusion_gt_gain": 0.0,
                "mmwave_entropy_bucket": "high",
            },
        ]
    )
