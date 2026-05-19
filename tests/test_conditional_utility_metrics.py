from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.diagnostics.communication_state_features import assign_buckets, fit_bucket_thresholds  # noqa: E402
from kd_sensing.diagnostics.conditional_utility import (  # noqa: E402
    aggregate_subset_metrics,
    build_conditional_utility_summary,
    compute_bucket_summary,
    compute_marginal_deltas,
    compute_subset_oracle,
    records_from_logits,
    write_json,
    write_table,
)
from kd_sensing.evaluation.metrics import DBA_TOP_K, calculate_dba_score  # noqa: E402


def test_dba_uses_deepsense_average_y1_to_y3_formula():
    logits = torch.tensor([[[6.0, 0.0, 0.0, 7.0, 8.0, 0.0]]])
    labels = torch.tensor([[0]])

    score = calculate_dba_score(logits, labels)

    assert DBA_TOP_K == 3
    assert score.tolist() == pytest.approx([(0.2 + 0.4 + 1.0) / 3])


def test_records_from_logits_match_aggregate_dba():
    logits = torch.tensor(
        [
            [[5.0, 1.0, 0.0, -1.0, -2.0, -3.0], [0.0, 2.0, 5.0, 1.0, -1.0, -2.0]],
            [[0.0, 3.0, 2.0, 1.0, -1.0, -2.0], [4.0, 3.0, 1.0, 0.0, -1.0, -2.0]],
        ]
    )
    labels = torch.tensor([[0, 3], [2, 0]])

    records = records_from_logits(
        logits,
        labels,
        subset_name="strong_only",
        modalities=("gps", "mmwave"),
        dba_delta=2,
    )
    frame = pd.DataFrame(records)
    metrics = aggregate_subset_metrics(frame)
    expected_dba = calculate_dba_score(logits, labels, delta=2)

    assert frame.loc[0, "pred_top1"] == 0
    assert frame.loc[0, "top1_hit"] == 1
    assert frame.loc[1, "top3_hit"] == 1
    assert frame.loc[1, "ce"] > 0
    assert metrics["strong_only"]["dba"] == pytest.approx(expected_dba.tolist())


def test_marginal_delta_signs_and_dummy_outputs(tmp_path: Path):
    labels = torch.tensor([[0, 1], [1, 2]])
    strong = torch.tensor(
        [
            [[4.0, 1.0, 0.0], [0.0, 4.0, 1.0]],
            [[3.0, 2.0, 0.0], [4.0, 1.0, 2.0]],
        ]
    )
    plus_image = torch.tensor(
        [
            [[5.0, 1.0, 0.0], [0.0, 5.0, 1.0]],
            [[0.5, 4.0, 0.0], [0.0, 1.0, 5.0]],
        ]
    )
    all_rows = []
    all_rows.extend(
        records_from_logits(strong, labels, subset_name="strong_only", modalities=("gps", "mmwave"))
    )
    all_rows.extend(
        records_from_logits(plus_image, labels, subset_name="strong_plus_image", modalities=("image", "gps", "mmwave"))
    )
    all_rows.extend(
        records_from_logits(plus_image, labels, subset_name="all", modalities=("image", "radar", "gps", "lidar", "mmwave"))
    )
    subset_frame = pd.DataFrame(all_rows)
    delta = compute_marginal_deltas(subset_frame, weak_modalities=("image",))
    assert delta["delta_ce"].mean() > 0
    assert delta["delta_top1"].mean() > 0

    subset_write = write_table(subset_frame, tmp_path, "subset_predictions")
    delta_write = write_table(delta, tmp_path, "conditional_utility_per_sample_delta")
    oracle_summary, oracle_rows = compute_subset_oracle(
        subset_frame,
        candidates=("strong_only", "strong_plus_image", "all"),
    )
    features = pd.DataFrame(
        {
            "sample_id": delta["sample_id"],
            "dataset_index": delta["dataset_index"],
            "horizon_idx": delta["horizon_idx"],
            "horizon_name": delta["horizon_name"],
            "mmwave_entropy": [0.1, 0.2, 0.3, 0.4],
            "beam_transition": [0, 1, 0, 1],
        }
    )
    thresholds = fit_bucket_thresholds(features, ["mmwave_entropy", "beam_transition"], quantiles=(0.5,), bucket_names=("low", "high"))
    bucketed = assign_buckets(features, thresholds)
    bucket = compute_bucket_summary(delta, bucketed, oracle_choices=oracle_rows, min_samples=1)
    bucket_path = tmp_path / "conditional_utility_by_bucket.csv"
    bucket.to_csv(bucket_path, index=False)
    summary = build_conditional_utility_summary(
        run_name="dummy",
        scene="scene32",
        num_samples=2,
        horizons=["t+1", "t+2"],
        aggregate_metrics=aggregate_subset_metrics(subset_frame),
        deltas=delta,
        oracle_summary=oracle_summary,
        teacher_summary={},
        bucket_summary=bucket,
        metadata={"table_outputs": {"subset_predictions": subset_write, "delta": delta_write}},
        diagnosis_thresholds={"min_bucket_samples": 1},
    )
    summary_path = write_json(summary, tmp_path / "conditional_utility_summary.json")
    write_json(oracle_summary, tmp_path / "oracle_subset_summary.json")

    assert Path(subset_write["path"]).exists()
    assert Path(delta_write["path"]).exists()
    assert bucket_path.exists()
    assert summary_path.exists()
    assert (tmp_path / "oracle_subset_summary.json").exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["run_name"] == "dummy"
