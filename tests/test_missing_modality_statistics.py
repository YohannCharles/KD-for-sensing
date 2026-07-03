import csv

from kd_sensing.diagnostics.missing_modality_statistics import build_statistical_summary, read_metric_rows


def test_statistical_summary_reads_scene31_rows_and_paired_delta(tmp_path):
    path = tmp_path / "eval_matrix.csv"
    _write_csv(
        path,
        [
            _row("baseline", 1, "full", 0.60),
            _row("baseline", 1, "missing_gps", 0.40),
            _row("baseline", 2, "full", 0.62),
            _row("baseline", 2, "missing_gps", 0.41),
            _row("candidate", 1, "full", 0.66),
            _row("candidate", 1, "missing_gps", 0.45),
            _row("candidate", 2, "full", 0.67),
            _row("candidate", 2, "missing_gps", 0.47),
        ],
    )

    result = build_statistical_summary(
        read_metric_rows(path),
        baseline_method="baseline",
        candidate_method="candidate",
        bootstrap_iterations=128,
        bootstrap_seed=7,
    )

    candidate = next(row for row in result["summary_rows"] if row["method"] == "candidate" and row["metric"] == "top1")
    assert candidate["seed_count"] == 2
    assert candidate["pattern_count"] == 2
    assert candidate["ci_status"] == "available"
    assert result["paired_comparison"]["status"] == "available"
    assert result["paired_comparison"]["win_count"] == 4
    assert result["paired_comparison"]["loss_count"] == 0
    assert result["paired_comparison"]["paired_delta_mean"] > 0
    assert result["claim_gate"]["statistical_claim_ready"] is True


def test_statistical_summary_warns_for_single_seed_and_missing_metric(tmp_path):
    path = tmp_path / "single_seed.csv"
    _write_csv(
        path,
        [
            _row("candidate", 1, "full", 0.66),
            {
                "method": "candidate",
                "seed": "1",
                "pattern": "missing_gps",
                "comparability_status": "strict",
            },
        ],
    )

    result = build_statistical_summary(
        read_metric_rows(path),
        candidate_method="candidate",
        min_seed_count=2,
        bootstrap_iterations=64,
    )

    top1 = next(row for row in result["summary_rows"] if row["metric"] == "top1")
    assert top1["std_status"] == "unavailable"
    assert top1["ci_status"] == "unavailable"
    assert result["claim_gate"]["statistical_claim_ready"] is False
    codes = {warning["code"] for warning in result["warnings"]}
    assert {"metric_missing", "insufficient_seed_count", "ci_unavailable", "claim_gate_not_ready"} <= codes


def test_paired_comparison_requires_matching_seed_split_pattern_keys():
    result = build_statistical_summary(
        [
            _row("baseline", 1, "full", 0.6),
            _row("candidate", 2, "full", 0.7),
        ],
        baseline_method="baseline",
        candidate_method="candidate",
    )

    assert result["paired_comparison"]["status"] == "unavailable"
    assert any(warning["code"] == "paired_keys_unavailable" for warning in result["warnings"])


def test_fresh_eval_wide_summary_rows_are_supported():
    result = build_statistical_summary(
        [
            {
                "run_name": "proto_seed1",
                "method": "proto",
                "seed": 1,
                "full": 0.5,
                "avg_missing": 0.4,
                "radar_only": 0.3,
                "comparability_status": "strict",
            },
            {
                "run_name": "proto_seed2",
                "method": "proto",
                "seed": 2,
                "full": 0.6,
                "avg_missing": 0.5,
                "radar_only": 0.4,
                "comparability_status": "strict",
            },
        ],
        candidate_method="proto",
        bootstrap_iterations=64,
    )

    row = next(item for item in result["summary_rows"] if item["method"] == "proto" and item["metric"] == "top1")
    assert row["count"] == 6
    assert row["seed_count"] == 2
    assert row["pattern_count"] == 3


def _row(method: str, seed: int, pattern: str, top1: float) -> dict[str, object]:
    return {
        "method": method,
        "seed": seed,
        "split": "test",
        "pattern": pattern,
        "metric_profile": "topk_dba",
        "label_space": "beam64",
        "comparability_status": "strict",
        "top1": top1,
    }


def _write_csv(path, rows):
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
