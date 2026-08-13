from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.analyze_mmw_cluster_bootstrap import (
    PROBE_METHODS,
    analyze,
    cluster_bootstrap,
    _load_ledger,
    _load_matrix,
)


def _write_synthetic_matrix(path: Path) -> None:
    sample_ids = []
    patterns = []
    labels = []
    groups = []
    domains = []
    for sample in range(16):
        for pattern in range(15):
            sample_ids.append(f"sample-{sample}")
            patterns.append(f"pattern-{pattern}")
            labels.append((sample + pattern) % 64)
            groups.append(f"trajectory-{sample}")
            domains.append(f"domain-{sample % 15}")
    torch.save(
        {
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_id": sample_ids,
            "pattern": patterns,
            "group_id": groups,
            "domain": domains,
        },
        path,
    )


def _write_synthetic_ledger(path: Path, *, drop: tuple[str, str] | None = None) -> None:
    fields = [
        "sample_id",
        "missing_pattern",
        "gt_beam",
        "method",
        "correct",
        "normalized_gain",
    ]
    rows = []
    for sample in range(16):
        for pattern in range(15):
            key = (f"sample-{sample}", f"pattern-{pattern}")
            if key == drop:
                continue
            gt = (sample + pattern) % 64
            for method_index, method in enumerate(PROBE_METHODS):
                rows.append(
                    {
                        "sample_id": key[0],
                        "missing_pattern": key[1],
                        "gt_beam": str(gt),
                        "method": method,
                        "correct": str(float(method_index == 0 or sample % 2 == 0)),
                        "normalized_gain": str(0.5 + 0.1 * method_index),
                    }
                )
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_cluster_bootstrap_resamples_whole_clusters_and_is_reproducible() -> None:
    values = np.asarray([1.0, 1.0, 1.0, 9.0])
    clusters = np.asarray(["a", "a", "a", "b"])
    first = cluster_bootstrap(values, clusters, replicates=200, seed=20260813)
    second = cluster_bootstrap(values, clusters, replicates=200, seed=20260813)
    assert first == second
    assert first["point_estimate"] == pytest.approx(3.0)
    assert first["cluster_macro_point"] == pytest.approx(5.0)
    assert first["rows"] == 4
    assert first["clusters"] == 2
    assert first["ci_low"] <= first["point_estimate"] <= first["ci_high"]


def test_ledger_fails_closed_on_missing_key(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.pt"
    ledger_path = tmp_path / "ledger.csv.gz"
    _write_synthetic_matrix(matrix_path)
    _write_synthetic_ledger(ledger_path, drop=("sample-3", "pattern-4"))
    matrix = _load_matrix(matrix_path, expected_rows=240, expected_samples=16, expected_patterns=15)
    with pytest.raises(ValueError, match="禁止静默 inner drop"):
        _load_ledger(ledger_path, matrix)


def test_analyze_writes_two_cluster_units_and_three_seed_summary(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.pt"
    ledger_path = tmp_path / "ledger.csv.gz"
    output_dir = tmp_path / "report"
    _write_synthetic_matrix(matrix_path)
    _write_synthetic_ledger(ledger_path)
    paths = {
        kind: [(ledger_path, matrix_path)] * 3
        for kind in ("prototype_only", "hard", "rmbp")
    }
    manifest = analyze(
        paths=paths,
        output_dir=output_dir,
        replicates=50,
        bootstrap_seed=20260813,
        expected_rows=240,
        expected_samples=16,
        expected_patterns=15,
    )
    assert manifest["result_count"] == 72
    assert manifest["summary_count"] == 24
    assert {row["cluster_unit"] for row in manifest["results"]} == {"trajectory", "domain"}
    assert {row["metric"] for row in manifest["results"]} == {"correct", "normalized_gain"}
    assert (output_dir / "paired_cluster_bootstrap.csv").is_file()
    assert (output_dir / "paired_cluster_bootstrap_summary.csv").is_file()
    assert (output_dir / "paired_cluster_bootstrap.md").is_file()
    saved = json.loads((output_dir / "paired_cluster_bootstrap.json").read_text(encoding="utf-8"))
    assert saved["claim_ineligible"] is True
    assert saved["outer_test_accessed"] is False
    assert saved["test_sealed"] is True
    assert saved["bootstrap"] == {"replicates": 50, "seed": 20260813, "ci": "percentile_95"}
