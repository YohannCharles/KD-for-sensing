from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from kd_sensing.diagnostics.jepa_benchmark_aggregation import (
    aggregate_robustness_summary,
    aggregate_shortcut_reliance,
)
from kd_sensing.diagnostics.jepa_benchmark_artifacts import _write_csv


@dataclass
class CoreBenchmarkTables:
    robustness_rows: list[dict[str, Any]]
    shortcut_rows: list[dict[str, Any]]


def write_core_benchmark_tables(
    metrics_rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    *,
    tables_dir: Path,
) -> CoreBenchmarkTables:
    robustness_rows = aggregate_robustness_summary(
        metrics_rows,
        primary_metric=str(manifest["metrics"]["primary"]),
    )
    shortcut_rows = aggregate_shortcut_reliance(metrics_rows, robustness_rows, manifest)
    _write_csv(tables_dir / "metrics_by_condition.csv", metrics_rows)
    _write_csv(tables_dir / "robustness_summary.csv", robustness_rows)
    _write_csv(tables_dir / "shortcut_reliance_summary.csv", shortcut_rows)
    return CoreBenchmarkTables(robustness_rows=robustness_rows, shortcut_rows=shortcut_rows)


__all__ = ["CoreBenchmarkTables", "write_core_benchmark_tables"]
