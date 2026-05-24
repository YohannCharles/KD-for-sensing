from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from kd_sensing.diagnostics.complementarity_summaries import (
    _best_by_metric,
    _display_float,
    _display_rate,
    _json_ready,
)

def write_outputs(
    cases: pd.DataFrame,
    summary: dict[str, Any],
    bucket_summary: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, str]:
    target = Path(output_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    cases_path = target / "complementarity_cases.csv.gz"
    summary_path = target / "complementarity_summary.json"
    bucket_path = target / "complementarity_by_bucket.csv"
    report_path = target / "complementarity_report.md"

    cases.to_csv(cases_path, index=False, compression="gzip")
    bucket_summary.to_csv(bucket_path, index=False)
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    report_path.write_text(render_report(summary, bucket_summary), encoding="utf-8")
    return {
        "cases": str(cases_path),
        "summary": str(summary_path),
        "bucket_summary": str(bucket_path),
        "report": str(report_path),
    }


def render_report(summary: dict[str, Any], bucket_summary: pd.DataFrame) -> str:
    global_metrics = summary.get("global", {})
    by_weak = summary.get("by_weak_modality", {})
    by_pair = summary.get("by_strong_weak_pair", {})
    best_weak = _best_by_metric(by_weak, "complementarity_rate")
    best_pair = _best_by_metric(by_pair, "complementarity_rate")
    rescue = global_metrics.get("rescue_rate_given_complementary", {}).get("value")
    negative = global_metrics.get("negative_transfer_rate", {}).get("value")
    probability = summary.get("metadata", {}).get("probability_metrics_available")
    bucket_line = "Bucket statistics unavailable."
    if not bucket_summary.empty:
        top_bucket = bucket_summary.sort_values(
            ["complementarity_rate", "rescue_count"], ascending=False, na_position="last"
        ).head(1)
        if not top_bucket.empty:
            row = top_bucket.iloc[0]
            bucket_line = (
                f"Top bucket: {row['bucket_feature']}={row['bucket_name']} "
                f"for {row['weak_modality']} / {row['horizon_name']} "
                f"(complementarity_rate={_display_float(row['complementarity_rate'])})."
            )
    lines = [
        "# Weak Modality Complementarity Report",
        "",
        f"- Total cases: {summary.get('total_cases', 0)}",
        f"- Complementarity rate: {_display_rate(global_metrics.get('complementarity_rate'))}",
        f"- Best weak modality by complementarity rate: {best_weak or 'n/a'}",
    ]
    if by_pair:
        available_pairs = [
            pair for pair, payload in by_pair.items()
            if isinstance(payload, dict) and payload.get("fusion_metrics_available")
        ]
        lines.extend(
            [
                f"- Best strong/weak pair by complementarity rate: {best_pair or 'n/a'}",
                f"- Strong/weak pairs with fusion metrics: {len(available_pairs)}/{len(by_pair)}",
            ]
        )
    lines.extend(
        [
            f"- Rescue rate given complementary: {_display_float(rescue)}",
            f"- Negative transfer rate: {_display_float(negative)}",
            f"- Net fusion gain count: {global_metrics.get('net_fusion_gain_count', 0)}",
            f"- Probability metrics available: {bool(probability)}",
            f"- {bucket_line}",
            "",
            "Case semantics: `strong_wrong_weak_correct` marks potential local complementarity; "
            "`rescue` means fusion used that complementarity; `unused_complementary` means fusion failed to use it; "
            "`negative_transfer` means fusion broke a strong-only correct prediction.",
            "",
        ]
    )
    return "\n".join(lines)




__all__ = ["render_report", "write_outputs"]
