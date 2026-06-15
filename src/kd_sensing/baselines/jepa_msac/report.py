from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from kd_sensing.utils.runtime_output_layout import PARTITION_ANALYSIS


DEFAULT_REPORT_ROOT = Path("outputs") / PARTITION_ANALYSIS / "jepa_msac"


def write_report(report: Mapping[str, Any], output_dir: str | Path | None = None) -> dict[str, str]:
    root = Path(output_dir) if output_dir is not None else DEFAULT_REPORT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": root / "jepa_msac_report.json",
        "markdown": root / "jepa_msac_report.md",
        "csv": root / "jepa_msac_summary.csv",
    }
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    _write_summary_csv(report, paths["csv"])
    return {key: str(value) for key, value in paths.items()}


def write_ablation_manifest(rows: Sequence[Mapping[str, Any]], output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(output_dir) if output_dir is not None else DEFAULT_REPORT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "workflow_family": "jepa_msac",
        "rows": [dict(row) for row in rows],
        "result_table_rows": [dict(row) for row in rows if str(row.get("run_status")) == "complete"],
    }
    path = root / "ablation_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"path": str(path), "row_count": len(rows), "result_row_count": len(payload["result_table_rows"])}


def build_ablation_row(
    *,
    config_path: str,
    checkpoint_provenance: str | None,
    run_status: str,
    metrics_path: str | None = None,
    claim_status: str = "unverified",
    caveat: str = "Not run as paper-aligned long experiment.",
    latent_dim: int = 64,
    mask_ratio: float = 0.5,
    mask_pattern: str = "random",
    modality_ablation: str = "none",
    training_mode: str = "frozen-head",
    loc_aux: bool = True,
    missing_history: str = "none",
) -> dict[str, Any]:
    return {
        "config_path": str(config_path),
        "checkpoint_provenance": checkpoint_provenance,
        "run_status": str(run_status),
        "metrics_path": metrics_path,
        "claim_status": str(claim_status),
        "caveat": str(caveat),
        "latent_dim": int(latent_dim),
        "mask_ratio": float(mask_ratio),
        "mask_pattern": str(mask_pattern),
        "modality_ablation": str(modality_ablation),
        "training_mode": str(training_mode),
        "localization_auxiliary": bool(loc_aux),
        "missing_history": str(missing_history),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# JEPA-MSAC Reproduction Report",
        "",
        f"- Claim status: `{report.get('claim_status', 'unverified')}`",
        f"- Stage: `{report.get('stage', 'unknown')}`",
        f"- Output boundary: `{report.get('output_dir', 'outputs/analysis/jepa_msac')}`",
        "",
        "## Caveats",
    ]
    caveats = report.get("caveats", ["No paper-aligned long training has been recorded."])
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(["", "## Metrics"])
    metrics = report.get("metrics", {})
    for group, payload in metrics.items() if isinstance(metrics, Mapping) else []:
        lines.append(f"- `{group}`: {json.dumps(payload, sort_keys=True)}")
    lines.append("")
    return "\n".join(lines)


def _write_summary_csv(report: Mapping[str, Any], path: Path) -> None:
    rows = []
    metrics = report.get("metrics", {})
    if isinstance(metrics, Mapping):
        for group, payload in metrics.items():
            if isinstance(payload, Mapping):
                rows.append({"metric_group": group, "available": payload.get("available"), "value": payload.get("value"), "reason": payload.get("reason")})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric_group", "available", "value", "reason"])
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["DEFAULT_REPORT_ROOT", "build_ablation_row", "write_ablation_manifest", "write_report"]
