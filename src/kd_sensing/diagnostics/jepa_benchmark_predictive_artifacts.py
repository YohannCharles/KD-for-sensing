import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from kd_sensing.data.difficulty.presets import PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE
from kd_sensing.diagnostics.jepa_benchmark_artifacts import _write_csv
from kd_sensing.diagnostics.jepa_benchmark_common import GPS_QUERY_ADVANTAGE_SLICE_TYPE, _json_ready
from kd_sensing.diagnostics.jepa_benchmark_manifest import _manifest_has_predictive_jepa
from kd_sensing.diagnostics.jepa_benchmark_predictive import aggregate_predictive_robustness_summary
from kd_sensing.diagnostics.jepa_benchmark_predictive_advantage_metrics import (
    aggregate_gps_query_advantage_margins,
    build_predictive_claim_gate,
    build_predictive_diagnostics_bundle_manifest,
    fusion_diagnostic_condition_rows,
    fusion_diagnostic_paired_margins,
    fusion_diagnostic_summary,
)


@dataclass
class PredictiveArtifactResults:
    predictive_summary: list[dict[str, Any]]
    paths: dict[str, Path | None]


def write_predictive_and_fusion_artifacts(
    metrics_rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    *,
    results_dir: Path,
    warnings: list[dict[str, Any]],
) -> PredictiveArtifactResults:
    paths: dict[str, Path | None] = {
        "predictive_condition_metrics": None,
        "predictive_regional_summary": None,
        "predictive_margin_vs_resnet": None,
        "predictive_warnings": None,
        "predictive_gps_query_advantage_metrics": None,
        "predictive_gps_query_advantage_margins": None,
        "predictive_claim_gate": None,
        "predictive_diagnostics_bundle_manifest": None,
        "fusion_diagnostic_condition_metrics": None,
        "fusion_diagnostic_paired_margins": None,
        "fusion_diagnostic_summary": None,
    }
    predictive_summary: list[dict[str, Any]] = []
    if _manifest_has_predictive_jepa(manifest):
        predictive_summary = _write_predictive_artifacts(
            metrics_rows,
            manifest,
            results_dir=results_dir,
            warnings=warnings,
            paths=paths,
        )
    _write_fusion_diagnostic_artifacts(metrics_rows, manifest, results_dir=results_dir, paths=paths)
    return PredictiveArtifactResults(predictive_summary=predictive_summary, paths=paths)


def _write_predictive_artifacts(
    metrics_rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    *,
    results_dir: Path,
    warnings: list[dict[str, Any]],
    paths: dict[str, Path | None],
) -> list[dict[str, Any]]:
    predictive_rows = [row for row in metrics_rows if str(row.get("suite_type")) == PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE]
    predictive_advantage_rows = [row for row in metrics_rows if str(row.get("suite_type")) == GPS_QUERY_ADVANTAGE_SLICE_TYPE]
    predictive_summary = aggregate_predictive_robustness_summary(
        metrics_rows,
        manifest,
        primary_metric=str(manifest["metrics"]["primary"]),
    )
    predictive_advantage_margins = aggregate_gps_query_advantage_margins(
        metrics_rows,
        manifest,
        primary_metric=str(manifest["metrics"]["primary"]),
    )
    predictive_claim_gate = build_predictive_claim_gate(
        predictive_summary,
        predictive_advantage_margins,
        manifest,
    )
    predictive_diagnostics_bundle = build_predictive_diagnostics_bundle_manifest(
        manifest,
        advantage_margins=predictive_advantage_margins,
    )
    paths["predictive_condition_metrics"] = results_dir / "predictive_condition_metrics.csv"
    paths["predictive_regional_summary"] = results_dir / "predictive_regional_summary.json"
    paths["predictive_margin_vs_resnet"] = results_dir / "predictive_margin_vs_resnet.json"
    paths["predictive_warnings"] = results_dir / "predictive_warnings.json"
    paths["predictive_gps_query_advantage_metrics"] = results_dir / "predictive_gps_query_advantage_metrics.csv"
    paths["predictive_gps_query_advantage_margins"] = results_dir / "predictive_gps_query_advantage_margins.json"
    paths["predictive_claim_gate"] = results_dir / "predictive_claim_gate.json"
    paths["predictive_diagnostics_bundle_manifest"] = results_dir / "predictive_diagnostics_bundle_manifest.json"
    _write_csv(paths["predictive_condition_metrics"], predictive_rows)
    _write_csv(paths["predictive_gps_query_advantage_metrics"], predictive_advantage_rows)
    paths["predictive_regional_summary"].write_text(
        json.dumps(_json_ready({"summary": predictive_summary}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["predictive_margin_vs_resnet"].write_text(
        json.dumps(
            _json_ready(
                {
                    "margins": [
                        {
                            key: row.get(key)
                            for key in (
                                "model",
                                "group",
                                "predictive_dba",
                                "clean_anchor_primary",
                                "S@drop<=0.02",
                                "S@drop<=0.05",
                                "AUC_retention",
                                "collapse_s",
                                "weakest_axis",
                                "resnet_predictive_dba",
                                "margin_vs_resnet_dba",
                                "claim_pass_5pt",
                                "claim_status",
                                "stress_summary_status",
                                "overall_cxd_dba",
                                "overall_cxd_delta_vs_resnet",
                            )
                        }
                        for row in predictive_summary
                    ]
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["predictive_warnings"].write_text(
        json.dumps(_json_ready({"warnings": warnings}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["predictive_gps_query_advantage_margins"].write_text(
        json.dumps(_json_ready({"margins": predictive_advantage_margins}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["predictive_claim_gate"].write_text(
        json.dumps(_json_ready({"claim_gate": predictive_claim_gate}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["predictive_diagnostics_bundle_manifest"].write_text(
        json.dumps(_json_ready(predictive_diagnostics_bundle), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return predictive_summary


def _write_fusion_diagnostic_artifacts(
    metrics_rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    *,
    results_dir: Path,
    paths: dict[str, Path | None],
) -> None:
    fusion_condition_rows = fusion_diagnostic_condition_rows(metrics_rows, manifest)
    if not fusion_condition_rows:
        return
    fusion_margin_rows = fusion_diagnostic_paired_margins(fusion_condition_rows, manifest)
    fusion_summary = fusion_diagnostic_summary(fusion_condition_rows, fusion_margin_rows, manifest)
    paths["fusion_diagnostic_condition_metrics"] = results_dir / "fusion_diagnostic_metrics.csv"
    paths["fusion_diagnostic_paired_margins"] = results_dir / "paired_margin_by_condition.csv"
    paths["fusion_diagnostic_summary"] = results_dir / "fusion_diagnostic_summary.json"
    _write_csv(paths["fusion_diagnostic_condition_metrics"], fusion_condition_rows)
    _write_csv(paths["fusion_diagnostic_paired_margins"], fusion_margin_rows)
    paths["fusion_diagnostic_summary"].write_text(
        json.dumps(_json_ready(fusion_summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = ["PredictiveArtifactResults", "write_predictive_and_fusion_artifacts"]
