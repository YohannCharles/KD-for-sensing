from pathlib import Path
from typing import Any, Callable, Mapping

from kd_sensing.diagnostics.jepa_benchmark_common import BenchmarkManifestError, WarningRecord
from kd_sensing.diagnostics.jepa_benchmark_manifest import evaluate_model_comparability


MetricSourceLoader = Callable[..., dict[str, Any]]
MetricRowsBuilder = Callable[..., list[dict[str, Any]]]


def evaluate_comparability_or_warn(
    manifest: Mapping[str, Any],
    *,
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    comparability = evaluate_model_comparability(manifest)
    if comparability["status"] == "failed" and str(manifest.get("comparability", {}).get("mode")) == "strict":
        fields = ", ".join(str(item["field"]) for item in comparability["inconsistent_fields"])
        raise BenchmarkManifestError(f"Benchmark models are not comparable under strict mode: {fields}")
    if comparability["status"] != "passed":
        warnings.append(
            WarningRecord(
                code="comparability_marked_unavailable",
                message="One or more declared comparability fields differ across models.",
            ).to_dict()
        )
    return comparability


def collect_model_metric_rows(
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    dry_run: bool,
    warnings: list[dict[str, Any]],
    comparability_status: str,
    metric_source_loader: MetricSourceLoader,
    metric_rows_builder: MetricRowsBuilder,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    model_summaries: dict[str, dict[str, Any]] = {}
    metrics_rows: list[dict[str, Any]] = []
    for model_name, model_spec in manifest["models"].items():
        source = metric_source_loader(
            model_name,
            model_spec,
            manifest,
            output_dir=output_dir,
            dry_run=dry_run,
            warnings=warnings,
        )
        model_summaries[model_name] = source
        metrics_rows.extend(
            metric_rows_builder(
                model_name,
                model_spec,
                source,
                manifest,
                comparability_status=comparability_status,
                dry_run=dry_run,
            )
        )
    return model_summaries, metrics_rows


__all__ = ["collect_model_metric_rows", "evaluate_comparability_or_warn"]
