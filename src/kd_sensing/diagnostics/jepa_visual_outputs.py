from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class VisualArtifactResults:
    comparison_rows: list[dict[str, Any]] = field(default_factory=list)
    case_rows: list[dict[str, Any]] = field(default_factory=list)
    robustness_rows: list[dict[str, Any]] = field(default_factory=list)
    embedding_neighbor_rows: list[dict[str, Any]] = field(default_factory=list)
    attention_faithfulness_context: dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    evidence_context: dict[str, Any] = field(default_factory=lambda: {"enabled": False})


def write_benchmark_analysis_artifacts(
    cfg: Mapping[str, Any],
    *,
    tables_dir: Path,
    figures_dir: Path,
    payload_dir: Path,
    registry: Any,
    warnings: list[str],
) -> dict[str, Any]:
    from kd_sensing.diagnostics import jepa_visual_analysis as jva

    return jva._write_benchmark_analysis_outputs(cfg, tables_dir, figures_dir, payload_dir, registry, warnings)


def write_model_analysis_artifacts(
    cfg: Mapping[str, Any],
    *,
    output_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
    payload_dir: Path,
    analyses: Mapping[str, Any],
    registry: Any,
    warnings: list[str],
    command: list[str] | None,
) -> VisualArtifactResults:
    from kd_sensing.diagnostics import jepa_visual_analysis as jva

    result = VisualArtifactResults()
    if analyses:
        jva._write_model_metrics(tables_dir / "model_metrics.csv", analyses)
        if bool(cfg.get("figures", {}).get("error_anatomy", True)):
            jva._write_error_anatomy_outputs(figures_dir, tables_dir, analyses, cfg, registry, warnings)
        result.comparison_rows = jva._write_comparison_outputs(tables_dir, analyses, cfg, warnings)
        result.embedding_neighbor_rows = jva._write_embedding_outputs(
            figures_dir,
            tables_dir,
            analyses,
            cfg,
            registry,
            warnings,
        )
        jva._write_attention_outputs(figures_dir, tables_dir, analyses, cfg, registry, warnings)
        result.attention_faithfulness_context = jva._write_attention_faithfulness_outputs(
            figures_dir,
            tables_dir,
            analyses,
            cfg,
            registry,
            warnings,
        )
        if bool(cfg.get("figures", {}).get("case_studies", True)):
            result.case_rows = jva._write_case_study_outputs(
                figures_dir,
                payload_dir,
                tables_dir,
                result.comparison_rows,
                analyses,
                cfg,
                registry,
                warnings,
            )
        if bool(cfg.get("figures", {}).get("robustness", True)):
            result.robustness_rows = jva._write_robustness_outputs(
                figures_dir,
                tables_dir,
                analyses,
                cfg,
                registry,
                warnings,
            )
    else:
        registry.skipped_output(tables_dir / "model_metrics.csv", reason="no_completed_models", kind="table")
        registry.skipped_output(tables_dir / "comparison_samples.csv", reason="no_completed_models", kind="table")

    if jva.gps_query_evidence_enabled(cfg):
        result.evidence_context = jva.write_gps_query_evidence_package(
            cfg,
            output_dir=output_dir,
            analyses=analyses,
            comparison_rows=result.comparison_rows,
            command=command,
            warnings=warnings,
            formats=jva._output_formats(cfg),
            dpi=int(cfg.get("outputs", {}).get("dpi", 180)),
            attention_faithfulness=result.attention_faithfulness_context,
        )
    return result


__all__ = ["VisualArtifactResults", "write_benchmark_analysis_artifacts", "write_model_analysis_artifacts"]
