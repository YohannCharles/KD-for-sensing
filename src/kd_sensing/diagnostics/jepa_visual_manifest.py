import json
from pathlib import Path
from typing import Any, Mapping


def write_visual_report_and_manifest(
    cfg: Mapping[str, Any],
    *,
    output_dir: Path,
    command: list[str] | None,
    analyses: Mapping[str, Any],
    warnings: list[str],
    model_failures: Mapping[str, str],
    dry_run: bool,
    registry: Any,
    benchmark_context: Mapping[str, Any],
    evidence_context: Mapping[str, Any],
    attention_faithfulness: Mapping[str, Any],
    comparison_rows: list[dict[str, Any]],
    embedding_neighbor_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    robustness_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    from kd_sensing.diagnostics import jepa_visual_analysis as jva

    manifest = jva._build_manifest(
        cfg,
        output_dir=output_dir,
        command=command,
        analyses=analyses,
        warnings=warnings,
        model_failures=model_failures,
        dry_run=dry_run,
        registry=registry,
        benchmark_context=benchmark_context,
        evidence_context=evidence_context,
        attention_faithfulness=attention_faithfulness,
    )
    report = jva._build_report(
        cfg,
        analyses=analyses,
        comparison_rows=comparison_rows,
        embedding_neighbor_rows=embedding_neighbor_rows,
        case_rows=case_rows,
        robustness_rows=robustness_rows,
        benchmark_context=benchmark_context,
        evidence_context=evidence_context,
        attention_faithfulness=attention_faithfulness,
        warnings=warnings,
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    manifest["outputs"] = registry.list_outputs()
    manifest["outputs"].append(
        {
            "path": "report.md",
            "kind": "report",
            "status": "generated",
            "size_bytes": int(report_path.stat().st_size),
        }
    )
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(jva._json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["outputs"].append(
        {
            "path": "analysis_manifest.json",
            "kind": "manifest",
            "status": "generated",
            "size_bytes": int(manifest_path.stat().st_size),
        }
    )
    manifest_path.write_text(json.dumps(jva._json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path, manifest_path


__all__ = ["write_visual_report_and_manifest"]
