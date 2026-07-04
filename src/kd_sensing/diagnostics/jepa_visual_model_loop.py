from pathlib import Path
from typing import Any, Callable, Mapping


AnalyzeModel = Callable[..., Any]


def run_model_analysis_loop(
    model_specs: Mapping[str, Mapping[str, Any]],
    cfg: Mapping[str, Any],
    *,
    tables_dir: Path,
    cache_dir: Path,
    warnings: list[str],
    dry_run: bool,
    analyzer: AnalyzeModel,
) -> tuple[dict[str, Any], dict[str, str]]:
    analyses: dict[str, Any] = {}
    model_failures: dict[str, str] = {}
    if dry_run:
        warnings.append("dry_run:no_model_forward_executed")
        return analyses, model_failures
    for model_name, model_spec in model_specs.items():
        try:
            analysis = analyzer(
                model_name,
                model_spec,
                cfg,
                tables_dir=tables_dir,
                cache_dir=cache_dir,
                warnings=warnings,
            )
        except Exception as exc:
            if not bool(model_spec.get("optional", False)):
                raise
            message = f"model_failed:{model_name}:{exc}"
            warnings.append(message)
            model_failures[model_name] = str(exc)
            continue
        analyses[model_name] = analysis
        warnings.extend(analysis.warnings)
    return analyses, model_failures


__all__ = ["run_model_analysis_loop"]
