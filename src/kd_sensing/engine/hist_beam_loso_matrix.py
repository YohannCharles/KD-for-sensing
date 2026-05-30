from __future__ import annotations

from typing import Any


def matrix_summary(plan: dict[str, Any]) -> dict[str, Any]:
    runs = list(plan.get("runs", []))
    return {
        "target_scenes": sorted(
            {_matrix_scene_value(run["target_scene"]) for run in runs if run.get("target_scene") is not None},
            key=str,
        ),
        "variants": sorted({str(run["variant"]) for run in runs if run.get("variant") is not None}),
        "budgets": sorted({int(run["budget"]) for run in runs if run.get("budget") is not None}),
        "seeds": sorted({int(run["seed"]) for run in runs if run.get("seed") is not None}),
        "run_count": len(runs),
        "profile": plan.get("profile") or plan.get("matrix_profile"),
        "matrix_scope": plan.get("matrix_scope"),
        "quick_validation": plan.get("quick_validation"),
        "modality_profile": plan.get("modality_profile"),
    }


def _matrix_scene_value(scene: Any) -> Any:
    try:
        return int(scene)
    except (TypeError, ValueError):
        return str(scene)


__all__ = ["matrix_summary"]
