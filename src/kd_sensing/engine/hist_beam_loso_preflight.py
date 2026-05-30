from __future__ import annotations

from pathlib import Path
from typing import Any

from kd_sensing.preprocessing.mmw_radar import materialize_mmw_radar_split_csv


def ensure_mmw_radar_csv_for_preflight(data_root: Path, csv_path: Path, scene: str) -> Path:
    result = materialize_mmw_radar_split_csv(data_root, csv_path, scene, require_maps=True)
    return Path(result["path"])


def preflight_error(
    scene: Any,
    resource_type: str,
    path: str | None,
    message: str,
    runs: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    return {
        "scene": scene,
        "resource_type": resource_type,
        "path": path,
        "message": message,
        "runs": [_run_identity(run) for run in (runs or [])],
    }


def _run_identity(run: dict[str, Any]) -> dict[str, Any]:
    identity = {
        "fold": run.get("fold"),
        "target_scene": run.get("target_scene"),
        "source_scenes": list(run.get("source_scenes", [])),
        "variant": run.get("variant"),
        "budget": run.get("budget"),
        "seed": run.get("seed"),
    }
    for key in (
        "dataset_family",
        "scene_family",
        "condition",
        "town",
        "protocol",
        "claim_scope",
        "cross_scene_claim_allowed",
        "profile",
        "modality_profile",
        "enabled_modalities",
        "excluded_sensitive_fields",
        "matrix_scope",
        "quick_validation",
    ):
        if run.get(key) is not None:
            identity[key] = run.get(key)
    return identity


__all__ = ["ensure_mmw_radar_csv_for_preflight", "preflight_error"]
