import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from kd_sensing.data.target_shot_splits import (
    TargetShotSplitConfig,
    build_target_shot_split,
    read_manifest_rows,
    write_target_shot_artifact,
)


@dataclass(frozen=True)
class MMWFold:
    fold_id: str
    target_scene: str
    source_scenes: tuple[str, ...]
    condition: str
    town: str
    protocol: str = "mmw_scenario_loso"
    claim_scope: str = "scenario_loso"
    cross_scene_claim_allowed: bool = True

    def metadata(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "target_scene": self.target_scene,
            "source_scenes": list(self.source_scenes),
            "dataset_family": "MMW",
            "scene_family": "MMW",
            "condition": self.condition,
            "town": self.town,
            "protocol": self.protocol,
            "claim_scope": self.claim_scope,
            "cross_scene_claim_allowed": self.cross_scene_claim_allowed,
        }


def load_mmw_data_availability(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path or "dataset/MMW/data_availability.json")
    if not source.exists():
        return {
            "dataset_family": "MMW",
            "ready_scenario_count": 0,
            "claim_scope": "unavailable",
            "cross_scene_claim_allowed": False,
            "entries": [],
            "unavailable_reason": f"availability_metadata_missing:{source}",
        }
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload.setdefault("entries", [])
    return payload


def ready_mmw_entries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = payload.get("entries", [])
    ready = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", ""))
        if status in {"ready_for_loso", "single_scene_ready", "ready"} and int(entry.get("window_count", 0) or 0) > 0:
            ready.append(dict(entry))
    return ready


def build_mmw_folds(payload: Mapping[str, Any], *, protocol: str = "scenario_loso") -> list[MMWFold]:
    ready = ready_mmw_entries(payload)
    if len(ready) < 2:
        if not ready:
            return []
        entry = ready[0]
        scenario = str(entry["scenario"])
        return [
            MMWFold(
                fold_id=f"smoke_{scenario}",
                target_scene=scenario,
                source_scenes=(scenario,),
                condition=str(entry.get("condition", payload.get("condition", "sunny"))),
                town=str(entry.get("town", "Town10")),
                protocol="mmw_single_scene_smoke",
                claim_scope="single_scene_smoke",
                cross_scene_claim_allowed=False,
            )
        ]
    folds = []
    for entry in ready:
        target = str(entry["scenario"])
        sources = tuple(str(item["scenario"]) for item in ready if str(item["scenario"]) != target)
        folds.append(
            MMWFold(
                fold_id=f"target_{target}",
                target_scene=target,
                source_scenes=sources,
                condition=str(entry.get("condition", payload.get("condition", "sunny"))),
                town=str(entry.get("town", "Town10")),
                protocol=f"mmw_{protocol}",
                claim_scope="scenario_loso",
                cross_scene_claim_allowed=True,
            )
        )
    return folds


def mmw_scene_data_roots(payload: Mapping[str, Any]) -> dict[str, str]:
    roots: dict[str, str] = {}
    for entry in ready_mmw_entries(payload):
        scenario = str(entry.get("scenario", ""))
        prepared = Path(str(entry.get("prepared_root", "")))
        if not scenario or not prepared:
            continue
        condition_root = prepared.parents[1] if len(prepared.parents) > 1 else prepared
        roots[scenario] = str(condition_root)
    return roots


def mmw_scene_csv_names(payload: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    csv_names: dict[str, dict[str, str]] = {}
    for entry in ready_mmw_entries(payload):
        scenario = str(entry.get("scenario", ""))
        if not scenario:
            continue
        csv_names[scenario] = {
            "train_csv_name": f"Prepared/{scenario}/splits/train.csv",
            "test_csv_name": f"Prepared/{scenario}/splits/test.csv",
        }
    return csv_names


def build_mmw_target_shot_split_artifact(
    *,
    csv_path: str | Path,
    config: TargetShotSplitConfig | Mapping[str, Any],
    output_path: str | Path,
    leakage_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = read_manifest_rows(csv_path, dataset_type="mmw")
    for row in rows:
        row.setdefault("scenario", row.get("sensor_scenario", row.get("scene_slug", "")))
        row.setdefault("weather", row.get("condition", ""))
        row.setdefault("beam_abs", row.get("beam_label", row.get("target_beam", "")))
        row.setdefault("beam_geo_source", _mmw_beam_geo_source(row))
    artifact = build_target_shot_split(rows, config, dataset_type="mmw", leakage_metadata=leakage_metadata)
    artifact["mmw_protocol"] = {
        "source_csv": str(csv_path),
        "beam_geo_source": _first_nonempty(row.get("beam_geo_source") for row in rows) or "unavailable",
        "geometry_unavailable_reason": _first_nonempty(row.get("geometry_unavailable_reason") for row in rows),
        "target_label_fraction": artifact.get("config_summary", {}).get("target_label_fraction"),
        "target_labeled_selected_sample_ids": artifact.get("sampling_manifest", {}).get("selected_sample_ids", []),
        "oracle_boundary": (
            "target_labeled beam/residual supervision is legal; target_unlabeled and target_test oracle fields "
            "remain excluded from adaptation, calibration, prototype update, temperature fitting and early stopping."
        ),
    }
    outputs = write_target_shot_artifact(artifact, output_path)
    artifact["artifact_paths"] = outputs
    return artifact


def _mmw_beam_geo_source(row: Mapping[str, Any]) -> str:
    if row.get("relative_geometry_json") or row.get("relative_azimuth") not in (None, ""):
        return "direct_relative_geometry"
    if row.get("beam_geo") not in (None, ""):
        return "codebook_mapping"
    return "unavailable"


def _first_nonempty(values) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


__all__ = [
    "MMWFold",
    "build_mmw_target_shot_split_artifact",
    "build_mmw_folds",
    "load_mmw_data_availability",
    "mmw_scene_csv_names",
    "mmw_scene_data_roots",
    "ready_mmw_entries",
]
