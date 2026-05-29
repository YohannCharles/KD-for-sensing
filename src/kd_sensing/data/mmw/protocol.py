from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


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


__all__ = [
    "MMWFold",
    "build_mmw_folds",
    "load_mmw_data_availability",
    "mmw_scene_csv_names",
    "mmw_scene_data_roots",
    "ready_mmw_entries",
]
