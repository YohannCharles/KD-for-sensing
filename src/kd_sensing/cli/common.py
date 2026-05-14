from __future__ import annotations

import argparse
import json
from typing import Iterable

from kd_sensing.config import load_config
from kd_sensing.data.scenes import DEEPSENSE_SCENES, resolve_deepsense_scene

DEFAULT_COMPARE_SCENES = (9, 32)


def collect_overrides(namespace: argparse.Namespace, unknown: Iterable[str]) -> list[str]:
    overrides = []
    overrides.extend(_scene_selection_overrides(getattr(namespace, "scenes", None)))
    for item in getattr(namespace, "override", []) or []:
        overrides.append(item)
    overrides.extend(item for item in unknown if "=" in item)
    return overrides


def print_result(result: dict) -> None:
    print(json.dumps(result, indent=2))


def load_cli_config(args: argparse.Namespace, unknown: Iterable[str]) -> dict:
    return load_config(args.config, collect_overrides(args, unknown))


def _scene_selection_overrides(raw: str | None) -> list[str]:
    if raw is None or not str(raw).strip():
        return []
    scenes = _parse_scene_selection(raw)
    if len(scenes) == 1:
        return [
            f"data.dataset.scene={scenes[0]}",
            "diagnostics.visualization.compare_scenes=null",
        ]
    return [f"diagnostics.visualization.compare_scenes={json.dumps(scenes)}"]


def _parse_scene_selection(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        if token.lower() == "all":
            values.extend(scene for scene in DEFAULT_COMPARE_SCENES if scene in DEEPSENSE_SCENES)
            continue
        values.append(resolve_deepsense_scene(token).scene_id)
    if not values:
        raise ValueError("--scenes must include at least one scene, for example --scenes 9,32.")
    unique: list[int] = []
    for scene_id in values:
        if scene_id not in unique:
            unique.append(scene_id)
    return unique
