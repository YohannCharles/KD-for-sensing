from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from tools.visualization.viewer_constants import (
    LOW_QUALITY_MEAN_THRESHOLD,
    LOW_QUALITY_MODALITY_THRESHOLD,
    SHOW_MODES,
)

def load_manifest(manifest_path: str | Path, project_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Load a viewer manifest from a JSON array, {"samples": [...]}, or JSONL file."""

    path = Path(manifest_path).expanduser()
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    samples: list[dict[str, Any]]
    first = text[0]
    if first in "[{":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
            samples = [item for item in payload["samples"] if isinstance(item, dict)]
        elif isinstance(payload, list):
            samples = [item for item in payload if isinstance(item, dict)]
        else:
            samples = _load_jsonl(text)
    else:
        samples = _load_jsonl(text)

    manifest_dir = path.parent
    root = Path(project_root).expanduser() if project_root is not None else _find_project_root(manifest_dir)
    for index, sample in enumerate(samples):
        sample.setdefault("_manifest_index", index)
        sample.setdefault("_global_index", index)
        sample.setdefault("_manifest_dir", str(manifest_dir))
        if root is not None:
            sample.setdefault("_project_root", str(root))
    return samples


def get_available_scenes(samples: Iterable[dict[str, Any]]) -> list[str]:
    values = []
    for sample in samples:
        scene = sample.get("scene_slug", sample.get("scene_id"))
        if scene is not None and str(scene).strip():
            values.append(str(scene))
    unique = sorted(set(values), key=_natural_key)
    return ["all", *unique] if unique else ["all"]


def get_available_splits(samples: Iterable[dict[str, Any]]) -> list[str]:
    values = [str(sample.get("split")) for sample in samples if sample.get("split") is not None]
    unique = sorted({value for value in values if value.strip()}, key=_natural_key)
    return ["all", *unique] if unique else ["all"]


def filter_samples(
    samples: Iterable[dict[str, Any]],
    scene: str | None = "all",
    split: str | None = "all",
    show_mode: str | None = "all",
) -> list[dict[str, Any]]:
    scene_filter = _none_if_all(scene)
    split_filter = _none_if_all(split)
    mode = str(show_mode or "all").strip().lower()
    if mode not in SHOW_MODES:
        mode = "all"

    filtered = []
    for sample in samples:
        if scene_filter is not None and not _sample_matches_scene(sample, scene_filter):
            continue
        if split_filter is not None and str(sample.get("split")) != split_filter:
            continue
        if mode == "correct only" and safe_get(sample, "prediction.correct") is not True:
            continue
        if mode == "wrong only" and safe_get(sample, "prediction.correct") is not False:
            continue
        if mode == "low quality only" and not _is_low_quality(sample.get("quality")):
            continue
        filtered.append(sample)
    return filtered


def safe_get(data: Any, path: str, default: Any = None) -> Any:
    current = data
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
            continue
        if isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return default
            current = current[index]
            continue
        return default
    return current


def resolve_path(
    path: str | Path | None,
    manifest_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path | None:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None

    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate

    bases = []
    if manifest_dir is not None:
        bases.append(Path(manifest_dir).expanduser())
    if project_root is not None:
        bases.append(Path(project_root).expanduser())
    for base in bases:
        resolved = base / candidate
        if resolved.exists():
            return resolved
    if bases:
        return bases[0] / candidate
    return candidate


def load_image_safe(
    path: str | Path | None,
    manifest_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Image.Image | None:
    resolved = resolve_path(path, manifest_dir=manifest_dir, project_root=project_root)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        return None
    try:
        with Image.open(resolved) as image:
            return image.convert("RGB").copy()
    except Exception:
        return None


def load_json_safe(
    path: str | Path | dict[str, Any] | list[Any] | None,
    manifest_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Any:
    if isinstance(path, (dict, list)):
        return path
    resolved = resolve_path(path, manifest_dir=manifest_dir, project_root=project_root)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        return None
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        return None



def manifest_context(sample: dict[str, Any]) -> tuple[str | None, str | None]:
    return sample.get("_manifest_dir"), sample.get("_project_root")


def clamp_index(index: Any, total: int) -> int:
    if total <= 0:
        return 0
    try:
        value = int(index)
    except (TypeError, ValueError):
        value = 0
    return max(0, min(total - 1, value))


def _load_jsonl(text: str) -> list[dict[str, Any]]:
    samples = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            samples.append(item)
    return samples


def _find_project_root(start: Path) -> Path | None:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
    return None


def _none_if_all(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "all"} else text


def _sample_matches_scene(sample: dict[str, Any], expected: str) -> bool:
    scene_values = [sample.get("scene_id"), sample.get("scene_slug")]
    return any(str(value) == expected for value in scene_values if value is not None)


def _is_low_quality(quality: Any) -> bool:
    if not isinstance(quality, dict):
        return False
    values = [value for _, value in _numeric_score_items(quality)]
    if not values:
        return False
    return any(value < LOW_QUALITY_MODALITY_THRESHOLD for value in values) or (
        float(np.mean(values)) < LOW_QUALITY_MEAN_THRESHOLD
    )


def _numeric_score_items(score_dict: dict[str, Any] | None) -> list[tuple[str, float]]:
    if not isinstance(score_dict, dict):
        return []
    items = []
    for key, value in score_dict.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            items.append((str(key), number))
    return items



def _looks_like_image_path(value: Any) -> bool:
    if not isinstance(value, (str, Path)):
        return False
    suffix = Path(str(value)).suffix.lower()
    return suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


def _natural_key(value: str) -> tuple[int, str]:
    text = str(value)
    digits = ""
    for char in reversed(text):
        if not char.isdigit():
            break
        digits = char + digits
    if digits:
        return (0, f"{text[: -len(digits)]}{int(digits):012d}")
    return (1, text)


__all__ = [
    "clamp_index",
    "filter_samples",
    "get_available_scenes",
    "get_available_splits",
    "load_image_safe",
    "load_json_safe",
    "load_manifest",
    "manifest_context",
    "resolve_path",
    "safe_get",
]
