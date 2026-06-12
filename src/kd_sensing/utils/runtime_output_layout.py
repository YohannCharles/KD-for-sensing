from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

from kd_sensing.data.scenes import (
    is_deepsense_dataset_type,
    resolve_deepsense_scene,
)


OUTPUT_ROOT = "outputs"
PARTITION_CACHE = "cache"
PARTITION_CLEANUP_MANIFESTS = "cleanup_manifests"
PARTITION_ANALYSIS = "analysis"
PARTITION_VISUAL_ANALYSIS = "visual_analysis"
PARTITION_EVALUATIONS = "evaluations"
PARTITION_ARCHIVE = "archive"
PARTITION_FEATURES = "features"
PARTITION_TRAINING = "training"
SCENEGROUP_PREFIX = "scenegroup_"

CANONICAL_RUNTIME_PARTITIONS = (
    PARTITION_CACHE,
    PARTITION_CLEANUP_MANIFESTS,
    PARTITION_ANALYSIS,
    PARTITION_VISUAL_ANALYSIS,
    PARTITION_EVALUATIONS,
    PARTITION_ARCHIVE,
)
DEFAULT_NON_RUN_PARTITIONS = (PARTITION_CACHE, PARTITION_ARCHIVE, PARTITION_CLEANUP_MANIFESTS)
PROTECTED_MAINLINE_PARTITIONS = (
    PARTITION_ANALYSIS,
    PARTITION_VISUAL_ANALYSIS,
    PARTITION_CACHE,
    PARTITION_CLEANUP_MANIFESTS,
    PARTITION_EVALUATIONS,
    PARTITION_ARCHIVE,
    PARTITION_FEATURES,
    PARTITION_TRAINING,
)
SCENE_LIST_KEYS = ("train_scenes", "validation_scenes", "val_scenes", "test_scenes", "eval_scenes", "scenes")


@dataclass(frozen=True)
class RuntimeOutputScope:
    kind: str
    slug: str
    scene_ids: tuple[int, ...]
    role_scenes: dict[str, tuple[int, ...]]
    source: str

    @property
    def scene_slugs(self) -> tuple[str, ...]:
        return tuple(f"scene{scene_id}" for scene_id in self.scene_ids)

    def to_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "scope_kind": self.kind,
            "scope_slug": self.slug,
            "scene_scope": self.kind,
            "scene_ids": list(self.scene_ids),
            "scene_slugs": list(self.scene_slugs),
            "source": self.source,
        }
        for role, scenes in self.role_scenes.items():
            metadata[f"{role}_scenes"] = list(scenes)
            metadata[f"{role}_scene_slugs"] = [f"scene{scene_id}" for scene_id in scenes]
        if self.kind == "scene" and self.scene_ids:
            metadata["scene_id"] = self.scene_ids[0]
            metadata["scene_slug"] = self.slug
        elif self.kind == "scenegroup":
            metadata["scene_slug"] = self.slug
        return metadata


def canonical_runtime_partitions(outputs_root: str | Path = OUTPUT_ROOT) -> dict[str, str]:
    root = Path(outputs_root)
    return {
        PARTITION_CACHE: str(root / PARTITION_CACHE),
        PARTITION_CLEANUP_MANIFESTS: str(root / PARTITION_CLEANUP_MANIFESTS),
        PARTITION_ANALYSIS: str(root / PARTITION_ANALYSIS),
        PARTITION_VISUAL_ANALYSIS: str(root / PARTITION_VISUAL_ANALYSIS),
        PARTITION_EVALUATIONS: str(root / PARTITION_EVALUATIONS),
        PARTITION_ARCHIVE: str(root / PARTITION_ARCHIVE),
        "scene": str(root / "scene<id>"),
        "scenegroup": str(root / f"{SCENEGROUP_PREFIX}<range-or-list>"),
    }


def scenegroup_slug(scene_ids: Any) -> str:
    normalized = _normalize_scene_ids(_as_sequence(scene_ids))
    if len(normalized) < 2:
        raise ValueError("A scenegroup scope requires at least two DeepSense6G scenes.")
    if _is_contiguous(normalized):
        body = f"s{normalized[0]}_s{normalized[-1]}"
    else:
        body = "_".join(f"s{scene_id}" for scene_id in normalized)
    return f"{SCENEGROUP_PREFIX}{body}"


def runtime_output_scope_from_config(
    cfg: Mapping[str, Any],
    *,
    purpose: str = "training",
) -> RuntimeOutputScope | None:
    dataset_cfg = _dataset_cfg(cfg)
    if not dataset_cfg:
        return None
    dataset_type = dataset_cfg.get("type", "deepsense6g")
    if not is_deepsense_dataset_type(dataset_type):
        return _generic_scope_from_dataset(dataset_cfg)

    role_scenes = _role_scenes(dataset_cfg)
    scene_ids, source = _scope_scene_ids(dataset_cfg, role_scenes, purpose=purpose)
    if not scene_ids:
        return None
    if len(scene_ids) == 1:
        slug = resolve_deepsense_scene(scene_ids[0], dataset_type=dataset_type).scene_slug
        return RuntimeOutputScope(
            kind="scene",
            slug=slug,
            scene_ids=scene_ids,
            role_scenes=role_scenes,
            source=source,
        )
    return RuntimeOutputScope(
        kind="scenegroup",
        slug=scenegroup_slug(scene_ids),
        scene_ids=scene_ids,
        role_scenes=role_scenes,
        source=source,
    )


def runtime_scope_metadata_from_config(
    cfg: Mapping[str, Any],
    *,
    purpose: str = "training",
) -> dict[str, Any]:
    scope = runtime_output_scope_from_config(cfg, purpose=purpose)
    return scope.to_metadata() if scope is not None else {}


def scoped_output_base(base: str | Path, cfg: Mapping[str, Any], *, purpose: str = "training") -> Path:
    output_cfg = cfg.get("output", {}) if isinstance(cfg.get("output"), Mapping) else {}
    root = Path(base)
    if output_cfg.get("group_by_scene", True) is False:
        return root
    scope = runtime_output_scope_from_config(cfg, purpose=purpose)
    if scope is None or root.name == scope.slug:
        return root
    return root / scope.slug


def evaluation_study_id_from_config(cfg: Mapping[str, Any]) -> str:
    output_cfg = cfg.get("output", {}) if isinstance(cfg.get("output"), Mapping) else {}
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), Mapping) else {}
    experiment = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), Mapping) else {}
    raw = (
        output_cfg.get("evaluation_study_id")
        or eval_cfg.get("study_id")
        or experiment.get("name")
        or output_cfg.get("evaluation_run_name")
        or output_cfg.get("run_name")
        or "evaluation"
    )
    return _safe_path_slug(str(raw))


def evaluation_output_base(base: str | Path, cfg: Mapping[str, Any]) -> Path:
    root = Path(base)
    study_id = evaluation_study_id_from_config(cfg)
    if root.name == study_id and root.parent.name == PARTITION_EVALUATIONS:
        return root
    if root.name == PARTITION_EVALUATIONS:
        return root / study_id
    return root / PARTITION_EVALUATIONS / study_id


def output_layout_summary(path: str | Path, *, outputs_root: str | Path | None = None) -> dict[str, Any]:
    target = Path(path)
    root = Path(outputs_root) if outputs_root is not None else _nearest_outputs_root(target)
    if root is None:
        return {
            "outputs_root": None,
            "canonical_partition": "outside_outputs",
            "scope_kind": None,
            "scope_slug": None,
            "legacy": False,
            "archive": False,
            "explicit_non_run_partition": False,
        }
    try:
        rel_parts = target.resolve().relative_to(root.resolve()).parts
    except (OSError, ValueError):
        try:
            rel_parts = target.relative_to(root).parts
        except ValueError:
            rel_parts = ()
    first = rel_parts[0] if rel_parts else ""
    partition = _partition_name(first)
    scene_ids = _scene_ids_from_partition(first)
    return {
        "outputs_root": str(root),
        "canonical_partition": partition,
        "scope_kind": _scope_kind_from_partition(first),
        "scope_slug": first if first else None,
        "scene_ids": scene_ids,
        "scene_slugs": [f"scene{scene_id}" for scene_id in scene_ids],
        "legacy": partition.startswith("legacy_"),
        "archive": partition == PARTITION_ARCHIVE or first == PARTITION_ARCHIVE,
        "explicit_non_run_partition": first in DEFAULT_NON_RUN_PARTITIONS and target.name == first,
    }


def is_default_outputs_root(path: str | Path) -> bool:
    return Path(path).name == OUTPUT_ROOT


def is_default_skipped_partition(path: str | Path) -> bool:
    target = Path(path)
    return target.name in DEFAULT_NON_RUN_PARTITIONS and target.parent.name == OUTPUT_ROOT


def _dataset_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    data = cfg.get("data", {}) if isinstance(cfg, Mapping) else {}
    dataset = data.get("dataset", {}) if isinstance(data, Mapping) else {}
    return dataset if isinstance(dataset, Mapping) else {}


def _role_scenes(dataset_cfg: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    aliases = {"val_scenes": "validation", "validation_scenes": "validation"}
    for key in SCENE_LIST_KEYS:
        if key not in dataset_cfg:
            continue
        role = aliases.get(key, key.replace("_scenes", ""))
        result[role] = _normalize_scene_ids(_as_sequence(dataset_cfg[key]))
    return result


def _scope_scene_ids(
    dataset_cfg: Mapping[str, Any],
    role_scenes: Mapping[str, tuple[int, ...]],
    *,
    purpose: str,
) -> tuple[tuple[int, ...], str]:
    if purpose == "evaluation":
        for role in ("eval", "test", "validation", "train", "scenes"):
            scenes = role_scenes.get(role)
            if scenes:
                key = "scenes" if role == "scenes" else f"{role}_scenes"
                return scenes, f"data.dataset.{key}"
    else:
        for role in ("train", "scenes"):
            scenes = role_scenes.get(role)
            if scenes:
                key = "scenes" if role == "scenes" else f"{role}_scenes"
                return scenes, f"data.dataset.{key}"
        collected: list[int] = []
        for role in ("validation", "test", "eval"):
            collected.extend(role_scenes.get(role, ()))
        if collected:
            return tuple(dict.fromkeys(sorted(collected))), "data.dataset.split_scenes"

    scene_value = dataset_cfg.get("scene", dataset_cfg.get("scene_id", dataset_cfg.get("scene_slug")))
    scene = resolve_deepsense_scene(scene_value, dataset_type=dataset_cfg.get("type", "deepsense6g"))
    return (scene.scene_id,), "data.dataset.scene"


def _generic_scope_from_dataset(dataset_cfg: Mapping[str, Any]) -> RuntimeOutputScope | None:
    raw = dataset_cfg.get("scene_slug") or dataset_cfg.get("scene") or dataset_cfg.get("scene_id")
    if raw in (None, ""):
        return None
    slug = _safe_path_slug(str(raw))
    return RuntimeOutputScope(kind="scene", slug=slug, scene_ids=(), role_scenes={}, source="data.dataset.scene")


def _normalize_scene_ids(values: tuple[Any, ...]) -> tuple[int, ...]:
    scenes = sorted({resolve_deepsense_scene(value).scene_id for value in values})
    return tuple(scenes)


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        if "," in stripped:
            return tuple(item.strip() for item in stripped.split(",") if item.strip())
        return (stripped,)
    if isinstance(value, (int, float)):
        return (value,)
    return tuple(value)


def _is_contiguous(scene_ids: tuple[int, ...]) -> bool:
    return list(scene_ids) == list(range(scene_ids[0], scene_ids[-1] + 1))


def _safe_path_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned or "evaluation"


def _nearest_outputs_root(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if candidate.name == OUTPUT_ROOT:
            return candidate
    return None


def _partition_name(first_part: str) -> str:
    if not first_part:
        return "outputs_root"
    if first_part in CANONICAL_RUNTIME_PARTITIONS or first_part == PARTITION_FEATURES:
        return first_part
    if re.fullmatch(r"scene\d+", first_part):
        return "scene"
    if first_part.startswith(SCENEGROUP_PREFIX):
        return "scenegroup"
    if first_part.isdigit():
        return "legacy_numeric_scene"
    if first_part == "best_checkpoints":
        return "legacy_registry"
    if first_part.startswith("eval_"):
        return "legacy_evaluation"
    return "legacy_root_run"


def _scope_kind_from_partition(first_part: str) -> str | None:
    if re.fullmatch(r"scene\d+", first_part):
        return "scene"
    if first_part.startswith(SCENEGROUP_PREFIX):
        return "scenegroup"
    return None


def _scene_ids_from_partition(first_part: str) -> list[int]:
    if re.fullmatch(r"scene\d+", first_part):
        return [int(first_part.replace("scene", "", 1))]
    if first_part.startswith(SCENEGROUP_PREFIX):
        return [int(match) for match in re.findall(r"s(\d+)", first_part)]
    return []
