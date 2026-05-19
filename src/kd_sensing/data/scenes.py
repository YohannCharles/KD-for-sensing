from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kd_sensing.data.layouts import deepsense6g_legacy_scene_root, deepsense6g_scene_root


DEFAULT_DEEPSENSE_SCENE_ID = 31
DEFAULT_TRAIN_CSV_NAME = "train_seqs_RA_GPS_LIDAR.csv"
DEFAULT_TEST_CSV_NAME = "test_seqs_RA_GPS_LIDAR.csv"
DEEPSENSE_DATASET_TYPES = {"deepsense6g"}
REMOVED_DEEPSENSE_DATASET_TYPES = {
    "scenario9": 9,
    "scenario31": 31,
    "scenario32": 32,
}


@dataclass(frozen=True)
class DeepSenseScene:
    scene_id: int
    scene_slug: str
    aliases: tuple[str, ...]
    default_data_root: str
    legacy_data_root: str
    default_train_csv_name: str = DEFAULT_TRAIN_CSV_NAME
    default_test_csv_name: str = DEFAULT_TEST_CSV_NAME

    def metadata(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_slug": self.scene_slug,
        }


DEEPSENSE_SCENES: dict[int, DeepSenseScene] = {
    9: DeepSenseScene(
        scene_id=9,
        scene_slug="scene9",
        aliases=("9", "scene9", "scenario9"),
        default_data_root=deepsense6g_scene_root(9),
        legacy_data_root=deepsense6g_legacy_scene_root(9),
    ),
    31: DeepSenseScene(
        scene_id=31,
        scene_slug="scene31",
        aliases=("31", "scene31", "scenario31"),
        default_data_root=deepsense6g_scene_root(31),
        legacy_data_root=deepsense6g_legacy_scene_root(31),
    ),
    32: DeepSenseScene(
        scene_id=32,
        scene_slug="scene32",
        aliases=("32", "scene32", "scenario32"),
        default_data_root=deepsense6g_scene_root(32),
        legacy_data_root=deepsense6g_legacy_scene_root(32),
    ),
}


def _normalize_scene_token(value: Any) -> str:
    return str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


_SCENE_ALIASES = {
    _normalize_scene_token(alias): scene_id
    for scene_id, scene in DEEPSENSE_SCENES.items()
    for alias in scene.aliases
}


def is_deepsense_dataset_type(dataset_type: Any) -> bool:
    return str(dataset_type or "").strip().lower() in DEEPSENSE_DATASET_TYPES


def resolve_deepsense_scene(scene: Any = None, *, dataset_type: Any = None) -> DeepSenseScene:
    _reject_removed_dataset_type(dataset_type)
    if scene is None:
        scene_id = DEFAULT_DEEPSENSE_SCENE_ID
    else:
        scene_id = _scene_id_from_value(scene)
    try:
        return DEEPSENSE_SCENES[scene_id]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported DeepSense6G scene '{scene}'. Supported scenes: {supported_scene_names()}."
        ) from exc


def normalize_deepsense_dataset_config(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_type = dataset_cfg.get("type", "deepsense6g")
    _reject_removed_dataset_type(dataset_type)
    if not is_deepsense_dataset_type(dataset_type):
        return dataset_cfg
    has_scene = any(key in dataset_cfg for key in ("scene", "scene_id", "scene_slug"))

    scene_value = dataset_cfg.get("scene", dataset_cfg.get("scene_id", dataset_cfg.get("scene_slug")))
    scene = resolve_deepsense_scene(scene_value, dataset_type=dataset_type)
    dataset_cfg["scene"] = scene.scene_id
    dataset_cfg["scene_id"] = scene.scene_id
    dataset_cfg["scene_slug"] = scene.scene_slug

    if is_deepsense_dataset_type(dataset_type):
        if not dataset_cfg.get("data_root"):
            dataset_cfg["data_root"] = scene.default_data_root
        if not dataset_cfg.get("train_csv_name"):
            dataset_cfg["train_csv_name"] = scene.default_train_csv_name
        if not dataset_cfg.get("test_csv_name"):
            dataset_cfg["test_csv_name"] = scene.default_test_csv_name
    return dataset_cfg


def normalize_deepsense_config(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = cfg.get("data", {}).get("dataset")
    if isinstance(dataset_cfg, dict):
        normalize_deepsense_dataset_config(dataset_cfg)
    return cfg


def retarget_deepsense_dataset_config(dataset_cfg: dict[str, Any], scene: Any) -> dict[str, Any]:
    """Retarget an already-normalized DeepSense6G dataset config to another scene.

    Config loading fills default roots/CSV names before diagnostics can expand
    compare_scenes. When switching scene afterward, replace only known built-in
    defaults so custom user paths are preserved.
    """

    dataset_type = str(dataset_cfg.get("type", "deepsense6g")).strip().lower()
    _reject_removed_dataset_type(dataset_type)

    target = resolve_deepsense_scene(scene, dataset_type=dataset_type)
    if _is_default_scene_value(dataset_cfg.get("data_root"), "default_data_root"):
        dataset_cfg["data_root"] = target.default_data_root
    if _is_default_scene_value(dataset_cfg.get("train_csv_name"), "default_train_csv_name"):
        dataset_cfg["train_csv_name"] = target.default_train_csv_name
    if _is_default_scene_value(dataset_cfg.get("test_csv_name"), "default_test_csv_name"):
        dataset_cfg["test_csv_name"] = target.default_test_csv_name

    dataset_cfg["scene"] = target.scene_id
    dataset_cfg["scene_id"] = target.scene_id
    dataset_cfg["scene_slug"] = target.scene_slug
    return dataset_cfg


def scene_metadata_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return {}
    dataset_type = dataset_cfg.get("type")
    if not is_deepsense_dataset_type(dataset_type):
        slug = dataset_cfg.get("scene") or dataset_cfg.get("scene_slug") or dataset_cfg.get("scene_id")
        return {"scene_id": slug, "scene_slug": str(slug)} if slug not in (None, "") else {}
    scene_value = dataset_cfg.get("scene", dataset_cfg.get("scene_id", dataset_cfg.get("scene_slug")))
    scene = resolve_deepsense_scene(scene_value, dataset_type=dataset_type)
    return scene.metadata()


def scene_slug_from_config(cfg: dict[str, Any]) -> str | None:
    metadata = scene_metadata_from_config(cfg)
    slug = metadata.get("scene_slug")
    return str(slug) if slug else None


def supported_scene_names() -> str:
    return ", ".join(scene.scene_slug for scene in DEEPSENSE_SCENES.values())


def _is_default_scene_value(value: Any, attr: str) -> bool:
    if value is None:
        return True
    text = str(value).strip().replace("\\", "/").rstrip("/")
    if not text:
        return True
    defaults = {str(getattr(scene, attr)).replace("\\", "/").rstrip("/") for scene in DEEPSENSE_SCENES.values()}
    return text in defaults or any(text.endswith(f"/{default}") for default in defaults)


def _scene_id_from_value(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    token = _normalize_scene_token(value)
    if token in _SCENE_ALIASES:
        return _SCENE_ALIASES[token]
    try:
        return int(token)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported DeepSense6G scene '{value}'. Supported scenes: {supported_scene_names()}."
        ) from exc


def _reject_removed_dataset_type(dataset_type: Any) -> None:
    dataset_type_key = str(dataset_type or "").strip().lower()
    if dataset_type_key in REMOVED_DEEPSENSE_DATASET_TYPES:
        scene_id = REMOVED_DEEPSENSE_DATASET_TYPES[dataset_type_key]
        raise ValueError(
            f"Removed DeepSense6G dataset type '{dataset_type_key}'. "
            f"Use data.dataset.type: deepsense6g with data.dataset.scene: {scene_id}."
        )
