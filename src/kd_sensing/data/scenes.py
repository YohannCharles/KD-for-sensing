from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_DEEPSENSE_SCENE_ID = 32
DEFAULT_TRAIN_CSV_NAME = "train_seqs_RA_GPS_LIDAR.csv"
DEFAULT_TEST_CSV_NAME = "test_seqs_RA_GPS_LIDAR.csv"
DEEPSENSE_DATASET_TYPES = {"deepsense6g", "scenario9", "scenario32"}


@dataclass(frozen=True)
class DeepSenseScene:
    scene_id: int
    scene_slug: str
    aliases: tuple[str, ...]
    default_data_root: str
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
        default_data_root="dataset/scenario9",
    ),
    32: DeepSenseScene(
        scene_id=32,
        scene_slug="scene32",
        aliases=("32", "scene32", "scenario32"),
        default_data_root="dataset/scenario32",
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
    if scene is None:
        dataset_type_key = str(dataset_type or "deepsense6g").strip().lower()
        if dataset_type_key == "scenario9":
            scene_id = 9
        elif dataset_type_key == "scenario32":
            scene_id = 32
        else:
            scene_id = DEFAULT_DEEPSENSE_SCENE_ID
    else:
        scene_id = _scene_id_from_value(scene)
        _validate_dataset_type_scene(dataset_type, scene_id)
    try:
        return DEEPSENSE_SCENES[scene_id]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported DeepSense6G scene '{scene}'. Supported scenes: {supported_scene_names()}."
        ) from exc


def normalize_deepsense_dataset_config(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_type = dataset_cfg.get("type", "deepsense6g")
    has_scene = any(key in dataset_cfg for key in ("scene", "scene_id", "scene_slug"))
    if not is_deepsense_dataset_type(dataset_type) and not has_scene:
        return dataset_cfg

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


def scene_metadata_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return {}
    if not any(key in dataset_cfg for key in ("scene", "scene_id", "scene_slug")) and not is_deepsense_dataset_type(
        dataset_cfg.get("type")
    ):
        return {}
    scene_value = dataset_cfg.get("scene", dataset_cfg.get("scene_id", dataset_cfg.get("scene_slug")))
    scene = resolve_deepsense_scene(scene_value, dataset_type=dataset_cfg.get("type"))
    return scene.metadata()


def scene_slug_from_config(cfg: dict[str, Any]) -> str | None:
    metadata = scene_metadata_from_config(cfg)
    slug = metadata.get("scene_slug")
    return str(slug) if slug else None


def supported_scene_names() -> str:
    return ", ".join(scene.scene_slug for scene in DEEPSENSE_SCENES.values())


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


def _validate_dataset_type_scene(dataset_type: Any, scene_id: int) -> None:
    dataset_type_key = str(dataset_type or "").strip().lower()
    if dataset_type_key == "scenario9" and scene_id != 9:
        raise ValueError("data.dataset.type=scenario9 conflicts with data.dataset.scene; use scene 9 or type deepsense6g.")
    if dataset_type_key == "scenario32" and scene_id != 32:
        raise ValueError(
            "data.dataset.type=scenario32 conflicts with data.dataset.scene; use scene 32 or type deepsense6g."
        )
