from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


DATASET_ROOT = PurePosixPath("dataset")
DEEPSENSE6G_FAMILY = "DeepSense6G"
MMW_FAMILY = "MMW"
RAYMOBTIME_FAMILY = "Raymobtime"
DEEPSENSE6G_SCENE_IDS = (9, 31, 32)
MMW_CONDITIONS = ("sunny", "rainy", "foggy")


@dataclass(frozen=True)
class DeepSense6GSceneLayout:
    scene_id: int

    @property
    def scenario_slug(self) -> str:
        return f"scenario{self.scene_id}"

    @property
    def canonical_root(self) -> str:
        return str(DATASET_ROOT / DEEPSENSE6G_FAMILY / self.scenario_slug)

    @property
    def legacy_root(self) -> str:
        return str(DATASET_ROOT / self.scenario_slug)

    @property
    def raw_csv_name(self) -> str:
        return f"{self.scenario_slug}.csv"

    @property
    def radar_csv_name(self) -> str:
        return f"{self.scenario_slug}_RA.csv"

    @property
    def radar_csv_path(self) -> str:
        return str(PurePosixPath(self.canonical_root) / self.radar_csv_name)


@dataclass(frozen=True)
class MMWConditionLayout:
    condition: str

    @property
    def root(self) -> str:
        return str(DATASET_ROOT / MMW_FAMILY / self.condition)

    @property
    def sensor_data_root(self) -> str:
        return str(PurePosixPath(self.root) / "Sensor_Data")

    @property
    def channel_data_root(self) -> str:
        return str(PurePosixPath(self.root) / "Channel_Data")

    def prepared_scenario_root(self, scenario: Any) -> str:
        scenario_key = str(scenario or "").strip()
        if not scenario_key:
            raise ValueError("MMW prepared scenario name must be non-empty.")
        return str(PurePosixPath(self.root) / "Prepared" / scenario_key)

    @property
    def required_subdirs(self) -> tuple[str, str]:
        return ("Sensor_Data", "Channel_Data")


@dataclass(frozen=True)
class RaymobtimeS008Layout:
    scenario: str = "s008"

    @property
    def root(self) -> str:
        return str(DATASET_ROOT / RAYMOBTIME_FAMILY / self.scenario)

    @property
    def baseline_root(self) -> str:
        return str(PurePosixPath(self.root) / "baseline_data")

    @property
    def raw_root(self) -> str:
        return str(PurePosixPath(self.root) / "raw_data")

    @property
    def cache_root(self) -> str:
        return str(PurePosixPath(self.root) / "cache")

    @property
    def coord_csv_name(self) -> str:
        return "CoordVehiclesRxPerScene_s008.csv"

    @property
    def ray_zip_name(self) -> str:
        return "ray_tracing_data_s008_carrier60GHz.zip"

    @property
    def required_paths(self) -> tuple[str, ...]:
        return (
            "baseline_data/beam_output",
            "baseline_data/coord_input",
            "baseline_data/lidar_input",
            "baseline_data/image_v2_input",
            f"raw_data/{self.coord_csv_name}",
            f"raw_data/{self.ray_zip_name}",
        )


def deepsense6g_scene_layout(scene_id: Any) -> DeepSense6GSceneLayout:
    scene_id = _normalize_int_token(scene_id, context="DeepSense6G scene")
    if scene_id not in DEEPSENSE6G_SCENE_IDS:
        raise ValueError(
            f"Unsupported DeepSense6G scene '{scene_id}'. Supported scenes: "
            f"{', '.join(str(item) for item in DEEPSENSE6G_SCENE_IDS)}."
        )
    return DeepSense6GSceneLayout(scene_id=scene_id)


def deepsense6g_scene_root(scene_id: Any) -> str:
    return deepsense6g_scene_layout(scene_id).canonical_root


def deepsense6g_legacy_scene_root(scene_id: Any) -> str:
    return deepsense6g_scene_layout(scene_id).legacy_root


def deepsense6g_radar_csv_path(scene_id: Any) -> str:
    return deepsense6g_scene_layout(scene_id).radar_csv_path


def mmw_condition_layout(condition: Any) -> MMWConditionLayout:
    condition_key = str(condition or "").strip().lower()
    if condition_key not in MMW_CONDITIONS:
        raise ValueError(
            f"Unsupported MMW condition '{condition}'. Supported conditions: {', '.join(MMW_CONDITIONS)}."
        )
    return MMWConditionLayout(condition=condition_key)


def raymobtime_s008_layout() -> RaymobtimeS008Layout:
    return RaymobtimeS008Layout()


def raymobtime_s008_root() -> str:
    return raymobtime_s008_layout().root


def _normalize_int_token(value: Any, *, context: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    for prefix in ("scenario", "scene"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"Unsupported {context} '{value}'.") from exc
