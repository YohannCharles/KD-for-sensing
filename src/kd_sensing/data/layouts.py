from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


DATASET_ROOT = PurePosixPath("dataset")
MMW_FAMILY = "MMW"
MMW_CONDITIONS = ("sunny", "rainy", "foggy")


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
        name = str(scenario or "").strip()
        if not name:
            raise ValueError("MMW prepared scenario name must be non-empty.")
        return str(PurePosixPath(self.root) / "Prepared" / name)


def mmw_condition_layout(condition: Any) -> MMWConditionLayout:
    normalized = str(condition or "").strip().lower()
    if normalized not in MMW_CONDITIONS:
        raise ValueError(f"Unsupported MMW condition {condition!r}; expected one of {MMW_CONDITIONS}.")
    return MMWConditionLayout(normalized)


__all__ = ["MMWConditionLayout", "MMW_CONDITIONS", "MMW_FAMILY", "mmw_condition_layout"]
