from __future__ import annotations

import importlib
import importlib.util
import json
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DeepVerseDependencyError(RuntimeError):
    """Raised when the external DeepVerse package or scenario files are unavailable."""


@dataclass
class DeepVerseDT31Generator:
    scenario_root: str | Path
    scenario: str = "DT31"
    config_m: str | Path | None = None
    scenes: list[int] | None = None
    enable_camera: bool = True
    enable_lidar: bool = True
    enable_radar: bool = True
    enable_comm: bool = True
    enable_position: bool = True

    def resolved_config_m(self) -> Path:
        if self.config_m:
            return Path(self.config_m).expanduser().resolve()
        return Path(self.scenario_root).expanduser().resolve() / self.scenario / "param" / "config.m"

    def validate_environment(self) -> None:
        config_path = self.resolved_config_m()
        if not config_path.exists():
            raise DeepVerseDependencyError(f"DeepVerse DT31 config.m not found: {config_path}")

        if importlib.util.find_spec("deepverse") is None:
            raise DeepVerseDependencyError(
                "Python package 'deepverse' is not installed. "
                "Install it inside the kd_mm_beam environment before generating DT31 cache."
            )

    def load_dataset(self, *, output_root: str | Path | None = None) -> Any:
        self.validate_environment()
        deepverse = importlib.import_module("deepverse")
        try:
            parameter_manager_cls = getattr(deepverse, "ParameterManager")
            dataset_cls = getattr(deepverse, "Dataset")
        except AttributeError as exc:
            raise DeepVerseDependencyError(
                "The installed 'deepverse' package does not expose ParameterManager and Dataset."
            ) from exc

        config_path = self.resolved_config_m()
        param_manager = parameter_manager_cls(str(config_path))
        params = getattr(param_manager, "params", None)
        if not isinstance(params, MutableMapping):
            raise DeepVerseDependencyError("DeepVerse ParameterManager.params is not a mutable mapping.")

        self._configure_params(params)
        if output_root is not None:
            output_path = Path(output_root)
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "used_generation_params.json").write_text(
                json.dumps(_jsonable(params), indent=2, sort_keys=True),
                encoding="utf-8",
            )

        return dataset_cls(param_manager)

    def _configure_params(self, params: MutableMapping[str, Any]) -> None:
        root = str(Path(self.scenario_root).expanduser().resolve())
        params["dataset_folder"] = root
        params["scenario"] = self.scenario
        if self.scenes is not None:
            params["scenes"] = list(self.scenes)

        _set_if_present(params, "camera", self.enable_camera)
        _set_if_present(params, "lidar", self.enable_lidar)
        _set_if_present(params, "position", self.enable_position)
        _set_nested_if_present(params, "radar", "enable", self.enable_radar)
        _set_nested_if_present(params, "comm", "enable", self.enable_comm)
        _set_nested_if_present(params, "comm", "generate_OFDM_channels", 1 if self.enable_comm else 0)


def _set_if_present(params: MutableMapping[str, Any], key: str, value: Any) -> None:
    if key in params:
        params[key] = value


def _set_nested_if_present(params: MutableMapping[str, Any], parent: str, key: str, value: Any) -> None:
    nested = params.get(parent)
    if isinstance(nested, MutableMapping) and key in nested:
        nested[key] = value


def _jsonable(value: Any) -> Any:
    if isinstance(value, MutableMapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
