"""Lightweight component registries used by config-driven workflows."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Iterable, Optional


class RegistryError(ValueError):
    """Raised when a registry cannot resolve or construct a component."""


class Registry:
    """Map string names to components and construct them from config dicts."""

    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, Callable[..., Any]] = {}
        self._removed: Dict[str, str] = {}

    def register_removed(self, name: str, message: str) -> None:
        self._removed[name] = message

    def register(self, name: Optional[str] = None, *, force: bool = False):
        """Register a component under ``name``.

        Can be used as ``@REGISTRY.register("name")`` or ``REGISTRY.register()(Cls)``.
        """

        def decorator(component: Callable[..., Any]):
            key = name or component.__name__
            if key in self._items and not force:
                available = ", ".join(self.list()) or "<empty>"
                raise RegistryError(
                    f"Duplicate registration '{key}' in registry '{self.name}'. "
                    f"Available names: {available}"
                )
            self._items[key] = component
            return component

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        if name in self._removed:
            available = ", ".join(self.list()) or "<empty>"
            raise RegistryError(
                f"Removed component '{name}' in registry '{self.name}'. "
                f"{self._removed[name]} Available names: {available}"
            )
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(self.list()) or "<empty>"
            raise RegistryError(
                f"Unknown component '{name}' in registry '{self.name}'. "
                f"Available names: {available}"
            ) from exc

    def list(self) -> list[str]:
        return sorted(self._items.keys())

    def build(self, cfg: Any = None, **extra_kwargs: Any) -> Any:
        if cfg is None:
            raise RegistryError(
                f"Missing config for registry '{self.name}'. Available names: "
                f"{', '.join(self.list()) or '<empty>'}"
            )
        if isinstance(cfg, str):
            cfg = {"type": cfg}
        if not isinstance(cfg, dict):
            raise RegistryError(
                f"Registry '{self.name}' expected a dict or string config, got {type(cfg).__name__}."
            )
        if "type" not in cfg:
            raise RegistryError(
                f"Missing required field 'type' for registry '{self.name}'. "
                f"Available names: {', '.join(self.list()) or '<empty>'}"
            )
        params = dict(cfg)
        component_name = params.pop("type")
        params.update(extra_kwargs)
        component = self.get(component_name)
        try:
            return component(**params)
        except TypeError as exc:
            required = _missing_required_params(component, params)
            missing = f" Missing required parameters: {', '.join(required)}." if required else ""
            available = ", ".join(self.list()) or "<empty>"
            raise RegistryError(
                f"Failed to build '{component_name}' from registry '{self.name}'.{missing} "
                f"Available names: {available}. Original error: {exc}"
            ) from exc


def _missing_required_params(component: Callable[..., Any], provided: dict[str, Any]) -> list[str]:
    try:
        signature = inspect.signature(component)
    except (TypeError, ValueError):
        return []
    missing = []
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty and name not in provided:
            missing.append(name)
    return missing


MODELS = Registry("models")
ENCODERS = Registry("encoders")
PROJECTORS = Registry("projectors")
REPRESENTATION_CORES = Registry("representation_cores")
HEADS = Registry("heads")
DATASETS = Registry("datasets")
LOSSES = Registry("losses")
METRICS = Registry("metrics")
DISTILLERS = Registry("distillers")
PREPROCESSORS = Registry("preprocessors")

DATASETS.register_removed(
    "scenario9",
    "Use {'type': 'deepsense6g', 'scene': 9}.",
)
DATASETS.register_removed(
    "scenario31",
    "Use {'type': 'deepsense6g', 'scene': 31}.",
)
DATASETS.register_removed(
    "scenario32",
    "Use {'type': 'deepsense6g', 'scene': 32}.",
)
MODELS.register_removed(
    "Fusion" + "ModalityNet",
    "Use the 'fusion_teacher' registry name or FusionTeacherModalityNet.",
)
MODELS.register_removed(
    "Student" + "ModalityNet",
    "Use the 'fusion_student' registry name or FusionStudentModalityNet.",
)


def registry_self_check() -> dict[str, str]:
    """Small self-check used by smoke scripts and docs."""

    local = Registry("self_check")

    @local.register("example")
    class Example:
        def __init__(self, value: int):
            self.value = value

    instance = local.build({"type": "example", "value": 7})
    if instance.value != 7:
        raise RegistryError("Registry self-check failed to build the example component.")

    try:
        local.build({"type": "missing"})
    except RegistryError as exc:
        unknown_message = str(exc)
    else:
        raise RegistryError("Registry self-check expected an unknown-name error.")

    try:
        local.register("example")(Example)
    except RegistryError as exc:
        duplicate_message = str(exc)
    else:
        raise RegistryError("Registry self-check expected a duplicate-name error.")

    return {
        "build": "ok",
        "unknown": unknown_message,
        "duplicate": duplicate_message,
    }


def import_default_components() -> None:
    """Import modules that register built-in components."""

    import kd_sensing.data.datasets.deepsense6g  # noqa: F401
    import kd_sensing.data.datasets.mmw  # noqa: F401
    import kd_sensing.data.datasets.multimodal_nf  # noqa: F401
    import kd_sensing.data.datasets.raymobtime_s008  # noqa: F401
    import kd_sensing.data.datasets.synthetic  # noqa: F401
    import kd_sensing.distillation.distillers  # noqa: F401
    import kd_sensing.distillation.losses  # noqa: F401
    import kd_sensing.evaluation.metrics  # noqa: F401
    import kd_sensing.models.fusion  # noqa: F401
    import kd_sensing.models.csi  # noqa: F401
    import kd_sensing.models.gps  # noqa: F401
    import kd_sensing.models.image  # noqa: F401
    import kd_sensing.models.image_encoders  # noqa: F401
    import kd_sensing.models.lidar  # noqa: F401
    import kd_sensing.models.modular  # noqa: F401
    import kd_sensing.models.mmwave  # noqa: F401
    import kd_sensing.models.radar  # noqa: F401
    import kd_sensing.models.raymobtime_s008  # noqa: F401
    import kd_sensing.preprocessing.csv  # noqa: F401
    import kd_sensing.preprocessing.lidar  # noqa: F401
    import kd_sensing.preprocessing.multimodal_nf  # noqa: F401
    import kd_sensing.preprocessing.raymobtime_s008  # noqa: F401
    import kd_sensing.preprocessing.sequences  # noqa: F401


__all__ = [
    "Registry",
    "RegistryError",
    "MODELS",
    "ENCODERS",
    "PROJECTORS",
    "REPRESENTATION_CORES",
    "HEADS",
    "DATASETS",
    "LOSSES",
    "METRICS",
    "DISTILLERS",
    "PREPROCESSORS",
    "registry_self_check",
    "import_default_components",
]
