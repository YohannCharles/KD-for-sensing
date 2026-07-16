"""Lightweight component registries used by config-driven workflows."""

import inspect
from typing import Any, Callable, Dict, Iterable, Optional


class RegistryError(ValueError):
    """Raised when a registry cannot resolve or construct a component."""


class Registry:
    """Map string names to components and construct them from config dicts."""

    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, Callable[..., Any]] = {}

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

    def build(self, cfg: dict[str, Any], **extra_kwargs: Any) -> Any:
        if not isinstance(cfg, dict):
            raise RegistryError(
                f"Registry '{self.name}' expected a dict config, got {type(cfg).__name__}."
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
PREPROCESSORS = Registry("preprocessors")

def import_default_components() -> None:
    """Import modules that register built-in components."""

    import kd_sensing.data.datasets.mmw  # noqa: F401
    import kd_sensing.losses.beam  # noqa: F401
    import kd_sensing.losses.u_mask_beam_jepa  # noqa: F401
    import kd_sensing.models.gps  # noqa: F401
    import kd_sensing.models.image_encoders  # noqa: F401
    import kd_sensing.models.lidar  # noqa: F401
    import kd_sensing.models.modular  # noqa: F401
    import kd_sensing.models.radar  # noqa: F401
    import kd_sensing.models.rmbp_mm  # noqa: F401
    import kd_sensing.models.tinyvit  # noqa: F401
    import kd_sensing.models.u_mask_beam_jepa  # noqa: F401
    import kd_sensing.preprocessing.mmw_radar  # noqa: F401


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
    "PREPROCESSORS",
    "import_default_components",
]
