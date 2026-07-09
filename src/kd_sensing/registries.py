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
PREPROCESSORS = Registry("preprocessors")
JEPA_DOWNSTREAM_POOLERS = Registry("jepa_downstream_poolers")
JEPA_VISUAL_TOKEN_ENCODERS = Registry("jepa_visual_token_encoders")
DIFFICULTY_OPERATORS = Registry("difficulty_operators")

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
DATASETS.register_removed(
    "scenario33",
    "Use {'type': 'deepsense6g', 'scene': 33}.",
)
DATASETS.register_removed(
    "scenario34",
    "Use {'type': 'deepsense6g', 'scene': 34}.",
)
MODELS.register_removed(
    "fusion_strong",
    "Use model.primary.type='modular_sequence' fusion with current modality encoders, projectors, early_concat_gru, and beam_head.",
)
MODELS.register_removed(
    "fusion_lightweight",
    "Use model.primary.type='modular_sequence' fusion or the current 'cls_token_transformer_fusion' lightweight route.",
)
MODELS.register_removed(
    "bev_fusion_2604",
    "BEV-Fusion 2604 reproduction was retired after final C2. Use U-MaskBeamJEPA or current MMW/CSI workflows.",
)
MODELS.register_removed(
    "vision_position_late_fusion",
    "Vision-Position baselines were retired after final C2. Use U-MaskBeamJEPA or current MMW/CSI workflows.",
)
MODELS.register_removed(
    "vision_position_transformer_fusion",
    "Vision-Position baselines were retired after final C2. Use U-MaskBeamJEPA or current MMW/CSI workflows.",
)
MODELS.register_removed(
    "gps_sequence_baseline",
    "Vision-Position GPS-only baseline was retired after final C2. Use current GPS encoders inside modular_sequence or MMW/CSI workflows.",
)
LOSSES.register_removed("logits_kd", "KD support has been removed. Use supervised or adaptation losses.")
LOSSES.register_removed("rkd", "KD support has been removed. Use supervised or adaptation losses.")


def import_default_components() -> None:
    """Import modules that register built-in components."""

    import kd_sensing.data.datasets.deepsense6g  # noqa: F401
    import kd_sensing.data.datasets.mmw  # noqa: F401
    import kd_sensing.data.datasets.synthetic  # noqa: F401
    import kd_sensing.evaluation.metrics  # noqa: F401
    import kd_sensing.losses.beam  # noqa: F401
    import kd_sensing.losses.physics_informed  # noqa: F401
    import kd_sensing.losses.u_mask_beam_jepa  # noqa: F401
    import kd_sensing.models.amr_net  # noqa: F401
    import kd_sensing.models.fusion  # noqa: F401
    import kd_sensing.models.csi_encoder  # noqa: F401
    import kd_sensing.models.geometry_prior  # noqa: F401
    import kd_sensing.models.gps  # noqa: F401
    import kd_sensing.models.image  # noqa: F401
    import kd_sensing.models.image_encoders  # noqa: F401
    import kd_sensing.models.jepa_downstream  # noqa: F401
    import kd_sensing.models.jepa  # noqa: F401
    import kd_sensing.models.lidar  # noqa: F401
    import kd_sensing.models.modular  # noqa: F401
    import kd_sensing.models.mmwave  # noqa: F401
    import kd_sensing.models.pinn_multimodal_beam  # noqa: F401
    import kd_sensing.models.radar  # noqa: F401
    import kd_sensing.models.rmbp_mm  # noqa: F401
    import kd_sensing.models.tinyvit  # noqa: F401
    import kd_sensing.models.u_mask_beam_jepa  # noqa: F401
    import kd_sensing.preprocessing.csv  # noqa: F401
    import kd_sensing.preprocessing.image_cache  # noqa: F401
    import kd_sensing.preprocessing.lidar  # noqa: F401
    import kd_sensing.preprocessing.sample_cache  # noqa: F401
    import kd_sensing.preprocessing.sequences  # noqa: F401


def import_default_difficulty_operators() -> None:
    """Import built-in difficulty operators without pulling them into light registry imports."""

    import kd_sensing.data.difficulty.operators  # noqa: F401


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
    "PREPROCESSORS",
    "JEPA_DOWNSTREAM_POOLERS",
    "JEPA_VISUAL_TOKEN_ENCODERS",
    "DIFFICULTY_OPERATORS",
    "import_default_components",
    "import_default_difficulty_operators",
]
