from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "GpsFeatureExtractor": "kd_sensing.models.gps",
    "GpsModalityNet": "kd_sensing.models.gps",
    "GpsStudentModalityNet": "kd_sensing.models.gps",
    "ImageFeatureExtractor": "kd_sensing.models.image",
    "ImageModalityNet": "kd_sensing.models.image",
    "ImageStudentModalityNet": "kd_sensing.models.image",
    "ResNet18ImageEncoder": "kd_sensing.models.image_encoders",
    "LidarFeatureExtractor": "kd_sensing.models.lidar",
    "LidarModalityNet": "kd_sensing.models.lidar",
    "LidarStudentModalityNet": "kd_sensing.models.lidar",
    "MmWaveFeatureExtractor": "kd_sensing.models.mmwave",
    "MmWaveModalityNet": "kd_sensing.models.mmwave",
    "MmWaveStudentModalityNet": "kd_sensing.models.mmwave",
    "RadarFeatureExtractor": "kd_sensing.models.radar",
    "RadarModalityNet": "kd_sensing.models.radar",
    "RadarStudentModalityNet": "kd_sensing.models.radar",
    "CLSTokenTransformerFusionNet": "kd_sensing.models.fusion",
    "CRAFFusionNet": "kd_sensing.models.fusion",
    "FusionTeacherModalityNet": "kd_sensing.models.fusion",
    "MARFFusionNet": "kd_sensing.models.fusion",
    "FusionStudentModalityNet": "kd_sensing.models.fusion",
    "TokenTransformerFusionNet": "kd_sensing.models.fusion",
    "BeamClassificationHead": "kd_sensing.models.modular",
    "EarlyConcatGRUCore": "kd_sensing.models.modular",
    "LinearProjector": "kd_sensing.models.modular",
    "ModularSequenceModel": "kd_sensing.models.modular",
    "SingleGRUCore": "kd_sensing.models.modular",
    "TokenTransformerCore": "kd_sensing.models.modular",
}

__all__ = list(_EXPORTS)

_REMOVED_ALIASES = {
    "Fusion" + "ModalityNet": "FusionTeacherModalityNet",
    "Student" + "ModalityNet": "FusionStudentModalityNet",
}


def __getattr__(name: str) -> Any:
    replacement = _REMOVED_ALIASES.get(name)
    if replacement is not None:
        raise AttributeError(f"{name} has been removed; use {replacement}.")
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__, *_REMOVED_ALIASES])
