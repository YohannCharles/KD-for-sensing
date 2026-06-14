from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "GpsFeatureExtractor": "kd_sensing.models.gps",
    "GpsLightweightModalityNet": "kd_sensing.models.gps",
    "GpsStrongModalityNet": "kd_sensing.models.gps",
    "ImageFeatureExtractor": "kd_sensing.models.image",
    "ImageLightweightModalityNet": "kd_sensing.models.image",
    "ImageStrongModalityNet": "kd_sensing.models.image",
    "GPSConditionedJEPA": "kd_sensing.models.jepa",
    "GPSQueryPool": "kd_sensing.models.jepa",
    "JepaContextImageEncoder": "kd_sensing.models.jepa",
    "JepaMaskSampler": "kd_sensing.models.jepa",
    "VisualPatchTokenEncoder": "kd_sensing.models.jepa",
    "ResNet18ImageEncoder": "kd_sensing.models.image_encoders",
    "LidarFeatureExtractor": "kd_sensing.models.lidar",
    "LidarLightweightModalityNet": "kd_sensing.models.lidar",
    "LidarStrongModalityNet": "kd_sensing.models.lidar",
    "MmWaveFeatureExtractor": "kd_sensing.models.mmwave",
    "MmWaveLightweightModalityNet": "kd_sensing.models.mmwave",
    "MmWaveStrongModalityNet": "kd_sensing.models.mmwave",
    "ObservabilityAwareFusion": "kd_sensing.models.observability_aware_fusion",
    "RadarFeatureExtractor": "kd_sensing.models.radar",
    "RadarLightweightModalityNet": "kd_sensing.models.radar",
    "RadarStrongModalityNet": "kd_sensing.models.radar",
    "PilotCSIChannelEstimator": "kd_sensing.models.csi",
    "PilotDualViewCSIEncoder": "kd_sensing.models.csi",
    "CLSTokenTransformerFusionNet": "kd_sensing.models.fusion",
    "FusionLightweightModalityNet": "kd_sensing.models.fusion",
    "FusionStrongModalityNet": "kd_sensing.models.fusion",
    "TokenTransformerFusionNet": "kd_sensing.models.fusion",
    "BeamClassificationHead": "kd_sensing.models.modular",
    "BEVFusion2604Net": "kd_sensing.models.bev_fusion_2604",
    "EarlyConcatGRUCore": "kd_sensing.models.modular",
    "GpsSequenceBaselineNet": "kd_sensing.models.vision_position",
    "LinearProjector": "kd_sensing.models.modular",
    "ModularSequenceModel": "kd_sensing.models.modular",
    "NextBeamQueryTransformerCore": "kd_sensing.models.modular",
    "SingleGRUCore": "kd_sensing.models.modular",
    "TokenTransformerCore": "kd_sensing.models.modular",
    "VisionPositionLateFusionNet": "kd_sensing.models.vision_position",
    "VisionPositionTransformerFusionNet": "kd_sensing.models.vision_position",
}

__all__ = list(_EXPORTS)

_REMOVED_ALIASES = {
    "Fusion" + "ModalityNet": "FusionStrongModalityNet",
    "Student" + "ModalityNet": "FusionLightweightModalityNet",
    "CRAFFusionNet": "CRAF has been retired; use cls_token_transformer_fusion or a current fusion model.",
    "MARFFusionNet": "MARF has been retired; use cls_token_transformer_fusion or a current fusion model.",
    "BottleneckPrivateAdapter": "HiST-Beam/Hist has been retired; no compatibility adapter is provided.",
    "HistBeamFusionNet": "HiST-Beam/Hist has been retired; use current supervised, adapter, GPS v2, CSI, JEPA, or viewer workflows.",
}


def __getattr__(name: str) -> Any:
    replacement = _REMOVED_ALIASES.get(name)
    if replacement is not None:
        if replacement.startswith(name):
            raise AttributeError(f"{name} has been removed; use {replacement}.")
        raise AttributeError(f"{name} has been removed. {replacement}")
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__, *_REMOVED_ALIASES])
