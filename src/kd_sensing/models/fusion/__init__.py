from kd_sensing.models.radar import RadarFeatureExtractor
from kd_sensing.models.lidar import LidarFeatureExtractor
from kd_sensing.models.mmwave import MmWaveFeatureExtractor

from .craf import CRAFFusionNet, PriorResidualGate, ReliabilityEstimator, TokenTransformerFusionNet, UniModalHead
from .marf import AnchorFusion, MARFFusionNet, ModalityRouter, ResidualAdapter
from .networks import (
    FusionTeacherModalityNet,
    FusionStudentModalityNet,
)

__all__ = [
    "AnchorFusion",
    "CRAFFusionNet",
    "FusionTeacherModalityNet",
    "LidarFeatureExtractor",
    "MARFFusionNet",
    "MmWaveFeatureExtractor",
    "ModalityRouter",
    "PriorResidualGate",
    "RadarFeatureExtractor",
    "ReliabilityEstimator",
    "ResidualAdapter",
    "FusionStudentModalityNet",
    "TokenTransformerFusionNet",
    "UniModalHead",
]

_REMOVED_ALIASES = {
    "Fusion" + "ModalityNet": "FusionTeacherModalityNet",
    "Student" + "ModalityNet": "FusionStudentModalityNet",
}


def __getattr__(name: str):
    replacement = _REMOVED_ALIASES.get(name)
    if replacement is not None:
        raise AttributeError(f"{name} has been removed; use {replacement}.")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
