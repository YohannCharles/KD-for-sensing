from .fusion import (
    CRAFFusionNet,
    FusionTeacherModalityNet,
    MARFFusionNet,
    RadarFeatureExtractor,
    FusionStudentModalityNet,
    TokenTransformerFusionNet,
)
from .gps import GpsFeatureExtractor, GpsModalityNet, GpsStudentModalityNet
from .image import ImageFeatureExtractor, ImageModalityNet, ImageStudentModalityNet
from .image_encoders import ResNet18ImageEncoder
from .lidar import LidarFeatureExtractor, LidarModalityNet, LidarStudentModalityNet
from .modular import (
    BeamClassificationHead,
    EarlyConcatGRUCore,
    LinearProjector,
    ModularSequenceModel,
    SingleGRUCore,
    TokenTransformerCore,
)
from .mmwave import MmWaveFeatureExtractor, MmWaveModalityNet, MmWaveStudentModalityNet
from .radar import RadarModalityNet, RadarStudentModalityNet

__all__ = [
    "GpsFeatureExtractor",
    "GpsModalityNet",
    "GpsStudentModalityNet",
    "ImageFeatureExtractor",
    "ImageModalityNet",
    "ImageStudentModalityNet",
    "ResNet18ImageEncoder",
    "LidarFeatureExtractor",
    "LidarModalityNet",
    "LidarStudentModalityNet",
    "MmWaveFeatureExtractor",
    "MmWaveModalityNet",
    "MmWaveStudentModalityNet",
    "RadarFeatureExtractor",
    "RadarModalityNet",
    "RadarStudentModalityNet",
    "CRAFFusionNet",
    "FusionTeacherModalityNet",
    "MARFFusionNet",
    "FusionStudentModalityNet",
    "TokenTransformerFusionNet",
    "BeamClassificationHead",
    "EarlyConcatGRUCore",
    "LinearProjector",
    "ModularSequenceModel",
    "SingleGRUCore",
    "TokenTransformerCore",
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
