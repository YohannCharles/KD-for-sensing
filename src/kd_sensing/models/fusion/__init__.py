from kd_sensing.models.radar import RadarFeatureExtractor
from kd_sensing.models.lidar import LidarFeatureExtractor
from kd_sensing.models.mmwave import MmWaveFeatureExtractor

from .cls_token_transformer import CLSTokenTransformerFusionNet
from .networks import (
    FusionLightweightModalityNet,
    FusionStrongModalityNet,
)
from .token_transformer import TokenTransformerFusionNet

__all__ = [
    "CLSTokenTransformerFusionNet",
    "FusionStrongModalityNet",
    "LidarFeatureExtractor",
    "MmWaveFeatureExtractor",
    "RadarFeatureExtractor",
    "FusionLightweightModalityNet",
    "TokenTransformerFusionNet",
]

_REMOVED_ALIASES = {
    "Fusion" + "ModalityNet": "FusionStrongModalityNet",
    "Student" + "ModalityNet": "FusionLightweightModalityNet",
    "CRAFFusionNet": "CRAF has been retired; use cls_token_transformer_fusion or a current fusion model.",
    "MARFFusionNet": "MARF has been retired; use cls_token_transformer_fusion or a current fusion model.",
    "PriorResidualGate": "teacher-prior CRAF has been retired.",
    "ReliabilityEstimator": "CRAF reliability gates have been retired.",
    "UniModalHead": "CRAF/MARF unimodal auxiliary heads have been retired.",
    "AnchorFusion": "MARF has been retired.",
    "ModalityRouter": "MARF has been retired.",
    "ResidualAdapter": "MARF has been retired.",
    "BottleneckPrivateAdapter": "HiST-Beam/Hist has been retired; no compatibility adapter is provided.",
    "HistBeamFusionNet": "HiST-Beam/Hist has been retired; use current supervised, adapter, GPS candidate, or residual fusion workflows.",
}


def __getattr__(name: str):
    replacement = _REMOVED_ALIASES.get(name)
    if replacement is not None:
        if replacement.startswith(name):
            raise AttributeError(f"{name} has been removed; use {replacement}.")
        raise AttributeError(f"{name} has been removed. {replacement}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
