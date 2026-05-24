from __future__ import annotations

MODALITIES = ("image", "lidar", "radar", "gps", "mmwave")
DISTRIBUTION_MODALITIES = (*MODALITIES, "fusion")
SHOW_MODES = ("all", "correct only", "wrong only", "low quality only")
LOW_QUALITY_MODALITY_THRESHOLD = 0.4
LOW_QUALITY_MEAN_THRESHOLD = 0.5

__all__ = [
    "DISTRIBUTION_MODALITIES",
    "LOW_QUALITY_MEAN_THRESHOLD",
    "LOW_QUALITY_MODALITY_THRESHOLD",
    "MODALITIES",
    "SHOW_MODES",
]
