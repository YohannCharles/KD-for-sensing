from typing import Any


TEMPORAL_MISSING_MODES = (
    "none",
    "frame_bernoulli",
    "modality_frame_bernoulli",
    "block",
    "stratified_modality_temporal",
)
TEMPORAL_AGGREGATION_MODES = ("last", "mean", "masked_mean", "flatten")


def normalize_temporal_missing_mode(value: Any) -> str:
    mode = str(value or "none").strip().lower()
    if mode not in TEMPORAL_MISSING_MODES:
        raise ValueError(f"temporal_missing_mode must be one of {TEMPORAL_MISSING_MODES}, got {value!r}.")
    return mode


def normalize_temporal_aggregation(value: Any) -> str:
    mode = str(value or "masked_mean").strip().lower()
    if mode not in TEMPORAL_AGGREGATION_MODES:
        raise ValueError(f"temporal_aggregation must be one of {TEMPORAL_AGGREGATION_MODES}, got {value!r}.")
    return mode


__all__ = [
    "TEMPORAL_AGGREGATION_MODES",
    "TEMPORAL_MISSING_MODES",
    "normalize_temporal_aggregation",
    "normalize_temporal_missing_mode",
]
