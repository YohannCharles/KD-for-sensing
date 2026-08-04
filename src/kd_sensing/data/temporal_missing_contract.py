from typing import Any


TEMPORAL_SUPERSET_PAYLOAD_KEY = "temporal_superset_payload"
TEMPORAL_MISSING_MODES = (
    "none",
    "stratified_modality_temporal",
    "balanced_pattern_schedule",
    "fixed_single_modality",
)


def normalize_temporal_missing_mode(value: Any) -> str:
    mode = str(value or "none").strip().lower()
    if mode not in TEMPORAL_MISSING_MODES:
        raise ValueError(f"temporal_missing_mode must be one of {TEMPORAL_MISSING_MODES}, got {value!r}.")
    return mode
__all__ = [
    "TEMPORAL_MISSING_MODES",
    "TEMPORAL_SUPERSET_PAYLOAD_KEY",
    "normalize_temporal_missing_mode",
]
