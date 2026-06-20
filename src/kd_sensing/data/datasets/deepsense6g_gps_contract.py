import math


GPS_FEATURE_DIMS = {
    "relative_polar": 3,
    "paper_calibrated_relative_polar": 3,
    "paper_distance_angle": 2,
}
SUPPORTED_GPS_FEATURE_MODE = "relative_polar"
CALIBRATED_GPS_FEATURE_MODES = {"paper_calibrated_relative_polar", "paper_distance_angle"}
PAPER_CALIBRATED_GPS_MODE = "paper_distance_angle"
PAPER_SCENE_CENTER_ANGLES_RAD = {
    31: -0.72,
    32: -0.8125375604986421 + math.pi / 2.0,
    33: 0.59,
    34: -0.51,
}
PAPER_DISTANCE_ANGLE_FEATURE_VERSION = "official_arctan_ratio_v1"


SUPPORTED_GPS_BEV_XY_SOURCES = ("history_relative_xy",)


def normalize_gps_feature_mode(mode: str | None) -> str:
    normalized = str(mode or SUPPORTED_GPS_FEATURE_MODE).strip().lower()
    if normalized not in GPS_FEATURE_DIMS:
        supported = ", ".join(sorted(GPS_FEATURE_DIMS))
        raise ValueError(f"Unsupported gps_feature_mode '{mode}'. Supported modes: {supported}.")
    return normalized


def resolve_gps_angle_offset(
    *,
    gps_feature_mode: str,
    scene_id: int | str,
    explicit_value: float | None,
    source: str | None,
) -> tuple[float | None, str]:
    if gps_feature_mode not in CALIBRATED_GPS_FEATURE_MODES:
        return None, "not_applicable"
    if explicit_value is not None:
        return float(explicit_value), "explicit"
    source_key = str(source or "paper_scene_default").strip().lower()
    if source_key in {"none", "zero", "disabled"}:
        return 0.0, source_key
    if source_key != "paper_scene_default":
        raise ValueError(
            "gps_angle_offset_source must be 'paper_scene_default', 'explicit', "
            "'none', 'zero', or 'disabled'."
        )
    normalized_scene_id = int(scene_id)
    if normalized_scene_id in PAPER_SCENE_CENTER_ANGLES_RAD:
        return float(PAPER_SCENE_CENTER_ANGLES_RAD[normalized_scene_id]), "paper_scene_default"
    return 0.0, "paper_scene_default_missing"


def resolve_gps_source_seq_len(
    *,
    seq_len: int,
    gps_seq_len: int | None,
    gps_source_seq_len: int | None,
) -> int:
    selected = gps_source_seq_len if gps_source_seq_len is not None else gps_seq_len
    resolved = int(selected) if selected is not None else int(seq_len)
    if resolved <= 0:
        raise ValueError("gps_source_seq_len must be positive when provided.")
    return resolved


def normalize_gps_bev_xy_source(value: str | None) -> str:
    source = str(value or "history_relative_xy").strip().lower()
    if source not in SUPPORTED_GPS_BEV_XY_SOURCES:
        supported = ", ".join(repr(item) for item in SUPPORTED_GPS_BEV_XY_SOURCES)
        raise ValueError(f"gps_bev_xy_source must be one of {supported}.")
    return source


__all__ = [
    "CALIBRATED_GPS_FEATURE_MODES",
    "GPS_FEATURE_DIMS",
    "PAPER_CALIBRATED_GPS_MODE",
    "PAPER_DISTANCE_ANGLE_FEATURE_VERSION",
    "PAPER_SCENE_CENTER_ANGLES_RAD",
    "SUPPORTED_GPS_BEV_XY_SOURCES",
    "SUPPORTED_GPS_FEATURE_MODE",
    "normalize_gps_bev_xy_source",
    "normalize_gps_feature_mode",
    "resolve_gps_angle_offset",
    "resolve_gps_source_seq_len",
]
