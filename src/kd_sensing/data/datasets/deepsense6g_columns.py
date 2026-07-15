from pathlib import Path
from typing import Iterable


def sorted_numbered_columns(columns: Iterable[str], prefix: str) -> list[str]:
    selected = []
    for col in columns:
        if not col.startswith(prefix):
            continue
        suffix = col[len(prefix) :]
        if suffix.isdigit():
            selected.append(col)
    return sorted(selected, key=lambda name: int(name[len(prefix) :]))


def validate_required_columns(
    csv_path: str | Path,
    enabled_modalities: tuple[str, ...],
    *,
    camera_cols: list[str],
    radar_cols: list[str],
    gps_cols: list[str],
    bs_gps_cols: list[str],
    future_gps_cols: list[str],
    future_bs_gps_cols: list[str],
    lidar_cols: list[str],
    mmwave_cols: list[str],
    csi_cols: list[str],
    beam_cols: list[str],
    future_beam_cols: list[str],
    seq_len: int | None,
    gps_seq_len: int | None,
    num_pred: int | None,
    include_position_targets: bool,
    include_history_position_targets: bool,
) -> None:
    path = Path(csv_path)
    minimum_seq = int(seq_len) if seq_len is not None else 1
    minimum_gps_seq = int(gps_seq_len) if gps_seq_len is not None else minimum_seq
    minimum_pred = int(num_pred) if num_pred is not None else 1
    requirements = {
        "beam": (beam_cols, minimum_seq, "beam1..beamN"),
        "future_beam": (future_beam_cols, minimum_pred, "future_beam1..future_beamN"),
    }
    if "image" in enabled_modalities:
        requirements["image"] = (camera_cols, minimum_seq, "camera1..cameraN")
    if "radar" in enabled_modalities:
        requirements["radar"] = (radar_cols, minimum_seq, "radar1..radarN")
    if "gps" in enabled_modalities or include_history_position_targets:
        requirements["gps"] = (gps_cols, minimum_gps_seq, "gps1..gpsN")
        requirements["bs_gps"] = (bs_gps_cols, minimum_gps_seq, "bs_gps1..bs_gpsN")
    if include_position_targets:
        requirements["future_gps"] = (future_gps_cols, minimum_pred, "future_gps1..future_gpsN")
        requirements["future_bs_gps"] = (
            future_bs_gps_cols,
            minimum_pred,
            "future_bs_gps1..future_bs_gpsN",
        )
    if "lidar" in enabled_modalities:
        requirements["lidar"] = (lidar_cols, minimum_seq, "lidar1..lidarN")
    if "mmwave" in enabled_modalities:
        requirements["mmwave"] = (mmwave_cols, minimum_seq, "mmwave1..mmwaveN")
    if "csi" in enabled_modalities:
        requirements["csi"] = (csi_cols, minimum_seq, "csi1..csiN")
    for name, (columns, minimum, expected) in requirements.items():
        if len(columns) < minimum:
            raise ValueError(
                required_columns_error(
                    name,
                    csv_path=path,
                    available=len(columns),
                    expected_columns=expected,
                    minimum=minimum,
                )
            )


def required_columns_error(
    name: str,
    *,
    csv_path: str | Path,
    available: int,
    expected_columns: str,
    minimum: int,
) -> str:
    hint = ""
    if name in {"future_gps", "future_bs_gps"}:
        hint = " Regenerate sequence CSVs with include_position_targets: true."
    return (
        f"{name} is enabled but {Path(csv_path)} contains {int(available)} {expected_columns} columns; "
        f"expected at least {int(minimum)}.{hint}"
    )


def ensure_gps_columns(
    *,
    root_csv: str | Path,
    gps_paths: list[list[str]] | None,
    bs_gps_paths: list[list[str]] | None,
    gps_feature_mode: str,
    supported_modes: Iterable[str],
) -> None:
    if gps_paths is None:
        raise ValueError(
            f"GPS is enabled but {root_csv} does not contain gps1..gpsN columns. "
            "Regenerate sequence CSVs with include_gps: true."
        )
    if gps_feature_mode not in supported_modes:
        supported = ", ".join(repr(item) for item in sorted(supported_modes))
        raise ValueError(
            f"Unsupported gps_feature_mode '{gps_feature_mode}'. "
            f"Supported modes: {supported}."
        )
    if bs_gps_paths is None:
        raise ValueError(
            f"gps_feature_mode '{gps_feature_mode}' requires bs_gps1..bs_gpsN columns in {root_csv}."
        )


def ensure_enabled_contract_columns(
    *,
    root_csv: str | Path,
    samples: object,
    use_gps: bool,
    use_gps_bev_xy: bool,
    use_mmwave: bool,
    use_csi: bool,
    use_lidar: bool,
    gps_feature_mode: str,
    supported_gps_modes: Iterable[str],
) -> None:
    if use_gps:
        ensure_gps_columns(
            root_csv=root_csv,
            gps_paths=getattr(samples, "gps_paths", None),
            bs_gps_paths=getattr(samples, "bs_gps_paths", None),
            gps_feature_mode=gps_feature_mode,
            supported_modes=supported_gps_modes,
        )
    if use_gps_bev_xy:
        ensure_gps_bev_xy_columns(
            root_csv=root_csv,
            gps_paths=getattr(samples, "gps_paths", None),
            bs_gps_paths=getattr(samples, "bs_gps_paths", None),
        )
    if use_mmwave:
        ensure_mmwave_columns(root_csv=root_csv, mmwave_paths=getattr(samples, "mmwave_paths", None))
    if use_csi:
        ensure_csi_columns(root_csv=root_csv, csi_paths=getattr(samples, "csi_paths", None))
    if use_lidar:
        ensure_lidar_columns(root_csv=root_csv, lidar_paths=getattr(samples, "lidar_paths", None))


def ensure_gps_bev_xy_columns(
    *,
    root_csv: str | Path,
    gps_paths: list[list[str]] | None,
    bs_gps_paths: list[list[str]] | None,
) -> None:
    if gps_paths is None or bs_gps_paths is None:
        raise ValueError(
            f"GPS BEV XY is enabled but {root_csv} does not contain gps1..gpsN "
            "and bs_gps1..bs_gpsN columns. Regenerate sequence CSVs with GPS and BS GPS history."
        )


def ensure_mmwave_columns(*, root_csv: str | Path, mmwave_paths: list[list[str]] | None) -> None:
    if mmwave_paths is None:
        raise ValueError(
            f"mmWave is enabled but {root_csv} does not contain mmwave1..mmwaveN columns. "
            "Regenerate sequence CSVs with include_mmwave: true."
        )


def ensure_csi_columns(*, root_csv: str | Path, csi_paths: list[list[str]] | None) -> None:
    if csi_paths is None:
        raise ValueError(
            f"CSI is enabled but {root_csv} does not contain csi1..csiN columns. "
            "Regenerate sequence CSVs with CSI export enabled."
        )


def ensure_lidar_columns(*, root_csv: str | Path, lidar_paths: list[list[str]] | None) -> None:
    if lidar_paths is None:
        raise ValueError(
            f"LiDAR is enabled but {root_csv} does not contain lidar1..lidarN columns. "
            "Regenerate sequence CSVs with include_lidar: true."
        )


__all__ = [
    "ensure_csi_columns",
    "ensure_enabled_contract_columns",
    "ensure_gps_bev_xy_columns",
    "ensure_gps_columns",
    "ensure_lidar_columns",
    "ensure_mmwave_columns",
    "required_columns_error",
    "sorted_numbered_columns",
    "validate_required_columns",
]
