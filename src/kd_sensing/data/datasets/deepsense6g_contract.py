from pathlib import Path
import re
from typing import Any

from kd_sensing.modalities import normalize_modalities


SUPPORTED_BEAM_TARGET_SOURCES = ("current", "future")


def normalize_beam_target_source(value: object) -> str:
    normalized = str(value or "future").strip().lower().replace("-", "_")
    if normalized in {"future", "future_beam", "future_beam1", "next"}:
        return "future"
    if normalized in {"current", "current_beam", "beam", "beam_last", "last_beam"}:
        return "current"
    supported = ", ".join(repr(item) for item in SUPPORTED_BEAM_TARGET_SOURCES)
    raise ValueError(f"beam_target_source must be one of {supported}.")


def validate_beam_target_source_contract(source: str, *, num_pred: int, seq_len: int) -> None:
    if source == "current" and int(num_pred) > int(seq_len):
        raise ValueError("beam_target_source='current' requires num_pred <= seq_len.")


def resolve_target_beam_paths(
    input_beam_paths: list[str],
    future_beam_paths: list[str],
    *,
    source: str,
    num_pred: int,
) -> list[str]:
    normalized = normalize_beam_target_source(source)
    if normalized == "current":
        return input_beam_paths[-int(num_pred) :]
    return future_beam_paths[: int(num_pred)]


def resolve_enabled_modalities(
    enabled_modalities: list[str] | tuple[str, ...] | None,
    *,
    use_gps: bool,
    use_lidar: bool,
    use_mmwave: bool,
    use_csi: bool,
) -> tuple[str, ...]:
    if enabled_modalities is None:
        selected = ["image", "radar"]
        if use_gps:
            selected.append("gps")
        if use_lidar:
            selected.append("lidar")
        if use_mmwave:
            selected.append("mmwave")
        if use_csi:
            selected.append("csi")
    else:
        selected = [str(modality) for modality in enabled_modalities]
    return normalize_modalities(selected, context="DeepSense6G modalities")


def resolve_sequence_csv_path(
    data_root: str | Path,
    scene: object,
    *,
    root_csv: str | None,
    csv_name: str | None,
    split: str,
    train_csv_name: str | None,
    val_csv_name: str | None,
    test_csv_name: str | None,
) -> Path:
    selected_csv = root_csv or csv_name
    if selected_csv is None:
        if split == "train":
            default_csv = getattr(scene, "default_train_csv_name")
            configured_csv = train_csv_name
        elif split in {"val", "validation"}:
            if not val_csv_name:
                raise ValueError("val_csv_name is required for an independent validation split.")
            default_csv = val_csv_name
            configured_csv = val_csv_name
        else:
            default_csv = getattr(scene, "default_test_csv_name")
            configured_csv = test_csv_name
        selected_csv = configured_csv or default_csv
    path = Path(selected_csv)
    if path.is_absolute():
        return path
    return Path(data_root) / path


def resolve_beam_label_cache_mode(config: bool | str) -> str:
    if isinstance(config, bool):
        return "eager" if config else "off"
    mode = str(config).lower()
    if mode in {"true", "yes", "on"}:
        return "eager"
    if mode in {"false", "no", "off", "none"}:
        return "off"
    if mode not in {"eager", "lazy"}:
        raise ValueError("beam_label_cache must be one of eager, lazy, off, true, or false.")
    return mode


def add_path_metadata(metadata: dict[str, Any], key: str, paths: list[list[str]] | None, idx: int) -> None:
    if not paths or idx >= len(paths) or not paths[idx]:
        return
    metadata[key] = str(paths[idx][-1])


def parse_sequence_position(path: str) -> tuple[str | None, int | None]:
    text = str(path)
    seq_id = None
    frame_idx = None
    seq_match = re.search(r"(?:^|[/_-])seq(?:uence)?[_-]?([A-Za-z0-9]+)", text, flags=re.IGNORECASE)
    if seq_match:
        seq_id = seq_match.group(1)
    frame_match = re.search(
        r"(?:frame|frm|camera|radar|beam|gps|lidar|mmwave|pwr)[_-]?(\d+)",
        Path(text).stem,
        flags=re.IGNORECASE,
    )
    if frame_match:
        frame_idx = int(frame_match.group(1))
    return seq_id, frame_idx


__all__ = [
    "SUPPORTED_BEAM_TARGET_SOURCES",
    "add_path_metadata",
    "normalize_beam_target_source",
    "parse_sequence_position",
    "resolve_beam_label_cache_mode",
    "resolve_enabled_modalities",
    "resolve_sequence_csv_path",
    "resolve_target_beam_paths",
    "validate_beam_target_source_contract",
]
