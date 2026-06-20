import os
from pathlib import Path
from typing import Any

import pandas as pd


def _ensure_csi_columns(data_root: str | Path, csv_name: str, scenario: str) -> str:
    root = Path(data_root)
    csv_path = Path(csv_name)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not csv_path.exists():
        return str(csv_name)
    frame = pd.read_csv(csv_path)
    if any(str(col).startswith("csi") for col in frame.columns):
        return str(csv_path.resolve())
    beam_cols = _numbered_columns(frame.columns, "beam")
    if not beam_cols:
        return str(csv_path.resolve())
    output_path = csv_path.with_name(f"{csv_path.stem}_with_csi{csv_path.suffix}")
    if _derived_csv_is_complete(output_path, prefix="csi", expected_rows=len(frame), expected_count=len(beam_cols)):
        return str(output_path.resolve())
    manifest_path = root / "Prepared" / scenario / "manifests" / "frame_manifest.csv"
    if not manifest_path.exists():
        raise ValueError(
            f"CSI is enabled for MMW dataset but {csv_path} has no csi columns and manifest is missing: "
            f"{manifest_path}"
        )
    manifest = pd.read_csv(manifest_path)
    if "beam_power_path" not in manifest.columns or "channel_path" not in manifest.columns:
        raise ValueError(
            f"Cannot derive CSI columns from {manifest_path}; expected beam_power_path and channel_path columns."
        )
    channel_by_beam = {
        _norm_path(row["beam_power_path"]): str(row["channel_path"])
        for _, row in manifest.iterrows()
        if str(row.get("beam_power_path", "")).strip() and str(row.get("channel_path", "")).strip()
    }
    missing: list[str] = []
    for idx, beam_col in enumerate(beam_cols, start=1):
        csi_values = []
        for value in frame[beam_col].tolist():
            key = _norm_path(value)
            channel_path = channel_by_beam.get(key)
            if channel_path is None:
                missing.append(str(value))
                channel_path = "-99"
            csi_values.append(channel_path)
        frame[f"csi{idx}"] = csi_values
    if missing:
        examples = ", ".join(missing[:3])
        raise ValueError(
            f"Could not derive CSI paths for {len(missing)} beam paths in {csv_path}; examples: {examples}."
        )
    _write_csv_atomic(frame, output_path)
    return str(output_path.resolve())

def _ensure_bs_gps_columns(data_root: str | Path, csv_name: str, scenario: str) -> str:
    root = Path(data_root)
    csv_path = Path(csv_name)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not csv_path.exists():
        return str(csv_name)
    frame = pd.read_csv(csv_path)
    if any(str(col).startswith("bs_gps") for col in frame.columns):
        return str(csv_path.resolve())
    gps_cols = _numbered_columns(frame.columns, "gps")
    if not gps_cols:
        return str(csv_path.resolve())
    output_path = csv_path.with_name(f"{csv_path.stem}_with_bs_gps{csv_path.suffix}")
    if _derived_csv_is_complete(output_path, prefix="bs_gps", expected_rows=len(frame), expected_count=len(gps_cols)):
        return str(output_path.resolve())
    for gps_col in gps_cols:
        suffix = gps_col[len("gps") :]
        frame[f"bs_gps{suffix}"] = [
            _rsu_gps_path_for_value(value, scenario)
            for value in frame[gps_col].tolist()
        ]
    _write_csv_atomic(frame, output_path)
    return str(output_path.resolve())

def _ensure_radar_columns(data_root: str | Path, csv_name: str, scenario: str) -> str:
    root = Path(data_root)
    csv_path = Path(csv_name)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not csv_path.exists():
        return str(csv_name)
    frame = pd.read_csv(csv_path)
    if any(str(col).startswith("radar") for col in frame.columns):
        return str(csv_path.resolve())
    beam_cols = _numbered_columns(frame.columns, "beam")
    if not beam_cols:
        return str(csv_path.resolve())
    output_path = csv_path.with_name(f"{csv_path.stem}_with_radar{csv_path.suffix}")
    if _derived_csv_is_complete(output_path, prefix="radar", expected_rows=len(frame), expected_count=len(beam_cols)):
        return str(output_path.resolve())
    missing: list[str] = []
    for beam_col in beam_cols:
        suffix = beam_col[len("beam") :]
        values = []
        for value in frame[beam_col].tolist():
            rel_path = _radar_path_for_value(value, scenario)
            if not (root / rel_path).exists():
                missing.append(rel_path)
            values.append(rel_path)
        frame[f"radar{suffix}"] = values
    if missing:
        examples = ", ".join(missing[:3])
        raise ValueError(
            f"Could not derive radar paths for {len(missing)} entries in {csv_path}; examples: {examples}. "
            "Generate MMW radar maps first with: conda run -n kd_mm_beam kd-sensing-preprocess "
            "--config configs/preprocess/mmw_radar_maps.yaml"
        )
    _write_csv_atomic(frame, output_path)
    return str(output_path.resolve())

def _derived_csv_is_complete(path: Path, *, prefix: str, expected_rows: int, expected_count: int) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    return len(frame) == int(expected_rows) and len(_numbered_columns(frame.columns, prefix)) >= int(expected_count)

def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(path)

def _numbered_columns(columns, prefix: str) -> list[str]:
    selected = []
    for col in columns:
        text = str(col)
        if not text.startswith(prefix):
            continue
        suffix = text[len(prefix) :]
        if suffix.isdigit():
            selected.append(text)
    return sorted(selected, key=lambda name: int(name[len(prefix) :]))

def _norm_path(value: object) -> str:
    return str(value).replace("\\", "/").lstrip("/")

def _rsu_gps_path_for_value(value: object, scenario: str) -> str:
    path = Path(_norm_path(value))
    frame_id = path.stem
    if not frame_id:
        return "-99"
    return (Path("Sensor_Data") / scenario / "rsu_1" / f"{frame_id}.yaml").as_posix()

def _radar_path_for_value(value: object, scenario: str) -> str:
    path = Path(_norm_path(value))
    frame_id = path.stem
    if not frame_id:
        return "-99"
    return (Path("Prepared") / scenario / "derived" / "radar_maps" / "rsu_1" / f"{frame_id}_RA.npy").as_posix()


__all__ = [
    "_ensure_bs_gps_columns",
    "_ensure_csi_columns",
    "_ensure_radar_columns",
]
