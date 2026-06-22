import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SENSOR_COLUMN_PREFIXES = {
    "camera": ("unit1_rgb_", "camera", "rgb"),
    "lidar": ("unit1_lidar_", "lidar"),
    "radar": ("unit1_radar_", "radar"),
    "gps": ("unit1_loc", "unit2_loc_", "gps", "bs_gps", "future_gps", "future_bs_gps"),
}
IDENTIFIER_ALIASES = {
    "scene": ("scene", "scene_id", "scenario", "scenario_id", "dataset", "dataset_id"),
    "sample": ("sample", "sample_id", "sample_index", "frame", "frame_id", "index"),
    "sequence": ("seq", "seq_id", "seq_index", "sequence", "sequence_id"),
    "timestamp": ("timestamp", "time", "utc_time", "frame_time"),
}
LABEL_EXACT_NAMES = {
    "label",
    "beam_label",
    "target_label",
    "target_beam",
    "true_beam",
    "optimal_beam",
    "future_beam_label",
    "label_beam_future",
}


@dataclass(frozen=True)
class CsvFieldResolution:
    camera_columns: tuple[str, ...]
    lidar_columns: tuple[str, ...]
    radar_columns: tuple[str, ...]
    gps_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    label_path_columns: tuple[str, ...]
    scene_columns: tuple[str, ...]
    sample_columns: tuple[str, ...]
    sequence_columns: tuple[str, ...]
    timestamp_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_csv_fields(columns: Iterable[str]) -> CsvFieldResolution:
    names = [str(column) for column in columns]
    labels = []
    label_paths = []
    modality_columns: dict[str, list[str]] = {key: [] for key in SENSOR_COLUMN_PREFIXES}
    identifiers: dict[str, list[str]] = {key: [] for key in IDENTIFIER_ALIASES}

    for column in names:
        lower = column.lower()
        for modality, prefixes in SENSOR_COLUMN_PREFIXES.items():
            if _matches_any_prefix(lower, prefixes, numbered_ok=modality != "gps"):
                modality_columns[modality].append(column)
                break
        if lower in LABEL_EXACT_NAMES or lower.startswith("future_beam_label"):
            labels.append(column)
        elif _is_beam_path_column(lower):
            label_paths.append(column)
        for name, aliases in IDENTIFIER_ALIASES.items():
            if lower in aliases or any(lower.startswith(alias + "_") for alias in aliases):
                identifiers[name].append(column)
                break

    return CsvFieldResolution(
        camera_columns=tuple(_sort_field_names(modality_columns["camera"])),
        lidar_columns=tuple(_sort_field_names(modality_columns["lidar"])),
        radar_columns=tuple(_sort_field_names(modality_columns["radar"])),
        gps_columns=tuple(_sort_field_names(modality_columns["gps"])),
        label_columns=tuple(_sort_field_names(labels)),
        label_path_columns=tuple(_sort_field_names(label_paths)),
        scene_columns=tuple(_sort_field_names(identifiers["scene"])),
        sample_columns=tuple(_sort_field_names(identifiers["sample"])),
        sequence_columns=tuple(_sort_field_names(identifiers["sequence"])),
        timestamp_columns=tuple(_sort_field_names(identifiers["timestamp"])),
    )


def check_dataset(
    data_root: str | Path,
    csv: str | Path,
    *,
    scene: str | int | None = None,
    num_beams: int = 64,
    beam_shift: int = 0,
    max_missing_examples: int = 5,
) -> dict[str, Any]:
    root = Path(data_root)
    csv_path = Path(csv)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    fields = resolve_csv_fields(frame.columns)
    sensor_stats = {
        "camera": _path_stats(frame, root, fields.camera_columns, max_examples=max_missing_examples),
        "lidar": _path_stats(frame, root, fields.lidar_columns, max_examples=max_missing_examples),
        "radar": _path_stats(frame, root, fields.radar_columns, max_examples=max_missing_examples),
        "gps": _path_stats(frame, root, fields.gps_columns, max_examples=max_missing_examples),
    }
    label_report = _label_report(
        frame,
        root,
        fields.label_columns,
        fields.label_path_columns,
        num_beams=int(num_beams),
        beam_shift=int(beam_shift),
        max_examples=max_missing_examples,
    )
    identifiers = _identifier_report(frame, fields)
    missing_sensor_refs = sum(int(item["missing_count"]) for item in sensor_stats.values())
    ok = bool(
        label_report["invalid_count"] == 0
        and missing_sensor_refs == 0
        and (fields.label_columns or fields.label_path_columns)
    )
    return {
        "ok": ok,
        "dataset_family": "BeamBench/DeepSense6G",
        "mock_data": _infer_mock_data(frame, root),
        "data_root": str(root),
        "csv": str(csv_path),
        "csv_exists": True,
        "row_count": int(len(frame)),
        "scene_argument": None if scene is None else str(scene),
        "num_beams": int(num_beams),
        "beam_shift": int(beam_shift),
        "fields": fields.to_dict(),
        "sensor_files": sensor_stats,
        "labels": label_report,
        "identifiers": identifiers,
        "missing_sensor_reference_count": int(missing_sensor_refs),
        "notes": [
            "This checker is read-only; it does not move, delete, or generate real dataset files.",
            "MOCK data may pass smoke checks but must not be reported as real BeamBench reproduction.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check BeamBench/DeepSense6G CSV fields and referenced files.")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root used to resolve relative CSV paths.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV path or path relative to --data-root.")
    parser.add_argument("--scene", type=str, help="Optional scene id/name to record in the report.")
    parser.add_argument("--num-beams", type=int, default=64, help="Number of valid beam classes.")
    parser.add_argument("--beam-shift", type=int, default=0, help="Subtract this shift before validating labels.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--max-missing-examples", type=int, default=5, help="Missing path examples per modality.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Kept for workflow clarity; the checker is always read-only.",
    )
    args = parser.parse_args(argv)
    try:
        report = check_dataset(
            args.data_root,
            args.csv,
            scene=args.scene,
            num_beams=args.num_beams,
            beam_shift=args.beam_shift,
            max_missing_examples=args.max_missing_examples,
        )
        return_code = 0 if report["ok"] else 1
    except Exception as exc:
        report = {
            "ok": False,
            "csv_exists": False,
            "data_root": str(args.data_root),
            "csv": str(args.csv),
            "error": str(exc),
        }
        return_code = 2
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return return_code


def _path_stats(
    frame: pd.DataFrame,
    root: Path,
    columns: Iterable[str],
    *,
    max_examples: int,
) -> dict[str, Any]:
    total = 0
    existing = 0
    missing = 0
    examples: list[str] = []
    used_columns = []
    for column in columns:
        if column not in frame.columns:
            continue
        refs = [_clean_cell(value) for value in frame[column].tolist()]
        refs = [value for value in refs if _looks_like_path(value)]
        if not refs:
            continue
        used_columns.append(column)
        for value in refs:
            total += 1
            path = _resolve_data_path(root, value)
            if path.exists():
                existing += 1
            else:
                missing += 1
                if len(examples) < max_examples:
                    examples.append(str(path))
    ratio = 0.0 if total == 0 else float(missing / total)
    return {
        "columns": list(used_columns),
        "column_count": int(len(used_columns)),
        "total_references": int(total),
        "existing_count": int(existing),
        "missing_count": int(missing),
        "missing_ratio": ratio,
        "missing_examples": examples,
    }


def _label_report(
    frame: pd.DataFrame,
    root: Path,
    label_columns: Iterable[str],
    label_path_columns: Iterable[str],
    *,
    num_beams: int,
    beam_shift: int,
    max_examples: int,
) -> dict[str, Any]:
    raw_values: list[int] = []
    sources: dict[str, int] = {}
    unreadable_examples: list[str] = []
    for column in label_columns:
        if column not in frame.columns:
            continue
        for value in frame[column].tolist():
            label = _coerce_label(value)
            if label is None:
                continue
            raw_values.append(label)
            sources[column] = sources.get(column, 0) + 1
    for column in label_path_columns:
        if column not in frame.columns:
            continue
        for value in frame[column].tolist():
            cleaned = _clean_cell(value)
            if not _looks_like_path(cleaned):
                continue
            path = _resolve_data_path(root, cleaned)
            try:
                label = _label_from_file(path)
            except Exception as exc:
                if len(unreadable_examples) < max_examples:
                    unreadable_examples.append(f"{path}: {exc}")
                continue
            raw_values.append(label)
            sources[column] = sources.get(column, 0) + 1
    shifted = [int(value) - int(beam_shift) for value in raw_values]
    invalid = [value for value in shifted if value < 0 or value >= int(num_beams)]
    invalid_examples = [
        {"raw": int(raw), "shifted": int(raw) - int(beam_shift)}
        for raw in raw_values
        if int(raw) - int(beam_shift) < 0 or int(raw) - int(beam_shift) >= int(num_beams)
    ][:max_examples]
    return {
        "columns": list(label_columns),
        "path_columns": list(label_path_columns),
        "source_counts": sources,
        "count": int(len(raw_values)),
        "invalid_count": int(len(invalid)),
        "raw_min": None if not raw_values else int(min(raw_values)),
        "raw_max": None if not raw_values else int(max(raw_values)),
        "shifted_min": None if not shifted else int(min(shifted)),
        "shifted_max": None if not shifted else int(max(shifted)),
        "beam_shift": int(beam_shift),
        "num_beams": int(num_beams),
        "basis_inference": _basis_inference(raw_values, num_beams=num_beams),
        "invalid_examples": invalid_examples,
        "unreadable_label_path_examples": unreadable_examples,
    }


def _identifier_report(frame: pd.DataFrame, fields: CsvFieldResolution) -> dict[str, Any]:
    return {
        "scene": _column_summary(frame, fields.scene_columns),
        "sample": _column_summary(frame, fields.sample_columns),
        "sequence": _column_summary(frame, fields.sequence_columns),
        "timestamp": _column_summary(frame, fields.timestamp_columns),
    }


def _column_summary(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, Any]:
    result = {"columns": list(columns), "available": bool(tuple(columns)), "unique_counts": {}}
    for column in columns:
        if column in frame.columns:
            result["unique_counts"][column] = int(frame[column].nunique(dropna=True))
    return result


def _label_from_file(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError("missing label file")
    if path.suffix.lower() in {".npy", ".npz"}:
        payload = np.load(path)
        values = payload[payload.files[0]] if hasattr(payload, "files") else payload
    else:
        values = np.loadtxt(path)
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("empty label file")
    if array.size == 1:
        return int(round(float(array[0])))
    return int(np.argmax(array))


def _basis_inference(values: list[int], *, num_beams: int) -> str:
    if not values:
        return "unavailable"
    lo = min(values)
    hi = max(values)
    if lo >= 0 and hi < int(num_beams):
        return "0-based-like"
    if lo >= 1 and hi <= int(num_beams):
        return "1-based-like"
    return "unknown"


def _infer_mock_data(frame: pd.DataFrame, root: Path) -> bool:
    if "mock_data" in frame.columns and frame["mock_data"].astype(str).str.lower().isin({"true", "1", "mock"}).any():
        return True
    marker = root / "MOCK_DATASET_MARKER.txt"
    return marker.exists()


def _resolve_data_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _coerce_label(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "-99"}:
        return None
    try:
        return int(round(float(text)))
    except ValueError:
        return None


def _clean_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _looks_like_path(value: str) -> bool:
    if not value or value.lower() in {"nan", "none", "-99"}:
        return False
    try:
        float(value)
        return False
    except ValueError:
        return True


def _matches_any_prefix(lower: str, prefixes: Iterable[str], *, numbered_ok: bool) -> bool:
    for prefix in prefixes:
        if lower == prefix or lower.startswith(prefix):
            return True
        if numbered_ok and lower.startswith(prefix.rstrip("_")) and lower[len(prefix.rstrip("_")) :].isdigit():
            return True
    return False


def _is_beam_path_column(lower: str) -> bool:
    if lower in {"beam_power_path", "future_beam_power_path"}:
        return True
    if lower.startswith("future_beam") and lower.replace("future_beam", "").isdigit():
        return True
    if lower.startswith("beam") and lower.replace("beam", "").isdigit():
        return True
    return False


def _sort_field_names(names: Iterable[str]) -> list[str]:
    return sorted(dict.fromkeys(names), key=_field_sort_key)


def _field_sort_key(name: str) -> tuple[str, int, str]:
    digits = ""
    for char in reversed(name):
        if char.isdigit():
            digits = char + digits
        else:
            break
    return (name[: len(name) - len(digits)], int(digits or -1), name)


if __name__ == "__main__":
    raise SystemExit(main())
