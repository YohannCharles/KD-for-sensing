import hashlib
from contextlib import ExitStack
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

from kd_sensing.data.transform_ops.io import atomic_save_npy
from kd_sensing.preprocessing.radar import Doppler_Angle, Radar_Cube, Range_Angle, Range_Doppler
from kd_sensing.registries import PREPROCESSORS
from kd_sensing.utils.paths import resolve_path


def _load_radar_data(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return loadmat(path)["data"]


def process_radar_and_create_new_csv(
    csv_path: str | Path,
    data_root: str | Path,
    output_csv_path: str | Path | None = None,
    output_suffix: str = "FFT",
    test_mode: bool = False,
    test_portion: float = 0.01,
    fft_tuple: list[int] | tuple[int, int, int] = (64, 256, 128),
    max_failure_rate: float = 0.0,
) -> pd.DataFrame:
    csv_path = resolve_path(csv_path)
    data_root = resolve_path(data_root)
    suffix = _validated_output_suffix(output_suffix)
    if output_csv_path is None:
        output_csv_path = csv_path.with_name(f"{csv_path.stem}_{suffix}.csv")
    output_csv_path = resolve_path(output_csv_path)
    fft_output_dir = (data_root / "unit1" / f"radar_data_{suffix}").resolve()
    _validate_output_boundaries(csv_path, output_csv_path, fft_output_dir)
    frame = pd.read_csv(csv_path)
    if test_mode:
        frame = frame.head(max(1, int(len(frame) * test_portion)))
    radar_columns = [col for col in frame.columns if "radar" in col.lower() and "unit" in col.lower()]
    resources: dict[str, Path] = {}
    references: dict[str, list[tuple[int, str]]] = {}
    failed_files: set[str] = set()
    failures: list[dict[str, str]] = []
    attempted = 0
    frame_new = frame.copy()
    for radar_col in radar_columns:
        for idx in range(len(frame)):
            radar_path = frame.loc[idx, radar_col]
            if pd.isna(radar_path) or radar_path == -99:
                continue
            raw_path = str(radar_path)
            try:
                resource_id, full_radar_path = _radar_resource_identity(data_root, raw_path)
            except ValueError as exc:
                resource_id = f"invalid:{raw_path}"
                if resource_id not in failed_files:
                    attempted += 1
                    failed_files.add(resource_id)
                    if len(failures) < 20:
                        failures.append({"path": raw_path, "reason": str(exc)})
                continue
            _ensure_disjoint(full_radar_path, fft_output_dir, "radar input", "FFT output directory")
            _ensure_disjoint(full_radar_path, output_csv_path, "radar input", "output CSV")
            resources[resource_id] = full_radar_path
            references.setdefault(resource_id, []).append((idx, radar_col))

    names = _resource_output_names(resources, suffix)
    processed_files: dict[str, Path] = {}
    fft_output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        output_stage_root = Path(
            stack.enter_context(
                tempfile.TemporaryDirectory(prefix=f".{fft_output_dir.name}.stage-", dir=fft_output_dir.parent)
            )
        )
        csv_stage_root = Path(
            stack.enter_context(
                tempfile.TemporaryDirectory(prefix=f".{output_csv_path.name}.stage-", dir=output_csv_path.parent)
            )
        )
        staged_output_dir = output_stage_root / "payload"
        if fft_output_dir.exists():
            shutil.copytree(fft_output_dir, staged_output_dir, symlinks=True)
        else:
            staged_output_dir.mkdir()

        iterator = tqdm(sorted(resources), desc="Processing radar resources")
        for resource_id in iterator:
            full_radar_path = resources[resource_id]
            new_filename = names[resource_id]
            staged_filepath = staged_output_dir / new_filename
            attempted += 1
            try:
                smp_radar = _load_radar_data(full_radar_path)
                radar_cube = Radar_Cube(smp_radar, fft_tuple, remove_mean=True)
                if suffix == "RA":
                    output = Range_Angle(radar_cube, mean=True, log_scale=True)
                elif suffix == "RD":
                    output = Range_Doppler(radar_cube, mean=True, log_scale=True)
                elif suffix == "DA":
                    output = Doppler_Angle(radar_cube, mean=True, log_scale=True)
                else:
                    output = radar_cube
                atomic_save_npy(staged_filepath, output)
                processed_files[resource_id] = fft_output_dir / new_filename
                for idx, radar_col in references[resource_id]:
                    frame_new.loc[idx, radar_col] = f"/unit1/radar_data_{suffix}/{new_filename}"
            except Exception as exc:  # noqa: BLE001 - aggregate bounded per-resource failures.
                failed_files.add(resource_id)
                if len(failures) < 20:
                    failures.append({"path": str(full_radar_path), "reason": str(exc)})

        _validate_batch_outcome(
            "radar FFT CSV preprocessing",
            attempted=attempted,
            succeeded=len(processed_files),
            failed=len(failed_files),
            failures=failures,
            max_failure_rate=max_failure_rate,
        )
        staged_csv = csv_stage_root / output_csv_path.name
        frame_new.to_csv(staged_csv, index=False)
        _publish_paths(
            [
                (staged_output_dir, fft_output_dir),
                (staged_csv, output_csv_path),
            ]
        )
    frame_new.attrs["preprocessing_report"] = {
        "attempted": attempted,
        "succeeded": len(processed_files),
        "failed": len(failed_files),
        "failures": failures,
    }
    return frame_new


def _validated_output_suffix(value: object) -> str:
    suffix = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", suffix):
        raise ValueError("output_suffix must contain only letters, digits, '_' or '-'.")
    return suffix


def _resource_output_names(resources: dict[str, Path], suffix: str) -> dict[str, str]:
    basename_owners: dict[str, set[str]] = {}
    for resource_id in resources:
        basename_owners.setdefault(Path(resource_id).name, set()).add(resource_id)
    names: dict[str, str] = {}
    for resource_id in resources:
        source = Path(resource_id)
        collision = len(basename_owners[source.name]) > 1
        digest = f"_{hashlib.sha256(resource_id.encode('utf-8')).hexdigest()[:12]}" if collision else ""
        names[resource_id] = f"{source.stem}{digest}_{suffix}.npy"
    return names


def _validate_output_boundaries(csv_path: Path, output_csv_path: Path, output_dir: Path) -> None:
    _ensure_disjoint(csv_path, output_csv_path, "input CSV", "output CSV")
    _ensure_disjoint(csv_path, output_dir, "input CSV", "FFT output directory")
    _ensure_disjoint(output_csv_path, output_dir, "output CSV", "FFT output directory")
    for path, label in ((output_csv_path, "output CSV"), (output_dir, "FFT output directory")):
        if path.is_symlink():
            raise ValueError(f"{label} must not be a symbolic link: {path}")


def _ensure_disjoint(left: Path, right: Path, left_name: str, right_name: str) -> None:
    left = left.resolve()
    right = right.resolve()
    if left == right or left.is_relative_to(right) or right.is_relative_to(left):
        raise ValueError(f"{left_name} and {right_name} must be disjoint: {left} vs {right}")


def _publish_paths(pairs: list[tuple[Path, Path]]) -> None:
    token = uuid.uuid4().hex
    published: list[tuple[Path, Path | None]] = []
    try:
        for staged, target in pairs:
            backup: Path | None = None
            if target.exists() or target.is_symlink():
                if target.is_symlink():
                    raise ValueError(f"Refusing to replace symbolic-link output: {target}")
                backup = target.with_name(f".{target.name}.{token}.backup")
                os.replace(target, backup)
            try:
                os.replace(staged, target)
            except Exception:
                if backup is not None:
                    os.replace(backup, target)
                raise
            published.append((target, backup))
    except Exception:
        for target, backup in reversed(published):
            _remove_path(target)
            if backup is not None and backup.exists():
                os.replace(backup, target)
        raise
    for _, backup in published:
        if backup is not None:
            try:
                _remove_path(backup)
            except OSError as exc:
                warnings.warn(f"Could not remove successful radar publish backup {backup}: {exc}", RuntimeWarning)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _radar_resource_identity(data_root: Path, raw_path: str) -> tuple[str, Path]:
    text = str(raw_path).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    candidate = (data_root / text.lstrip("/")).resolve()
    root = data_root.resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Radar input escapes data_root: {raw_path}") from exc
    return relative.as_posix(), candidate


def _validate_batch_outcome(
    name: str,
    *,
    attempted: int,
    succeeded: int,
    failed: int,
    failures: list[dict[str, str]],
    max_failure_rate: float,
) -> None:
    threshold = float(max_failure_rate)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("max_failure_rate must be between 0 and 1.")
    detail = f"attempted={attempted}, succeeded={succeeded}, failed={failed}, examples={failures[:20]}"
    if attempted <= 0 or succeeded <= 0:
        raise RuntimeError(f"{name} produced zero successful resources; {detail}")
    if failed / attempted > threshold:
        raise RuntimeError(f"{name} exceeded max_failure_rate={threshold}; {detail}")
    if failed:
        warnings.warn(f"{name} completed with allowed failures; {detail}", RuntimeWarning, stacklevel=2)


@PREPROCESSORS.register("radar_fft_csv")
class CSVFFTPreprocessor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        return process_radar_and_create_new_csv(**self.kwargs)
