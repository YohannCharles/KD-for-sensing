import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
import uuid
import warnings

import pandas as pd
from tqdm.auto import tqdm

from kd_sensing.data.layouts import deepsense6g_image_cache_root, mmw_image_cache_root, runtime_cache_root
from kd_sensing.data.transform_ops.image import build_rgb_imagenet_transform, read_image_array
from kd_sensing.data.transform_ops.image_cache import (
    IMAGE_DERIVED_CACHE_VERSION,
    ImageDerivedCache,
    ImageDerivedCacheConfig,
)
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.registries import PREPROCESSORS
from kd_sensing.utils.paths import resolve_path


def prewarm_image_derived_cache(
    csv_path: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    *,
    csv_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    data_root: str | Path,
    cache_dir: str | Path | None = None,
    image_prefix: str = "camera",
    image_columns: list[str] | tuple[str, ...] | None = None,
    image_size: list[int] | tuple[int, int] = (224, 224),
    image_profile: str = "rgb_imagenet",
    transform_version: str = IMAGE_DERIVED_CACHE_VERSION,
    policy: str = "auto",
    overwrite: bool = False,
    progress: bool = True,
    max_failure_rate: float = 0.0,
) -> dict[str, Any]:
    if policy not in {"auto", "rebuild"}:
        raise ValueError("image-derived cache prewarm requires policy 'auto' or 'rebuild'.")
    resolved_csv_paths = [resolve_path(path) for path in _normalize_csv_paths(csv_path, csv_paths)]
    root = resolve_path(data_root)
    cache_candidate = _resolve_image_cache_root(root, cache_dir)
    if cache_candidate.is_symlink():
        raise ValueError(f"Image cache root must not be a symbolic link: {cache_candidate}")
    cache_root = cache_candidate.resolve()
    _validate_cache_boundaries(root, cache_root, resolved_csv_paths)
    transform = build_rgb_imagenet_transform(image_size)
    rel_paths: set[str] = set()
    for csv_item in resolved_csv_paths:
        frame = pd.read_csv(csv_item, na_values="").fillna("")
        selected_columns = list(image_columns or _sorted_prefixed_columns(frame.columns, image_prefix))
        if not selected_columns:
            raise ValueError(f"No image columns with prefix '{image_prefix}' found in {csv_item}.")
        for column in selected_columns:
            for value in frame[column].tolist():
                text = str(value).strip()
                if text and text != "-99":
                    rel_paths.add(_normalized_resource_identity(root, text))
    scanned = len(rel_paths)
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{cache_root.name}.stage-", dir=cache_root.parent) as temporary:
        staged_cache_root = Path(temporary) / "payload"
        if cache_root.exists():
            if not cache_root.is_dir():
                raise ValueError(f"Image cache root must be a directory: {cache_root}")
            shutil.copytree(cache_root, staged_cache_root, symlinks=True)
        else:
            staged_cache_root.mkdir()
        cache = ImageDerivedCache(
            ImageDerivedCacheConfig(
                cache_dir=staged_cache_root,
                policy="rebuild" if overwrite or policy == "rebuild" else "auto",
                image_profile=str(image_profile),
                image_size=(int(image_size[0]), int(image_size[1])),
                transform_version=str(transform_version),
            )
        )
        skipped = 0
        failures = 0
        failure_examples: list[dict[str, str]] = []
        iterator = sorted(rel_paths)
        if progress:
            iterator = tqdm(iterator, desc="Image-derived cache", unit="frame")
        for rel_path in iterator:
            try:
                if not overwrite and cache.load(root, rel_path) is not None:
                    skipped += 1
                    continue
                image = read_image_array(joined_resource(root, rel_path))
                tensor = transform(image)
                if cache.store(root, rel_path, tensor) is None:
                    failures += 1
                    if len(failure_examples) < 20:
                        failure_examples.append({"path": rel_path, "reason": "cache store failed"})
            except Exception as exc:  # noqa: BLE001 - aggregate bounded per-resource failures.
                failures += 1
                if len(failure_examples) < 20:
                    failure_examples.append({"path": rel_path, "reason": str(exc)})
        _validate_batch_outcome(
            "image-derived cache preprocessing",
            attempted=scanned,
            succeeded=int(cache.generated + skipped),
            failed=failures,
            failures=failure_examples,
            max_failure_rate=max_failure_rate,
        )
        report = {
            "type": "image_derived_cache",
            "csv_paths": [str(path) for path in resolved_csv_paths],
            "data_root": str(root),
            "cache_dir": str(cache_root),
            "scanned": int(scanned),
            "generated": int(cache.generated),
            "skipped": int(skipped),
            "failed": int(failures),
            "failures": failure_examples,
            "coverage": float((cache.generated + skipped) / scanned) if scanned else 1.0,
            "cache_total_bytes": int(cache.summary()["cache_total_bytes"]),
            "image_profile": str(image_profile),
            "image_size": [int(image_size[0]), int(image_size[1])],
            "transform_version": str(transform_version),
        }
        (staged_cache_root / "prewarm_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish_directory(staged_cache_root, cache_root)
    return report


def _normalized_resource_identity(data_root: Path, raw_path: str) -> str:
    text = str(raw_path).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    candidate = (data_root / text.lstrip("/")).resolve()
    try:
        return candidate.relative_to(data_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Image input escapes data_root: {raw_path}") from exc


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


def _validate_cache_boundaries(data_root: Path, cache_root: Path, csv_paths: list[Path]) -> None:
    _ensure_disjoint(data_root, cache_root, "data root", "image cache root")
    for csv_path in csv_paths:
        _ensure_disjoint(csv_path, cache_root, "input CSV", "image cache root")


def _ensure_disjoint(left: Path, right: Path, left_name: str, right_name: str) -> None:
    left = left.resolve()
    right = right.resolve()
    if left == right or left.is_relative_to(right) or right.is_relative_to(left):
        raise ValueError(f"{left_name} and {right_name} must be disjoint: {left} vs {right}")


def _publish_directory(staged: Path, target: Path) -> None:
    backup: Path | None = None
    token = uuid.uuid4().hex
    if target.exists() or target.is_symlink():
        if target.is_symlink():
            raise ValueError(f"Refusing to replace symbolic-link image cache root: {target}")
        backup = target.with_name(f".{target.name}.{token}.backup")
        os.replace(target, backup)
    try:
        os.replace(staged, target)
    except Exception:
        if backup is not None:
            os.replace(backup, target)
        raise
    if backup is not None:
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            warnings.warn(f"Could not remove successful image cache backup {backup}: {exc}", RuntimeWarning)


def _resolve_image_cache_root(data_root: Path, cache_dir: str | Path | None) -> Path:
    if cache_dir is None:
        return _default_image_cache_root(data_root)
    path = Path(cache_dir).expanduser()
    if path.is_absolute():
        return path
    first_part = path.parts[0] if path.parts else ""
    if first_part in {"outputs", "dataset", "cache", "logs"}:
        return resolve_path(path)
    return data_root / path


def _default_image_cache_root(data_root: Path) -> Path:
    normalized = data_root.as_posix()
    match = re.search(r"(?:^|/)DeepSense6G/scenario(\d+)(?:/|$)", normalized)
    if match:
        return resolve_path(deepsense6g_image_cache_root(match.group(1)))
    match = re.search(r"(?:^|/)MMW/([^/]+)(?:/|$)", normalized)
    if match:
        return resolve_path(mmw_image_cache_root(match.group(1)))
    return resolve_path(Path(runtime_cache_root()) / "image_derived")


def _normalize_csv_paths(
    csv_path: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    csv_paths: list[str | Path] | tuple[str | Path, ...] | None,
) -> list[str | Path]:
    selected: list[str | Path] = []
    if csv_paths:
        selected.extend(csv_paths)
    if csv_path is not None:
        if isinstance(csv_path, (list, tuple)):
            selected.extend(csv_path)
        else:
            selected.append(csv_path)
    if not selected:
        raise ValueError("prewarm_image_derived_cache requires csv_path or csv_paths.")
    return selected


def _sorted_prefixed_columns(columns, prefix: str) -> list[str]:
    selected = []
    for col in columns:
        text = str(col)
        if not text.startswith(prefix):
            continue
        suffix = text[len(prefix) :]
        if suffix.isdigit():
            selected.append(text)
    return sorted(selected, key=lambda name: int(name[len(prefix) :]))


@PREPROCESSORS.register("image_derived_cache")
class ImageDerivedCachePreprocessor:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def run(self):
        return prewarm_image_derived_cache(**self.kwargs)


__all__ = ["ImageDerivedCachePreprocessor", "prewarm_image_derived_cache"]
