import json
from pathlib import Path
import re
from typing import Any

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
) -> dict[str, Any]:
    if policy not in {"auto", "rebuild"}:
        raise ValueError("image-derived cache prewarm requires policy 'auto' or 'rebuild'.")
    resolved_csv_paths = _normalize_csv_paths(csv_path, csv_paths)
    root = resolve_path(data_root)
    cache_root = _resolve_image_cache_root(root, cache_dir)
    cache = ImageDerivedCache(
        ImageDerivedCacheConfig(
            cache_dir=cache_root,
            policy="rebuild" if overwrite or policy == "rebuild" else "auto",
            image_profile=str(image_profile),
            image_size=(int(image_size[0]), int(image_size[1])),
            transform_version=str(transform_version),
        )
    )
    transform = build_rgb_imagenet_transform(image_size)
    rel_paths: set[str] = set()
    for csv_item in resolved_csv_paths:
        frame = pd.read_csv(resolve_path(csv_item), na_values="").fillna("")
        selected_columns = list(image_columns or _sorted_prefixed_columns(frame.columns, image_prefix))
        if not selected_columns:
            raise ValueError(f"No image columns with prefix '{image_prefix}' found in {csv_item}.")
        rel_paths.update(
            str(value).strip()
            for column in selected_columns
            for value in frame[column].tolist()
            if str(value).strip() and str(value).strip() != "-99"
        )
    scanned = len(rel_paths)
    skipped = 0
    failures = 0
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
        except Exception:
            failures += 1
    report = {
        "type": "image_derived_cache",
        "csv_paths": [str(resolve_path(path)) for path in resolved_csv_paths],
        "data_root": str(root),
        "cache_dir": str(cache_root),
        "scanned": int(scanned),
        "generated": int(cache.generated),
        "skipped": int(skipped),
        "failed": int(failures + cache.failures),
        "coverage": float((cache.generated + skipped) / scanned) if scanned else 1.0,
        "cache_total_bytes": int(cache.summary()["cache_total_bytes"]),
        "image_profile": str(image_profile),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "transform_version": str(transform_version),
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "prewarm_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


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
