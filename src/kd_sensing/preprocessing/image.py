from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from kd_sensing.data.transforms import (
    build_image_transform,
    build_motion_mask_pair,
    atomic_save_npy,
    image_motion_cache_path,
    parameterized_image_motion_cache_dir,
    write_image_motion_cache_metadata,
)
from kd_sensing.registries import PREPROCESSORS
from kd_sensing.utils.paths import resolve_path


def generate_image_motion_cache(
    csv_path: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    csv_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    data_root: str | Path = "dataset/scenario32",
    cache_dir: str | Path = "dataset/scenario32/image_motion_cache",
    camera_prefix: str = "camera",
    camera_columns: list[str] | tuple[str, ...] | None = None,
    image_size: list[int] | tuple[int, int] = (224, 224),
    gaussian_sigma: float = 1.0,
    threshold_ratio: float = 0.1,
    threshold_strategy: str = "relative_max",
    grayscale: str = "rgb2gray",
    cache_version: str = "v1",
    overwrite: bool = False,
    progress: bool = True,
) -> dict[str, str | int]:
    resolved_csv_paths = _normalize_csv_paths(csv_path, csv_paths)
    data_root = resolve_path(data_root)
    cache_dir = parameterized_image_motion_cache_dir(
        resolve_path(cache_dir),
        image_size=image_size,
        gaussian_sigma=gaussian_sigma,
        threshold_ratio=threshold_ratio,
        threshold_strategy=threshold_strategy,
        grayscale=grayscale,
        cache_version=cache_version,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    pairs: set[tuple[str, str]] = set()
    for csv_item in resolved_csv_paths:
        frame = pd.read_csv(resolve_path(csv_item), na_values="").fillna("")
        selected_columns = list(camera_columns or _sorted_prefixed_columns(frame.columns, camera_prefix))
        if not selected_columns:
            raise ValueError(f"No image columns with prefix '{camera_prefix}' found in {csv_item}.")
        for _, row in frame.iterrows():
            paths = [str(row[column]).strip() for column in selected_columns]
            paths = [path for path in paths if path and path != "-99"]
            pairs.update(zip(paths[:-1], paths[1:]))

    transform = build_image_transform(image_size)
    generated = 0
    skipped = 0
    iterator = sorted(pairs)
    if progress:
        iterator = tqdm(iterator, desc="Image motion cache", unit="pair")
    for previous_rel_path, current_rel_path in iterator:
        npy_path = image_motion_cache_path(cache_dir, previous_rel_path, current_rel_path)
        if npy_path.exists() and not overwrite:
            skipped += 1
            continue
        mask = build_motion_mask_pair(
            data_root,
            previous_rel_path,
            current_rel_path,
            transform=transform,
            image_size=image_size,
            gaussian_sigma=gaussian_sigma,
            threshold_ratio=threshold_ratio,
            threshold_strategy=threshold_strategy,
            grayscale=grayscale,
        )
        atomic_save_npy(npy_path, mask)
        generated += 1

    write_image_motion_cache_metadata(
        cache_dir,
        data_root=data_root,
        csv_paths=[str(resolve_path(path)) for path in resolved_csv_paths],
        generated=generated,
        skipped=skipped,
        image_size=image_size,
        gaussian_sigma=gaussian_sigma,
        threshold_ratio=threshold_ratio,
        threshold_strategy=threshold_strategy,
        grayscale=grayscale,
        cache_version=cache_version,
    )
    return {
        "cache_dir": str(cache_dir),
        "count": len(pairs),
        "generated": generated,
        "skipped": skipped,
    }


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
        raise ValueError("generate_image_motion_cache requires csv_path or csv_paths.")
    return selected


def _sorted_prefixed_columns(columns, prefix: str) -> list[str]:
    selected = []
    for col in columns:
        if not col.startswith(prefix):
            continue
        suffix = col[len(prefix) :]
        if suffix.isdigit():
            selected.append(col)
    return sorted(selected, key=lambda name: int(name[len(prefix) :]))


@PREPROCESSORS.register("image_motion_cache")
class ImageMotionCachePreprocessor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        return generate_image_motion_cache(**self.kwargs)
