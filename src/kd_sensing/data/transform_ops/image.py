from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import io
from skimage.color import rgb2gray
import torch

from kd_sensing.data.transform_ops.io import atomic_save_npy, joined_resource


DEFAULT_IMAGE_MOTION_CACHE_VERSION = "v1"
DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY = "relative_max"
DEFAULT_IMAGE_MOTION_GRAYSCALE = "rgb2gray"


def build_image_transform(image_size: list[int] | tuple[int, int] = (224, 224)):
    height, width = tuple(image_size)

    def transform(array):
        image = Image.fromarray(array)
        return image.resize((width, height))

    return transform


def image_motion_cache_config_hash(
    *,
    image_size: list[int] | tuple[int, int] = (224, 224),
    gaussian_sigma: float = 1.0,
    threshold_ratio: float = 0.1,
    threshold_strategy: str = DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY,
    grayscale: str = DEFAULT_IMAGE_MOTION_GRAYSCALE,
    cache_version: str = DEFAULT_IMAGE_MOTION_CACHE_VERSION,
) -> str:
    payload = image_motion_cache_config_payload(
        image_size=image_size,
        gaussian_sigma=gaussian_sigma,
        threshold_ratio=threshold_ratio,
        threshold_strategy=threshold_strategy,
        grayscale=grayscale,
        cache_version=cache_version,
    )
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"motion_{digest}"


def image_motion_cache_config_payload(
    *,
    image_size: list[int] | tuple[int, int] = (224, 224),
    gaussian_sigma: float = 1.0,
    threshold_ratio: float = 0.1,
    threshold_strategy: str = DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY,
    grayscale: str = DEFAULT_IMAGE_MOTION_GRAYSCALE,
    cache_version: str = DEFAULT_IMAGE_MOTION_CACHE_VERSION,
) -> dict[str, object]:
    return {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "gaussian_sigma": float(gaussian_sigma),
        "threshold_ratio": float(threshold_ratio),
        "threshold_strategy": str(threshold_strategy),
        "grayscale": str(grayscale),
        "cache_version": str(cache_version),
    }


def parameterized_image_motion_cache_dir(
    cache_dir: str | Path,
    *,
    image_size: list[int] | tuple[int, int] = (224, 224),
    gaussian_sigma: float = 1.0,
    threshold_ratio: float = 0.1,
    threshold_strategy: str = DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY,
    grayscale: str = DEFAULT_IMAGE_MOTION_GRAYSCALE,
    cache_version: str = DEFAULT_IMAGE_MOTION_CACHE_VERSION,
) -> Path:
    return Path(cache_dir) / image_motion_cache_config_hash(
        image_size=image_size,
        gaussian_sigma=gaussian_sigma,
        threshold_ratio=threshold_ratio,
        threshold_strategy=threshold_strategy,
        grayscale=grayscale,
        cache_version=cache_version,
    )


def image_motion_cache_path(cache_dir: str | Path, previous_rel_path: str, current_rel_path: str) -> Path:
    pair_key = f"{str(previous_rel_path).lstrip('/')}->{str(current_rel_path).lstrip('/')}"
    digest = hashlib.sha1(pair_key.encode("utf-8")).hexdigest()[:16]
    current_safe = str(current_rel_path).lstrip("/").replace("\\", "/").replace("/", "__").replace("..", "__")
    stem = Path(current_safe).with_suffix("").name
    return Path(cache_dir) / f"{stem}_{digest}.npy"


def write_image_motion_cache_metadata(
    cache_dir: str | Path,
    *,
    data_root: str | Path,
    csv_paths: list[str] | tuple[str, ...] | None = None,
    generated: int = 0,
    skipped: int = 0,
    image_size: list[int] | tuple[int, int] = (224, 224),
    gaussian_sigma: float = 1.0,
    threshold_ratio: float = 0.1,
    threshold_strategy: str = DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY,
    grayscale: str = DEFAULT_IMAGE_MOTION_GRAYSCALE,
    cache_version: str = DEFAULT_IMAGE_MOTION_CACHE_VERSION,
) -> Path:
    target_dir = Path(cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "type": "image_motion_cache",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_root": str(data_root),
        "csv_paths": list(csv_paths or []),
        "generated": int(generated),
        "skipped": int(skipped),
        "parameters": image_motion_cache_config_payload(
            image_size=image_size,
            gaussian_sigma=gaussian_sigma,
            threshold_ratio=threshold_ratio,
            threshold_strategy=threshold_strategy,
            grayscale=grayscale,
            cache_version=cache_version,
        ),
    }
    path = target_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def build_motion_mask_pair(
    data_root: str | Path,
    previous_rel_path: str,
    current_rel_path: str,
    *,
    transform=None,
    image_size: list[int] | tuple[int, int] = (224, 224),
    gaussian_sigma: float = 1.0,
    threshold_ratio: float = 0.1,
    threshold_strategy: str = DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY,
    grayscale: str = DEFAULT_IMAGE_MOTION_GRAYSCALE,
) -> np.ndarray:
    if threshold_strategy != DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY:
        raise ValueError(f"Unsupported image motion threshold strategy '{threshold_strategy}'.")
    if grayscale != DEFAULT_IMAGE_MOTION_GRAYSCALE:
        raise ValueError(f"Unsupported image motion grayscale mode '{grayscale}'.")
    transform = transform or build_image_transform(image_size)
    previous = _load_grayscale_image(data_root, previous_rel_path, transform, gaussian_sigma)
    current = _load_grayscale_image(data_root, current_rel_path, transform, gaussian_sigma)
    if previous.shape != current.shape:
        raise ValueError(f"Motion mask frames must share one image size, got {previous.shape} and {current.shape}.")
    diff = np.abs(current - previous)
    max_pixel_value = float(np.max(diff))
    threshold_value = float(threshold_ratio) * max_pixel_value
    return (diff > threshold_value).astype(np.uint8)


def load_motion_masks(
    data_root: str | Path,
    rgb_paths: list[str],
    seq_len: int,
    transform=None,
    *,
    image_size: list[int] | tuple[int, int] = (224, 224),
    gaussian_sigma: float = 1.0,
    threshold_ratio: float = 0.1,
    threshold_strategy: str = DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY,
    grayscale: str = DEFAULT_IMAGE_MOTION_GRAYSCALE,
    cache_dir: str | Path | None = None,
    use_cache: bool = False,
    write_cache: bool = False,
) -> torch.Tensor:
    if threshold_strategy != DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY:
        raise ValueError(f"Unsupported image motion threshold strategy '{threshold_strategy}'.")
    if grayscale != DEFAULT_IMAGE_MOTION_GRAYSCALE:
        raise ValueError(f"Unsupported image motion grayscale mode '{grayscale}'.")
    if cache_dir is not None and (use_cache or write_cache):
        return torch.tensor(
            _load_motion_masks_with_cache(
                data_root,
                rgb_paths,
                seq_len,
                transform=transform,
                image_size=image_size,
                gaussian_sigma=gaussian_sigma,
                threshold_ratio=threshold_ratio,
                threshold_strategy=threshold_strategy,
                grayscale=grayscale,
                cache_dir=cache_dir,
                use_cache=use_cache,
                write_cache=write_cache,
            ),
            dtype=torch.float32,
        )
    transform = transform or build_image_transform(image_size)
    image_val = None
    image_motion_masks = None
    for i, rel_path in enumerate(rgb_paths[-seq_len:]):
        img = _load_grayscale_image(data_root, rel_path, transform, gaussian_sigma)
        if image_val is None:
            height, width = img.shape
            image_val = np.zeros((seq_len, height, width))
            image_motion_masks = np.zeros((seq_len - 1, height, width))
        elif img.shape != image_val.shape[1:]:
            raise ValueError(
                f"Motion mask frames must share one image size, got {img.shape} and {image_val.shape[1:]}."
            )
        image_val[i, ...] = img
        if i >= 1:
            diff = np.abs(image_val[i, ...] - image_val[i - 1, ...])
            max_pixel_value = np.max(diff)
            threshold_value = float(threshold_ratio) * max_pixel_value
            image_motion_masks[i - 1, ...] = (diff > threshold_value).astype(np.uint8)
    if image_motion_masks is None:
        image_motion_masks = np.zeros((seq_len - 1, int(image_size[0]), int(image_size[1])))
    return torch.tensor(image_motion_masks, dtype=torch.float32)


def _load_grayscale_image(
    data_root: str | Path,
    rel_path: str,
    transform,
    gaussian_sigma: float,
) -> np.ndarray:
    img = transform(io.imread(joined_resource(data_root, rel_path)))
    img = rgb2gray(np.asarray(img))
    return gaussian_filter(img, sigma=float(gaussian_sigma))


def _load_motion_masks_with_cache(
    data_root: str | Path,
    rgb_paths: list[str],
    seq_len: int,
    *,
    transform=None,
    image_size: list[int] | tuple[int, int] = (224, 224),
    gaussian_sigma: float = 1.0,
    threshold_ratio: float = 0.1,
    threshold_strategy: str = DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY,
    grayscale: str = DEFAULT_IMAGE_MOTION_GRAYSCALE,
    cache_dir: str | Path,
    use_cache: bool,
    write_cache: bool,
) -> np.ndarray:
    selected = list(rgb_paths[-seq_len:])
    height, width = int(image_size[0]), int(image_size[1])
    if len(selected) < 2:
        return np.zeros((max(seq_len - 1, 0), height, width), dtype=np.uint8)
    transform = transform or build_image_transform(image_size)
    cache_root = Path(cache_dir)
    if write_cache:
        cache_root.mkdir(parents=True, exist_ok=True)
    masks = []
    build_pair = _current_public_symbol("build_motion_mask_pair", build_motion_mask_pair)
    for previous_rel_path, current_rel_path in zip(selected[:-1], selected[1:]):
        path = image_motion_cache_path(cache_root, previous_rel_path, current_rel_path)
        if use_cache and path.exists():
            mask = np.load(path).astype(np.uint8)
        else:
            mask = build_pair(
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
            if write_cache:
                atomic_save_npy(path, mask.astype(np.uint8))
        masks.append(mask.astype(np.uint8))
    return np.stack(masks, axis=0).astype(np.uint8)


def _current_public_symbol(name: str, fallback):
    import sys

    facade = sys.modules.get("kd_sensing.data.transforms")
    return getattr(facade, name, fallback) if facade is not None else fallback


__all__ = [
    "DEFAULT_IMAGE_MOTION_CACHE_VERSION",
    "DEFAULT_IMAGE_MOTION_GRAYSCALE",
    "DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY",
    "build_image_transform",
    "build_motion_mask_pair",
    "image_motion_cache_config_hash",
    "image_motion_cache_config_payload",
    "image_motion_cache_path",
    "load_motion_masks",
    "parameterized_image_motion_cache_dir",
    "write_image_motion_cache_metadata",
]
