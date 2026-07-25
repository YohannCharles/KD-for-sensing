import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from kd_sensing.data.transform_ops.io import joined_resource


DEFAULT_IMAGE_PROFILE = "rgb_imagenet"
IMAGE_DERIVED_CACHE_VERSION = "rgb_imagenet_derived_v1"
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)


def read_image_array(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image).copy()


def build_image_transform(image_size: list[int] | tuple[int, int] = (224, 224)):
    height, width = tuple(image_size)

    def transform(array):
        image = Image.fromarray(array)
        return image.resize((width, height))

    return transform


def build_rgb_imagenet_transform(image_size: list[int] | tuple[int, int] = (224, 224)):
    height, width = tuple(int(value) for value in image_size)
    mean = torch.tensor(IMAGENET_RGB_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_RGB_STD, dtype=torch.float32).view(3, 1, 1)

    def transform(array) -> torch.Tensor:
        image = Image.fromarray(np.asarray(array)).convert("RGB").resize((width, height), Image.BILINEAR)
        values = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(values).permute(2, 0, 1).contiguous()
        return (tensor - mean) / std

    return transform


def load_rgb_imagenet_frames(
    data_root: str | Path,
    rgb_paths: list[str],
    seq_len: int,
    transform=None,
    *,
    image_size: list[int] | tuple[int, int] = (224, 224),
    cache_dir: str | Path | None = None,
    strict_cache: bool = False,
) -> torch.Tensor:
    transform = transform or build_rgb_imagenet_transform(image_size)
    selected = list(rgb_paths[-seq_len:])
    frames = []
    for rel_path in selected:
        frame = None
        if cache_dir is not None:
            try:
                frame = load_rgb_imagenet_cache_frame(
                    data_root,
                    rel_path,
                    cache_dir=cache_dir,
                    image_size=image_size,
                )
            except (FileNotFoundError, ValueError):
                if strict_cache:
                    raise
        if frame is None:
            image = read_image_array(joined_resource(data_root, rel_path))
            frame = transform(image)
        if not torch.is_tensor(frame):
            raise TypeError("RGB/ImageNet transform must return a torch.Tensor.")
        if frame.shape != (3, int(image_size[0]), int(image_size[1])):
            raise ValueError(
                "RGB/ImageNet frames must have shape "
                f"[3, {int(image_size[0])}, {int(image_size[1])}], got {tuple(frame.shape)}."
            )
        frames.append(frame.to(dtype=torch.float32))
    if not frames:
        height, width = int(image_size[0]), int(image_size[1])
        return torch.empty((0, 3, height, width), dtype=torch.float32)
    return torch.stack(frames, dim=0)


def image_derived_cache_path(
    data_root: str | Path,
    rel_path: str,
    *,
    cache_dir: str | Path,
    image_size: list[int] | tuple[int, int] = (224, 224),
    image_profile: str = DEFAULT_IMAGE_PROFILE,
    transform_version: str = IMAGE_DERIVED_CACHE_VERSION,
) -> tuple[Path, dict[str, Any]]:
    source = joined_resource(data_root, rel_path)
    stat = source.stat()
    fingerprint = {
        "path": str(source),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    height, width = (int(value) for value in image_size)
    payload = {
        "source_path": str(rel_path),
        "source_fingerprint": fingerprint,
        "image_size": [height, width],
        "image_profile": str(image_profile),
        "transform_version": str(transform_version),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    path = Path(cache_dir) / str(image_profile) / f"{height}x{width}" / f"{digest}.npy"
    return path, payload


def load_rgb_imagenet_cache_frame(
    data_root: str | Path,
    rel_path: str,
    *,
    cache_dir: str | Path,
    image_size: list[int] | tuple[int, int] = (224, 224),
) -> torch.Tensor:
    path, expected = image_derived_cache_path(
        data_root,
        rel_path,
        cache_dir=cache_dir,
        image_size=image_size,
    )
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Strict RGB cache miss for {rel_path}: {path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid RGB cache metadata: {metadata_path}") from exc
    if (
        metadata.get("version") != "image_derived_cache_metadata_v1"
        or any(metadata.get(key) != value for key, value in expected.items())
    ):
        raise ValueError(f"RGB cache metadata mismatch for {rel_path}: {metadata_path}")
    array = np.load(path, allow_pickle=False)
    expected_shape = (3, int(image_size[0]), int(image_size[1]))
    if array.shape != expected_shape or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"RGB cache tensor mismatch for {rel_path}: {path}")
    return torch.from_numpy(array)


__all__ = [
    "DEFAULT_IMAGE_PROFILE",
    "IMAGE_DERIVED_CACHE_VERSION",
    "IMAGENET_RGB_MEAN",
    "IMAGENET_RGB_STD",
    "build_image_transform",
    "build_rgb_imagenet_transform",
    "image_derived_cache_path",
    "load_rgb_imagenet_cache_frame",
    "load_rgb_imagenet_frames",
    "read_image_array",
]
