from pathlib import Path

import numpy as np
from PIL import Image
import torch

from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.data.transform_ops.image_cache import ImageDerivedCache


DEFAULT_IMAGE_PROFILE = "rgb_imagenet"
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
    image_cache: ImageDerivedCache | None = None,
) -> torch.Tensor:
    transform = transform or build_rgb_imagenet_transform(image_size)
    selected = list(rgb_paths[-seq_len:])
    frames = []
    for rel_path in selected:
        frame = image_cache.load(data_root, rel_path) if image_cache is not None else None
        if frame is None:
            image = read_image_array(joined_resource(data_root, rel_path))
            frame = transform(image)
            if image_cache is not None:
                image_cache.store(data_root, rel_path, frame)
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


__all__ = [
    "DEFAULT_IMAGE_PROFILE",
    "IMAGENET_RGB_MEAN",
    "IMAGENET_RGB_STD",
    "build_image_transform",
    "build_rgb_imagenet_transform",
    "load_rgb_imagenet_frames",
    "read_image_array",
]
