from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kd_sensing.data.transform_ops.io import atomic_save_npy, joined_resource


IMAGE_DERIVED_CACHE_VERSION = "rgb_imagenet_derived_v1"


@dataclass(frozen=True)
class ImageDerivedCacheConfig:
    cache_dir: Path
    policy: str = "off"
    image_profile: str = "rgb_imagenet"
    image_size: tuple[int, int] = (224, 224)
    transform_version: str = IMAGE_DERIVED_CACHE_VERSION

    @property
    def use_cache(self) -> bool:
        return self.policy in {"read_only", "auto"}

    @property
    def write_cache(self) -> bool:
        return self.policy in {"auto", "rebuild"}


class ImageDerivedCache:
    def __init__(self, config: ImageDerivedCacheConfig) -> None:
        self.config = config
        self.hits = 0
        self.misses = 0
        self.generated = 0
        self.skipped = 0
        self.failures = 0

    def load(self, data_root: str | Path, rel_path: str) -> torch.Tensor | None:
        if not self.config.use_cache:
            return None
        image_path = joined_resource(data_root, rel_path)
        fingerprint = image_fingerprint(image_path)
        cache_path = self.cache_path(rel_path, fingerprint)
        metadata_path = image_cache_metadata_path(cache_path)
        if not cache_path.exists() or not metadata_path.exists():
            self.misses += 1
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not self._metadata_matches(metadata, rel_path, fingerprint):
                self.misses += 1
                return None
            array = np.load(cache_path)
            tensor = torch.from_numpy(np.asarray(array)).to(dtype=torch.float32)
        except Exception:
            self.misses += 1
            return None
        expected_shape = (3, int(self.config.image_size[0]), int(self.config.image_size[1]))
        if tuple(tensor.shape) != expected_shape:
            self.misses += 1
            return None
        self.hits += 1
        return tensor

    def store(self, data_root: str | Path, rel_path: str, tensor: torch.Tensor) -> Path | None:
        if not self.config.write_cache:
            self.skipped += 1
            return None
        image_path = joined_resource(data_root, rel_path)
        try:
            fingerprint = image_fingerprint(image_path)
            cache_path = self.cache_path(rel_path, fingerprint)
            array = tensor.detach().cpu().to(dtype=torch.float32).numpy()
            atomic_save_npy(cache_path, array)
            metadata = image_cache_metadata(
                source_path=str(rel_path),
                source_fingerprint=fingerprint,
                image_size=self.config.image_size,
                image_profile=self.config.image_profile,
                transform_version=self.config.transform_version,
                dtype=str(array.dtype),
                shape=list(array.shape),
            )
            image_cache_metadata_path(cache_path).write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.generated += 1
            return cache_path
        except Exception:
            self.failures += 1
            return None

    def cache_path(self, rel_path: str, fingerprint: dict[str, Any]) -> Path:
        height, width = self.config.image_size
        key_payload = {
            "source_path": str(rel_path),
            "source_fingerprint": fingerprint,
            "image_size": [int(height), int(width)],
            "image_profile": self.config.image_profile,
            "transform_version": self.config.transform_version,
        }
        digest = hashlib.sha1(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.config.cache_dir / self.config.image_profile / f"{int(height)}x{int(width)}" / f"{digest}.npy"

    def summary(self) -> dict[str, Any]:
        total = self.hits + self.misses
        cache_total_bytes = _directory_size(self.config.cache_dir)
        return {
            "policy": self.config.policy,
            "cache_dir": str(self.config.cache_dir),
            "transform_version": self.config.transform_version,
            "image_profile": self.config.image_profile,
            "image_size": [int(self.config.image_size[0]), int(self.config.image_size[1])],
            "hits": int(self.hits),
            "misses": int(self.misses),
            "generated": int(self.generated),
            "skipped": int(self.skipped),
            "failures": int(self.failures),
            "coverage": float(self.hits / total) if total else None,
            "cache_total_bytes": int(cache_total_bytes),
        }

    def _metadata_matches(self, metadata: dict[str, Any], rel_path: str, fingerprint: dict[str, Any]) -> bool:
        return (
            metadata.get("version") == "image_derived_cache_metadata_v1"
            and metadata.get("source_path") == str(rel_path)
            and metadata.get("source_fingerprint") == fingerprint
            and metadata.get("image_size") == [int(self.config.image_size[0]), int(self.config.image_size[1])]
            and metadata.get("image_profile") == self.config.image_profile
            and metadata.get("transform_version") == self.config.transform_version
        )


def image_fingerprint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    return {
        "path": str(source),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def image_cache_metadata(
    *,
    source_path: str,
    source_fingerprint: dict[str, Any],
    image_size: tuple[int, int] | list[int],
    image_profile: str,
    transform_version: str,
    dtype: str,
    shape: list[int],
) -> dict[str, Any]:
    return {
        "version": "image_derived_cache_metadata_v1",
        "source_path": str(source_path),
        "source_fingerprint": dict(source_fingerprint),
        "image_size": [int(value) for value in image_size],
        "image_profile": str(image_profile),
        "transform_version": str(transform_version),
        "dtype": str(dtype),
        "shape": [int(value) for value in shape],
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def image_cache_metadata_path(cache_path: str | Path) -> Path:
    path = Path(cache_path)
    return path.with_suffix(path.suffix + ".json")


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(int(item.stat().st_size) for item in path.rglob("*") if item.is_file())


__all__ = [
    "IMAGE_DERIVED_CACHE_VERSION",
    "ImageDerivedCache",
    "ImageDerivedCacheConfig",
    "image_cache_metadata",
    "image_cache_metadata_path",
    "image_fingerprint",
]
