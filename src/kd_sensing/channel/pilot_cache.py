from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable

import numpy as np

from kd_sensing.channel.sparse_pilot_simulator import SIMULATOR_VERSION


@dataclass(frozen=True)
class PilotCacheSpec:
    codebook_hash: str
    frequency_positions_hz: tuple[float, ...]
    subcarrier_spacing_hz: float
    frequency_index_mode: str
    nt: int
    nr: int
    simulator_version: str = SIMULATOR_VERSION


class PilotCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def key(self, channel_path: str | Path, spec: PilotCacheSpec) -> tuple[str, dict[str, object]]:
        source = Path(channel_path).resolve()
        stat = source.stat()
        channel_hash = _sha256_file(source)
        metadata: dict[str, object] = {
            "channel_path": str(source),
            "channel_size": int(stat.st_size),
            "channel_sha256": channel_hash,
            "spec": json.loads(json.dumps(asdict(spec), sort_keys=True)),
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest(), metadata

    def load(self, channel_path: str | Path, spec: PilotCacheSpec) -> np.ndarray | None:
        key, expected = self.key(channel_path, spec)
        path = self.root / f"{key}.npz"
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
            values = np.asarray(payload["candidate_g"], dtype=np.complex64)
        if metadata != expected:
            return None
        return values

    def store(self, channel_path: str | Path, spec: PilotCacheSpec, candidate_g: np.ndarray) -> Path:
        key, metadata = self.key(channel_path, spec)
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{key}.npz"
        values = np.asarray(candidate_g, dtype=np.complex64)
        with tempfile.NamedTemporaryFile(dir=self.root, prefix=f".{key}.", suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                candidate_g=values,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temporary, target)
        return target

    def get_or_compute(
        self,
        channel_path: str | Path,
        spec: PilotCacheSpec,
        compute: Callable[[], np.ndarray],
    ) -> np.ndarray:
        cached = self.load(channel_path, spec)
        if cached is not None:
            return cached
        values = np.asarray(compute(), dtype=np.complex64)
        self.store(channel_path, spec, values)
        return values


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["PilotCache", "PilotCacheSpec"]
