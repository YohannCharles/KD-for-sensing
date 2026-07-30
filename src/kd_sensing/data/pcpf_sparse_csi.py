from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from kd_sensing.channel.pilot_cache import PilotCache, PilotCacheSpec
from kd_sensing.channel.probe_codebook import load_probe_codebook
from kd_sensing.channel.sparse_pilot_simulator import load_path_channel, simulate_candidate_pilots


PCPF_SPARSE_CSI_MODALITY = "csi"
PCPF_SPARSE_CSI_PATTERN_INDICES = (0, 1)
PCPF_SPARSE_CSI_FREQUENCY_INDICES = (0, 15)
PCPF_SPARSE_CSI_FREQUENCY_POSITIONS_HZ = (-61_440_000.0, 61_320_000.0)
PCPF_SPARSE_CSI_SELECTION = {
    "schema_version": 1,
    "selection_rule": "tspc_v2_nested_prefix_v1",
    "mother_shape": [5, 32, 16],
    "history_indices": [0, 1, 2, 3, 4],
    "pattern_indices": list(PCPF_SPARSE_CSI_PATTERN_INDICES),
    "frequency_indices": list(PCPF_SPARSE_CSI_FREQUENCY_INDICES),
}
PCPF_SPARSE_CSI_SELECTION_SHA256 = hashlib.sha256(
    json.dumps(PCPF_SPARSE_CSI_SELECTION, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

_CONFIG_FIELDS = {
    "enabled",
    "codebook_path",
    "codebook_sha256",
    "codebook_hash",
    "cache_root",
    "selection_sha256",
}


class PCPFSparseCSISidecar:
    """Deterministically load the fixed historical TSPC 2x2 observation."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        raw = dict(config)
        unknown = sorted(set(raw) - _CONFIG_FIELDS)
        if unknown:
            raise ValueError(f"PCPF sparse CSI config contains unsupported fields: {unknown}.")
        if raw.get("enabled") is not True:
            raise ValueError("PCPF sparse CSI sidecar requires enabled=true.")
        if str(raw.get("selection_sha256", "")).lower() != PCPF_SPARSE_CSI_SELECTION_SHA256:
            raise ValueError("PCPF sparse CSI selection SHA256 does not match the fixed TSPC-V2 2x2 selection.")

        self.codebook_path = _required_file(raw.get("codebook_path"), "codebook_path")
        expected_file_hash = _required_sha256(raw.get("codebook_sha256"), "codebook_sha256")
        actual_file_hash = _sha256_file(self.codebook_path)
        if actual_file_hash != expected_file_hash:
            raise ValueError(
                f"PCPF sparse CSI codebook file SHA256 mismatch: expected={expected_file_hash}, actual={actual_file_hash}."
            )
        self.codebook = load_probe_codebook(self.codebook_path)
        expected_codebook_hash = _required_sha256(raw.get("codebook_hash"), "codebook_hash")
        if self.codebook.hash != expected_codebook_hash:
            raise ValueError(
                f"PCPF sparse CSI logical codebook hash mismatch: expected={expected_codebook_hash}, "
                f"actual={self.codebook.hash}."
            )
        if self.codebook.tx.shape[0] != 32 or self.codebook.rx.shape[0] != 32:
            raise ValueError("PCPF sparse CSI requires the audited 32-pattern mother codebook.")

        cache_root = raw.get("cache_root")
        if not isinstance(cache_root, (str, Path)) or not str(cache_root).strip():
            raise ValueError("PCPF sparse CSI cache_root must be a non-empty path.")
        self.cache = PilotCache(Path(cache_root).resolve())
        self.cache_spec = PilotCacheSpec(
            codebook_hash=self.codebook.hash,
            frequency_positions_hz=PCPF_SPARSE_CSI_FREQUENCY_POSITIONS_HZ,
            subcarrier_spacing_hz=120_000.0,
            frequency_index_mode="centered",
            nt=64,
            nr=16,
        )
        self.codebook_file_sha256 = actual_file_hash
        self._memory_cache: dict[str, np.ndarray] = {}

    @property
    def identity(self) -> dict[str, Any]:
        cache_spec = {
            "codebook_hash": self.cache_spec.codebook_hash,
            "frequency_positions_hz": list(self.cache_spec.frequency_positions_hz),
            "subcarrier_spacing_hz": self.cache_spec.subcarrier_spacing_hz,
            "frequency_index_mode": self.cache_spec.frequency_index_mode,
            "nt": self.cache_spec.nt,
            "nr": self.cache_spec.nr,
            "simulator_version": self.cache_spec.simulator_version,
        }
        cache_spec_sha256 = hashlib.sha256(
            json.dumps(cache_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": 1,
            "source": "historical_channel_path_domain_simulator",
            "selection": dict(PCPF_SPARSE_CSI_SELECTION),
            "selection_sha256": PCPF_SPARSE_CSI_SELECTION_SHA256,
            "codebook_path": str(self.codebook_path),
            "codebook_hash": self.codebook.hash,
            "codebook_file_sha256": self.codebook_file_sha256,
            "spatial_selection": {
                "type": "paired_tx_rx_probe_patterns",
                "tx_pattern_indices": list(PCPF_SPARSE_CSI_PATTERN_INDICES),
                "rx_pattern_indices": list(PCPF_SPARSE_CSI_PATTERN_INDICES),
                "direct_antenna_indices": None,
                "tx_elements_per_pattern": 64,
                "rx_elements_per_pattern": 16,
            },
            "frequency_positions_hz": list(PCPF_SPARSE_CSI_FREQUENCY_POSITIONS_HZ),
            "cache_identity": {
                "root": str(self.cache.root.resolve()),
                "spec": cache_spec,
                "spec_sha256": cache_spec_sha256,
            },
            "complex_re_per_frame": 4,
            "mother_re_per_frame": 32 * 16,
            "sampling_ratio": 4.0 / float(32 * 16),
            "snr_available": False,
            "awgn_enabled": False,
            "pilot_dropout_enabled": False,
            "synthetic_corruption_enabled": False,
        }

    def load_history(
        self,
        channel_refs: Sequence[str | Path],
        *,
        history_frame_ids: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        refs = tuple(Path(value).resolve() for value in channel_refs)
        frame_ids = tuple(int(value) for value in history_frame_ids)
        if len(refs) != 5 or len(frame_ids) != 5:
            raise ValueError("PCPF sparse CSI requires exactly five historical channel refs and frame ids.")
        if any(right != left + 1 for left, right in zip(frame_ids, frame_ids[1:], strict=False)):
            raise ValueError("PCPF sparse CSI history frame ids must be consecutive and increasing.")
        frames = [self._load_frame(path) for path in refs]
        history = np.stack(frames).astype(np.complex64, copy=False)
        if history.shape != (5, 2, 2) or not np.isfinite(history).all():
            raise ValueError(f"PCPF sparse CSI history must be finite complex [5,2,2], got {history.shape}.")
        return {
            "csi": torch.from_numpy(history),
            "csi_pattern_ids": torch.tensor(PCPF_SPARSE_CSI_PATTERN_INDICES, dtype=torch.long).expand(5, -1).clone(),
            "csi_frequency_positions": torch.tensor(PCPF_SPARSE_CSI_FREQUENCY_POSITIONS_HZ, dtype=torch.float32),
            "csi_frequency_ids": torch.tensor(PCPF_SPARSE_CSI_FREQUENCY_INDICES, dtype=torch.long),
            "csi_pilot_mask": torch.ones(5, 2, 2, dtype=torch.bool),
            "csi_valid_mask": torch.ones(5, dtype=torch.bool),
            "csi_snr_available": torch.tensor(False),
        }

    def _load_frame(self, path: Path) -> np.ndarray:
        key = str(path)
        cached = self._memory_cache.get(key)
        if cached is not None:
            return cached.copy()
        if not path.is_file():
            raise FileNotFoundError(f"PCPF sparse CSI historical channel is missing: {path}")

        def compute() -> np.ndarray:
            matrices, delays = load_path_channel(path)
            return simulate_candidate_pilots(
                matrices[None, None, :, None, :, :, None],
                delays[None, None, None, :],
                self.codebook,
                np.asarray(PCPF_SPARSE_CSI_FREQUENCY_POSITIONS_HZ, dtype=np.float64),
            )

        mother = self.cache.get_or_compute(path, self.cache_spec, compute)
        if mother.shape != (32, 2):
            raise ValueError(f"PCPF sparse CSI cached mother frame must have shape [32,2], got {mother.shape}: {path}")
        selected = np.asarray(mother[list(PCPF_SPARSE_CSI_PATTERN_INDICES)], dtype=np.complex64)
        self._memory_cache[key] = selected
        return selected.copy()


def _required_file(value: Any, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"PCPF sparse CSI {field} must be a non-empty path.")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PCPF sparse CSI {field} is missing: {path}")
    return path


def _required_sha256(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"PCPF sparse CSI {field} must be a 64-character hexadecimal digest.")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "PCPFSparseCSISidecar",
    "PCPF_SPARSE_CSI_FREQUENCY_INDICES",
    "PCPF_SPARSE_CSI_FREQUENCY_POSITIONS_HZ",
    "PCPF_SPARSE_CSI_MODALITY",
    "PCPF_SPARSE_CSI_PATTERN_INDICES",
    "PCPF_SPARSE_CSI_SELECTION",
    "PCPF_SPARSE_CSI_SELECTION_SHA256",
]
