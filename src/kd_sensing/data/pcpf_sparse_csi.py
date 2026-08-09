from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from kd_sensing.channel.pilot_cache import PilotCache, PilotCacheSpec
from kd_sensing.channel.probe_codebook import load_probe_codebook
from kd_sensing.channel.sparse_pilot_simulator import load_path_channel, simulate_candidate_pilots
from kd_sensing.data.mmw.trajectory_protocol import split_cache_identity


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
PCPF_SPARSE_CSI_4X2_PATTERN_INDICES = (0, 1, 2, 3)
PCPF_SPARSE_CSI_4X2_SELECTION = {
    **PCPF_SPARSE_CSI_SELECTION,
    "pattern_indices": list(PCPF_SPARSE_CSI_4X2_PATTERN_INDICES),
}
PCPF_SPARSE_CSI_4X2_SELECTION_SHA256 = hashlib.sha256(
    json.dumps(PCPF_SPARSE_CSI_4X2_SELECTION, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
_SUPPORTED_SELECTIONS = {
    PCPF_SPARSE_CSI_SELECTION_SHA256: PCPF_SPARSE_CSI_SELECTION,
    PCPF_SPARSE_CSI_4X2_SELECTION_SHA256: PCPF_SPARSE_CSI_4X2_SELECTION,
}
PCPF_SPARSE_CSI_PACKED_CACHE_SCHEMA_VERSION = 3

_CONFIG_FIELDS = {
    "enabled",
    "codebook_path",
    "codebook_sha256",
    "codebook_hash",
    "cache_root",
    "cache_manifest_path",
    "selection_sha256",
    "packed_cache_path",
    "packed_cache_sha256",
    "packed_cache_protocol_id",
    "packed_cache_protocol_fingerprint",
    "packed_cache_manifest_version",
    "packed_cache_split_seed",
    "packed_cache_split_manifest",
    "packed_cache_split_identity",
}


class PCPFSparseCSISidecar:
    """Deterministically load a pre-registered historical TSPC observation."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        raw = dict(config)
        unknown = sorted(set(raw) - _CONFIG_FIELDS)
        if unknown:
            raise ValueError(f"PCPF sparse CSI config contains unsupported fields: {unknown}.")
        if raw.get("enabled") is not True:
            raise ValueError("PCPF sparse CSI sidecar requires enabled=true.")
        self.selection_sha256 = _required_sha256(raw.get("selection_sha256"), "selection_sha256")
        if self.selection_sha256 not in _SUPPORTED_SELECTIONS:
            raise ValueError("PCPF sparse CSI selection SHA256 is not a pre-registered TSPC-V2 selection.")
        self.selection = _SUPPORTED_SELECTIONS[self.selection_sha256]
        self.pattern_indices = tuple(int(value) for value in self.selection["pattern_indices"])
        self.frequency_indices = tuple(int(value) for value in self.selection["frequency_indices"])
        if self.frequency_indices != PCPF_SPARSE_CSI_FREQUENCY_INDICES:
            raise ValueError("PCPF sparse CSI selection uses unsupported frequency indices.")
        self.selected_shape = (len(self.pattern_indices), len(self.frequency_indices))

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
        self.cache_spec_payload = {
            "codebook_hash": self.cache_spec.codebook_hash,
            "frequency_positions_hz": list(self.cache_spec.frequency_positions_hz),
            "subcarrier_spacing_hz": self.cache_spec.subcarrier_spacing_hz,
            "frequency_index_mode": self.cache_spec.frequency_index_mode,
            "nt": self.cache_spec.nt,
            "nr": self.cache_spec.nr,
            "simulator_version": self.cache_spec.simulator_version,
        }
        self.cache_spec_sha256 = hashlib.sha256(
            json.dumps(self.cache_spec_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._memory_cache: dict[str, np.ndarray] = {}
        self._cache_keys: dict[str, str] = {}
        self._packed_cache_path: Path | None = None
        self._packed_cache_sha256: str | None = None
        self._packed_cache_metadata: dict[str, Any] | None = None
        self._packed_frame_index: dict[str, int] | None = None
        self._packed_frames: np.ndarray | None = None

        packed_fields = (
            raw.get("packed_cache_path"),
            raw.get("packed_cache_sha256"),
            raw.get("packed_cache_protocol_id"),
            raw.get("packed_cache_protocol_fingerprint"),
            raw.get("packed_cache_manifest_version"),
            raw.get("packed_cache_split_seed"),
            raw.get("packed_cache_split_manifest"),
            raw.get("packed_cache_split_identity"),
        )
        if any(value is not None for value in packed_fields):
            if not all(value is not None for value in packed_fields):
                raise ValueError(
                    "PCPF sparse CSI packed cache requires path, SHA256, protocol ID/fingerprint, "
                    "manifest version, split seed, split manifest, and the full split identity together."
                )
            packed_path = _required_file(packed_fields[0], "packed_cache_path")
            packed_sha256 = _required_sha256(packed_fields[1], "packed_cache_sha256")
            protocol_id = str(packed_fields[2]).strip()
            if not protocol_id:
                raise ValueError("PCPF sparse CSI packed_cache_protocol_id must be non-empty.")
            protocol_fingerprint = _required_sha256(
                packed_fields[3], "packed_cache_protocol_fingerprint"
            )
            manifest_version = _required_nonnegative_int(
                packed_fields[4], "packed_cache_manifest_version"
            )
            split_seed = _required_nonnegative_int(packed_fields[5], "packed_cache_split_seed")
            split_manifest = str(Path(str(packed_fields[6])).resolve())
            if not isinstance(packed_fields[7], Mapping):
                raise ValueError("PCPF sparse CSI packed_cache_split_identity must be a mapping.")
            cache_split_identity = split_cache_identity(packed_fields[7])
            metadata, frame_index, frames = _load_packed_cache(
                str(packed_path), packed_sha256, *self.selected_shape
            )
            self._validate_packed_cache(
                metadata,
                protocol_id,
                protocol_fingerprint,
                manifest_version,
                split_seed,
                split_manifest,
                cache_split_identity,
            )
            self._packed_cache_path = packed_path
            self._packed_cache_sha256 = packed_sha256
            self._packed_cache_metadata = metadata
            self._packed_frame_index = frame_index
            self._packed_frames = frames

    @property
    def identity(self) -> dict[str, Any]:
        identity = {
            "schema_version": 1,
            "source": "historical_channel_path_domain_simulator",
            "selection": dict(self.selection),
            "selection_sha256": self.selection_sha256,
            "codebook_path": str(self.codebook_path),
            "codebook_hash": self.codebook.hash,
            "codebook_file_sha256": self.codebook_file_sha256,
            "spatial_selection": {
                "type": "paired_tx_rx_probe_patterns",
                "tx_pattern_indices": list(self.pattern_indices),
                "rx_pattern_indices": list(self.pattern_indices),
                "direct_antenna_indices": None,
                "tx_elements_per_pattern": 64,
                "rx_elements_per_pattern": 16,
            },
            "frequency_positions_hz": list(PCPF_SPARSE_CSI_FREQUENCY_POSITIONS_HZ),
            "cache_identity": {
                "root": str(self.cache.root.resolve()),
                "spec": self.cache_spec_payload,
                "spec_sha256": self.cache_spec_sha256,
            },
            "complex_re_per_frame": self.selected_shape[0] * self.selected_shape[1],
            "mother_re_per_frame": 32 * 16,
            "sampling_ratio": float(self.selected_shape[0] * self.selected_shape[1]) / float(32 * 16),
            "snr_available": False,
            "awgn_enabled": False,
            "pilot_dropout_enabled": False,
            "synthetic_corruption_enabled": False,
        }
        if self._packed_cache_metadata is not None:
            identity["packed_cache"] = {
                "path": str(self._packed_cache_path),
                "sha256": self._packed_cache_sha256,
                "protocol_id": self._packed_cache_metadata["protocol_id"],
                "protocol_fingerprint": self._packed_cache_metadata["protocol_fingerprint"],
                "manifest_version": self._packed_cache_metadata["manifest_version"],
                "split_seed": self._packed_cache_metadata["split_seed"],
                "split_manifest": self._packed_cache_metadata["split_manifest"],
                "split_identity": {
                    key: self._packed_cache_metadata[key]
                    for key in split_cache_identity(self._packed_cache_metadata)
                },
                "entry_count": self._packed_cache_metadata["entry_count"],
                "strict": True,
            }
        return identity

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
        expected_shape = (5, *self.selected_shape)
        if history.shape != expected_shape or not np.isfinite(history).all():
            raise ValueError(f"PCPF sparse CSI history must be finite complex {expected_shape}, got {history.shape}.")
        return {
            "csi": torch.from_numpy(history),
            "csi_pattern_ids": torch.tensor(self.pattern_indices, dtype=torch.long).expand(5, -1).clone(),
            "csi_frequency_positions": torch.tensor(PCPF_SPARSE_CSI_FREQUENCY_POSITIONS_HZ, dtype=torch.float32),
            "csi_frequency_ids": torch.tensor(self.frequency_indices, dtype=torch.long),
            "csi_pilot_mask": torch.ones(expected_shape, dtype=torch.bool),
            "csi_valid_mask": torch.ones(5, dtype=torch.bool),
            "csi_snr_available": torch.tensor(False),
        }

    def _load_frame(self, path: Path) -> np.ndarray:
        key = str(path)
        if self._packed_frame_index is not None:
            index = self._packed_frame_index.get(key)
            if index is None:
                raise FileNotFoundError(f"PCPF sparse CSI strict packed cache miss: {path}")
            assert self._packed_frames is not None
            return self._packed_frames[index].copy()
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

        mother, cache_key = self.cache.get_or_compute_with_key(path, self.cache_spec, compute)
        if mother.shape != (32, 2):
            raise ValueError(f"PCPF sparse CSI cached mother frame must have shape [32,2], got {mother.shape}: {path}")
        selected = np.asarray(mother[list(self.pattern_indices)], dtype=np.complex64)
        self._memory_cache[key] = selected
        self._cache_keys[key] = cache_key
        return selected.copy()

    def export_packed_cache(
        self,
        path: str | Path,
        *,
        protocol_id: str,
        protocol_fingerprint: str,
        manifest_version: int,
        split_seed: int,
        split_manifest: str | Path,
        split_identity: Mapping[str, Any],
        roles: Mapping[str, Mapping[str, int]],
    ) -> dict[str, Any]:
        fingerprint = _required_sha256(protocol_fingerprint, "packed_cache_protocol_fingerprint")
        manifest_version = _required_nonnegative_int(manifest_version, "packed_cache_manifest_version")
        split_seed = _required_nonnegative_int(split_seed, "packed_cache_split_seed")
        split_manifest = str(Path(split_manifest).resolve())
        cache_split_identity = split_cache_identity(split_identity)
        if (
            cache_split_identity["split_protocol"] != str(protocol_id)
            or int(cache_split_identity["split_seed"]) != split_seed
        ):
            raise ValueError("PCPF sparse CSI packed cache split identity conflicts with protocol arguments.")
        paths = sorted(self._memory_cache)
        if not paths or set(paths) != set(self._cache_keys):
            raise ValueError("PCPF sparse CSI packed cache export requires complete prefilled frames and cache keys.")
        normalized_roles = {
            role: {
                "sample_count": int(values["sample_count"]),
                "unique_channel_count": int(values["unique_channel_count"]),
            }
            for role, values in roles.items()
        }
        if sum(values["unique_channel_count"] for values in normalized_roles.values()) != len(paths):
            raise ValueError("PCPF sparse CSI packed cache role counts do not match the unique channel union.")
        metadata = {
            "schema_version": PCPF_SPARSE_CSI_PACKED_CACHE_SCHEMA_VERSION,
            "status": "passed",
            **cache_split_identity,
            "protocol_id": str(protocol_id),
            "protocol_fingerprint": fingerprint,
            "manifest_version": manifest_version,
            "split_seed": split_seed,
            "split_manifest": split_manifest,
            "selection_sha256": self.selection_sha256,
            "codebook_hash": self.codebook.hash,
            "codebook_file_sha256": self.codebook_file_sha256,
            "cache_root": str(self.cache.root.resolve()),
            "cache_spec_sha256": self.cache_spec_sha256,
            "entry_count": len(paths),
            "roles": normalized_roles,
            "test_evaluated": False,
            "outer_test_accessed": False,
        }
        values = np.stack([self._memory_cache[channel_path] for channel_path in paths]).astype(
            np.complex64, copy=False
        )
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".npz",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                channel_paths=np.asarray(paths),
                cache_keys=np.asarray([self._cache_keys[channel_path] for channel_path in paths]),
                selected_g=values,
            )
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        digest = _sha256_file(target)
        return {
            "path": str(target),
            "sha256": digest,
            "size_bytes": target.stat().st_size,
            "entry_count": len(paths),
            "metadata": metadata,
        }

    def _validate_packed_cache(
        self,
        metadata: Mapping[str, Any],
        protocol_id: str,
        protocol_fingerprint: str,
        manifest_version: int,
        split_seed: int,
        split_manifest: str,
        cache_split_identity: Mapping[str, Any],
    ) -> None:
        expected = {
            "schema_version": PCPF_SPARSE_CSI_PACKED_CACHE_SCHEMA_VERSION,
            "status": "passed",
            **split_cache_identity(cache_split_identity),
            "protocol_id": protocol_id,
            "protocol_fingerprint": protocol_fingerprint,
            "manifest_version": manifest_version,
            "split_seed": split_seed,
            "split_manifest": split_manifest,
            "selection_sha256": self.selection_sha256,
            "codebook_hash": self.codebook.hash,
            "codebook_file_sha256": self.codebook_file_sha256,
            "cache_root": str(self.cache.root.resolve()),
            "cache_spec_sha256": self.cache_spec_sha256,
            "test_evaluated": False,
            "outer_test_accessed": False,
        }
        mismatches = {
            key: {"expected": value, "actual": metadata.get(key)}
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"PCPF sparse CSI packed cache identity mismatch: {mismatches}")


@lru_cache(maxsize=8)
def _load_packed_cache(
    path_text: str,
    expected_sha256: str,
    num_patterns: int,
    num_frequencies: int,
) -> tuple[dict[str, Any], dict[str, int], np.ndarray]:
    path = Path(path_text)
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"PCPF sparse CSI packed cache SHA256 mismatch: expected={expected_sha256}, actual={actual_sha256}."
        )
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        channel_paths = [str(value) for value in np.asarray(payload["channel_paths"]).tolist()]
        cache_keys = [str(value) for value in np.asarray(payload["cache_keys"]).tolist()]
        frames = np.asarray(payload["selected_g"], dtype=np.complex64)
    if not isinstance(metadata, dict):
        raise ValueError(f"PCPF sparse CSI packed cache metadata must be a mapping: {path}")
    if channel_paths != sorted(channel_paths) or len(channel_paths) != len(set(channel_paths)):
        raise ValueError(f"PCPF sparse CSI packed cache paths must be sorted and unique: {path}")
    if any(not Path(value).is_absolute() for value in channel_paths):
        raise ValueError(f"PCPF sparse CSI packed cache paths must be absolute: {path}")
    if len(cache_keys) != len(channel_paths) or any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in cache_keys
    ):
        raise ValueError(f"PCPF sparse CSI packed cache keys are invalid: {path}")
    if (
        frames.shape != (len(channel_paths), num_patterns, num_frequencies)
        or not np.iscomplexobj(frames)
        or not np.isfinite(frames).all()
    ):
        raise ValueError(f"PCPF sparse CSI packed cache values are invalid: {path}")
    if int(metadata.get("entry_count", -1)) != len(channel_paths):
        raise ValueError(f"PCPF sparse CSI packed cache entry count changed: {path}")
    frames.setflags(write=False)
    return metadata, {channel_path: index for index, channel_path in enumerate(channel_paths)}, frames


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


def _required_nonnegative_int(value: Any, field: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"PCPF sparse CSI {field} must be a non-negative integer.") from error
    if normalized < 0:
        raise ValueError(f"PCPF sparse CSI {field} must be a non-negative integer.")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "PCPFSparseCSISidecar",
    "PCPF_SPARSE_CSI_4X2_PATTERN_INDICES",
    "PCPF_SPARSE_CSI_4X2_SELECTION",
    "PCPF_SPARSE_CSI_4X2_SELECTION_SHA256",
    "PCPF_SPARSE_CSI_PACKED_CACHE_SCHEMA_VERSION",
    "PCPF_SPARSE_CSI_FREQUENCY_INDICES",
    "PCPF_SPARSE_CSI_FREQUENCY_POSITIONS_HZ",
    "PCPF_SPARSE_CSI_MODALITY",
    "PCPF_SPARSE_CSI_PATTERN_INDICES",
    "PCPF_SPARSE_CSI_SELECTION",
    "PCPF_SPARSE_CSI_SELECTION_SHA256",
]
