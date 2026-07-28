from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CODEBOOK_VERSION = "phase_only_probe_codebook_v1"
METHODS = ("random_qpsk", "fixed_dft_beams", "multice_interleaved")


@dataclass(frozen=True)
class ProbeCodebook:
    tx: np.ndarray
    rx: np.ndarray
    metadata: dict[str, Any]

    @property
    def hash(self) -> str:
        return str(self.metadata["codebook_hash"])

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            tx=np.asarray(self.tx, dtype=np.complex64),
            rx=np.asarray(self.rx, dtype=np.complex64),
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )
        return target


def generate_probe_codebook(
    nt: int,
    nr: int,
    *,
    num_patterns: int = 32,
    seed: int = 2026,
    method: str = "random_qpsk",
    candidate_pool_size: int | None = None,
) -> ProbeCodebook:
    nt, nr, count = int(nt), int(nr), int(num_patterns)
    method = str(method).strip().lower()
    if nt <= 0 or nr <= 0 or count <= 0:
        raise ValueError("nt, nr and num_patterns must be positive.")
    if method not in METHODS:
        raise ValueError(f"Unsupported probe codebook method {method!r}; expected one of {METHODS}.")

    if method == "random_qpsk":
        pool_size = max(count, int(candidate_pool_size or count * 8))
        rng = np.random.default_rng(int(seed))
        tx_pool = _qpsk_patterns(rng, pool_size, nt)
        rx_pool = _qpsk_patterns(rng, pool_size, nr)
        selected = _greedy_low_coherence_pairs(tx_pool, rx_pool, count)
        tx, rx = tx_pool[selected], rx_pool[selected]
        generation = "greedy_low_coherence_qpsk_pool"
    elif method == "fixed_dft_beams":
        indices = np.arange(count, dtype=np.int64)
        tx = _dft_patterns(nt, indices % nt)
        rx = _dft_patterns(nr, (indices * 5) % nr)
        pool_size = count
        generation = "deterministic_dft_pair_scan"
    else:
        indices = np.arange(count, dtype=np.int64)
        tx = _interleaved_phase_patterns(nt, indices)
        rx = _interleaved_phase_patterns(nr, indices * 3 + 1)
        pool_size = count
        generation = "multice_style_interleaved_phase_groups"

    tx = np.asarray(tx, dtype=np.complex64)
    rx = np.asarray(rx, dtype=np.complex64)
    _validate_codebook_arrays(tx, rx, nt=nt, nr=nr, num_patterns=count)
    coherence = _pairwise_coherence(tx, rx)
    metadata: dict[str, Any] = {
        "version": CODEBOOK_VERSION,
        "method": method,
        "generation_method": generation,
        "nt": nt,
        "nr": nr,
        "num_patterns": count,
        "seed": int(seed),
        "candidate_pool_size": int(pool_size),
        "max_pairwise_coherence": float(coherence.max(initial=0.0)),
        "mean_pairwise_coherence": float(coherence.mean()) if coherence.size else 0.0,
    }
    metadata["codebook_hash"] = _codebook_hash(tx, rx, metadata)
    return ProbeCodebook(tx=tx, rx=rx, metadata=metadata)


def load_probe_codebook(path: str | Path) -> ProbeCodebook:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        tx = np.asarray(payload["tx"], dtype=np.complex64)
        rx = np.asarray(payload["rx"], dtype=np.complex64)
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    _validate_codebook_arrays(
        tx,
        rx,
        nt=int(metadata["nt"]),
        nr=int(metadata["nr"]),
        num_patterns=int(metadata["num_patterns"]),
    )
    expected = _codebook_hash(tx, rx, {key: value for key, value in metadata.items() if key != "codebook_hash"})
    if metadata.get("codebook_hash") != expected:
        raise ValueError(f"Probe codebook hash mismatch: {source}")
    return ProbeCodebook(tx=tx, rx=rx, metadata=metadata)


def _qpsk_patterns(rng: np.random.Generator, count: int, width: int) -> np.ndarray:
    phases = rng.integers(0, 4, size=(count, width)) * (np.pi / 2.0)
    return np.exp(1j * phases) / np.sqrt(float(width))


def _dft_patterns(width: int, indices: np.ndarray) -> np.ndarray:
    elements = np.arange(width, dtype=np.float64)[None, :]
    return np.exp(-2j * np.pi * indices[:, None] * elements / float(width)) / np.sqrt(float(width))


def _interleaved_phase_patterns(width: int, indices: np.ndarray) -> np.ndarray:
    elements = np.arange(width, dtype=np.int64)[None, :]
    phases = ((elements % 4) * (indices[:, None] % 4 + 1) + elements // 4 + indices[:, None]) % 4
    return np.exp(1j * phases * (np.pi / 2.0)) / np.sqrt(float(width))


def _greedy_low_coherence_pairs(tx: np.ndarray, rx: np.ndarray, count: int) -> np.ndarray:
    if count > tx.shape[0]:
        raise ValueError("num_patterns cannot exceed candidate_pool_size.")
    tx_corr = np.abs(tx @ tx.conj().T)
    rx_corr = np.abs(rx @ rx.conj().T)
    pair_corr = tx_corr * rx_corr
    selected = [0]
    available = np.ones(tx.shape[0], dtype=bool)
    available[0] = False
    while len(selected) < count:
        candidates = np.flatnonzero(available)
        worst = pair_corr[np.ix_(candidates, np.asarray(selected))].max(axis=1)
        chosen = int(candidates[int(np.argmin(worst))])
        selected.append(chosen)
        available[chosen] = False
    return np.asarray(selected, dtype=np.int64)


def _pairwise_coherence(tx: np.ndarray, rx: np.ndarray) -> np.ndarray:
    combined = np.abs(tx @ tx.conj().T) * np.abs(rx @ rx.conj().T)
    return combined[np.triu_indices(combined.shape[0], k=1)]


def _validate_codebook_arrays(tx: np.ndarray, rx: np.ndarray, *, nt: int, nr: int, num_patterns: int) -> None:
    if tx.shape != (num_patterns, nt) or rx.shape != (num_patterns, nr):
        raise ValueError(f"Probe codebook shapes must be {(num_patterns, nt)} and {(num_patterns, nr)}.")
    for name, values, width in (("tx", tx, nt), ("rx", rx, nr)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} probe codebook contains non-finite values.")
        if not np.allclose(np.linalg.norm(values, axis=1), 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{name} probe patterns must have unit norm.")
        if not np.allclose(np.abs(values), 1.0 / np.sqrt(float(width)), rtol=1e-6, atol=1e-6):
            raise ValueError(f"{name} probe patterns must be constant modulus.")


def _codebook_hash(tx: np.ndarray, rx: np.ndarray, metadata: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(tx).view(np.uint8))
    digest.update(np.ascontiguousarray(rx).view(np.uint8))
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


__all__ = ["CODEBOOK_VERSION", "METHODS", "ProbeCodebook", "generate_probe_codebook", "load_probe_codebook"]
