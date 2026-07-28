from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.channel.probe_codebook import ProbeCodebook


SIMULATOR_VERSION = "path_domain_sparse_pilot_v1"
FREQUENCY_INDEX_MODES = ("zero_based", "centered")


def load_path_channel(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        if "a" not in payload or "tau" not in payload:
            raise ValueError(f"Channel NPZ must contain a and tau: {source}")
        return parse_path_channel(payload["a"], payload["tau"])


def parse_path_channel(a: Any, tau: Any) -> tuple[np.ndarray, np.ndarray]:
    matrices = np.asarray(a)
    delays = np.asarray(tau, dtype=np.float64)
    if matrices.ndim != 7 or any(matrices.shape[index] != 1 for index in (0, 1, 3, 6)):
        raise ValueError(f"a must have shape [1,1,Nr,1,Nt,L,1], got {matrices.shape}.")
    matrices = np.asarray(matrices[0, 0, :, 0, :, :, 0], dtype=np.complex64)
    delays = delays.reshape(-1)
    if matrices.shape[2] != delays.size or delays.size == 0:
        raise ValueError(f"a path axis {matrices.shape[2]} does not match tau length {delays.size}.")
    if not np.all(np.isfinite(matrices)) or not np.all(np.isfinite(delays)):
        raise ValueError("Channel path matrices and delays must be finite.")
    return matrices, delays


def pilot_subcarrier_indices(
    num_subcarriers: int,
    num_pilot_subcarriers: int,
    *,
    pattern: str = "uniform",
) -> np.ndarray:
    count, selected = int(num_subcarriers), int(num_pilot_subcarriers)
    if count <= 0 or selected <= 0 or selected > count:
        raise ValueError("Require 0 < num_pilot_subcarriers <= num_subcarriers.")
    if str(pattern) != "uniform":
        raise ValueError("Only pilot_frequency_pattern='uniform' is supported.")
    indices = np.rint(np.linspace(0, count - 1, selected)).astype(np.int64)
    if np.unique(indices).size != selected:
        raise ValueError("Uniform pilot subcarrier selection produced duplicate indices.")
    return indices


def frequency_offsets_hz(
    indices: np.ndarray,
    *,
    num_subcarriers: int,
    subcarrier_spacing_hz: float,
    mode: str,
) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode == "auto":
        raise ValueError("frequency_index_mode=auto requires dataset metadata or a bound consistency audit.")
    if mode not in FREQUENCY_INDEX_MODES:
        raise ValueError(f"Unsupported frequency index mode {mode!r}.")
    positions = np.asarray(indices, dtype=np.int64).reshape(-1)
    count = int(num_subcarriers)
    spacing = float(subcarrier_spacing_hz)
    if count <= 0 or spacing <= 0 or np.any(positions < 0) or np.any(positions >= count):
        raise ValueError("Subcarrier indices, count and spacing are inconsistent.")
    origin = 0.0 if mode == "zero_based" else float(count // 2)
    return (positions.astype(np.float64) - origin) * spacing


def simulate_candidate_pilots(
    a: Any,
    tau: Any,
    codebook: ProbeCodebook,
    frequency_positions_hz: np.ndarray,
) -> np.ndarray:
    matrices, delays = parse_path_channel(a, tau)
    if matrices.shape[:2] != (codebook.rx.shape[1], codebook.tx.shape[1]):
        raise ValueError(
            f"Channel (Nr,Nt)={matrices.shape[:2]} does not match codebook "
            f"({codebook.rx.shape[1]},{codebook.tx.shape[1]})."
        )
    spatial = np.einsum("rm,mnl,rn->rl", codebook.rx.conj(), matrices, codebook.tx, optimize=True)
    phase = np.exp(-2j * np.pi * np.asarray(frequency_positions_hz, dtype=np.float64)[:, None] * delays[None, :])
    return np.asarray(spatial @ phase.T, dtype=np.complex64)


def explicit_frequency_channel(a: Any, tau: Any, frequency_positions_hz: np.ndarray) -> np.ndarray:
    matrices, delays = parse_path_channel(a, tau)
    phase = np.exp(-2j * np.pi * np.asarray(frequency_positions_hz, dtype=np.float64)[:, None] * delays[None, :])
    return np.asarray(np.einsum("mnl,kl->kmn", matrices, phase, optimize=True), dtype=np.complex64)


def project_explicit_channel(channel: np.ndarray, codebook: ProbeCodebook) -> np.ndarray:
    values = np.asarray(channel)
    if values.ndim != 3:
        raise ValueError("Explicit frequency channel must have shape [K,Nr,Nt].")
    return np.asarray(
        np.einsum("rm,kmn,rn->rk", codebook.rx.conj(), values, codebook.tx, optimize=True),
        dtype=np.complex64,
    )


def add_awgn(
    observations: np.ndarray,
    snr_db: float,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    signal = np.asarray(observations, dtype=np.complex64)
    power = float(np.mean(np.abs(signal) ** 2))
    variance = power / (10.0 ** (float(snr_db) / 10.0))
    if not np.isfinite(variance):
        raise ValueError("AWGN variance is not finite.")
    noise = np.sqrt(variance / 2.0) * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )
    return np.asarray(signal + noise, dtype=np.complex64), variance


def apply_pilot_dropout(
    observations: np.ndarray,
    probability: float,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(observations, dtype=np.complex64)
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("pilot dropout probability must be in [0,1].")
    valid = rng.random(values.shape) >= probability
    return np.where(valid, values, 0.0).astype(np.complex64), valid


__all__ = [
    "FREQUENCY_INDEX_MODES",
    "SIMULATOR_VERSION",
    "add_awgn",
    "apply_pilot_dropout",
    "explicit_frequency_channel",
    "frequency_offsets_hz",
    "load_path_channel",
    "parse_path_channel",
    "pilot_subcarrier_indices",
    "project_explicit_channel",
    "simulate_candidate_pilots",
]
