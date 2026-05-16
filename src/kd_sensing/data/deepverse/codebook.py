from __future__ import annotations

import numpy as np


def make_ula_dft_codebook(num_ant: int, num_beams: int = 64) -> np.ndarray:
    """Return a normalized ULA/DFT-style codebook with shape [num_ant, num_beams]."""

    if num_ant <= 0:
        raise ValueError(f"num_ant must be positive, got {num_ant}.")
    if num_beams <= 0:
        raise ValueError(f"num_beams must be positive, got {num_beams}.")

    antennas = np.arange(num_ant, dtype=np.float32)[:, None]
    beams = np.arange(num_beams, dtype=np.float32)[None, :]
    codebook = np.exp(-2j * np.pi * antennas * beams / float(num_beams)).astype(np.complex64)
    codebook /= np.linalg.norm(codebook, axis=0, keepdims=True).clip(min=1e-12)
    return codebook


def compute_beam_gain(channel: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Compute mean beam gain for a channel using a [N_ant, N_beam] codebook."""

    h = np.asarray(channel)
    cb = np.asarray(codebook)
    if cb.ndim != 2:
        raise ValueError(f"codebook must have shape [num_ant, num_beams], got {cb.shape}.")
    if h.ndim == 1:
        h = h[:, None]
    elif h.ndim == 2:
        if h.shape[0] == cb.shape[0]:
            pass
        elif h.shape[1] == cb.shape[0]:
            h = h.T
        else:
            raise ValueError(f"channel shape {h.shape} is incompatible with codebook shape {cb.shape}.")
    elif h.ndim >= 3:
        if h.shape[0] == cb.shape[0]:
            h = h[:, 0, ...].reshape(cb.shape[0], -1)
        elif h.shape[1] == cb.shape[0]:
            h = h[0, :, ...].reshape(cb.shape[0], -1)
        else:
            raise ValueError(f"channel shape {h.shape} is incompatible with codebook shape {cb.shape}.")
    else:
        raise ValueError(f"channel must have at least one dimension, got {h.shape}.")

    if h.shape[0] != cb.shape[0]:
        raise ValueError(f"channel antenna dimension {h.shape[0]} does not match codebook {cb.shape[0]}.")

    projected = np.conjugate(cb).T @ h.astype(np.complex64, copy=False)
    gains = np.mean(np.abs(projected) ** 2, axis=1).astype(np.float32)
    if not np.all(np.isfinite(gains)):
        raise ValueError("computed beam gains contain NaN or Inf.")
    return gains


def beam_entropy(gains: np.ndarray) -> float:
    values = np.asarray(gains, dtype=np.float64)
    total = float(np.sum(values))
    if total <= 0.0 or not np.isfinite(total):
        return 0.0
    probs = values / total
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log(probs)))
