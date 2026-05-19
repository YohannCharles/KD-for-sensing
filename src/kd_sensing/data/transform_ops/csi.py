from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.data.transform_ops.io import joined_resource


CSI_FIELD_CANDIDATES = ("csi", "channel", "channels", "h", "H", "array")


def read_csi_tensor(data_root: str | Path, rel_path: str) -> np.ndarray:
    path = joined_resource(data_root, rel_path)
    if not path.exists():
        raise FileNotFoundError(f"CSI file not found: {path}")
    try:
        payload = _load_payload(path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSI file {path}: {exc}") from exc
    array, source_field = _select_csi_array(payload)
    return coerce_csi_real_imag(array, path=path, source_field=source_field)


def load_csi_sequence(data_root: str | Path, csi_paths: list[str], *, seq_len: int) -> np.ndarray:
    selected = csi_paths[-int(seq_len) :]
    tensors = [read_csi_tensor(data_root, path) for path in selected if _valid_csi_path(path)]
    if not tensors:
        raise ValueError("No valid CSI paths were provided for the requested sequence.")
    expanded = []
    for tensor in tensors:
        if tensor.ndim == 3:
            expanded.append(tensor)
        elif tensor.ndim == 4 and tensor.shape[0] == 1:
            expanded.append(tensor[0])
        else:
            raise ValueError(
                "CSI sequence entries must be single-frame tensors with shape [Nsc, Nant, 2]; "
                f"got shape {tuple(tensor.shape)} from one entry."
            )
    return np.stack(expanded, axis=0).astype(np.float32)


def coerce_csi_real_imag(array: Any, *, path: str | Path = "<array>", source_field: str = "array") -> np.ndarray:
    source = Path(path) if path != "<array>" else path
    values = np.asarray(array)
    if np.iscomplexobj(values):
        values = np.stack([values.real, values.imag], axis=-1)
    elif values.dtype.names and {"real", "imag"}.issubset(values.dtype.names):
        values = np.stack([values["real"], values["imag"]], axis=-1)
    else:
        values = np.asarray(values)
        if values.ndim >= 1 and values.shape[-1] == 2:
            pass
        elif values.ndim == 1:
            values = np.stack([values.astype(np.float32), np.zeros_like(values, dtype=np.float32)], axis=-1)
        else:
            raise ValueError(
                f"CSI file {source} field '{source_field}' has unsupported shape {tuple(values.shape)}; "
                "expected complex [Nsc,Nant] or real/imag [...,2]."
            )
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2 and values.shape[-1] == 2:
        values = values.reshape(1, values.shape[0], 2)
    if values.ndim not in {3, 4} or int(values.shape[-1]) != 2:
        raise ValueError(
            f"CSI file {source} field '{source_field}' has unsupported shape {tuple(values.shape)}; "
            "expected [Nsc,Nant,2] or [T,Nsc,Nant,2]."
        )
    if values.ndim == 3 and values.shape[0] == 1 and values.shape[1] != 1:
        # 1-D channel vectors are treated as one subcarrier with Nant antennas.
        pass
    if not np.isfinite(values).all():
        raise ValueError(f"CSI file {source} field '{source_field}' contains NaN or Inf values.")
    return values.astype(np.float32)


@dataclass
class CSIRMSNormalizer:
    rms: float
    sample_count: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.rms)) or float(self.rms) <= 0.0:
            raise ValueError(f"CSI RMS must be positive and finite, got {self.rms}.")
        self.rms = float(self.rms)
        self.sample_count = int(self.sample_count)

    def transform(self, csi: np.ndarray) -> np.ndarray:
        return (np.asarray(csi, dtype=np.float32) / float(self.rms)).astype(np.float32)

    def to_dict(self) -> dict[str, float | int]:
        return {"rms": float(self.rms), "sample_count": int(self.sample_count)}

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, rms=np.asarray(self.rms, dtype=np.float32), sample_count=np.asarray(self.sample_count))

    @classmethod
    def load(cls, path: str | Path) -> "CSIRMSNormalizer":
        with np.load(Path(path)) as payload:
            return cls(rms=float(payload["rms"]), sample_count=int(payload.get("sample_count", 0)))


class CSIRMSStreamingStats:
    def __init__(self) -> None:
        self.power_sum = 0.0
        self.count = 0
        self.sample_count = 0

    def update(self, csi: np.ndarray) -> None:
        values = np.asarray(csi, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != 2:
            raise ValueError(f"CSI RMS stats expected real/imag last dimension, got {tuple(values.shape)}.")
        if not np.isfinite(values).all():
            raise ValueError("CSI RMS stats received NaN or Inf values.")
        power = values[..., 0] ** 2 + values[..., 1] ** 2
        self.power_sum += float(power.sum())
        self.count += int(power.size)
        self.sample_count += 1

    def finalize(self) -> CSIRMSNormalizer:
        if self.count <= 0:
            raise ValueError("Cannot fit CSI RMS normalizer from zero CSI values.")
        rms = float(np.sqrt(self.power_sum / self.count))
        return CSIRMSNormalizer(rms=max(rms, 1e-8), sample_count=self.sample_count)


def fit_csi_rms_from_sequences(sequences: list[np.ndarray]) -> CSIRMSNormalizer:
    stats = CSIRMSStreamingStats()
    for sequence in sequences:
        stats.update(sequence)
    return stats.finalize()


def _load_payload(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as payload:
            return {key: payload[key] for key in payload.files}
    if suffix == ".npy":
        raw = np.load(path, allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.shape == () and isinstance(raw.item(), dict):
            return dict(raw.item())
        if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1 and isinstance(raw.reshape(-1)[0], dict):
            return dict(raw.reshape(-1)[0])
        return {"array": raw}
    raise ValueError(f"Unsupported CSI file extension for {path}; expected .npy or .npz.")


def _select_csi_array(payload: dict[str, Any]) -> tuple[Any, str]:
    lookup = {str(key).lower(): key for key in payload}
    for candidate in CSI_FIELD_CANDIDATES:
        key = lookup.get(candidate.lower())
        if key is not None:
            return payload[key], str(key)
    if len(payload) == 1:
        key = next(iter(payload))
        return payload[key], str(key)
    derived = _derive_channel_from_paths(payload)
    if derived is not None:
        return derived, "path_gains+departure_angles"
    raise ValueError(f"CSI payload has no supported field; available fields: {sorted(payload)}.")


def _derive_channel_from_paths(payload: dict[str, Any], *, tx_antennas: int = 64) -> np.ndarray | None:
    gain_key = _first_present(payload, ("a", "gain", "gains", "path_gain", "path_gains", "alpha", "alphas"))
    angle_key = _first_present(
        payload,
        (
            "aod",
            "ao_d",
            "tx_angle",
            "departure_angle",
            "azimuth_of_departure",
            "phi_t",
            "glob_phi_t",
            "theta_t",
            "glob_theta_t",
        ),
    )
    if gain_key is None or angle_key is None:
        return None
    gains = _coerce_complex_array(payload[gain_key]).reshape(-1)
    angles = np.asarray(payload[angle_key], dtype=np.float64).reshape(-1)
    if gains.size == 0 or angles.size == 0:
        return None
    count = min(gains.size, angles.size)
    radians = _angles_to_radians(angles[:count])
    antennas = np.arange(int(tx_antennas), dtype=np.float64)[:, None]
    steering = np.exp(1j * np.pi * antennas * np.sin(radians)[None, :])
    channel = steering @ gains[:count].astype(np.complex128)[:, None]
    return channel.reshape(-1).astype(np.complex64)


def _first_present(payload: dict[str, Any], names: tuple[str, ...]) -> str | None:
    lookup = {str(key).lower(): str(key) for key in payload}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def _coerce_complex_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.names and {"real", "imag"}.issubset(array.dtype.names):
        return array["real"] + 1j * array["imag"]
    if np.iscomplexobj(array):
        return array.astype(np.complex64)
    if array.ndim > 0 and array.shape[-1] == 2:
        return (array[..., 0] + 1j * array[..., 1]).astype(np.complex64)
    return array.astype(np.complex64)


def _angles_to_radians(angles: np.ndarray) -> np.ndarray:
    values = np.asarray(angles, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size and np.nanmax(np.abs(finite)) > 2.0 * np.pi:
        return np.deg2rad(values)
    return values


def _valid_csi_path(path: object) -> bool:
    text = str(path).strip()
    return bool(text) and text != "-99"


__all__ = [
    "CSIRMSNormalizer",
    "CSIRMSStreamingStats",
    "coerce_csi_real_imag",
    "fit_csi_rms_from_sequences",
    "load_csi_sequence",
    "read_csi_tensor",
]
