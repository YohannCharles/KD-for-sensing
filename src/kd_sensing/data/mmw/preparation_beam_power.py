from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.data.beam_codebook import compute_beam_gain, make_ula_dft_codebook
from kd_sensing.data.mmw.preparation_config import ALGORITHM_VERSION



def load_channel_payload(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path)
    diagnostics = {"path": str(source), "fields": [], "shape": None}
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=True) as payload:
            data = {key: payload[key] for key in payload.files}
    elif source.suffix.lower() == ".npy":
        raw = np.load(source, allow_pickle=True)
        diagnostics["shape"] = tuple(raw.shape)
        if isinstance(raw, np.ndarray) and raw.shape == () and isinstance(raw.item(), dict):
            data = dict(raw.item())
        elif isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1 and isinstance(raw.reshape(-1)[0], dict):
            data = dict(raw.reshape(-1)[0])
        else:
            data = {"array": raw}
    else:
        raise ValueError(f"Unsupported channel file extension for {source}; expected .npy or .npz.")
    diagnostics["fields"] = sorted(data.keys())
    if not data:
        raise ValueError(f"Channel file {source} contains no fields.")
    return data, diagnostics


def derive_beam_power_from_file(
    path: str | Path,
    *,
    num_beams: int = 64,
    tx_antennas: int = 64,
    rx_antennas: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    payload, diagnostics = load_channel_payload(path)
    power, source_field = derive_beam_power(
        payload,
        num_beams=num_beams,
        tx_antennas=tx_antennas,
    )
    path_count = _channel_path_count(payload, source_field)
    return power, {
        "algorithm_version": ALGORITHM_VERSION,
        "codebook_type": "ula_dft",
        "num_beams": int(num_beams),
        "tx_antennas": int(tx_antennas),
        "rx_antennas": int(rx_antennas),
        "source_channel_field": source_field,
        "path_count": int(path_count),
        "diagnostics": diagnostics,
    }


def derive_beam_power(
    payload: dict[str, Any],
    *,
    num_beams: int = 64,
    tx_antennas: int = 64,
) -> tuple[np.ndarray, str]:
    channel_field = _first_present(payload, ("channel", "channels", "h", "H", "csi", "array", "a"))
    if channel_field is not None:
        channel = _coerce_complex_array(payload[channel_field])
        if channel.size == int(num_beams) and channel.ndim == 1:
            power = np.abs(channel.astype(np.complex64)) ** 2
        else:
            codebook = make_ula_dft_codebook(int(tx_antennas), int(num_beams))
            power = _compute_beam_gain_for_channel(channel, codebook)
        return _validate_power(power, expected_dim=num_beams, source=channel_field), channel_field

    gain_field = _first_present(payload, ("a", "gain", "gains", "path_gain", "path_gains", "alpha", "alphas"))
    angle_field = _first_present(
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
    if gain_field is None or angle_field is None:
        raise ValueError(
            "Channel payload must contain an equivalent channel field or both path gains and AoD angles; "
            f"available fields: {sorted(payload.keys())}."
        )
    gains = _coerce_complex_array(payload[gain_field]).reshape(-1)
    angles = np.asarray(payload[angle_field], dtype=np.float64).reshape(-1)
    if gains.size == 0 or angles.size == 0:
        raise ValueError("Channel path gains and AoD angles must be non-empty.")
    count = min(gains.size, angles.size)
    gains = gains[:count]
    angles = angles[:count]
    radians = _angles_to_radians(angles)
    antennas = np.arange(int(tx_antennas), dtype=np.float64)[:, None]
    steering = np.exp(1j * np.pi * antennas * np.sin(radians)[None, :])
    channel = steering @ gains.astype(np.complex128)[:, None]
    codebook = make_ula_dft_codebook(int(tx_antennas), int(num_beams))
    power = compute_beam_gain(channel, codebook)
    return _validate_power(power, expected_dim=num_beams, source=f"{gain_field}+{angle_field}"), f"{gain_field}+{angle_field}"


def _write_power_vector(path: Path, power: np.ndarray, *, expected_dim: int, source_path: Path) -> None:
    vector = _validate_power(power, expected_dim=expected_dim, source=str(source_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, vector, fmt="%.9g")


def _validate_power(power: np.ndarray, *, expected_dim: int, source: str) -> np.ndarray:
    vector = np.asarray(power, dtype=np.float64).reshape(-1)
    if vector.size != int(expected_dim):
        raise ValueError(f"Derived power vector from {source} has {vector.size} values; expected {expected_dim}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"Derived power vector from {source} contains NaN or Inf.")
    return vector.astype(np.float32)


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


def _compute_beam_gain_for_channel(channel: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    array = np.asarray(channel)
    antenna_axes = [idx for idx, size in enumerate(array.shape) if int(size) == int(codebook.shape[0])]
    if antenna_axes and antenna_axes[0] != 0:
        array = np.moveaxis(array, antenna_axes[0], 0).reshape(codebook.shape[0], -1)
    return compute_beam_gain(array, codebook)


def _angles_to_radians(angles: np.ndarray) -> np.ndarray:
    values = np.asarray(angles, dtype=np.float64)
    if np.nanmax(np.abs(values)) > (2.0 * np.pi + 1e-6):
        values = np.deg2rad(values)
    return values


def _beam_histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[int] = Counter()
    for row in rows:
        for key, value in row.items():
            if key.startswith("future_beam_label"):
                counter[int(value)] += 1
    if not counter:
        for row in rows:
            label = row.get("target_label")
            if label is not None:
                counter[int(label)] += 1
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _channel_field_summary(meta: dict[str, Any]) -> dict[str, Any]:
    diagnostics = meta.get("diagnostics") if isinstance(meta.get("diagnostics"), dict) else {}
    return {
        "source_channel_field": meta.get("source_channel_field"),
        "fields": diagnostics.get("fields", []),
        "shape": diagnostics.get("shape"),
        "path_count": meta.get("path_count", 0),
    }


def _channel_path_count(payload: dict[str, Any], source_field: str) -> int:
    if "+" in str(source_field):
        first = str(source_field).split("+", 1)[0]
        try:
            return int(np.asarray(payload[first]).reshape(-1).shape[0])
        except Exception:
            return 0
    for key in ("path_gain", "path_gains", "gain", "gains", "alpha", "alphas", "aod"):
        if key in payload:
            try:
                return int(np.asarray(payload[key]).reshape(-1).shape[0])
            except Exception:
                return 0
    return 0

__all__ = [
    'load_channel_payload',
    'derive_beam_power_from_file',
    'derive_beam_power'
]
