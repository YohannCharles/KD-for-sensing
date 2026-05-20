from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.data.transform_ops.io import joined_resource


CSI_FIELD_CANDIDATES = ("csi", "channel", "channels", "h", "H", "array")
PATH_GAIN_FIELD_CANDIDATES = ("a", "gain", "gains", "path_gain", "path_gains", "alpha", "alphas")
AOD_FIELD_CANDIDATES = (
    "aod",
    "ao_d",
    "tx_angle",
    "departure_angle",
    "azimuth_of_departure",
    "phi_t",
    "glob_phi_t",
    "theta_t",
    "glob_theta_t",
)
AOA_FIELD_CANDIDATES = (
    "aoa",
    "ao_a",
    "rx_angle",
    "arrival_angle",
    "azimuth_of_arrival",
    "phi_r",
    "glob_phi_r",
    "theta_r",
    "glob_theta_r",
)
DELAY_FIELD_CANDIDATES = ("delay", "delays", "tau", "taus", "path_delay", "path_delays")


_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "clean": {
        "snr_db": None,
        "path_dropout_rate": 0.0,
        "dominant_path_attenuation": None,
        "delay_noise_std_ns": 0.0,
        "delay_quantization_ns": None,
        "angle_noise_std_deg": 0.0,
        "antenna_phase_error_std_deg": 0.0,
        "temporal_shift_choices": (0,),
    },
    "medium": {
        "snr_db": 10.0,
        "path_dropout_rate": 0.2,
        "dominant_path_attenuation": None,
        "delay_noise_std_ns": 0.5,
        "delay_quantization_ns": None,
        "angle_noise_std_deg": 3.0,
        "antenna_phase_error_std_deg": 10.0,
        "temporal_shift_choices": (-1, 0, 1),
    },
    "hard": {
        "snr_db": 5.0,
        "path_dropout_rate": 0.3,
        "dominant_path_attenuation": 0.5,
        "delay_noise_std_ns": 1.0,
        "delay_quantization_ns": None,
        "angle_noise_std_deg": 5.0,
        "antenna_phase_error_std_deg": 20.0,
        "temporal_shift_choices": (-2, -1, 0, 1, 2),
    },
}

_DEGRADATION_ALIASES = {
    "path_dropout": "path_dropout_rate",
    "dropout_rate": "path_dropout_rate",
    "dominant_attenuation": "dominant_path_attenuation",
    "delay_noise_ns": "delay_noise_std_ns",
    "delay_noise": "delay_noise_std_ns",
    "delay_quantization_step_ns": "delay_quantization_ns",
    "angle_noise_deg": "angle_noise_std_deg",
    "angle_noise": "angle_noise_std_deg",
    "antenna_phase_error_deg": "antenna_phase_error_std_deg",
    "antenna_phase_error": "antenna_phase_error_std_deg",
    "phase_error_std_deg": "antenna_phase_error_std_deg",
    "temporal_shift": "temporal_shift_choices",
}


@dataclass(frozen=True)
class CSIDegradationConfig:
    enabled: bool = False
    profile: str = "clean"
    snr_db: float | None = None
    path_dropout_rate: float = 0.0
    dominant_path_attenuation: float | None = None
    delay_noise_std_ns: float = 0.0
    delay_quantization_ns: float | None = None
    angle_noise_std_deg: float = 0.0
    antenna_phase_error_std_deg: float = 0.0
    temporal_shift_choices: tuple[int, ...] = (0,)
    temporal_fill_mode: str = "clamp"
    seed: int | None = None
    tx_antennas: int = 64
    num_subcarriers: int = 1
    carrier_frequency_hz: float = 60.0e9
    subcarrier_spacing_hz: float = 120.0e3

    def __post_init__(self) -> None:
        profile = str(self.profile or "clean").strip().lower()
        if profile not in _PROFILE_DEFAULTS:
            available = ", ".join(sorted(_PROFILE_DEFAULTS))
            raise ValueError(f"Unknown CSI degradation profile '{self.profile}'. Available profiles: {available}.")
        shifts = tuple(int(value) for value in self.temporal_shift_choices) or (0,)
        fill_mode = str(self.temporal_fill_mode or "clamp").strip().lower()
        if fill_mode != "clamp":
            raise ValueError("csi_degradation.temporal_fill_mode currently supports only 'clamp'.")
        dropout = float(self.path_dropout_rate)
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError(f"path_dropout_rate must be in [0, 1], got {self.path_dropout_rate}.")
        dominant = self.dominant_path_attenuation
        if dominant is not None and float(dominant) < 0.0:
            raise ValueError(
                f"dominant_path_attenuation must be non-negative when provided, got {dominant}."
            )
        if int(self.tx_antennas) <= 0:
            raise ValueError("tx_antennas must be positive.")
        if int(self.num_subcarriers) <= 0:
            raise ValueError("num_subcarriers must be positive.")
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "path_dropout_rate", dropout)
        object.__setattr__(self, "temporal_shift_choices", shifts)
        object.__setattr__(self, "temporal_fill_mode", fill_mode)
        if self.snr_db is not None:
            object.__setattr__(self, "snr_db", float(self.snr_db))
        if dominant is not None:
            object.__setattr__(self, "dominant_path_attenuation", float(dominant))
        object.__setattr__(self, "delay_noise_std_ns", float(self.delay_noise_std_ns or 0.0))
        if self.delay_quantization_ns is not None:
            object.__setattr__(self, "delay_quantization_ns", float(self.delay_quantization_ns))
        object.__setattr__(self, "angle_noise_std_deg", float(self.angle_noise_std_deg or 0.0))
        object.__setattr__(
            self,
            "antenna_phase_error_std_deg",
            float(self.antenna_phase_error_std_deg or 0.0),
        )
        if self.seed is not None:
            object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "tx_antennas", int(self.tx_antennas))
        object.__setattr__(self, "num_subcarriers", int(self.num_subcarriers))
        object.__setattr__(self, "carrier_frequency_hz", float(self.carrier_frequency_hz))
        object.__setattr__(self, "subcarrier_spacing_hz", float(self.subcarrier_spacing_hz))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "profile": self.profile,
            "snr_db": self.snr_db,
            "path_dropout_rate": float(self.path_dropout_rate),
            "dominant_path_attenuation": self.dominant_path_attenuation,
            "delay_noise_std_ns": float(self.delay_noise_std_ns),
            "delay_quantization_ns": self.delay_quantization_ns,
            "angle_noise_std_deg": float(self.angle_noise_std_deg),
            "antenna_phase_error_std_deg": float(self.antenna_phase_error_std_deg),
            "temporal_shift_choices": list(self.temporal_shift_choices),
            "temporal_fill_mode": self.temporal_fill_mode,
            "seed": self.seed,
            "tx_antennas": int(self.tx_antennas),
            "num_subcarriers": int(self.num_subcarriers),
            "carrier_frequency_hz": float(self.carrier_frequency_hz),
            "subcarrier_spacing_hz": float(self.subcarrier_spacing_hz),
        }


def resolve_csi_degradation_config(config: CSIDegradationConfig | dict[str, Any] | bool | None) -> CSIDegradationConfig:
    if isinstance(config, CSIDegradationConfig):
        return config
    if config is None:
        return CSIDegradationConfig()
    if isinstance(config, bool):
        profile = "medium" if config else "clean"
        return CSIDegradationConfig(enabled=bool(config), profile=profile, **_PROFILE_DEFAULTS[profile])
    if not isinstance(config, dict):
        raise TypeError("csi_degradation must be a mapping, boolean, CSIDegradationConfig, or None.")

    raw_profile = config.get("profile")
    profile = str(raw_profile or ("medium" if bool(config.get("enabled", False)) else "clean")).strip().lower()
    if profile not in _PROFILE_DEFAULTS:
        available = ", ".join(sorted(_PROFILE_DEFAULTS))
        raise ValueError(f"Unknown CSI degradation profile '{profile}'. Available profiles: {available}.")
    enabled_default = profile != "clean" if "enabled" not in config else bool(config.get("enabled"))
    values = dict(_PROFILE_DEFAULTS[profile])
    for raw_key, value in config.items():
        if raw_key in {"enabled", "profile"}:
            continue
        key = _DEGRADATION_ALIASES.get(str(raw_key), str(raw_key))
        values[key] = value
    values["temporal_shift_choices"] = _coerce_shift_choices(values.get("temporal_shift_choices", (0,)))
    return CSIDegradationConfig(enabled=enabled_default, profile=profile, **values)


def csi_degradation_sample_seed(
    config: CSIDegradationConfig | dict[str, Any] | bool | None,
    *,
    split: str | None,
    sample_index: int | None,
    csi_paths: list[str] | tuple[str, ...],
    sample_key: str | None = None,
) -> int:
    resolved = resolve_csi_degradation_config(config)
    base_seed = 0 if resolved.seed is None else int(resolved.seed)
    key = "|".join(
        [
            str(base_seed),
            str(split or ""),
            "" if sample_index is None else str(int(sample_index)),
            str(sample_key or ""),
            "|".join(str(path) for path in csi_paths),
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2**32)


def read_csi_tensor(
    data_root: str | Path,
    rel_path: str,
    *,
    degradation: CSIDegradationConfig | dict[str, Any] | bool | None = None,
    rng: np.random.Generator | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> np.ndarray:
    path = joined_resource(data_root, rel_path)
    if not path.exists():
        raise FileNotFoundError(f"CSI file not found: {path}")
    try:
        payload = _load_payload(path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSI file {path}: {exc}") from exc
    config = resolve_csi_degradation_config(degradation)
    if config.enabled:
        if rng is None:
            seed = csi_degradation_sample_seed(config, split=None, sample_index=None, csi_paths=[str(rel_path)])
            rng = np.random.default_rng(seed)
        array, source_field = degrade_csi_payload(payload, config=config, rng=rng, diagnostics=diagnostics)
    else:
        array, source_field = _select_csi_array(payload)
    return coerce_csi_real_imag(array, path=path, source_field=source_field)


def load_csi_sequence(
    data_root: str | Path,
    csi_paths: list[str],
    *,
    seq_len: int,
    degradation: CSIDegradationConfig | dict[str, Any] | bool | None = None,
    split: str | None = None,
    sample_index: int | None = None,
    sample_key: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> np.ndarray:
    selected = [str(path) for path in csi_paths[-int(seq_len) :] if _valid_csi_path(path)]
    config = resolve_csi_degradation_config(degradation)
    frame_diagnostics: list[dict[str, Any]] = []
    if config.enabled:
        sample_seed = csi_degradation_sample_seed(
            config,
            split=split,
            sample_index=sample_index,
            csi_paths=selected,
            sample_key=sample_key,
        )
        rng = np.random.default_rng(sample_seed)
        shifted, shift = _temporal_shift_paths(selected, config=config, rng=rng)
        selected = shifted
        if diagnostics is not None:
            diagnostics.update(
                {
                    "enabled": True,
                    "profile": config.profile,
                    "resolved_parameters": config.to_dict(),
                    "seed": config.seed,
                    "sample_seed": int(sample_seed),
                    "temporal_shift": int(shift),
                    "temporal_fill_mode": config.temporal_fill_mode,
                    "input_paths": list(csi_paths[-int(seq_len) :]),
                    "selected_paths": list(selected),
                    "skipped_operators": [],
                }
            )
    else:
        rng = None
        if diagnostics is not None:
            diagnostics.update({"enabled": False, "profile": config.profile, "resolved_parameters": config.to_dict()})
    tensors = []
    for path in selected:
        frame_diag: dict[str, Any] = {}
        tensors.append(
            read_csi_tensor(
                data_root,
                path,
                degradation=config if config.enabled else None,
                rng=rng,
                diagnostics=frame_diag if config.enabled else None,
            )
        )
        if config.enabled:
            frame_diagnostics.append(frame_diag)
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
    sequence = np.stack(expanded, axis=0).astype(np.float32)
    if diagnostics is not None and config.enabled:
        diagnostics["frames"] = frame_diagnostics
        skipped = {
            str(operator)
            for frame_diag in frame_diagnostics
            for operator in frame_diag.get("skipped_operators", [])
        }
        diagnostics["skipped_operators"] = sorted(skipped)
    return sequence


def degrade_csi_payload(
    payload: dict[str, Any],
    *,
    config: CSIDegradationConfig | dict[str, Any] | bool | None,
    rng: np.random.Generator | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[np.ndarray, str]:
    resolved = resolve_csi_degradation_config(config)
    if not resolved.enabled:
        return _select_csi_array(payload)
    if rng is None:
        rng = np.random.default_rng(csi_degradation_sample_seed(resolved, split=None, sample_index=None, csi_paths=[]))
    diag = diagnostics if diagnostics is not None else {}
    diag.setdefault("profile", resolved.profile)
    diag.setdefault("resolved_parameters", resolved.to_dict())
    diag.setdefault("skipped_operators", [])

    path_components = _extract_path_components(payload)
    if path_components is not None:
        diag["source_mode"] = "path"
        diag["source_fields"] = {
            "gain": path_components["gain_key"],
            "angle": path_components["angle_key"],
            "delay": path_components.get("delay_key"),
        }
        complex_channel = _degrade_path_components(path_components, config=resolved, rng=rng, diagnostics=diag)
        complex_channel = _apply_antenna_phase_error(complex_channel, config=resolved, rng=rng, diagnostics=diag)
        return _complex_to_real_imag(complex_channel), "degraded_path_gains+departure_angles"

    array, source_field = _select_csi_array(payload)
    clean = coerce_csi_real_imag(array, source_field=source_field)
    complex_channel = _real_imag_to_complex(clean)
    diag["source_mode"] = "tensor"
    diag["source_fields"] = {"csi": source_field}
    _skip_path_level_operators(resolved, diag)
    complex_channel = _apply_complex_awgn(complex_channel, config=resolved, rng=rng, diagnostics=diag, source="tensor")
    complex_channel = _apply_antenna_phase_error(complex_channel, config=resolved, rng=rng, diagnostics=diag)
    return _complex_to_real_imag(complex_channel), source_field


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


def _coerce_shift_choices(value: Any) -> tuple[int, ...]:
    if value is None:
        return (0,)
    if isinstance(value, (int, np.integer)):
        return (int(value),)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return (0,)
        if "," in text:
            return tuple(int(part.strip()) for part in text.split(",") if part.strip())
        return (int(text),)
    return tuple(int(item) for item in value) or (0,)


def _temporal_shift_paths(
    paths: list[str],
    *,
    config: CSIDegradationConfig,
    rng: np.random.Generator,
) -> tuple[list[str], int]:
    if not paths:
        return [], 0
    choices = np.asarray(config.temporal_shift_choices, dtype=np.int64)
    shift = int(rng.choice(choices)) if choices.size else 0
    if shift == 0:
        return list(paths), 0
    shifted = []
    last_index = len(paths) - 1
    for idx in range(len(paths)):
        source_idx = min(max(idx + shift, 0), last_index)
        shifted.append(paths[source_idx])
    return shifted, shift


def _extract_path_components(payload: dict[str, Any]) -> dict[str, Any] | None:
    gain_key = _first_present(payload, PATH_GAIN_FIELD_CANDIDATES)
    angle_key = _first_present(payload, AOD_FIELD_CANDIDATES + AOA_FIELD_CANDIDATES)
    if gain_key is None or angle_key is None:
        return None
    gains = _coerce_complex_array(payload[gain_key]).reshape(-1).astype(np.complex128)
    angles = np.asarray(payload[angle_key], dtype=np.float64).reshape(-1)
    if gains.size == 0 or angles.size == 0:
        return None
    count = min(gains.size, angles.size)
    delay_key = _first_present(payload, DELAY_FIELD_CANDIDATES)
    delays = None
    if delay_key is not None:
        delays = _coerce_delay_seconds(payload[delay_key]).reshape(-1)
        if delays.size:
            count = min(count, delays.size)
        else:
            delays = None
    components = {
        "gains": gains[:count],
        "angles_rad": _angles_to_radians(angles[:count]),
        "gain_key": gain_key,
        "angle_key": angle_key,
        "delay_key": delay_key,
    }
    if delays is not None:
        components["delays_s"] = delays[:count]
    return components


def _degrade_path_components(
    components: dict[str, Any],
    *,
    config: CSIDegradationConfig,
    rng: np.random.Generator,
    diagnostics: dict[str, Any],
) -> np.ndarray:
    gains = np.asarray(components["gains"], dtype=np.complex128).copy()
    angles = np.asarray(components["angles_rad"], dtype=np.float64).copy()
    delays = np.asarray(components.get("delays_s", np.zeros(gains.shape, dtype=np.float64)), dtype=np.float64).copy()

    gains = _apply_path_dropout(gains, config=config, diagnostics=diagnostics)
    gains = _apply_dominant_path_attenuation(gains, config=config, diagnostics=diagnostics)
    gains = _apply_complex_awgn(gains, config=config, rng=rng, diagnostics=diagnostics, source="path_gain")
    delays = _apply_delay_degradation(
        delays,
        has_delay=components.get("delay_key") is not None,
        config=config,
        rng=rng,
        diagnostics=diagnostics,
    )
    angles = _apply_angle_noise(angles, config=config, rng=rng, diagnostics=diagnostics)
    return _derive_channel_from_path_arrays(
        gains,
        angles,
        delays,
        tx_antennas=config.tx_antennas,
        num_subcarriers=config.num_subcarriers,
        carrier_frequency_hz=config.carrier_frequency_hz,
        subcarrier_spacing_hz=config.subcarrier_spacing_hz,
    )


def _apply_complex_awgn(
    values: np.ndarray,
    *,
    config: CSIDegradationConfig,
    rng: np.random.Generator,
    diagnostics: dict[str, Any],
    source: str,
) -> np.ndarray:
    if config.snr_db is None:
        return values
    complex_values = np.asarray(values, dtype=np.complex128)
    power = float(np.mean(np.abs(complex_values) ** 2)) if complex_values.size else 0.0
    if not np.isfinite(power) or power <= 0.0:
        _append_skipped(diagnostics, f"awgn_{source}_zero_power")
        return complex_values
    snr_linear = float(10.0 ** (float(config.snr_db) / 10.0))
    noise_variance = power / snr_linear
    std = float(np.sqrt(noise_variance / 2.0))
    noise = rng.normal(0.0, std, size=complex_values.shape) + 1j * rng.normal(0.0, std, size=complex_values.shape)
    diagnostics["awgn"] = {
        "source": source,
        "snr_db": float(config.snr_db),
        "signal_power": power,
        "complex_noise_variance": float(noise_variance),
    }
    return complex_values + noise


def _apply_path_dropout(
    gains: np.ndarray,
    *,
    config: CSIDegradationConfig,
    diagnostics: dict[str, Any],
) -> np.ndarray:
    rate = float(config.path_dropout_rate)
    if rate <= 0.0:
        return gains
    if gains.size <= 1:
        _append_skipped(diagnostics, "path_dropout_insufficient_paths")
        return gains
    power = np.abs(gains) ** 2
    dominant_idx = int(np.argmax(power))
    candidate_indices = [idx for idx in np.argsort(power).tolist() if int(idx) != dominant_idx]
    drop_count = min(int(round(rate * gains.size)), len(candidate_indices))
    if drop_count <= 0:
        return gains
    dropped = [int(idx) for idx in candidate_indices[:drop_count]]
    degraded = gains.copy()
    degraded[dropped] = 0.0 + 0.0j
    diagnostics["path_dropout"] = {
        "rate": rate,
        "dropped_count": int(len(dropped)),
        "dropped_indices": dropped,
        "dominant_index": dominant_idx,
    }
    return degraded


def _apply_dominant_path_attenuation(
    gains: np.ndarray,
    *,
    config: CSIDegradationConfig,
    diagnostics: dict[str, Any],
) -> np.ndarray:
    factor = config.dominant_path_attenuation
    if factor is None:
        return gains
    if gains.size == 0:
        _append_skipped(diagnostics, "dominant_path_attenuation_no_paths")
        return gains
    dominant_idx = int(np.argmax(np.abs(gains) ** 2))
    degraded = gains.copy()
    degraded[dominant_idx] *= float(factor)
    diagnostics["dominant_path_attenuation"] = {
        "factor": float(factor),
        "dominant_index": dominant_idx,
    }
    return degraded


def _apply_delay_degradation(
    delays_s: np.ndarray,
    *,
    has_delay: bool,
    config: CSIDegradationConfig,
    rng: np.random.Generator,
    diagnostics: dict[str, Any],
) -> np.ndarray:
    noise_std_ns = float(config.delay_noise_std_ns)
    quantization_ns = config.delay_quantization_ns
    if not has_delay:
        if noise_std_ns > 0.0:
            _append_skipped(diagnostics, "delay_noise")
        if quantization_ns is not None and float(quantization_ns) > 0.0:
            _append_skipped(diagnostics, "delay_quantization")
        return delays_s
    degraded = np.asarray(delays_s, dtype=np.float64).copy()
    if noise_std_ns > 0.0:
        degraded += rng.normal(0.0, noise_std_ns * 1e-9, size=degraded.shape)
        diagnostics["delay_noise"] = {"std_ns": noise_std_ns}
    if quantization_ns is not None and float(quantization_ns) > 0.0:
        step = float(quantization_ns) * 1e-9
        degraded = np.round(degraded / step) * step
        diagnostics["delay_quantization"] = {"step_ns": float(quantization_ns)}
    return degraded


def _apply_angle_noise(
    angles_rad: np.ndarray,
    *,
    config: CSIDegradationConfig,
    rng: np.random.Generator,
    diagnostics: dict[str, Any],
) -> np.ndarray:
    std_deg = float(config.angle_noise_std_deg)
    if std_deg <= 0.0:
        return angles_rad
    degraded = np.asarray(angles_rad, dtype=np.float64).copy()
    degraded += rng.normal(0.0, np.deg2rad(std_deg), size=degraded.shape)
    diagnostics["angle_noise"] = {"std_deg": std_deg}
    return degraded


def _apply_antenna_phase_error(
    channel: np.ndarray,
    *,
    config: CSIDegradationConfig,
    rng: np.random.Generator,
    diagnostics: dict[str, Any],
) -> np.ndarray:
    std_deg = float(config.antenna_phase_error_std_deg)
    if std_deg <= 0.0:
        return channel
    complex_channel = np.asarray(channel, dtype=np.complex128)
    if complex_channel.ndim == 0:
        _append_skipped(diagnostics, "antenna_phase_error_scalar_channel")
        return complex_channel
    nant = int(complex_channel.shape[-1])
    phase = rng.normal(0.0, np.deg2rad(std_deg), size=(nant,))
    shape = (1,) * (complex_channel.ndim - 1) + (nant,)
    diagnostics["antenna_phase_error"] = {"std_deg": std_deg, "antenna_count": nant}
    return complex_channel * np.exp(1j * phase).reshape(shape)


def _derive_channel_from_path_arrays(
    gains: np.ndarray,
    angles_rad: np.ndarray,
    delays_s: np.ndarray | None,
    *,
    tx_antennas: int,
    num_subcarriers: int,
    carrier_frequency_hz: float,
    subcarrier_spacing_hz: float,
) -> np.ndarray:
    count = min(np.asarray(gains).size, np.asarray(angles_rad).size)
    if count <= 0:
        return np.zeros((int(num_subcarriers), int(tx_antennas)), dtype=np.complex64)
    gains = np.asarray(gains, dtype=np.complex128).reshape(-1)[:count]
    angles = np.asarray(angles_rad, dtype=np.float64).reshape(-1)[:count]
    if delays_s is None:
        delays = np.zeros(count, dtype=np.float64)
    else:
        delays = np.asarray(delays_s, dtype=np.float64).reshape(-1)[:count]
        if delays.size < count:
            delays = np.pad(delays, (0, count - delays.size))
    antennas = np.arange(int(tx_antennas), dtype=np.float64)[:, None]
    steering = np.exp(1j * np.pi * antennas * np.sin(angles)[None, :])
    centered = np.arange(int(num_subcarriers), dtype=np.float64) - (int(num_subcarriers) - 1) / 2.0
    frequencies = float(carrier_frequency_hz) + centered * float(subcarrier_spacing_hz)
    delay_phase = np.exp(-1j * 2.0 * np.pi * frequencies[:, None] * delays[None, :])
    channel = np.einsum("sp,np,p->sn", delay_phase, steering, gains, optimize=True)
    return channel.astype(np.complex64)


def _real_imag_to_complex(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array[..., 0].astype(np.float64) + 1j * array[..., 1].astype(np.float64)


def _complex_to_real_imag(values: np.ndarray) -> np.ndarray:
    complex_values = np.asarray(values)
    output = np.stack([complex_values.real, complex_values.imag], axis=-1).astype(np.float32)
    np.nan_to_num(output, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return output


def _coerce_delay_seconds(value: Any) -> np.ndarray:
    delays = np.asarray(value, dtype=np.float64)
    finite = delays[np.isfinite(delays)]
    if finite.size and np.nanmax(np.abs(finite)) > 1e-6:
        return delays * 1e-9
    return delays


def _skip_path_level_operators(config: CSIDegradationConfig, diagnostics: dict[str, Any]) -> None:
    if config.path_dropout_rate > 0.0:
        _append_skipped(diagnostics, "path_dropout")
    if config.dominant_path_attenuation is not None:
        _append_skipped(diagnostics, "dominant_path_attenuation")
    if config.delay_noise_std_ns > 0.0:
        _append_skipped(diagnostics, "delay_noise")
    if config.delay_quantization_ns is not None and float(config.delay_quantization_ns) > 0.0:
        _append_skipped(diagnostics, "delay_quantization")
    if config.angle_noise_std_deg > 0.0:
        _append_skipped(diagnostics, "angle_noise")


def _append_skipped(diagnostics: dict[str, Any], operator: str) -> None:
    skipped = diagnostics.setdefault("skipped_operators", [])
    if operator not in skipped:
        skipped.append(operator)


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
    "CSIDegradationConfig",
    "CSIRMSNormalizer",
    "CSIRMSStreamingStats",
    "csi_degradation_sample_seed",
    "coerce_csi_real_imag",
    "degrade_csi_payload",
    "fit_csi_rms_from_sequences",
    "load_csi_sequence",
    "read_csi_tensor",
    "resolve_csi_degradation_config",
]
