from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


PATH_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "aod": ("aod", "AoD", "departure_angle", "departure", "phi_t", "azimuth_departure"),
    "aoa": ("aoa", "AoA", "arrival_angle", "arrival", "phi_r", "azimuth_arrival"),
    "delay": ("delay", "tau", "time_delay", "tof"),
    "gain_real": ("gain_real", "real", "alpha_real"),
    "gain_imag": ("gain_imag", "imag", "alpha_imag"),
    "path_mask": ("path_mask", "mask", "valid"),
}


@dataclass(frozen=True)
class PhysicsAdapterConfig:
    required_fields: tuple[str, ...] = ()
    field_map: Mapping[str, str] | None = None
    use_csi_input: bool = False
    csi_input_mode: str = "none"
    history_len: int = 5
    partial_subcarrier_ratio: float = 0.25
    partial_antenna_ratio: float = 0.25
    csi_noise_snr_db: float = 20.0
    allow_oracle_full_csi_input: bool = False
    pilot_subcarrier_stride: int = 4
    pilot_antenna_stride: int = 4
    pilot_pattern: str = "grid"
    pilot_random_seed: int = 0
    num_pred: int = 1
    num_paths: int | None = None

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | bool | None) -> "PhysicsAdapterConfig":
        if value is True:
            return cls()
        if not isinstance(value, Mapping):
            return cls()
        required = tuple(str(item) for item in value.get("required_fields", ()) or ())
        field_map = value.get("field_map") if isinstance(value.get("field_map"), Mapping) else None
        return cls(
            required_fields=required,
            field_map=field_map,
            use_csi_input=bool(value.get("use_csi_input", False)),
            csi_input_mode=str(value.get("csi_input_mode", "none")),
            history_len=int(value.get("history_len", 5)),
            partial_subcarrier_ratio=float(value.get("partial_subcarrier_ratio", 0.25)),
            partial_antenna_ratio=float(value.get("partial_antenna_ratio", 0.25)),
            csi_noise_snr_db=float(value.get("csi_noise_snr_db", 20.0)),
            allow_oracle_full_csi_input=bool(value.get("allow_oracle_full_csi_input", False)),
            pilot_subcarrier_stride=int(value.get("pilot_subcarrier_stride", 4)),
            pilot_antenna_stride=int(value.get("pilot_antenna_stride", 4)),
            pilot_pattern=str(value.get("pilot_pattern", "grid")),
            pilot_random_seed=int(value.get("pilot_random_seed", 0)),
            num_pred=int(value.get("num_pred", 1)),
            num_paths=int(value["num_paths"]) if value.get("num_paths") is not None else None,
        )


def build_mmw_physics_targets(
    sample: Mapping[str, Any],
    config: Mapping[str, Any] | PhysicsAdapterConfig | bool | None = None,
) -> dict[str, Any]:
    cfg = config if isinstance(config, PhysicsAdapterConfig) else PhysicsAdapterConfig.from_value(config)
    metadata = _metadata(sample)
    targets: dict[str, Any] = {
        "metadata": {
            "sample_id": str(sample.get("sample_id", metadata.get("sample_id", ""))),
            "scene": str(metadata.get("scenario", metadata.get("scene", ""))),
            "condition": str(metadata.get("condition", "")),
            "town": str(metadata.get("town", "")),
            "field_mapping": {},
            "sources": {},
        },
        "unavailable_reasons": {},
    }

    _add_csi(targets, sample, cfg)
    _add_beamspace_power(targets, sample)
    _add_path_params(targets, sample, cfg)
    _enforce_required(targets, cfg, sample)
    return targets


def sort_paths_by_gain_magnitude(path_params: torch.Tensor, path_mask: torch.Tensor | None = None) -> torch.Tensor:
    if path_params.ndim < 2 or path_params.shape[-1] < 5:
        raise ValueError(f"path_params must have shape [..., P, 5+], got {tuple(path_params.shape)}.")
    gain = path_params[..., 3].square() + path_params[..., 4].square()
    if path_mask is not None:
        gain = gain.masked_fill(~path_mask.to(dtype=torch.bool, device=path_params.device), float("-inf"))
    order = torch.argsort(gain, dim=-1, descending=True)
    return torch.gather(path_params, -2, order.unsqueeze(-1).expand(*order.shape, path_params.shape[-1]))


def physics_shape_summary(sample: Mapping[str, Any]) -> dict[str, Any]:
    physics = sample.get("physics_targets")
    if not isinstance(physics, Mapping):
        physics = build_mmw_physics_targets(sample)
    path = physics.get("path_params")
    csi = physics.get("csi_target")
    csi_input = physics.get("csi_input", sample.get("csi_input"))
    beam = physics.get("beamspace_power")
    return {
        "image": _shape(sample.get("image")),
        "csi_input": _shape(csi_input),
        "csi_target": _shape(csi),
        "beamspace_power": _shape(beam),
        "path_params": _shape(path),
        "num_subcarriers": _dim(csi, -3),
        "num_antennas": _dim(csi, -2),
        "num_beams": _dim(beam, -1),
        "num_paths": _dim(path, -2),
        "modality_availability": sample.get("modality_availability", {}),
        "unavailable_reasons": dict(physics.get("unavailable_reasons", {})),
    }


def _add_csi(targets: dict[str, Any], sample: Mapping[str, Any], cfg: PhysicsAdapterConfig) -> None:
    value = sample.get("csi")
    if torch.is_tensor(value):
        target = _current_csi_target(value.to(dtype=torch.float32), cfg.num_pred)
        csi_input, csi_mask = _csi_input_from_target(value.to(dtype=torch.float32), target, cfg)
        targets["csi_target"] = target
        targets["csi_target_valid"] = torch.isfinite(target).all()
        targets["csi_input_valid"] = torch.tensor(csi_input is not None)
        if csi_input is not None:
            targets["csi_input"] = csi_input
        if csi_mask is not None:
            targets["csi_observation_mask"] = csi_mask
            observed = csi_mask.to(dtype=torch.float32).mean().item()
            targets["metadata"]["sources"]["csi_input"] = f"sparse_pilot:{cfg.pilot_pattern}"
            targets["metadata"]["csi_input"] = {
                "mode": cfg.csi_input_mode,
                "pilot_pattern": cfg.pilot_pattern,
                "pilot_subcarrier_stride": cfg.pilot_subcarrier_stride,
                "pilot_antenna_stride": cfg.pilot_antenna_stride,
                "observed_fraction": observed,
            }
        targets["csi_input_mode"] = cfg.csi_input_mode
        targets["metadata"]["sources"]["csi_target"] = "sample.csi_full_current"
        return
    targets["csi_target_valid"] = torch.tensor(False)
    targets["csi_input_valid"] = torch.tensor(False)
    targets["csi_input_mode"] = "none"
    targets["unavailable_reasons"]["csi_target"] = "missing_csi_full_current"


def _current_csi_target(csi: torch.Tensor, num_pred: int) -> torch.Tensor:
    horizon = max(int(num_pred), 1)
    if csi.ndim >= 4:
        return csi[-horizon:, ...]
    return csi


def _csi_input_from_target(
    csi_sequence: torch.Tensor,
    csi_target: torch.Tensor,
    cfg: PhysicsAdapterConfig,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    mode = cfg.csi_input_mode.strip().lower()
    if not cfg.use_csi_input or mode == "none":
        return None, None
    if mode == "oracle_full":
        if not cfg.allow_oracle_full_csi_input:
            raise RuntimeError(
                "csi_input_mode='oracle_full' requires allow_oracle_full_csi_input=true; "
                "current full CSI as input is an oracle upper-bound and may leak labels."
            )
        return csi_target.clone(), None
    if mode == "history":
        cutoff = max(int(csi_sequence.shape[0]) - max(int(cfg.num_pred), 1), 0)
        history = csi_sequence[:cutoff][-max(int(cfg.history_len), 1) :, ...]
        return (history.clone() if history.numel() else torch.zeros_like(csi_target[:1])), None
    if mode == "partial":
        return _partial_csi(csi_target, cfg), None
    if mode == "sparse_pilot":
        return _sparse_pilot_csi(csi_target, cfg)
    if mode == "noisy":
        return _noisy_csi(csi_target, cfg.csi_noise_snr_db), None
    if mode == "compressed":
        return csi_target.mean(dim=(-3, -2), keepdim=True), None
    raise ValueError(f"Unsupported csi_input_mode {cfg.csi_input_mode!r}.")


def _partial_csi(csi: torch.Tensor, cfg: PhysicsAdapterConfig) -> torch.Tensor:
    result = torch.zeros_like(csi)
    subcarriers = max(1, int(csi.shape[-3] * max(min(cfg.partial_subcarrier_ratio, 1.0), 0.0)))
    antennas = max(1, int(csi.shape[-2] * max(min(cfg.partial_antenna_ratio, 1.0), 0.0)))
    result[..., :subcarriers, :antennas, :] = csi[..., :subcarriers, :antennas, :]
    return result


def _sparse_pilot_csi(csi: torch.Tensor, cfg: PhysicsAdapterConfig) -> tuple[torch.Tensor, torch.Tensor]:
    pattern = cfg.pilot_pattern.strip().lower()
    if pattern not in {"grid", "random"}:
        raise ValueError(f"Unsupported sparse pilot pattern {cfg.pilot_pattern!r}.")
    mask = torch.zeros(csi.shape[:-1], dtype=torch.bool, device=csi.device)
    sub_stride = max(int(cfg.pilot_subcarrier_stride), 1)
    ant_stride = max(int(cfg.pilot_antenna_stride), 1)
    if pattern == "grid":
        mask[..., ::sub_stride, ::ant_stride] = True
    else:
        probability = min(1.0, 1.0 / float(sub_stride * ant_stride))
        generator = torch.Generator(device=csi.device)
        generator.manual_seed(int(cfg.pilot_random_seed))
        mask = torch.rand(mask.shape, generator=generator, device=csi.device).lt(probability)
        if not torch.any(mask):
            mask[..., 0, 0] = True
    return csi * mask.unsqueeze(-1).to(dtype=csi.dtype), mask


def _noisy_csi(csi: torch.Tensor, snr_db: float) -> torch.Tensor:
    power = csi.square().mean().clamp_min(1e-12)
    noise_power = power / (10.0 ** (float(snr_db) / 10.0))
    return csi + torch.randn_like(csi) * noise_power.sqrt()


def _add_beamspace_power(targets: dict[str, Any], sample: Mapping[str, Any]) -> None:
    for key in ("beamspace_power_label", "beam_power", "target_beam_distribution"):
        value = sample.get(key)
        if torch.is_tensor(value):
            targets["beamspace_power"] = value.to(dtype=torch.float32)
            valid = sample.get("beamspace_power_available")
            targets["beamspace_power_valid"] = valid.to(dtype=torch.bool) if torch.is_tensor(valid) else torch.isfinite(value).all(dim=-1)
            targets["metadata"]["sources"]["beamspace_power"] = key
            return
    targets["beamspace_power_valid"] = torch.tensor(False)
    targets["unavailable_reasons"]["beamspace_power"] = "missing_beamspace_power"


def _add_path_params(targets: dict[str, Any], sample: Mapping[str, Any], cfg: PhysicsAdapterConfig) -> None:
    value = sample.get("path_params")
    if torch.is_tensor(value):
        params = value.to(dtype=torch.float32)
        if params.shape[-1] >= 5:
            targets["path_params"] = params
            targets["path_mask"] = _path_mask(sample, params)
            targets["metadata"]["sources"]["path_params"] = "sample.path_params"
            return
    payload = _first_mapping(value)
    if payload is None:
        payload = _first_mapping(sample.get("path_semantic_diagnostics"))
    mapped, mapping = _map_path_payload(payload, cfg.field_map)
    if mapped is None:
        csi_payload = _path_payload_from_csi_metadata(sample)
        mapped, mapping = _map_path_payload(csi_payload, cfg.field_map)
    if mapped is None:
        targets["path_mask"] = torch.tensor(False)
        targets["unavailable_reasons"]["path_params"] = "missing_path_params"
        return
    mapped = _fit_path_count(mapped, cfg.num_paths)
    targets["path_params"] = mapped
    targets["path_mask"] = mapped[..., 5].to(dtype=torch.bool) if mapped.shape[-1] > 5 else torch.ones(mapped.shape[:-1], dtype=torch.bool)
    targets["metadata"]["field_mapping"].update(mapping)
    targets["metadata"]["sources"]["path_params"] = "path_payload"


def _map_path_payload(
    payload: Mapping[str, Any] | None,
    explicit: Mapping[str, str] | None,
) -> tuple[torch.Tensor | None, dict[str, str]]:
    if not isinstance(payload, Mapping):
        return None, {}
    values: dict[str, torch.Tensor] = {}
    mapping: dict[str, str] = {}
    for standard, aliases in PATH_FIELD_ALIASES.items():
        candidates = (explicit.get(standard),) if explicit and explicit.get(standard) else aliases
        for key in candidates:
            if key in payload:
                values[standard] = torch.as_tensor(payload[key], dtype=torch.float32)
                mapping[str(key)] = standard
                break
    complex_gain = _complex_gain(payload)
    if complex_gain is not None:
        values.setdefault("gain_real", complex_gain.real.to(torch.float32))
        values.setdefault("gain_imag", complex_gain.imag.to(torch.float32))
        mapping.setdefault("complex_gain", "gain_real/gain_imag")
    required = ("aod", "aoa", "delay", "gain_real", "gain_imag")
    if any(key not in values for key in required):
        return None, mapping
    base_shape = torch.broadcast_shapes(*(values[key].shape for key in required))
    columns = [values[key].expand(base_shape) for key in required]
    mask = values.get("path_mask")
    if mask is None:
        mask = torch.ones(base_shape, dtype=torch.float32)
    columns.append(mask.to(dtype=torch.float32).expand(base_shape))
    params = torch.stack(columns, dim=-1)
    return sort_paths_by_gain_magnitude(params, params[..., 5].to(torch.bool)), mapping


def _fit_path_count(params: torch.Tensor, count: int | None) -> torch.Tensor:
    if count is None:
        return params
    target = max(int(count), 1)
    current = int(params.shape[-2])
    if current >= target:
        return params[..., :target, :]
    pad_shape = (*params.shape[:-2], target - current, params.shape[-1])
    pad = torch.zeros(pad_shape, dtype=params.dtype, device=params.device)
    return torch.cat((params, pad), dim=-2)


def _complex_gain(payload: Mapping[str, Any]) -> torch.Tensor | None:
    for key in ("complex_gain", "gain", "alpha", "a"):
        if key not in payload:
            continue
        value = torch.as_tensor(payload[key])
        if torch.is_complex(value):
            return value
        if value.shape[-1:] == (2,):
            return torch.complex(value[..., 0].to(torch.float32), value[..., 1].to(torch.float32))
    return None


def _path_mask(sample: Mapping[str, Any], params: torch.Tensor) -> torch.Tensor:
    value = sample.get("path_mask")
    if torch.is_tensor(value):
        return value.to(dtype=torch.bool)
    if params.shape[-1] > 5:
        return params[..., 5].to(dtype=torch.bool)
    return torch.ones(params.shape[:-1], dtype=torch.bool, device=params.device)


def _enforce_required(targets: Mapping[str, Any], cfg: PhysicsAdapterConfig, sample: Mapping[str, Any]) -> None:
    aliases = {"csi": "csi_target"}
    fields = [aliases.get(field, field) for field in cfg.required_fields]
    missing = [field for field in fields if f"{field}_valid" in targets and not bool(torch.as_tensor(targets[f"{field}_valid"]).any())]
    missing += [field for field in fields if field not in targets and f"{field}_valid" not in targets]
    if missing:
        metadata = _metadata(sample)
        raise RuntimeError(
            "Required MMW physics field is unavailable: "
            f"sample_id={sample.get('sample_id', metadata.get('sample_id', '<unknown>'))}, "
            f"scene={metadata.get('scenario', metadata.get('scene', '<unknown>'))}, "
            f"field={','.join(missing)}, available_keys={sorted(str(key) for key in sample.keys())}."
        )


def _metadata(sample: Mapping[str, Any]) -> Mapping[str, Any]:
    value = sample.get("metadata")
    return value if isinstance(value, Mapping) else {}


def _first_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        raw = value.get("raw") if isinstance(value.get("raw"), Mapping) else value
        return raw
    if isinstance(value, (list, tuple)) and value:
        return _first_mapping(value[0])
    return None


def _path_payload_from_csi_metadata(sample: Mapping[str, Any]) -> Mapping[str, Any] | None:
    metadata = _metadata(sample)
    data_root = metadata.get("data_root")
    rel_paths = metadata.get("csi_path")
    if not data_root or not rel_paths:
        return None
    rel_path = _last_path(rel_paths)
    if rel_path is None:
        return None
    path = Path(str(data_root)) / rel_path
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            if not {"phi_t", "phi_r", "tau", "a"}.issubset(data.files):
                return None
            delay = np.asarray(data["tau"], dtype=np.float32).reshape(-1)
            paths = int(delay.shape[0])
            aod = np.asarray(data["phi_t"], dtype=np.float32).reshape(-1)[:paths]
            aoa = np.asarray(data["phi_r"], dtype=np.float32).reshape(-1)[:paths]
            gain = np.asarray(data["a"]).reshape(-1, paths).mean(axis=0).astype(np.complex64)
    except Exception:
        return None
    mask = np.isfinite(delay) & np.isfinite(aod) & np.isfinite(aoa) & np.isfinite(gain.real) & np.isfinite(gain.imag)
    return {
        "aod": aod,
        "aoa": aoa,
        "delay": delay,
        "gain_real": gain.real,
        "gain_imag": gain.imag,
        "path_mask": mask.astype(np.float32),
    }


def _last_path(value: Any) -> str | None:
    if isinstance(value, str) and value and value != "-99":
        return value
    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            result = _last_path(item)
            if result is not None:
                return result
    return None


def _shape(value: Any) -> list[int] | None:
    return list(value.shape) if torch.is_tensor(value) else None


def _dim(value: Any, index: int) -> int | None:
    if not torch.is_tensor(value):
        return None
    try:
        return int(value.shape[index])
    except IndexError:
        return None


__all__ = [
    "PATH_FIELD_ALIASES",
    "PhysicsAdapterConfig",
    "build_mmw_physics_targets",
    "physics_shape_summary",
    "sort_paths_by_gain_magnitude",
]
