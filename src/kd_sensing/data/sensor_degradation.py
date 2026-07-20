"""Deterministic online sensor degradation for the MMW PGCD screen."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from kd_sensing.data.transform_ops.image import IMAGENET_RGB_MEAN, IMAGENET_RGB_STD
from kd_sensing.modalities import MODALITY_ORDER


SEVERITY_SCALARS = (0.0, 0.25, 0.5, 0.75, 1.0)
SENSOR_CORRUPTIONS = {
    "image": ("gaussian_blur", "patch_occlusion", "exposure_sensor_noise"),
    "lidar": ("point_dropout", "range_dependent_dropout", "coordinate_jitter"),
    "radar": ("detection_dropout", "coordinate_jitter", "false_clutter"),
    "gps": ("white_position_noise", "slow_bias_drift", "sudden_jump", "outage"),
}
TRAIN_CORRUPTIONS = {
    "image": ("gaussian_blur", "patch_occlusion"),
    "lidar": ("point_dropout",),
    "radar": ("detection_dropout",),
    "gps": ("slow_bias_drift",),
}
UNSEEN_CORRUPTIONS = {
    "image": "exposure_sensor_noise",
    "lidar": "coordinate_jitter",
    "radar": "false_clutter",
    "gps": "sudden_jump",
}
_BATCH_KEYS = {
    "image": ("image",),
    "radar": ("radar_ra", "radar_da"),
    "gps": ("gps",),
    "lidar": ("lidar",),
}
_FORBIDDEN_TENSOR_TOKENS = (
    "channel",
    "csi",
    "path_gain",
    "path_feature",
    "beam_gain",
    "beam_power",
    "ray_tracing",
)


@dataclass(frozen=True)
class SensorDegradationResult:
    corrupted_inputs: torch.Tensor
    availability_mask: torch.Tensor
    degradation_metadata: dict[str, Any]


@dataclass(frozen=True)
class BatchDegradationResult:
    corrupted_batch: dict[str, Any]
    availability_mask: torch.Tensor
    severity: torch.Tensor
    corrupted_mask: torch.Tensor
    stale_mask: torch.Tensor
    corruption_types: tuple[tuple[str, ...], ...]
    sample_modes: tuple[str, ...]


def assert_pgcd_channel_free(config: Mapping[str, Any], batch: Mapping[str, Any] | None = None) -> None:
    """Fail closed before PGCD can consume privileged communication data."""

    flags = _find_boolean_flags(config)
    use_channel = flags.get("use_channel", False)
    use_csi = flags.get("use_csi", False)
    use_path_features = flags.get("use_path_features", False)
    use_channel_gain_target = flags.get("use_channel_gain_target", False)
    if any((use_channel, use_csi, use_path_features, use_channel_gain_target)):
        raise ValueError("PGCD forbids channel, CSI, path-feature, and channel-gain inputs or targets.")
    assert not use_channel
    assert not use_csi
    assert not use_path_features
    assert not use_channel_gain_target
    dataset = config.get("data", {}).get("dataset", {}) if isinstance(config.get("data"), Mapping) else {}
    if isinstance(dataset, Mapping) and dataset.get("include_router_utility_targets", False) is not False:
        raise ValueError("PGCD requires data.dataset.include_router_utility_targets=false.")
    if batch is None:
        return
    forbidden = [
        str(key)
        for key, value in batch.items()
        if torch.is_tensor(value) and any(token in str(key).strip().lower() for token in _FORBIDDEN_TENSOR_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"PGCD batch contains forbidden communication tensor fields: {sorted(forbidden)}.")


class SensorDegradationGenerator:
    """Apply representation-aware L0-L4 degradation without touching source files."""

    def __init__(self, global_seed: int, *, modalities: Sequence[str] = MODALITY_ORDER) -> None:
        self.global_seed = int(global_seed)
        self.modalities = tuple(str(item) for item in modalities)
        if self.modalities != MODALITY_ORDER:
            raise ValueError(f"PGCD requires modalities {list(MODALITY_ORDER)}.")

    def generate(
        self,
        sensor_inputs: torch.Tensor,
        sensor_name: str,
        sample_id: str,
        weather: str,
        severity: int,
        corruption_type: str,
        *,
        training: bool,
        source_frame_ids: Sequence[Any] | None = None,
        variant_id: int = 0,
        gps_scaler_mean: Any | None = None,
        gps_scaler_scale: Any | None = None,
    ) -> SensorDegradationResult:
        name = str(sensor_name).strip().lower()
        level = int(severity)
        corruption = str(corruption_type).strip().lower()
        if name not in SENSOR_CORRUPTIONS:
            raise ValueError(f"Unsupported PGCD sensor {sensor_name!r}.")
        if level not in range(5):
            raise ValueError("PGCD severity must be one of L0-L4 encoded as 0..4.")
        if corruption not in SENSOR_CORRUPTIONS[name] and corruption != "one_step_stale":
            raise ValueError(f"Unsupported {name} corruption {corruption!r}.")
        value = torch.as_tensor(sensor_inputs)
        if value.ndim < 2:
            raise ValueError("Sensor degradation inputs must include time and feature dimensions.")
        timesteps = int(value.shape[0])
        available = torch.ones(timesteps, dtype=torch.bool, device=value.device)
        stale = torch.zeros_like(available)
        metadata: dict[str, Any] = {
            "sensor": name,
            "sample_id": str(sample_id),
            "weather": str(weather),
            "severity": level,
            "severity_scalar": SEVERITY_SCALARS[level],
            "corruption_type": corruption,
            "training": bool(training),
            "variant_id": int(variant_id),
        }
        if level == 0:
            return SensorDegradationResult(value.clone(), available, {**metadata, "intensity": 0.0, "stale_mask": stale})
        if level == 4:
            return SensorDegradationResult(torch.zeros_like(value), torch.zeros_like(available), {**metadata, "intensity": 1.0, "stale_mask": stale})

        generator = torch.Generator(device="cpu").manual_seed(
            _derived_seed(self.global_seed, sample_id, name, corruption, variant_id, tuple(source_frame_ids or ()))
        )
        if corruption == "one_step_stale":
            corrupted = value.clone()
            if timesteps > 1:
                corrupted[1:] = value[:-1]
                stale[1:] = True
            return SensorDegradationResult(corrupted, available, {**metadata, "intensity": level / 3.0, "stale_mask": stale})
        if name == "image":
            corrupted, extra = _corrupt_image(value, corruption, level, generator)
        elif name == "lidar":
            corrupted, extra = _corrupt_lidar(value, corruption, level, generator)
        elif name == "radar":
            corrupted, extra = _corrupt_radar(value, corruption, level, generator)
        else:
            corrupted, available, extra = _corrupt_gps(
                value,
                corruption,
                level,
                generator,
                scaler_mean=gps_scaler_mean,
                scaler_scale=gps_scaler_scale,
            )
        corrupted = _share_source_corruption(corrupted, source_frame_ids)
        return SensorDegradationResult(corrupted, available, {**metadata, **extra, "stale_mask": stale})

    def apply_batch(
        self,
        batch: Mapping[str, Any],
        *,
        training: bool,
        epoch: int = 0,
        step: int = 0,
        variant_id: int = 0,
        fixed: Mapping[str, Any] | None = None,
    ) -> BatchDegradationResult:
        sample_ids = _batch_sample_ids(batch)
        if not sample_ids:
            raise ValueError("PGCD degradation requires stable sample identities.")
        batch_size, timesteps = _batch_time_shape(batch)
        if len(sample_ids) != batch_size:
            raise ValueError("PGCD sample identity count must match batch size.")
        result = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
        base = batch.get("modality_temporal_mask")
        availability = (
            torch.as_tensor(base, dtype=torch.bool).clone()
            if torch.is_tensor(base)
            else torch.ones(batch_size, timesteps, len(self.modalities), dtype=torch.bool)
        )
        severity = torch.zeros(batch_size, timesteps, len(self.modalities), dtype=torch.int64)
        corrupted_mask = torch.zeros_like(availability)
        stale_mask = torch.zeros_like(availability)
        type_rows: list[tuple[str, ...]] = []
        modes: list[str] = []
        weather_rows = _batch_weather(batch, batch_size)
        gps_mean = batch.get("gps_scaler_mean")
        gps_scale = batch.get("gps_scaler_scale")
        for sample_index, sample_id in enumerate(sample_ids):
            plan = self._sample_plan(
                sample_id,
                training=training,
                epoch=epoch,
                step=step,
                variant_id=variant_id,
                fixed=fixed,
            )
            modes.append(plan["mode"])
            per_sensor = ["clean"] * len(self.modalities)
            for sensor, level, corruption in plan["items"]:
                modality_index = self.modalities.index(sensor)
                tensor = _stack_sensor(result, sensor, sample_index)
                generated = self.generate(
                    tensor,
                    sensor,
                    sample_id,
                    weather_rows[sample_index],
                    level,
                    corruption,
                    training=training,
                    source_frame_ids=_source_ids(batch, sample_index, modality_index),
                    variant_id=variant_id,
                    gps_scaler_mean=_batch_row(gps_mean, sample_index),
                    gps_scaler_scale=_batch_row(gps_scale, sample_index),
                )
                _unstack_sensor(result, sensor, sample_index, generated.corrupted_inputs)
                local_available = generated.availability_mask.cpu() & availability[sample_index, :, modality_index]
                availability[sample_index, :, modality_index] = local_available
                severity[sample_index, :, modality_index] = int(level)
                corrupted_mask[sample_index, :, modality_index] = int(level) > 0
                stale_value = generated.degradation_metadata.get("stale_mask")
                if torch.is_tensor(stale_value):
                    stale_mask[sample_index, :, modality_index] = stale_value.cpu().bool()
                per_sensor[modality_index] = corruption
            type_rows.append(tuple(per_sensor))
        if not bool(availability.any(dim=(1, 2)).all().item()):
            raise ValueError("PGCD sampling removed every sensor block from a sample.")
        return BatchDegradationResult(
            corrupted_batch=result,
            availability_mask=availability,
            severity=severity,
            corrupted_mask=corrupted_mask,
            stale_mask=stale_mask,
            corruption_types=tuple(type_rows),
            sample_modes=tuple(modes),
        )

    def _sample_plan(
        self,
        sample_id: str,
        *,
        training: bool,
        epoch: int,
        step: int,
        variant_id: int,
        fixed: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if fixed is not None:
            sensors = tuple(str(item) for item in fixed.get("sensors", (fixed.get("sensor"),)) if item)
            levels = fixed.get("severities", (fixed.get("severity", 0),))
            if isinstance(levels, int):
                levels = (levels,)
            corruptions = fixed.get("corruption_types", (fixed.get("corruption_type", ""),))
            if isinstance(corruptions, str):
                corruptions = (corruptions,)
            items = tuple((sensor, int(levels[min(index, len(levels) - 1)]), str(corruptions[min(index, len(corruptions) - 1)])) for index, sensor in enumerate(sensors))
            return {"mode": str(fixed.get("mode", "fixed")), "items": items}
        rng = random.Random(_derived_seed(self.global_seed, sample_id, epoch if training else 0, step if training else 0, variant_id, "plan"))
        draw = rng.random()
        if draw < 0.20:
            return {"mode": "clean_only", "items": ()}
        if draw < 0.60:
            mode, count = "single_sensor_corruption", 1
        elif draw < 0.80:
            mode, count = "two_sensor_corruption", 2
        elif draw < 0.90:
            sensor = rng.choice(self.modalities)
            return {"mode": "temporal_block_corruption", "items": ((sensor, rng.choice((1, 2, 3)), "one_step_stale"),)}
        else:
            sensor = rng.choice(self.modalities)
            return {"mode": "single_sensor_missing", "items": ((sensor, 4, TRAIN_CORRUPTIONS[sensor][0]),)}
        sensors = rng.sample(self.modalities, count)
        items = tuple(
            (sensor, rng.choices((1, 2, 3, 4), weights=(0.30, 0.30, 0.30, 0.10), k=1)[0], rng.choice(TRAIN_CORRUPTIONS[sensor]))
            for sensor in sensors
        )
        return {"mode": mode, "items": items}


def _corrupt_image(value: torch.Tensor, corruption: str, level: int, generator: torch.Generator):
    if corruption == "gaussian_blur":
        sigma = (0.0, 0.7, 1.4, 2.25)[level]
        radius = max(1, math.ceil(3.0 * sigma))
        coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
        kernel_1d /= kernel_1d.sum()
        kernel = torch.outer(kernel_1d, kernel_1d).to(device=value.device, dtype=value.dtype)
        flat = value.reshape(-1, value.shape[-3], value.shape[-2], value.shape[-1])
        weight = kernel.expand(flat.shape[1], 1, -1, -1)
        corrupted = F.conv2d(F.pad(flat, (radius, radius, radius, radius), mode="reflect"), weight, groups=flat.shape[1])
        return corrupted.reshape_as(value), {"intensity": sigma, "sigma": sigma}
    if corruption == "patch_occlusion":
        fraction = (0.0, 0.10, 0.25, 0.40)[level]
        corrupted = value.clone()
        height, width = value.shape[-2:]
        rectangles = 1 + int(torch.randint(0, 3, (), generator=generator).item())
        for frame in corrupted.reshape(-1, *value.shape[-3:]):
            area = fraction * height * width / rectangles
            for _ in range(rectangles):
                aspect = float(torch.empty(()).uniform_(0.6, 1.6, generator=generator).item())
                block_h = min(height, max(1, round(math.sqrt(area / aspect))))
                block_w = min(width, max(1, round(math.sqrt(area * aspect))))
                top = int(torch.randint(0, max(height - block_h + 1, 1), (), generator=generator).item())
                left = int(torch.randint(0, max(width - block_w + 1, 1), (), generator=generator).item())
                frame[..., top : top + block_h, left : left + block_w] = 0
        return corrupted, {"intensity": fraction, "occlusion_fraction": fraction, "rectangle_count": rectangles}
    mean = torch.tensor(IMAGENET_RGB_MEAN, device=value.device, dtype=value.dtype).reshape(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_RGB_STD, device=value.device, dtype=value.dtype).reshape(1, 3, 1, 1)
    raw = (value * std + mean).clamp(0, 1)
    brightness = (1.0, 0.85, 0.70, 0.55)[level]
    contrast = (1.0, 0.90, 0.75, 0.60)[level]
    noise_std = (0.0, 0.02, 0.05, 0.08)[level]
    centered = (raw - raw.mean(dim=(-2, -1), keepdim=True)) * contrast + raw.mean(dim=(-2, -1), keepdim=True)
    noise = torch.randn(raw.shape, generator=generator, dtype=torch.float32).to(device=raw.device, dtype=raw.dtype)
    corrupted = (centered * brightness + noise * noise_std).clamp(0, 1)
    return (corrupted - mean) / std, {"intensity": noise_std, "brightness": brightness, "contrast": contrast}


def _corrupt_lidar(value: torch.Tensor, corruption: str, level: int, generator: torch.Generator):
    retain = (1.0, 0.80, 0.55, 0.30)[level]
    if corruption == "coordinate_jitter":
        pixels = (0, 2, 6, 11)[level]
        shifted = torch.stack([_zero_shift(frame, _randint(generator, -pixels, pixels), _randint(generator, -pixels, pixels)) for frame in value])
        return shifted, {"intensity": pixels * (60.0 / value.shape[-1]), "pixel_shift_limit": pixels}
    occupied = value.abs().amax(dim=-3, keepdim=True).gt(0)
    keep_shape = (*value.shape[:-3], 1, value.shape[-2], value.shape[-1])
    draw = torch.rand(keep_shape, generator=generator, dtype=torch.float32).to(value.device)
    if corruption == "range_dependent_dropout":
        y = torch.linspace(-1, 1, value.shape[-2], device=value.device).reshape(1, 1, -1, 1)
        x = torch.linspace(-1, 1, value.shape[-1], device=value.device).reshape(1, 1, 1, -1)
        radial = torch.sqrt(x.square() + y.square()).clamp(0, 1)
        probability = (retain + (1.0 - retain) * (1.0 - radial)).expand(keep_shape)
        keep = draw < probability
    else:
        keep = draw < retain
    corrupted = value * (keep | ~occupied).to(dtype=value.dtype)
    return corrupted, {"intensity": 1.0 - retain, "retain_probability": retain}


def _corrupt_radar(value: torch.Tensor, corruption: str, level: int, generator: torch.Generator):
    retain = (1.0, 0.80, 0.55, 0.30)[level]
    if corruption == "coordinate_jitter":
        pixels = (0, 1, 3, 5)[level]
        shifted = torch.stack([_zero_shift(frame, _randint(generator, -pixels, pixels), _randint(generator, -pixels, pixels)) for frame in value])
        return shifted, {"intensity": pixels, "pixel_shift_limit": pixels}
    if corruption == "false_clutter":
        fraction = (0.0, 0.0025, 0.0075, 0.015)[level]
        draw = torch.rand((value.shape[0], 1, value.shape[-2], value.shape[-1]), generator=generator).to(value.device)
        clutter = draw < fraction
        amplitude = torch.rand(clutter.shape, generator=generator, dtype=torch.float32).to(value.device, value.dtype)
        upper = value.detach().float().amax().clamp_min(1e-6).to(value.dtype)
        corrupted = torch.where(clutter.expand_as(value), torch.maximum(value, amplitude.expand_as(value) * upper), value)
        return corrupted, {"intensity": fraction, "clutter_fraction": fraction, "clutter_mask": clutter}
    occupied = value.abs().amax(dim=1, keepdim=True).gt(0)
    draw = torch.rand(occupied.shape, generator=generator, dtype=torch.float32).to(value.device)
    keep = draw < retain
    corrupted = value * (keep | ~occupied).to(dtype=value.dtype)
    return corrupted, {"intensity": 1.0 - retain, "retain_probability": retain}


def _corrupt_gps(
    value: torch.Tensor,
    corruption: str,
    level: int,
    generator: torch.Generator,
    *,
    scaler_mean: Any | None,
    scaler_scale: Any | None,
):
    mean, scale = _gps_scaler(value, scaler_mean, scaler_scale)
    physical = value * scale + mean
    radius = physical[..., 0].clamp_min(1e-6)
    xy = torch.stack((radius * physical[..., 2], radius * physical[..., 1]), dim=-1)
    magnitudes = (0.0, 0.5, 1.5, 3.0)
    magnitude = magnitudes[level]
    direction = torch.randn((2,), generator=generator, dtype=torch.float32)
    direction /= direction.norm().clamp_min(1e-6)
    direction = direction.to(device=value.device, dtype=value.dtype)
    available = torch.ones(value.shape[0], dtype=torch.bool, device=value.device)
    if corruption == "slow_bias_drift":
        relative_time = torch.linspace(0, 1, value.shape[0], device=value.device, dtype=value.dtype).unsqueeze(-1)
        xy = xy + relative_time * direction * magnitude
    elif corruption == "white_position_noise":
        noise = torch.randn(xy.shape, generator=generator, dtype=torch.float32).to(value.device, value.dtype)
        xy = xy + noise * magnitude
    elif corruption == "sudden_jump":
        start = max(1, value.shape[0] // 2)
        xy = xy.clone()
        xy[start:] += direction * magnitude
    elif corruption == "outage":
        count = min(value.shape[0], level)
        available[-count:] = False
        xy = xy.clone()
        xy[-count:] = 0
    noisy_radius = torch.linalg.vector_norm(xy, dim=-1).clamp_min(1e-6)
    converted = torch.stack((noisy_radius, xy[..., 1] / noisy_radius, xy[..., 0] / noisy_radius), dim=-1)
    converted = (converted - mean) / scale
    converted = converted.masked_fill(~available.unsqueeze(-1), 0)
    return converted, available, {"intensity": magnitude, "final_offset_m": magnitude}


def _gps_scaler(value: torch.Tensor, mean: Any | None, scale: Any | None):
    if mean is None and scale is None:
        return torch.zeros((1, 3), device=value.device, dtype=value.dtype), torch.ones((1, 3), device=value.device, dtype=value.dtype)
    mean_tensor = torch.as_tensor(mean, device=value.device, dtype=value.dtype).reshape(1, 3)
    scale_tensor = torch.as_tensor(scale, device=value.device, dtype=value.dtype).reshape(1, 3)
    if bool((scale_tensor <= 0).any().item()):
        raise ValueError("PGCD GPS scaler scale must be positive.")
    return mean_tensor, scale_tensor


def _stack_sensor(batch: Mapping[str, Any], sensor: str, index: int) -> torch.Tensor:
    tensors = [batch[key][index] for key in _BATCH_KEYS[sensor] if torch.is_tensor(batch.get(key))]
    if len(tensors) != len(_BATCH_KEYS[sensor]):
        raise ValueError(f"PGCD batch is missing {sensor} tensor inputs.")
    return tensors[0] if len(tensors) == 1 else torch.stack(tensors, dim=1)


def _unstack_sensor(batch: dict[str, Any], sensor: str, index: int, value: torch.Tensor) -> None:
    keys = _BATCH_KEYS[sensor]
    if len(keys) == 1:
        batch[keys[0]][index] = value
        return
    for channel, key in enumerate(keys):
        batch[key][index] = value[:, channel]


def _zero_shift(value: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
    result = torch.zeros_like(value)
    height, width = value.shape[-2:]
    source_y = slice(max(-shift_y, 0), min(height - shift_y, height))
    target_y = slice(max(shift_y, 0), min(height + shift_y, height))
    source_x = slice(max(-shift_x, 0), min(width - shift_x, width))
    target_x = slice(max(shift_x, 0), min(width + shift_x, width))
    result[..., target_y, target_x] = value[..., source_y, source_x]
    return result


def _share_source_corruption(value: torch.Tensor, source_frame_ids: Sequence[Any] | None) -> torch.Tensor:
    if source_frame_ids is None:
        return value
    if len(source_frame_ids) != value.shape[0]:
        raise ValueError("source_frame_ids length must match the sensor history window.")
    result = value.clone()
    first: dict[str, int] = {}
    for index, identity in enumerate(source_frame_ids):
        key = str(identity)
        if key in first:
            result[index] = result[first[key]]
        else:
            first[key] = index
    return result


def _randint(generator: torch.Generator, low: int, high: int) -> int:
    return int(torch.randint(int(low), int(high) + 1, (), generator=generator).item()) if high > low else int(low)


def _batch_sample_ids(batch: Mapping[str, Any]) -> list[str]:
    value = batch.get("sample_id")
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if torch.is_tensor(value):
        return [str(item.item()) for item in value.reshape(-1)]
    return [str(value)] if value not in (None, "") else []


def _batch_weather(batch: Mapping[str, Any], batch_size: int) -> list[str]:
    metadata = batch.get("domain_metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get("condition", "")
        if isinstance(value, (list, tuple)) and len(value) == batch_size:
            return [str(item) for item in value]
        return [str(value)] * batch_size
    return [""] * batch_size


def _batch_time_shape(batch: Mapping[str, Any]) -> tuple[int, int]:
    for keys in _BATCH_KEYS.values():
        for key in keys:
            value = batch.get(key)
            if torch.is_tensor(value) and value.ndim >= 3:
                return int(value.shape[0]), int(value.shape[1])
    raise ValueError("PGCD batch has no sensor sequence tensor.")


def _batch_row(value: Any, index: int) -> Any | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value)
    return tensor[index] if tensor.ndim == 2 else tensor


def _source_ids(batch: Mapping[str, Any], sample: int, modality: int) -> Sequence[Any] | None:
    value = batch.get("source_frame_ids")
    if value is None:
        return None
    try:
        return value[sample][modality]
    except (IndexError, KeyError, TypeError):
        raise ValueError("source_frame_ids must have shape [B,M,T].") from None


def _find_boolean_flags(value: Any) -> dict[str, bool]:
    result = {key: False for key in ("use_channel", "use_csi", "use_path_features", "use_channel_gain_target")}
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in result:
                result[normalized] = result[normalized] or bool(item)
            nested = _find_boolean_flags(item)
            result = {name: result[name] or nested[name] for name in result}
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _find_boolean_flags(item)
            result = {name: result[name] or nested[name] for name in result}
    return result


def _derived_seed(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


__all__ = [
    "BatchDegradationResult",
    "SEVERITY_SCALARS",
    "SENSOR_CORRUPTIONS",
    "TRAIN_CORRUPTIONS",
    "UNSEEN_CORRUPTIONS",
    "SensorDegradationGenerator",
    "SensorDegradationResult",
    "assert_pgcd_channel_free",
]
