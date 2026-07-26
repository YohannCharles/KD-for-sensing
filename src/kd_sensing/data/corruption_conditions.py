"""Graded sensor corruption conditions for the Router observability screen.

The 45 conditions are reconstructed from the condition ids recorded in
``outputs/prototype_collapse_diagnostics/diagnostic_sample_manifest.csv``.  The
original injector was removed with the diagnostics tool, so these operators are
a documented reimplementation of that condition table, not a byte-identical
restoration.  Only the condition inventory, severity ladder and severity scalars
are restored verbatim.

Every operator is deterministic given an explicit generator and acts on the
canonical U0 fusion inputs:

    image  [B, T, 3, 224, 224]      radar [B, T, 2, 128, 64]
    lidar  [B, T, 3, 224, 224]      gps   [B, T, 3]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch
import torch.nn.functional as F

from kd_sensing.modalities import MODALITY_ORDER


SEVERITY_SCALARS: dict[int, float] = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
BATCH_KEYS: dict[str, str] = {name: f"{name}_batch" for name in MODALITY_ORDER}


@dataclass(frozen=True)
class CorruptionCondition:
    """One entry of the restored 45-condition table."""

    condition_id: str
    modality: str
    corruption_type: str
    severity: int

    @property
    def severity_scalar(self) -> float:
        return SEVERITY_SCALARS[self.severity]

    @property
    def is_clean(self) -> bool:
        return self.corruption_type == "clean"

    @property
    def forces_missing(self) -> bool:
        return self.corruption_type == "missing"


_GRADED: dict[str, tuple[str, ...]] = {
    "image": ("gaussian_blur", "patch_occlusion", "exposure_sensor_noise"),
    "radar": ("coordinate_jitter", "detection_dropout", "false_clutter"),
    "gps": ("white_position_noise", "slow_bias_drift", "sudden_jump"),
    "lidar": ("point_dropout", "coordinate_jitter", "range_dependent_dropout"),
}


def _build_conditions() -> tuple[CorruptionCondition, ...]:
    conditions = [CorruptionCondition("clean", "all", "clean", 0)]
    for modality in ("image", "radar", "gps", "lidar"):
        for corruption in _GRADED[modality]:
            for severity in (1, 2, 3):
                conditions.append(
                    CorruptionCondition(f"{modality}_{corruption}_l{severity}", modality, corruption, severity)
                )
        conditions.append(CorruptionCondition(f"{modality}_one_step_stale_l2", modality, "one_step_stale", 2))
        conditions.append(CorruptionCondition(f"{modality}_missing_l4", modality, "missing", 4))
    return tuple(conditions)


CONDITIONS: tuple[CorruptionCondition, ...] = _build_conditions()
CONDITIONS_BY_ID: dict[str, CorruptionCondition] = {item.condition_id: item for item in CONDITIONS}
CONDITION_IDS: tuple[str, ...] = tuple(item.condition_id for item in CONDITIONS)

if len(CONDITIONS) != 45:
    raise RuntimeError(f"The restored corruption table must contain 45 conditions, built {len(CONDITIONS)}.")


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def _randn(reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(reference.shape, generator=generator, device=reference.device, dtype=reference.dtype)


def _rand(shape: tuple[int, ...], reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.rand(shape, generator=generator, device=reference.device, dtype=reference.dtype)


def _randint(high: int, count: int, reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randint(0, max(int(high), 1), (count,), generator=generator, device=reference.device)


def _one_step_stale(value: torch.Tensor) -> torch.Tensor:
    """Delay the sequence by one acquisition step; step 0 repeats itself."""
    if value.shape[1] < 2:
        return value.clone()
    index = (torch.arange(value.shape[1], device=value.device) - 1).clamp_min(0)
    return value.index_select(1, index).contiguous()


def _gaussian_blur(value: torch.Tensor, sigma: float) -> torch.Tensor:
    batch, steps, channels, height, width = value.shape
    radius = max(1, int(round(3.0 * sigma)))
    offsets = torch.arange(-radius, radius + 1, device=value.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (offsets / float(sigma)) ** 2)
    kernel = (kernel / kernel.sum()).to(dtype=value.dtype)
    flat = value.reshape(batch * steps, channels, height, width)
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    flat = F.conv2d(F.pad(flat, (radius, radius, 0, 0), mode="reflect"), horizontal, groups=channels)
    flat = F.conv2d(F.pad(flat, (0, 0, radius, radius), mode="reflect"), vertical, groups=channels)
    return flat.reshape(batch, steps, channels, height, width)


def _patch_occlusion(value: torch.Tensor, fraction: float, generator: torch.Generator) -> torch.Tensor:
    batch, steps, _, height, width = value.shape
    side_h = max(1, int(round(height * float(fraction) ** 0.5)))
    side_w = max(1, int(round(width * float(fraction) ** 0.5)))
    rows = torch.arange(height, device=value.device).view(1, 1, height, 1)
    columns = torch.arange(width, device=value.device).view(1, 1, 1, width)
    top = _randint(max(height - side_h, 1), batch * steps, value, generator).view(batch, steps, 1, 1)
    left = _randint(max(width - side_w, 1), batch * steps, value, generator).view(batch, steps, 1, 1)
    inside = (rows >= top) & (rows < top + side_h) & (columns >= left) & (columns < left + side_w)
    return value.masked_fill(inside.unsqueeze(2), 0.0)


def _exposure_sensor_noise(value: torch.Tensor, gain: float, noise: float, generator: torch.Generator) -> torch.Tensor:
    batch, steps = value.shape[:2]
    shape = (batch, steps) + (1,) * (value.ndim - 2)
    scale = 1.0 + (_rand((batch, steps), value, generator) * 2.0 - 1.0).view(shape) * float(gain)
    return value * scale + _randn(value, generator) * float(noise)


def _spatial_jitter(value: torch.Tensor, shift: int, generator: torch.Generator) -> torch.Tensor:
    """Per-(sample, step) integer roll over the two spatial axes."""
    batch, steps = value.shape[:2]
    span = 2 * int(shift) + 1
    vertical = (_randint(span, batch * steps, value, generator) - int(shift)).tolist()
    horizontal = (_randint(span, batch * steps, value, generator) - int(shift)).tolist()
    result = value.clone()
    for index in range(batch * steps):
        row, column = divmod(index, steps)
        result[row, column] = torch.roll(value[row, column], shifts=(vertical[index], horizontal[index]), dims=(-2, -1))
    return result


def _cell_dropout(value: torch.Tensor, probability: torch.Tensor | float, generator: torch.Generator) -> torch.Tensor:
    keep = (_rand(value.shape, value, generator) >= probability).to(dtype=value.dtype)
    return value * keep


def _range_dependent_dropout(value: torch.Tensor, maximum: float, generator: torch.Generator) -> torch.Tensor:
    height, width = value.shape[-2:]
    rows = torch.linspace(-1.0, 1.0, height, device=value.device, dtype=value.dtype).view(-1, 1)
    columns = torch.linspace(-1.0, 1.0, width, device=value.device, dtype=value.dtype).view(1, -1)
    radius = (rows.pow(2) + columns.pow(2)).sqrt()
    probability = (radius / radius.max()).clamp(0.0, 1.0) * float(maximum)
    return _cell_dropout(value, probability, generator)


def _false_clutter(value: torch.Tensor, rate: float, generator: torch.Generator) -> torch.Tensor:
    hits = _rand(value.shape, value, generator) < float(rate)
    amplitude = value.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return torch.where(hits, amplitude * _rand(value.shape, value, generator), value)


def _slow_bias_drift(value: torch.Tensor, scale: float, generator: torch.Generator) -> torch.Tensor:
    batch, steps, features = value.shape
    ramp = torch.linspace(0.0, 1.0, steps, device=value.device, dtype=value.dtype).view(1, steps, 1)
    direction = _randn(value.new_empty((batch, 1, features)), generator)
    return value + direction * ramp * float(scale)


def _sudden_jump(value: torch.Tensor, scale: float, generator: torch.Generator) -> torch.Tensor:
    batch, steps, features = value.shape
    onset = _randint(steps, batch, value, generator).view(batch, 1, 1)
    active = (torch.arange(steps, device=value.device).view(1, steps, 1) >= onset).to(dtype=value.dtype)
    offset = _randn(value.new_empty((batch, 1, features)), generator) * float(scale)
    return value + offset * active


_IMAGE_LIKE: dict[str, Callable[[torch.Tensor, int, torch.Generator], torch.Tensor]] = {
    "gaussian_blur": lambda value, level, generator: _gaussian_blur(value, {1: 1.0, 2: 2.0, 3: 3.0}[level]),
    "patch_occlusion": lambda value, level, generator: _patch_occlusion(value, {1: 0.10, 2: 0.25, 3: 0.40}[level], generator),
    "exposure_sensor_noise": lambda value, level, generator: _exposure_sensor_noise(
        value, {1: 0.2, 2: 0.4, 3: 0.6}[level], {1: 0.05, 2: 0.10, 3: 0.20}[level], generator
    ),
    "point_dropout": lambda value, level, generator: _cell_dropout(value, {1: 0.2, 2: 0.4, 3: 0.6}[level], generator),
    "coordinate_jitter": lambda value, level, generator: _spatial_jitter(value, {1: 4, 2: 8, 3: 16}[level], generator),
    "range_dependent_dropout": lambda value, level, generator: _range_dependent_dropout(
        value, {1: 0.3, 2: 0.6, 3: 0.9}[level], generator
    ),
}

_RADAR: dict[str, Callable[[torch.Tensor, int, torch.Generator], torch.Tensor]] = {
    "coordinate_jitter": lambda value, level, generator: _spatial_jitter(value, {1: 2, 2: 4, 3: 8}[level], generator),
    "detection_dropout": lambda value, level, generator: _cell_dropout(value, {1: 0.2, 2: 0.4, 3: 0.6}[level], generator),
    "false_clutter": lambda value, level, generator: _false_clutter(value, {1: 0.02, 2: 0.05, 3: 0.10}[level], generator),
}

_GPS: dict[str, Callable[[torch.Tensor, int, torch.Generator], torch.Tensor]] = {
    "white_position_noise": lambda value, level, generator: value
    + _randn(value, generator) * {1: 0.10, 2: 0.25, 3: 0.50}[level],
    "slow_bias_drift": lambda value, level, generator: _slow_bias_drift(value, {1: 0.2, 2: 0.5, 3: 1.0}[level], generator),
    "sudden_jump": lambda value, level, generator: _sudden_jump(value, {1: 0.5, 2: 1.0, 3: 2.0}[level], generator),
}

_OPERATORS: dict[str, dict[str, Callable[[torch.Tensor, int, torch.Generator], torch.Tensor]]] = {
    "image": _IMAGE_LIKE,
    "lidar": _IMAGE_LIKE,
    "radar": _RADAR,
    "gps": _GPS,
}


def apply_condition(
    value: torch.Tensor,
    condition: CorruptionCondition,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply one condition to one modality tensor, leaving the caller's tensor intact."""
    if condition.is_clean:
        return value
    if condition.forces_missing:
        return torch.zeros_like(value)
    if condition.corruption_type == "one_step_stale":
        return _one_step_stale(value)
    operators = _OPERATORS.get(condition.modality)
    if operators is None or condition.corruption_type not in operators:
        raise ValueError(f"Unsupported corruption condition: {condition.condition_id!r}")
    return operators[condition.corruption_type](value, condition.severity, generator)


def apply_batch_conditions(
    inputs: Mapping[str, torch.Tensor],
    condition_ids: list[str],
    *,
    seed: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Apply one condition per sample and report which modalities are forced missing.

    Samples sharing a condition are corrupted together so the operators stay
    vectorised, and the generator is seeded from ``(seed, condition index)``.  The
    guarantee is therefore reproducibility for a fixed row ordering, which is what
    the cache build relies on: it runs exactly once over a ``shuffle=False`` loader
    and records the resulting cache digest.  Regrouping the same samples into
    different batches would consume the noise stream differently, so callers must
    not re-derive corrupted inputs on the fly and expect cache-identical values.
    """
    batch = len(condition_ids)
    result = {key: value.clone() for key, value in inputs.items()}
    forced = torch.zeros((batch, len(MODALITY_ORDER)), dtype=torch.bool, device=next(iter(inputs.values())).device)
    for condition_id in sorted(set(condition_ids)):
        condition = CONDITIONS_BY_ID.get(condition_id)
        if condition is None:
            raise ValueError(f"Unknown corruption condition id: {condition_id!r}")
        if condition.is_clean:
            continue
        rows = torch.tensor(
            [index for index, value in enumerate(condition_ids) if value == condition_id],
            dtype=torch.long,
            device=forced.device,
        )
        column = MODALITY_ORDER.index(condition.modality)
        if condition.forces_missing:
            forced[rows, column] = True
        key = BATCH_KEYS[condition.modality]
        generator = torch.Generator(device=result[key].device)
        generator.manual_seed(int(seed) * 1_000_003 + CONDITION_IDS.index(condition_id))
        result[key][rows] = apply_condition(result[key][rows], condition, generator)
    return result, forced


def condition_table() -> list[dict[str, object]]:
    """Serialisable inventory used for run manifests and tests."""
    return [
        {
            "condition_id": item.condition_id,
            "modality": item.modality,
            "corruption_type": item.corruption_type,
            "severity": item.severity,
            "severity_scalar": item.severity_scalar,
        }
        for item in CONDITIONS
    ]


__all__ = [
    "BATCH_KEYS",
    "CONDITIONS",
    "CONDITIONS_BY_ID",
    "CONDITION_IDS",
    "CorruptionCondition",
    "SEVERITY_SCALARS",
    "apply_batch_conditions",
    "apply_condition",
    "condition_table",
]
