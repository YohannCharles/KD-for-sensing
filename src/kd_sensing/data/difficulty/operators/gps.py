from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from kd_sensing.data.difficulty.schema import (
    DifficultyContext,
    DifficultyOperatorConfig,
    DifficultyOperatorOutcome,
    DifficultyProfile,
    DifficultyWarning,
)


SCENARIO_C_CANONICAL_CONDITIONS = (
    {
        "id": "C0_sync",
        "severity": 0.0,
        "max_delay_steps": 0,
        "gps_stride": 1,
        "gps_dropout_prob": 0.0,
        "fallback": "zero_fill",
        "use_forward_fill": True,
        "random_delay": False,
    },
    {
        "id": "C1_mild_stale",
        "severity": 1.0,
        "max_delay_steps": 1,
        "gps_stride": 1,
        "gps_dropout_prob": 0.0,
        "fallback": "zero_fill",
        "use_forward_fill": True,
        "random_delay": False,
    },
    {
        "id": "C2_low_rate",
        "severity": 2.0,
        "max_delay_steps": 2,
        "gps_stride": 2,
        "gps_dropout_prob": 0.1,
        "fallback": "forward_fill",
        "use_forward_fill": True,
        "random_delay": False,
    },
    {
        "id": "C3_random_async",
        "severity": 3.0,
        "max_delay_steps": 4,
        "gps_stride_choices": [1, 2, 3],
        "gps_dropout_prob": 0.3,
        "fallback": "forward_fill",
        "use_forward_fill": True,
        "random_delay": True,
    },
    {
        "id": "C4_severe_async",
        "severity": 4.0,
        "max_delay_steps": 4,
        "gps_stride_choices": [2, 3, 4],
        "gps_dropout_prob": 0.5,
        "fallback": "forward_fill",
        "use_forward_fill": True,
        "random_delay": True,
    },
)


class _BaseGpsOperator:
    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    def _generator(
        self,
        profile: DifficultyProfile,
        config: DifficultyOperatorConfig,
        context: DifficultyContext,
    ) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(context.derived_seed(profile, config)))
        return generator

    def _missing_gps(
        self,
        profile: DifficultyProfile,
        config: DifficultyOperatorConfig,
        *,
        severity: float,
        fallback: str = "skip",
    ) -> DifficultyOperatorOutcome | None:
        return DifficultyOperatorOutcome(
            warnings=(
                DifficultyWarning(
                    code="missing_gps",
                    message="GPS tensor is unavailable; GPS difficulty operator was skipped.",
                    profile_id=profile.id,
                    operator=config.type,
                    condition=profile.condition,
                    severity=float(severity),
                    fallback=fallback,
                ),
            )
        )


class GpsCleanOperator(_BaseGpsOperator):
    def __call__(
        self,
        batch: dict[str, Any],
        *,
        config: DifficultyOperatorConfig,
        profile: DifficultyProfile,
        context: DifficultyContext,
    ) -> DifficultyOperatorOutcome:
        return DifficultyOperatorOutcome(metadata={"operator": config.type, "modality": "gps", "state": "clean"})


class GpsGaussianJitterOperator(_BaseGpsOperator):
    def __call__(
        self,
        batch: dict[str, Any],
        *,
        config: DifficultyOperatorConfig,
        profile: DifficultyProfile,
        context: DifficultyContext,
    ) -> DifficultyOperatorOutcome:
        gps = batch.get("gps")
        if not torch.is_tensor(gps):
            return self._missing_gps(profile, config, severity=profile.severity) or DifficultyOperatorOutcome()
        generator = self._generator(profile, config, context)
        std = float(self.params.get("std", self.params.get("std_meters", 1.0))) * float(profile.severity)
        noise = torch.randn(gps.shape, generator=generator, dtype=torch.float32).to(gps.device) * std
        batch["gps"] = gps + noise.to(dtype=gps.dtype)
        return DifficultyOperatorOutcome(
            metadata={"std": std, "seed": int(context.derived_seed(profile, config)), "input_space": "gps_tensor"}
        )


class GpsCumulativeDriftOperator(_BaseGpsOperator):
    def __call__(
        self,
        batch: dict[str, Any],
        *,
        config: DifficultyOperatorConfig,
        profile: DifficultyProfile,
        context: DifficultyContext,
    ) -> DifficultyOperatorOutcome:
        gps = batch.get("gps")
        if not torch.is_tensor(gps):
            return self._missing_gps(profile, config, severity=profile.severity) or DifficultyOperatorOutcome()
        generator = self._generator(profile, config, context)
        drift_scale = float(self.params.get("drift_scale", 1.0)) * float(profile.severity)
        steps = int(gps.shape[1]) if gps.ndim >= 3 else 1
        direction = torch.randn((gps.shape[0], 1, gps.shape[-1]), generator=generator, dtype=torch.float32).to(gps.device)
        if gps.ndim >= 3:
            ramp = torch.linspace(0.0, 1.0, steps=max(steps, 1), dtype=torch.float32, device=gps.device).reshape(1, steps, 1)
            drift = direction * ramp * drift_scale
        else:
            drift = direction.squeeze(1) * drift_scale
        batch["gps"] = gps + drift.to(dtype=gps.dtype)
        return DifficultyOperatorOutcome(metadata={"drift_scale": drift_scale, "steps": steps})


class GpsMissingOperator(_BaseGpsOperator):
    def __call__(
        self,
        batch: dict[str, Any],
        *,
        config: DifficultyOperatorConfig,
        profile: DifficultyProfile,
        context: DifficultyContext,
    ) -> DifficultyOperatorOutcome:
        gps = batch.get("gps")
        if not torch.is_tensor(gps):
            return self._missing_gps(profile, config, severity=profile.severity) or DifficultyOperatorOutcome()
        generator = self._generator(profile, config, context)
        probability = float(self.params.get("probability", self.params.get("dropout_prob", profile.severity)))
        probability = max(0.0, min(probability, 1.0))
        if gps.ndim >= 3:
            dropout_mask = torch.rand(gps.shape[:2], generator=generator, dtype=torch.float32).to(gps.device) < probability
            keep = (~dropout_mask).reshape(*dropout_mask.shape, *([1] * (gps.ndim - 2)))
            valid_mask = ~dropout_mask
        else:
            dropout_mask = torch.rand((gps.shape[0],), generator=generator, dtype=torch.float32).to(gps.device) < probability
            keep = (~dropout_mask).reshape(gps.shape[0], *([1] * (gps.ndim - 1)))
            valid_mask = ~dropout_mask
        batch["gps"] = gps * keep.to(dtype=gps.dtype)
        batch["gps_dropout_mask"] = dropout_mask
        batch["gps_valid_mask"] = valid_mask
        batch["gps_missing_mask"] = valid_mask.reshape(valid_mask.shape[0], -1).all(dim=1)
        affected = int(dropout_mask.sum().item())
        warnings: tuple[DifficultyWarning, ...] = ()
        if affected:
            warnings = (
                DifficultyWarning(
                    code="gps_missing_zero_fill",
                    message="Missing GPS was represented as zero-filled GPS with input reliability masks.",
                    profile_id=profile.id,
                    operator=config.type,
                    condition=profile.condition,
                    severity=float(profile.severity),
                    sample_count=int(gps.shape[0]),
                    affected_count=affected,
                    fallback=str(self.params.get("fallback", profile.fallback)),
                ),
            )
        return DifficultyOperatorOutcome(
            metadata={"dropout_probability": probability, "affected_count": affected},
            warnings=warnings,
        )


class GpsDistractorOperator(_BaseGpsOperator):
    def __call__(
        self,
        batch: dict[str, Any],
        *,
        config: DifficultyOperatorConfig,
        profile: DifficultyProfile,
        context: DifficultyContext,
    ) -> DifficultyOperatorOutcome:
        gps = batch.get("gps")
        if not torch.is_tensor(gps):
            return self._missing_gps(profile, config, severity=profile.severity) or DifficultyOperatorOutcome()
        if gps.shape[0] < 2:
            return DifficultyOperatorOutcome(
                warnings=(
                    DifficultyWarning(
                        code="gps_distractor_unavailable",
                        message="At least two samples are required for GPS distractor intervention.",
                        profile_id=profile.id,
                        operator=config.type,
                        condition=profile.condition,
                        severity=float(profile.severity),
                        sample_count=int(gps.shape[0]),
                        fallback="identity",
                    ),
                )
            )
        generator = self._generator(profile, config, context)
        shift = int(torch.randint(1, int(gps.shape[0]), (1,), generator=generator).item())
        batch["gps"] = torch.roll(gps, shifts=shift, dims=0)
        batch["gps_distractor_shift"] = shift
        return DifficultyOperatorOutcome(metadata={"batch_roll_shift": shift})


class GpsTemporalDelayOperator(_BaseGpsOperator):
    def __call__(
        self,
        batch: dict[str, Any],
        *,
        config: DifficultyOperatorConfig,
        profile: DifficultyProfile,
        context: DifficultyContext,
    ) -> DifficultyOperatorOutcome:
        modality = str(self.params.get("modality", config.modality or "gps"))
        tensor = batch.get(modality)
        if not torch.is_tensor(tensor):
            return DifficultyOperatorOutcome(
                warnings=(
                    DifficultyWarning(
                        code="temporal_modality_unavailable",
                        message=f"{modality} tensor is unavailable; temporal difficulty operator was skipped.",
                        profile_id=profile.id,
                        operator=config.type,
                        condition=profile.condition,
                        severity=float(profile.severity),
                        fallback="skip",
                    ),
                )
            )
        fallback = str(self.params.get("fallback", profile.fallback if profile.fallback != "identity" else "clamp"))
        stride = max(1, int(round(float(self.params.get("gps_stride", self.params.get("stride", 1))))))
        if config.type in {"sampling_rate_mismatch", "gps_low_rate_stride"}:
            stride = max(1, int(round(float(self.params.get("gps_stride", self.params.get("stride", profile.severity))))))
            frame_offset = stride
        else:
            frame_offset = int(round(float(self.params.get("max_delay_steps", self.params.get("delay_steps", profile.severity)))))
        if tensor.ndim < 3:
            if tensor.shape[0] > 1 and frame_offset > 0:
                batch[modality] = torch.roll(tensor, shifts=frame_offset % int(tensor.shape[0]), dims=0)
            return DifficultyOperatorOutcome(
                warnings=(
                    DifficultyWarning(
                        code="temporal_delay_insufficient_history",
                        message="Tensor has no explicit temporal axis; batch-roll fallback was used.",
                        profile_id=profile.id,
                        operator=config.type,
                        condition=profile.condition,
                        severity=float(profile.severity),
                        sample_count=int(tensor.shape[0]) if tensor.ndim else None,
                        fallback="batch_roll",
                    ),
                )
            )
        if config.type in {"sampling_rate_mismatch", "gps_low_rate_stride"}:
            shifted, source_index, delay_steps, valid_mask = _low_rate_stride(tensor, stride=stride, fallback=fallback)
        else:
            shifted, source_index, delay_steps, valid_mask = _fixed_delay(tensor, delay=frame_offset, fallback=fallback)
        batch[modality] = shifted
        if modality == "gps":
            steps = int(tensor.shape[1])
            stale_mask = source_index.ge(0) & (source_index < torch.arange(steps, dtype=torch.long).reshape(1, steps))
            batch["gps_valid_mask"] = valid_mask.to(device=tensor.device)
            batch["gps_stale_mask"] = stale_mask.to(device=tensor.device)
            batch["gps_delay_steps"] = delay_steps.to(device=tensor.device)
            batch["gps_source_index"] = source_index.to(device=tensor.device)
        return DifficultyOperatorOutcome(
            metadata={
                "modality": modality,
                "frame_offset": frame_offset,
                "gps_stride": stride,
                "fallback": fallback,
            }
        )


class ScenarioCAsyncPositionFeedbackOperator(_BaseGpsOperator):
    def __call__(
        self,
        batch: dict[str, Any],
        *,
        config: DifficultyOperatorConfig,
        profile: DifficultyProfile,
        context: DifficultyContext,
    ) -> DifficultyOperatorOutcome:
        gps = batch.get("gps")
        if not torch.is_tensor(gps):
            return DifficultyOperatorOutcome(
                warnings=(
                    DifficultyWarning(
                        code="scenario_c_gps_unavailable",
                        message="GPS tensor is unavailable; Scenario C perturbation was skipped.",
                        profile_id=profile.id,
                        operator=config.type,
                        condition=profile.condition,
                        severity=float(profile.severity),
                        fallback="skip",
                    ),
                )
            )
        if gps.ndim < 3:
            return DifficultyOperatorOutcome(
                warnings=(
                    DifficultyWarning(
                        code="scenario_c_temporal_axis_unavailable",
                        message="GPS tensor has no explicit temporal axis; Scenario C perturbation was skipped.",
                        profile_id=profile.id,
                        operator=config.type,
                        condition=profile.condition,
                        severity=float(profile.severity),
                        sample_count=int(gps.shape[0]) if gps.ndim else None,
                        fallback="skip",
                    ),
                )
            )
        generator = self._generator(profile, config, context)
        condition = _scenario_c_condition_for_severity(self.params, profile.severity)
        return _apply_scenario_c(batch, gps, condition, profile, config, generator=generator, seed=context.derived_seed(profile, config))


def _fixed_delay(
    tensor: torch.Tensor,
    *,
    delay: int,
    fallback: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = int(tensor.shape[0])
    steps = int(tensor.shape[1])
    source_index = torch.full((batch_size, steps), -1, dtype=torch.long)
    delay_steps = torch.zeros((batch_size, steps), dtype=torch.long)
    valid_mask = torch.zeros((batch_size, steps), dtype=torch.bool)
    shifted = torch.zeros_like(tensor)
    delay = max(0, min(int(delay), max(steps - 1, 0)))
    for step in range(steps):
        source = step - delay
        delay_steps[:, step] = delay
        if source >= 0:
            source_index[:, step] = source
            valid_mask[:, step] = True
            shifted[:, step] = tensor[:, source]
        elif fallback in {"clamp", "forward_fill"} and steps > 0:
            shifted[:, step] = tensor[:, 0]
    return shifted, source_index, delay_steps, valid_mask


def _low_rate_stride(
    tensor: torch.Tensor,
    *,
    stride: int,
    fallback: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = int(tensor.shape[0])
    steps = int(tensor.shape[1])
    source_index = torch.full((batch_size, steps), -1, dtype=torch.long)
    delay_steps = torch.zeros((batch_size, steps), dtype=torch.long)
    valid_mask = torch.zeros((batch_size, steps), dtype=torch.bool)
    shifted = torch.zeros_like(tensor)
    stride = max(1, int(stride))
    for step in range(steps):
        source = (step // stride) * stride
        if source <= step:
            source_index[:, step] = source
            delay_steps[:, step] = step - source
            valid_mask[:, step] = True
            shifted[:, step] = tensor[:, source]
        elif fallback in {"clamp", "forward_fill"}:
            shifted[:, step] = tensor[:, 0]
    return shifted, source_index, delay_steps, valid_mask


def _apply_scenario_c(
    batch: dict[str, Any],
    gps: torch.Tensor,
    condition: Mapping[str, Any],
    profile: DifficultyProfile,
    config: DifficultyOperatorConfig,
    *,
    generator: torch.Generator,
    seed: int,
) -> DifficultyOperatorOutcome:
    condition_id = str(condition.get("id", profile.condition))
    batch_size = int(gps.shape[0])
    steps = int(gps.shape[1])
    source_index = torch.full((batch_size, steps), -1, dtype=torch.long)
    delay_steps = torch.zeros((batch_size, steps), dtype=torch.long)
    valid_mask = torch.zeros((batch_size, steps), dtype=torch.bool)
    dropout_mask = torch.zeros((batch_size, steps), dtype=torch.bool)
    stride_per_sample = _scenario_c_stride_per_sample(condition, batch_size=batch_size, generator=generator)
    delay_matrix = _scenario_c_delay_matrix(condition, batch_size=batch_size, steps=steps, generator=generator)
    timestamp_based = bool(condition.get("timestamp_based", False)) or "delay_seconds" in condition
    warnings: list[DifficultyWarning] = []

    image_time = gps_time = None
    if timestamp_based:
        image_time = _metadata_time_matrix(
            batch.get("metadata"),
            names=("image_timestamp", "image_timestamps", "image_time", "image_times", "timestamps", "timestamp"),
            batch_size=batch_size,
            steps=steps,
        )
        gps_time = _metadata_time_matrix(
            batch.get("metadata"),
            names=("gps_timestamp", "gps_timestamps", "gps_time", "gps_times", "timestamps", "timestamp"),
            batch_size=batch_size,
            steps=steps,
        )
        if image_time is None or gps_time is None:
            warnings.append(
                DifficultyWarning(
                    code="scenario_c_timestamp_fallback_frame_index",
                    message="Timestamp-based Scenario C delay requested but timestamps are unavailable; frame-index delay was used.",
                    profile_id=profile.id,
                    operator=config.type,
                    condition=condition_id,
                    severity=float(profile.severity),
                    sample_count=batch_size,
                    fallback="frame_index",
                )
            )
            timestamp_based = False

    if timestamp_based and image_time is not None and gps_time is not None:
        _fill_timestamp_source_indices(
            source_index,
            delay_steps,
            valid_mask,
            image_time=image_time,
            gps_time=gps_time,
            delay_seconds=float(condition.get("delay_seconds", profile.severity)),
            stride_per_sample=stride_per_sample,
            use_forward_fill=bool(condition.get("use_forward_fill", True)),
        )
    else:
        _fill_frame_source_indices(
            source_index,
            delay_steps,
            valid_mask,
            delay_matrix=delay_matrix,
            stride_per_sample=stride_per_sample,
            use_forward_fill=bool(condition.get("use_forward_fill", True)),
        )

    dropout_prob = float(condition.get("gps_dropout_prob", 0.0) or 0.0)
    if dropout_prob > 0.0:
        dropout_mask = torch.rand((batch_size, steps), generator=generator, dtype=torch.float32) < dropout_prob
        valid_mask &= ~dropout_mask
        source_index = torch.where(dropout_mask, torch.full_like(source_index, -1), source_index)

    async_gps = torch.zeros_like(gps)
    fallback = str(condition.get("fallback", "zero_fill"))
    for batch_index in range(batch_size):
        for step_index in range(steps):
            source = int(source_index[batch_index, step_index].item())
            if source >= 0:
                async_gps[batch_index, step_index] = gps[batch_index, source]
            elif fallback in {"clamp", "forward_fill"} and steps > 0:
                async_gps[batch_index, step_index] = gps[batch_index, 0]
    stale_mask = source_index.ge(0) & (source_index < torch.arange(steps, dtype=torch.long).reshape(1, steps))

    batch["gps"] = async_gps
    batch["gps_async"] = async_gps.clone()
    batch["gps_valid_mask"] = valid_mask.to(device=gps.device)
    batch["gps_stale_mask"] = stale_mask.to(device=gps.device)
    batch["gps_delay_steps"] = delay_steps.to(device=gps.device)
    batch["gps_source_index"] = source_index.to(device=gps.device)
    batch["gps_dropout_mask"] = dropout_mask.to(device=gps.device)
    batch["gps_async_condition"] = condition_id
    batch["gps_async_parameters"] = {
        "condition": condition_id,
        "max_delay_steps": int(condition.get("max_delay_steps", 0)),
        "gps_stride": int(condition.get("gps_stride", 0) or 0),
        "gps_stride_choices": list(condition.get("gps_stride_choices", []) or []),
        "gps_dropout_prob": dropout_prob,
        "fallback": fallback,
        "use_forward_fill": bool(condition.get("use_forward_fill", True)),
        "timestamp_based": bool(timestamp_based),
        "seed": int(seed),
    }

    invalid_count = int((~valid_mask).sum().item())
    stale_count = int(stale_mask.sum().item())
    dropout_count = int(dropout_mask.sum().item())
    if invalid_count:
        warnings.append(
            DifficultyWarning(
                code="scenario_c_invalid_gps_zero_fill",
                message="Scenario C produced stale or missing GPS entries; invalid entries were marked with gps_valid_mask.",
                profile_id=profile.id,
                operator=config.type,
                condition=condition_id,
                severity=float(profile.severity),
                sample_count=batch_size,
                affected_count=invalid_count,
                fallback=fallback,
            )
        )
    if stale_count:
        warnings.append(
            DifficultyWarning(
                code="scenario_c_stale_gps",
                message="Scenario C reused non-future historical GPS; stale entries are marked by gps_stale_mask and gps_delay_steps.",
                profile_id=profile.id,
                operator=config.type,
                condition=condition_id,
                severity=float(profile.severity),
                sample_count=batch_size,
                affected_count=stale_count,
                fallback="forward_fill" if bool(condition.get("use_forward_fill", True)) else fallback,
            )
        )
    if dropout_count:
        warnings.append(
            DifficultyWarning(
                code="scenario_c_gps_dropout",
                message="Scenario C GPS dropout was applied deterministically.",
                profile_id=profile.id,
                operator=config.type,
                condition=condition_id,
                severity=float(profile.severity),
                sample_count=batch_size,
                affected_count=dropout_count,
                fallback=fallback,
            )
        )
    return DifficultyOperatorOutcome(
        metadata={
            "condition": condition_id,
            "input_space": "gps_tensor",
            "source_index": "gps_source_index",
            "valid_mask": "gps_valid_mask",
            "stale_mask": "gps_stale_mask",
            "dropout_mask": "gps_dropout_mask",
            "parameters": dict(batch["gps_async_parameters"]),
        },
        warnings=tuple(warnings),
    )


def _scenario_c_condition_for_severity(params: Mapping[str, Any], severity: float) -> dict[str, Any]:
    conditions = params.get("scenario_c_conditions", params.get("conditions", []))
    if not isinstance(conditions, (list, tuple)) or not conditions:
        conditions = SCENARIO_C_CANONICAL_CONDITIONS
    for condition in conditions:
        if isinstance(condition, Mapping) and math.isclose(float(condition.get("severity", 0.0)), float(severity), abs_tol=1e-9):
            return dict(condition)
    return dict(conditions[-1]) if isinstance(conditions[-1], Mapping) else {}


def _scenario_c_stride_per_sample(
    condition: Mapping[str, Any],
    *,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    choices = condition.get("gps_stride_choices")
    if isinstance(choices, (list, tuple)) and choices:
        values = torch.tensor([int(item) for item in choices], dtype=torch.long)
        indices = torch.randint(0, int(values.numel()), (batch_size,), generator=generator)
        return values[indices]
    stride = int(condition.get("gps_stride", 1) or 1)
    return torch.full((batch_size,), max(stride, 1), dtype=torch.long)


def _scenario_c_delay_matrix(
    condition: Mapping[str, Any],
    *,
    batch_size: int,
    steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    max_delay = max(0, int(condition.get("max_delay_steps", condition.get("delay_steps", 0)) or 0))
    if max_delay <= 0:
        return torch.zeros((batch_size, steps), dtype=torch.long)
    if bool(condition.get("random_delay", False)):
        return torch.randint(0, max_delay + 1, (batch_size, steps), generator=generator, dtype=torch.long)
    return torch.full((batch_size, steps), max_delay, dtype=torch.long)


def _fill_frame_source_indices(
    source_index: torch.Tensor,
    delay_steps: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    delay_matrix: torch.Tensor,
    stride_per_sample: torch.Tensor,
    use_forward_fill: bool,
) -> None:
    batch_size, steps = source_index.shape
    for batch_index in range(batch_size):
        stride = max(1, int(stride_per_sample[batch_index].item()))
        for step_index in range(steps):
            requested = max(0, int(delay_matrix[batch_index, step_index].item()))
            base = step_index - requested
            if base < 0:
                delay_steps[batch_index, step_index] = requested
                continue
            if stride > 1:
                if use_forward_fill:
                    source = (base // stride) * stride
                elif base % stride == 0:
                    source = base
                else:
                    delay_steps[batch_index, step_index] = step_index - base
                    continue
            else:
                source = base
            if source < 0 or source > step_index:
                delay_steps[batch_index, step_index] = max(0, step_index - source)
                continue
            source_index[batch_index, step_index] = source
            delay_steps[batch_index, step_index] = step_index - source
            valid_mask[batch_index, step_index] = True


def _fill_timestamp_source_indices(
    source_index: torch.Tensor,
    delay_steps: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    image_time: torch.Tensor,
    gps_time: torch.Tensor,
    delay_seconds: float,
    stride_per_sample: torch.Tensor,
    use_forward_fill: bool,
) -> None:
    batch_size, steps = source_index.shape
    for batch_index in range(batch_size):
        stride = max(1, int(stride_per_sample[batch_index].item()))
        gps_row = gps_time[batch_index]
        for step_index in range(steps):
            threshold = float(image_time[batch_index, step_index].item()) - float(delay_seconds)
            candidates = []
            for candidate in range(step_index + 1):
                if float(gps_row[candidate].item()) <= threshold:
                    if stride <= 1 or candidate % stride == 0 or use_forward_fill:
                        candidates.append(candidate)
            if not candidates:
                continue
            source = max(candidates)
            if stride > 1 and use_forward_fill:
                sampled = [candidate for candidate in candidates if candidate % stride == 0]
                if sampled:
                    source = max(sampled)
            elif stride > 1 and source % stride != 0:
                continue
            source_index[batch_index, step_index] = source
            delay_steps[batch_index, step_index] = step_index - source
            valid_mask[batch_index, step_index] = True


def _metadata_time_matrix(
    metadata: Any,
    *,
    names: tuple[str, ...],
    batch_size: int,
    steps: int,
) -> torch.Tensor | None:
    if not isinstance(metadata, Mapping):
        return None
    for name in names:
        if name not in metadata:
            continue
        value = metadata[name]
        try:
            tensor = torch.as_tensor(value, dtype=torch.float64)
        except Exception:
            continue
        if tensor.ndim == 1:
            if int(tensor.shape[0]) == steps:
                tensor = tensor.reshape(1, steps).expand(batch_size, steps)
            elif int(tensor.shape[0]) == batch_size:
                tensor = tensor.reshape(batch_size, 1).expand(batch_size, steps)
            else:
                continue
        elif tensor.ndim >= 2:
            tensor = tensor.reshape(int(tensor.shape[0]), int(tensor.shape[1]), *tensor.shape[2:])
            if tensor.ndim > 2:
                tensor = tensor[..., 0]
        else:
            continue
        if int(tensor.shape[0]) >= batch_size and int(tensor.shape[1]) >= steps:
            return tensor[:batch_size, :steps].clone()
    return None
