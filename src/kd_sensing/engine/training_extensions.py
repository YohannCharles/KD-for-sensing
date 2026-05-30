from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from kd_sensing.engine.model_output import ModelOutput


@dataclass(frozen=True)
class ExtensionContext:
    cfg: dict[str, Any]
    task: str
    model_cfg: dict[str, Any]
    training_cfg: dict[str, Any]
    student_model: Any
    teacher_model: Any | None
    distiller: Any
    task_criterion: Any
    run_dir: Path
    device: torch.device
    num_pred: int
    num_classes: int
    seq_length_student: int
    seq_length_teacher: int
    non_blocking: bool


@dataclass(frozen=True)
class ForwardControls:
    force_modality_mask: torch.Tensor | None = None

    def merge(self, other: "ForwardControls | None") -> "ForwardControls":
        if other is None:
            return self
        return ForwardControls(
            force_modality_mask=other.force_modality_mask
            if other.force_modality_mask is not None
            else self.force_modality_mask,
        )


@dataclass
class BatchState:
    epoch: int
    step: int
    batch: dict[str, torch.Tensor]
    labels: torch.Tensor
    soft_beam_targets: torch.Tensor | None
    student_output: ModelOutput
    student_logits: torch.Tensor
    controls: ForwardControls
    teacher_logits: torch.Tensor | None = None
    teacher_input_features: torch.Tensor | None = None
    teacher_output_features: torch.Tensor | None = None
    teacher_diagnostics: dict[str, Any] = field(default_factory=dict)
    total_loss: torch.Tensor | None = None
    task_loss: torch.Tensor | None = None
    distill_loss: torch.Tensor | None = None
    active_modalities: list[str] | None = None


@dataclass(frozen=True)
class BaseLossResult:
    total_loss: torch.Tensor
    task_loss: torch.Tensor
    distill_loss: torch.Tensor
    teacher_diagnostics: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)
    active_modalities: list[str] | None = None


@dataclass(frozen=True)
class LossBundle:
    total: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)


class EpochDiagnosticsAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, diagnostics: dict[str, Any] | None) -> None:
        if not isinstance(diagnostics, dict):
            return
        for key, value in diagnostics.items():
            if isinstance(value, (int, float)):
                self.sums[key] = self.sums.get(key, 0.0) + float(value)
                self.counts[key] = self.counts.get(key, 0) + 1

    def mean(self) -> dict[str, float]:
        return {
            key: float(value / max(self.counts.get(key, 0), 1))
            for key, value in self.sums.items()
            if self.counts.get(key, 0) > 0
        }


class TrainingExtension:
    name = "base"

    def setup(self, context: ExtensionContext) -> Any:
        return None

    def checkpoint_loads(self, state: Any) -> list[dict[str, Any]]:
        return []

    def before_epoch(self, context: ExtensionContext, state: Any, *, epoch: int) -> None:
        return None

    def before_forward(
        self,
        context: ExtensionContext,
        state: Any,
        batch: dict[str, torch.Tensor],
        labels: torch.Tensor,
        *,
        epoch: int,
    ) -> ForwardControls:
        return ForwardControls()

    def compute_base_loss(
        self,
        context: ExtensionContext,
        state: Any,
        batch_state: BatchState,
    ) -> BaseLossResult | None:
        return None

    def after_forward(
        self,
        context: ExtensionContext,
        state: Any,
        batch_state: BatchState,
    ) -> LossBundle | None:
        return None

    def after_backward(
        self,
        context: ExtensionContext,
        state: Any,
        batch_state: BatchState,
    ) -> None:
        return None

    def after_epoch(
        self,
        context: ExtensionContext,
        state: Any,
        *,
        epoch: int,
    ) -> dict[str, Any]:
        return {}


class NoOpTrainingExtension(TrainingExtension):
    name = "noop"
