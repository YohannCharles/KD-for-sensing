from collections.abc import Mapping, MutableMapping
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
    primary_model: Any
    task_criterion: Any
    run_dir: Path
    device: torch.device
    num_pred: int
    num_classes: int
    seq_length: int
    non_blocking: bool


@dataclass(frozen=True)
class ForwardControls:
    force_modality_mask: torch.Tensor | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "ForwardControls | None") -> "ForwardControls":
        if other is None:
            return self
        return ForwardControls(
            force_modality_mask=other.force_modality_mask
            if other.force_modality_mask is not None
            else self.force_modality_mask,
            model_kwargs={**self.model_kwargs, **other.model_kwargs},
        )


@dataclass
class BatchState:
    epoch: int
    step: int
    batch: dict[str, torch.Tensor]
    labels: torch.Tensor
    primary_output: ModelOutput
    primary_logits: torch.Tensor
    controls: ForwardControls
    total_loss: torch.Tensor | None = None
    task_loss: torch.Tensor | None = None
    auxiliary_loss: torch.Tensor | None = None


@dataclass(frozen=True)
class BaseLossResult:
    total_loss: torch.Tensor
    task_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LossBundle:
    total: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)


class EpochDiagnosticsAccumulator:
    _SUM_PREFIXES = ("temporal_missing/mask_count/",)

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
            key: float(value if key.startswith(self._SUM_PREFIXES) else value / max(self.counts.get(key, 0), 1))
            for key, value in self.sums.items()
            if self.counts.get(key, 0) > 0
        }


class TrainingExtension:
    name = "base"
    state_schema_version = 1
    stateless = False

    def setup(self, context: ExtensionContext) -> Any:
        return None

    def checkpoint_loads(self, state: Any) -> list[dict[str, Any]]:
        return []

    def state_dict(self, state: Any) -> dict[str, Any]:
        """Return the minimal mutable extension state needed for exact resume."""
        if state is None:
            return {}
        if isinstance(state, Mapping):
            return dict(state)
        raise TypeError(
            f"Training extension {self.name!r} must implement state_dict/load_state_dict "
            "or explicitly declare stateless=True."
        )

    def load_state_dict(self, state: Any, payload: Mapping[str, Any]) -> None:
        if state is None:
            if payload:
                raise TypeError(f"Training extension {self.name!r} cannot restore state into None.")
            return
        if isinstance(state, MutableMapping):
            state.clear()
            state.update(dict(payload))
            return
        raise TypeError(
            f"Training extension {self.name!r} must implement load_state_dict "
            "or explicitly declare stateless=True."
        )

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
        step: int = 0,
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

    def after_optimizer_step(
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
    stateless = True
