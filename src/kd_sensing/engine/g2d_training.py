from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from kd_sensing.diagnostics.g2d_diagnostics import G2DDiagnosticsAccumulator
from kd_sensing.engine.model_output import ModelOutput, adapt_model_output
from kd_sensing.engine.optim import build_model
from kd_sensing.engine.runtime import forward_task_model
from kd_sensing.engine.training_extensions import BaseLossResult, BatchState, ExtensionContext, TrainingExtension
from kd_sensing.distillation.g2d_smp import apply_smp_gradient_mask
from kd_sensing.modalities import MODALITY_ORDER, normalize_modalities
from kd_sensing.utils.artifact_registry import CheckpointResolution, resolve_teacher_checkpoint
from kd_sensing.utils.checkpoint import checkpoint_load_summary, load_model_state
from kd_sensing.utils.paths import resolve_path


TEACHER_MODEL_TYPES = {
    "image": "modular_sequence",
    "radar": "radar_teacher",
    "gps": "gps_teacher",
    "lidar": "modular_sequence",
    "mmwave": "mmwave_teacher",
}


@dataclass(frozen=True)
class TeacherLoadRecord:
    modality: str
    checkpoint: str | None
    source: str
    strict: bool
    summary: dict[str, Any] | None


class TeacherEnsemble(nn.Module):
    def __init__(
        self,
        teachers: dict[str, nn.Module],
        *,
        num_pred: int,
        num_classes: int,
        load_records: list[TeacherLoadRecord] | None = None,
    ):
        super().__init__()
        self.teachers = nn.ModuleDict(teachers)
        self.modalities = tuple(teachers.keys())
        self.num_pred = int(num_pred)
        self.num_classes = int(num_classes)
        self.load_records = list(load_records or [])
        for teacher in self.teachers.values():
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        seq_length: int,
        num_pred: int,
        device: torch.device,
        non_blocking: bool = False,
    ) -> dict[str, ModelOutput]:
        outputs: dict[str, ModelOutput] = {}
        allowed_slots = int(seq_length) + int(num_pred) - 1
        for modality, teacher in self.teachers.items():
            raw = forward_task_model(
                teacher,
                modality,
                batch,
                seq_length=seq_length,
                num_pred=num_pred,
                device=device,
                non_blocking=non_blocking,
            )
            output = adapt_model_output(raw)
            logits = normalize_teacher_logits(
                output.logits,
                num_pred=num_pred,
                num_classes=self.num_classes,
                modality=modality,
                allowed_slots=allowed_slots,
            )
            outputs[modality] = ModelOutput(
                logits=logits,
                input_features=output.input_features,
                output_features=output.output_features,
                diagnostics=output.diagnostics,
            )
        return outputs

    def load_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "modality": record.modality,
                "checkpoint": record.checkpoint,
                "source": record.source,
                "strict": record.strict,
                "summary": record.summary,
            }
            for record in self.load_records
        ]


@dataclass
class G2DExtensionState:
    teacher_ensemble: TeacherEnsemble | None
    checkpoint_loads: list[dict[str, Any]]
    accumulator: G2DDiagnosticsAccumulator | None = None


class G2DTrainingExtension(TrainingExtension):
    name = "g2d"

    def setup(self, context: ExtensionContext) -> G2DExtensionState:
        if context.cfg.get("distillation", {}).get("type", "no_kd") != "g2d":
            return G2DExtensionState(None, [])
        teacher_ensemble = build_g2d_teacher_ensemble(context.cfg, context.device)
        return G2DExtensionState(teacher_ensemble, teacher_ensemble.load_summary())

    def checkpoint_loads(self, state: G2DExtensionState) -> list[dict[str, Any]]:
        return list(state.checkpoint_loads)

    def before_epoch(self, context: ExtensionContext, state: G2DExtensionState, *, epoch: int) -> None:
        del epoch
        enabled = context.cfg.get("distillation", {}).get("g2d", {}).get("diagnostics", {}).get("enabled", True)
        state.accumulator = (
            G2DDiagnosticsAccumulator(
                num_pred=context.num_pred,
                horizon_names=getattr(
                    context.distiller,
                    "horizon_names",
                    [f"t+{idx + 1}" for idx in range(context.num_pred)],
                ),
            )
            if state.teacher_ensemble is not None and enabled
            else None
        )

    def compute_base_loss(
        self,
        context: ExtensionContext,
        state: G2DExtensionState,
        batch_state: BatchState,
    ) -> BaseLossResult | None:
        if state.teacher_ensemble is None:
            return None
        teacher_outputs = state.teacher_ensemble(
            batch_state.batch,
            seq_length=context.seq_length_teacher,
            num_pred=context.num_pred,
            device=context.device,
            non_blocking=context.non_blocking,
        )
        student_output = ModelOutput(
            logits=batch_state.student_logits,
            input_features=batch_state.student_output.input_features,
            output_features=batch_state.student_output.output_features,
            diagnostics=batch_state.student_output.diagnostics,
        )
        result = context.distiller.compute(
            student_output,
            teacher_outputs,
            batch_state.labels,
            epoch=batch_state.epoch,
        )
        if state.accumulator is not None:
            state.accumulator.update(result.diagnostics)
        active_modalities = result.active_modalities if getattr(context.distiller, "smp_enabled", False) else None
        return BaseLossResult(
            total_loss=result.total_loss,
            task_loss=result.supervised_loss,
            distill_loss=result.distill_loss,
            teacher_diagnostics={},
            diagnostics=g2d_scalar_diagnostics(result.diagnostics),
            active_modalities=active_modalities,
        )

    def after_backward(
        self,
        context: ExtensionContext,
        state: G2DExtensionState,
        batch_state: BatchState,
    ) -> None:
        del state
        if batch_state.active_modalities is not None and getattr(context.distiller, "smp_enabled", False):
            apply_smp_gradient_mask(context.student_model, batch_state.active_modalities)

    def after_epoch(
        self,
        context: ExtensionContext,
        state: G2DExtensionState,
        *,
        epoch: int,
    ) -> dict[str, Any]:
        if state.accumulator is None:
            return {}
        path = state.accumulator.write_epoch(context.run_dir, epoch=epoch + 1)
        payload = state.accumulator.finalize(epoch=epoch + 1)
        return {
            "g2d_diagnostics_path": str(path),
            "g2d_active_modalities": payload.get("active_modalities", []),
        }


def build_g2d_teacher_ensemble(cfg: dict[str, Any], device: torch.device) -> TeacherEnsemble:
    g2d_cfg = cfg.get("distillation", {}).get("g2d", {})
    teachers_cfg = g2d_cfg.get("teachers") or {name: {} for name in MODALITY_ORDER}
    modalities = normalize_modalities(tuple(teachers_cfg.keys()), context="G2D teacher modalities")
    teachers: dict[str, nn.Module] = {}
    records: list[TeacherLoadRecord] = []
    for modality in modalities:
        teacher_cfg = dict(teachers_cfg.get(modality) or {})
        model_cfg = _teacher_model_cfg(cfg, modality, teacher_cfg)
        teacher = build_model(model_cfg).to(device)
        strict = bool(teacher_cfg.get("strict_load", cfg.get("checkpoint", {}).get("strict_load", True)))
        required = bool(teacher_cfg.get("required", True))
        checkpoint = _resolve_teacher_weight(cfg, modality, teacher_cfg)
        summary = None
        if checkpoint.path is None or not checkpoint.path.exists():
            if required:
                raise FileNotFoundError(
                    f"G2D teacher checkpoint for modality '{modality}' was not found. "
                    f"Resolution: {checkpoint.to_dict()}"
                )
        else:
            load_result = load_model_state(
                checkpoint.path,
                teacher,
                role=f"g2d_{modality}_teacher",
                map_location=device,
                strict=strict,
            )
            summary = checkpoint_load_summary(load_result)
            if summary is not None:
                summary.update(
                    {
                        "source": checkpoint.source,
                        "metadata": checkpoint.metadata,
                        "registry_dir": str(checkpoint.registry_dir) if checkpoint.registry_dir is not None else None,
                        "candidates": checkpoint.candidates,
                    }
                )
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        teachers[modality] = teacher
        records.append(
            TeacherLoadRecord(
                modality=modality,
                checkpoint=str(checkpoint.path) if checkpoint.path is not None else None,
                source=checkpoint.source,
                strict=strict,
                summary=summary,
            )
        )
    return TeacherEnsemble(
        teachers,
        num_pred=int(cfg.get("model", {}).get("num_pred", 3)),
        num_classes=int(cfg.get("model", {}).get("num_classes", 64)),
        load_records=records,
    )


def normalize_teacher_logits(
    logits: torch.Tensor,
    *,
    num_pred: int,
    num_classes: int,
    modality: str,
    allowed_slots: int | None = None,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"G2D {modality} teacher logits must have shape [B,T,C], got {tuple(logits.shape)}.")
    if int(logits.shape[-1]) != int(num_classes):
        raise ValueError(
            f"G2D {modality} teacher logits expected classes={num_classes}, got shape {tuple(logits.shape)}."
        )
    if int(logits.shape[1]) == int(num_pred):
        return logits
    if allowed_slots is not None and int(logits.shape[1]) == int(allowed_slots):
        return logits[:, -int(num_pred) :, :]
    raise ValueError(
        f"Expected G2D {modality} teacher logits horizon={num_pred}, got shape {tuple(logits.shape)}. "
        "Teacher outputs with legacy current/beam8 slots are not accepted."
    )


def g2d_scalar_diagnostics(diagnostics: dict) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for key, value in (diagnostics.get("loss") or {}).items():
        if isinstance(value, (int, float)):
            scalars[f"loss/g2d_{key}"] = float(value)
    for modality, values in (diagnostics.get("teacher_confidence") or {}).items():
        if isinstance(values, dict):
            avg = values.get("avg")
            if isinstance(avg, (int, float)):
                scalars[f"g2d/teacher_confidence/{modality}"] = float(avg)
    active = diagnostics.get("active_modalities")
    if isinstance(active, (list, tuple)):
        scalars["g2d/active_count"] = float(len(active))
    return scalars


def _teacher_model_cfg(cfg: dict[str, Any], modality: str, teacher_cfg: dict[str, Any]) -> dict[str, Any]:
    feature_size = int(cfg.get("model", {}).get("feature_size", 64))
    num_classes = int(cfg.get("model", {}).get("num_classes", 64))
    num_pred = int(cfg.get("model", {}).get("num_pred", 3))
    if modality == "image":
        model_cfg = _modular_image_teacher_cfg(feature_size=feature_size, num_classes=num_classes, num_pred=num_pred)
    elif modality == "lidar":
        model_cfg = _modular_lidar_teacher_cfg(feature_size=feature_size, num_classes=num_classes, num_pred=num_pred)
    else:
        model_cfg = {
            "type": TEACHER_MODEL_TYPES[modality],
            "feature_size": feature_size,
            "num_classes": num_classes,
            "gru_params": [64, 64, 1],
        }
    default_fields = {
        "radar": {"radar_channels": 2},
        "gps": {"gps_input_size": 3},
        "lidar": {"lidar_channels": 3},
        "mmwave": {"mmwave_input_size": 64},
    }
    model_cfg.update(default_fields.get(modality, {}))
    role_cfg = cfg.get("model", {}).get("teacher", {})
    allowed_keys = set(default_fields.get(modality, {}))
    for key in allowed_keys:
        if key in role_cfg:
            model_cfg[key] = role_cfg[key]
    configured_model = teacher_cfg.get("model") or {}
    model_cfg.update(deepcopy(configured_model))
    return model_cfg


def _modular_image_teacher_cfg(*, feature_size: int, num_classes: int, num_pred: int) -> dict[str, Any]:
    return {
        "type": "modular_sequence",
        "modalities": ["image"],
        "image_profile": "rgb_imagenet",
        "feature_size": feature_size,
        "d_model": feature_size,
        "num_classes": num_classes,
        "num_pred": num_pred,
        "encoders": {
            "image": {
                "type": "resnet18_imagenet_rgb",
                "output_dim": feature_size,
                "pretrained": True,
                "weights": "DEFAULT",
                "freeze_backbone": True,
                "unfreeze_stages": ["layer4"],
                "dropout": 0.1,
            }
        },
        "representation_core": {
            "type": "single_gru",
            "d_model": feature_size,
            "hidden_size": feature_size,
            "num_layers": 1,
        },
        "heads": {"beam": {"type": "beam_head", "dropout": 0.1}},
    }


def _modular_lidar_teacher_cfg(*, feature_size: int, num_classes: int, num_pred: int) -> dict[str, Any]:
    return {
        "type": "modular_sequence",
        "modalities": ["lidar"],
        "feature_size": feature_size,
        "d_model": feature_size,
        "num_classes": num_classes,
        "num_pred": num_pred,
        "lidar_channels": 3,
        "encoders": {
            "lidar": {
                "type": "lidar_cnn",
                "output_dim": feature_size,
                "lidar_channels": 3,
            }
        },
        "representation_core": {
            "type": "single_gru",
            "d_model": feature_size,
            "hidden_size": feature_size,
            "num_layers": 1,
        },
        "heads": {"beam": {"type": "beam_head", "dropout": 0.1}},
    }


def _resolve_teacher_weight(cfg: dict[str, Any], modality: str, teacher_cfg: dict[str, Any]) -> CheckpointResolution:
    raw = teacher_cfg.get("checkpoint", teacher_cfg.get("teacher_model_name", None))
    if raw is not None:
        candidate = Path(str(raw)).expanduser()
        if candidate.is_absolute() or "/" in str(raw) or "\\" in str(raw):
            path = resolve_path(candidate)
            return CheckpointResolution(path=path, source="explicit", requested=str(raw))
    weight_name = "best.pth" if raw is None else str(raw)
    modality_cfg = deepcopy(cfg)
    modality_cfg.setdefault("experiment", {})["task"] = modality
    modality_cfg["experiment"]["name"] = f"{modality}_teacher_no_kd"
    modality_cfg.setdefault("output", {})["run_name"] = f"{modality}_teacher_no_kd"
    modality_cfg.setdefault("distillation", {})["type"] = "no_kd"
    modality_cfg["distillation"]["teacher_model_name"] = None
    return resolve_teacher_checkpoint(modality_cfg, weight_name)


__all__ = [
    "G2DTrainingExtension",
    "TeacherEnsemble",
    "TeacherLoadRecord",
    "build_g2d_teacher_ensemble",
    "g2d_scalar_diagnostics",
    "normalize_teacher_logits",
]
