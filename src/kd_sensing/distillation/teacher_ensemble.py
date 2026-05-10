from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from kd_sensing.engine.batch import (
    forward_model,
    prepare_gps_inputs,
    prepare_image_inputs,
    prepare_lidar_inputs,
    prepare_mmwave_inputs,
    prepare_radar_inputs,
)
from kd_sensing.engine.model_output import ModelOutput, adapt_model_output
from kd_sensing.engine.optim import build_model
from kd_sensing.modalities import MODALITY_ORDER, normalize_modalities
from kd_sensing.utils.artifact_registry import CheckpointResolution, resolve_teacher_checkpoint
from kd_sensing.utils.checkpoint import checkpoint_load_summary, load_model_state
from kd_sensing.utils.paths import resolve_path


TEACHER_MODEL_TYPES = {
    "image": "image_teacher",
    "radar": "radar_teacher",
    "gps": "gps_teacher",
    "lidar": "lidar_teacher",
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
            prepared = _prepare_teacher_input(
                modality,
                batch,
                seq_length=seq_length,
                num_pred=num_pred,
                device=device,
                non_blocking=non_blocking,
            )
            raw = _forward_single_teacher(teacher, modality, prepared)
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


def _teacher_model_cfg(cfg: dict[str, Any], modality: str, teacher_cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = {
        "type": TEACHER_MODEL_TYPES[modality],
        "feature_size": int(cfg.get("model", {}).get("feature_size", 64)),
        "num_classes": int(cfg.get("model", {}).get("num_classes", 64)),
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


def _prepare_teacher_input(
    modality: str,
    batch: dict[str, torch.Tensor],
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    non_blocking: bool,
) -> torch.Tensor:
    preparers = {
        "image": prepare_image_inputs,
        "radar": prepare_radar_inputs,
        "gps": prepare_gps_inputs,
        "lidar": prepare_lidar_inputs,
        "mmwave": prepare_mmwave_inputs,
    }
    return preparers[modality](
        batch,
        seq_length=seq_length,
        num_pred=num_pred,
        device=device,
        non_blocking=non_blocking,
    )


def _forward_single_teacher(model: nn.Module, modality: str, prepared: torch.Tensor):
    if modality == "image":
        return forward_model(model, "image", image_batch=prepared)
    if modality == "radar":
        return forward_model(model, "radar", radar_batch=prepared)
    if modality == "gps":
        return forward_model(model, "gps", gps_batch=prepared)
    if modality == "lidar":
        return forward_model(model, "lidar", lidar_batch=prepared)
    if modality == "mmwave":
        return forward_model(model, "mmwave", mmwave_batch=prepared)
    raise ValueError(f"Unsupported G2D teacher modality '{modality}'.")


__all__ = [
    "TeacherEnsemble",
    "TeacherLoadRecord",
    "build_g2d_teacher_ensemble",
    "normalize_teacher_logits",
]
