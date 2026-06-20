from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.engine.training_extensions import BatchState, ExtensionContext, LossBundle, TrainingExtension
from kd_sensing.utils.paths import resolve_path


class TeacherGuidanceTrainingExtension(TrainingExtension):
    name = "teacher_guidance"

    def setup(self, context: ExtensionContext) -> dict[str, Any]:
        cfg = teacher_guidance_config(context.cfg)
        state: dict[str, Any] = {"config": cfg, "tensor": None, "tensor_kind": ""}
        if not cfg["enabled"]:
            return state
        tensor_path = cfg.get("probabilities_path") or cfg.get("logits_path")
        if tensor_path:
            path = resolve_path(tensor_path)
            if path is None or not path.exists():
                if cfg["allow_missing"]:
                    state["missing_reason"] = f"teacher tensor artifact not found: {tensor_path}"
                    return state
                raise FileNotFoundError(f"Teacher guidance tensor artifact not found: {tensor_path}")
            payload = torch.load(path, map_location="cpu")
            tensor = payload.get("probabilities") if isinstance(payload, dict) and cfg.get("probabilities_path") else payload
            if isinstance(payload, dict) and not torch.is_tensor(tensor):
                for key in ("logits", "teacher_logits", "teacher_probabilities"):
                    candidate = payload.get(key)
                    if torch.is_tensor(candidate):
                        tensor = candidate
                        break
            if not torch.is_tensor(tensor):
                raise ValueError("Teacher guidance tensor artifact must contain a tensor or logits/probabilities mapping.")
            state["tensor"] = tensor
            state["tensor_kind"] = "probabilities" if cfg.get("probabilities_path") else "logits"
            state["tensor_path"] = str(path)
        return state

    def checkpoint_loads(self, state: Any) -> list[dict[str, Any]]:
        if not isinstance(state, dict):
            return []
        cfg = state.get("config", {})
        if not isinstance(cfg, dict) or not cfg.get("enabled", False):
            return []
        checkpoint_path = cfg.get("checkpoint_path")
        if not checkpoint_path and not state.get("tensor_path"):
            return []
        return [
            {
                "role": "teacher_guidance_stabilization",
                "path": str(checkpoint_path or state.get("tensor_path")),
                "provenance": cfg.get("checkpoint_provenance", cfg.get("provenance", "")),
                "temperature": float(cfg.get("temperature", 1.0)),
                "weight": float(cfg.get("weight", 0.0)),
                "detach_policy": cfg.get("detach_policy", "detach_teacher"),
                "enabled_splits": list(cfg.get("enabled_splits", ["train"])),
            }
        ]

    def after_forward(
        self,
        context: ExtensionContext,
        state: Any,
        batch_state: BatchState,
    ) -> LossBundle | None:
        if not isinstance(state, dict):
            return None
        cfg = state.get("config", {})
        if not isinstance(cfg, dict) or not cfg.get("enabled", False):
            return None
        if "train" not in {str(item) for item in cfg.get("enabled_splits", ["train"])}:
            return None
        weight = float(cfg.get("weight", 0.0) or 0.0)
        if weight <= 0.0:
            return None
        teacher, kind = _teacher_tensor_for_batch(batch_state, state, cfg=cfg)
        if teacher is None:
            if cfg.get("allow_missing", False):
                zero = batch_state.primary_logits.sum() * 0.0
                return LossBundle(
                    total=zero,
                    components={"teacher_guidance": zero},
                    diagnostics={"loss/teacher_guidance": 0.0, "teacher_guidance/sample_count": 0.0},
                )
            raise ValueError("Teacher guidance is enabled but no teacher logits/probabilities were provided.")
        teacher = teacher.to(device=batch_state.primary_logits.device, dtype=batch_state.primary_logits.dtype)
        if teacher.shape != batch_state.primary_logits.shape:
            raise ValueError(
                "Teacher guidance tensor shape must match student logits, "
                f"got {tuple(teacher.shape)} and {tuple(batch_state.primary_logits.shape)}."
            )
        if bool(cfg.get("detach_teacher", True)) or str(cfg.get("detach_policy", "detach_teacher")) == "detach_teacher":
            teacher = teacher.detach()
        temperature = float(cfg.get("temperature", 1.0) or 1.0)
        if temperature <= 0.0:
            raise ValueError(f"loss.teacher_guidance.temperature must be positive, got {temperature}.")
        teacher_prob = teacher if kind == "probabilities" else torch.softmax(teacher / temperature, dim=-1)
        teacher_prob = teacher_prob / teacher_prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        student_log_prob = F.log_softmax(batch_state.primary_logits / temperature, dim=-1)
        per_slot = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
        valid = torch.isfinite(per_slot)
        loss = per_slot[valid].mean() * (temperature * temperature) * weight if torch.any(valid) else per_slot.sum() * 0.0
        return LossBundle(
            total=loss,
            components={"teacher_guidance": loss},
            diagnostics={
                "loss/teacher_guidance": float(loss.detach().cpu().item()),
                "loss/geometry_teacher_kl": float(loss.detach().cpu().item()),
                "teacher_guidance/weight": weight,
                "teacher_guidance/temperature": temperature,
                "teacher_guidance/sample_count": float(int(valid.sum().detach().cpu().item())),
                "teacher_guidance/detach_teacher": float(bool(cfg.get("detach_teacher", True))),
            },
        )

    def after_epoch(self, context: ExtensionContext, state: Any, *, epoch: int) -> dict[str, Any]:
        if not isinstance(state, dict):
            return {}
        cfg = state.get("config", {})
        if not isinstance(cfg, dict) or not cfg.get("enabled", False):
            return {}
        return {
            "teacher_guidance": {
                "enabled": True,
                "mode": "opt_in_stabilization",
                "checkpoint_path": cfg.get("checkpoint_path", ""),
                "checkpoint_provenance": cfg.get("checkpoint_provenance", cfg.get("provenance", "")),
                "temperature": float(cfg.get("temperature", 1.0)),
                "weight": float(cfg.get("weight", 0.0)),
                "detach_policy": cfg.get("detach_policy", "detach_teacher"),
                "enabled_splits": list(cfg.get("enabled_splits", ["train"])),
            }
        }


def teacher_guidance_config(cfg: dict[str, Any]) -> dict[str, Any]:
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg.get("loss"), dict) else {}
    raw = loss_cfg.get("teacher_guidance", {})
    if raw is True:
        raw = {"enabled": True}
    if not isinstance(raw, dict):
        raw = {}
    resolved = dict(raw)
    resolved.setdefault("enabled", False)
    resolved.setdefault("weight", 0.0)
    resolved.setdefault("temperature", 1.0)
    resolved.setdefault("detach_policy", "detach_teacher")
    resolved.setdefault("detach_teacher", True)
    resolved.setdefault("enabled_splits", ["train"])
    resolved.setdefault("allow_missing", False)
    return resolved


def _teacher_tensor_for_batch(
    batch_state: BatchState,
    state: dict[str, Any],
    *,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor | None, str]:
    for key, kind in (
        ("teacher_probabilities", "probabilities"),
        ("geometry_teacher_probabilities", "probabilities"),
        ("teacher_logits", "logits"),
        ("geometry_teacher_logits", "logits"),
    ):
        value = batch_state.batch.get(key)
        if torch.is_tensor(value):
            return value, kind
    tensor = state.get("tensor")
    if not torch.is_tensor(tensor):
        return None, ""
    indices = _teacher_indices(batch_state.batch, cfg=cfg)
    if indices is None:
        return tensor[: batch_state.primary_logits.shape[0]], str(state.get("tensor_kind", "logits") or "logits")
    return tensor.index_select(0, indices.to(dtype=torch.long, device=tensor.device)), str(state.get("tensor_kind", "logits") or "logits")


def _teacher_indices(batch: dict[str, Any], *, cfg: dict[str, Any]) -> torch.Tensor | None:
    for key in (cfg.get("index_key"), "teacher_index", "sample_index", "dataset_index"):
        if not key:
            continue
        value = batch.get(str(key))
        if torch.is_tensor(value):
            return value.reshape(-1).detach().cpu()
    return None


__all__ = ["TeacherGuidanceTrainingExtension", "teacher_guidance_config"]
