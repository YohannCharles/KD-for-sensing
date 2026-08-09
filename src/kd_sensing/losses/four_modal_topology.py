from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.engine.training_extensions import BaseLossResult, BatchState, ExtensionContext, TrainingExtension
from kd_sensing.losses.beam_prototype_alignment import (
    BeamPrototypeBank,
    TOPOLOGY_IDS,
    make_soft_beam_labels,
    prototype_alignment_loss,
)


_FIELDS = frozenset(
    {
        "enabled",
        "lambda_fused_hard",
        "lambda_unimodal",
        "unimodal_hard_weight",
        "unimodal_soft_weight",
        "use_beam_prototype_alignment",
        "lambda_proto",
        "lambda_modality_proto",
        "beam_label_sigma",
        "prototype_topology",
    }
)


def four_modal_topology_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    loss = cfg.get("loss", {})
    if not isinstance(loss, Mapping):
        raise ValueError("loss must be a mapping.")
    raw = loss.get("four_modal_topology", {})
    if not isinstance(raw, Mapping):
        raise ValueError("loss.four_modal_topology must be a mapping.")
    unknown = sorted(set(raw) - _FIELDS)
    if unknown:
        raise ValueError(f"loss.four_modal_topology contains unsupported fields: {unknown}.")
    result = {
        "enabled": bool(raw.get("enabled", False)),
        "lambda_fused_hard": _finite(raw.get("lambda_fused_hard", 1.0), "lambda_fused_hard"),
        "lambda_unimodal": _finite(raw.get("lambda_unimodal", 1.0), "lambda_unimodal"),
        "unimodal_hard_weight": _finite(raw.get("unimodal_hard_weight", 1.0), "unimodal_hard_weight"),
        "unimodal_soft_weight": _finite(raw.get("unimodal_soft_weight", 0.5), "unimodal_soft_weight"),
        "use_beam_prototype_alignment": bool(raw.get("use_beam_prototype_alignment", True)),
        "lambda_proto": _finite(raw.get("lambda_proto", 0.2), "lambda_proto"),
        "lambda_modality_proto": _finite(raw.get("lambda_modality_proto", 0.1), "lambda_modality_proto"),
        "beam_label_sigma": _finite(raw.get("beam_label_sigma", 2.0), "beam_label_sigma", positive=True),
        "prototype_topology": _prototype_topology(raw.get("prototype_topology")),
    }
    if not result["enabled"]:
        raise ValueError("four_modal_topology_predictor requires loss.four_modal_topology.enabled=true.")
    return result


def four_modal_topology_loss(
    output: Mapping[str, Any],
    labels: torch.Tensor,
    *,
    prototype_bank: BeamPrototypeBank,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    logits = _prediction_logits(output["logits"])
    hard = labels.to(device=logits.device, dtype=torch.long).reshape(logits.shape[0], -1)[:, 0]
    valid = hard.ne(-100)
    if not bool(valid.any().item()):
        raise ValueError("Topology predictor loss has no valid labels.")
    available = torch.as_tensor(output["available_modalities"], device=logits.device, dtype=torch.bool)
    unimodal_logits = torch.as_tensor(output["unimodal_logits"])
    if tuple(available.shape) != tuple(unimodal_logits.shape[:2]) or unimodal_logits.ndim != 3:
        raise ValueError("unimodal_logits and available_modalities must have shapes [B,4,64] and [B,4].")
    fused_hard = F.cross_entropy(logits, hard, ignore_index=-100)
    safe_hard = hard.masked_fill(~valid, 0)
    per_modality_hard = F.cross_entropy(
        unimodal_logits.reshape(-1, unimodal_logits.shape[-1]),
        safe_hard.unsqueeze(1).expand(-1, unimodal_logits.shape[1]).reshape(-1),
        reduction="none",
    ).reshape_as(available)
    loss_mask = available & valid.unsqueeze(1)
    denominator = loss_mask.sum(dim=1).clamp_min(1).to(torch.float32)
    unimodal_hard = ((per_modality_hard * loss_mask).sum(dim=1) / denominator)[valid].mean()

    topology = config["prototype_topology"]
    soft_target = make_soft_beam_labels(
        safe_hard,
        int(unimodal_logits.shape[-1]),
        float(config["beam_label_sigma"]),
        circular=True,
        topology_id=topology["id"],
        topology_permutation=topology["permutation"],
    ).to(device=logits.device, dtype=torch.float32)
    per_modality_soft = -(
        soft_target.unsqueeze(1) * F.log_softmax(unimodal_logits.float(), dim=-1)
    ).sum(dim=-1)
    unimodal_soft = ((per_modality_soft * loss_mask).sum(dim=1) / denominator)[valid].mean()
    unimodal = (
        float(config["unimodal_hard_weight"]) * unimodal_hard
        + float(config["unimodal_soft_weight"]) * unimodal_soft
    )

    prototype = logits.sum() * 0.0
    prototype_diagnostics: dict[str, float] = {}
    if bool(config["use_beam_prototype_alignment"]):
        prototype, prototype_diagnostics = prototype_alignment_loss(
            prototype_bank,
            hard[valid],
            fused_features=torch.as_tensor(output["output_features"])[valid],
            modality_features=torch.as_tensor(output["modality_features"])[valid],
            mask=available[valid],
            beam_label_sigma=float(config["beam_label_sigma"]),
            circular=True,
            topology_id=topology["id"],
            topology_permutation=topology["permutation"],
            lambda_proto=float(config["lambda_proto"]),
            lambda_modality_proto=float(config["lambda_modality_proto"]),
        )
    total = (
        float(config["lambda_fused_hard"]) * fused_hard
        + float(config["lambda_unimodal"]) * unimodal
        + prototype
    )
    diagnostics = {
        "loss/four_modal_topology_total": _scalar(total),
        "loss/fused_hard": _scalar(fused_hard),
        "loss/unimodal": _scalar(unimodal),
        "loss/unimodal_hard": _scalar(unimodal_hard),
        "loss/unimodal_soft": _scalar(unimodal_soft),
        **prototype_diagnostics,
    }
    return {"loss": total, "task_loss": fused_hard, "diagnostics": diagnostics}


class FourModalTopologyTrainingExtension(TrainingExtension):
    name = "four_modal_topology"
    state_schema_version = 1

    def setup(self, context: ExtensionContext) -> dict[str, Any]:
        config = four_modal_topology_config(context.cfg)
        model_topology = context.primary_model.prototype_topology_metadata()
        expected = config["prototype_topology"]
        for key in ("id", "descriptor_sha256", "audit_sha256"):
            if str(model_topology.get(key, "")) != str(expected.get(key, "")):
                raise ValueError("Model and loss prototype topology provenance do not match.")
        return {"config": config}

    def state_dict(self, state: Any) -> dict[str, Any]:
        del state
        return {}

    def load_state_dict(self, state: Any, payload: Mapping[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("four_modal_topology extension state must be a mapping.")
        if payload:
            raise ValueError("four_modal_topology extension has no mutable resume state.")

    def compute_base_loss(
        self,
        context: ExtensionContext,
        state: Any,
        batch_state: BatchState,
    ) -> BaseLossResult | None:
        output = {
            "logits": batch_state.primary_logits,
            "input_features": batch_state.primary_output.input_features,
            "output_features": batch_state.primary_output.output_features,
            **batch_state.primary_output.diagnostics,
        }
        result = four_modal_topology_loss(
            output,
            batch_state.labels,
            prototype_bank=context.primary_model.prototype_bank,
            config=state["config"],
        )
        return BaseLossResult(
            total_loss=result["loss"],
            task_loss=result["task_loss"],
            auxiliary_loss=result["loss"] - result["task_loss"],
            diagnostics=dict(result["diagnostics"]),
        )


def _prototype_topology(value: Any) -> dict[str, Any]:
    raw = {"id": "cyclic_index_v1"} if value is None else {"id": value} if isinstance(value, str) else dict(value)
    unknown = sorted(set(raw) - {"id", "permutation", "descriptor_sha256", "audit_path", "audit_sha256"})
    if unknown:
        raise ValueError(f"prototype_topology contains unsupported fields: {unknown}.")
    topology_id = str(raw.get("id", "")).strip().lower()
    if topology_id not in TOPOLOGY_IDS or topology_id == "linear_index_v1":
        raise ValueError("four_modal_topology requires a supported circular topology.")
    raw_permutation = raw.get("permutation")
    if topology_id == "permuted_index_v1":
        permutation = [int(item) for item in raw_permutation] if isinstance(raw_permutation, (list, tuple)) else []
        if len(permutation) != 64 or set(permutation) != set(range(64)):
            raise ValueError("permuted_index_v1 requires a 64-label bijection.")
    else:
        if raw_permutation not in (None, [], ()):
            raise ValueError(f"Topology {topology_id!r} does not accept a permutation.")
        permutation = None
    descriptor = str(raw.get("descriptor_sha256", "")).strip().lower()
    audit_path = str(raw.get("audit_path", "")).strip()
    audit = str(raw.get("audit_sha256", "")).strip().lower()
    if topology_id == "ula_dft_phase_cycle_v1":
        if not _is_sha256(descriptor) or not audit_path or not _is_sha256(audit):
            raise ValueError("ULA-DFT topology requires descriptor/audit SHA256 and audit path.")
    elif descriptor or audit_path or audit:
        raise ValueError(f"Topology {topology_id!r} does not accept physical provenance.")
    return {
        "id": topology_id,
        "permutation": permutation,
        "descriptor_sha256": descriptor,
        "audit_path": audit_path,
        "audit_sha256": audit,
    }


def _prediction_logits(value: Any) -> torch.Tensor:
    logits = torch.as_tensor(value)
    if logits.ndim == 3:
        logits = logits[:, -1]
    if logits.ndim != 2:
        raise ValueError("Prediction logits must have shape [B,C] or [B,T,C].")
    return logits


def _finite(value: Any, field: str, *, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        comparator = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be finite and {comparator}.")
    return result


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


__all__ = [
    "FourModalTopologyTrainingExtension",
    "four_modal_topology_config",
    "four_modal_topology_loss",
]
