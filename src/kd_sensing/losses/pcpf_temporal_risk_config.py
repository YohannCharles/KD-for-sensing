from __future__ import annotations

import math
from typing import Any, Mapping

from kd_sensing.losses.beam_prototype_alignment import TOPOLOGY_IDS
from kd_sensing.models.pcpf_temporal_risk import TRAINING_STAGES


_FIELDS = frozenset(
    {
        "enabled",
        "lambda_fused_hard",
        "lambda_unimodal",
        "unimodal_hard_weight",
        "unimodal_soft_weight",
        "fused_soft_weight",
        "use_beam_prototype_alignment",
        "lambda_proto",
        "lambda_modality_proto",
        "beam_label_sigma",
        "prototype_topology",
        "risk_supervision_enabled",
        "lambda_risk",
        "lambda_rank",
        "rank_margin",
        "lambda_concentration",
        "stage3_topology_weight",
        "stage_preparation",
        "stage2_gate",
    }
)


def pcpf_temporal_risk_config(cfg: dict[str, Any]) -> dict[str, Any]:
    loss = cfg.get("loss", {})
    if not isinstance(loss, Mapping):
        raise ValueError("loss must be a mapping.")
    raw = loss.get("pcpf_temporal_risk", {})
    if not isinstance(raw, Mapping):
        raise ValueError("loss.pcpf_temporal_risk must be a mapping.")
    raw = dict(raw)
    unknown = sorted(set(raw) - _FIELDS)
    if unknown:
        raise ValueError(f"loss.pcpf_temporal_risk contains unsupported fields: {unknown}.")
    primary = cfg.get("model", {}).get("primary", {})
    if not isinstance(primary, Mapping):
        raise ValueError("model.primary must be a mapping.")
    stage = str(primary.get("training_stage", "")).strip().lower()
    if stage not in TRAINING_STAGES:
        raise ValueError(f"PCPF-T training_stage must be one of {list(TRAINING_STAGES)}.")
    topology = _prototype_topology(raw.get("prototype_topology"))
    result = {
        "enabled": bool(raw.get("enabled", False)),
        "training_stage": stage,
        "lambda_fused_hard": _finite(raw.get("lambda_fused_hard", 1.0), "lambda_fused_hard", non_negative=True),
        "lambda_unimodal": _finite(raw.get("lambda_unimodal", 1.0), "lambda_unimodal", non_negative=True),
        "unimodal_hard_weight": _finite(
            raw.get("unimodal_hard_weight", 1.0), "unimodal_hard_weight", non_negative=True
        ),
        "unimodal_soft_weight": _finite(
            raw.get("unimodal_soft_weight", 0.5), "unimodal_soft_weight", non_negative=True
        ),
        "fused_soft_weight": _finite(raw.get("fused_soft_weight", 0.0), "fused_soft_weight", non_negative=True),
        "use_beam_prototype_alignment": bool(raw.get("use_beam_prototype_alignment", True)),
        "lambda_proto": _finite(raw.get("lambda_proto", 0.2), "lambda_proto", non_negative=True),
        "lambda_modality_proto": _finite(
            raw.get("lambda_modality_proto", 0.1), "lambda_modality_proto", non_negative=True
        ),
        "beam_label_sigma": _finite(raw.get("beam_label_sigma", 2.0), "beam_label_sigma", positive=True),
        "prototype_topology": topology,
        "risk_supervision_enabled": bool(raw.get("risk_supervision_enabled", True)),
        "lambda_risk": _finite(raw.get("lambda_risk", 1.0), "lambda_risk", non_negative=True),
        "lambda_rank": _finite(raw.get("lambda_rank", 0.2), "lambda_rank", non_negative=True),
        "rank_margin": _finite(raw.get("rank_margin", 0.05), "rank_margin", non_negative=True),
        "lambda_concentration": _finite(
            raw.get("lambda_concentration", 0.5), "lambda_concentration", non_negative=True
        ),
        "stage3_topology_weight": _finite(
            raw.get("stage3_topology_weight", 0.0), "stage3_topology_weight", non_negative=True
        ),
        "stage_preparation": _stage_preparation(raw.get("stage_preparation")),
        "stage2_gate": _stage2_gate(raw.get("stage2_gate")),
    }
    if not result["enabled"]:
        raise ValueError("pcpf_temporal_risk_fusion requires loss.pcpf_temporal_risk.enabled=true.")
    return result


def _prototype_topology(value: Any) -> dict[str, Any]:
    if value is None:
        raw: dict[str, Any] = {"id": "cyclic_index_v1"}
    elif isinstance(value, str):
        raw = {"id": value}
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise ValueError("prototype_topology must be a string or mapping.")
    unknown = sorted(set(raw) - {"id", "permutation", "descriptor_sha256", "audit_path", "audit_sha256"})
    if unknown:
        raise ValueError(f"prototype_topology contains unsupported fields: {unknown}.")
    topology_id = str(raw.get("id", "")).strip().lower()
    if topology_id not in TOPOLOGY_IDS or topology_id == "linear_index_v1":
        raise ValueError("PCPF-T requires a supported circular prototype topology.")
    raw_permutation = raw.get("permutation")
    if topology_id == "permuted_index_v1":
        if not isinstance(raw_permutation, (list, tuple)):
            raise ValueError("permuted_index_v1 requires prototype_topology.permutation.")
        permutation = [int(value) for value in raw_permutation]
        if len(permutation) != 64 or set(permutation) != set(range(64)):
            raise ValueError("prototype_topology.permutation must be a 64-label bijection.")
    else:
        if raw_permutation not in (None, [], ()):
            raise ValueError(f"prototype topology {topology_id!r} does not accept a permutation.")
        permutation = None
    descriptor_sha256 = str(raw.get("descriptor_sha256", "")).strip().lower()
    audit_path = str(raw.get("audit_path", "")).strip()
    audit_sha256 = str(raw.get("audit_sha256", "")).strip().lower()
    if topology_id == "ula_dft_phase_cycle_v1":
        if len(descriptor_sha256) != 64 or any(char not in "0123456789abcdef" for char in descriptor_sha256):
            raise ValueError("ula_dft_phase_cycle_v1 requires a valid descriptor_sha256.")
        if not audit_path:
            raise ValueError("ula_dft_phase_cycle_v1 requires prototype_topology.audit_path.")
        if len(audit_sha256) != 64 or any(char not in "0123456789abcdef" for char in audit_sha256):
            raise ValueError("ula_dft_phase_cycle_v1 requires a valid audit_sha256.")
    elif descriptor_sha256 or audit_path or audit_sha256:
        raise ValueError(f"prototype topology {topology_id!r} does not accept physical audit fields.")
    return {
        "id": topology_id,
        "permutation": permutation,
        "descriptor_sha256": descriptor_sha256,
        "audit_path": audit_path,
        "audit_sha256": audit_sha256,
    }


def _stage_preparation(value: Any) -> dict[str, Any]:
    if value is None:
        raw: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise ValueError("stage_preparation must be a mapping.")
    unknown = sorted(set(raw) - {"enabled", "max_batches", "smoke_only"})
    if unknown:
        raise ValueError(f"stage_preparation contains unsupported fields: {unknown}.")
    max_batches = raw.get("max_batches")
    if max_batches is not None and (isinstance(max_batches, bool) or int(max_batches) <= 0):
        raise ValueError("stage_preparation.max_batches must be null or a positive integer.")
    smoke_only = bool(raw.get("smoke_only", False))
    if max_batches is not None and not smoke_only:
        raise ValueError("A bounded stage preparation pass is allowed only with smoke_only=true.")
    return {
        "enabled": bool(raw.get("enabled", True)),
        "max_batches": int(max_batches) if max_batches is not None else None,
        "smoke_only": smoke_only,
    }


def _stage2_gate(value: Any) -> dict[str, Any]:
    if value is None:
        raw: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise ValueError("stage2_gate must be a mapping.")
    allowed = {
        "overall_spearman_min",
        "minimum_positive_modalities",
        "require_each_weather_positive",
        "upper_lower_gap_min",
        "minimum_risk_std",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"stage2_gate contains unsupported fields: {unknown}.")
    minimum_modalities = int(raw.get("minimum_positive_modalities", 3))
    if not 0 <= minimum_modalities <= 5:
        raise ValueError("stage2_gate.minimum_positive_modalities must be in [0,5].")
    return {
        "overall_spearman_min": _finite(
            raw.get("overall_spearman_min", 0.20), "stage2_gate.overall_spearman_min"
        ),
        "minimum_positive_modalities": minimum_modalities,
        "require_each_weather_positive": bool(raw.get("require_each_weather_positive", True)),
        "upper_lower_gap_min": _finite(
            raw.get("upper_lower_gap_min", 0.0), "stage2_gate.upper_lower_gap_min", non_negative=True
        ),
        "minimum_risk_std": _finite(
            raw.get("minimum_risk_std", 1e-4), "stage2_gate.minimum_risk_std", positive=True
        ),
    }


def _finite(value: Any, field: str, *, positive: bool = False, non_negative: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite.")
    if positive and result <= 0:
        raise ValueError(f"{field} must be positive.")
    if non_negative and result < 0:
        raise ValueError(f"{field} must be non-negative.")
    return result


__all__ = ["pcpf_temporal_risk_config"]
