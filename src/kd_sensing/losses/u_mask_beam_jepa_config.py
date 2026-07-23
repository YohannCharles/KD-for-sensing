import math
from typing import Any

from kd_sensing.losses.beam_prototype_alignment import TOPOLOGY_IDS


_FIELDS = frozenset(
    {
        "enabled",
        "router_supervision",
        "use_beam_prototype_alignment",
        "lambda_proto",
        "lambda_modality_proto",
        "beam_label_sigma",
        "prototype_target_circular",
        "prototype_topology",
        "circular_beam_distance",
        "router_oracle_weight",
        "router_oracle_target_mode",
        "router_oracle_temperature",
        "missing_mask",
        "superset_consistency",
    }
)


def u_mask_beam_jepa_config(cfg: dict[str, Any]) -> dict[str, Any]:
    retired = [name for name in ("bcacl", "cmsbl") if name in cfg]
    if retired:
        raise ValueError(f"Clean U0 does not support retired training sections: {retired}.")
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg.get("loss"), dict) else {}
    raw = loss_cfg.get("u_mask_beam_jepa", {})
    if not isinstance(raw, dict):
        raise ValueError("loss.u_mask_beam_jepa must be a mapping.")
    unknown = sorted(set(raw) - _FIELDS)
    if unknown:
        raise ValueError(f"U-Mask current surface does not support fields: {unknown}.")
    model = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    primary = model.get("primary", {}) if isinstance(model.get("primary"), dict) else {}
    head_type = str(primary.get("head_type", "prototype")).strip().lower()
    if head_type not in {"prototype", "classifier"}:
        raise ValueError("Clean U0 head_type must be prototype or classifier.")
    fusion_type = str(primary.get("fusion_type", "supervised_router")).strip().lower()
    if fusion_type != "supervised_router":
        raise ValueError("Clean U0 supports fusion_type=supervised_router only.")

    use_bpa = bool(raw.get("use_beam_prototype_alignment", False)) and head_type == "prototype"
    lambda_proto = _finite(raw.get("lambda_proto", 0.0), "lambda_proto", non_negative=True) if use_bpa else 0.0
    lambda_modality_proto = (
        _finite(raw.get("lambda_modality_proto", 0.0), "lambda_modality_proto", non_negative=True)
        if use_bpa
        else 0.0
    )
    topology = _resolve_prototype_topology(raw, use_bpa=use_bpa)
    router_weight = _finite(raw.get("router_oracle_weight", 0.1), "router_oracle_weight", non_negative=True)
    router_supervision = str(raw.get("router_supervision", "oracle")).strip().lower()
    if router_supervision != "oracle":
        raise ValueError("Clean U0 supports router_supervision=oracle only.")
    target_mode = str(raw.get("router_oracle_target_mode", "hard_first")).strip().lower()
    if target_mode != "hard_first":
        raise ValueError("Clean U0 supports router_oracle_target_mode=hard_first only.")
    return {
        "enabled": bool(raw.get("enabled", False)),
        "head_type": head_type,
        "fusion_type": fusion_type,
        "use_beam_prototype_alignment": use_bpa,
        "lambda_proto": lambda_proto,
        "lambda_modality_proto": lambda_modality_proto,
        "beam_proto_temperature": float(primary.get("beam_proto_temperature", 0.2)),
        "beam_label_sigma": _finite(raw.get("beam_label_sigma", 1.0), "beam_label_sigma", positive=True),
        "prototype_target_circular": bool(topology["circular"]),
        "prototype_topology": topology,
        "superset_consistency": _resolve_superset(raw.get("superset_consistency")),
        "missing_mask": _resolve_missing_mask(raw.get("missing_mask")),
        "router_supervision": router_supervision,
        "router_oracle_weight": router_weight,
        "router_oracle_target_mode": target_mode,
        "router_oracle_temperature": _finite(
            raw.get("router_oracle_temperature", 1.0), "router_oracle_temperature", positive=True
        ),
        "circular_beam_distance": bool(raw.get("circular_beam_distance", True)),
    }


def _resolve_missing_mask(value: Any) -> dict[str, Any]:
    if value is None:
        raw: dict[str, Any] = {}
    elif isinstance(value, dict):
        raw = dict(value)
    else:
        raise ValueError("loss.u_mask_beam_jepa.missing_mask must be a mapping.")
    mode = str(raw.get("mode", "random")).strip().lower()
    if mode not in {"random", "external"}:
        raise ValueError("loss.u_mask_beam_jepa.missing_mask.mode must be random or external.")
    if mode == "external":
        random_fields = {"p_missing", "always_available_indices", "ensure_at_least_one"} & raw.keys()
        if random_fields:
            raise ValueError(f"external missing-mask mode must not declare random fields: {sorted(random_fields)}.")
        return {"mode": mode}
    return {
        **raw,
        "mode": mode,
        "p_missing": raw.get("p_missing", 0.25),
        "ensure_at_least_one": bool(raw.get("ensure_at_least_one", True)),
    }


def _resolve_superset(value: Any) -> dict[str, Any]:
    if value is None:
        raw: dict[str, Any] = {}
    elif isinstance(value, dict):
        raw = dict(value)
    else:
        raise ValueError("superset_consistency must be a mapping.")
    unknown = sorted(set(raw) - {"enabled", "confidence_gated_kl", "kl_weight", "temperature"})
    if unknown:
        raise ValueError(f"Unsupported superset_consistency fields: {unknown}.")
    return {
        "enabled": bool(raw.get("enabled", False)),
        "confidence_gated_kl": bool(raw.get("confidence_gated_kl", False)),
        "kl_weight": _finite(raw.get("kl_weight", 0.2), "superset_consistency.kl_weight", non_negative=True),
        "temperature": _finite(raw.get("temperature", 2.0), "superset_consistency.temperature", positive=True),
    }


def _resolve_prototype_topology(raw: dict[str, Any], *, use_bpa: bool) -> dict[str, Any]:
    if not use_bpa:
        return {"id": "not_applicable", "circular": False, "permutation": None, "descriptor_sha256": "", "audit_path": ""}
    legacy_circular = bool(raw.get("prototype_target_circular", True))
    value = raw.get("prototype_topology")
    if value is None:
        payload: dict[str, Any] = {"id": "cyclic_index_v1" if legacy_circular else "linear_index_v1"}
    elif isinstance(value, str):
        payload = {"id": value}
    elif isinstance(value, dict):
        payload = dict(value)
    else:
        raise ValueError("prototype_topology must be a mapping, string, or omitted legacy selector.")
    topology_id = str(payload.get("id", "")).strip().lower()
    if topology_id not in TOPOLOGY_IDS:
        raise ValueError(f"Unsupported prototype topology {topology_id!r}.")
    circular = topology_id != "linear_index_v1"
    if "prototype_target_circular" in raw and legacy_circular != circular:
        raise ValueError("prototype_target_circular conflicts with prototype_topology.id.")
    raw_permutation = payload.get("permutation")
    if topology_id == "permuted_index_v1":
        if not isinstance(raw_permutation, (list, tuple)):
            raise ValueError("permuted_index_v1 requires prototype_topology.permutation.")
        permutation = [int(item) for item in raw_permutation]
        if len(permutation) != 64 or set(permutation) != set(range(64)):
            raise ValueError("prototype topology permutation must be a 64-label bijection.")
    else:
        if raw_permutation not in (None, [], ()):
            raise ValueError(f"prototype topology {topology_id!r} must not declare a permutation.")
        permutation = None
    descriptor_sha256 = str(payload.get("descriptor_sha256", ""))
    audit_path = str(payload.get("audit_path", ""))
    if topology_id == "ula_dft_phase_cycle_v1":
        if len(descriptor_sha256) != 64 or any(char not in "0123456789abcdef" for char in descriptor_sha256.lower()):
            raise ValueError("ula_dft_phase_cycle_v1 requires a 64-character topology descriptor_sha256.")
        if not audit_path:
            raise ValueError("ula_dft_phase_cycle_v1 requires prototype_topology.audit_path.")
    elif descriptor_sha256 or audit_path:
        raise ValueError(f"prototype topology {topology_id!r} must not declare physical audit provenance.")
    return {
        "id": topology_id,
        "circular": circular,
        "permutation": permutation,
        "descriptor_sha256": descriptor_sha256,
        "audit_path": audit_path,
    }


def _finite(
    value: Any,
    field: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite.")
    if positive and result <= 0.0:
        raise ValueError(f"{field} must be positive.")
    if non_negative and result < 0.0:
        raise ValueError(f"{field} must be non-negative.")
    return result


__all__ = ["u_mask_beam_jepa_config"]
