from typing import Any


def u_mask_beam_jepa_config(cfg: dict[str, Any]) -> dict[str, Any]:
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg.get("loss"), dict) else {}
    raw = loss_cfg.get("u_mask_beam_jepa", {})
    if not isinstance(raw, dict):
        raise ValueError("loss.u_mask_beam_jepa must be a mapping.")
    model = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    primary = model.get("primary", {}) if isinstance(model.get("primary"), dict) else {}

    head_type = str(primary.get("head_type", "prototype")).strip().lower()
    if head_type not in {"prototype", "classifier"}:
        raise ValueError("T2 head_type must be prototype or classifier.")
    fusion_type = str(primary.get("fusion_type", "supervised_router")).strip().lower()
    if fusion_type not in {"supervised_router", "reliability_mean"}:
        raise ValueError("T2 fusion_type must be supervised_router or reliability_mean.")
    use_bpa = bool(raw.get("use_beam_prototype_alignment", False))
    lambda_proto = float(raw.get("lambda_proto", 0.0))
    lambda_modality_proto = float(raw.get("lambda_modality_proto", 0.0))
    use_cma = bool(raw.get("use_amber_cma_analogue", False))
    if head_type == "classifier":
        use_bpa = False
        lambda_proto = 0.0
        lambda_modality_proto = 0.0
    if not use_bpa:
        lambda_proto = 0.0
        lambda_modality_proto = 0.0
    if use_bpa and use_cma:
        raise ValueError("BPA and the AMBER CMA analogue are mutually exclusive.")

    prototype_target_circular = bool(raw.get("prototype_target_circular", True))
    router_circular = bool(raw.get("circular_beam_distance", True))
    router_supervision = str(raw.get("router_supervision", "oracle")).strip().lower()
    if router_supervision != "oracle":
        raise ValueError("T2 supervised_router requires router_supervision='oracle'.")
    router_oracle_weight = float(raw.get("router_oracle_weight", 0.1))
    if fusion_type == "reliability_mean" and router_oracle_weight != 0.0:
        raise ValueError("reliability_mean fusion requires router_oracle_weight=0.")
    superset = _resolve_superset(raw.get("superset_consistency"))
    missing_mask = raw.get("missing_mask", {"p_missing": 0.25, "ensure_at_least_one": True})
    if not isinstance(missing_mask, dict):
        raise ValueError("loss.u_mask_beam_jepa.missing_mask must be a mapping.")

    return {
        "enabled": bool(raw.get("enabled", False)),
        "head_type": head_type,
        "fusion_type": fusion_type,
        "use_beam_prototype_alignment": use_bpa,
        "lambda_proto": lambda_proto,
        "lambda_modality_proto": lambda_modality_proto,
        "beam_proto_temperature": float(primary.get("beam_proto_temperature", 0.2)),
        "beam_label_sigma": float(raw.get("beam_label_sigma", 1.0)),
        "prototype_target_circular": prototype_target_circular,
        "use_amber_cma_analogue": use_cma,
        "lambda_amber_cma": float(raw.get("lambda_amber_cma", 0.2)),
        "amber_cma_temperature": float(raw.get("amber_cma_temperature", 0.2)),
        "superset_consistency": superset,
        "missing_mask": dict(missing_mask),
        "router_supervision": router_supervision,
        "router_oracle_weight": router_oracle_weight,
        "circular_beam_distance": router_circular,
    }


def _resolve_superset(value: Any) -> dict[str, Any]:
    if value is None:
        raw: dict[str, Any] = {}
    elif isinstance(value, dict):
        raw = dict(value)
    else:
        raise ValueError("superset_consistency must be a mapping.")
    if bool(raw.get("beam_monotonic_rank", False)):
        raise ValueError("beam_monotonic_rank has been retired; T2 keeps only superset KL.")
    if float(raw.get("feature_l2_weight", 0.0)) != 0.0:
        raise ValueError("superset feature L2 has been retired; T2 keeps only superset KL.")
    temperature = float(raw.get("temperature", 2.0))
    weight = float(raw.get("kl_weight", 0.2))
    if temperature <= 0.0 or weight < 0.0:
        raise ValueError("superset KL temperature must be positive and its weight non-negative.")
    return {
        "enabled": bool(raw.get("enabled", False)),
        "confidence_gated_kl": bool(raw.get("confidence_gated_kl", False)),
        "kl_weight": weight,
        "temperature": temperature,
    }
