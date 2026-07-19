import math
from typing import Any

from kd_sensing.losses.beam_prototype_alignment import TOPOLOGY_IDS


ROUTER_ORACLE_TARGET_MODES = frozenset(
    {
        "hard_first",
        "hard_confidence_tie",
        "soft_uniform_tie",
        "soft_confidence_tie",
        "distance_soft",
        "distance_confidence_soft",
        "beam_power_soft",
        "beam_power_expected_soft",
    }
)

ROUTER_VARIANTS = frozenset({"current", "patr", "h2r", "core", "unified_hpr"})
DYNAMIC_ROUTER_SUPERVISION_MODES = frozenset({"label_topology", "beam_power"})
_DYNAMIC_ROUTER_FIELDS = frozenset(
    {
        "supervision",
        "utility_temperature",
        "quality_regression_weight",
        "fused_utility_weight",
        "frame_rank_weight",
        "residual_anchor_weight",
        "paired_joint",
    }
)
_PAIRED_JOINT_FIELDS = frozenset(
    {
        "enabled",
        "panel_path",
        "panel_sha256",
        "corruption_seed",
        "monotonic_weight",
        "monotonic_margin_scale",
        "quality_drop_epsilon",
    }
)
_ROUTER_COMPONENT_OVERRIDE_FIELDS = frozenset(
    {
        "router_use_prior_anchor",
        "router_use_window_temporal_evidence",
        "router_use_hierarchical_cell_gate",
        "router_use_consensus_evidence",
    }
)
_ROUTER_VARIANT_COMPONENTS = {
    "current": {
        "prior_anchored_residual": False,
        "window_temporal_evidence": False,
        "hierarchical_cell_gate": False,
        "consensus_evidence": False,
    },
    "patr": {
        "prior_anchored_residual": True,
        "window_temporal_evidence": True,
        "hierarchical_cell_gate": False,
        "consensus_evidence": False,
    },
    "h2r": {
        "prior_anchored_residual": True,
        "window_temporal_evidence": False,
        "hierarchical_cell_gate": True,
        "consensus_evidence": False,
    },
    "core": {
        "prior_anchored_residual": True,
        "window_temporal_evidence": False,
        "hierarchical_cell_gate": False,
        "consensus_evidence": True,
    },
    "unified_hpr": {
        "prior_anchored_residual": True,
        "window_temporal_evidence": True,
        "hierarchical_cell_gate": True,
        "consensus_evidence": True,
    },
}


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
    if fusion_type not in {"supervised_router", "reliability_mean", "uniform_mean"}:
        raise ValueError("T2 fusion_type must be supervised_router, reliability_mean, or uniform_mean.")
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

    prototype_topology = _resolve_prototype_topology(raw, use_bpa=use_bpa)
    prototype_target_circular = bool(prototype_topology["circular"])
    router_circular = bool(raw.get("circular_beam_distance", True))
    router_supervision = str(raw.get("router_supervision", "oracle")).strip().lower()
    if router_supervision != "oracle":
        raise ValueError("T2 supervised_router requires router_supervision='oracle'.")
    router_oracle_weight = float(raw.get("router_oracle_weight", 0.1))
    router_oracle_target_mode = str(raw.get("router_oracle_target_mode", "hard_first")).strip().lower()
    if router_oracle_target_mode not in ROUTER_ORACLE_TARGET_MODES:
        raise ValueError(
            "router_oracle_target_mode must be one of "
            f"{sorted(ROUTER_ORACLE_TARGET_MODES)}."
        )
    router_oracle_temperature = float(raw.get("router_oracle_temperature", 1.0))
    if router_oracle_temperature <= 0.0:
        raise ValueError("router_oracle_temperature must be positive.")
    router_oracle_beam_temperature = float(raw.get("router_oracle_beam_temperature", 1.0))
    if router_oracle_beam_temperature <= 0.0:
        raise ValueError("router_oracle_beam_temperature must be positive.")
    if fusion_type in {"reliability_mean", "uniform_mean"} and router_oracle_weight != 0.0:
        raise ValueError(f"{fusion_type} fusion requires router_oracle_weight=0.")
    superset = _resolve_superset(raw.get("superset_consistency"))
    missing_mask = _resolve_missing_mask(raw.get("missing_mask"))
    quality_pairing = _resolve_router_quality_pairing(raw.get("router_quality_pairing"))
    dataset = cfg.get("data", {}).get("dataset", {})
    raw_utility_targets = dataset.get("include_router_utility_targets", False) if isinstance(dataset, dict) else False
    utility_targets = bool(raw_utility_targets)
    if (router_oracle_target_mode.startswith("beam_power_") or quality_pairing["enabled"]) and not utility_targets:
        raise ValueError(
            "Beam-power Router supervision requires data.dataset.include_router_utility_targets=true."
        )
    router_variant, dynamic_router = resolve_dynamic_router_config(
        cfg,
        raw=raw,
        primary=primary,
        fusion_type=fusion_type,
        head_type=head_type,
        use_bpa=use_bpa,
        router_oracle_weight=router_oracle_weight,
        router_oracle_target_mode=router_oracle_target_mode,
        router_quality_pairing=quality_pairing,
        include_router_utility_targets=raw_utility_targets,
    )

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
        "prototype_topology": prototype_topology,
        "use_amber_cma_analogue": use_cma,
        "lambda_amber_cma": float(raw.get("lambda_amber_cma", 0.2)),
        "amber_cma_temperature": float(raw.get("amber_cma_temperature", 0.2)),
        "superset_consistency": superset,
        "missing_mask": missing_mask,
        "router_supervision": router_supervision,
        "router_oracle_weight": router_oracle_weight,
        "router_oracle_target_mode": router_oracle_target_mode,
        "router_oracle_temperature": router_oracle_temperature,
        "router_oracle_beam_temperature": router_oracle_beam_temperature,
        "circular_beam_distance": router_circular,
        "router_quality_pairing": quality_pairing,
        "router_variant": router_variant,
        "dynamic_router": dynamic_router,
    }


def resolve_dynamic_router_config(
    cfg: dict[str, Any],
    *,
    raw: dict[str, Any] | None = None,
    primary: dict[str, Any] | None = None,
    fusion_type: str | None = None,
    head_type: str | None = None,
    use_bpa: bool | None = None,
    router_oracle_weight: float | None = None,
    router_oracle_target_mode: str | None = None,
    router_quality_pairing: dict[str, Any] | None = None,
    include_router_utility_targets: Any = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve the opt-in inner-only dynamic Router contract."""

    loss_cfg = cfg.get("loss", {}) if isinstance(cfg.get("loss"), dict) else {}
    raw_loss = loss_cfg.get("u_mask_beam_jepa", {})
    if raw is None:
        raw = raw_loss if isinstance(raw_loss, dict) else {}
    model = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    if primary is None:
        value = model.get("primary", {})
        primary = value if isinstance(value, dict) else {}

    variant = str(primary.get("router_variant", "current")).strip().lower()
    if variant not in ROUTER_VARIANTS:
        raise ValueError(f"model.primary.router_variant must be one of {sorted(ROUTER_VARIANTS)}.")
    forbidden = sorted(_ROUTER_COMPONENT_OVERRIDE_FIELDS & primary.keys())
    if forbidden:
        raise ValueError(
            "Router components are selected only by model.primary.router_variant; "
            f"remove independent component fields {forbidden}."
        )

    dynamic_present = "dynamic_router" in raw
    dynamic_raw = raw.get("dynamic_router")
    if variant == "current":
        if dynamic_present:
            raise ValueError("router_variant=current must not declare loss.u_mask_beam_jepa.dynamic_router.")
        return variant, _disabled_dynamic_router_config()

    if not isinstance(dynamic_raw, dict):
        raise ValueError("Candidate router variants require loss.u_mask_beam_jepa.dynamic_router mapping.")
    _reject_unknown_fields(dynamic_raw, _DYNAMIC_ROUTER_FIELDS, "loss.u_mask_beam_jepa.dynamic_router")

    resolved_fusion = str(fusion_type or primary.get("fusion_type", "supervised_router")).strip().lower()
    resolved_head = str(head_type or primary.get("head_type", "prototype")).strip().lower()
    resolved_bpa = bool(raw.get("use_beam_prototype_alignment", False) if use_bpa is None else use_bpa)
    if resolved_fusion != "supervised_router":
        raise ValueError("Candidate router variants require model.primary.fusion_type=supervised_router.")
    if resolved_head != "prototype":
        raise ValueError("Candidate router variants require model.primary.head_type=prototype.")
    if not resolved_bpa:
        raise ValueError("Candidate router variants require use_beam_prototype_alignment=true.")

    oracle_weight = float(raw.get("router_oracle_weight", 0.1) if router_oracle_weight is None else router_oracle_weight)
    oracle_mode = str(
        raw.get("router_oracle_target_mode", "hard_first")
        if router_oracle_target_mode is None
        else router_oracle_target_mode
    ).strip().lower()
    pairing = (
        _resolve_router_quality_pairing(raw.get("router_quality_pairing"))
        if router_quality_pairing is None
        else router_quality_pairing
    )
    if oracle_weight != 0.0:
        raise ValueError("Candidate router variants require router_oracle_weight=0 to avoid duplicate supervision.")
    if oracle_mode.startswith("beam_power_"):
        raise ValueError("Candidate router variants must not reuse beam-power router_oracle_target_mode.")
    if bool(pairing.get("enabled", False)):
        raise ValueError("Candidate router variants must not enable legacy router_quality_pairing.")

    supervision = str(dynamic_raw.get("supervision", "")).strip().lower()
    if supervision not in DYNAMIC_ROUTER_SUPERVISION_MODES:
        raise ValueError(
            "loss.u_mask_beam_jepa.dynamic_router.supervision must be label_topology or beam_power."
        )
    dataset = cfg.get("data", {}).get("dataset", {})
    raw_utility_targets = (
        dataset.get("include_router_utility_targets", False)
        if include_router_utility_targets is None and isinstance(dataset, dict)
        else include_router_utility_targets
    )
    if type(raw_utility_targets) is not bool:
        raise ValueError("data.dataset.include_router_utility_targets must be boolean for candidate Routers.")
    utility_targets = raw_utility_targets
    if supervision == "beam_power" and not utility_targets:
        raise ValueError("beam_power dynamic Router supervision requires include_router_utility_targets=true.")
    if supervision == "label_topology" and utility_targets:
        raise ValueError("label_topology dynamic Router supervision requires include_router_utility_targets=false.")

    utility_temperature = _finite_float(
        dynamic_raw.get("utility_temperature", 1.0),
        "dynamic_router.utility_temperature",
        positive=True,
    )
    quality_weight = _finite_float(
        dynamic_raw.get("quality_regression_weight", 0.1),
        "dynamic_router.quality_regression_weight",
        non_negative=True,
    )
    fused_weight = _finite_float(
        dynamic_raw.get("fused_utility_weight", 0.1),
        "dynamic_router.fused_utility_weight",
        non_negative=True,
    )
    frame_rank_weight = _finite_float(
        dynamic_raw.get(
            "frame_rank_weight",
            0.1 if _ROUTER_VARIANT_COMPONENTS[variant]["hierarchical_cell_gate"] else 0.0,
        ),
        "dynamic_router.frame_rank_weight",
        non_negative=True,
    )
    residual_anchor_weight = _finite_float(
        dynamic_raw.get("residual_anchor_weight", 0.01),
        "dynamic_router.residual_anchor_weight",
        non_negative=True,
    )
    if quality_weight == 0.0 and fused_weight == 0.0:
        raise ValueError("Candidate dynamic Router requires a positive quality or fused utility weight.")
    hierarchical = _ROUTER_VARIANT_COMPONENTS[variant]["hierarchical_cell_gate"]
    if hierarchical and frame_rank_weight == 0.0:
        raise ValueError("H2R and Unified-HPR require a positive dynamic_router.frame_rank_weight.")
    if not hierarchical and frame_rank_weight != 0.0:
        raise ValueError("dynamic_router.frame_rank_weight is only valid for H2R and Unified-HPR.")
    paired_joint = _resolve_paired_joint(dynamic_raw.get("paired_joint"))

    return variant, {
        "enabled": True,
        "variant": variant,
        "components": dict(_ROUTER_VARIANT_COMPONENTS[variant]),
        "supervision": supervision,
        "requires_beam_power": supervision == "beam_power",
        "utility_temperature": utility_temperature,
        "quality_regression_weight": quality_weight,
        "fused_utility_weight": fused_weight,
        "frame_rank_weight": frame_rank_weight,
        "residual_anchor_weight": residual_anchor_weight,
        "paired_joint": paired_joint,
    }


def _disabled_dynamic_router_config() -> dict[str, Any]:
    return {
        "enabled": False,
        "variant": "current",
        "components": dict(_ROUTER_VARIANT_COMPONENTS["current"]),
        "supervision": None,
        "requires_beam_power": False,
        "utility_temperature": 1.0,
        "quality_regression_weight": 0.0,
        "fused_utility_weight": 0.0,
        "frame_rank_weight": 0.0,
        "residual_anchor_weight": 0.0,
        "paired_joint": {
            "enabled": False,
            "panel_path": "",
            "panel_sha256": "",
            "corruption_seed": 20260719,
            "monotonic_weight": 0.0,
            "monotonic_margin_scale": 0.0,
            "quality_drop_epsilon": 0.0,
        },
    }


def _resolve_paired_joint(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Candidate dynamic Router requires dynamic_router.paired_joint mapping.")
    raw = dict(value)
    _reject_unknown_fields(raw, _PAIRED_JOINT_FIELDS, "loss.u_mask_beam_jepa.dynamic_router.paired_joint")
    if type(raw.get("enabled")) is not bool or not raw["enabled"]:
        raise ValueError("Candidate dynamic Router requires paired_joint.enabled=true.")
    raw_panel_path = raw.get("panel_path", "")
    if not isinstance(raw_panel_path, str) or not raw_panel_path.strip():
        raise ValueError("paired_joint.panel_path must be non-empty.")
    panel_path = raw_panel_path.strip()
    panel_sha256 = str(raw.get("panel_sha256", "")).strip().lower()
    if len(panel_sha256) != 64 or any(char not in "0123456789abcdef" for char in panel_sha256):
        raise ValueError("paired_joint.panel_sha256 must be a 64-character SHA256 digest.")
    corruption_seed = raw.get("corruption_seed", 20260719)
    if type(corruption_seed) is not int or corruption_seed < 0:
        raise ValueError("paired_joint.corruption_seed must be a non-negative integer.")
    monotonic_weight = _finite_float(
        raw.get("monotonic_weight", 0.05),
        "paired_joint.monotonic_weight",
        positive=True,
    )
    margin_scale = _finite_float(
        raw.get("monotonic_margin_scale", 0.25),
        "paired_joint.monotonic_margin_scale",
        positive=True,
    )
    epsilon = _finite_float(
        raw.get("quality_drop_epsilon", 0.01),
        "paired_joint.quality_drop_epsilon",
        positive=True,
    )
    return {
        "enabled": True,
        "panel_path": panel_path,
        "panel_sha256": panel_sha256,
        "corruption_seed": corruption_seed,
        "monotonic_weight": monotonic_weight,
        "monotonic_margin_scale": margin_scale,
        "quality_drop_epsilon": epsilon,
    }


def _reject_unknown_fields(raw: dict[str, Any], allowed: frozenset[str], path: str) -> None:
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {path} fields: {unknown}.")


def _finite_float(value: Any, path: str, *, positive: bool = False, non_negative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a finite number.")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a finite number.") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{path} must be a finite number.")
    if positive and resolved <= 0.0:
        raise ValueError(f"{path} must be positive.")
    if non_negative and resolved < 0.0:
        raise ValueError(f"{path} must be non-negative.")
    return resolved


def _resolve_router_quality_pairing(value: Any) -> dict[str, Any]:
    if value is None:
        raw: dict[str, Any] = {}
    elif isinstance(value, dict):
        raw = dict(value)
    else:
        raise ValueError("loss.u_mask_beam_jepa.router_quality_pairing must be a mapping.")
    result = {
        "enabled": bool(raw.get("enabled", False)),
        "utility_mode": str(raw.get("utility_mode", "argmax")).strip().lower(),
        "target_temperature": float(raw.get("target_temperature", 0.1)),
        "beam_temperature": float(raw.get("beam_temperature", 1.0)),
        "start_epoch_index": int(raw.get("start_epoch_index", 0)),
        "max_target_entropy": (
            None if raw.get("max_target_entropy") is None else float(raw["max_target_entropy"])
        ),
        "utility_weight": float(raw.get("utility_weight", 0.0)),
        "monotonic_weight": float(raw.get("monotonic_weight", 0.0)),
        "monotonic_margin_scale": float(raw.get("monotonic_margin_scale", 0.25)),
        "quality_drop_epsilon": float(raw.get("quality_drop_epsilon", 0.01)),
        "corruption_seed": int(raw.get("corruption_seed", 20260719)),
    }
    if any(result[key] < 0.0 for key in ("utility_weight", "monotonic_weight", "monotonic_margin_scale", "quality_drop_epsilon")):
        raise ValueError("Router quality-pairing weights, margin, and epsilon must be non-negative.")
    if result["utility_mode"] not in {"argmax", "expected"}:
        raise ValueError("router_quality_pairing.utility_mode must be argmax or expected.")
    if result["target_temperature"] <= 0.0 or result["beam_temperature"] <= 0.0:
        raise ValueError("Router quality-pairing target and beam temperatures must be positive.")
    if result["enabled"] and result["utility_weight"] == 0.0 and result["monotonic_weight"] == 0.0:
        raise ValueError("Enabled router_quality_pairing requires a positive utility or monotonic weight.")
    if result["corruption_seed"] < 0:
        raise ValueError("router_quality_pairing.corruption_seed must be non-negative.")
    if result["start_epoch_index"] < 0:
        raise ValueError("router_quality_pairing.start_epoch_index must be non-negative.")
    if result["max_target_entropy"] is not None and result["max_target_entropy"] <= 0.0:
        raise ValueError("router_quality_pairing.max_target_entropy must be positive when set.")
    return result


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


def _resolve_prototype_topology(raw: dict[str, Any], *, use_bpa: bool) -> dict[str, Any]:
    if not use_bpa:
        return {
            "id": "not_applicable",
            "circular": False,
            "permutation": None,
            "descriptor_sha256": "",
            "audit_path": "",
        }
    legacy_circular = bool(raw.get("prototype_target_circular", True))
    value = raw.get("prototype_topology")
    if value is None:
        topology_id = "cyclic_index_v1" if legacy_circular else "linear_index_v1"
        payload: dict[str, Any] = {"id": topology_id}
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
