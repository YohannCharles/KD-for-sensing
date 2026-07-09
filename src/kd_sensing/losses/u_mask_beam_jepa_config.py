import warnings
from typing import Any


def u_mask_beam_jepa_config(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = cfg.get("loss", {}).get("u_mask_beam_jepa", {}) if isinstance(cfg.get("loss"), dict) else {}
    if raw is True:
        raw = {"enabled": True}
    if not isinstance(raw, dict):
        raw = {}
    resolved = dict(raw)
    resolved.setdefault("enabled", False)
    resolved.setdefault("lambda_teacher", 0.5)
    resolved.setdefault("lambda_jepa_global", resolved.get("lambda_jepa", 1.0))
    resolved.setdefault("lambda_modality_nll", 1.0)
    resolved.setdefault("use_teacher", cfg.get("model", {}).get("primary", {}).get("use_teacher", True))
    resolved.setdefault("use_jepa_loss", cfg.get("model", {}).get("primary", {}).get("use_jepa_loss", True))
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    primary_cfg = cfg.get("model", {}).get("primary", {}) if isinstance(cfg.get("model"), dict) else {}
    head_type = str(primary_cfg.get("head_type", training_cfg.get("head_type", "legacy")) or "legacy").strip().lower()
    use_modality_proto = training_cfg.get("use_modality_prototype_loss", primary_cfg.get("use_modality_prototype_loss", True))
    modality_proto_weight = training_cfg.get("modality_proto_weight", primary_cfg.get("modality_proto_weight"))
    beam_proto_weight = training_cfg.get("beam_proto_align_weight", primary_cfg.get("beam_proto_align_weight"))
    for key, default in {
        "mask_sampler": training_cfg.get("mask_sampler", "random_missing"),
        "pattern_probs": training_cfg.get("pattern_probs"),
        "use_beam_prototype_alignment": training_cfg.get(
            "use_beam_prototype_alignment", training_cfg.get("use_btapa", primary_cfg.get("use_beam_prototype_alignment", False))
        ),
        "lambda_proto": training_cfg.get("lambda_proto", training_cfg.get("btapa_lambda", 0.0))
        if beam_proto_weight is None
        else beam_proto_weight,
        "lambda_modality_proto": training_cfg.get(
            "lambda_modality_proto",
            0.0 if _is_false(use_modality_proto) else (0.0 if modality_proto_weight is None else modality_proto_weight),
        ),
        "lambda_supcon": training_cfg.get("lambda_supcon", 0.0),
        "lambda_teacher_proto": training_cfg.get("lambda_teacher_proto", 0.0),
        "beam_proto_temperature": training_cfg.get("beam_proto_temperature", primary_cfg.get("beam_proto_temperature", 0.2)),
        "use_beam_topology_proto": training_cfg.get(
            "use_beam_topology_proto", training_cfg.get("use_btapa", primary_cfg.get("use_beam_topology_proto", False))
        ),
        "proto_target_type": training_cfg.get("proto_target_type"),
        "tau_beam": training_cfg.get("btapa_tau_beam", training_cfg.get("tau_beam", 2.0)),
        "circular_beam_distance": training_cfg.get("circular_beam_distance", training_cfg.get("circular_distance")),
        "btapa_include_fusion": training_cfg.get("btapa_include_fusion", True),
        "btapa_include_modalities": training_cfg.get("btapa_include_modalities", True),
        "btapa_fusion_weight": training_cfg.get("btapa_fusion_weight", 1.0),
        "btapa_modality_weight": training_cfg.get("btapa_modality_weight"),
        "use_adba_aware_proto": training_cfg.get("use_adba_aware_proto", False),
        "lambda_adba_proto": training_cfg.get("lambda_adba_proto", 0.0),
        "adba_margin": training_cfg.get("adba_margin", 3),
        "beam_label_sigma": training_cfg.get("beam_label_sigma", 1.0),
        "beam_label_circular": training_cfg.get(
            "beam_label_circular",
            training_cfg.get("use_circular_soft_targets", primary_cfg.get("use_circular_soft_targets", True)),
        ),
        "use_full_to_partial_kd": training_cfg.get(
            "use_full_to_partial_kd", primary_cfg.get("use_full_to_partial_kd", False)
        ),
        "kd_teacher_mode": training_cfg.get("kd_teacher_mode", primary_cfg.get("kd_teacher_mode", "disabled")),
        "lambda_full_to_partial_kd": training_cfg.get("lambda_full_to_partial_kd", 0.0),
        "lambda_feature_kd": training_cfg.get("lambda_feature_kd", 0.0),
        "lambda_prototype_kd": training_cfg.get("lambda_prototype_kd", 0.0),
        "kd_temperature": training_cfg.get("kd_temperature", 1.0),
        "use_full_aux_loss": training_cfg.get("use_full_aux_loss", False),
        "lambda_full_aux": training_cfg.get("lambda_full_aux", 0.0),
        "full_aux_proto": training_cfg.get("full_aux_proto", False),
        "missing_pattern_sampler": training_cfg.get("missing_pattern_sampler", training_cfg.get("mask_sampler", "default")),
        "pattern_sampling_weights": training_cfg.get("pattern_sampling_weights", {}),
        "adaptive_alpha": training_cfg.get("adaptive_alpha", 0.5),
        "adaptive_temperature": training_cfg.get("adaptive_temperature", 1.0),
        "adaptive_ema_beta": training_cfg.get("adaptive_ema_beta", 0.9),
        "adaptive_score_mode": training_cfg.get("adaptive_score_mode", "gap_to_full"),
        "adaptive_min_prob": training_cfg.get("adaptive_min_prob", 0.05),
        "adaptive_max_prob": training_cfg.get("adaptive_max_prob", 0.40),
        "adaptive_update_freq": training_cfg.get("adaptive_update_freq", "step"),
        "adaptive_warmup_epochs": training_cfg.get("adaptive_warmup_epochs", 3),
        "curriculum_schedule": training_cfg.get("curriculum_schedule", {}),
        "use_pattern_conditional_btapa": training_cfg.get("use_pattern_conditional_btapa", False),
        "btapa_apply_patterns": training_cfg.get("btapa_apply_patterns", ()),
        "btapa_disable_on_patterns": training_cfg.get("btapa_disable_on_patterns", ()),
        "btapa_fallback_to_ordinary_proto": training_cfg.get("btapa_fallback_to_ordinary_proto", True),
        "ordinary_proto_target_type": training_cfg.get("ordinary_proto_target_type", "gaussian"),
        "use_hard_pattern_weight": training_cfg.get(
            "use_pattern_loss_weight", training_cfg.get("use_hard_pattern_weight", False)
        ),
        "pattern_loss_weights": training_cfg.get("pattern_loss_weights", {}),
        "hard_patterns": training_cfg.get("hard_patterns", ()),
        "hard_pattern_weight": training_cfg.get("hard_pattern_weight", 1.0),
        "apply_pattern_weight_to_ce": training_cfg.get("apply_pattern_weight_to_ce", True),
        "apply_pattern_weight_to_proto": training_cfg.get(
            "apply_pattern_weight_to_proto", training_cfg.get("hard_pattern_weight_apply_to_proto", False)
        ),
        "use_weak_pattern_kd": training_cfg.get("use_weak_pattern_kd", False),
        "kd_apply_patterns": training_cfg.get("kd_apply_patterns", ()),
        "lambda_kd": training_cfg.get("lambda_kd", 0.0),
        "use_light_latent_pred": training_cfg.get("use_light_latent_pred", False),
        "latent_pred_target": training_cfg.get("latent_pred_target", "full_fused"),
        "latent_pred_apply_patterns": training_cfg.get("latent_pred_apply_patterns", ()),
        "lambda_latent_pred": training_cfg.get("lambda_latent_pred", 0.0),
        "latent_pred_loss": training_cfg.get("latent_pred_loss", "cosine"),
        "mpdro": training_cfg.get("mpdro", {}),
        "router_supervision": training_cfg.get("router_supervision", primary_cfg.get("router_supervision", "none")),
        "router_distill_weight": training_cfg.get("router_distill_weight", primary_cfg.get("router_distill_weight", 0.0)),
        "temporal_router_distill_weight": training_cfg.get(
            "temporal_router_distill_weight",
            primary_cfg.get("temporal_router_distill_weight", primary_cfg.get("router_distill_weight", 0.0)),
        ),
    }.items():
        resolved.setdefault(key, default)
    if _is_false(use_modality_proto):
        resolved["lambda_modality_proto"] = 0.0
        resolved["btapa_modality_weight"] = 0.0
    if beam_proto_weight is not None:
        resolved["lambda_proto"] = float(beam_proto_weight)
    if head_type == "classifier":
        resolved["use_beam_prototype_alignment"] = False
        resolved["lambda_proto"] = 0.0
        resolved["lambda_modality_proto"] = 0.0
        resolved["btapa_modality_weight"] = 0.0
    if training_cfg.get("use_beam_prototype_alignment") is False or primary_cfg.get("use_beam_prototype_alignment") is False:
        resolved["use_beam_prototype_alignment"] = False
    if "beam_proto_align_weight" in training_cfg or "beam_proto_align_weight" in primary_cfg:
        resolved["lambda_proto"] = float(beam_proto_weight or 0.0)
    use_gaussian = training_cfg.get("use_gaussian_beam_targets", primary_cfg.get("use_gaussian_beam_targets"))
    use_circular = training_cfg.get("use_circular_soft_targets", primary_cfg.get("use_circular_soft_targets"))
    if _is_false(use_gaussian) and _is_false(use_circular):
        resolved["proto_target_type"] = "onehot"
        resolved["beam_label_circular"] = False
        resolved["circular_beam_distance"] = False
    if resolved.get("proto_target_type") is None:
        resolved["proto_target_type"] = "beam_soft" if bool(resolved.get("use_beam_topology_proto", False)) else "gaussian"
    if resolved.get("circular_beam_distance") is None:
        resolved["circular_beam_distance"] = bool(resolved.get("beam_label_circular", True))
    if resolved.get("btapa_modality_weight") is None:
        resolved["btapa_modality_weight"] = resolved.get("lambda_modality_proto", 0.0)
    resolved["missing_mask"] = _resolve_missing_mask_config(resolved)
    return resolved


def _is_false(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"0", "false", "no", "off", "none", ""}
    return value is False


def _resolve_missing_mask_config(raw: dict[str, Any]) -> dict[str, Any]:
    has_missing_mask = "missing_mask" in raw
    has_missing = "missing" in raw
    if has_missing_mask and has_missing:
        warnings.warn(
            "loss.u_mask_beam_jepa.missing is ignored because missing_mask is set; "
            "use missing_mask for U-MaskBeamJEPA missing-mask config.",
            UserWarning,
            stacklevel=3,
        )
    elif has_missing:
        warnings.warn(
            "loss.u_mask_beam_jepa.missing is deprecated; please rename it to missing_mask.",
            UserWarning,
            stacklevel=3,
        )
        return dict(raw["missing"])
    if has_missing_mask:
        return dict(raw["missing_mask"])
    return {"p_missing": 0.25, "ensure_at_least_one": True}
