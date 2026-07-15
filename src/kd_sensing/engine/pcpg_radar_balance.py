from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.engine.training_extensions import BatchState, ExtensionContext, LossBundle, TrainingExtension


SUPPORTED_HARD_PATTERNS = {
    "full",
    "drop1",
    "drop2",
    "drop3",
    "miss1",
    "miss2",
    "miss3",
    "image_only",
    "lidar_only",
    "radar_only",
    "gps_only",
    "missing_image",
    "missing_lidar",
    "missing_radar",
    "missing_gps",
}
SOFT_STATIC_HARD_SUBSET_WEIGHTS = {
    "full": 0.75,
    "miss1": 1.0,
    "miss2": 1.15,
    "miss3": 1.35,
    "radar_only": 1.50,
    "missing_image": 1.35,
}
ROUTER_MODALITY_ORDER = ("image", "lidar", "radar", "gps")


class PCPGRadarBalanceTrainingExtension(TrainingExtension):
    name = "pcpg_radar_balance"

    def setup(self, context: ExtensionContext) -> dict[str, Any]:
        return pcpg_radar_balance_config(context.cfg)

    def after_forward(
        self,
        context: ExtensionContext,
        state: Any,
        batch_state: BatchState,
    ) -> LossBundle | None:
        cfg = state if isinstance(state, dict) else {}
        if not bool(cfg.get("enabled", False)):
            return None
        zero = batch_state.primary_logits.sum() * 0.0
        total = zero
        diagnostics: dict[str, Any] = {}

        branch_total, branch_diag = _branch_aux_loss(context, cfg, batch_state, zero)
        total = total + branch_total
        diagnostics.update(branch_diag)

        hard_total, hard_diag = _hard_subset_extra(context, cfg, batch_state, zero)
        total = total + hard_total
        diagnostics.update(hard_diag)

        jepa_total, jepa_diag = _jepa_alignment_loss(cfg, batch_state, zero)
        total = total + jepa_total
        diagnostics.update(jepa_diag)

        bprr_total, bprr_diag = _bprr_gate_extra(context, cfg, batch_state, zero)
        total = total + bprr_total
        diagnostics.update(bprr_diag)

        router_total, router_diag = _supervised_router_extra(context, cfg, batch_state, zero)
        total = total + router_total
        diagnostics.update(router_diag)

        return LossBundle(total=total, components={"unimodal": branch_total}, diagnostics=diagnostics)


def pcpg_radar_balance_config(cfg: dict[str, Any]) -> dict[str, Any]:
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg.get("loss"), dict) else {}
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    raw = loss_cfg.get("pcpg_radar_balance", {})
    base = dict(raw) if isinstance(raw, dict) else {"enabled": bool(raw)}
    u_mask_cfg = loss_cfg.get("u_mask_beam_jepa", {})
    u_mask_cfg = u_mask_cfg if isinstance(u_mask_cfg, dict) else {}

    branch = loss_cfg.get("branch_aux_loss", base.get("branch_aux_loss", False))
    radar = loss_cfg.get("radar_protect_loss", base.get("radar_protect_loss", False))
    hard = loss_cfg.get("hard_subset_weighting", base.get("hard_subset_weighting", False))
    use_jepa = loss_cfg.get("use_jepa", base.get("use_jepa", False))

    base["branch_aux_loss"] = _enabled(branch)
    base["radar_protect_loss"] = _enabled(radar)
    base["hard_subset_weighting"] = _hard_subset_weighting_config(hard)
    base["hard_subset_weighting_type"] = str(base["hard_subset_weighting"].get("mode", "none"))
    base["use_jepa"] = bool(use_jepa)
    base["unimodal_aux_weight"] = float(
        loss_cfg.get("unimodal_aux_weight", base.get("unimodal_aux_weight", _weight(branch, 0.0)))
    )
    base["radar_aux_weight"] = float(
        loss_cfg.get("radar_aux_weight", base.get("radar_aux_weight", _weight(radar, 0.0)))
    )
    base["branch_aux_loss"] = bool(base["branch_aux_loss"] or base["unimodal_aux_weight"] > 0.0)
    base["radar_protect_loss"] = bool(base["radar_protect_loss"] or base["radar_aux_weight"] > 0.0)
    base["hard_subset_alpha"] = float(loss_cfg.get("hard_subset_alpha", base.get("hard_subset_alpha", 1.5)))
    base["hard_subset_focus"] = loss_cfg.get("hard_subset_focus", base.get("hard_subset_focus", list(SUPPORTED_HARD_PATTERNS)))
    base["jepa_weight"] = float(loss_cfg.get("jepa_weight", base.get("jepa_weight", 0.0)))
    base["bprr_gate_balance_weight"] = float(
        loss_cfg.get("bprr_gate_balance_weight", base.get("bprr_gate_balance_weight", 0.0))
    )
    base["bprr_radar_gate_reg_weight"] = float(
        loss_cfg.get("bprr_radar_gate_reg_weight", base.get("bprr_radar_gate_reg_weight", 0.0))
    )
    base["bprr_radar_gate_floor"] = float(
        loss_cfg.get("bprr_radar_gate_floor", base.get("bprr_radar_gate_floor", 0.10))
    )
    base["bprr_radar_gate_reg_patterns"] = loss_cfg.get(
        "bprr_radar_gate_reg_patterns",
        base.get("bprr_radar_gate_reg_patterns", ["radar_only", "missing_image", "miss3"]),
    )
    model_cfg = cfg.get("model", {}).get("primary", {}) if isinstance(cfg.get("model"), dict) else {}
    base["router_supervision"] = str(
        loss_cfg.get(
            "router_supervision",
            base.get("router_supervision", model_cfg.get("router_supervision", "none")),
        )
    ).strip().lower()
    base["router_distill_weight"] = float(
        loss_cfg.get(
            "router_distill_weight",
            base.get("router_distill_weight", model_cfg.get("router_distill_weight", 0.0)),
        )
    )
    base["router_distill_temperature"] = float(
        loss_cfg.get(
            "router_distill_temperature",
            base.get("router_distill_temperature", model_cfg.get("router_distill_temperature", 1.0)),
        )
    )
    base["router_focus_patterns"] = loss_cfg.get(
        "router_focus_patterns",
        base.get("router_focus_patterns", model_cfg.get("router_focus_patterns", ["missing_image", "miss2", "drop2"])),
    )
    base["router_fuse_level"] = str(
        loss_cfg.get("router_fuse_level", base.get("router_fuse_level", model_cfg.get("router_fuse_level", "logits")))
    ).strip().lower()
    base["circular_beam_distance"] = _bool_value(
        loss_cfg.get(
            "circular_beam_distance",
            base.get(
                "circular_beam_distance",
                u_mask_cfg.get(
                    "circular_beam_distance",
                    training_cfg.get(
                        "circular_beam_distance",
                        training_cfg.get("beam_label_circular", True),
                    ),
                ),
            ),
        )
    )
    for key in (
        "router_use_pattern_features",
        "router_use_reliability_features",
        "router_use_prototype_margin",
        "router_use_entropy",
        "router_use_confidence",
        "router_use_logit_norm",
    ):
        base[key] = _bool_value(loss_cfg.get(key, base.get(key, model_cfg.get(key, True))))
    if base["router_supervision"] not in {"oracle", "pattern_best", "none"}:
        raise ValueError("router_supervision must be one of oracle, pattern_best, or none.")
    if base["router_fuse_level"] != "logits":
        raise ValueError("supervised_router currently supports router_fuse_level='logits' only.")
    base["enabled"] = bool(
        base.get("enabled", False)
        or base["branch_aux_loss"]
        or base["radar_protect_loss"]
        or base["hard_subset_weighting"].get("enabled", False)
        or (base["use_jepa"] and base["jepa_weight"] > 0.0)
        or base["bprr_gate_balance_weight"] > 0.0
        or base["bprr_radar_gate_reg_weight"] > 0.0
        or (base["router_supervision"] != "none" and base["router_distill_weight"] > 0.0)
    )
    return base


def hard_subset_sample_weight(
    pattern_name: str,
    *,
    mode: str = "static",
    alpha: float = 1.5,
    full_weight: float = 0.5,
    unknown_weight: float = 1.0,
    focus: list[str] | tuple[str, ...] | str | None = None,
) -> float:
    mode_name = str(mode or "none").strip().lower()
    if mode_name in {"none", "false", "off", "0"}:
        return 1.0
    if mode_name == "soft_static":
        return soft_static_hard_subset_weight(pattern_name, unknown_weight=unknown_weight)
    return static_hard_subset_weight(
        pattern_name,
        alpha=alpha,
        full_weight=full_weight,
        unknown_weight=unknown_weight,
        focus=focus,
    )


def soft_static_hard_subset_weight(pattern_name: str, *, unknown_weight: float = 1.0) -> float:
    pattern = _canonical_pattern_alias(pattern_name)
    if pattern in SOFT_STATIC_HARD_SUBSET_WEIGHTS:
        return float(SOFT_STATIC_HARD_SUBSET_WEIGHTS[pattern])
    missing_count = missing_count_from_pattern_name(pattern_name)
    if missing_count == 1:
        return float(SOFT_STATIC_HARD_SUBSET_WEIGHTS["miss1"])
    if missing_count == 2:
        return float(SOFT_STATIC_HARD_SUBSET_WEIGHTS["miss2"])
    if missing_count == 3:
        return float(SOFT_STATIC_HARD_SUBSET_WEIGHTS["miss3"])
    return float(unknown_weight)


def static_hard_subset_weight(
    pattern_name: str,
    *,
    alpha: float = 1.5,
    full_weight: float = 0.5,
    unknown_weight: float = 1.0,
    focus: list[str] | tuple[str, ...] | None = None,
) -> float:
    pattern = str(pattern_name or "unknown").strip().lower()
    if isinstance(focus, str):
        focus = [item.strip() for item in focus.split(",") if item.strip()]
    focus_set = {str(item).strip().lower() for item in (focus or SUPPORTED_HARD_PATTERNS)}
    if pattern == "full":
        return float(full_weight)
    if pattern in focus_set:
        return float(alpha)
    if pattern.endswith("_only") and pattern in {"image_only", "lidar_only", "radar_only"}:
        return float(alpha)
    if pattern.startswith("missing_") and len([item for item in pattern.removeprefix("missing_").split("_") if item]) >= 3:
        return float(alpha)
    return float(unknown_weight)


def missing_count_from_pattern_name(pattern_name: str) -> int | None:
    pattern = _canonical_pattern_alias(pattern_name)
    if pattern == "full":
        return 0
    if pattern in {"miss1", "miss2", "miss3"}:
        return int(pattern[-1])
    if pattern.endswith("_only"):
        return 3
    if pattern.startswith("missing_"):
        return len([item for item in pattern.removeprefix("missing_").split("_") if item])
    return None


def pattern_name_from_available_mask(mask: torch.Tensor, modalities: list[str] | tuple[str, ...]) -> str:
    return pattern_names_from_available_mask(mask.reshape(1, -1), modalities)[0]


def pattern_names_from_available_mask(mask: torch.Tensor, modalities: list[str] | tuple[str, ...]) -> list[str]:
    values = mask.detach().to(device="cpu", dtype=torch.bool).tolist()
    return [_pattern_name_from_values(row, modalities) for row in values]


def _pattern_name_from_values(values: list[bool], modalities: list[str] | tuple[str, ...]) -> str:
    names = [str(item) for item in modalities]
    available = [name for name, keep in zip(names, values) if keep]
    missing = [name for name, keep in zip(names, values) if not keep]
    if len(available) == len(names):
        return "full"
    if len(available) == 1:
        return f"{available[0]}_only"
    if len(missing) == 1:
        return f"missing_{missing[0]}"
    if len(missing) >= 3:
        return "miss3"
    return "missing_" + "_".join(missing) if missing else "unknown"


def _branch_aux_loss(
    context: ExtensionContext,
    cfg: dict[str, Any],
    batch_state: BatchState,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    diagnostics = batch_state.primary_output.diagnostics
    unimodal_logits = diagnostics.get("pcpg_unimodal_logits")
    if not torch.is_tensor(unimodal_logits):
        unimodal_logits = diagnostics.get("unimodal_logits")
    if not torch.is_tensor(unimodal_logits):
        if bool(cfg.get("radar_protect_loss", False)) and float(cfg.get("radar_aux_weight", 0.0)) != 0.0:
            labels = batch_state.labels[:, 0] if batch_state.labels.ndim > 1 else batch_state.labels.reshape(-1)
            valid = labels.ne(-100)
            logits = batch_state.primary_logits[:, 0, :] if batch_state.primary_logits.ndim == 3 else batch_state.primary_logits
            per_sample = F.cross_entropy(logits, labels, reduction="none")
            valid_f = valid.to(dtype=per_sample.dtype)
            valid_count = valid_f.sum()
            fallback = (per_sample * valid_f).sum() / valid_count.clamp_min(1.0)
            available = valid_count.gt(0).to(dtype=per_sample.dtype)
            return float(cfg.get("radar_aux_weight", 0.0)) * fallback, {
                "loss/radar_aux": fallback.detach(),
                "pcpg/radar_aux_fallback_used": available.detach(),
                "pcpg/branch_aux_available": zero.detach(),
            }
        return zero, {"pcpg/branch_aux_available": 0.0}
    mask = _available_mask(diagnostics, batch_state, unimodal_logits)
    labels = batch_state.labels[:, 0] if batch_state.labels.ndim > 1 else batch_state.labels.reshape(-1)
    valid = labels.ne(-100)
    num_modalities = min(int(unimodal_logits.shape[1]), int(mask.shape[1]))
    logits = unimodal_logits[:, :num_modalities, :]
    active = valid.unsqueeze(1) & mask[:, :num_modalities].to(device=valid.device, dtype=torch.bool)
    flat_labels = labels.unsqueeze(1).expand(-1, num_modalities).reshape(-1)
    per_sample = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), flat_labels, reduction="none").reshape_as(active)
    active_f = active.to(dtype=per_sample.dtype)
    modality_counts = active_f.sum(dim=0)
    modality_losses = (per_sample * active_f).sum(dim=0) / modality_counts.clamp_min(1.0)
    active_modalities = modality_counts.gt(0).to(dtype=per_sample.dtype)
    branch_loss = (modality_losses * active_modalities).sum() / active_modalities.sum().clamp_min(1.0)

    modalities = list(getattr(context.primary_model, "modalities", ())) or [f"modality_{i}" for i in range(num_modalities)]
    radar_loss = zero
    radar_acc = zero
    if "radar" in modalities[:num_modalities]:
        radar_index = modalities.index("radar")
        radar_loss = modality_losses[radar_index]
        radar_active = active_f[:, radar_index]
        radar_acc = (
            logits[:, radar_index, :].argmax(dim=-1).eq(labels).to(dtype=per_sample.dtype) * radar_active
        ).sum() / radar_active.sum().clamp_min(1.0)
    total = zero
    if bool(cfg.get("branch_aux_loss", False)) and float(cfg.get("unimodal_aux_weight", 0.0)) != 0.0:
        total = total + float(cfg.get("unimodal_aux_weight", 0.0)) * branch_loss
    if bool(cfg.get("radar_protect_loss", False)) and float(cfg.get("radar_aux_weight", 0.0)) != 0.0:
        total = total + float(cfg.get("radar_aux_weight", 0.0)) * radar_loss
    return total, {
        "loss/unimodal_aux": branch_loss.detach(),
        "loss/radar_aux": radar_loss.detach(),
        "pcpg/radar_aux_accuracy": radar_acc.detach(),
        "pcpg/branch_aux_available": valid.any().to(dtype=per_sample.dtype).detach(),
    }


def _hard_subset_extra(
    context: ExtensionContext,
    cfg: dict[str, Any],
    batch_state: BatchState,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    hard_cfg = cfg.get("hard_subset_weighting", {})
    if not isinstance(hard_cfg, dict) or not bool(hard_cfg.get("enabled", False)) or batch_state.task_loss is None:
        return zero, {}
    diagnostics = batch_state.primary_output.diagnostics
    mask = _available_mask(diagnostics, batch_state, batch_state.primary_logits)
    modalities = list(getattr(context.primary_model, "modalities", ())) or [f"modality_{i}" for i in range(mask.shape[1])]
    names = pattern_names_from_available_mask(mask, modalities)
    weights = torch.tensor(
        [
            hard_subset_sample_weight(
                name,
                mode=str(hard_cfg.get("mode", cfg.get("hard_subset_weighting_type", "static"))),
                alpha=float(cfg.get("hard_subset_alpha", 1.5)),
                full_weight=float(hard_cfg.get("full_weight", 0.5)),
                unknown_weight=float(hard_cfg.get("unknown_weight", 1.0)),
                focus=cfg.get("hard_subset_focus"),
            )
            for name in names
        ],
        device=batch_state.primary_logits.device,
        dtype=batch_state.primary_logits.dtype,
    )
    mean_weight = weights.mean()
    extra = batch_state.task_loss * (mean_weight - 1.0)
    return extra, {
        "loss/hard_subset_extra": extra.detach(),
        "pcpg/hard_subset_weight_mean": mean_weight.detach(),
    }


def _jepa_alignment_loss(cfg: dict[str, Any], batch_state: BatchState, zero: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    if not bool(cfg.get("use_jepa", False)) or float(cfg.get("jepa_weight", 0.0)) == 0.0:
        return zero, {}
    diagnostics = batch_state.primary_output.diagnostics
    target = diagnostics.get("u_star")
    student = batch_state.primary_output.output_features
    if not torch.is_tensor(target) or student is None:
        return zero, {"pcpg/jepa_alignment_available": 0.0}
    loss = F.mse_loss(F.normalize(student, dim=-1), F.normalize(target.to(device=student.device), dim=-1))
    weighted = float(cfg.get("jepa_weight", 0.0)) * loss
    return weighted, {
        "loss/jepa_latent_alignment": loss.detach(),
        "pcpg/jepa_alignment_available": 1.0,
    }


def _bprr_gate_extra(
    context: ExtensionContext,
    cfg: dict[str, Any],
    batch_state: BatchState,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    balance_weight = float(cfg.get("bprr_gate_balance_weight", 0.0))
    radar_weight = float(cfg.get("bprr_radar_gate_reg_weight", 0.0))
    if balance_weight == 0.0 and radar_weight == 0.0:
        return zero, {}
    diagnostics = batch_state.primary_output.diagnostics
    if diagnostics.get("reliability_fusion_mode") != "bprr":
        return zero, {"bprr/gate_regularization_available": 0.0}
    gate = diagnostics.get("bprr_gate_weights", diagnostics.get("reliability_fusion_weights"))
    if not torch.is_tensor(gate):
        return zero, {"bprr/gate_regularization_available": 0.0}
    mask = _available_mask(diagnostics, batch_state, gate)
    modalities = list(getattr(context.primary_model, "modalities", ())) or [f"modality_{i}" for i in range(gate.shape[1])]
    total, raw = bprr_gate_regularization(
        gate,
        mask,
        modalities,
        balance_weight=balance_weight,
        radar_weight=radar_weight,
        radar_floor=float(cfg.get("bprr_radar_gate_floor", 0.10)),
        radar_patterns=cfg.get("bprr_radar_gate_reg_patterns"),
    )
    total = total if torch.is_tensor(total) else zero
    return total, {
        "loss/bprr_gate_balance": raw["balance"].detach(),
        "loss/bprr_radar_gate": raw["radar"].detach(),
        "loss/bprr_gate_regularization": total.detach(),
        "bprr/gate_regularization_available": 1.0,
    }


def _supervised_router_extra(
    context: ExtensionContext,
    cfg: dict[str, Any],
    batch_state: BatchState,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    diagnostics = batch_state.primary_output.diagnostics
    gate = diagnostics.get("supervised_router_gate_weights", diagnostics.get("reliability_fusion_weights"))
    if not torch.is_tensor(gate):
        return zero, {}
    mask = _available_mask(diagnostics, batch_state, gate)
    modalities = list(getattr(context.primary_model, "modalities", ())) or [f"modality_{i}" for i in range(gate.shape[1])]
    return supervised_router_distill_extra(cfg, diagnostics, batch_state.labels, mask, modalities, zero)


def supervised_router_distill_extra(
    cfg: dict[str, Any],
    diagnostics: dict[str, Any],
    labels: torch.Tensor,
    available_mask: torch.Tensor,
    modalities: list[str] | tuple[str, ...],
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    supervision = str(cfg.get("router_supervision", "none")).strip().lower()
    weight = float(cfg.get("router_distill_weight", 0.0))
    if supervision == "none" or weight == 0.0:
        return zero, {}
    if diagnostics.get("reliability_fusion_mode") != "supervised_router":
        return zero, {"router/supervision_available": 0.0}
    gate_logits = diagnostics.get("supervised_router_gate_logits", diagnostics.get("router_gate_logits"))
    gate_weights = diagnostics.get("supervised_router_gate_weights", diagnostics.get("reliability_fusion_weights"))
    unimodal_logits = diagnostics.get("unimodal_logits", diagnostics.get("pcpg_unimodal_logits"))
    if not torch.is_tensor(gate_logits) or not torch.is_tensor(gate_weights):
        return zero, {"router/supervision_available": 0.0}
    labels_flat = labels[:, 0] if labels.ndim > 1 else labels.reshape(-1)
    available = available_mask.to(device=gate_logits.device, dtype=torch.bool)
    patterns = pattern_names_from_available_mask(available, modalities)
    focus = router_focus_mask(patterns, cfg.get("router_focus_patterns"), device=gate_logits.device)
    valid = labels_flat.to(device=gate_logits.device).ne(-100) & focus & (available.sum(dim=1) > 1)
    if supervision == "oracle" and torch.is_tensor(unimodal_logits):
        targets = supervised_router_oracle_targets(
            unimodal_logits,
            labels_flat,
            available,
            circular_beam_distance=bool(cfg.get("circular_beam_distance", True)),
        )
    else:
        targets = pattern_best_router_targets(patterns, available, modalities)
    temperature = max(float(cfg.get("router_distill_temperature", 1.0)), 1e-6)
    masked_logits = gate_logits.to(dtype=torch.float32).masked_fill(~available, torch.finfo(torch.float32).min)
    safe_logits = torch.where(available.any(dim=1, keepdim=True), masked_logits, torch.zeros_like(masked_logits))
    per_sample = F.cross_entropy(safe_logits / temperature, targets.to(device=gate_logits.device), reduction="none")
    valid_f = valid.to(dtype=per_sample.dtype)
    loss = (per_sample * valid_f).sum() / valid_f.sum().clamp_min(1.0)
    predicted = gate_weights.argmax(dim=1)
    active_targets = targets.to(device=gate_logits.device)[valid]
    active_predicted = predicted[valid]
    total = weight * loss
    diagnostics_out = router_diagnostics_from_targets(
        gate_weights,
        active_predicted,
        active_targets,
        patterns,
        valid,
        modalities,
    )
    diagnostics_out.update(
        {
            "loss/router_distill": loss.detach(),
            "router/supervision_available": 1.0,
            "router/distill_active_rate": valid_f.mean().detach(),
        }
    )
    return total, diagnostics_out


def supervised_router_oracle_targets(
    unimodal_logits: torch.Tensor,
    labels: torch.Tensor,
    available_mask: torch.Tensor,
    *,
    circular_beam_distance: bool = True,
) -> torch.Tensor:
    if unimodal_logits.ndim != 3:
        raise ValueError(f"unimodal_logits must have shape [B, M, C], got {tuple(unimodal_logits.shape)}.")
    available = available_mask.to(device=unimodal_logits.device, dtype=torch.bool)
    if available.shape != unimodal_logits.shape[:2]:
        raise ValueError(f"available_mask shape {tuple(available.shape)} must match logits {tuple(unimodal_logits.shape[:2])}.")
    labels_flat = labels.to(device=unimodal_logits.device, dtype=torch.long).reshape(-1)
    safe_labels = labels_flat.clamp_min(0)
    predictions = unimodal_logits.argmax(dim=-1)
    distance = (predictions - safe_labels.view(-1, 1)).abs()
    if circular_beam_distance:
        distance = torch.minimum(distance, unimodal_logits.shape[-1] - distance)
    ce = F.cross_entropy(
        unimodal_logits.reshape(-1, unimodal_logits.shape[-1]),
        safe_labels.repeat_interleave(unimodal_logits.shape[1]),
        reduction="none",
    ).view(unimodal_logits.shape[:2])
    confidence = torch.softmax(unimodal_logits, dim=-1).amax(dim=-1)
    order = torch.arange(unimodal_logits.shape[1], device=unimodal_logits.device, dtype=torch.float32).view(1, -1)
    score = distance.to(torch.float32) * 1_000_000.0 + ce * 1_000.0 - confidence * 0.001 + order * 1e-6
    score = score.masked_fill(~available, torch.finfo(score.dtype).max)
    target = score.argmin(dim=1)
    first_available = available.to(dtype=torch.int64).argmax(dim=1)
    return torch.where(available.any(dim=1), target, first_available)


def supervised_router_masked_softmax(logits: torch.Tensor, available_mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    available = available_mask.to(device=logits.device, dtype=torch.bool)
    if logits.shape != available.shape:
        raise ValueError(f"router logits shape {tuple(logits.shape)} must match mask {tuple(available.shape)}.")
    masked = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked, dim=-1) * available.to(dtype=logits.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    return torch.where(available.any(dim=-1, keepdim=True), weights, torch.zeros_like(weights))


def router_focus_mask(
    pattern_names: list[str] | tuple[str, ...],
    focus_patterns: list[str] | tuple[str, ...] | str | None,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    focus = _pattern_set(focus_patterns or ("missing_image", "miss2", "drop2"))
    values = [router_focus_pattern_enabled(pattern, focus) for pattern in pattern_names]
    return torch.tensor(values, device=device, dtype=torch.bool)


def router_focus_pattern_enabled(pattern_name: str, focus_patterns: set[str] | list[str] | tuple[str, ...] | str | None) -> bool:
    pattern = _canonical_pattern_alias(pattern_name)
    focus = focus_patterns if isinstance(focus_patterns, set) else _pattern_set(focus_patterns)
    aliases = {pattern}
    count = missing_count_from_pattern_name(pattern)
    if "all_multimodal" in focus:
        return count is not None and count <= 2
    if count is not None:
        aliases.update({f"miss{count}", f"drop{count}"})
    return bool(aliases & set(focus))


def pattern_best_router_targets(
    pattern_names: list[str] | tuple[str, ...],
    available_mask: torch.Tensor,
    modalities: list[str] | tuple[str, ...],
) -> torch.Tensor:
    names = [str(item) for item in modalities]
    priorities = {
        "missing_image": ("radar", "lidar", "gps", "image"),
        "miss2": ("radar", "lidar", "gps", "image"),
        "drop2": ("radar", "lidar", "gps", "image"),
    }
    targets: list[int] = []
    available = available_mask.detach().to(device="cpu", dtype=torch.bool).tolist()
    for row, pattern in zip(available, pattern_names):
        canonical = _canonical_pattern_alias(pattern)
        priority = priorities.get(canonical)
        count = missing_count_from_pattern_name(canonical)
        if priority is None and count == 2:
            priority = priorities["miss2"]
        selected = None
        for name in priority or names:
            if name in names:
                index = names.index(name)
                if row[index]:
                    selected = index
                    break
        if selected is None:
            selected = next((index for index, enabled in enumerate(row) if enabled), 0)
        targets.append(selected)
    return torch.tensor(targets, device=available_mask.device, dtype=torch.long)


def router_diagnostics_from_targets(
    gate_weights: torch.Tensor,
    active_predicted: torch.Tensor,
    active_targets: torch.Tensor,
    pattern_names: list[str],
    valid_mask: torch.Tensor,
    modalities: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    weights = gate_weights.detach()
    active = valid_mask.to(device=weights.device, dtype=torch.bool)
    active_f = active.to(dtype=weights.dtype)
    active_count = active_f.sum()
    names = [str(item) for item in modalities]
    correct = active_predicted.eq(active_targets).to(dtype=torch.float32)
    out["router/oracle_acc"] = (correct.sum() / active_count.clamp_min(1.0)).detach()
    out["router_oracle_acc"] = out["router/oracle_acc"]
    for modality in ROUTER_MODALITY_ORDER:
        if modality in names:
            index = names.index(modality)
            out[f"oracle_target_{modality}_rate"] = (
                active_targets.eq(index).to(dtype=weights.dtype).sum() / active_count.clamp_min(1.0)
            ).detach()
            out[f"mean_gate_{modality}"] = (
                (weights[:, index] * active_f).sum() / active_count.clamp_min(1.0)
            ).detach()
    target_by_row = torch.zeros_like(weights.argmax(dim=1), dtype=torch.long)
    target_by_row[active] = active_targets
    prediction_by_row = weights.argmax(dim=1)
    correct_by_row = prediction_by_row.eq(target_by_row).to(dtype=weights.dtype)
    pattern_tensor = {
        "missing_image": router_focus_mask(pattern_names, ["missing_image"], device=weights.device),
        "drop2": router_focus_mask(pattern_names, ["miss2", "drop2"], device=weights.device),
    }
    for key, mask in pattern_tensor.items():
        selected = active & mask
        selected_f = selected.to(dtype=weights.dtype)
        selected_count = selected_f.sum()
        out[f"router_oracle_acc_{key}"] = (
            (correct_by_row * selected_f).sum() / selected_count.clamp_min(1.0)
        ).detach()
        if "radar" in names:
            out[f"radar_gate_{key}"] = (
                (weights[:, names.index("radar")] * selected_f).sum() / selected_count.clamp_min(1.0)
            ).detach()
    gate_entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)
    out["gate_entropy"] = ((gate_entropy * active_f).sum() / active_count.clamp_min(1.0)).detach()
    return out


def bprr_gate_regularization(
    gate: torch.Tensor,
    available_mask: torch.Tensor,
    modalities: list[str] | tuple[str, ...],
    *,
    balance_weight: float = 0.0,
    radar_weight: float = 0.0,
    radar_floor: float = 0.10,
    radar_patterns: list[str] | tuple[str, ...] | str | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    available = available_mask.to(device=gate.device, dtype=torch.bool)
    if gate.shape != available.shape:
        raise ValueError(f"bprr gate shape {tuple(gate.shape)} must match available mask {tuple(available.shape)}.")
    zero = gate.sum() * 0.0
    available_float = available.to(dtype=gate.dtype)
    counts = available_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    prior = available_float / counts
    balance = (((gate - prior) ** 2) * available_float).sum(dim=1) / counts.squeeze(1)
    balance_loss = balance.mean() if balance.numel() else zero

    names = [str(item) for item in modalities]
    if "radar" not in names:
        radar_loss = zero
    else:
        radar_index = names.index("radar")
        hard = _pattern_set(radar_patterns or ("radar_only", "missing_image", "miss3"))
        row_patterns = pattern_names_from_available_mask(available, names)
        hard_patterns = torch.tensor(
            [pattern in hard and pattern != "radar_only" for pattern in row_patterns],
            device=gate.device,
            dtype=torch.bool,
        )
        active = available[:, radar_index] & available.sum(dim=1).gt(1) & hard_patterns
        active_f = active.to(dtype=gate.dtype)
        radar_loss = (F.relu(float(radar_floor) - gate[:, radar_index]) * active_f).sum() / active_f.sum().clamp_min(1.0)
    total = float(balance_weight) * balance_loss + float(radar_weight) * radar_loss
    return total, {"balance": balance_loss, "radar": radar_loss}


def _available_mask(diagnostics: dict[str, Any], batch_state: BatchState, reference: torch.Tensor) -> torch.Tensor:
    for key in ("pcpg_available_mask", "reliability_fusion_available_mask", "missing_mask"):
        value = diagnostics.get(key)
        if torch.is_tensor(value) and value.ndim == 2:
            return value.to(device=reference.device, dtype=torch.bool)
    value = batch_state.controls.model_kwargs.get("missing_mask")
    if torch.is_tensor(value) and value.ndim == 2:
        return value.to(device=reference.device, dtype=torch.bool)
    batch_size = int(reference.shape[0])
    num_modalities = int(getattr(batch_state.primary_output.input_features, "shape", (batch_size, 1))[1])
    return torch.ones(batch_size, num_modalities, device=reference.device, dtype=torch.bool)


def _enabled(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get("enabled", True))
    return bool(value)


def _weight(value: Any, default: float) -> float:
    if isinstance(value, dict):
        return float(value.get("weight", default))
    return default


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", "none", ""}
    return bool(value)


def _hard_subset_weighting_config(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        cfg = dict(value)
        mode = str(cfg.get("mode", "static" if bool(cfg.get("enabled", False)) else "none")).strip().lower()
        if mode in {"false", "off", "0"}:
            mode = "none"
        cfg["mode"] = mode
        cfg["enabled"] = bool(cfg.get("enabled", mode != "none"))
        return cfg
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in {"", "none", "false", "off", "0"}:
            return {"enabled": False, "mode": "none"}
        if mode not in {"static", "soft_static", "dynamic"}:
            raise ValueError("hard_subset_weighting must be one of none, static, soft_static, or dynamic.")
        return {"enabled": True, "mode": mode}
    return {"enabled": bool(value), "mode": "static" if bool(value) else "none"}


def _canonical_pattern_alias(pattern_name: str) -> str:
    pattern = str(pattern_name or "unknown").strip().lower()
    aliases = {"drop1": "miss1", "drop2": "miss2", "drop3": "miss3"}
    return aliases.get(pattern, pattern)


def _pattern_set(value: list[str] | tuple[str, ...] | str | None) -> set[str]:
    if isinstance(value, str):
        return {_canonical_pattern_alias(item) for item in value.split(",") if item.strip()}
    return {_canonical_pattern_alias(str(item)) for item in (value or []) if str(item).strip()}
