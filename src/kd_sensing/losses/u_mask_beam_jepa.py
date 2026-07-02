import warnings
from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.data.missing_mask import (
    get_missing_pattern_name,
    make_pattern_mask,
    sample_missing_mask,
    sample_pattern_balanced_mask,
)
from kd_sensing.engine.training_extensions import BaseLossResult, BatchState, ExtensionContext, ForwardControls, TrainingExtension
from kd_sensing.losses.beam_prototype_alignment import prototype_alignment_loss, supervised_contrastive_loss


def u_mask_beam_jepa_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    lambda_teacher: float = 0.5,
    lambda_jepa_global: float | None = None,
    lambda_modality_nll: float = 1.0,
    lambda_jepa: float | None = None,
    use_teacher: bool = True,
    use_jepa_loss: bool = True,
    logvar_min: float = -6.0,
    logvar_max: float = 2.0,
    teacher_output: dict[str, torch.Tensor] | None = None,
    prototype_bank: torch.nn.Module | None = None,
    use_beam_prototype_alignment: bool = False,
    lambda_proto: float = 0.0,
    lambda_modality_proto: float = 0.0,
    lambda_supcon: float = 0.0,
    lambda_teacher_proto: float = 0.0,
    beam_label_sigma: float = 1.0,
    beam_label_circular: bool = True,
    proto_target_type: str = "gaussian",
    tau_beam: float = 2.0,
    circular_beam_distance: bool | None = None,
    btapa_include_fusion: bool = True,
    btapa_include_modalities: bool = True,
    btapa_fusion_weight: float = 1.0,
    btapa_modality_weight: float | None = None,
    use_adba_aware_proto: bool = False,
    lambda_adba_proto: float = 0.0,
    adba_margin: int = 3,
    use_full_to_partial_kd: bool = False,
    lambda_full_to_partial_kd: float = 0.0,
    lambda_feature_kd: float = 0.0,
    lambda_prototype_kd: float = 0.0,
    kd_temperature: float = 1.0,
    sample_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    if lambda_jepa_global is None:
        lambda_jepa_global = 1.0 if lambda_jepa is None else float(lambda_jepa)
    logits = output["logits"]
    labels = labels.to(device=logits.device, dtype=torch.long)
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    loss_beam = _sample_weighted_ce(logits, labels[:, : logits.shape[1]], sample_weights)
    zero = logits.sum() * 0.0
    teacher_logits = output.get("teacher_logits")
    if use_teacher and torch.is_tensor(teacher_logits):
        if teacher_logits.ndim == 2:
            teacher_logits = teacher_logits.unsqueeze(1)
        loss_teacher = F.cross_entropy(
            teacher_logits.reshape(-1, teacher_logits.shape[-1]),
            labels[:, : teacher_logits.shape[1]].reshape(-1),
        )
    else:
        loss_teacher = zero
    if use_jepa_loss:
        target = output["u_star"].detach()
        logvar_b = output["logvar_B"].clamp(float(logvar_min), float(logvar_max))
        loss_jepa_global = _global_jepa_loss(output["mu_B"], logvar_b, target)
        loss_modality_nll = _modality_uncertainty_loss(
            output["modality_mu_B"],
            output["modality_logvar_B"].clamp(float(logvar_min), float(logvar_max)),
            target,
            output.get("missing_mask"),
        )
    else:
        loss_jepa_global = zero
        loss_modality_nll = zero
    loss = (
        loss_beam
        + float(lambda_teacher) * loss_teacher
        + float(lambda_jepa_global) * loss_jepa_global
        + float(lambda_modality_nll) * loss_modality_nll
    )
    diagnostics_extra: dict[str, float] = {}
    if use_beam_prototype_alignment and prototype_bank is not None:
        proto_loss, proto_diag = prototype_alignment_loss(
            prototype_bank,
            labels,
            fused_features=output["output_features"],
            modality_features=output.get("modality_features"),
            mask=output.get("missing_mask"),
            teacher_features=(teacher_output or {}).get("output_features"),
            beam_label_sigma=beam_label_sigma,
            beam_label_circular=beam_label_circular,
            proto_target_type=proto_target_type,
            tau_beam=tau_beam,
            circular_beam_distance=circular_beam_distance,
            lambda_proto=lambda_proto,
            lambda_modality_proto=lambda_modality_proto,
            lambda_teacher_proto=lambda_teacher_proto,
            btapa_include_fusion=btapa_include_fusion,
            btapa_include_modalities=btapa_include_modalities,
            btapa_fusion_weight=btapa_fusion_weight,
            btapa_modality_weight=btapa_modality_weight,
            use_adba_aware_proto=use_adba_aware_proto,
            lambda_adba_proto=lambda_adba_proto,
            adba_margin=adba_margin,
        )
        loss = loss + proto_loss
        diagnostics_extra.update(proto_diag)
        if float(lambda_supcon) != 0.0:
            supcon, supcon_diag = supervised_contrastive_loss(output["output_features"], labels[:, 0], temperature=kd_temperature)
            loss = loss + float(lambda_supcon) * supcon
            diagnostics_extra.update(supcon_diag)
    if use_full_to_partial_kd and teacher_output is not None:
        teacher_logits = teacher_output["logits"].detach()
        logit_kd = _logit_kd_loss(logits, teacher_logits, temperature=kd_temperature)
        student_feature = output["output_features"]
        teacher_feature = teacher_output["output_features"].detach()
        feature_kd = F.mse_loss(student_feature, teacher_feature)
        prototype_kd = logits.sum() * 0.0
        if prototype_bank is not None and float(lambda_prototype_kd) != 0.0:
            prototype_kd = _logit_kd_loss(
                prototype_bank(student_feature),
                prototype_bank(teacher_feature).detach(),
                temperature=kd_temperature,
            )
        loss = (
            loss
            + float(lambda_full_to_partial_kd) * logit_kd
            + float(lambda_feature_kd) * feature_kd
            + float(lambda_prototype_kd) * prototype_kd
        )
        diagnostics_extra.update(
            {
                "loss/full_to_partial_kd": float(logit_kd.detach().cpu().item()),
                "loss/feature_kd": float(feature_kd.detach().cpu().item()),
                "loss/prototype_kd": float(prototype_kd.detach().cpu().item()),
                "teacher_top1": _top1(teacher_logits, labels),
                "student_top1": _top1(logits, labels),
                "kd_gap": float((teacher_logits.detach() - logits.detach()).abs().mean().cpu().item()),
            }
        )
    diagnostics = _loss_diagnostics(
        logits,
        labels,
        output,
        loss_beam=loss_beam,
        loss_teacher=loss_teacher,
        loss_jepa_global=loss_jepa_global,
        loss_modality_nll=loss_modality_nll,
        use_teacher=use_teacher,
        use_jepa_loss=use_jepa_loss,
    )
    diagnostics.update(diagnostics_extra)
    return {
        "loss": loss,
        "loss_beam": loss_beam,
        "loss_teacher": loss_teacher,
        "loss_jepa_global": loss_jepa_global,
        "loss_modality_nll": loss_modality_nll,
        "loss_jepa": loss_jepa_global + loss_modality_nll,
        "diagnostics": diagnostics,
    }


def _global_jepa_loss(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mu.shape != target.shape:
        raise ValueError(f"mu_B shape {tuple(mu.shape)} does not match u_star {tuple(target.shape)}.")
    if logvar.shape != mu.shape:
        raise ValueError(f"logvar_B shape {tuple(logvar.shape)} does not match mu_B {tuple(mu.shape)}.")
    return (0.5 * ((target - mu).pow(2) * torch.exp(-logvar) + logvar)).mean()


def _modality_uncertainty_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mu.ndim != 3:
        raise ValueError(f"modality_mu_B must have shape [B, M, D], got {tuple(mu.shape)}.")
    if target.shape != (mu.shape[0], mu.shape[-1]):
        raise ValueError(f"u_star shape {tuple(target.shape)} does not match modality_mu_B {tuple(mu.shape)}.")
    if logvar.shape[:2] != mu.shape[:2] or logvar.shape[-1] not in {1, mu.shape[-1]}:
        raise ValueError(
            "modality_logvar_B must have shape [B, M, D] or [B, M, 1], "
            f"got {tuple(logvar.shape)} for modality_mu_B {tuple(mu.shape)}."
        )
    if logvar.shape[-1] == 1:
        logvar = logvar.expand_as(mu)
    if mask is None:
        mask = torch.ones(mu.shape[:2], dtype=torch.bool, device=mu.device)
    mask = mask.to(device=mu.device, dtype=torch.bool)
    if tuple(mask.shape) != tuple(mu.shape[:2]):
        raise ValueError(f"missing_mask must have shape {tuple(mu.shape[:2])}, got {tuple(mask.shape)}.")
    available = mask.unsqueeze(-1).to(dtype=mu.dtype)
    nll = 0.5 * ((target.unsqueeze(1) - mu).pow(2) * torch.exp(-logvar) + logvar)
    return (nll * available).sum() / (available.sum().clamp_min(1.0) * int(mu.shape[-1]))


def _sample_weighted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None,
) -> torch.Tensor:
    if sample_weights is None:
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    weights = sample_weights.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    if int(weights.numel()) != int(logits.shape[0]):
        raise ValueError(f"sample_weights must have shape [B], got {tuple(sample_weights.shape)} for B={logits.shape[0]}.")
    per_token = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").view(
        logits.shape[0], -1
    )
    valid = labels.reshape(logits.shape[0], -1).ne(-100)
    per_sample = (per_token * valid.to(dtype=per_token.dtype)).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    active = valid.any(dim=1).to(dtype=weights.dtype)
    return (per_sample * weights * active).sum() / (weights * active).sum().clamp_min(1e-6)


class UMaskBeamJEPATrainingExtension(TrainingExtension):
    name = "u_mask_beam_jepa"

    def setup(self, context: ExtensionContext) -> dict[str, Any]:
        cfg = u_mask_beam_jepa_config(context.cfg)
        if cfg.get("kd_teacher_mode") == "checkpoint":
            raise NotImplementedError(
                "loss.u_mask_beam_jepa.kd_teacher_mode=checkpoint is pending; use online_full or disable "
                "full-to-partial stabilization."
            )
        return {"config": cfg}

    def before_forward(
        self,
        context: ExtensionContext,
        state: Any,
        batch: dict[str, torch.Tensor],
        labels: torch.Tensor,
        *,
        epoch: int,
    ) -> ForwardControls:
        del epoch
        cfg = state.get("config", {}) if isinstance(state, dict) else {}
        if not cfg.get("enabled", False):
            return ForwardControls()
        modalities = tuple(getattr(context.primary_model, "modalities", context.model_cfg.get("primary", {}).get("modalities", ())))
        mask_cfg = cfg.get("missing_mask", {})
        pattern = cfg.get("pattern") or cfg.get("missing_pattern")
        if cfg.get("mask_sampler") == "pattern_balanced":
            mask, pattern_names, pattern_ids = sample_pattern_balanced_mask(
                int(labels.shape[0]),
                modalities,
                cfg.get("pattern_probs"),
                device=context.device,
            )
            state["pattern_names"] = pattern_names
            state["pattern_ids"] = pattern_ids
        elif isinstance(pattern, dict):
            mask = make_pattern_mask(
                int(labels.shape[0]),
                modalities,
                available_modalities=pattern.get("available_modalities"),
                pattern_mask=pattern.get("pattern_mask"),
                device=context.device,
            )
        else:
            mask = sample_missing_mask(
                int(labels.shape[0]),
                len(modalities),
                mask_cfg.get("p_missing", 0.25),
                always_available_indices=mask_cfg.get("always_available_indices"),
                ensure_at_least_one=bool(mask_cfg.get("ensure_at_least_one", True)),
                device=context.device,
            )
            state["pattern_names"] = None
            state["pattern_ids"] = None
        if cfg.get("use_full_to_partial_kd", False) and cfg.get("kd_teacher_mode") == "online_full":
            state["online_teacher"] = _online_full_teacher(context, batch, modalities, labels, cfg)
        else:
            state["online_teacher"] = None
        return ForwardControls(model_kwargs={"missing_mask": mask})

    def compute_base_loss(self, context: ExtensionContext, state: Any, batch_state: BatchState) -> BaseLossResult | None:
        cfg = state.get("config", {}) if isinstance(state, dict) else {}
        if not cfg.get("enabled", False):
            return None
        output = {
            "logits": batch_state.primary_logits,
            "output_features": batch_state.primary_output.output_features,
            "input_features": batch_state.primary_output.input_features,
            **batch_state.primary_output.diagnostics,
        }
        result = u_mask_beam_jepa_loss(
            output,
            batch_state.labels,
            lambda_teacher=float(cfg.get("lambda_teacher", 0.5)),
            lambda_jepa_global=float(cfg.get("lambda_jepa_global", cfg.get("lambda_jepa", 1.0))),
            lambda_modality_nll=float(cfg.get("lambda_modality_nll", 1.0)),
            use_teacher=bool(cfg.get("use_teacher", True)),
            use_jepa_loss=bool(cfg.get("use_jepa_loss", True)),
            logvar_min=float(cfg.get("logvar_min", -6.0)),
            logvar_max=float(cfg.get("logvar_max", 2.0)),
            teacher_output=state.get("online_teacher"),
            prototype_bank=getattr(context.primary_model, "prototype_bank", None),
            use_beam_prototype_alignment=bool(cfg.get("use_beam_prototype_alignment", False)),
            lambda_proto=float(cfg.get("lambda_proto", 0.0)),
            lambda_modality_proto=float(cfg.get("lambda_modality_proto", 0.0)),
            lambda_supcon=float(cfg.get("lambda_supcon", 0.0)),
            lambda_teacher_proto=float(cfg.get("lambda_teacher_proto", 0.0)),
            beam_label_sigma=float(cfg.get("beam_label_sigma", 1.0)),
            beam_label_circular=bool(cfg.get("beam_label_circular", True)),
            proto_target_type=str(cfg.get("proto_target_type", "gaussian")),
            tau_beam=float(cfg.get("tau_beam", 2.0)),
            circular_beam_distance=bool(cfg.get("circular_beam_distance", cfg.get("beam_label_circular", True))),
            btapa_include_fusion=bool(cfg.get("btapa_include_fusion", True)),
            btapa_include_modalities=bool(cfg.get("btapa_include_modalities", True)),
            btapa_fusion_weight=float(cfg.get("btapa_fusion_weight", 1.0)),
            btapa_modality_weight=float(cfg.get("btapa_modality_weight", cfg.get("lambda_modality_proto", 0.0))),
            use_adba_aware_proto=bool(cfg.get("use_adba_aware_proto", False)),
            lambda_adba_proto=float(cfg.get("lambda_adba_proto", 0.0)),
            adba_margin=int(cfg.get("adba_margin", 3)),
            use_full_to_partial_kd=bool(cfg.get("use_full_to_partial_kd", False)),
            lambda_full_to_partial_kd=float(cfg.get("lambda_full_to_partial_kd", cfg.get("lambda_kd", 0.0))),
            lambda_feature_kd=float(cfg.get("lambda_feature_kd", 0.0)),
            lambda_prototype_kd=float(cfg.get("lambda_prototype_kd", 0.0)),
            kd_temperature=float(cfg.get("kd_temperature", 1.0)),
            sample_weights=_hard_pattern_weights(
                cfg,
                state.get("pattern_names"),
                batch_state.controls.model_kwargs.get("missing_mask"),
                getattr(context.primary_model, "modalities", ()),
            ),
        )
        loss = result["loss"]
        full_aux_ce = loss.sum() * 0.0
        if bool(cfg.get("use_full_aux_loss", False)):
            full_aux_ce = _full_aux_ce(context, batch_state)
            loss = loss + float(cfg.get("lambda_full_aux", 0.0)) * full_aux_ce
        auxiliary = loss - result["loss_beam"]
        diagnostics = dict(result["diagnostics"])
        diagnostics.update(
            {
                "ce_loss": float(result["loss_beam"].detach().cpu().item()),
                "beam_ce_loss": float(result["loss_beam"].detach().cpu().item()),
                "partial_ce": float(result["loss_beam"].detach().cpu().item()),
                "proto_loss": float(diagnostics.get("loss/prototype_total", 0.0)),
                "btapa_fusion_loss": float(diagnostics.get("loss/btapa_fusion", 0.0)),
                "btapa_modality_loss": float(diagnostics.get("loss/btapa_modality", 0.0)),
                "adba_proto_loss": float(diagnostics.get("loss/adba_proto", 0.0)),
                "full_aux_ce": float(full_aux_ce.detach().cpu().item()),
                "total_loss": float(loss.detach().cpu().item()),
            }
        )
        diagnostics.update(_pattern_diagnostics(state.get("pattern_names")))
        return BaseLossResult(
            total_loss=loss,
            task_loss=result["loss_beam"],
            auxiliary_loss=auxiliary,
            diagnostics=diagnostics,
        )


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
    for key, default in {
        "mask_sampler": training_cfg.get("mask_sampler", "random_missing"),
        "pattern_probs": training_cfg.get("pattern_probs"),
        "use_beam_prototype_alignment": training_cfg.get(
            "use_beam_prototype_alignment", primary_cfg.get("use_beam_prototype_alignment", False)
        ),
        "lambda_proto": training_cfg.get("lambda_proto", 0.0),
        "lambda_modality_proto": training_cfg.get("lambda_modality_proto", 0.0),
        "lambda_supcon": training_cfg.get("lambda_supcon", 0.0),
        "lambda_teacher_proto": training_cfg.get("lambda_teacher_proto", 0.0),
        "beam_proto_temperature": training_cfg.get("beam_proto_temperature", primary_cfg.get("beam_proto_temperature", 0.2)),
        "use_beam_topology_proto": training_cfg.get(
            "use_beam_topology_proto", primary_cfg.get("use_beam_topology_proto", False)
        ),
        "proto_target_type": training_cfg.get("proto_target_type"),
        "tau_beam": training_cfg.get("tau_beam", 2.0),
        "circular_beam_distance": training_cfg.get("circular_beam_distance", training_cfg.get("circular_distance")),
        "btapa_include_fusion": training_cfg.get("btapa_include_fusion", True),
        "btapa_include_modalities": training_cfg.get("btapa_include_modalities", True),
        "btapa_fusion_weight": training_cfg.get("btapa_fusion_weight", 1.0),
        "btapa_modality_weight": training_cfg.get("btapa_modality_weight"),
        "use_adba_aware_proto": training_cfg.get("use_adba_aware_proto", False),
        "lambda_adba_proto": training_cfg.get("lambda_adba_proto", 0.0),
        "adba_margin": training_cfg.get("adba_margin", 3),
        "beam_label_sigma": training_cfg.get("beam_label_sigma", 1.0),
        "beam_label_circular": training_cfg.get("beam_label_circular", True),
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
        "use_hard_pattern_weight": training_cfg.get("use_hard_pattern_weight", False),
        "hard_patterns": training_cfg.get("hard_patterns", ()),
        "hard_pattern_weight": training_cfg.get("hard_pattern_weight", 1.0),
        "hard_pattern_weight_apply_to_proto": training_cfg.get("hard_pattern_weight_apply_to_proto", False),
    }.items():
        resolved.setdefault(key, default)
    if resolved.get("proto_target_type") is None:
        resolved["proto_target_type"] = "beam_soft" if bool(resolved.get("use_beam_topology_proto", False)) else "gaussian"
    if resolved.get("circular_beam_distance") is None:
        resolved["circular_beam_distance"] = bool(resolved.get("beam_label_circular", True))
    if resolved.get("btapa_modality_weight") is None:
        resolved["btapa_modality_weight"] = resolved.get("lambda_modality_proto", 0.0)
    resolved["missing_mask"] = _resolve_missing_mask_config(resolved)
    return resolved


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


def _loss_diagnostics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    output: dict[str, torch.Tensor],
    *,
    loss_beam: torch.Tensor,
    loss_teacher: torch.Tensor,
    loss_jepa_global: torch.Tensor,
    loss_modality_nll: torch.Tensor,
    use_teacher: bool,
    use_jepa_loss: bool,
) -> dict[str, float]:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels[:, : logits.shape[1]].reshape(-1)
    top1 = flat_logits.argmax(dim=-1).eq(flat_labels).float().mean()
    k = min(5, int(flat_logits.shape[-1]))
    top5 = flat_logits.topk(k, dim=-1).indices.eq(flat_labels.unsqueeze(-1)).any(dim=-1).float().mean()
    diagnostics = {
        "loss_beam": float(loss_beam.detach().cpu().item()),
        "loss_teacher": float(loss_teacher.detach().cpu().item()),
        "loss_jepa_global": float(loss_jepa_global.detach().cpu().item()),
        "loss_modality_nll": float(loss_modality_nll.detach().cpu().item()),
        "loss_jepa": float((loss_jepa_global + loss_modality_nll).detach().cpu().item()),
        "loss/u_mask_beam": float(loss_beam.detach().cpu().item()),
        "loss/u_mask_teacher": float(loss_teacher.detach().cpu().item()),
        "loss/u_mask_jepa_global": float(loss_jepa_global.detach().cpu().item()),
        "loss/u_mask_modality_nll": float(loss_modality_nll.detach().cpu().item()),
        "loss/u_mask_jepa": float((loss_jepa_global + loss_modality_nll).detach().cpu().item()),
        "top1_acc": float(top1.detach().cpu().item()),
        "top5_acc": float(top5.detach().cpu().item()),
        "accuracy/top1": float(top1.detach().cpu().item()),
        "accuracy/top5": float(top5.detach().cpu().item()),
        "u_mask/use_teacher": float(bool(use_teacher)),
        "u_mask/use_jepa_loss": float(bool(use_jepa_loss)),
    }
    reliability = output.get("modality_reliability")
    if torch.is_tensor(reliability):
        value = float(reliability.detach().mean().cpu().item())
        diagnostics["mean_modality_reliability"] = value
        diagnostics["u_mask/mean_modality_reliability"] = value
    global_reliability = output.get("global_reliability")
    if torch.is_tensor(global_reliability):
        value = float(global_reliability.detach().mean().cpu().item())
        diagnostics["mean_global_reliability"] = value
        diagnostics["u_mask/mean_global_reliability"] = value
    return diagnostics


def _online_full_teacher(
    context: ExtensionContext,
    batch: dict[str, torch.Tensor],
    modalities: tuple[str, ...],
    labels: torch.Tensor,
    cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    del labels, cfg
    from kd_sensing.engine.runtime import run_model_step

    batch_size = next(int(value.shape[0]) for value in batch.values() if torch.is_tensor(value) and value.ndim > 0)
    full_mask = torch.ones(batch_size, len(modalities), dtype=torch.bool, device=context.device)
    with torch.no_grad():
        step = run_model_step(
            context.primary_model,
            context.task,
            batch,
            model_cfg=context.model_cfg["primary"],
            seq_length=context.seq_length,
            num_pred=context.num_pred,
            device=context.device,
            non_blocking=context.non_blocking,
            extra_model_kwargs={"missing_mask": full_mask},
        )
    return {
        "logits": step.logits.detach(),
        "output_features": step.model_output.output_features.detach(),
    }


def _full_aux_ce(context: ExtensionContext, batch_state: BatchState) -> torch.Tensor:
    from kd_sensing.engine.runtime import run_model_step

    mask = batch_state.controls.model_kwargs.get("missing_mask")
    labels = batch_state.labels
    if torch.is_tensor(mask) and bool(mask.to(dtype=torch.bool).all().item()):
        logits = batch_state.primary_logits
    else:
        batch_size = int(labels.shape[0])
        modalities = tuple(getattr(context.primary_model, "modalities", context.model_cfg.get("primary", {}).get("modalities", ())))
        full_mask = torch.ones(batch_size, len(modalities), dtype=torch.bool, device=context.device)
        step = run_model_step(
            context.primary_model,
            context.task,
            batch_state.batch,
            model_cfg=context.model_cfg["primary"],
            seq_length=context.seq_length,
            num_pred=context.num_pred,
            device=context.device,
            non_blocking=context.non_blocking,
            extra_model_kwargs={"missing_mask": full_mask},
        )
        logits = step.logits
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, : logits.shape[1]].reshape(-1))


def _hard_pattern_weights(
    cfg: dict[str, Any],
    pattern_names: list[str] | None,
    mask: torch.Tensor | None,
    modalities: tuple[str, ...],
) -> torch.Tensor | None:
    if not bool(cfg.get("use_hard_pattern_weight", False)):
        return None
    hard_patterns = {str(item) for item in cfg.get("hard_patterns", ())}
    if not hard_patterns:
        return None
    if pattern_names is None and torch.is_tensor(mask):
        pattern_names = [
            get_missing_pattern_name(row.detach().cpu(), modalities)
            for row in mask.to(dtype=torch.bool)
        ]
    if not pattern_names:
        return None
    value = float(cfg.get("hard_pattern_weight", 1.0))
    weights = [value if name in hard_patterns else 1.0 for name in pattern_names]
    return torch.tensor(weights, device=mask.device if torch.is_tensor(mask) else None)


def _logit_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, *, temperature: float) -> torch.Tensor:
    if student_logits.ndim == 3:
        student_logits = student_logits.reshape(-1, student_logits.shape[-1])
    if teacher_logits.ndim == 3:
        teacher_logits = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    temperature = float(temperature)
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)


def _top1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.ndim == 3:
        logits = logits[:, 0, :]
    if labels.ndim > 1:
        labels = labels[:, 0]
    return float(logits.argmax(dim=-1).eq(labels.to(device=logits.device)).float().mean().detach().cpu().item())


def _pattern_diagnostics(pattern_names: list[str] | None) -> dict[str, float]:
    if not pattern_names:
        return {}
    total = max(len(pattern_names), 1)
    diagnostics = {"u_mask/pattern_batch_size": float(total)}
    for name in sorted(set(pattern_names)):
        diagnostics[f"u_mask/pattern/{name}"] = float(pattern_names.count(name) / total)
    return diagnostics


__all__ = ["UMaskBeamJEPATrainingExtension", "u_mask_beam_jepa_config", "u_mask_beam_jepa_loss"]
