import warnings
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.data.difficulty.operators.temporal import TEMPORAL_SUPERSET_PAYLOAD_KEY
from kd_sensing.data.missing_mask import (
    get_missing_pattern_name,
    make_pattern_mask,
    sample_missing_mask,
    sample_pattern_balanced_mask,
)
from kd_sensing.engine.evaluation_pass_runtime import sample_ids_from_batch
from kd_sensing.engine.training_extensions import BaseLossResult, BatchState, ExtensionContext, ForwardControls, TrainingExtension
from kd_sensing.losses.modality_alignment_contrastive import amber_cma_analogue_loss
from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config
from kd_sensing.losses.u_mask_beam_jepa_mpdro import (
    core_pattern_names as _core_pattern_names,
    csv_float as _csv_float,
    mpdro_enabled as _mpdro_enabled,
    mpdro_sample_weights as _mpdro_sample_weights,
    new_mpdro_state as _new_mpdro_state,
    write_mpdro_group_log as _write_mpdro_group_log,
)
from kd_sensing.losses.u_mask_beam_jepa_prototype import add_prototype_alignment_losses
from kd_sensing.utils.missing_patterns import canonical_missing_pattern_name


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
    use_amber_cma_analogue: bool = False,
    lambda_amber_cma: float = 0.2,
    amber_cma_temperature: float = 0.2,
    sample_ids: list[str] | tuple[str, ...] | None = None,
    lambda_teacher_proto: float = 0.0,
    beam_label_sigma: float = 1.0,
    beam_label_circular: bool = True,
    prototype_target_circular: bool | None = None,
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
    use_superset_confidence_gated_kl: bool = False,
    lambda_superset_consistency: float = 0.0,
    superset_temperature: float = 2.0,
    use_beam_monotonic_rank: bool = False,
    lambda_beam_monotonic_rank: float = 0.0,
    beam_monotonic_tolerance: float = 0.0,
    sample_weights: torch.Tensor | None = None,
    proto_sample_weights: torch.Tensor | None = None,
    pattern_names: list[str] | None = None,
    use_pattern_conditional_btapa: bool = False,
    btapa_apply_patterns: list[str] | tuple[str, ...] | None = None,
    btapa_disable_on_patterns: list[str] | tuple[str, ...] | None = None,
    btapa_fallback_to_ordinary_proto: bool = True,
    ordinary_proto_target_type: str = "gaussian",
    apply_pattern_weight_to_proto: bool = False,
    beam_criterion: Any | None = None,
    router_supervision: str = "none",
    router_distill_weight: float = 0.0,
) -> dict[str, Any]:
    if lambda_jepa_global is None:
        lambda_jepa_global = 1.0 if lambda_jepa is None else float(lambda_jepa)
    if use_beam_prototype_alignment and use_amber_cma_analogue:
        raise ValueError(
            "use_beam_prototype_alignment and use_amber_cma_analogue are mutually exclusive; "
            "disable BPA when replacing it with the AMBER CMA analogue."
        )
    if use_amber_cma_analogue and float(lambda_amber_cma) < 0.0:
        raise ValueError("lambda_amber_cma must be non-negative.")
    logits = output["logits"]
    labels = labels.to(device=logits.device, dtype=torch.long)
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    unweighted_loss_beam = _beam_supervised_loss(logits, labels[:, : logits.shape[1]], None, beam_criterion)
    loss_beam = _beam_supervised_loss(logits, labels[:, : logits.shape[1]], sample_weights, beam_criterion)
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
    loss, proto_diag = add_prototype_alignment_losses(
        loss,
        output,
        labels,
        teacher_output=teacher_output,
        prototype_bank=prototype_bank,
        use_beam_prototype_alignment=use_beam_prototype_alignment,
        lambda_proto=lambda_proto,
        lambda_modality_proto=lambda_modality_proto,
        lambda_supcon=lambda_supcon,
        lambda_teacher_proto=lambda_teacher_proto,
        beam_label_sigma=beam_label_sigma,
        beam_label_circular=beam_label_circular,
        prototype_target_circular=prototype_target_circular,
        proto_target_type=proto_target_type,
        tau_beam=tau_beam,
        circular_beam_distance=circular_beam_distance,
        btapa_include_fusion=btapa_include_fusion,
        btapa_include_modalities=btapa_include_modalities,
        btapa_fusion_weight=btapa_fusion_weight,
        btapa_modality_weight=btapa_modality_weight,
        use_adba_aware_proto=use_adba_aware_proto,
        lambda_adba_proto=lambda_adba_proto,
        adba_margin=adba_margin,
        use_pattern_conditional_btapa=use_pattern_conditional_btapa,
        pattern_names=pattern_names,
        btapa_apply_patterns=btapa_apply_patterns,
        btapa_disable_on_patterns=btapa_disable_on_patterns,
        btapa_fallback_to_ordinary_proto=btapa_fallback_to_ordinary_proto,
        ordinary_proto_target_type=ordinary_proto_target_type,
        proto_sample_weights=proto_sample_weights if apply_pattern_weight_to_proto else None,
        kd_temperature=1.0,
    )
    diagnostics_extra.update(proto_diag)
    loss_amber_cma = zero
    if use_amber_cma_analogue:
        loss_amber_cma, cma_diag = amber_cma_analogue_loss(
            output["output_features"],
            output["modality_features"],
            output["missing_mask"],
            sample_ids,
            temperature=amber_cma_temperature,
        )
        weighted_cma = float(lambda_amber_cma) * loss_amber_cma
        loss = loss + weighted_cma
        diagnostics_extra.update(cma_diag)
        diagnostics_extra["loss/amber_cma_weighted"] = float(weighted_cma.detach().cpu().item())
    if use_superset_confidence_gated_kl or use_beam_monotonic_rank:
        if teacher_output is None or not torch.is_tensor(teacher_output.get("logits")):
            raise ValueError("Enabled superset consistency requires an online same-model superset output.")
        superset_logits = teacher_output["logits"].detach()
        if use_superset_confidence_gated_kl:
            weighted_kl, raw_kl, gate = _confidence_gated_temperature_kl(
                logits,
                superset_logits,
                labels,
                temperature=superset_temperature,
            )
            loss = loss + float(lambda_superset_consistency) * weighted_kl
            diagnostics_extra.update(
                {
                    "loss/superset_consistency": float(weighted_kl.detach().cpu().item()),
                    "superset_consistency/raw_kl": float(raw_kl.detach().cpu().item()),
                    "superset_consistency/weighted_kl": float(weighted_kl.detach().cpu().item()),
                    "superset_consistency/gate_mean": float(gate.mean().cpu().item()),
                    "superset_consistency/gate_active_ratio": float(gate.gt(0).float().mean().cpu().item()),
                    "superset_consistency/teacher_top1": _top1(superset_logits, labels),
                    "superset_consistency/student_top1": _top1(logits, labels),
                    "superset_consistency/feature_l2_weight": 0.0,
                }
            )
        if use_beam_monotonic_rank:
            rank_loss, teacher_risk, student_risk, partial_excess, superset_worse = _beam_monotonic_ranking_loss(
                logits,
                superset_logits,
                labels,
                tolerance=beam_monotonic_tolerance,
            )
            loss = loss + float(lambda_beam_monotonic_rank) * rank_loss
            risk_gap = student_risk - teacher_risk
            diagnostics_extra.update(
                {
                    "loss/beam_monotonic_rank": float(rank_loss.detach().cpu().item()),
                    "beam_monotonic_rank/teacher_risk": float(teacher_risk.mean().cpu().item()),
                    "beam_monotonic_rank/student_risk": float(student_risk.mean().detach().cpu().item()),
                    "beam_monotonic_rank/risk_gap": float(risk_gap.mean().detach().cpu().item()),
                    "beam_monotonic_rank/partial_excess_violation_rate": float(
                        partial_excess.float().mean().cpu().item()
                    ),
                    "beam_monotonic_rank/superset_worse_rate": float(
                        superset_worse.float().mean().cpu().item()
                    ),
                }
            )
    router_loss, router_diag = _hard_router_oracle_losses(
        output,
        labels,
        router_supervision=router_supervision,
        router_distill_weight=router_distill_weight,
        circular_beam_distance=(
            bool(beam_label_circular) if circular_beam_distance is None else bool(circular_beam_distance)
        ),
    )
    if torch.is_tensor(router_loss):
        loss = loss + router_loss
        diagnostics_extra.update(router_diag)
    diagnostics = _loss_diagnostics(
        logits,
        labels,
        output,
        loss_beam=loss_beam,
        unweighted_loss_beam=unweighted_loss_beam,
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
        "unweighted_loss_beam": unweighted_loss_beam,
        "loss_teacher": loss_teacher,
        "loss_jepa_global": loss_jepa_global,
        "loss_modality_nll": loss_modality_nll,
        "loss_amber_cma": loss_amber_cma,
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


def _beam_supervised_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None,
    beam_criterion: Any | None,
) -> torch.Tensor:
    if beam_criterion is not None and sample_weights is None:
        return beam_criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    return _sample_weighted_ce(logits, labels, sample_weights)


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
        return {"config": cfg, "adaptive_sampler": _new_adaptive_sampler_state(), "mpdro": _new_mpdro_state()}

    def before_forward(
        self,
        context: ExtensionContext,
        state: Any,
        batch: dict[str, torch.Tensor],
        labels: torch.Tensor,
        *,
        epoch: int,
    ) -> ForwardControls:
        cfg = state.get("config", {}) if isinstance(state, dict) else {}
        if not cfg.get("enabled", False):
            return ForwardControls()
        modalities = tuple(getattr(context.primary_model, "modalities", context.model_cfg.get("primary", {}).get("modalities", ())))
        mask_cfg = cfg.get("missing_mask", {})
        pattern = cfg.get("pattern") or cfg.get("missing_pattern")
        sampler = str(cfg.get("missing_pattern_sampler", cfg.get("mask_sampler", "default")) or "default")
        pattern_probs = _pattern_probs_for_sampler(cfg, modalities, epoch=epoch, state=state)
        if sampler in {
            "adaptive_pattern",
            "pattern_balanced",
            "uniform",
            "weak_single_oversample",
            "sensing_only_oversample",
            "missing_gps_oversample",
            "curriculum_easy_to_hard",
            "curriculum_hard_to_easy",
        }:
            mask, pattern_names, pattern_ids = sample_pattern_balanced_mask(
                int(labels.shape[0]),
                modalities,
                pattern_probs,
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
            state["pattern_names"] = [
                get_missing_pattern_name(row.detach().cpu(), modalities)
                for row in mask.to(dtype=torch.bool)
            ]
            state["pattern_ids"] = None
        _update_pattern_counts(state, state.get("pattern_names"))
        superset_cfg = cfg.get("superset_consistency", {})
        superset_active = bool(
            isinstance(superset_cfg, dict)
            and superset_cfg.get("enabled", False)
            and (superset_cfg.get("confidence_gated_kl", False) or superset_cfg.get("beam_monotonic_rank", False))
        )
        if superset_active:
            state["online_teacher"] = _online_full_teacher(
                context,
                batch,
                modalities,
                labels,
                cfg,
                use_temporal_superset=superset_active,
            )
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
        pattern_weights = _hard_pattern_weights(
            cfg,
            state.get("pattern_names"),
            batch_state.controls.model_kwargs.get("missing_mask"),
            getattr(context.primary_model, "modalities", ()),
        )
        ce_weights = pattern_weights if bool(cfg.get("apply_pattern_weight_to_ce", True)) else None
        mpdro_weights, mpdro_diagnostics = _mpdro_sample_weights(
            cfg,
            state,
            state.get("pattern_names"),
            batch_state.primary_logits,
            batch_state.labels,
            epoch=batch_state.epoch,
        )
        if ce_weights is not None and mpdro_weights is not None:
            ce_weights = ce_weights.to(device=mpdro_weights.device, dtype=mpdro_weights.dtype) * mpdro_weights
        elif mpdro_weights is not None:
            ce_weights = mpdro_weights
        proto_weights = pattern_weights if bool(cfg.get("apply_pattern_weight_to_proto", False)) else None
        superset_cfg = cfg.get("superset_consistency", {})
        superset_enabled = bool(isinstance(superset_cfg, dict) and superset_cfg.get("enabled", False))
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
            use_amber_cma_analogue=bool(cfg.get("use_amber_cma_analogue", False)),
            lambda_amber_cma=float(cfg.get("lambda_amber_cma", 0.2)),
            amber_cma_temperature=float(cfg.get("amber_cma_temperature", 0.2)),
            sample_ids=(
                sample_ids_from_batch(batch_state.batch)
                if bool(cfg.get("use_amber_cma_analogue", False))
                else None
            ),
            lambda_teacher_proto=float(cfg.get("lambda_teacher_proto", 0.0)),
            beam_label_sigma=float(cfg.get("beam_label_sigma", 1.0)),
            beam_label_circular=bool(cfg.get("beam_label_circular", True)),
            prototype_target_circular=bool(
                cfg.get("prototype_target_circular", cfg.get("beam_label_circular", True))
            ),
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
            use_superset_confidence_gated_kl=bool(
                superset_enabled and superset_cfg.get("confidence_gated_kl", False)
            ),
            lambda_superset_consistency=float(superset_cfg.get("kl_weight", 0.0)),
            superset_temperature=float(superset_cfg.get("temperature", 2.0)),
            use_beam_monotonic_rank=bool(
                superset_enabled and superset_cfg.get("beam_monotonic_rank", False)
            ),
            lambda_beam_monotonic_rank=float(superset_cfg.get("rank_weight", 0.0)),
            beam_monotonic_tolerance=float(superset_cfg.get("rank_tolerance", 0.0)),
            sample_weights=ce_weights,
            proto_sample_weights=proto_weights,
            pattern_names=state.get("pattern_names"),
            use_pattern_conditional_btapa=bool(cfg.get("use_pattern_conditional_btapa", False)),
            btapa_apply_patterns=cfg.get("btapa_apply_patterns", ()),
            btapa_disable_on_patterns=cfg.get("btapa_disable_on_patterns", ()),
            btapa_fallback_to_ordinary_proto=bool(cfg.get("btapa_fallback_to_ordinary_proto", True)),
            ordinary_proto_target_type=str(cfg.get("ordinary_proto_target_type", "gaussian")),
            apply_pattern_weight_to_proto=bool(cfg.get("apply_pattern_weight_to_proto", False)),
            beam_criterion=_u_mask_beam_criterion(context),
            router_supervision=str(cfg.get("router_supervision", "none")),
            router_distill_weight=float(cfg.get("router_distill_weight", 0.0)),
        )
        _adaptive_sampler_update(
            cfg,
            state,
            state.get("pattern_names"),
            result["unweighted_loss_beam"],
        )
        loss = result["loss"]
        full_aux_ce = loss.sum() * 0.0
        if bool(cfg.get("use_full_aux_loss", False)):
            full_aux_ce = _full_aux_ce(context, batch_state)
            loss = loss + float(cfg.get("lambda_full_aux", 0.0)) * full_aux_ce
        auxiliary = loss - result["loss_beam"]
        diagnostics = dict(result["diagnostics"])
        diagnostics.update(mpdro_diagnostics)
        diagnostics.update(
            {
                "ce_loss": float(result["unweighted_loss_beam"].detach().cpu().item()),
                "beam_ce_loss": float(result["unweighted_loss_beam"].detach().cpu().item()),
                "partial_ce": float(result["unweighted_loss_beam"].detach().cpu().item()),
                "weighted_ce_loss": float(result["loss_beam"].detach().cpu().item()),
                "avg_sample_weight": _avg_sample_weight(pattern_weights),
                "proto_loss": float(diagnostics.get("loss/prototype_total", 0.0)),
                "btapa_fusion_loss": float(diagnostics.get("loss/btapa_fusion", 0.0)),
                "btapa_modality_loss": float(diagnostics.get("loss/btapa_modality", 0.0)),
                "adba_proto_loss": float(diagnostics.get("loss/adba_proto", 0.0)),
                "full_aux_ce": float(full_aux_ce.detach().cpu().item()),
                "total_loss": float(loss.detach().cpu().item()),
            }
        )
        diagnostics.update(_pattern_diagnostics(state.get("pattern_names")))
        diagnostics.update(_adaptive_sampler_diagnostics(state))
        return BaseLossResult(
            total_loss=loss,
            task_loss=result["loss_beam"],
            auxiliary_loss=auxiliary,
            diagnostics=diagnostics,
        )
    def before_epoch(self, context: ExtensionContext, state: Any, *, epoch: int) -> None:
        if isinstance(state, dict):
            state["pattern_epoch_counts"] = Counter()
            state["mpdro_epoch_batches"] = Counter()
            _prepare_adaptive_sampler_epoch(context, state, epoch=epoch)

    def after_epoch(self, context: ExtensionContext, state: Any, *, epoch: int) -> dict[str, Any]:
        if not isinstance(state, dict):
            return {}
        cfg = state.get("config", {})
        counts = state.get("pattern_epoch_counts")
        if not isinstance(cfg, dict):
            return {}
        sampler = str(cfg.get("missing_pattern_sampler", cfg.get("mask_sampler", "default")) or "default")
        metrics: dict[str, Any] = {}
        if isinstance(counts, Counter) and counts and sampler not in {"default", "random_missing", "adaptive_pattern"}:
            path = _write_pattern_counts(context.cfg, counts, epoch=epoch)
            metrics["pattern_sampling"] = {"path": str(path), "sampler": sampler}
        if sampler == "adaptive_pattern":
            path = _write_adaptive_sampler_log(context, state, epoch=epoch)
            metrics["adaptive_sampler"] = {"path": str(path)}
        if _mpdro_enabled(cfg):
            path = _write_mpdro_group_log(context, state, epoch=epoch)
            metrics["mpdro"] = {"path": str(path)}
        return metrics
def _loss_diagnostics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    output: dict[str, torch.Tensor],
    *,
    loss_beam: torch.Tensor,
    unweighted_loss_beam: torch.Tensor,
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
        "ce_loss": float(unweighted_loss_beam.detach().cpu().item()),
        "weighted_ce_loss": float(loss_beam.detach().cpu().item()),
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
    *,
    use_temporal_superset: bool = False,
) -> dict[str, torch.Tensor]:
    del labels, cfg
    from kd_sensing.engine.runtime import run_model_step

    teacher_batch = batch
    if use_temporal_superset:
        teacher_batch, teacher_mask = _restore_temporal_superset(batch, modalities, context.device)
    else:
        batch_size = next(int(value.shape[0]) for value in batch.values() if torch.is_tensor(value) and value.ndim > 0)
        teacher_mask = torch.ones(batch_size, len(modalities), dtype=torch.bool, device=context.device)
    module_states = [(module, module.training) for module in context.primary_model.modules()]
    try:
        context.primary_model.eval()
        with torch.no_grad():
            step = run_model_step(
                context.primary_model,
                context.task,
                teacher_batch,
                model_cfg=context.model_cfg["primary"],
                seq_length=context.seq_length,
                num_pred=context.num_pred,
                device=context.device,
                non_blocking=context.non_blocking,
                extra_model_kwargs={"missing_mask": teacher_mask},
            )
    finally:
        for module, training in module_states:
            module.training = training
    result = {"logits": step.logits.detach()}
    if torch.is_tensor(step.model_output.output_features):
        result["output_features"] = step.model_output.output_features.detach()
    return result


def _restore_temporal_superset(
    batch: dict[str, Any],
    modalities: tuple[str, ...],
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    payload = batch.get(TEMPORAL_SUPERSET_PAYLOAD_KEY)
    if not isinstance(payload, dict):
        raise ValueError(
            "Temporal superset consistency requires temporal_missing.preserve_unmasked_for_superset=true."
        )
    inputs = payload.get("inputs")
    base_mask = payload.get("base_mask")
    payload_modalities = tuple(payload.get("modalities", ()))
    if not isinstance(inputs, dict) or not torch.is_tensor(base_mask):
        raise ValueError("Temporal superset payload must contain input tensor references and a base mask.")
    if payload_modalities != modalities:
        raise ValueError(
            f"Temporal superset modalities {payload_modalities} do not match model modalities {modalities}."
        )
    base_mask = base_mask.to(dtype=torch.bool)
    student_mask = batch.get("modality_temporal_mask")
    if not torch.is_tensor(student_mask) or tuple(student_mask.shape) != tuple(base_mask.shape):
        raise ValueError("Temporal superset and student masks must have matching [B,T,M] shapes.")
    student_mask = student_mask.to(device=base_mask.device, dtype=torch.bool)
    if bool((student_mask & ~base_mask).any().item()):
        raise ValueError("Temporal student mask must be a subset of the preserved superset mask.")
    if not bool(student_mask.any(dim=(1, 2)).all().item()) or not bool(base_mask.any(dim=(1, 2)).all().item()):
        raise ValueError("Temporal student and superset masks must retain at least one history cell per sample.")
    teacher_batch = dict(batch)
    teacher_batch.update(inputs)
    teacher_batch["modality_temporal_mask"] = base_mask
    teacher_batch["temporal_mask"] = base_mask.any(dim=2)
    teacher_batch["available_modalities"] = base_mask.any(dim=1)
    for index, modality in enumerate(modalities):
        valid = base_mask[:, :, index]
        teacher_batch[f"{modality}_valid_mask"] = valid
        teacher_batch[f"{modality}_dropout_mask"] = ~valid
        teacher_batch[f"{modality}_missing_mask"] = ~valid
    return teacher_batch, base_mask.any(dim=1).to(device=device)


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
    pattern_weights = {
        canonical_missing_pattern_name(key): float(value)
        for key, value in dict(cfg.get("pattern_loss_weights", {}) or {}).items()
    }
    hard_patterns = {canonical_missing_pattern_name(item) for item in cfg.get("hard_patterns", ())}
    if not pattern_weights and not hard_patterns:
        return None
    if pattern_names is None and torch.is_tensor(mask):
        pattern_names = [
            get_missing_pattern_name(row.detach().cpu(), modalities)
            for row in mask.to(dtype=torch.bool)
        ]
    if not pattern_names:
        return None
    value = float(cfg.get("hard_pattern_weight", 1.0))
    weights = []
    for name in pattern_names:
        canonical = canonical_missing_pattern_name(name)
        weights.append(pattern_weights.get(canonical, value if canonical in hard_patterns else 1.0))
    return torch.tensor(weights, device=mask.device if torch.is_tensor(mask) else None)


def _avg_sample_weight(sample_weights: torch.Tensor | None) -> float:
    if sample_weights is None:
        return 1.0
    return float(sample_weights.detach().float().mean().cpu().item())


def _u_mask_beam_criterion(context: ExtensionContext):
    loss_cfg = context.cfg.get("loss", {}) if isinstance(context.cfg.get("loss"), dict) else {}
    loss_type = str(loss_cfg.get("type", "cross_entropy"))
    if loss_type in {"beam_neighborhood_ce", "label_smoothing_ce"}:
        return context.task_criterion
    return None


def _logit_kd_loss_per_sample(student_logits: torch.Tensor, teacher_logits: torch.Tensor, *, temperature: float) -> torch.Tensor:
    if student_logits.ndim == 2:
        student_logits = student_logits.unsqueeze(1)
    if teacher_logits.ndim == 2:
        teacher_logits = teacher_logits.unsqueeze(1)
    if tuple(student_logits.shape) != tuple(teacher_logits.shape):
        raise ValueError(
            "student and teacher logits must have matching shape for superset consistency, "
            f"got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}."
        )
    temperature = float(temperature)
    per_slot = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return per_slot.mean(dim=1) * (temperature**2)


def _confidence_gated_temperature_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("Superset consistency temperature must be positive.")
    per_sample = _logit_kd_loss_per_sample(student_logits, teacher_logits, temperature=temperature)
    if teacher_logits.ndim == 2:
        teacher_logits = teacher_logits.unsqueeze(1)
    targets = labels.unsqueeze(1) if labels.ndim == 1 else labels
    targets = targets[:, : teacher_logits.shape[1]].to(device=teacher_logits.device, dtype=torch.long)
    if tuple(targets.shape) != tuple(teacher_logits.shape[:2]):
        raise ValueError("Superset teacher logits and labels must have matching batch/prediction dimensions.")
    valid = targets.ne(-100)
    safe_targets = targets.masked_fill(~valid, 0)
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    entropy = -(teacher_prob * teacher_prob.clamp_min(torch.finfo(teacher_prob.dtype).tiny).log()).sum(dim=-1)
    if int(teacher_logits.shape[-1]) <= 1:
        raise ValueError("Superset consistency requires at least two beam classes.")
    normalized_entropy = entropy / math.log(int(teacher_logits.shape[-1]))
    mean_entropy = (normalized_entropy * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    correct = ((teacher_logits.detach().argmax(dim=-1) == safe_targets) | ~valid).all(dim=1) & valid.any(dim=1)
    gate = (correct.to(dtype=teacher_prob.dtype) * (1.0 - mean_entropy).clamp(0.0, 1.0)).detach()
    weighted = (per_sample * gate).sum() / gate.sum().clamp_min(torch.finfo(per_sample.dtype).eps)
    return weighted, per_sample.mean(), gate


def _circular_beam_risk(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    if logits.ndim != 3:
        raise ValueError(f"Beam logits must have shape [B,P,C], got {tuple(logits.shape)}.")
    targets = labels.unsqueeze(1) if labels.ndim == 1 else labels
    targets = targets[:, : logits.shape[1]].to(device=logits.device, dtype=torch.long)
    if tuple(targets.shape) != tuple(logits.shape[:2]):
        raise ValueError("Beam logits and labels must have matching batch/prediction dimensions.")
    valid = targets.ne(-100)
    safe_targets = targets.masked_fill(~valid, 0)
    classes = int(logits.shape[-1])
    beam_ids = torch.arange(classes, device=logits.device).view(1, 1, classes)
    distance = (beam_ids - safe_targets.unsqueeze(-1)).abs()
    circular_distance = torch.minimum(distance, classes - distance).to(dtype=logits.dtype)
    per_slot = (F.softmax(logits, dim=-1) * circular_distance).sum(dim=-1)
    return (per_slot * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def _beam_monotonic_ranking_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if tuple(student_logits.shape) != tuple(teacher_logits.shape):
        raise ValueError("Student and superset logits must have matching shapes for beam monotonic ranking.")
    teacher_risk = _circular_beam_risk(teacher_logits.detach(), labels).detach()
    student_risk = _circular_beam_risk(student_logits, labels)
    partial_excess = student_risk - teacher_risk - float(tolerance)
    superset_worse = teacher_risk > student_risk
    return (
        F.relu(partial_excess).mean(),
        teacher_risk,
        student_risk,
        partial_excess.detach().gt(0),
        superset_worse.detach(),
    )


def _top1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.ndim == 3:
        logits = logits[:, 0, :]
    if labels.ndim > 1:
        labels = labels[:, 0]
    return float(logits.argmax(dim=-1).eq(labels.to(device=logits.device)).float().mean().detach().cpu().item())


def _hard_router_oracle_losses(
    output: dict[str, Any],
    labels: torch.Tensor,
    *,
    router_supervision: str,
    router_distill_weight: float,
    circular_beam_distance: bool = True,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if str(router_supervision).strip().lower() != "oracle":
        return None, {}
    terms: list[torch.Tensor] = []
    diagnostics: dict[str, float] = {}
    modality_weight = float(router_distill_weight)
    if (
        modality_weight != 0.0
        and torch.is_tensor(output.get("router_gate_logits"))
        and torch.is_tensor(output.get("unimodal_logits"))
    ):
        loss, diag = _modality_oracle_ce(
            output["router_gate_logits"],
            output["unimodal_logits"],
            labels,
            output.get("missing_mask"),
            prefix="router",
            circular_beam_distance=circular_beam_distance,
        )
        terms.append(modality_weight * loss)
        diagnostics.update(diag)
    if not terms:
        return None, {}
    total = sum(terms)
    diagnostics["loss/router_oracle_total"] = float(total.detach().cpu().item())
    diagnostics["router_oracle_hard_target_fallback"] = 1.0
    return total, diagnostics


def _modality_oracle_ce(
    gate_logits: torch.Tensor,
    unimodal_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    prefix: str,
    circular_beam_distance: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if mask is None:
        mask = torch.ones(gate_logits.shape, dtype=torch.bool, device=gate_logits.device)
    mask = mask.to(device=gate_logits.device, dtype=torch.bool)
    targets = _oracle_argmin(
        unimodal_logits,
        labels,
        mask,
        circular_beam_distance=circular_beam_distance,
    )
    active = targets.ne(-100) & mask.sum(dim=-1).gt(1)
    if not bool(active.any().item()):
        zero = gate_logits.sum() * 0.0
        return zero, {f"loss/{prefix}_oracle": 0.0, f"{prefix}_oracle_active_ratio": 0.0}
    masked_gate = gate_logits.masked_fill(~mask, torch.finfo(gate_logits.dtype).min)
    loss = F.cross_entropy(masked_gate[active], targets[active])
    pred = masked_gate.argmax(dim=-1)
    acc = pred[active].eq(targets[active]).float().mean()
    return loss, {
        f"loss/{prefix}_oracle": float(loss.detach().cpu().item()),
        f"{prefix}_oracle_acc": float(acc.detach().cpu().item()),
        f"{prefix}_oracle_active_ratio": float(active.float().mean().detach().cpu().item()),
    }


def _oracle_argmin(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    circular_beam_distance: bool = True,
) -> torch.Tensor:
    target = labels.to(device=logits.device, dtype=torch.long)
    if target.ndim > 1:
        target = target[:, 0]
    pred = logits.argmax(dim=-1)
    while target.ndim < pred.ndim:
        target = target.unsqueeze(-1)
    diff = (pred - target).abs()
    if circular_beam_distance:
        diff = torch.minimum(diff, int(logits.shape[-1]) - diff)
    errors = diff.to(dtype=logits.dtype)
    available = mask.to(device=logits.device, dtype=torch.bool)
    masked = errors.masked_fill(~available, torch.finfo(errors.dtype).max)
    oracle = masked.argmin(dim=-1)
    return torch.where(available.any(dim=-1), oracle, torch.full_like(oracle, -100))


def _pattern_diagnostics(pattern_names: list[str] | None) -> dict[str, float]:
    if not pattern_names:
        return {}
    total = max(len(pattern_names), 1)
    diagnostics = {"u_mask/pattern_batch_size": float(total)}
    for name in sorted(set(pattern_names)):
        diagnostics[f"u_mask/pattern/{name}"] = float(pattern_names.count(name) / total)
    return diagnostics


def _new_adaptive_sampler_state() -> dict[str, Any]:
    return {
        "ema_loss": {},
        "ema_acc": {},
        "num_samples": Counter(),
        "last_probs": {},
        "last_scores": {},
        "warnings": set(),
    }


def _pattern_probs_for_sampler(
    cfg: dict[str, Any],
    modalities: tuple[str, ...],
    *,
    epoch: int,
    state: Any | None = None,
) -> dict[str, float] | list[str] | tuple[str, ...] | None:
    sampler = str(cfg.get("missing_pattern_sampler", cfg.get("mask_sampler", "default")) or "default")
    if sampler == "pattern_balanced":
        return cfg.get("pattern_probs")
    if sampler == "adaptive_pattern":
        if isinstance(state, dict) and isinstance(state.get("adaptive_current_probs"), dict):
            return state["adaptive_current_probs"]
        if isinstance(state, dict):
            return _adaptive_pattern_probs(cfg, state, modalities, epoch=epoch)
        return _uniform_pattern_probs(modalities)
    if sampler in {"default", "random_missing"}:
        return None
    if sampler in {"curriculum_easy_to_hard", "curriculum_hard_to_easy"}:
        scheduled = _scheduled_patterns(cfg.get("curriculum_schedule", {}), epoch=epoch)
        return scheduled or _core_pattern_names(modalities)
    weights = {name: 1.0 for name in _core_pattern_names(modalities)}
    for name, value in dict(cfg.get("pattern_sampling_weights", {}) or {}).items():
        canonical = canonical_missing_pattern_name(name)
        if canonical in weights:
            weights[canonical] = float(value)
    return weights


def _prepare_adaptive_sampler_epoch(context: ExtensionContext, state: dict[str, Any], *, epoch: int) -> None:
    cfg = state.get("config", {})
    if str(cfg.get("missing_pattern_sampler", cfg.get("mask_sampler", "default")) or "default") != "adaptive_pattern":
        return
    modalities = tuple(getattr(context.primary_model, "modalities", context.model_cfg.get("primary", {}).get("modalities", ())))
    state["adaptive_current_probs"] = _adaptive_pattern_probs(cfg, state, modalities, epoch=epoch)


def _adaptive_pattern_probs(
    cfg: dict[str, Any],
    state: dict[str, Any],
    modalities: tuple[str, ...],
    *,
    epoch: int,
) -> dict[str, float]:
    adaptive = _adaptive_state(state)
    patterns = _core_pattern_names(modalities)
    uniform = _uniform_pattern_probs(modalities)
    scores = {pattern: 0.0 for pattern in patterns}
    if not patterns:
        _adaptive_warn(adaptive, "no core missing patterns are available; using uniform probabilities.")
        return uniform
    sampler_cfg = _adaptive_cfg(cfg)
    if int(epoch) < int(sampler_cfg["warmup_epochs"]):
        adaptive["last_scores"] = scores
        adaptive["last_probs"] = uniform
        return uniform
    score_mode = str(sampler_cfg["score_mode"]).lower()
    if score_mode == "acc_gap":
        _adaptive_warn(adaptive, "score_mode=acc_gap is configured but pattern-wise training accuracy is unavailable; using uniform probabilities.")
        adaptive["last_scores"] = scores
        adaptive["last_probs"] = uniform
        return uniform
    ema_loss = adaptive.get("ema_loss", {})
    missing = [pattern for pattern in patterns if pattern not in ema_loss]
    if missing:
        _adaptive_warn(adaptive, f"missing EMA loss for patterns {missing}; using uniform probabilities.")
        adaptive["last_scores"] = scores
        adaptive["last_probs"] = uniform
        return uniform
    if score_mode == "gap_to_full":
        if "full" not in ema_loss:
            _adaptive_warn(adaptive, "missing full EMA loss for gap_to_full; using uniform probabilities.")
            adaptive["last_scores"] = scores
            adaptive["last_probs"] = uniform
            return uniform
        full_loss = float(ema_loss["full"])
        scores = {pattern: max(0.0, float(ema_loss[pattern]) - full_loss) for pattern in patterns}
    elif score_mode == "loss":
        scores = {pattern: float(ema_loss[pattern]) for pattern in patterns}
    else:
        _adaptive_warn(adaptive, f"unknown score_mode={score_mode!r}; using uniform probabilities.")
        adaptive["last_scores"] = scores
        adaptive["last_probs"] = uniform
        return uniform
    score_values = torch.tensor([scores[pattern] for pattern in patterns], dtype=torch.float32)
    hard = torch.softmax(score_values / float(sampler_cfg["temperature"]), dim=0)
    alpha = float(sampler_cfg["alpha"])
    raw = torch.tensor([uniform[pattern] for pattern in patterns], dtype=torch.float32) * (1.0 - alpha) + hard * alpha
    clipped = raw.clamp(float(sampler_cfg["min_prob"]), float(sampler_cfg["max_prob"]))
    normalized = clipped / clipped.sum().clamp_min(1e-12)
    probs = {pattern: float(value) for pattern, value in zip(patterns, normalized.tolist())}
    adaptive["last_scores"] = scores
    adaptive["last_probs"] = probs
    return probs


def _adaptive_sampler_update(
    cfg: dict[str, Any],
    state: Any,
    pattern_names: list[str] | None,
    loss: torch.Tensor,
) -> None:
    if not isinstance(state, dict) or not pattern_names:
        return
    if str(cfg.get("missing_pattern_sampler", cfg.get("mask_sampler", "default")) or "default") != "adaptive_pattern":
        return
    if str(cfg.get("adaptive_update_freq", "step")) != "step":
        return
    adaptive = _adaptive_state(state)
    value = float(loss.detach().float().cpu().item())
    beta = float(_adaptive_cfg(cfg)["ema_beta"])
    counts = Counter(canonical_missing_pattern_name(name) for name in pattern_names)
    for pattern, count in counts.items():
        previous = adaptive["ema_loss"].get(pattern)
        adaptive["ema_loss"][pattern] = value if previous is None else beta * float(previous) + (1.0 - beta) * value
        adaptive["num_samples"][pattern] += int(count)


def _write_adaptive_sampler_log(context: ExtensionContext, state: dict[str, Any], *, epoch: int) -> Path:
    adaptive = _adaptive_state(state)
    modalities = tuple(getattr(context.primary_model, "modalities", context.model_cfg.get("primary", {}).get("modalities", ())))
    patterns = _core_pattern_names(modalities)
    probs = dict(adaptive.get("last_probs") or state.get("adaptive_current_probs") or _uniform_pattern_probs(modalities))
    scores = dict(adaptive.get("last_scores") or {pattern: 0.0 for pattern in patterns})
    counts = state.get("pattern_epoch_counts")
    if not isinstance(counts, Counter):
        counts = Counter()
    path = context.run_dir / "adaptive_sampler_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "pattern", "ema_loss", "ema_acc", "score", "sampling_prob", "num_samples"],
        )
        if write_header:
            writer.writeheader()
        for pattern in patterns:
            writer.writerow(
                {
                    "epoch": int(epoch) + 1,
                    "pattern": pattern,
                    "ema_loss": _csv_float(adaptive["ema_loss"].get(pattern)),
                    "ema_acc": _csv_float(adaptive["ema_acc"].get(pattern)),
                    "score": _csv_float(scores.get(pattern, 0.0)),
                    "sampling_prob": _csv_float(probs.get(pattern, 0.0)),
                    "num_samples": int(counts.get(pattern, 0)),
                }
            )
    summary = " ".join(f"{pattern}={probs.get(pattern, 0.0):.3f}" for pattern in patterns)
    print(f"[AdaptiveSampler] epoch={int(epoch) + 1} probs: {summary}")
    return path


def _adaptive_sampler_diagnostics(state: Any) -> dict[str, float]:
    if not isinstance(state, dict):
        return {}
    adaptive = state.get("adaptive_sampler")
    if not isinstance(adaptive, dict):
        return {}
    diagnostics = {}
    for pattern, prob in dict(adaptive.get("last_probs") or {}).items():
        diagnostics[f"adaptive_sampler/prob/{pattern}"] = float(prob)
    return diagnostics


def _adaptive_state(state: dict[str, Any]) -> dict[str, Any]:
    adaptive = state.setdefault("adaptive_sampler", _new_adaptive_sampler_state())
    adaptive.setdefault("ema_loss", {})
    adaptive.setdefault("ema_acc", {})
    adaptive.setdefault("num_samples", Counter())
    adaptive.setdefault("last_probs", {})
    adaptive.setdefault("last_scores", {})
    adaptive.setdefault("warnings", set())
    return adaptive


def _adaptive_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    nested = cfg.get("adaptive_sampler", {})
    if not isinstance(nested, dict):
        nested = {}

    def value(name: str, default: Any) -> Any:
        return cfg.get(f"adaptive_{name}", nested.get(name, default))

    temperature = max(float(value("temperature", 1.0)), 1e-6)
    return {
        "alpha": min(max(float(value("alpha", 0.5)), 0.0), 1.0),
        "temperature": temperature,
        "ema_beta": min(max(float(value("ema_beta", 0.9)), 0.0), 0.9999),
        "score_mode": str(value("score_mode", "gap_to_full")),
        "min_prob": max(float(value("min_prob", 0.05)), 0.0),
        "max_prob": min(max(float(value("max_prob", 0.40)), 0.0), 1.0),
        "warmup_epochs": max(int(value("warmup_epochs", 3)), 0),
    }


def _uniform_pattern_probs(modalities: tuple[str, ...]) -> dict[str, float]:
    patterns = _core_pattern_names(modalities)
    if not patterns:
        return {}
    prob = 1.0 / len(patterns)
    return {pattern: prob for pattern in patterns}


def _adaptive_warn(adaptive: dict[str, Any], message: str) -> None:
    warnings_seen = adaptive.setdefault("warnings", set())
    if message in warnings_seen:
        return
    warnings.warn(f"[AdaptiveSampler] {message}", UserWarning, stacklevel=3)
    warnings_seen.add(message)


def _scheduled_patterns(schedule: Any, *, epoch: int) -> list[str]:
    if not isinstance(schedule, dict):
        return []
    epoch_number = int(epoch) + 1
    for key, raw_patterns in schedule.items():
        text = str(key)
        if not text.startswith("epochs_"):
            continue
        bounds = text.removeprefix("epochs_").split("_")
        if len(bounds) != 2:
            continue
        start, end = int(bounds[0]), int(bounds[1])
        if start <= epoch_number <= end:
            return [canonical_missing_pattern_name(item) for item in raw_patterns]
    return []


def _update_pattern_counts(state: Any, pattern_names: list[str] | None) -> None:
    if not isinstance(state, dict) or not pattern_names:
        return
    counts = state.setdefault("pattern_epoch_counts", Counter())
    if isinstance(counts, Counter):
        counts.update(pattern_names)


def _write_pattern_counts(cfg: dict[str, Any], counts: Counter, *, epoch: int) -> Path:
    run_name = str(cfg.get("output", {}).get("run_name") or cfg.get("experiment", {}).get("name") or "run")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in run_name)
    out_dir = Path("outputs/scene31/analysis/pattern_sampling_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{safe}_pattern_counts.csv"
    write_header = not path.exists()
    total = sum(int(value) for value in counts.values())
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "pattern", "count", "ratio"])
        if write_header:
            writer.writeheader()
        for pattern, count in sorted(counts.items()):
            writer.writerow(
                {
                    "epoch": int(epoch) + 1,
                    "pattern": pattern,
                    "count": int(count),
                    "ratio": float(count / max(total, 1)),
                }
            )
    return path


__all__ = ["UMaskBeamJEPATrainingExtension", "u_mask_beam_jepa_config", "u_mask_beam_jepa_loss"]
