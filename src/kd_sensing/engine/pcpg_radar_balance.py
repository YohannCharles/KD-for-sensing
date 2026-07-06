from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.engine.training_extensions import BatchState, ExtensionContext, LossBundle, TrainingExtension


SUPPORTED_HARD_PATTERNS = {"image_only", "lidar_only", "radar_only", "missing_image", "miss3"}


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
        diagnostics: dict[str, float] = {}

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

        return LossBundle(total=total, components={"unimodal": branch_total}, diagnostics=diagnostics)


def pcpg_radar_balance_config(cfg: dict[str, Any]) -> dict[str, Any]:
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg.get("loss"), dict) else {}
    raw = loss_cfg.get("pcpg_radar_balance", {})
    base = dict(raw) if isinstance(raw, dict) else {"enabled": bool(raw)}

    branch = loss_cfg.get("branch_aux_loss", base.get("branch_aux_loss", False))
    radar = loss_cfg.get("radar_protect_loss", base.get("radar_protect_loss", False))
    hard = loss_cfg.get("hard_subset_weighting", base.get("hard_subset_weighting", False))
    use_jepa = loss_cfg.get("use_jepa", base.get("use_jepa", False))

    base["branch_aux_loss"] = _enabled(branch)
    base["radar_protect_loss"] = _enabled(radar)
    base["hard_subset_weighting"] = hard if isinstance(hard, dict) else {"enabled": bool(hard)}
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
    base["enabled"] = bool(
        base.get("enabled", False)
        or base["branch_aux_loss"]
        or base["radar_protect_loss"]
        or base["hard_subset_weighting"].get("enabled", False)
        or (base["use_jepa"] and base["jepa_weight"] > 0.0)
        or base["bprr_gate_balance_weight"] > 0.0
        or base["bprr_radar_gate_reg_weight"] > 0.0
    )
    return base


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


def pattern_name_from_available_mask(mask: torch.Tensor, modalities: list[str] | tuple[str, ...]) -> str:
    values = [bool(item) for item in mask.detach().cpu().tolist()]
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
) -> tuple[torch.Tensor, dict[str, float]]:
    diagnostics = batch_state.primary_output.diagnostics
    unimodal_logits = diagnostics.get("pcpg_unimodal_logits")
    if not torch.is_tensor(unimodal_logits):
        if bool(cfg.get("radar_protect_loss", False)) and float(cfg.get("radar_aux_weight", 0.0)) != 0.0:
            labels = batch_state.labels[:, 0] if batch_state.labels.ndim > 1 else batch_state.labels.reshape(-1)
            valid = labels.ne(-100)
            if bool(valid.any().item()):
                logits = batch_state.primary_logits[:, 0, :] if batch_state.primary_logits.ndim == 3 else batch_state.primary_logits
                fallback = F.cross_entropy(logits, labels, reduction="none")[valid].mean()
                return float(cfg.get("radar_aux_weight", 0.0)) * fallback, {
                    "loss/radar_aux": float(fallback.detach().cpu().item()),
                    "pcpg/radar_aux_fallback_used": 1.0,
                    "pcpg/branch_aux_available": 0.0,
                }
        return zero, {"pcpg/branch_aux_available": 0.0}
    mask = _available_mask(diagnostics, batch_state, unimodal_logits)
    labels = batch_state.labels[:, 0] if batch_state.labels.ndim > 1 else batch_state.labels.reshape(-1)
    valid = labels.ne(-100)
    if not bool(valid.any().item()):
        return zero, {"pcpg/branch_aux_available": 0.0}

    modalities = list(getattr(context.primary_model, "modalities", ())) or [f"modality_{i}" for i in range(unimodal_logits.shape[1])]
    losses: list[torch.Tensor] = []
    radar_loss = zero
    radar_acc = 0.0
    for index, modality in enumerate(modalities[: int(unimodal_logits.shape[1])]):
        active = valid & mask[:, index].to(device=valid.device, dtype=torch.bool)
        if not bool(active.any().item()):
            continue
        per_sample = F.cross_entropy(unimodal_logits[:, index, :], labels, reduction="none")
        loss = per_sample[active].mean()
        losses.append(loss)
        if modality == "radar":
            radar_loss = loss
            radar_acc = float(unimodal_logits[active, index, :].argmax(dim=-1).eq(labels[active]).float().mean().detach().cpu().item())
    branch_loss = torch.stack(losses).mean() if losses else zero
    total = zero
    if bool(cfg.get("branch_aux_loss", False)) and float(cfg.get("unimodal_aux_weight", 0.0)) != 0.0:
        total = total + float(cfg.get("unimodal_aux_weight", 0.0)) * branch_loss
    if bool(cfg.get("radar_protect_loss", False)) and float(cfg.get("radar_aux_weight", 0.0)) != 0.0:
        total = total + float(cfg.get("radar_aux_weight", 0.0)) * radar_loss
    return total, {
        "loss/unimodal_aux": float(branch_loss.detach().cpu().item()),
        "loss/radar_aux": float(radar_loss.detach().cpu().item()),
        "pcpg/radar_aux_accuracy": radar_acc,
        "pcpg/branch_aux_available": 1.0,
    }


def _hard_subset_extra(
    context: ExtensionContext,
    cfg: dict[str, Any],
    batch_state: BatchState,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    hard_cfg = cfg.get("hard_subset_weighting", {})
    if not isinstance(hard_cfg, dict) or not bool(hard_cfg.get("enabled", False)) or batch_state.task_loss is None:
        return zero, {}
    diagnostics = batch_state.primary_output.diagnostics
    mask = _available_mask(diagnostics, batch_state, batch_state.primary_logits)
    modalities = list(getattr(context.primary_model, "modalities", ())) or [f"modality_{i}" for i in range(mask.shape[1])]
    names = [pattern_name_from_available_mask(row, modalities) for row in mask]
    weights = torch.tensor(
        [
            static_hard_subset_weight(
                name,
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
        "loss/hard_subset_extra": float(extra.detach().cpu().item()),
        "pcpg/hard_subset_weight_mean": float(mean_weight.detach().cpu().item()),
    }


def _jepa_alignment_loss(cfg: dict[str, Any], batch_state: BatchState, zero: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
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
        "loss/jepa_latent_alignment": float(loss.detach().cpu().item()),
        "pcpg/jepa_alignment_available": 1.0,
    }


def _bprr_gate_extra(
    context: ExtensionContext,
    cfg: dict[str, Any],
    batch_state: BatchState,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
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
        "loss/bprr_gate_balance": float(raw["balance"].detach().cpu().item()),
        "loss/bprr_radar_gate": float(raw["radar"].detach().cpu().item()),
        "loss/bprr_gate_regularization": float(total.detach().cpu().item()),
        "bprr/gate_regularization_available": 1.0,
    }


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
        row_patterns = [pattern_name_from_available_mask(row, names) for row in available]
        active = torch.tensor(
            [
                bool(row[radar_index].item())
                and int(row.sum().item()) > 1
                and row_patterns[index] in hard
                and row_patterns[index] != "radar_only"
                for index, row in enumerate(available)
            ],
            device=gate.device,
            dtype=torch.bool,
        )
        if bool(active.any().item()):
            radar_loss = F.relu(float(radar_floor) - gate[active, radar_index]).mean()
        else:
            radar_loss = zero
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


def _pattern_set(value: list[str] | tuple[str, ...] | str | None) -> set[str]:
    if isinstance(value, str):
        return {item.strip().lower() for item in value.split(",") if item.strip()}
    return {str(item).strip().lower() for item in (value or []) if str(item).strip()}
