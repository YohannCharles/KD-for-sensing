from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.engine.model_output import ModelOutput
from kd_sensing.modalities import MODALITY_ORDER, normalize_modalities
from kd_sensing.registries import DISTILLERS

from .g2d_smp import SMPScheduler


DEFAULT_HORIZON_NAMES = ("t+1", "t+2", "t+3")
DEFAULT_STUDENT_FEATURE_DIMS = {
    "image": 192,
    "radar": 192,
    "gps": 96,
    "lidar": 192,
    "mmwave": 96,
    "csi": 64,
}


@dataclass
class G2DStepResult:
    total_loss: torch.Tensor
    supervised_loss: torch.Tensor
    feature_kd_loss: torch.Tensor
    logit_kd_loss: torch.Tensor
    distill_loss: torch.Tensor
    diagnostics: dict[str, Any]
    active_modalities: list[str]
    modality_ranking: dict[str, list[str]]
    teacher_confidence: dict[str, torch.Tensor]


@DISTILLERS.register("g2d")
class G2DDistiller(nn.Module):
    def __init__(
        self,
        task_criterion: nn.Module,
        *,
        g2d: dict[str, Any] | None = None,
        mode: str | None = None,
        num_pred: int = 3,
        num_classes: int = 64,
        feature_size: int = 64,
        modalities: list[str] | tuple[str, ...] | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        del kwargs
        cfg = dict(g2d or {})
        self.task_criterion = task_criterion
        self.mode = str(mode or cfg.get("mode", "lite"))
        if self.mode == "horizon":
            self.mode = "horizon_diagnostic"
        if self.mode not in {"lite", "global", "horizon_diagnostic"}:
            raise ValueError("G2D mode must be one of lite, global, horizon_diagnostic.")
        self.modalities = list(normalize_modalities(tuple(modalities or cfg.get("modalities") or MODALITY_ORDER), context="G2D modalities"))
        self.num_pred = int(cfg.get("num_pred", num_pred))
        self.num_classes = int(cfg.get("num_classes", num_classes))
        self.horizon_names = horizon_names(self.num_pred)

        loss_cfg = dict(cfg.get("loss") or {})
        self.supervised_weight = float(loss_cfg.get("supervised_weight", cfg.get("supervised_weight", 1.0)))
        self.feature_weight = float(loss_cfg.get("feature_weight", cfg.get("feature_weight", 0.1)))
        self.logit_weight = float(loss_cfg.get("logit_weight", cfg.get("logit_weight", 0.5)))
        self.temperature = float(loss_cfg.get("temperature", cfg.get("temperature", 4.0)))
        self.ignore_index = int(loss_cfg.get("ignore_index", getattr(task_criterion, "ignore_index", -100)))
        self.horizons = _resolve_horizons(loss_cfg.get("horizons", cfg.get("horizons", "all")), self.num_pred)

        feature_cfg = dict(loss_cfg.get("feature_align") or cfg.get("feature_align") or {})
        self.feature_enabled = bool(feature_cfg.get("enabled", self.feature_weight > 0.0))
        self.feature_pool = str(feature_cfg.get("pool", "last"))
        self.feature_source = str(feature_cfg.get("source", "auto"))
        self.feature_normalize = bool(feature_cfg.get("normalize", True))
        self.projection_mode = str(feature_cfg.get("projection", "auto"))
        self.target_feature_dim = int(feature_cfg.get("projection_dim", feature_size))
        self.projections = nn.ModuleDict()
        student_dims = {**DEFAULT_STUDENT_FEATURE_DIMS, **dict(feature_cfg.get("student_dims") or {})}
        if self.projection_mode == "auto":
            for modality in self.modalities:
                input_dim = int(student_dims.get(modality, self.target_feature_dim))
                if input_dim != self.target_feature_dim:
                    self.projections[modality] = nn.Linear(input_dim, self.target_feature_dim)

        logit_cfg = dict(loss_cfg.get("logit_align") or cfg.get("logit_align") or {})
        self.logit_enabled = bool(logit_cfg.get("enabled", self.logit_weight > 0.0))

        smp_cfg = dict(cfg.get("smp") or {})
        self.smp_enabled = bool(smp_cfg.get("enabled", self.mode == "global"))
        self.smp_scheduler = SMPScheduler(
            self.modalities,
            per_modality_tau=int((smp_cfg.get("tau") or {}).get("per_modality", smp_cfg.get("per_modality_tau", 5))),
            joint_tau=int((smp_cfg.get("tau") or {}).get("joint", smp_cfg.get("joint_tau", 30))),
            prioritize_low_confidence_first=bool(smp_cfg.get("prioritize_low_confidence_first", True)),
        )

    def compute(
        self,
        student_output: ModelOutput,
        teacher_outputs: Mapping[str, ModelOutput],
        labels: torch.Tensor,
        *,
        epoch: int = 0,
    ) -> G2DStepResult:
        student_logits = validate_logits(
            student_output.logits,
            num_pred=self.num_pred,
            num_classes=self.num_classes,
            name="student logits",
        )
        labels = validate_labels(labels, num_pred=self.num_pred, batch_size=student_logits.shape[0])
        selected_student = _select_horizons(student_logits, self.horizons)
        selected_labels = _select_label_horizons(labels, self.horizons)
        supervised_loss = F.cross_entropy(
            selected_student.reshape(-1, selected_student.shape[-1]),
            selected_labels.reshape(-1),
            ignore_index=self.ignore_index,
        )

        teacher_logits: dict[str, torch.Tensor] = {}
        teacher_confidence: dict[str, torch.Tensor] = {}
        for modality in self.modalities:
            if modality not in teacher_outputs:
                raise KeyError(f"G2D teacher output missing modality '{modality}'.")
            logits = validate_logits(
                teacher_outputs[modality].logits,
                num_pred=self.num_pred,
                num_classes=self.num_classes,
                name=f"{modality} teacher logits",
            ).detach()
            teacher_logits[modality] = logits
            teacher_confidence[modality] = teacher_confidence_from_logits(
                logits,
                labels,
                num_pred=self.num_pred,
                modality=modality,
                ignore_index=self.ignore_index,
            )

        logit_kd_loss = self._logit_kd_loss(selected_student, teacher_logits) if self.logit_enabled else _zero_like(supervised_loss)
        feature_kd_loss = self._feature_kd_loss(student_output, teacher_outputs) if self.feature_enabled else _zero_like(supervised_loss)
        total_loss = (
            self.supervised_weight * supervised_loss
            + self.feature_weight * feature_kd_loss
            + self.logit_weight * logit_kd_loss
        )
        distill_loss = self.feature_weight * feature_kd_loss + self.logit_weight * logit_kd_loss

        ranking = modality_rankings(teacher_confidence, self.horizon_names)
        confidence_avg = {name: float(values.mean().detach().cpu().item()) for name, values in teacher_confidence.items()}
        active_modalities = (
            self.smp_scheduler.active_modalities(epoch, confidence_avg)
            if self.smp_enabled
            else list(self.modalities)
        )
        student_branch_conf = student_branch_confidence_from_output(
            student_output,
            labels,
            self.modalities,
            self.horizon_names,
            ignore_index=self.ignore_index,
        )
        diagnostics = {
            "num_pred": self.num_pred,
            "horizon_names": list(self.horizon_names),
            "teacher_confidence": confidence_to_json(teacher_confidence, self.horizon_names),
            "modality_ranking_weak_to_strong": ranking,
            "active_modalities": list(active_modalities),
            "loss": {
                "supervised": float(supervised_loss.detach().cpu().item()),
                "feature_kd": float(feature_kd_loss.detach().cpu().item()),
                "logit_kd": float(logit_kd_loss.detach().cpu().item()),
                "distill": float(distill_loss.detach().cpu().item()),
                "total": float(total_loss.detach().cpu().item()),
            },
        }
        if student_branch_conf:
            diagnostics["student_branch_confidence"] = student_branch_conf
            diagnostics["confidence_ratio"] = confidence_ratio(student_branch_conf, diagnostics["teacher_confidence"])
        return G2DStepResult(
            total_loss=total_loss,
            supervised_loss=supervised_loss,
            feature_kd_loss=feature_kd_loss,
            logit_kd_loss=logit_kd_loss,
            distill_loss=distill_loss,
            diagnostics=diagnostics,
            active_modalities=list(active_modalities),
            modality_ranking=ranking,
            teacher_confidence=teacher_confidence,
        )

    def _logit_kd_loss(self, student_logits: torch.Tensor, teacher_logits: Mapping[str, torch.Tensor]) -> torch.Tensor:
        losses = []
        flat_student = student_logits.reshape(-1, student_logits.shape[-1])
        for modality in self.modalities:
            selected_teacher = _select_horizons(teacher_logits[modality], self.horizons).reshape(-1, student_logits.shape[-1])
            student_log_prob = F.log_softmax(flat_student / self.temperature, dim=-1)
            teacher_prob = F.softmax(selected_teacher.detach() / self.temperature, dim=-1)
            losses.append(F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (self.temperature ** 2))
        return torch.stack(losses).mean() if losses else _zero_like(student_logits)

    def _feature_kd_loss(
        self,
        student_output: ModelOutput,
        teacher_outputs: Mapping[str, ModelOutput],
    ) -> torch.Tensor:
        losses = []
        for modality in self.modalities:
            student_feature = extract_modality_feature(
                student_output,
                modality,
                pool=self.feature_pool,
                source="modality",
            )
            teacher_feature = extract_modality_feature(
                teacher_outputs[modality],
                modality,
                pool=self.feature_pool,
                source=self.feature_source,
            )
            if student_feature is None or teacher_feature is None:
                raise ValueError(
                    f"G2D feature KD requires both student and teacher features for modality '{modality}'. "
                    f"student={None if student_feature is None else tuple(student_feature.shape)}, "
                    f"teacher={None if teacher_feature is None else tuple(teacher_feature.shape)}."
                )
            aligned_student = self._align_student_feature(modality, student_feature, teacher_feature)
            teacher_feature = teacher_feature.detach()
            if self.feature_normalize:
                aligned_student = F.normalize(aligned_student, dim=-1)
                teacher_feature = F.normalize(teacher_feature, dim=-1)
            losses.append(F.mse_loss(aligned_student, teacher_feature))
        if not losses:
            device = student_output.logits.device
            return torch.tensor(0.0, dtype=student_output.logits.dtype, device=device)
        return torch.stack(losses).mean()

    def _align_student_feature(
        self,
        modality: str,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
    ) -> torch.Tensor:
        if student_feature.shape[-1] == teacher_feature.shape[-1]:
            return student_feature
        if self.projection_mode != "auto":
            raise ValueError(
                f"G2D feature dim mismatch for '{modality}': "
                f"student={student_feature.shape[-1]}, teacher={teacher_feature.shape[-1]}."
            )
        projection = self.projections[modality] if modality in self.projections else None
        if (
            projection is None
            or getattr(projection, "in_features", None) != student_feature.shape[-1]
            or getattr(projection, "out_features", None) != teacher_feature.shape[-1]
        ):
            projection = nn.Linear(student_feature.shape[-1], teacher_feature.shape[-1]).to(
                device=student_feature.device,
                dtype=student_feature.dtype,
            )
            self.projections[modality] = projection
        return projection(student_feature)


def validate_logits(
    logits: torch.Tensor,
    *,
    num_pred: int,
    num_classes: int | None = None,
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(logits):
        raise TypeError(f"{name} must be a Tensor, got {type(logits).__name__}.")
    if logits.ndim != 3:
        raise ValueError(f"{name} must have shape [B,{num_pred},C], got {tuple(logits.shape)}.")
    if int(logits.shape[1]) != int(num_pred):
        raise ValueError(
            f"Expected {name} horizon={num_pred}, got shape {tuple(logits.shape)}. "
            "G2D requires future-only logits and will not silently drop current/legacy slots."
        )
    if num_classes is not None and int(logits.shape[-1]) != int(num_classes):
        raise ValueError(f"Expected {name} classes={num_classes}, got shape {tuple(logits.shape)}.")
    return logits


def validate_labels(labels: torch.Tensor, *, num_pred: int, batch_size: int | None = None) -> torch.Tensor:
    if not torch.is_tensor(labels):
        raise TypeError(f"labels must be a Tensor, got {type(labels).__name__}.")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [B,{num_pred}], got {tuple(labels.shape)}.")
    if int(labels.shape[1]) != int(num_pred):
        raise ValueError(f"Expected labels horizon={num_pred}, got shape {tuple(labels.shape)}.")
    if batch_size is not None and int(labels.shape[0]) != int(batch_size):
        raise ValueError(f"Expected labels batch={batch_size}, got shape {tuple(labels.shape)}.")
    return labels


def teacher_confidence_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_pred: int,
    modality: str = "teacher",
    ignore_index: int = -100,
) -> torch.Tensor:
    logits = validate_logits(logits, num_pred=num_pred, name=f"{modality} confidence logits")
    labels = validate_labels(labels, num_pred=num_pred, batch_size=logits.shape[0])
    probs = F.softmax(logits, dim=-1)
    safe_labels = labels.clamp_min(0).unsqueeze(-1)
    gathered = probs.gather(dim=-1, index=safe_labels).squeeze(-1)
    valid = labels.ne(ignore_index)
    confidence = []
    for horizon in range(num_pred):
        horizon_valid = valid[:, horizon]
        if torch.any(horizon_valid):
            confidence.append(gathered[:, horizon][horizon_valid].mean())
        else:
            confidence.append(gathered[:, horizon].sum() * 0.0)
    return torch.stack(confidence)


def extract_modality_feature(
    output: ModelOutput,
    modality: str,
    *,
    pool: str = "last",
    source: str = "auto",
) -> torch.Tensor | None:
    diagnostics = output.diagnostics or {}
    feature = None
    modality_features = diagnostics.get("modality_features")
    if isinstance(modality_features, dict) and torch.is_tensor(modality_features.get(modality)):
        feature = modality_features[modality]
    if feature is None and torch.is_tensor(diagnostics.get("token_features")):
        modalities = diagnostics.get("modalities")
        if modalities is not None:
            names = [str(name) for name in modalities]
            if modality in names:
                tokens = diagnostics["token_features"]
                if tokens.ndim == 4:
                    feature = tokens[:, names.index(modality), :, :]
    if feature is None and source in {"auto", "input"} and torch.is_tensor(output.input_features):
        feature = output.input_features
    if feature is None and source in {"auto", "output", "modality"} and torch.is_tensor(output.output_features):
        feature = output.output_features
    if feature is None:
        return None
    return pool_feature(feature, pool=pool)


def pool_feature(feature: torch.Tensor, *, pool: str = "last") -> torch.Tensor:
    if feature.ndim == 2:
        return feature
    if feature.ndim < 2:
        raise ValueError(f"feature must include a batch dimension, got {tuple(feature.shape)}.")
    if feature.ndim > 3:
        feature = feature.reshape(feature.shape[0], feature.shape[1], -1)
    if pool == "last":
        return feature[:, -1, :]
    if pool == "mean":
        return feature.mean(dim=1)
    raise ValueError(f"Unsupported G2D feature pool '{pool}'.")


def modality_rankings(confidence: Mapping[str, torch.Tensor], horizon_labels: list[str]) -> dict[str, list[str]]:
    modalities = list(confidence)
    avg = {name: float(values.mean().detach().cpu().item()) for name, values in confidence.items()}
    rankings = {"avg": sorted(modalities, key=lambda name: (avg[name], modalities.index(name)))}
    for idx, horizon in enumerate(horizon_labels):
        rankings[horizon] = sorted(
            modalities,
            key=lambda name: (float(confidence[name][idx].detach().cpu().item()), modalities.index(name)),
        )
    return rankings


def confidence_to_json(confidence: Mapping[str, torch.Tensor], horizon_labels: list[str]) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for modality, values in confidence.items():
        detached = values.detach().cpu()
        item = {horizon: float(detached[idx].item()) for idx, horizon in enumerate(horizon_labels)}
        item["avg"] = float(detached.mean().item())
        payload[modality] = item
    return payload


def student_branch_confidence_from_output(
    output: ModelOutput,
    labels: torch.Tensor,
    modalities: list[str],
    horizon_labels: list[str],
    *,
    ignore_index: int,
) -> dict[str, dict[str, float]]:
    logits = output.diagnostics.get("unimodal_logits") if isinstance(output.diagnostics, dict) else None
    if not torch.is_tensor(logits) or logits.ndim != 4:
        return {}
    names = [str(name) for name in output.diagnostics.get("modalities", modalities)]
    results: dict[str, torch.Tensor] = {}
    for index, modality in enumerate(names):
        if modality not in modalities or index >= logits.shape[1]:
            continue
        results[modality] = teacher_confidence_from_logits(
            logits[:, index, :, :],
            labels,
            num_pred=labels.shape[1],
            modality=f"student {modality}",
            ignore_index=ignore_index,
        )
    return confidence_to_json(results, horizon_labels) if results else {}


def confidence_ratio(
    student_confidence: Mapping[str, Mapping[str, float]],
    teacher_confidence: Mapping[str, Mapping[str, float]],
    *,
    eps: float = 1e-8,
) -> dict[str, dict[str, float]]:
    ratios: dict[str, dict[str, float]] = {}
    for modality, student_values in student_confidence.items():
        teacher_values = teacher_confidence.get(modality)
        if not teacher_values:
            continue
        ratios[modality] = {
            key: float(value) / max(float(teacher_values.get(key, 0.0)), eps)
            for key, value in student_values.items()
        }
    return ratios


def horizon_names(num_pred: int) -> list[str]:
    if int(num_pred) == 3:
        return list(DEFAULT_HORIZON_NAMES)
    return [f"t+{idx + 1}" for idx in range(int(num_pred))]


def _resolve_horizons(raw: Any, num_pred: int) -> list[int]:
    if raw in (None, "all"):
        return list(range(int(num_pred)))
    if not isinstance(raw, (list, tuple)):
        raise ValueError("G2D horizons must be 'all' or a list of zero-based horizon indices.")
    values = [int(value) for value in raw]
    invalid = [value for value in values if value < 0 or value >= int(num_pred)]
    if invalid:
        raise ValueError(f"G2D horizons out of range for num_pred={num_pred}: {invalid}.")
    return values


def _select_horizons(logits: torch.Tensor, horizons: list[int]) -> torch.Tensor:
    index = torch.as_tensor(horizons, dtype=torch.long, device=logits.device)
    return logits.index_select(1, index)


def _select_label_horizons(labels: torch.Tensor, horizons: list[int]) -> torch.Tensor:
    index = torch.as_tensor(horizons, dtype=torch.long, device=labels.device)
    return labels.index_select(1, index)


def _zero_like(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.sum() * 0.0


__all__ = [
    "G2DDistiller",
    "G2DStepResult",
    "confidence_to_json",
    "extract_modality_feature",
    "horizon_names",
    "modality_rankings",
    "pool_feature",
    "teacher_confidence_from_logits",
    "validate_labels",
    "validate_logits",
]
