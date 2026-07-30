"""Small, inference-legal primitives for SMSL R5 experiments.

The module intentionally owns no model or data loader.  SMSL is a local
training policy layered on top of the frozen M4 protocol, so keeping its
feature construction and losses here makes the data-use boundary testable.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


SMSL_ARMS = ("a0", "a1", "a2", "a3", "c1", "c2", "c3")
F2_FEATURE_NAMES = (
    "mask_image_available",
    "mask_lidar_available",
    "mask_radar_available",
    "mask_gps_available",
    "missing_top1_top2_gap",
    "missing_entropy",
    "missing_embedding_norm",
    "available_unimodal_agreement",
    "available_unimodal_pairwise_disagreement",
)
FORBIDDEN_F2_FEATURE_TOKENS = (
    "label",
    "target",
    "rank",
    "full",
    "future",
    "trajectory",
    "metadata",
    "weather",
    "scene",
    "power",
)


def validate_legal_feature_names(names: tuple[str, ...] = F2_FEATURE_NAMES) -> None:
    """Fail closed when a purported deployment feature crosses the contract."""
    if tuple(names) != F2_FEATURE_NAMES:
        raise ValueError("SMSL F2 feature order changed from the registered legal contract.")
    prohibited = [
        name
        for name in names
        if any(token in name.lower() for token in FORBIDDEN_F2_FEATURE_TOKENS)
    ]
    if prohibited:
        raise ValueError(f"SMSL F2 features are not inference-legal: {prohibited}")


def _availability(mask: torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if value.ndim == 1:
        value = value.unsqueeze(0).expand(batch, -1)
    if value.shape != (batch, 4) or not bool(value.any(dim=1).all()):
        raise ValueError("SMSL availability must be a non-empty [B,4] mask.")
    return value


def legal_f2_features(
    availability: torch.Tensor,
    missing_logits: torch.Tensor,
    missing_embedding: torch.Tensor,
    unimodal_logits: torch.Tensor,
) -> torch.Tensor:
    """Build F2 only from the current missing-view prediction state.

    Labels, Full-view outputs, future frames, trajectory metadata, and target
    ranks are deliberately absent from both the signature and the result.
    """
    logits = torch.as_tensor(missing_logits, dtype=torch.float32)
    embedding = torch.as_tensor(missing_embedding, dtype=torch.float32, device=logits.device)
    unimodal = torch.as_tensor(unimodal_logits, dtype=torch.float32, device=logits.device)
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("SMSL missing logits must be [B,C] with C >= 2.")
    if embedding.ndim != 2 or embedding.shape[0] != logits.shape[0]:
        raise ValueError("SMSL missing embedding must align as [B,D].")
    if unimodal.shape[:2] != (logits.shape[0], 4) or unimodal.ndim != 3:
        raise ValueError("SMSL unimodal logits must be [B,4,C].")
    mask = _availability(availability, logits.shape[0], logits.device)
    top_two = logits.topk(2, dim=-1).values
    probability = torch.softmax(logits, dim=-1)
    entropy = -(probability * probability.clamp_min(torch.finfo(probability.dtype).tiny).log()).sum(dim=-1)
    prediction = unimodal.argmax(dim=-1)
    pair_available = mask[:, :, None] & mask[:, None, :]
    upper = torch.triu(torch.ones(4, 4, dtype=torch.bool, device=logits.device), diagonal=1)
    pair_mask = pair_available & upper
    pair_count = pair_mask.sum(dim=(1, 2))
    disagreement = prediction[:, :, None].ne(prediction[:, None, :]).to(logits.dtype)
    pairwise_disagreement = (disagreement * pair_mask.to(logits.dtype)).sum(dim=(1, 2)) / pair_count.clamp_min(1)
    # Modal prediction counts are tiny (four views), so one equality matrix is
    # clearer and safer than introducing a label-dependent pseudo-target.
    agreement_matrix = prediction[:, :, None].eq(prediction[:, None, :]) & pair_available
    agreement = agreement_matrix.to(logits.dtype).sum(dim=(1, 2)) / mask.sum(dim=1).square().clamp_min(1)
    features = torch.cat(
        (
            mask.to(dtype=logits.dtype),
            (top_two[:, :1] - top_two[:, 1:2]),
            entropy[:, None],
            embedding.norm(dim=-1, keepdim=True),
            agreement[:, None],
            pairwise_disagreement[:, None],
        ),
        dim=1,
    )
    if features.shape[1] != len(F2_FEATURE_NAMES) or not bool(torch.isfinite(features).all()):
        raise ValueError("SMSL F2 feature construction produced an invalid value.")
    return features


def severe_availability(availability: torch.Tensor) -> torch.Tensor:
    """Return the R5 target scope: one or two available modalities only."""
    mask = torch.as_tensor(availability, dtype=torch.bool)
    if mask.ndim != 2 or mask.shape[1] != 4:
        raise ValueError("SMSL availability must have shape [B,4].")
    return mask.sum(dim=1).le(2)


def normalized_risk_weights(
    risk: torch.Tensor,
    *,
    alpha: float,
    minimum: float = 0.5,
    maximum: float = 2.0,
) -> torch.Tensor:
    """Convert detached F2 risk to bounded, batch-mean-normalized weights."""
    if float(alpha) < 0 or float(minimum) <= 0 or float(maximum) < float(minimum):
        raise ValueError("SMSL risk-weight parameters are invalid.")
    value = torch.as_tensor(risk, dtype=torch.float32)
    if value.ndim != 1 or value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError("SMSL risk must be a finite non-empty [B] tensor.")
    bounded = value.detach().clamp(0.05, 0.95)
    raw = 1.0 + float(alpha) * bounded
    normalized = raw / raw.mean().clamp_min(torch.finfo(raw.dtype).tiny)
    return normalized.clamp(float(minimum), float(maximum))


def normalized_hard_weights(
    per_sample_loss: torch.Tensor,
    *,
    minimum: float = 0.5,
    maximum: float = 2.0,
) -> torch.Tensor:
    """A1's label-side hard weighting, kept separate from F2 by design."""
    values = torch.as_tensor(per_sample_loss, dtype=torch.float32)
    if values.ndim != 1 or values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("SMSL hard weights require finite non-empty [B] losses.")
    normalized = values.detach() / values.detach().mean().clamp_min(torch.finfo(values.dtype).tiny)
    return normalized.clamp(float(minimum), float(maximum))


def shuffled_weights(weights: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """C2 negative control: preserve the exact batch distribution, shuffle IDs."""
    values = torch.as_tensor(weights)
    if values.ndim != 1:
        raise ValueError("SMSL shuffled weights require a [B] tensor.")
    return values[torch.randperm(values.numel(), device=values.device, generator=generator)]


def directional_margin_distillation(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> Mapping[str, torch.Tensor]:
    """Penalize a missing-view margin below the frozen Full teacher's margin.

    The Full teacher is used only on teacher-correct samples.  All arithmetic is
    FP32 and the denominator is the number of teacher-correct examples, which
    keeps A3/C3 comparable across availability masks.
    """
    student = torch.as_tensor(student_logits, dtype=torch.float32)
    teacher = torch.as_tensor(teacher_logits, dtype=torch.float32, device=student.device).detach()
    target = torch.as_tensor(labels, dtype=torch.long, device=student.device).reshape(-1)
    if student.ndim != 2 or teacher.shape != student.shape or target.numel() != student.shape[0]:
        raise ValueError("SMSL directional margin inputs must align as [B,C], [B,C], [B].")
    if not bool(((target >= 0) & (target < student.shape[1])).all()):
        raise ValueError("SMSL directional margin labels are out of range.")
    sample_weights = (
        torch.ones_like(target, dtype=student.dtype)
        if weights is None
        else torch.as_tensor(weights, dtype=student.dtype, device=student.device).detach().reshape(-1)
    )
    if (
        sample_weights.shape != target.shape
        or not bool(torch.isfinite(sample_weights).all())
        or not bool(sample_weights.ge(0).all())
    ):
        raise ValueError("SMSL directional margin weights must be finite and align with [B].")
    teacher_prediction = teacher.argmax(dim=-1)
    teacher_correct = teacher_prediction.eq(target)
    excluded = teacher.clone()
    excluded.scatter_(1, target[:, None], -torch.inf)
    confuser = excluded.argmax(dim=-1)
    teacher_target = teacher.gather(1, target[:, None]).squeeze(1)
    teacher_confuser = teacher.gather(1, confuser[:, None]).squeeze(1)
    student_target = student.gather(1, target[:, None]).squeeze(1)
    student_confuser = student.gather(1, confuser[:, None]).squeeze(1)
    teacher_margin = teacher_target - teacher_confuser
    student_margin = student_target - student_confuser
    violation = F.relu(teacher_margin - student_margin)
    eligible = teacher_correct.to(dtype=student.dtype)
    denominator = eligible.sum()
    loss = (violation * eligible * sample_weights).sum() / denominator.clamp_min(1.0)
    return {
        "loss": loss,
        "teacher_correct": teacher_correct,
        "confuser": confuser,
        "teacher_margin": teacher_margin,
        "student_margin": student_margin,
        "violation": violation,
        "denominator": denominator,
    }


__all__ = [
    "F2_FEATURE_NAMES",
    "FORBIDDEN_F2_FEATURE_TOKENS",
    "SMSL_ARMS",
    "directional_margin_distillation",
    "legal_f2_features",
    "normalized_hard_weights",
    "normalized_risk_weights",
    "severe_availability",
    "shuffled_weights",
    "validate_legal_feature_names",
]
