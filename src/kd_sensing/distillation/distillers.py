from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.registries import DISTILLERS


class KnowledgeDistillationLoss(nn.Module):
    """KD loss covering no-KD, logits KD, and relational KD."""

    def __init__(
        self,
        task_criterion: nn.Module,
        kd_mode: int = 0,
        temperature: float = 3.0,
        alpha: float = 0.4,
        rkd_pairs_per_anchor: int = 4,
        rkd_distance_weight: float = 10.0,
        rkd_angle_weight: float = 10.0,
        **_: object,
    ):
        super().__init__()
        self.task_criterion = task_criterion
        self.kd_mode = kd_mode
        self.temperature = temperature
        self.alpha = alpha
        self.rkd_pairs_per_anchor = rkd_pairs_per_anchor
        self.rkd_distance_weight = rkd_distance_weight
        self.rkd_angle_weight = rkd_angle_weight
        self.kl_div = nn.KLDivLoss(reduction="none")
        self.mse_loss = nn.MSELoss()

    def select_pairs(self, batch_size: int, k: int) -> torch.Tensor:
        pairs = []
        for i in range(batch_size):
            positive_indices = list(range(batch_size))
            positive_indices.remove(i)
            k_actual = min(k, len(positive_indices))
            if k_actual > 0:
                selected = torch.randperm(len(positive_indices))[:k_actual]
                pairs.extend([[i, positive_indices[idx]] for idx in selected])
        return torch.tensor(pairs) if pairs else torch.empty(0, 2, dtype=torch.long)

    def compute_euclidean_distance(self, features: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        if pairs.numel() == 0:
            return torch.empty(0, device=features.device)
        features_flat = features.reshape(features.size(0), -1)
        anchor_features = features_flat[pairs[:, 0]]
        positive_features = features_flat[pairs[:, 1]]
        distances = torch.norm(anchor_features - positive_features, p=2, dim=1)
        mean_distance = distances.mean() if distances.numel() > 0 else torch.tensor(1.0, device=features.device)
        if mean_distance > 0:
            distances = distances / mean_distance
        return distances

    def compute_cosine_distance(self, features: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        if pairs.numel() == 0:
            return torch.empty(0, device=features.device)
        features_flat = features.reshape(features.size(0), -1)
        anchor_features = features_flat[pairs[:, 0]]
        positive_features = features_flat[pairs[:, 1]]
        return 1 - F.cosine_similarity(anchor_features, positive_features, dim=1)

    def relational_knowledge_distillation_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = student_features.size(0)
        pairs = self.select_pairs(batch_size, self.rkd_pairs_per_anchor)
        if pairs.numel() == 0:
            return torch.tensor(0.0, device=student_features.device)
        pairs = pairs.to(student_features.device)
        student_euclidean = self.compute_euclidean_distance(student_features, pairs)
        student_cosine = self.compute_cosine_distance(student_features, pairs)
        teacher_euclidean = self.compute_euclidean_distance(teacher_features, pairs)
        teacher_cosine = self.compute_cosine_distance(teacher_features, pairs)
        distance_loss = self.mse_loss(student_euclidean, teacher_euclidean)
        angle_loss = self.mse_loss(student_cosine, teacher_cosine)
        return self.rkd_distance_weight * distance_loss + self.rkd_angle_weight * angle_loss

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        student_input_features: torch.Tensor | None = None,
        teacher_input_features: torch.Tensor | None = None,
        student_output_features: torch.Tensor | None = None,
        teacher_output_features: torch.Tensor | None = None,
        current_alpha: float | None = None,
        soft_targets: torch.Tensor | None = None,
    ):
        task_loss = self.task_criterion(student_logits, soft_targets if soft_targets is not None else targets)
        distillation_loss = torch.tensor(0.0, device=student_logits.device)

        if self.kd_mode == 0:
            return task_loss, task_loss, distillation_loss

        if self.kd_mode == 1:
            student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
            kl_loss = self.kl_div(student_soft, teacher_soft)
            distillation_loss = kl_loss.sum(dim=1).mean() * (self.temperature ** 2)
        elif self.kd_mode == 2:
            if student_output_features is None or teacher_output_features is None:
                raise ValueError(
                    "Relational KD requires real student and teacher output_features; "
                    "model outputs did not provide the required feature tensors."
                )
            distillation_loss = self.relational_knowledge_distillation_loss(
                student_output_features,
                teacher_output_features,
            )
        else:
            raise ValueError(f"Unsupported kd_mode: {self.kd_mode}")

        alpha = self.alpha if current_alpha is None else current_alpha
        total_loss = (1 - alpha) * task_loss + alpha * distillation_loss
        return total_loss, task_loss, distillation_loss


@DISTILLERS.register("no_kd")
class NoKDDistiller(KnowledgeDistillationLoss):
    def __init__(self, task_criterion: nn.Module, **kwargs: object):
        super().__init__(task_criterion=task_criterion, kd_mode=0, **kwargs)


@DISTILLERS.register("logits_kd")
class LogitsKDDistiller(KnowledgeDistillationLoss):
    def __init__(self, task_criterion: nn.Module, **kwargs: object):
        super().__init__(task_criterion=task_criterion, kd_mode=1, **kwargs)


@DISTILLERS.register("rkd")
class RKDDistiller(KnowledgeDistillationLoss):
    def __init__(self, task_criterion: nn.Module, **kwargs: object):
        super().__init__(task_criterion=task_criterion, kd_mode=2, **kwargs)
