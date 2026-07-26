"""Router input variants for the frozen-U0 Router observability screen.

The diagnostics found exactly one supported hypothesis: the supervised router
reads only post-projection scalars, while the pre-projection encoder features
predict degradation severity far better.  The defensible claim is therefore
**router input insufficiency**, not prototype-induced information destruction --
the erasure hypothesis was tested and is not supported at the aggregate level.

Four nested arms isolate that claim:

    Q0  existing router scalars only            (retraining control)
    Q1  Q0 + per-modality prototype-space state (already-downstream control)
    Q2  Q1 + pre-projection quality embedding   (treatment)
    Q3  Q2 with pre-projection features permuted across samples (capacity control)

Nothing here trains or touches the frozen U0 backbone; only the router head and
the quality branch carry gradients.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from kd_sensing.modalities import MODALITY_ORDER


ARMS: tuple[str, ...] = ("q0", "q1", "q2", "q3")
ARM_DESCRIPTIONS: dict[str, str] = {
    "q0": "existing router scalars only",
    "q1": "q0 + per-modality prototype-space state",
    "q2": "q1 + pre-projection quality embedding",
    "q3": "q2 with pre-projection features permuted across samples",
}
PROTOTYPE_STATE_KEYS: tuple[str, ...] = (
    "nearest_distance",
    "distance_margin",
    "entropy",
    "restoration_residual_norm",
)


def uses_quality_branch(arm: str) -> bool:
    return arm in {"q2", "q3"}


def uses_prototype_state(arm: str) -> bool:
    return arm in {"q1", "q2", "q3"}


class ModalityQualityBranch(nn.Module):
    """Per-modality MLP over the input of each encoder's final linear layer."""

    def __init__(
        self,
        preprojection_dims: Mapping[str, int],
        *,
        embedding_dim: int = 8,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        if self.embedding_dim <= 0 or int(hidden_dim) <= 0:
            raise ValueError("embedding_dim and hidden_dim must be positive.")
        missing = [name for name in MODALITY_ORDER if name not in preprojection_dims]
        if missing:
            raise ValueError(f"Pre-projection dimensions are missing for {missing}.")
        self.preprojection_dims = {name: int(preprojection_dims[name]) for name in MODALITY_ORDER}
        self.branches = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(self.preprojection_dims[name]),
                    nn.Linear(self.preprojection_dims[name], int(hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden_dim), self.embedding_dim),
                )
                for name in MODALITY_ORDER
            }
        )

    def forward(self, preprojection: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Map ``{modality: [B, D_pre]}`` to a stacked ``[B, M, embedding_dim]``."""
        embeddings = []
        for name in MODALITY_ORDER:
            value = preprojection[name]
            if value.ndim != 2 or value.shape[-1] != self.preprojection_dims[name]:
                raise ValueError(
                    f"{name} pre-projection features must have shape [B, {self.preprojection_dims[name]}], "
                    f"got {tuple(value.shape)}."
                )
            embeddings.append(self.branches[name](value))
        return torch.stack(embeddings, dim=1)


class RouterHead(nn.Module):
    """Trainable replacement for U0's supervised router, with a widened input."""

    def __init__(
        self,
        feature_count: int,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        use_pattern_bias: bool = True,
    ) -> None:
        super().__init__()
        total = int(feature_count) + len(MODALITY_ORDER)
        if int(feature_count) <= 0:
            raise ValueError("feature_count must be positive.")
        # Same shape as UMaskBeamJEPA.supervised_router; only the input width differs.
        self.mlp = nn.Sequential(
            nn.LayerNorm(total),
            nn.Linear(total, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        self.pattern_bias = (
            nn.Linear(len(MODALITY_ORDER), len(MODALITY_ORDER), bias=False) if use_pattern_bias else None
        )

    def forward(self, features: torch.Tensor, available: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[:2] != available.shape:
            raise ValueError("features must be [B, M, F] and match available [B, M].")
        identity = torch.eye(len(MODALITY_ORDER), device=features.device, dtype=features.dtype)
        identity = identity.unsqueeze(0).expand(features.shape[0], -1, -1)
        logits = self.mlp(torch.cat([features, identity], dim=-1)).squeeze(-1)
        if self.pattern_bias is not None:
            logits = logits + self.pattern_bias(available.to(dtype=features.dtype))
        mask = available.to(dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every sample must keep at least one available modality.")
        masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits, torch.softmax(masked, dim=1) * mask.to(dtype=logits.dtype)


class RouterObservabilityModel(nn.Module):
    """Assemble one arm's router input and route cached frozen-U0 representations."""

    def __init__(
        self,
        arm: str,
        *,
        scalar_feature_count: int,
        preprojection_dims: Mapping[str, int],
        embedding_dim: int = 8,
        hidden_dim: int = 64,
        quality_hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"Unknown router observability arm: {arm!r}")
        self.arm = arm
        self.scalar_feature_count = int(scalar_feature_count)
        feature_count = self.scalar_feature_count
        if uses_prototype_state(arm):
            feature_count += len(PROTOTYPE_STATE_KEYS)
        self.quality_branch = (
            ModalityQualityBranch(
                preprojection_dims,
                embedding_dim=embedding_dim,
                hidden_dim=quality_hidden_dim,
                dropout=dropout,
            )
            if uses_quality_branch(arm)
            else None
        )
        if self.quality_branch is not None:
            feature_count += int(embedding_dim)
        self.feature_count = feature_count
        self.router = RouterHead(feature_count, hidden_dim=hidden_dim, dropout=dropout)
        self.register_buffer("quality_mean", torch.zeros(len(MODALITY_ORDER), int(embedding_dim)), persistent=True)
        self.register_buffer("quality_mean_fitted", torch.tensor(False), persistent=True)

    # -- router input ------------------------------------------------------

    def router_features(
        self,
        scalars: torch.Tensor,
        prototype_state: Mapping[str, torch.Tensor] | None,
        preprojection: Mapping[str, torch.Tensor] | None,
        *,
        permutation: torch.Tensor | None = None,
        ablate_quality: bool = False,
    ) -> torch.Tensor:
        parts = [scalars]
        if uses_prototype_state(self.arm):
            if prototype_state is None:
                raise ValueError(f"Arm {self.arm} requires per-modality prototype state.")
            parts.append(
                torch.stack([prototype_state[key] for key in PROTOTYPE_STATE_KEYS], dim=-1).to(dtype=scalars.dtype)
            )
        if self.quality_branch is not None:
            if ablate_quality:
                if not bool(self.quality_mean_fitted):
                    raise ValueError("Inference-time ablation requires the train-only mean embedding.")
                embedding = self.quality_mean.to(dtype=scalars.dtype).unsqueeze(0).expand(scalars.shape[0], -1, -1)
            else:
                if preprojection is None:
                    raise ValueError(f"Arm {self.arm} requires pre-projection features.")
                source = preprojection
                if self.arm == "q3":
                    if permutation is None:
                        raise ValueError("Arm q3 requires an explicit cross-sample permutation.")
                    # Same tensors, same parameters, sample alignment destroyed.
                    source = {name: value[permutation] for name, value in preprojection.items()}
                embedding = self.quality_branch(source).to(dtype=scalars.dtype)
            parts.append(embedding)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        scalars: torch.Tensor,
        available: torch.Tensor,
        unimodal_logits: torch.Tensor,
        *,
        prototype_state: Mapping[str, torch.Tensor] | None = None,
        preprojection: Mapping[str, torch.Tensor] | None = None,
        permutation: torch.Tensor | None = None,
        ablate_quality: bool = False,
    ) -> dict[str, torch.Tensor]:
        features = self.router_features(
            scalars,
            prototype_state,
            preprojection,
            permutation=permutation,
            ablate_quality=ablate_quality,
        )
        logits, weights = self.router(features, available)
        return {
            "router_features": features,
            "router_gate_logits": logits,
            "router_gate_weights": weights,
            "logits": (weights.unsqueeze(-1) * unimodal_logits).sum(dim=1),
        }

    # -- inference-time ablation support -----------------------------------

    @torch.no_grad()
    def set_quality_mean(self, mean: torch.Tensor) -> None:
        """Install the train-only mean quality embedding used by the ablation."""
        if self.quality_branch is None:
            raise RuntimeError("Only arms with a quality branch support the mean-embedding ablation.")
        value = torch.as_tensor(mean, dtype=self.quality_mean.dtype, device=self.quality_mean.device)
        if tuple(value.shape) != tuple(self.quality_mean.shape):
            raise ValueError(f"quality mean must have shape {tuple(self.quality_mean.shape)}, got {tuple(value.shape)}.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("quality mean must be finite.")
        self.quality_mean.copy_(value)
        self.quality_mean_fitted.fill_(True)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


__all__ = [
    "ARMS",
    "ARM_DESCRIPTIONS",
    "PROTOTYPE_STATE_KEYS",
    "ModalityQualityBranch",
    "RouterHead",
    "RouterObservabilityModel",
    "uses_prototype_state",
    "uses_quality_branch",
]
