from typing import Any

import torch
import torch.nn as nn


class BCACLModule(nn.Module):
    """Training-only modality-private and shared Beam classification heads."""

    def __init__(
        self,
        *,
        modalities: tuple[str, ...] | list[str],
        input_dim: int,
        num_classes: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        projection = config["projection"]
        self.projection_dim = int(projection["dim"])
        dropout = float(projection["dropout"])
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.input_dim, self.projection_dim),
                    nn.LayerNorm(self.projection_dim) if projection["layer_norm"] else nn.Identity(),
                    nn.Dropout(dropout) if dropout else nn.Identity(),
                )
                for name in self.modalities
            }
        )
        self.private_heads = nn.ModuleDict(
            {name: nn.Linear(self.projection_dim, self.num_classes) for name in self.modalities}
        )
        self.shared_head = nn.Linear(self.projection_dim, self.num_classes)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        expected = (len(self.modalities), self.input_dim)
        if features.ndim != 3 or tuple(features.shape[1:]) != expected:
            raise ValueError(
                f"BCACL modality features must have shape [B,{expected[0]},{expected[1]}], "
                f"got {tuple(features.shape)}."
            )
        projected = torch.stack(
            [self.projections[name](features[:, index]) for index, name in enumerate(self.modalities)],
            dim=1,
        )
        private_logits = torch.stack(
            [self.private_heads[name](projected[:, index]) for index, name in enumerate(self.modalities)],
            dim=1,
        )
        return {
            "features": projected,
            "private_logits": private_logits,
            "shared_logits": self.shared_head(projected),
        }


__all__ = ["BCACLModule"]
