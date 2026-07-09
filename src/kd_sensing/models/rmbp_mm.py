from typing import Any

import torch
import torch.nn as nn

from kd_sensing.registries import REPRESENTATION_CORES


@REPRESENTATION_CORES.register("rmbp_channel_attention_fusion")
class RMBPChannelAttentionFusionCore(nn.Module):
    supports_missing_modality_metadata = True

    def __init__(
        self,
        d_model: int,
        modality_count: int,
        bottleneck_dim: int | None = None,
        output_dim: int | None = None,
        dropout: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.modality_count = int(modality_count)
        self.bottleneck_dim = int(bottleneck_dim or max(1, self.modality_count // 2))
        self.output_dim = int(output_dim or d_model)
        if min(self.d_model, self.modality_count, self.bottleneck_dim, self.output_dim) <= 0:
            raise ValueError("rmbp_channel_attention_fusion dimensions must be positive.")

        self.shared_mlp = nn.Sequential(
            nn.Linear(self.modality_count, self.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.bottleneck_dim, self.modality_count),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.modality_count * self.d_model),
            nn.Dropout(float(dropout)),
            nn.Linear(self.modality_count * self.d_model, self.output_dim),
        )
        self.last_attention_weights: torch.Tensor | None = None
        self.last_availability_mask: torch.Tensor | None = None

    def forward(self, features: torch.Tensor, *, modality_available: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"rmbp_channel_attention_fusion expects [B,K,T,D], got {tuple(features.shape)}.")
        batch_size, modality_count, seq_len, d_model = features.shape
        if int(modality_count) != self.modality_count or int(d_model) != self.d_model:
            raise ValueError(
                "rmbp_channel_attention_fusion received incompatible shape: "
                f"expected K={self.modality_count}, D={self.d_model}, got {tuple(features.shape)}."
            )
        availability = self._availability(modality_available, features)
        features_bt = features.permute(0, 2, 1, 3).contiguous()
        avg_pool = features_bt.mean(dim=-1)
        max_pool = features_bt.max(dim=-1).values
        weights = torch.sigmoid(self.shared_mlp(avg_pool) + self.shared_mlp(max_pool))
        weights = weights * availability.permute(0, 2, 1).to(dtype=weights.dtype)
        weighted = features_bt * weights.unsqueeze(-1)
        output = self.output_projection(weighted.reshape(batch_size, seq_len, self.modality_count * self.d_model))
        self.last_attention_weights = weights.detach()
        self.last_availability_mask = availability.detach()
        return output

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "rmbp_channel_attention_fusion",
            "model_group": "RMBP-MM",
            "paper_alignment": "channel_attention_fusion",
            "fusion_type": "rmbp_channel_attention",
            "d_model": self.d_model,
            "output_dim": self.output_dim,
            "modality_count": self.modality_count,
            "bottleneck_dim": self.bottleneck_dim,
            "pooling": ["global_average_over_feature", "global_max_over_feature"],
            "attention_activation": "sigmoid",
            "missing_modality_strategy": "mask_attention_weights_to_zero",
            "consumes_missing_modality_metadata": True,
        }

    def _availability(self, modality_available: torch.Tensor | None, features: torch.Tensor) -> torch.Tensor:
        if modality_available is None:
            return torch.ones(features.shape[:3], dtype=torch.bool, device=features.device)
        value = torch.as_tensor(modality_available, dtype=torch.bool, device=features.device)
        if value.ndim != 3 or tuple(value.shape) != tuple(features.shape[:3]):
            raise ValueError(
                "rmbp_channel_attention_fusion modality_available must match "
                f"{tuple(features.shape[:3])}, got {tuple(value.shape)}."
            )
        return value


__all__ = ["RMBPChannelAttentionFusionCore"]
