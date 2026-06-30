from typing import Any

import torch
import torch.nn as nn


class ReliabilityBiasedMissingAwareAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_modalities: int,
        *,
        num_heads: int = 4,
        beta_reliability: float = 1.0,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_modalities = int(num_modalities)
        self.num_heads = int(num_heads)
        if self.d_model <= 0 or self.num_modalities <= 0 or self.num_heads <= 0:
            raise ValueError("d_model, num_modalities, and num_heads must be positive.")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads for RBMA attention.")
        self.head_dim = self.d_model // self.num_heads
        self.beta_reliability = float(beta_reliability)
        self.eps = float(eps)
        self.query = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.key = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)
        self.out = nn.Sequential(nn.LayerNorm(self.d_model), nn.Dropout(float(dropout)), nn.Linear(self.d_model, self.d_model))
        self.modality_bias = nn.Parameter(torch.zeros(self.num_modalities + 1))
        self.attention_dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        reliability: torch.Tensor,
        *,
        global_token: torch.Tensor | None = None,
        global_reliability: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if z.ndim != 3:
            raise ValueError(f"z must have shape [B, M, D], got {tuple(z.shape)}.")
        batch_size, modality_count, d_model = z.shape
        if modality_count != self.num_modalities or d_model != self.d_model:
            raise ValueError(
                f"z must have shape [B, {self.num_modalities}, {self.d_model}], got {tuple(z.shape)}."
            )
        mask = mask.to(device=z.device, dtype=torch.bool)
        if tuple(mask.shape) != (batch_size, modality_count):
            raise ValueError(f"mask must have shape [{batch_size}, {modality_count}], got {tuple(mask.shape)}.")
        reliability = reliability.to(device=z.device, dtype=z.dtype)
        if reliability.ndim == 3 and reliability.shape[-1] == 1:
            reliability = reliability.squeeze(-1)
        if tuple(reliability.shape) != (batch_size, modality_count):
            raise ValueError(
                f"reliability must have shape [{batch_size}, {modality_count}] or [..., 1], got {tuple(reliability.shape)}."
            )

        tokens = z
        availability = mask
        reliability_values = reliability.clamp_min(0.0)
        token_names = ["modality"] * modality_count
        if global_token is not None:
            if tuple(global_token.shape) != (batch_size, d_model):
                raise ValueError(f"global_token must have shape [{batch_size}, {d_model}], got {tuple(global_token.shape)}.")
            tokens = torch.cat([tokens, global_token.unsqueeze(1)], dim=1)
            availability = torch.cat([availability, torch.ones(batch_size, 1, dtype=torch.bool, device=z.device)], dim=1)
            if global_reliability is None:
                global_rel = torch.ones(batch_size, 1, dtype=z.dtype, device=z.device)
            else:
                global_rel = global_reliability.to(device=z.device, dtype=z.dtype).reshape(batch_size, 1)
            reliability_values = torch.cat([reliability_values, global_rel.clamp_min(0.0)], dim=1)
            token_names.append("global")

        if not availability.any(dim=1).all():
            empty = (~availability.any(dim=1)).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            raise ValueError(f"RBMA attention requires at least one available modality or global token; empty samples={empty}.")

        token_count = tokens.shape[1]
        keys = self.key(tokens).view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(tokens).view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (keys * self.query.view(1, self.num_heads, 1, self.head_dim)).sum(dim=-1) / (self.head_dim**0.5)
        reliability_log_bias = self.beta_reliability * reliability_values.clamp_min(self.eps).log()
        bias = self.modality_bias[:token_count].view(1, 1, token_count)
        scores = scores + reliability_log_bias.unsqueeze(1) + bias
        scores = scores.masked_fill(~availability.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        fused = (weights.unsqueeze(-1) * values).sum(dim=2).reshape(batch_size, self.d_model)
        fused = self.out(fused)
        diagnostics = {
            "rbma_attention_weights": weights.detach(),
            "rbma_attention_mean": weights.detach().mean(dim=1),
            "rbma_mask": availability.detach(),
            "rbma_reliability_log_bias": reliability_log_bias.detach(),
            "rbma_reliability_log_finite": bool(torch.isfinite(reliability_log_bias).all().detach().cpu().item()),
            "rbma_token_names": token_names,
        }
        if global_token is not None:
            diagnostics["rbma_global_attention_mean"] = weights[:, :, -1].detach().mean(dim=1)
        return {"fused": fused, "attention_weights": weights, "diagnostics": diagnostics}


__all__ = ["ReliabilityBiasedMissingAwareAttention"]
