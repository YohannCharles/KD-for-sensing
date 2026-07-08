import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.registries import REPRESENTATION_CORES


@REPRESENTATION_CORES.register("amber_full_adaptive_mask_transformer")
class AmberFullAdaptiveMaskTransformerCore(nn.Module):
    supports_missing_modality_metadata = True
    supports_reliability_metadata = True
    supports_spatial_modality_tokens = True

    def __init__(
        self,
        d_model: int,
        modality_count: int,
        num_heads: int = 4,
        modality_layers: int = 1,
        fusion_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        max_spatial_tokens: int = 256,
        output_dim: int | None = None,
        mask_token_strategy: str = "learned_per_modality",
        num_cma_queries: int = 4,
        cma_dim: int | None = None,
        cma_temperature: float = 0.2,
        modality_indicator_temperature: float = 1.0,
        enable_auxiliary: bool = True,
        auxiliary_loss_weights: dict[str, float] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.modality_count = int(modality_count)
        self.num_heads = int(num_heads)
        self.modality_layers = int(modality_layers)
        self.fusion_layers = int(fusion_layers)
        self.max_seq_len = int(max_seq_len)
        self.max_spatial_tokens = int(max_spatial_tokens)
        self.output_dim = int(output_dim or d_model)
        self.mask_token_strategy = str(mask_token_strategy)
        self.num_cma_queries = max(int(num_cma_queries), self.modality_count)
        self.cma_dim = int(cma_dim or d_model)
        self.cma_temperature = float(cma_temperature)
        self.modality_indicator_temperature = max(float(modality_indicator_temperature), 1e-6)
        self.enable_auxiliary = bool(enable_auxiliary)
        self.auxiliary_loss_weights = dict(auxiliary_loss_weights or {})
        if min(self.d_model, self.modality_count, self.output_dim, self.max_seq_len, self.max_spatial_tokens, self.cma_dim) <= 0:
            raise ValueError("amber_full_adaptive_mask_transformer dimensions must be positive.")
        if self.num_heads <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads}).")
        if self.mask_token_strategy not in {"learned_per_modality", "learned_shared"}:
            raise ValueError("mask_token_strategy must be 'learned_per_modality' or 'learned_shared'.")

        token_count = self.modality_count if self.mask_token_strategy == "learned_per_modality" else 1
        self.mask_tokens = nn.Parameter(torch.zeros(token_count, self.d_model))
        self.modality_embedding = nn.Embedding(self.modality_count, self.d_model)
        self.time_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.spatial_embedding = nn.Embedding(self.max_spatial_tokens, self.d_model)
        self.modality_indicator_logits = nn.Parameter(torch.zeros(self.modality_count))
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(float(dropout))
        self.modality_branches = nn.ModuleList(
            [
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self.d_model,
                        nhead=self.num_heads,
                        dropout=float(dropout),
                        dim_feedforward=max(self.d_model * 4, 64),
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    ),
                    num_layers=self.modality_layers,
                )
                for _ in range(self.modality_count)
            ]
        )
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads,
                dropout=float(dropout),
                dim_feedforward=max(self.d_model * 4, 64),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=self.fusion_layers,
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_projection = nn.Identity() if self.output_dim == self.d_model else nn.Linear(self.d_model, self.output_dim)
        self.cma_projection = nn.Linear(self.d_model, self.cma_dim)
        self.cma_modality_queries = nn.Parameter(torch.zeros(self.modality_count, self.cma_dim))
        self.cma_fusion_query = nn.Parameter(torch.zeros(1, self.cma_dim))
        self.last_amber_full_auxiliary: dict[str, Any] | None = None
        self.last_amber_full_attention_mask: torch.Tensor | None = None
        nn.init.trunc_normal_(self.mask_tokens, std=0.02)
        nn.init.trunc_normal_(self.fusion_token, std=0.02)
        nn.init.trunc_normal_(self.cma_modality_queries, std=0.02)
        nn.init.trunc_normal_(self.cma_fusion_query, std=0.02)

    def forward(self, features: torch.Tensor, *, modality_available: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim == 4:
            features = features.unsqueeze(3)
        if features.ndim != 5:
            raise ValueError(f"amber_full_adaptive_mask_transformer expects [B,K,T,D] or [B,K,T,S,D], got {tuple(features.shape)}.")
        batch_size, modality_count, seq_len, spatial_tokens, d_model = features.shape
        if int(modality_count) != self.modality_count or int(d_model) != self.d_model:
            raise ValueError(
                "amber_full_adaptive_mask_transformer received incompatible shape: "
                f"expected K={self.modality_count}, D={self.d_model}, got {tuple(features.shape)}."
            )
        if int(seq_len) > self.max_seq_len:
            raise ValueError(f"AMBER full seq_len {int(seq_len)} exceeds max_seq_len={self.max_seq_len}.")
        if int(spatial_tokens) > self.max_spatial_tokens:
            raise ValueError(f"AMBER full spatial token count {int(spatial_tokens)} exceeds max_spatial_tokens={self.max_spatial_tokens}.")

        availability = self._availability(modality_available, features)
        indicator = self._modality_indicator(features.device, features.dtype)
        masked = torch.where(availability.unsqueeze(-1), features, self._mask_tokens(features.device, features.dtype))
        masked = masked * indicator.view(1, self.modality_count, 1, 1, 1)
        tokens = self._add_position(masked)
        modality_features = torch.stack(
            [
                branch(tokens[:, index].reshape(batch_size, int(seq_len) * int(spatial_tokens), self.d_model)).view(
                    batch_size,
                    int(seq_len),
                    int(spatial_tokens),
                    self.d_model,
                )
                for index, branch in enumerate(self.modality_branches)
            ],
            dim=1,
        )
        modality_frame_tokens = modality_features.permute(0, 2, 1, 3, 4).reshape(
            batch_size,
            int(seq_len),
            self.modality_count * int(spatial_tokens),
            self.d_model,
        )
        frame_tokens = [self._fusion_tokens(batch_size, seq_len, features), modality_frame_tokens]
        sequence = torch.cat(frame_tokens, dim=2).reshape(
            batch_size,
            int(seq_len) * self._tokens_per_step(int(spatial_tokens)),
            self.d_model,
        )
        key_padding = self._fusion_key_padding_mask(availability)
        memory = self.fusion_transformer(sequence, src_key_padding_mask=key_padding)
        memory = memory.view(batch_size, int(seq_len), self._tokens_per_step(int(spatial_tokens)), self.d_model)
        fusion = self.output_norm(memory[:, :, 0])
        self.last_amber_full_attention_mask = key_padding.detach()
        self.last_amber_full_auxiliary = self._auxiliary_payload(fusion, modality_features, availability, indicator)
        return self.output_projection(fusion)

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "amber_full_adaptive_mask_transformer",
            "reproduction_scope": "amber_full_local",
            "d_model": self.d_model,
            "output_dim": self.output_dim,
            "modality_count": self.modality_count,
            "enabled_modalities": "config_order",
            "history_beam_usage": "disabled",
            "mask_strategy": "adaptive_key_padding_attention_mask",
            "mask_token_strategy": self.mask_token_strategy,
            "max_spatial_tokens": self.max_spatial_tokens,
            "modality_specific_transformer_layers": self.modality_layers,
            "fusion_transformer_layers": self.fusion_layers,
            "cma_enabled": self.num_cma_queries > 0,
            "cma_type": "class_query_cross_attention",
            "cma_temperature": self.cma_temperature,
            "modality_indicator_enabled": True,
            "modality_indicator_temperature": self.modality_indicator_temperature,
            "l2_regularization_source": "modality_indicator",
            "auxiliary_loss_weights": self.auxiliary_loss_weights,
            "consumes_missing_modality_metadata": True,
            "consumes_reliability_metadata": True,
            "output_boundary": "outputs/analysis/local_baselines/amber_full_architecture/",
        }

    def _add_position(self, features: torch.Tensor) -> torch.Tensor:
        seq_len = int(features.shape[2])
        spatial_tokens = int(features.shape[3])
        time = self.time_embedding(torch.arange(seq_len, device=features.device)).view(1, 1, seq_len, 1, self.d_model)
        modality = self.modality_embedding(torch.arange(self.modality_count, device=features.device)).view(
            1, self.modality_count, 1, 1, self.d_model
        )
        spatial = self.spatial_embedding(torch.arange(spatial_tokens, device=features.device)).view(
            1, 1, 1, spatial_tokens, self.d_model
        )
        return self.input_dropout(self.input_norm(features + time + modality + spatial))

    def _auxiliary_payload(
        self,
        fusion: torch.Tensor,
        modality_features: torch.Tensor,
        availability: torch.Tensor,
        indicator: torch.Tensor,
    ) -> dict[str, Any] | None:
        if not (self.training and self.enable_auxiliary):
            return None
        modality_bt = modality_features.mean(dim=3).permute(0, 2, 1, 3).contiguous()
        modality_available = availability.any(dim=3)
        indicator_l2 = (
            indicator.view(1, self.modality_count, 1).pow(2)
            * modality_available.to(dtype=indicator.dtype)
        ).sum(dim=1).mean()
        token_embeddings = F.normalize(self.cma_projection(modality_features), dim=-1)
        fusion_embeddings = F.normalize(self.cma_projection(fusion) + self.cma_fusion_query.view(1, 1, -1), dim=-1)
        modality_query_embeddings = self._class_query_embeddings(token_embeddings, availability)
        cma_logits = torch.einsum("btd,bktd->btk", fusion_embeddings, modality_query_embeddings) / max(self.cma_temperature, 1e-6)
        return {
            "modality_specific_features": modality_bt,
            "fusion_features": fusion,
            "fusion_token": fusion,
            "alignment_target": _available_mean(modality_bt, modality_available.permute(0, 2, 1).contiguous()),
            "cma_fusion_embeddings": fusion_embeddings,
            "cma_modality_embeddings": modality_query_embeddings.permute(0, 2, 1, 3).contiguous(),
            "cma_fusion_query_embeddings": fusion_embeddings,
            "cma_modality_query_embeddings": modality_query_embeddings,
            "cma_query_embeddings": F.normalize(self.cma_modality_queries, dim=-1),
            "cma_logits": cma_logits,
            "modality_indicator_weights": indicator,
            "modality_l2_regularization": indicator_l2,
            "availability_mask": modality_available,
            "token_availability_mask": availability,
            "mask_provenance": "input_valid_or_dropout_masks",
            "l2_regularization_source": "modality_indicator",
        }

    def _availability(self, modality_available: torch.Tensor | None, features: torch.Tensor) -> torch.Tensor:
        if modality_available is None:
            return torch.ones(features.shape[:4], dtype=torch.bool, device=features.device)
        value = torch.as_tensor(modality_available, dtype=torch.bool, device=features.device)
        if value.ndim == 3 and tuple(value.shape) == tuple(features.shape[:3]):
            return value.unsqueeze(3).expand(-1, -1, -1, int(features.shape[3]))
        if value.ndim != 4 or tuple(value.shape) != tuple(features.shape[:4]):
            raise ValueError(
                "amber_full_adaptive_mask_transformer modality_available must match "
                f"{tuple(features.shape[:3])} or {tuple(features.shape[:4])}."
            )
        return value

    def _fusion_key_padding_mask(self, availability: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, spatial_tokens = availability.shape
        per_step = torch.zeros(
            (batch_size, seq_len, self._tokens_per_step(int(spatial_tokens))),
            dtype=torch.bool,
            device=availability.device,
        )
        per_step[:, :, 1:] = ~availability.permute(0, 2, 1, 3).reshape(
            batch_size,
            seq_len,
            self.modality_count * int(spatial_tokens),
        )
        return per_step.reshape(batch_size, seq_len * self._tokens_per_step(int(spatial_tokens)))

    def _fusion_tokens(self, batch_size: int, seq_len: int, features: torch.Tensor) -> torch.Tensor:
        return self.fusion_token.to(device=features.device, dtype=features.dtype).expand(batch_size, seq_len, 1, -1)

    def _mask_tokens(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tokens = self.mask_tokens
        if self.mask_token_strategy == "learned_shared":
            tokens = tokens.expand(self.modality_count, -1)
        return tokens.to(device=device, dtype=dtype).view(1, self.modality_count, 1, 1, self.d_model)

    def _modality_indicator(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        logits = self.modality_indicator_logits.to(device=device, dtype=dtype)
        return torch.softmax(logits / self.modality_indicator_temperature, dim=0)

    def _class_query_embeddings(self, token_embeddings: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        queries = F.normalize(self.cma_modality_queries, dim=-1).to(
            device=token_embeddings.device,
            dtype=token_embeddings.dtype,
        )
        scores = torch.einsum("kc,bktsc->bkts", queries, token_embeddings) / math.sqrt(float(self.cma_dim))
        scores = scores.masked_fill(~availability, torch.finfo(scores.dtype).min)
        valid = availability.any(dim=3, keepdim=True)
        safe_scores = torch.where(valid, scores, torch.zeros_like(scores))
        weights = torch.softmax(safe_scores, dim=3).masked_fill(~availability, 0.0)
        denom = weights.sum(dim=3, keepdim=True).clamp_min(1e-8)
        return ((weights / denom).unsqueeze(-1) * token_embeddings).sum(dim=3)

    def _tokens_per_step(self, spatial_tokens: int) -> int:
        return 1 + self.modality_count * int(spatial_tokens)


def _available_mean(features: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
    weights = availability.to(dtype=features.dtype).unsqueeze(-1)
    denom = weights.sum(dim=2).clamp_min(1.0)
    return (features * weights).sum(dim=2) / denom


__all__ = ["AmberFullAdaptiveMaskTransformerCore"]
