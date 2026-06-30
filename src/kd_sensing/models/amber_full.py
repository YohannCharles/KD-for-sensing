from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.registries import REPRESENTATION_CORES


@REPRESENTATION_CORES.register("amber_full_adaptive_mask_transformer")
class AmberFullAdaptiveMaskTransformerCore(nn.Module):
    supports_missing_modality_metadata = True
    supports_reliability_metadata = True

    def __init__(
        self,
        d_model: int,
        modality_count: int,
        num_heads: int = 4,
        modality_layers: int = 1,
        fusion_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        output_dim: int | None = None,
        mask_token_strategy: str = "learned_per_modality",
        include_history_beam: bool = True,
        num_cma_queries: int = 4,
        cma_dim: int | None = None,
        cma_temperature: float = 0.2,
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
        self.output_dim = int(output_dim or d_model)
        self.mask_token_strategy = str(mask_token_strategy)
        self.include_history_beam = bool(include_history_beam)
        self.num_cma_queries = int(num_cma_queries)
        self.cma_dim = int(cma_dim or d_model)
        self.cma_temperature = float(cma_temperature)
        self.enable_auxiliary = bool(enable_auxiliary)
        self.auxiliary_loss_weights = dict(auxiliary_loss_weights or {})
        if min(self.d_model, self.modality_count, self.output_dim, self.max_seq_len, self.cma_dim) <= 0:
            raise ValueError("amber_full_adaptive_mask_transformer dimensions must be positive.")
        if self.num_heads <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads}).")
        if self.mask_token_strategy not in {"learned_per_modality", "learned_shared"}:
            raise ValueError("mask_token_strategy must be 'learned_per_modality' or 'learned_shared'.")

        token_count = self.modality_count if self.mask_token_strategy == "learned_per_modality" else 1
        self.mask_tokens = nn.Parameter(torch.zeros(token_count, self.d_model))
        self.modality_embedding = nn.Embedding(self.modality_count, self.d_model)
        self.time_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.history_beam_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
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
        self.history_branch = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads,
                dropout=float(dropout),
                dim_feedforward=max(self.d_model * 4, 64),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=max(self.modality_layers, 1),
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
        self.cma_queries = nn.Parameter(torch.zeros(self.num_cma_queries, self.cma_dim))
        self.last_amber_full_auxiliary: dict[str, Any] | None = None
        self.last_amber_full_attention_mask: torch.Tensor | None = None
        nn.init.trunc_normal_(self.mask_tokens, std=0.02)
        nn.init.trunc_normal_(self.fusion_token, std=0.02)
        nn.init.trunc_normal_(self.history_beam_token, std=0.02)
        nn.init.trunc_normal_(self.cma_queries, std=0.02)

    def forward(self, features: torch.Tensor, *, modality_available: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"amber_full_adaptive_mask_transformer expects [B,K,T,D], got {tuple(features.shape)}.")
        batch_size, modality_count, seq_len, d_model = features.shape
        if int(modality_count) != self.modality_count or int(d_model) != self.d_model:
            raise ValueError(
                "amber_full_adaptive_mask_transformer received incompatible shape: "
                f"expected K={self.modality_count}, D={self.d_model}, got {tuple(features.shape)}."
            )
        if int(seq_len) > self.max_seq_len:
            raise ValueError(f"AMBER full seq_len {int(seq_len)} exceeds max_seq_len={self.max_seq_len}.")

        availability = self._availability(modality_available, features)
        masked = torch.where(availability.unsqueeze(-1), features, self._mask_tokens(features.device, features.dtype))
        tokens = self._add_position(masked)
        modality_features = torch.stack(
            [branch(tokens[:, index]) for index, branch in enumerate(self.modality_branches)],
            dim=1,
        )
        frame_tokens = [self._fusion_tokens(batch_size, seq_len, features), modality_features.permute(0, 2, 1, 3)]
        if self.include_history_beam:
            history = self.history_branch(self._history_tokens(batch_size, seq_len, features))
            frame_tokens.append(history.unsqueeze(2))
        sequence = torch.cat(frame_tokens, dim=2).reshape(batch_size, seq_len * self._tokens_per_step(), self.d_model)
        key_padding = self._fusion_key_padding_mask(availability)
        memory = self.fusion_transformer(sequence, src_key_padding_mask=key_padding)
        memory = memory.view(batch_size, seq_len, self._tokens_per_step(), self.d_model)
        fusion = self.output_norm(memory[:, :, 0])
        self.last_amber_full_attention_mask = key_padding.detach()
        self.last_amber_full_auxiliary = self._auxiliary_payload(fusion, modality_features, availability)
        return self.output_projection(fusion)

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "amber_full_adaptive_mask_transformer",
            "reproduction_scope": "amber_full_local",
            "d_model": self.d_model,
            "output_dim": self.output_dim,
            "modality_count": self.modality_count,
            "enabled_modalities": "config_order",
            "history_beam_usage": "learned_history_beam_token" if self.include_history_beam else "disabled",
            "mask_strategy": "adaptive_key_padding_attention_mask",
            "mask_token_strategy": self.mask_token_strategy,
            "modality_specific_transformer_layers": self.modality_layers,
            "fusion_transformer_layers": self.fusion_layers,
            "cma_enabled": self.num_cma_queries > 0,
            "cma_temperature": self.cma_temperature,
            "auxiliary_loss_weights": self.auxiliary_loss_weights,
            "consumes_missing_modality_metadata": True,
            "consumes_reliability_metadata": True,
            "output_boundary": "outputs/analysis/local_baselines/amber_full_architecture/",
        }

    def _add_position(self, features: torch.Tensor) -> torch.Tensor:
        seq_len = int(features.shape[2])
        time = self.time_embedding(torch.arange(seq_len, device=features.device)).view(1, 1, seq_len, self.d_model)
        modality = self.modality_embedding(torch.arange(self.modality_count, device=features.device)).view(
            1, self.modality_count, 1, self.d_model
        )
        return self.input_dropout(self.input_norm(features + time + modality))

    def _auxiliary_payload(
        self,
        fusion: torch.Tensor,
        modality_features: torch.Tensor,
        availability: torch.Tensor,
    ) -> dict[str, Any] | None:
        if not (self.training and self.enable_auxiliary):
            return None
        modality_bt = modality_features.permute(0, 2, 1, 3).contiguous()
        fusion_embeddings = F.normalize(self.cma_projection(fusion), dim=-1)
        modality_embeddings = F.normalize(self.cma_projection(modality_bt), dim=-1)
        cma_logits = torch.einsum("btd,btkd->btk", fusion_embeddings, modality_embeddings) / max(self.cma_temperature, 1e-6)
        return {
            "modality_specific_features": modality_bt,
            "fusion_features": fusion,
            "fusion_token": fusion,
            "alignment_target": _available_mean(modality_bt, availability.permute(0, 2, 1).contiguous()),
            "cma_fusion_embeddings": fusion_embeddings,
            "cma_modality_embeddings": modality_embeddings,
            "cma_query_embeddings": F.normalize(self.cma_queries, dim=-1),
            "cma_logits": cma_logits,
            "availability_mask": availability,
            "mask_provenance": "input_valid_or_dropout_masks",
        }

    def _availability(self, modality_available: torch.Tensor | None, features: torch.Tensor) -> torch.Tensor:
        if modality_available is None:
            return torch.ones(features.shape[:3], dtype=torch.bool, device=features.device)
        value = torch.as_tensor(modality_available, dtype=torch.bool, device=features.device)
        if value.ndim != 3 or tuple(value.shape) != tuple(features.shape[:3]):
            raise ValueError(f"amber_full_adaptive_mask_transformer modality_available must match {tuple(features.shape[:3])}.")
        return value

    def _fusion_key_padding_mask(self, availability: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len = availability.shape
        per_step = torch.zeros((batch_size, seq_len, self._tokens_per_step()), dtype=torch.bool, device=availability.device)
        per_step[:, :, 1 : 1 + self.modality_count] = ~availability.permute(0, 2, 1)
        return per_step.reshape(batch_size, seq_len * self._tokens_per_step())

    def _fusion_tokens(self, batch_size: int, seq_len: int, features: torch.Tensor) -> torch.Tensor:
        return self.fusion_token.to(device=features.device, dtype=features.dtype).expand(batch_size, seq_len, 1, -1)

    def _history_tokens(self, batch_size: int, seq_len: int, features: torch.Tensor) -> torch.Tensor:
        return self.history_beam_token.to(device=features.device, dtype=features.dtype).expand(batch_size, seq_len, -1)

    def _mask_tokens(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tokens = self.mask_tokens
        if self.mask_token_strategy == "learned_shared":
            tokens = tokens.expand(self.modality_count, -1)
        return tokens.to(device=device, dtype=dtype).view(1, self.modality_count, 1, self.d_model)

    def _tokens_per_step(self) -> int:
        return 1 + self.modality_count + int(self.include_history_beam)


def _available_mean(features: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
    weights = availability.to(dtype=features.dtype).unsqueeze(-1)
    denom = weights.sum(dim=2).clamp_min(1.0)
    return (features * weights).sum(dim=2) / denom


__all__ = ["AmberFullAdaptiveMaskTransformerCore"]
