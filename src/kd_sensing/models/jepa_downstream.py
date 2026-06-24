from typing import Any

import torch
import torch.nn as nn

from kd_sensing.registries import JEPA_DOWNSTREAM_POOLERS


@JEPA_DOWNSTREAM_POOLERS.register("mean")
class MeanPatchPooler(nn.Module):
    pooler_type = "mean"
    required_context_modalities: tuple[str, ...] = ()
    context_feature_source = "none"
    context_feature_kwargs: dict[str, str] = {}

    def __init__(self, latent_dim: int | None = None, output_mode: str = "frame", **_: Any) -> None:
        super().__init__()
        self.latent_dim = None if latent_dim is None else int(latent_dim)
        self.output_mode = _normalize_pooler_output_mode(output_mode)
        self.last_diagnostics: dict[str, Any] = {}

    def forward(
        self,
        patch_tokens: torch.Tensor,
        condition_features: torch.Tensor | None = None,
        *,
        token_metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del condition_features
        if patch_tokens.ndim != 4:
            raise ValueError(
                f"MeanPatchPooler patch tokens must have shape [B, T, N, D], got {tuple(patch_tokens.shape)}."
            )
        if self.latent_dim is not None and int(patch_tokens.shape[-1]) != self.latent_dim:
            raise ValueError(
                f"MeanPatchPooler expected patch token dim {self.latent_dim}, got {tuple(patch_tokens.shape)}."
            )
        self.last_diagnostics = _pooler_token_diagnostics(
            token_metadata=token_metadata,
            attention_map=None,
            output_mode=self.output_mode,
            query_count=int(patch_tokens.shape[2]) if self.output_mode == "tokens" else 1,
            condition_source="none",
        )
        if self.output_mode == "tokens":
            return patch_tokens
        return patch_tokens.mean(dim=2)

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": self.pooler_type,
            "latent_dim": self.latent_dim,
            "output_mode": self.output_mode,
            "requires_condition": False,
        }


@JEPA_DOWNSTREAM_POOLERS.register("gps_query_attention")
class GPSQueryPool(nn.Module):
    pooler_type = "gps_query_attention"

    def __init__(
        self,
        *,
        latent_dim: int = 64,
        condition_dim: int = 64,
        k_queries: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
        return_attention: bool = False,
        hidden_dim: int | None = None,
        condition_source: str = "projected_gps",
        output_mode: str = "frame",
        per_head_attention: bool = False,
        return_attention_heads: bool | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.k_queries = int(k_queries)
        self.num_heads = int(num_heads)
        self.return_attention = bool(return_attention)
        self.per_head_attention = bool(per_head_attention if return_attention_heads is None else return_attention_heads)
        self.output_mode = _normalize_pooler_output_mode(output_mode)
        self.condition_source = _normalize_condition_source(condition_source)
        self.required_context_modalities = ("gps",)
        self.context_feature_source = _context_feature_source_kind(self.condition_source)
        self.context_feature_kwargs = {"gps": "gps_condition_features"}
        if self.latent_dim <= 0 or self.condition_dim <= 0:
            raise ValueError("GPSQueryPool latent_dim and condition_dim must be positive.")
        if self.k_queries <= 0:
            raise ValueError(f"GPSQueryPool k_queries must be positive, got {k_queries}.")
        if self.num_heads <= 0:
            raise ValueError(f"GPSQueryPool num_heads must be positive, got {num_heads}.")
        if self.latent_dim % self.num_heads != 0:
            raise ValueError(
                f"GPSQueryPool latent_dim ({self.latent_dim}) must be divisible by num_heads ({self.num_heads})."
            )
        hidden = int(hidden_dim or max(self.condition_dim, self.latent_dim))
        self.gps_to_q = nn.Sequential(
            nn.LayerNorm(self.condition_dim),
            nn.Linear(self.condition_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.k_queries * self.latent_dim),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.latent_dim)
        self.last_attention_map: torch.Tensor | None = None
        self.last_attention_heads: torch.Tensor | None = None
        self.last_diagnostics: dict[str, Any] = {}

    def forward(
        self,
        patch_tokens: torch.Tensor,
        condition_features: torch.Tensor | None = None,
        *,
        return_attention: bool | None = None,
        token_metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if condition_features is None:
            raise ValueError("GPSQueryPool requires GPS condition features.")
        self._validate_inputs(patch_tokens, condition_features)
        batch_size, seq_len, num_tokens, latent_dim = patch_tokens.shape
        condition = condition_features.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        flat_tokens = patch_tokens.reshape(batch_size * seq_len, num_tokens, latent_dim)
        flat_condition = condition.reshape(batch_size * seq_len, self.condition_dim)
        queries = self.gps_to_q(flat_condition).reshape(batch_size * seq_len, self.k_queries, self.latent_dim)
        with_attention = self.return_attention if return_attention is None else bool(return_attention)
        attended, attention = self.attention(
            queries,
            flat_tokens,
            flat_tokens,
            need_weights=with_attention,
            average_attn_weights=not self.per_head_attention,
        )
        attended = self.output_norm(attended)
        if self.output_mode == "tokens":
            pooled = attended.reshape(batch_size, seq_len, self.k_queries, self.latent_dim)
        else:
            pooled_queries = attended.mean(dim=1)
            pooled = pooled_queries.reshape(batch_size, seq_len, self.latent_dim)
        if not with_attention:
            self.last_attention_map = None
            self.last_attention_heads = None
            self.last_diagnostics = _pooler_token_diagnostics(
                token_metadata=token_metadata,
                attention_map=None,
                attended_tokens=pooled if self.output_mode == "tokens" else pooled.unsqueeze(2),
                output_mode=self.output_mode,
                query_count=self.k_queries,
                condition_source=self.context_feature_source,
            )
            return pooled
        if attention is None:
            raise RuntimeError("GPSQueryPool requested attention diagnostics but MultiheadAttention returned None.")
        if self.per_head_attention:
            attention_heads = attention.detach().reshape(batch_size, seq_len, self.num_heads, self.k_queries, num_tokens)
            attention_map = attention_heads.mean(dim=2)
        else:
            attention_heads = None
            attention_map = attention.detach().reshape(batch_size, seq_len, self.k_queries, num_tokens)
        self.last_attention_map = attention_map
        self.last_attention_heads = attention_heads
        self.last_diagnostics = _pooler_token_diagnostics(
            token_metadata=token_metadata,
            attention_map=attention_map,
            attended_tokens=pooled
            if self.output_mode == "tokens"
            else attended.reshape(batch_size, seq_len, self.k_queries, latent_dim),
            output_mode=self.output_mode,
            query_count=self.k_queries,
            condition_source=self.context_feature_source,
            attention_heads=attention_heads,
            attention_head_aggregation="per_head" if self.per_head_attention else "averaged",
        )
        return pooled, attention_map

    def _validate_inputs(self, patch_tokens: torch.Tensor, condition_features: torch.Tensor) -> None:
        if patch_tokens.ndim != 4:
            raise ValueError(f"GPSQueryPool patch tokens must have shape [B, T, N, D], got {tuple(patch_tokens.shape)}.")
        if condition_features.ndim != 3:
            raise ValueError(
                "GPSQueryPool condition features must have shape [B, T, C], "
                f"got {tuple(condition_features.shape)} for patch tokens {tuple(patch_tokens.shape)}."
            )
        if int(patch_tokens.shape[-1]) != self.latent_dim:
            raise ValueError(
                f"GPSQueryPool expected patch token latent dim {self.latent_dim}, "
                f"got patch tokens {tuple(patch_tokens.shape)}."
            )
        if int(condition_features.shape[-1]) != self.condition_dim:
            raise ValueError(
                f"GPSQueryPool expected condition feature dim {self.condition_dim}, "
                f"got condition features {tuple(condition_features.shape)} for patch tokens {tuple(patch_tokens.shape)}."
            )
        if tuple(patch_tokens.shape[:2]) != tuple(condition_features.shape[:2]):
            raise ValueError(
                "GPSQueryPool patch tokens and condition features must share batch/time dimensions; "
                f"patch tokens shape {tuple(patch_tokens.shape)}, "
                f"condition feature shape {tuple(condition_features.shape)}."
            )

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": self.pooler_type,
            "latent_dim": self.latent_dim,
            "condition_dim": self.condition_dim,
            "k_queries": self.k_queries,
            "num_heads": self.num_heads,
            "condition_source": self.condition_source,
            "output_mode": self.output_mode,
            "k_tokens": self.k_queries if self.output_mode == "tokens" else 1,
            "return_attention": self.return_attention,
            "per_head_attention": self.per_head_attention,
            "attention_diagnostics": self.return_attention,
            "requires_condition": True,
            "required_context_modalities": list(self.required_context_modalities),
            "context_feature_source": self.context_feature_source,
        }


@JEPA_DOWNSTREAM_POOLERS.register("learned_query_attention")
class LearnedQueryPool(nn.Module):
    pooler_type = "learned_query_attention"
    required_context_modalities: tuple[str, ...] = ()
    context_feature_source = "none"
    context_feature_kwargs: dict[str, str] = {}

    def __init__(
        self,
        *,
        latent_dim: int = 64,
        k_queries: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        return_attention: bool = False,
        output_mode: str = "frame",
        **_: Any,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.k_queries = int(k_queries)
        self.num_heads = int(num_heads)
        self.return_attention = bool(return_attention)
        self.output_mode = _normalize_pooler_output_mode(output_mode)
        if self.latent_dim <= 0 or self.k_queries <= 0:
            raise ValueError("LearnedQueryPool latent_dim and k_queries must be positive.")
        if self.num_heads <= 0 or self.latent_dim % self.num_heads != 0:
            raise ValueError("LearnedQueryPool num_heads must be positive and divide latent_dim.")
        self.query_tokens = nn.Parameter(torch.zeros(self.k_queries, self.latent_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.latent_dim)
        self.last_attention_map: torch.Tensor | None = None
        self.last_diagnostics: dict[str, Any] = {}

    def forward(
        self,
        patch_tokens: torch.Tensor,
        condition_features: torch.Tensor | None = None,
        *,
        return_attention: bool | None = None,
        token_metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del condition_features
        if patch_tokens.ndim != 4:
            raise ValueError(
                f"LearnedQueryPool patch tokens must have shape [B, T, N, D], got {tuple(patch_tokens.shape)}."
            )
        if int(patch_tokens.shape[-1]) != self.latent_dim:
            raise ValueError(
                f"LearnedQueryPool expected patch token latent dim {self.latent_dim}, "
                f"got patch tokens {tuple(patch_tokens.shape)}."
            )
        batch_size, seq_len, num_tokens, latent_dim = patch_tokens.shape
        flat_tokens = patch_tokens.reshape(batch_size * seq_len, num_tokens, latent_dim)
        queries = self.query_tokens.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        queries = queries.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        with_attention = self.return_attention if return_attention is None else bool(return_attention)
        attended, attention = self.attention(
            queries,
            flat_tokens,
            flat_tokens,
            need_weights=with_attention,
            average_attn_weights=True,
        )
        attended = self.output_norm(attended)
        if self.output_mode == "tokens":
            pooled = attended.reshape(batch_size, seq_len, self.k_queries, self.latent_dim)
        else:
            pooled = attended.mean(dim=1).reshape(batch_size, seq_len, self.latent_dim)
        attended_tokens = pooled if self.output_mode == "tokens" else attended.reshape(
            batch_size, seq_len, self.k_queries, latent_dim
        )
        if not with_attention:
            self.last_attention_map = None
            self.last_diagnostics = _pooler_token_diagnostics(
                token_metadata=token_metadata,
                attention_map=None,
                attended_tokens=attended_tokens,
                output_mode=self.output_mode,
                query_count=self.k_queries,
                condition_source="none",
            )
            return pooled
        if attention is None:
            raise RuntimeError("LearnedQueryPool requested attention diagnostics but MultiheadAttention returned None.")
        attention_map = attention.detach().reshape(batch_size, seq_len, self.k_queries, num_tokens)
        self.last_attention_map = attention_map
        self.last_diagnostics = _pooler_token_diagnostics(
            token_metadata=token_metadata,
            attention_map=attention_map,
            attended_tokens=attended_tokens,
            output_mode=self.output_mode,
            query_count=self.k_queries,
            condition_source="none",
        )
        return pooled, attention_map

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": self.pooler_type,
            "latent_dim": self.latent_dim,
            "k_queries": self.k_queries,
            "num_heads": self.num_heads,
            "condition_source": "none",
            "output_mode": self.output_mode,
            "k_tokens": self.k_queries if self.output_mode == "tokens" else 1,
            "return_attention": self.return_attention,
            "attention_diagnostics": self.return_attention,
            "requires_condition": False,
            "required_context_modalities": [],
            "context_feature_source": "none",
        }


@JEPA_DOWNSTREAM_POOLERS.register("self_attention")
class SelfAttentionPool(nn.Module):
    pooler_type = "self_attention"
    required_context_modalities: tuple[str, ...] = ()
    context_feature_source = "none"
    context_feature_kwargs: dict[str, str] = {}

    def __init__(
        self,
        *,
        latent_dim: int = 64,
        k_tokens: int | None = None,
        k_queries: int | None = None,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_mode: str = "frame",
        **_: Any,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.k_tokens = int(k_tokens if k_tokens is not None else k_queries if k_queries is not None else 2)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.output_mode = _normalize_pooler_output_mode(output_mode)
        if self.latent_dim <= 0 or self.k_tokens <= 0:
            raise ValueError("SelfAttentionPool latent_dim and k_tokens must be positive.")
        if self.num_heads <= 0 or self.latent_dim % self.num_heads != 0:
            raise ValueError("SelfAttentionPool num_heads must be positive and divide latent_dim.")
        self.summary_tokens = nn.Parameter(torch.zeros(self.k_tokens, self.latent_dim))
        nn.init.trunc_normal_(self.summary_tokens, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.latent_dim * 4,
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(self.num_layers, 0))
        self.output_norm = nn.LayerNorm(self.latent_dim)
        self.last_attention_map: torch.Tensor | None = None
        self.last_diagnostics: dict[str, Any] = {}

    def forward(
        self,
        patch_tokens: torch.Tensor,
        condition_features: torch.Tensor | None = None,
        *,
        token_metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del condition_features
        if patch_tokens.ndim != 4:
            raise ValueError(
                f"SelfAttentionPool patch tokens must have shape [B, T, N, D], got {tuple(patch_tokens.shape)}."
            )
        if int(patch_tokens.shape[-1]) != self.latent_dim:
            raise ValueError(
                f"SelfAttentionPool expected patch token latent dim {self.latent_dim}, "
                f"got patch tokens {tuple(patch_tokens.shape)}."
            )
        batch_size, seq_len, num_tokens, latent_dim = patch_tokens.shape
        flat_tokens = patch_tokens.reshape(batch_size * seq_len, num_tokens, latent_dim)
        summary = self.summary_tokens.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        summary = summary.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        encoded = self.encoder(torch.cat([summary, flat_tokens], dim=1))[:, : self.k_tokens]
        encoded = self.output_norm(encoded)
        if self.output_mode == "tokens":
            pooled = encoded.reshape(batch_size, seq_len, self.k_tokens, self.latent_dim)
            attended_tokens = pooled
        else:
            attended_tokens = encoded.reshape(batch_size, seq_len, self.k_tokens, self.latent_dim)
            pooled = encoded.mean(dim=1).reshape(batch_size, seq_len, self.latent_dim)
        self.last_attention_map = None
        self.last_diagnostics = _pooler_token_diagnostics(
            token_metadata=token_metadata,
            attention_map=None,
            attended_tokens=attended_tokens,
            output_mode=self.output_mode,
            query_count=self.k_tokens,
            condition_source="none",
        )
        return pooled

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": self.pooler_type,
            "latent_dim": self.latent_dim,
            "k_tokens": self.k_tokens if self.output_mode == "tokens" else 1,
            "k_queries": self.k_tokens,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "condition_source": "none",
            "output_mode": self.output_mode,
            "return_attention": False,
            "attention_diagnostics": False,
            "requires_condition": False,
            "required_context_modalities": [],
            "context_feature_source": "none",
        }


@JEPA_DOWNSTREAM_POOLERS.register("hybrid_residual_query")
class HybridResidualQueryPool(nn.Module):
    pooler_type = "hybrid_residual_query"

    def __init__(
        self,
        *,
        latent_dim: int = 64,
        condition_dim: int = 64,
        content_queries: int = 2,
        gps_queries: int | None = None,
        k_queries: int | None = None,
        num_heads: int = 4,
        dropout: float = 0.0,
        hidden_dim: int | None = None,
        condition_source: str = "projected_gps",
        require_condition: bool | None = None,
        gps_required: bool | None = None,
        residual_alpha_init: float = 0.1,
        return_attention: bool = False,
        output_mode: str = "frame",
        per_head_attention: bool = False,
        return_attention_heads: bool | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.content_queries = int(content_queries)
        self.gps_queries = int(gps_queries if gps_queries is not None else k_queries if k_queries is not None else 2)
        self.num_heads = int(num_heads)
        self.return_attention = bool(return_attention)
        self.per_head_attention = bool(per_head_attention if return_attention_heads is None else return_attention_heads)
        self.output_mode = _normalize_pooler_output_mode(output_mode)
        self.condition_source = _normalize_condition_source(condition_source)
        if require_condition is None:
            require_condition = True if gps_required is None else bool(gps_required)
        self.require_condition = bool(require_condition)
        self.required_context_modalities = ("gps",) if self.require_condition else ()
        self.context_feature_source = _context_feature_source_kind(self.condition_source) if self.require_condition else "none"
        self.context_feature_kwargs = {"gps": "gps_condition_features"} if self.require_condition else {}
        if self.latent_dim <= 0 or self.condition_dim <= 0:
            raise ValueError("HybridResidualQueryPool latent_dim and condition_dim must be positive.")
        if self.content_queries <= 0 or self.gps_queries <= 0:
            raise ValueError("HybridResidualQueryPool content_queries and gps_queries must be positive.")
        if self.num_heads <= 0:
            raise ValueError(f"HybridResidualQueryPool num_heads must be positive, got {num_heads}.")
        if self.latent_dim % self.num_heads != 0:
            raise ValueError(
                "HybridResidualQueryPool latent_dim "
                f"({self.latent_dim}) must be divisible by num_heads ({self.num_heads})."
            )
        hidden = int(hidden_dim or max(self.latent_dim * 2, self.condition_dim))
        self.content_query = nn.Parameter(torch.empty(self.content_queries, self.latent_dim))
        self.content_attention = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.gps_to_q = nn.Sequential(
            nn.LayerNorm(self.condition_dim),
            nn.Linear(self.condition_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.gps_queries * self.latent_dim),
        )
        self.gps_attention = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.residual_mlp = nn.Sequential(
            nn.LayerNorm(self.latent_dim * 2),
            nn.Linear(self.latent_dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.latent_dim),
        )
        self.output_norm = nn.LayerNorm(self.latent_dim)
        self.residual_alpha_init = float(residual_alpha_init)
        self.residual_alpha = nn.Parameter(torch.tensor(self.residual_alpha_init, dtype=torch.float32))
        self.last_attention_map: torch.Tensor | None = None
        self.last_attention_maps: dict[str, torch.Tensor] = {}
        self.last_attention_heads: dict[str, torch.Tensor] = {}
        self.last_diagnostics: dict[str, Any] = {}
        nn.init.trunc_normal_(self.content_query, std=0.02)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        condition_features: torch.Tensor | None = None,
        *,
        return_attention: bool | None = None,
        token_metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        self._validate_patch_tokens(patch_tokens)
        batch_size, seq_len, num_tokens, latent_dim = patch_tokens.shape
        mean_latent = patch_tokens.mean(dim=2)
        flat_tokens = patch_tokens.reshape(batch_size * seq_len, num_tokens, latent_dim)
        with_attention = self.return_attention if return_attention is None else bool(return_attention)

        content_query = self.content_query.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        content_query = content_query.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        content_attended, content_attention = self.content_attention(
            content_query,
            flat_tokens,
            flat_tokens,
            need_weights=with_attention,
            average_attn_weights=not self.per_head_attention,
        )
        content_latent = content_attended.mean(dim=1).reshape(batch_size, seq_len, latent_dim)

        gps_latent = mean_latent
        gps_attention_map: torch.Tensor | None = None
        gps_attended_tokens: torch.Tensor | None = None
        gps_available = condition_features is not None
        if condition_features is None:
            if self.require_condition:
                raise ValueError("HybridResidualQueryPool requires GPS condition features.")
        else:
            self._validate_condition_features(patch_tokens, condition_features)
            condition = condition_features.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
            flat_condition = condition.reshape(batch_size * seq_len, self.condition_dim)
            gps_query = self.gps_to_q(flat_condition).reshape(batch_size * seq_len, self.gps_queries, self.latent_dim)
            gps_attended, gps_attention = self.gps_attention(
                gps_query,
                flat_tokens,
                flat_tokens,
                need_weights=with_attention,
                average_attn_weights=not self.per_head_attention,
            )
            gps_latent = gps_attended.mean(dim=1).reshape(batch_size, seq_len, latent_dim)
            gps_attended_tokens = gps_attended
            if with_attention:
                if gps_attention is None:
                    raise RuntimeError(
                        "HybridResidualQueryPool requested GPS attention diagnostics but attention returned None."
                    )
                if self.per_head_attention:
                    gps_attention_heads = gps_attention.detach().reshape(
                        batch_size,
                        seq_len,
                        self.num_heads,
                        self.gps_queries,
                        num_tokens,
                    )
                    gps_attention_map = gps_attention_heads.mean(dim=2)
                else:
                    gps_attention_map = gps_attention.detach().reshape(batch_size, seq_len, self.gps_queries, num_tokens)

        content_delta = content_latent - mean_latent
        gps_delta = gps_latent - mean_latent if gps_available else torch.zeros_like(mean_latent)
        residual = self.residual_mlp(torch.cat([content_delta, gps_delta], dim=-1))
        alpha = self.residual_alpha.to(dtype=patch_tokens.dtype, device=patch_tokens.device).clamp(0.0, 1.0)
        pooled = self.output_norm(mean_latent + alpha * residual)
        if self.output_mode == "tokens":
            query_pieces = [content_attended]
            if gps_attended_tokens is not None:
                query_pieces.append(gps_attended_tokens)
            pooled = self.output_norm(torch.cat(query_pieces, dim=1)).reshape(batch_size, seq_len, -1, latent_dim)

        attention_maps: dict[str, torch.Tensor] = {}
        attention_heads: dict[str, torch.Tensor] = {}
        if with_attention:
            if content_attention is None:
                raise RuntimeError(
                    "HybridResidualQueryPool requested content attention diagnostics but attention returned None."
                )
            if self.per_head_attention:
                content_heads = content_attention.detach().reshape(
                    batch_size,
                    seq_len,
                    self.num_heads,
                    self.content_queries,
                    num_tokens,
                )
                attention_heads["content"] = content_heads
                attention_maps["content"] = content_heads.mean(dim=2)
            else:
                attention_maps["content"] = content_attention.detach().reshape(
                    batch_size,
                    seq_len,
                    self.content_queries,
                    num_tokens,
                )
            if gps_attention_map is not None:
                attention_maps["gps"] = gps_attention_map
                if self.per_head_attention and "gps_attention_heads" in locals():
                    attention_heads["gps"] = gps_attention_heads
        self.last_attention_maps = attention_maps
        self.last_attention_heads = attention_heads
        self.last_attention_map = attention_maps["gps"] if "gps" in attention_maps else attention_maps.get("content")
        self.last_diagnostics = {
            "type": self.pooler_type,
            "gps_condition_available": bool(gps_available),
            "requires_condition": self.require_condition,
            "residual_alpha": float(alpha.detach().cpu().item()),
            "residual_alpha_init": self.residual_alpha_init,
            "content_queries": self.content_queries,
            "gps_queries": self.gps_queries,
            "attention_diagnostics": bool(with_attention),
            "attention_head_aggregation": "per_head" if self.per_head_attention else "averaged",
            "last_attention_map_source": "gps" if "gps" in attention_maps else ("content" if "content" in attention_maps else "unavailable"),
            "last_attention_map_exposes_gps_branch": "gps" in attention_maps,
            "branch_attention": _branch_attention_diagnostics(
                attention_maps,
                attention_heads=attention_heads,
                unavailable={
                    "gps": "" if gps_available else "condition_features_unavailable",
                    "content": "",
                },
                exposed_branch="gps" if "gps" in attention_maps else "content",
            ),
            "output_mode": self.output_mode,
            "k_tokens": int(pooled.shape[2]) if pooled.ndim == 4 else 1,
            **_pooler_token_diagnostics(
                token_metadata=token_metadata,
                attention_map=self.last_attention_map,
                output_mode=self.output_mode,
                query_count=int(pooled.shape[2]) if pooled.ndim == 4 else 1,
                condition_source=self.context_feature_source,
            ),
        }
        return pooled

    def _validate_patch_tokens(self, patch_tokens: torch.Tensor) -> None:
        if patch_tokens.ndim != 4:
            raise ValueError(
                f"HybridResidualQueryPool patch tokens must have shape [B, T, N, D], got {tuple(patch_tokens.shape)}."
            )
        if int(patch_tokens.shape[-1]) != self.latent_dim:
            raise ValueError(
                f"HybridResidualQueryPool expected patch token latent dim {self.latent_dim}, "
                f"got patch tokens {tuple(patch_tokens.shape)}."
            )

    def _validate_condition_features(self, patch_tokens: torch.Tensor, condition_features: torch.Tensor) -> None:
        if condition_features.ndim != 3:
            raise ValueError(
                "HybridResidualQueryPool condition features must have shape [B, T, C], "
                f"got {tuple(condition_features.shape)} for patch tokens {tuple(patch_tokens.shape)}."
            )
        if int(condition_features.shape[-1]) != self.condition_dim:
            raise ValueError(
                f"HybridResidualQueryPool expected condition feature dim {self.condition_dim}, "
                f"got condition features {tuple(condition_features.shape)}."
            )
        if tuple(patch_tokens.shape[:2]) != tuple(condition_features.shape[:2]):
            raise ValueError(
                "HybridResidualQueryPool patch tokens and condition features must share batch/time dimensions; "
                f"patch tokens shape {tuple(patch_tokens.shape)}, "
                f"condition feature shape {tuple(condition_features.shape)}."
            )

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": self.pooler_type,
            "latent_dim": self.latent_dim,
            "condition_dim": self.condition_dim,
            "content_queries": self.content_queries,
            "gps_queries": self.gps_queries,
            "num_heads": self.num_heads,
            "condition_source": self.condition_source,
            "output_mode": self.output_mode,
            "k_tokens": self.content_queries + self.gps_queries if self.output_mode == "tokens" else 1,
            "return_attention": self.return_attention,
            "per_head_attention": self.per_head_attention,
            "attention_diagnostics": self.return_attention,
            "requires_condition": self.require_condition,
            "residual_alpha_init": self.residual_alpha_init,
            "residual_alpha": float(self.residual_alpha.detach().cpu().clamp(0.0, 1.0).item()),
            "required_context_modalities": list(self.required_context_modalities),
            "context_feature_source": self.context_feature_source,
        }


@JEPA_DOWNSTREAM_POOLERS.register("predictive_gps_query")
@JEPA_DOWNSTREAM_POOLERS.register("predictive_gps_query_plus_plus")
class PredictiveGPSQueryPool(nn.Module):
    pooler_type = "predictive_gps_query"
    forbidden_condition_fields = (
        "condition",
        "predictive_condition_id",
        "predictive_condition",
        "gps_condition",
        "image_condition",
        "c_idx",
        "d_idx",
    )

    def __init__(
        self,
        *,
        latent_dim: int = 64,
        condition_dim: int = 64,
        content_queries: int = 2,
        gps_queries: int | None = None,
        k_queries: int | None = None,
        num_heads: int = 4,
        dropout: float = 0.0,
        hidden_dim: int | None = None,
        condition_source: str = "projected_gps",
        residual_scale_init: float | None = None,
        residual_alpha_init: float | None = None,
        temporal_predictor: dict[str, Any] | str | None = None,
        reliability_gate: dict[str, Any] | str | None = None,
        return_attention: bool = False,
        output_mode: str = "frame",
        per_head_attention: bool = False,
        return_attention_heads: bool | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.content_queries = int(content_queries)
        self.gps_queries = int(gps_queries if gps_queries is not None else k_queries if k_queries is not None else 2)
        self.num_heads = int(num_heads)
        self.return_attention = bool(return_attention)
        self.per_head_attention = bool(per_head_attention if return_attention_heads is None else return_attention_heads)
        self.output_mode = _normalize_pooler_output_mode(output_mode)
        self.condition_source = _normalize_condition_source(condition_source)
        self.required_context_modalities = ("gps",)
        self.context_feature_source = _context_feature_source_kind(self.condition_source)
        self.context_feature_kwargs = {"gps": "gps_condition_features"}
        if self.latent_dim <= 0 or self.condition_dim <= 0:
            raise ValueError("PredictiveGPSQueryPool latent_dim and condition_dim must be positive.")
        if self.content_queries <= 0 or self.gps_queries <= 0:
            raise ValueError("PredictiveGPSQueryPool content_queries and gps_queries must be positive.")
        if self.num_heads <= 0:
            raise ValueError(f"PredictiveGPSQueryPool num_heads must be positive, got {num_heads}.")
        if self.latent_dim % self.num_heads != 0:
            raise ValueError(
                "PredictiveGPSQueryPool latent_dim "
                f"({self.latent_dim}) must be divisible by num_heads ({self.num_heads})."
            )
        hidden = int(hidden_dim or max(self.latent_dim * 2, self.condition_dim))
        self.temporal_predictor_config = _normalize_predictive_temporal_config(temporal_predictor)
        self.reliability_gate_config = _normalize_predictive_gate_config(reliability_gate)
        self.temporal_predictor_type = str(self.temporal_predictor_config["type"])
        self.history_window = int(self.temporal_predictor_config["history_window"])
        self.temporal_fallback = str(self.temporal_predictor_config["insufficient_history"])
        self.reliability_gate_type = str(self.reliability_gate_config["type"])

        self.content_query = nn.Parameter(torch.empty(self.content_queries, self.latent_dim))
        self.content_attention = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.gps_to_q = nn.Sequential(
            nn.LayerNorm(self.condition_dim),
            nn.Linear(self.condition_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.gps_queries * self.latent_dim),
        )
        self.gps_attention = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        if self.temporal_predictor_type == "gru":
            self.temporal_gru = nn.GRU(self.latent_dim, self.latent_dim, batch_first=True)
            self.temporal_projection = nn.Linear(self.latent_dim, self.latent_dim)
        elif self.temporal_predictor_type in {"mean", "disabled", "none"}:
            self.temporal_gru = None
            self.temporal_projection = nn.Identity()
        else:
            raise ValueError("PredictiveGPSQueryPool temporal_predictor.type must be gru, mean, or disabled.")
        gate_hidden = int(self.reliability_gate_config.get("hidden_dim") or max(8, self.latent_dim // 2))
        gate_input_dim = 6
        if self.reliability_gate_type in {"mlp", "linear"}:
            self.reliability_gate = nn.Sequential(
                nn.LayerNorm(gate_input_dim),
                nn.Linear(gate_input_dim, gate_hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(gate_hidden, 3),
            )
        elif self.reliability_gate_type in {"uniform", "fixed"}:
            self.reliability_gate = None
        else:
            raise ValueError("PredictiveGPSQueryPool reliability_gate.type must be mlp, linear, uniform, or fixed.")
        scale_init = residual_scale_init if residual_scale_init is not None else residual_alpha_init
        self.residual_scale_init = float(0.1 if scale_init is None else scale_init)
        self.residual_scale = nn.Parameter(torch.tensor(self.residual_scale_init, dtype=torch.float32))
        self.output_norm = nn.LayerNorm(self.latent_dim)
        self.last_attention_map: torch.Tensor | None = None
        self.last_attention_maps: dict[str, torch.Tensor] = {}
        self.last_attention_heads: dict[str, torch.Tensor] = {}
        self.last_diagnostics: dict[str, Any] = {}
        self.last_current_latent: torch.Tensor | None = None
        self.last_gps_residual_latent: torch.Tensor | None = None
        self.last_temporal_predicted_latent: torch.Tensor | None = None
        self.last_gate_weights: torch.Tensor | None = None
        nn.init.trunc_normal_(self.content_query, std=0.02)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        condition_features: torch.Tensor | None = None,
        *,
        image_valid_mask: torch.Tensor | None = None,
        image_observability_score: torch.Tensor | None = None,
        gps_valid_mask: torch.Tensor | None = None,
        gps_counterfactual_mask: torch.Tensor | None = None,
        benchmark_condition_metadata: dict[str, Any] | None = None,
        return_attention: bool | None = None,
        token_metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if condition_features is None:
            raise ValueError("PredictiveGPSQueryPool requires GPS condition features.")
        self._validate_inputs(patch_tokens, condition_features)
        batch_size, seq_len, num_tokens, latent_dim = patch_tokens.shape
        flat_tokens = patch_tokens.reshape(batch_size * seq_len, num_tokens, latent_dim)
        with_attention = self.return_attention if return_attention is None else bool(return_attention)

        content_query = self.content_query.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        content_query = content_query.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        content_attended, content_attention = self.content_attention(
            content_query,
            flat_tokens,
            flat_tokens,
            need_weights=with_attention,
            average_attn_weights=not self.per_head_attention,
        )
        content_latent = content_attended.mean(dim=1).reshape(batch_size, seq_len, latent_dim)

        condition = condition_features.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        flat_condition = condition.reshape(batch_size * seq_len, self.condition_dim)
        gps_query = self.gps_to_q(flat_condition).reshape(batch_size * seq_len, self.gps_queries, self.latent_dim)
        gps_attended, gps_attention = self.gps_attention(
            gps_query,
            flat_tokens,
            flat_tokens,
            need_weights=with_attention,
            average_attn_weights=not self.per_head_attention,
        )
        gps_latent = gps_attended.mean(dim=1).reshape(batch_size, seq_len, latent_dim)
        scale = self.residual_scale.to(dtype=patch_tokens.dtype, device=patch_tokens.device).clamp(0.0, 1.0)
        gps_residual_latent = content_latent + scale * (gps_latent - content_latent)

        temporal_latent, temporal_metadata = self._predict_temporal_latent(content_latent)
        gate_weights, gate_metadata = self._gate_weights(
            content_latent,
            temporal_latent,
            gps_residual_latent,
            temporal_available=torch.as_tensor(
                temporal_metadata["availability_mask"],
                dtype=torch.bool,
                device=patch_tokens.device,
            ),
            gps_available_mask=_mask_or_default(
                gps_valid_mask,
                batch_size=batch_size,
                steps=seq_len,
                device=patch_tokens.device,
                dtype=torch.bool,
                default=True,
                name="gps_valid_mask",
            ),
            image_valid_mask=_mask_or_default(
                image_valid_mask,
                batch_size=batch_size,
                steps=seq_len,
                device=patch_tokens.device,
                dtype=torch.bool,
                default=True,
                name="image_valid_mask",
            ),
            image_observability_score=_mask_or_default(
                image_observability_score,
                batch_size=batch_size,
                steps=seq_len,
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
                default=1.0,
                name="image_observability_score",
            ),
        )
        stacked = torch.stack([content_latent, temporal_latent, gps_residual_latent], dim=2)
        pooled = self.output_norm((stacked * gate_weights.unsqueeze(-1)).sum(dim=2))
        if self.output_mode == "tokens":
            pooled = self.output_norm(torch.cat([content_attended, gps_attended], dim=1)).reshape(
                batch_size,
                seq_len,
                self.content_queries + self.gps_queries,
                latent_dim,
            )

        attention_maps: dict[str, torch.Tensor] = {}
        attention_heads: dict[str, torch.Tensor] = {}
        if with_attention:
            if content_attention is None or gps_attention is None:
                raise RuntimeError("PredictiveGPSQueryPool requested attention diagnostics but attention returned None.")
            if self.per_head_attention:
                content_heads = content_attention.detach().reshape(
                    batch_size,
                    seq_len,
                    self.num_heads,
                    self.content_queries,
                    num_tokens,
                )
                gps_heads = gps_attention.detach().reshape(
                    batch_size,
                    seq_len,
                    self.num_heads,
                    self.gps_queries,
                    num_tokens,
                )
                attention_heads["content"] = content_heads
                attention_heads["gps"] = gps_heads
                attention_maps["content"] = content_heads.mean(dim=2)
                attention_maps["gps"] = gps_heads.mean(dim=2)
            else:
                attention_maps["content"] = content_attention.detach().reshape(
                    batch_size,
                    seq_len,
                    self.content_queries,
                    num_tokens,
                )
                attention_maps["gps"] = gps_attention.detach().reshape(batch_size, seq_len, self.gps_queries, num_tokens)
        self.last_attention_maps = attention_maps
        self.last_attention_heads = attention_heads
        self.last_attention_map = attention_maps.get("gps")
        self.last_current_latent = content_latent.detach()
        self.last_temporal_predicted_latent = temporal_latent.detach()
        self.last_gps_residual_latent = gps_residual_latent.detach()
        self.last_gate_weights = gate_weights.detach()
        self.last_diagnostics = self._diagnostics(
            gate_weights=gate_weights,
            gate_metadata=gate_metadata,
            temporal_metadata=temporal_metadata,
            gps_counterfactual_mask=gps_counterfactual_mask,
            benchmark_condition_metadata=benchmark_condition_metadata,
            attention_maps=attention_maps,
            attention_heads=attention_heads,
            residual_scale=scale,
            token_metadata=token_metadata,
            output_mode=self.output_mode,
            output_query_count=int(pooled.shape[2]) if pooled.ndim == 4 else 1,
        )
        return pooled

    def _predict_temporal_latent(self, content_latent: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        batch_size, steps, _ = content_latent.shape
        predicted = torch.zeros_like(content_latent)
        availability = torch.zeros((batch_size, steps), dtype=torch.bool, device=content_latent.device)
        source_ranges: list[list[int] | None] = []
        insufficient = 0
        for step in range(steps):
            start = max(0, step - self.history_window)
            end = step
            if end > start and self.temporal_predictor_type not in {"disabled", "none"}:
                history = content_latent[:, start:end, :]
                if self.temporal_predictor_type == "gru" and self.temporal_gru is not None:
                    output, _ = self.temporal_gru(history)
                    predicted[:, step, :] = self.temporal_projection(output[:, -1, :])
                else:
                    predicted[:, step, :] = history.mean(dim=1)
                availability[:, step] = True
                source_ranges.append([start, end - 1])
            else:
                insufficient += batch_size
                predicted[:, step, :] = _predictive_insufficient_history(content_latent[:, step, :], self.temporal_fallback)
                source_ranges.append(None)
        return predicted, {
            "predictor_type": self.temporal_predictor_type,
            "history_window": self.history_window,
            "source_history_range": source_ranges,
            "availability_mask": availability.detach().cpu().tolist(),
            "available": bool(availability.any().item()),
            "available_count": int(availability.sum().item()),
            "insufficient_history_count": int(insufficient),
            "fallback_strategy": self.temporal_fallback,
            "source_history_range_policy": "strictly_past",
        }

    def _gate_weights(
        self,
        content_latent: torch.Tensor,
        temporal_latent: torch.Tensor,
        gps_residual_latent: torch.Tensor,
        *,
        temporal_available: torch.Tensor,
        gps_available_mask: torch.Tensor,
        image_valid_mask: torch.Tensor,
        image_observability_score: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        scale = max(float(self.latent_dim), 1.0) ** 0.5
        temporal_consistency = torch.linalg.vector_norm(content_latent - temporal_latent, dim=-1, keepdim=True) / scale
        gps_consistency = torch.linalg.vector_norm(content_latent - gps_residual_latent, dim=-1, keepdim=True) / scale
        gate_inputs = torch.cat(
            [
                temporal_available.to(dtype=content_latent.dtype).unsqueeze(-1),
                gps_available_mask.to(dtype=content_latent.dtype).unsqueeze(-1),
                image_valid_mask.to(dtype=content_latent.dtype).unsqueeze(-1),
                image_observability_score.to(dtype=content_latent.dtype).unsqueeze(-1),
                temporal_consistency,
                gps_consistency,
            ],
            dim=-1,
        )
        if self.reliability_gate is None:
            logits = torch.zeros((*content_latent.shape[:2], 3), dtype=content_latent.dtype, device=content_latent.device)
        else:
            logits = self.reliability_gate(gate_inputs)
        logits = logits.clone()
        logits[..., 1] = torch.where(temporal_available, logits[..., 1], torch.full_like(logits[..., 1], -1.0e4))
        logits[..., 2] = torch.where(gps_available_mask, logits[..., 2], torch.full_like(logits[..., 2], -1.0e4))
        weights = torch.softmax(logits, dim=-1)
        return weights, {
            "input_fields": [
                "temporal_available",
                "gps_valid_mask",
                "image_valid_mask",
                "image_observability_score",
                "current_temporal_latent_l2",
                "current_gps_residual_l2",
            ],
            "temporal_consistency_mean": float(temporal_consistency.detach().mean().cpu().item()),
            "gps_consistency_mean": float(gps_consistency.detach().mean().cpu().item()),
        }

    def _diagnostics(
        self,
        *,
        gate_weights: torch.Tensor,
        gate_metadata: dict[str, Any],
        temporal_metadata: dict[str, Any],
        gps_counterfactual_mask: torch.Tensor | None,
        benchmark_condition_metadata: dict[str, Any] | None,
        attention_maps: dict[str, torch.Tensor],
        attention_heads: dict[str, torch.Tensor],
        residual_scale: torch.Tensor,
        token_metadata: dict[str, Any] | None,
        output_mode: str,
        output_query_count: int,
    ) -> dict[str, Any]:
        gate_mean = gate_weights.detach().mean(dim=(0, 1)).cpu().tolist()
        gps_counterfactual_count = 0
        if torch.is_tensor(gps_counterfactual_mask):
            gps_counterfactual_count = int(gps_counterfactual_mask.detach().to(dtype=torch.bool).sum().cpu().item())
        blocked_present = sorted(
            field
            for field in self.forbidden_condition_fields
            if isinstance(benchmark_condition_metadata, dict) and field in benchmark_condition_metadata
        )
        attention_summary: dict[str, Any] = {}
        for name, value in attention_maps.items():
            attention_summary[name] = {
                "shape": [int(dim) for dim in value.shape],
                "mean_entropy": _attention_entropy(value),
            }
        return {
            "type": self.pooler_type,
            "branch_availability": {
                "current_content": True,
                "temporal_predicted": bool(temporal_metadata.get("available", False)),
                "gps_residual": True,
            },
            "gate_type": self.reliability_gate_type,
            "gate_weight_mean": {
                "current_content": float(gate_mean[0]),
                "temporal_predicted": float(gate_mean[1]),
                "gps_residual": float(gate_mean[2]),
            },
            "gate_input_fields": gate_metadata["input_fields"],
            "condition_id_consumed": False,
            "blocked_condition_fields": list(self.forbidden_condition_fields),
            "blocked_condition_fields_present": blocked_present,
            "residual_scale": float(residual_scale.detach().cpu().item()),
            "residual_scale_init": self.residual_scale_init,
            "gps_query_attention_summary": attention_summary.get("gps", {}),
            "content_attention_summary": attention_summary.get("content", {}),
            "attention_head_aggregation": "per_head" if self.per_head_attention else "averaged",
            "last_attention_map_source": "gps" if "gps" in attention_maps else "unavailable",
            "last_attention_map_exposes_gps_branch": "gps" in attention_maps,
            "branch_attention": _branch_attention_diagnostics(
                attention_maps,
                attention_heads=attention_heads,
                unavailable={"content": "", "gps": ""},
                exposed_branch="gps",
            ),
            "temporal_predictor_type": self.temporal_predictor_type,
            "history_window": self.history_window,
            "temporal_source_history_range": temporal_metadata["source_history_range"],
            "temporal_source_history_range_policy": temporal_metadata["source_history_range_policy"],
            "temporal_availability_mask": temporal_metadata["availability_mask"],
            "insufficient_history_count": temporal_metadata["insufficient_history_count"],
            "fallback_strategy": temporal_metadata["fallback_strategy"],
            "gps_counterfactual_count": gps_counterfactual_count,
            "output_mode": output_mode,
            "k_tokens": int(output_query_count),
            "latent_consistency": {
                "current_temporal_l2_mean": gate_metadata["temporal_consistency_mean"],
                "current_gps_residual_l2_mean": gate_metadata["gps_consistency_mean"],
            },
            **_pooler_token_diagnostics(
                token_metadata=token_metadata,
                attention_map=attention_maps.get("gps"),
                output_mode=output_mode,
                query_count=output_query_count,
                condition_source=self.context_feature_source,
            ),
        }

    def _validate_inputs(self, patch_tokens: torch.Tensor, condition_features: torch.Tensor) -> None:
        if patch_tokens.ndim != 4:
            raise ValueError(
                f"PredictiveGPSQueryPool patch tokens must have shape [B, T, N, D], got {tuple(patch_tokens.shape)}."
            )
        if condition_features.ndim != 3:
            raise ValueError(
                "PredictiveGPSQueryPool condition features must have shape [B, T, C], "
                f"got {tuple(condition_features.shape)} for patch tokens {tuple(patch_tokens.shape)}."
            )
        if int(patch_tokens.shape[-1]) != self.latent_dim:
            raise ValueError(
                f"PredictiveGPSQueryPool expected patch token latent dim {self.latent_dim}, "
                f"got patch tokens {tuple(patch_tokens.shape)}."
            )
        if int(condition_features.shape[-1]) != self.condition_dim:
            raise ValueError(
                f"PredictiveGPSQueryPool expected condition feature dim {self.condition_dim}, "
                f"got condition features {tuple(condition_features.shape)}."
            )
        if tuple(patch_tokens.shape[:2]) != tuple(condition_features.shape[:2]):
            raise ValueError(
                "PredictiveGPSQueryPool patch tokens and condition features must share batch/time dimensions; "
                f"patch tokens shape {tuple(patch_tokens.shape)}, "
                f"condition feature shape {tuple(condition_features.shape)}."
            )

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": self.pooler_type,
            "latent_dim": self.latent_dim,
            "condition_dim": self.condition_dim,
            "content_queries": self.content_queries,
            "gps_queries": self.gps_queries,
            "num_heads": self.num_heads,
            "condition_source": self.condition_source,
            "output_mode": self.output_mode,
            "k_tokens": self.content_queries + self.gps_queries if self.output_mode == "tokens" else 1,
            "return_attention": self.return_attention,
            "per_head_attention": self.per_head_attention,
            "attention_diagnostics": self.return_attention,
            "requires_condition": True,
            "required_context_modalities": list(self.required_context_modalities),
            "context_feature_source": self.context_feature_source,
            "residual_scale_init": self.residual_scale_init,
            "residual_scale": float(self.residual_scale.detach().cpu().clamp(0.0, 1.0).item()),
            "temporal_predictor_type": self.temporal_predictor_type,
            "history_window": self.history_window,
            "reliability_gate_type": self.reliability_gate_type,
            "forbidden_condition_fields": list(self.forbidden_condition_fields),
        }


class IdentityJepaAdapter(nn.Module):
    adapter_type = "identity"

    def __init__(self, latent_dim: int | None = None, output_dim: int | None = None, **_: Any) -> None:
        super().__init__()
        self.latent_dim = None if latent_dim is None else int(latent_dim)
        self.output_dim = self.latent_dim if output_dim is None else int(output_dim)
        if self.latent_dim is not None and self.output_dim != self.latent_dim:
            raise ValueError(
                "identity JEPA adapter requires output_dim to equal latent_dim, "
                f"got output_dim={self.output_dim}, latent_dim={self.latent_dim}."
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": self.adapter_type,
            "latent_dim": self.latent_dim,
            "output_dim": self.output_dim,
        }


def build_jepa_downstream_pooler(cfg: Any = None, **extra_kwargs: Any) -> nn.Module:
    return JEPA_DOWNSTREAM_POOLERS.build(_normalize_component_config(cfg, default_type="mean"), **extra_kwargs)


def build_jepa_downstream_adapter(cfg: Any = None, **extra_kwargs: Any) -> nn.Module:
    params = _normalize_component_config(cfg, default_type="identity")
    params.update(extra_kwargs)
    adapter_type = str(params.pop("type", "identity")).strip().lower()
    if adapter_type != "identity":
        raise ValueError(f"Unsupported JEPA downstream adapter '{adapter_type}'. Only 'identity' is available.")
    return IdentityJepaAdapter(**params)


def normalize_jepa_downstream_pooler_config(
    *,
    pooler: Any = None,
    pooling: str | None = None,
    gps_query_pool: dict[str, Any] | None = None,
    latent_dim: int,
) -> dict[str, Any]:
    if pooler is not None:
        cfg = _normalize_component_config(pooler, default_type="mean")
    else:
        pooling_name = str(pooling or "mean").strip().lower()
        if pooling_name == "mean":
            cfg = {"type": "mean"}
        elif pooling_name == "gps_query_attention":
            cfg = {"type": "gps_query_attention", **dict(gps_query_pool or {})}
        else:
            cfg = {"type": pooling_name}
    cfg.setdefault("latent_dim", int(latent_dim))
    cfg["type"] = _normalize_pooler_type(cfg.get("type", "mean"))
    cfg["output_mode"] = _normalize_pooler_output_mode(cfg.get("output_mode", "frame"))
    if str(cfg.get("type")) == "gps_query_attention":
        cfg.setdefault("condition_dim", int(latent_dim))
        cfg.setdefault("k_queries", 4)
        cfg.setdefault("num_heads", 4)
        cfg.setdefault("dropout", 0.0)
        cfg.setdefault("return_attention", False)
        cfg.setdefault("per_head_attention", False)
        cfg["condition_source"] = _normalize_condition_source(cfg.get("condition_source", "projected_gps"))
    elif str(cfg.get("type")) == "learned_query_attention":
        cfg.setdefault("k_queries", 2)
        cfg.setdefault("num_heads", 4)
        cfg.setdefault("dropout", 0.0)
        cfg.setdefault("return_attention", False)
    elif str(cfg.get("type")) == "self_attention":
        cfg.setdefault("k_tokens", cfg.get("k_queries", 2))
        cfg.setdefault("num_heads", 4)
        cfg.setdefault("num_layers", 1)
        cfg.setdefault("dropout", 0.0)
    elif str(cfg.get("type")) == "hybrid_residual_query":
        cfg.setdefault("condition_dim", int(latent_dim))
        cfg.setdefault("content_queries", 2)
        cfg.setdefault("gps_queries", cfg.get("k_queries", 2))
        cfg.setdefault("num_heads", 4)
        cfg.setdefault("dropout", 0.0)
        cfg.setdefault("return_attention", False)
        cfg.setdefault("per_head_attention", False)
        cfg.setdefault("residual_alpha_init", 0.1)
        cfg.setdefault("require_condition", True)
        cfg["condition_source"] = _normalize_condition_source(cfg.get("condition_source", "projected_gps"))
    elif str(cfg.get("type")) == "predictive_gps_query":
        cfg.setdefault("condition_dim", int(latent_dim))
        cfg.setdefault("content_queries", 2)
        cfg.setdefault("gps_queries", cfg.get("k_queries", 2))
        cfg.setdefault("num_heads", 4)
        cfg.setdefault("dropout", 0.0)
        cfg.setdefault("return_attention", False)
        cfg.setdefault("per_head_attention", False)
        cfg.setdefault("residual_scale_init", cfg.get("residual_alpha_init", 0.1))
        cfg.setdefault("temporal_predictor", {"type": "gru", "history_window": 4, "insufficient_history": "zero"})
        cfg.setdefault("reliability_gate", {"type": "mlp"})
        cfg["condition_source"] = _normalize_condition_source(cfg.get("condition_source", "projected_gps"))
    return cfg


def normalize_jepa_downstream_adapter_config(
    *,
    adapter: Any = None,
    latent_dim: int,
    output_dim: int,
) -> dict[str, Any]:
    cfg = _normalize_component_config(adapter, default_type="identity")
    cfg.setdefault("latent_dim", int(latent_dim))
    cfg.setdefault("output_dim", int(output_dim))
    return cfg


def _normalize_component_config(cfg: Any, *, default_type: str) -> dict[str, Any]:
    if cfg is None:
        return {"type": default_type}
    if isinstance(cfg, str):
        return {"type": cfg}
    if not isinstance(cfg, dict):
        raise ValueError(f"JEPA downstream component config must be a dict or string, got {type(cfg).__name__}.")
    resolved = dict(cfg)
    resolved.setdefault("type", default_type)
    return resolved


def _normalize_pooler_type(value: Any) -> str:
    pooler_type = str(value or "mean").strip().lower()
    aliases = {
        "hybrid": "hybrid_residual_query",
        "hybrid_query": "hybrid_residual_query",
        "hybrid_residual": "hybrid_residual_query",
        "predictive_gps_query++": "predictive_gps_query",
        "predictive_gps_query_plus_plus": "predictive_gps_query",
        "gps_query_plus_plus": "predictive_gps_query",
    }
    return aliases.get(pooler_type, pooler_type)


def _normalize_predictive_temporal_config(raw: dict[str, Any] | str | None) -> dict[str, Any]:
    if raw is None:
        cfg: dict[str, Any] = {"type": "gru"}
    elif isinstance(raw, str):
        cfg = {"type": raw}
    elif isinstance(raw, dict):
        cfg = dict(raw)
    else:
        raise ValueError("PredictiveGPSQueryPool temporal_predictor must be a dict, string, or None.")
    predictor_type = str(cfg.get("type", cfg.get("predictor_type", "gru"))).strip().lower() or "gru"
    aliases = {"causal_gru": "gru", "history_mean": "mean", "off": "disabled"}
    predictor_type = aliases.get(predictor_type, predictor_type)
    if predictor_type not in {"gru", "mean", "disabled", "none"}:
        raise ValueError("PredictiveGPSQueryPool temporal_predictor.type must be gru, mean, or disabled.")
    history_window = int(cfg.get("history_window", cfg.get("window", 4)) or 4)
    if history_window <= 0:
        raise ValueError(f"PredictiveGPSQueryPool temporal history_window must be positive, got {history_window}.")
    fallback = str(cfg.get("insufficient_history", cfg.get("fallback", "zero"))).strip().lower() or "zero"
    if fallback not in {"raw", "skip", "zero", "clamp"}:
        raise ValueError("PredictiveGPSQueryPool insufficient_history must be raw, skip, zero, or clamp.")
    return {
        **cfg,
        "type": predictor_type,
        "history_window": history_window,
        "insufficient_history": fallback,
    }


def _normalize_predictive_gate_config(raw: dict[str, Any] | str | None) -> dict[str, Any]:
    if raw is None:
        cfg: dict[str, Any] = {"type": "mlp"}
    elif isinstance(raw, str):
        cfg = {"type": raw}
    elif isinstance(raw, dict):
        cfg = dict(raw)
    else:
        raise ValueError("PredictiveGPSQueryPool reliability_gate must be a dict, string, or None.")
    gate_type = str(cfg.get("type", cfg.get("gate_type", "mlp"))).strip().lower() or "mlp"
    aliases = {"reliability_mlp": "mlp", "none": "uniform"}
    gate_type = aliases.get(gate_type, gate_type)
    if gate_type not in {"mlp", "linear", "uniform", "fixed"}:
        raise ValueError("PredictiveGPSQueryPool reliability_gate.type must be mlp, linear, uniform, or fixed.")
    hidden_dim = int(cfg.get("hidden_dim", cfg.get("gate_hidden_dim", 0)) or 0)
    if hidden_dim < 0:
        raise ValueError("PredictiveGPSQueryPool reliability_gate.hidden_dim must be non-negative.")
    return {**cfg, "type": gate_type, "hidden_dim": hidden_dim}


def _predictive_insufficient_history(current: torch.Tensor, strategy: str) -> torch.Tensor:
    if strategy in {"raw", "clamp"}:
        return current
    if strategy in {"zero", "skip"}:
        return torch.zeros_like(current)
    raise ValueError("PredictiveGPSQueryPool insufficient_history must be raw, skip, zero, or clamp.")


def _mask_or_default(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
    default: bool | float,
    name: str,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size, steps), default, dtype=dtype, device=device)
    tensor = value.to(device=device, dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1).expand(-1, steps)
    if tensor.ndim != 2 or tuple(tensor.shape) != (batch_size, steps):
        raise ValueError(f"{name} must have shape [B, T] for PredictiveGPSQueryPool, got {tuple(value.shape)}.")
    return tensor


def _normalize_pooler_output_mode(value: Any) -> str:
    mode = str(value or "frame").strip().lower()
    aliases = {
        "pooled": "frame",
        "mean": "frame",
        "frames": "frame",
        "k_tokens": "tokens",
        "queries": "tokens",
        "query_tokens": "tokens",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"frame", "tokens"}:
        raise ValueError("JEPA downstream pooler output_mode must be 'frame' or 'tokens'.")
    return mode


def _pooler_token_diagnostics(
    *,
    token_metadata: dict[str, Any] | None,
    attention_map: torch.Tensor | None,
    output_mode: str,
    query_count: int,
    condition_source: str,
    attended_tokens: torch.Tensor | None = None,
    attention_heads: torch.Tensor | None = None,
    attention_head_aggregation: str = "averaged",
) -> dict[str, Any]:
    metadata = dict(token_metadata or {})
    token_count = metadata.get("token_count")
    if token_count is None and torch.is_tensor(attention_map):
        token_count = int(attention_map.shape[-1])
    diagnostics: dict[str, Any] = {
        "token_grid": metadata.get("token_grid"),
        "token_count": token_count,
        "variant_id": metadata.get("variant_id"),
        "checkpoint_policy": metadata.get("checkpoint_policy"),
        "condition_feature_source": condition_source,
        "output_mode": output_mode,
        "query_count": int(query_count),
        "k_queries": int(query_count),
        "k_tokens": int(query_count) if output_mode == "tokens" else 1,
        "attention_head_aggregation": attention_head_aggregation,
        "diagnostics_status": "missing_attention",
    }
    if torch.is_tensor(attention_map):
        diagnostics["attention_shape"] = [int(dim) for dim in attention_map.shape]
        diagnostics["attention_return_shape"] = [int(dim) for dim in attention_map.shape]
        diagnostics["attention_diagnostics_shape"] = [int(dim) for dim in attention_map.shape]
        diagnostics["attention_output_shape"] = [int(dim) for dim in attention_map.shape]
        diagnostics["attention_entropy"] = _attention_entropy(attention_map)
        diagnostics["effective_patch_count"] = _attention_effective_patch_count(attention_map)
        diagnostics["attention_peakiness"] = _attention_peakiness(attention_map)
        diagnostics["query_diversity"] = _attention_query_diversity(attention_map)
        diagnostics["diagnostics_status"] = "available"
    if torch.is_tensor(attention_heads):
        diagnostics["attention_per_head_shape"] = [int(dim) for dim in attention_heads.shape]
        diagnostics["attention_head_count"] = int(attention_heads.shape[2]) if attention_heads.ndim >= 3 else 0
        diagnostics["head_aggregation_method"] = "mean_heads_for_last_attention_map"
    if torch.is_tensor(attended_tokens):
        diagnostics["attended_latent_similarity"] = _attended_latent_similarity(attended_tokens)
        diagnostics["output_shape"] = [int(dim) for dim in attended_tokens.shape]
    return diagnostics


def _branch_attention_diagnostics(
    attention_maps: dict[str, torch.Tensor],
    *,
    attention_heads: dict[str, torch.Tensor],
    unavailable: dict[str, str],
    exposed_branch: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("content", "gps"):
        value = attention_maps.get(name)
        if torch.is_tensor(value):
            item: dict[str, Any] = {
                "available": True,
                "shape": [int(dim) for dim in value.shape],
                "mean_entropy": _attention_entropy(value),
                "exposed_as_last_attention_map": name == exposed_branch,
            }
            heads = attention_heads.get(name)
            if torch.is_tensor(heads):
                item["per_head_shape"] = [int(dim) for dim in heads.shape]
                item["head_count"] = int(heads.shape[2]) if heads.ndim >= 3 else 0
            out[name] = item
        else:
            out[name] = {
                "available": False,
                "unavailable_reason": unavailable.get(name) or "attention_not_requested",
                "exposed_as_last_attention_map": False,
            }
    return out


def _attention_entropy(attention: torch.Tensor) -> float:
    probs = attention.detach().to(dtype=torch.float32).clamp_min(1.0e-12)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return float(entropy.mean().cpu().item())


def _attention_peakiness(attention: torch.Tensor) -> float:
    probs = attention.detach().to(dtype=torch.float32)
    return float(probs.max(dim=-1).values.mean().cpu().item())


def _attention_effective_patch_count(attention: torch.Tensor) -> float:
    probs = attention.detach().to(dtype=torch.float32).clamp_min(1.0e-12)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return float(entropy.exp().mean().cpu().item())


def _attention_query_diversity(attention: torch.Tensor) -> float:
    probs = attention.detach().to(dtype=torch.float32)
    if probs.ndim != 4 or int(probs.shape[2]) < 2:
        return 0.0
    rows = torch.nn.functional.normalize(probs, dim=-1)
    sim = torch.matmul(rows, rows.transpose(-1, -2))
    k = int(rows.shape[2])
    off_diag = sim[..., ~torch.eye(k, dtype=torch.bool, device=sim.device)].reshape(*sim.shape[:2], k, k - 1)
    return float((1.0 - off_diag).mean().cpu().item())


def _attended_latent_similarity(tokens: torch.Tensor) -> float:
    values = tokens.detach().to(dtype=torch.float32)
    if values.ndim != 4 or int(values.shape[2]) < 2:
        return 1.0
    rows = torch.nn.functional.normalize(values, dim=-1)
    sim = torch.matmul(rows, rows.transpose(-1, -2))
    k = int(rows.shape[2])
    off_diag = sim[..., ~torch.eye(k, dtype=torch.bool, device=sim.device)].reshape(*sim.shape[:2], k, k - 1)
    return float(off_diag.mean().cpu().item())


def _normalize_condition_source(value: Any) -> str:
    source = str(value or "projected_gps").strip().lower()
    aliases = {
        "projected": "projected_gps",
        "gps_projected": "projected_gps",
        "encoded": "encoded_gps",
        "gps_encoded": "encoded_gps",
        "raw": "raw_gps",
        "gps_raw": "raw_gps",
    }
    source = aliases.get(source, source)
    if source not in {"projected_gps", "encoded_gps", "raw_gps"}:
        raise ValueError(
            "GPS-query JEPA pooler condition_source must be one of "
            "'projected_gps', 'encoded_gps', or 'raw_gps'."
        )
    return source


def _context_feature_source_kind(condition_source: str) -> str:
    if condition_source.startswith("encoded_"):
        return "encoded"
    if condition_source.startswith("raw_"):
        return "raw"
    return "projected"


__all__ = [
    "GPSQueryPool",
    "IdentityJepaAdapter",
    "LearnedQueryPool",
    "MeanPatchPooler",
    "PredictiveGPSQueryPool",
    "SelfAttentionPool",
    "build_jepa_downstream_adapter",
    "build_jepa_downstream_pooler",
    "normalize_jepa_downstream_adapter_config",
    "normalize_jepa_downstream_pooler_config",
]
