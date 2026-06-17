from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from kd_sensing.registries import JEPA_DOWNSTREAM_ADAPTERS, JEPA_DOWNSTREAM_POOLERS


@JEPA_DOWNSTREAM_POOLERS.register("mean")
class MeanPatchPooler(nn.Module):
    pooler_type = "mean"
    required_context_modalities: tuple[str, ...] = ()
    context_feature_source = "none"
    context_feature_kwargs: dict[str, str] = {}

    def __init__(self, latent_dim: int | None = None, **_: Any) -> None:
        super().__init__()
        self.latent_dim = None if latent_dim is None else int(latent_dim)

    def forward(self, patch_tokens: torch.Tensor, condition_features: torch.Tensor | None = None) -> torch.Tensor:
        del condition_features
        if patch_tokens.ndim != 4:
            raise ValueError(
                f"MeanPatchPooler patch tokens must have shape [B, T, N, D], got {tuple(patch_tokens.shape)}."
            )
        if self.latent_dim is not None and int(patch_tokens.shape[-1]) != self.latent_dim:
            raise ValueError(
                f"MeanPatchPooler expected patch token dim {self.latent_dim}, got {tuple(patch_tokens.shape)}."
            )
        return patch_tokens.mean(dim=2)

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": self.pooler_type,
            "latent_dim": self.latent_dim,
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
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.k_queries = int(k_queries)
        self.num_heads = int(num_heads)
        self.return_attention = bool(return_attention)
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

    def forward(
        self,
        patch_tokens: torch.Tensor,
        condition_features: torch.Tensor | None = None,
        *,
        return_attention: bool | None = None,
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
            average_attn_weights=True,
        )
        pooled_queries = self.output_norm(attended).mean(dim=1)
        pooled = pooled_queries.reshape(batch_size, seq_len, self.latent_dim)
        if not with_attention:
            self.last_attention_map = None
            return pooled
        if attention is None:
            raise RuntimeError("GPSQueryPool requested attention diagnostics but MultiheadAttention returned None.")
        attention_map = attention.detach().reshape(batch_size, seq_len, self.k_queries, num_tokens)
        self.last_attention_map = attention_map
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
            "return_attention": self.return_attention,
            "attention_diagnostics": self.return_attention,
            "requires_condition": True,
            "required_context_modalities": list(self.required_context_modalities),
            "context_feature_source": self.context_feature_source,
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
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.content_queries = int(content_queries)
        self.gps_queries = int(gps_queries if gps_queries is not None else k_queries if k_queries is not None else 2)
        self.num_heads = int(num_heads)
        self.return_attention = bool(return_attention)
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
        self.last_diagnostics: dict[str, Any] = {}
        nn.init.trunc_normal_(self.content_query, std=0.02)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        condition_features: torch.Tensor | None = None,
        *,
        return_attention: bool | None = None,
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
            average_attn_weights=True,
        )
        content_latent = content_attended.mean(dim=1).reshape(batch_size, seq_len, latent_dim)

        gps_latent = mean_latent
        gps_attention_map: torch.Tensor | None = None
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
                average_attn_weights=True,
            )
            gps_latent = gps_attended.mean(dim=1).reshape(batch_size, seq_len, latent_dim)
            if with_attention:
                if gps_attention is None:
                    raise RuntimeError(
                        "HybridResidualQueryPool requested GPS attention diagnostics but attention returned None."
                    )
                gps_attention_map = gps_attention.detach().reshape(batch_size, seq_len, self.gps_queries, num_tokens)

        content_delta = content_latent - mean_latent
        gps_delta = gps_latent - mean_latent if gps_available else torch.zeros_like(mean_latent)
        residual = self.residual_mlp(torch.cat([content_delta, gps_delta], dim=-1))
        alpha = self.residual_alpha.to(dtype=patch_tokens.dtype, device=patch_tokens.device).clamp(0.0, 1.0)
        pooled = self.output_norm(mean_latent + alpha * residual)

        attention_maps: dict[str, torch.Tensor] = {}
        if with_attention:
            if content_attention is None:
                raise RuntimeError(
                    "HybridResidualQueryPool requested content attention diagnostics but attention returned None."
                )
            attention_maps["content"] = content_attention.detach().reshape(
                batch_size,
                seq_len,
                self.content_queries,
                num_tokens,
            )
            if gps_attention_map is not None:
                attention_maps["gps"] = gps_attention_map
        self.last_attention_maps = attention_maps
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
            "return_attention": self.return_attention,
            "attention_diagnostics": self.return_attention,
            "requires_condition": self.require_condition,
            "residual_alpha_init": self.residual_alpha_init,
            "residual_alpha": float(self.residual_alpha.detach().cpu().clamp(0.0, 1.0).item()),
            "required_context_modalities": list(self.required_context_modalities),
            "context_feature_source": self.context_feature_source,
        }


@JEPA_DOWNSTREAM_ADAPTERS.register("identity")
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
    return JEPA_DOWNSTREAM_ADAPTERS.build(_normalize_component_config(cfg, default_type="identity"), **extra_kwargs)


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
    if str(cfg.get("type")) == "gps_query_attention":
        cfg.setdefault("condition_dim", int(latent_dim))
        cfg.setdefault("k_queries", 4)
        cfg.setdefault("num_heads", 4)
        cfg.setdefault("dropout", 0.0)
        cfg.setdefault("return_attention", False)
        cfg["condition_source"] = _normalize_condition_source(cfg.get("condition_source", "projected_gps"))
    elif str(cfg.get("type")) == "hybrid_residual_query":
        cfg.setdefault("condition_dim", int(latent_dim))
        cfg.setdefault("content_queries", 2)
        cfg.setdefault("gps_queries", cfg.get("k_queries", 2))
        cfg.setdefault("num_heads", 4)
        cfg.setdefault("dropout", 0.0)
        cfg.setdefault("return_attention", False)
        cfg.setdefault("residual_alpha_init", 0.1)
        cfg.setdefault("require_condition", True)
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
    }
    return aliases.get(pooler_type, pooler_type)


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
    "MeanPatchPooler",
    "build_jepa_downstream_adapter",
    "build_jepa_downstream_pooler",
    "normalize_jepa_downstream_adapter_config",
    "normalize_jepa_downstream_pooler_config",
]
