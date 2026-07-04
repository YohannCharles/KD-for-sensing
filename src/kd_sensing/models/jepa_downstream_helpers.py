from typing import Any

import torch


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
