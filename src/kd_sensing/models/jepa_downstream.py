from typing import Any, Mapping

import torch
import torch.nn as nn


class MeanPatchPooler(nn.Module):
    pooler_type = "mean"
    output_mode = "frame"
    required_context_modalities: tuple[str, ...] = ()
    context_feature_source = "none"

    def __init__(self, latent_dim: int | None = None, **_: Any) -> None:
        super().__init__()
        self.latent_dim = None if latent_dim is None else int(latent_dim)
        if self.latent_dim is not None and self.latent_dim <= 0:
            raise ValueError("MeanPatchPooler latent_dim must be positive.")

    def forward(self, patch_tokens: torch.Tensor, **_: Any) -> torch.Tensor:
        if patch_tokens.ndim != 4:
            raise ValueError(
                f"MeanPatchPooler patch tokens must have shape [B, T, N, D], got {tuple(patch_tokens.shape)}."
            )
        if self.latent_dim is not None and patch_tokens.shape[-1] != self.latent_dim:
            raise ValueError(
                f"MeanPatchPooler expected patch token dim {self.latent_dim}, got {tuple(patch_tokens.shape)}."
            )
        return patch_tokens.mean(dim=2)

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": self.pooler_type,
            "latent_dim": self.latent_dim,
            "output_mode": self.output_mode,
        }


def build_jepa_downstream_pooler(cfg: Any = None, **extra_kwargs: Any) -> MeanPatchPooler:
    params = _mean_pooler_config(cfg)
    params.update(extra_kwargs)
    params.pop("type", None)
    return MeanPatchPooler(**params)


def normalize_jepa_downstream_pooler_config(
    *,
    pooler: Any = None,
    pooling: str | None = None,
    gps_query_pool: Mapping[str, Any] | None = None,
    latent_dim: int,
) -> dict[str, Any]:
    if gps_query_pool:
        raise ValueError("gps_query_pool has been retired; current JEPA downstream supports only pooling='mean'.")
    cfg = _mean_pooler_config(pooler if pooler is not None else {"type": pooling or "mean"})
    cfg.setdefault("latent_dim", int(latent_dim))
    return cfg


def _mean_pooler_config(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        params: dict[str, Any] = {"type": "mean"}
    elif isinstance(cfg, str):
        params = {"type": cfg}
    elif isinstance(cfg, Mapping):
        params = dict(cfg)
    else:
        raise ValueError(f"JEPA downstream pooler config must be a dict, string, or None, got {type(cfg).__name__}.")

    pooler_type = str(params.get("type", "mean")).strip().lower()
    if pooler_type != "mean":
        raise ValueError(
            f"JEPA downstream pooler {pooler_type!r} has been retired; current path supports only 'mean'."
        )
    output_mode = str(params.pop("output_mode", "frame")).strip().lower()
    if output_mode != "frame":
        raise ValueError("JEPA downstream K-token output has been retired; current mean path returns [B, T, D].")
    params["type"] = "mean"
    return params


__all__ = [
    "MeanPatchPooler",
    "build_jepa_downstream_pooler",
    "normalize_jepa_downstream_pooler_config",
]
