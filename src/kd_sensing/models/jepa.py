from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from kd_sensing.modalities import image_profile_spec, validate_image_encoder_profile
from kd_sensing.models.jepa_downstream import (
    GPSQueryPool,
    build_jepa_downstream_adapter,
    build_jepa_downstream_pooler,
    normalize_jepa_downstream_adapter_config,
    normalize_jepa_downstream_pooler_config,
)
from kd_sensing.registries import ENCODERS, MODELS
from kd_sensing.utils.checkpoint import CheckpointLoadError


@dataclass(frozen=True)
class JepaMaskSample:
    context_mask: torch.Tensor
    target_mask: torch.Tensor
    loss_mask: torch.Tensor
    context_indices: torch.Tensor
    target_indices: torch.Tensor
    diagnostics: dict[str, float | str]


class VisualPatchTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        image_channels: int = 3,
        latent_dim: int = 64,
        patch_size: int = 16,
        depth: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        max_tokens: int = 256,
        image_profile: str | None = None,
    ) -> None:
        super().__init__()
        validate_image_encoder_profile(
            encoder_name="gps_conditioned_jepa",
            image_profile=image_profile,
            expected_channels=image_profile_spec(image_profile).channels,
            actual_channels=image_channels,
        )
        self.latent_dim = int(latent_dim)
        self.patch_size = int(patch_size)
        self.max_tokens = int(max_tokens)
        self.patch_embed = nn.Conv2d(
            int(image_channels),
            self.latent_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens, self.latent_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=int(num_heads),
            dim_feedforward=max(int(self.latent_dim * float(mlp_ratio)), self.latent_dim),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(int(depth), 0)) if int(depth) > 0 else nn.Identity()
        self.norm = nn.LayerNorm(self.latent_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, image_batch: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if image_batch.ndim != 5:
            raise ValueError(f"JEPA image_batch must have shape [B, T, C, H, W], got {tuple(image_batch.shape)}.")
        batch_size, seq_len, channels, height, width = image_batch.shape
        frames = image_batch.reshape(batch_size * seq_len, channels, height, width)
        patches = self.patch_embed(frames)
        grid_size = (int(patches.shape[-2]), int(patches.shape[-1]))
        tokens = patches.flatten(2).transpose(1, 2)
        if tokens.shape[1] > self.max_tokens:
            raise ValueError(
                f"JEPA visual encoder produced {tokens.shape[1]} tokens, exceeding max_tokens={self.max_tokens}."
            )
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :].to(dtype=tokens.dtype, device=tokens.device)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens.reshape(batch_size, seq_len, tokens.shape[1], self.latent_dim), grid_size


class JepaMaskSampler(nn.Module):
    def __init__(
        self,
        *,
        mode: str = "random",
        context_ratio: float = 0.6,
        target_ratio: float = 0.2,
        seed: int = 0,
        angle_feature_index: int = 1,
        angle_concentration: float = 3.0,
    ) -> None:
        super().__init__()
        self.mode = str(mode).strip().lower()
        if self.mode not in {"random", "gps_angle_biased"}:
            raise ValueError("JEPA mask sampler mode must be random or gps_angle_biased.")
        self.context_ratio = _ratio(context_ratio, "context_ratio")
        self.target_ratio = _ratio(target_ratio, "target_ratio")
        if self.context_ratio + self.target_ratio > 1.0:
            raise ValueError("JEPA context_ratio + target_ratio must not exceed 1.0.")
        self.seed = int(seed)
        self.angle_feature_index = int(angle_feature_index)
        self.angle_concentration = float(angle_concentration)

    def sample(
        self,
        *,
        batch_size: int,
        seq_len: int,
        num_tokens: int,
        grid_size: tuple[int, int],
        gps_batch: torch.Tensor,
        epoch: int = 0,
        step: int = 0,
        device: torch.device | None = None,
    ) -> JepaMaskSample:
        device = device or gps_batch.device
        n_context = min(max(1, int(round(num_tokens * self.context_ratio))), max(num_tokens - 1, 1))
        n_target = min(max(1, int(round(num_tokens * self.target_ratio))), max(num_tokens - n_context, 1))
        context_indices = torch.empty(batch_size, seq_len, n_context, dtype=torch.long, device=device)
        target_indices = torch.empty(batch_size, seq_len, n_target, dtype=torch.long, device=device)
        gps_cpu = gps_batch.detach().cpu()
        for batch_idx in range(batch_size):
            for time_idx in range(seq_len):
                gen = torch.Generator().manual_seed(self._sample_seed(epoch, step, batch_idx, time_idx))
                target = self._sample_target_indices(
                    n_target,
                    num_tokens=num_tokens,
                    grid_size=grid_size,
                    gps=gps_cpu[batch_idx, time_idx],
                    generator=gen,
                )
                target_set = set(int(value) for value in target.tolist())
                remaining = torch.tensor([idx for idx in range(num_tokens) if idx not in target_set], dtype=torch.long)
                order = remaining[torch.randperm(len(remaining), generator=gen)]
                context = order[:n_context]
                target_indices[batch_idx, time_idx] = target.to(device=device)
                context_indices[batch_idx, time_idx] = context.to(device=device)
        context_mask = _indices_to_mask(context_indices, num_tokens)
        target_mask = _indices_to_mask(target_indices, num_tokens)
        if torch.any(context_mask & target_mask):
            raise RuntimeError("JEPA sampler produced overlapping context and target masks.")
        loss_mask = torch.ones(batch_size, seq_len, n_target, dtype=torch.bool, device=device)
        diagnostics = {
            "jepa/mask_mode": self.mode,
            "jepa/mask_context_ratio": float(context_mask.float().mean().detach().cpu().item()),
            "jepa/mask_target_ratio": float(target_mask.float().mean().detach().cpu().item()),
            "jepa/target_tokens": float(n_target),
        }
        return JepaMaskSample(
            context_mask=context_mask,
            target_mask=target_mask,
            loss_mask=loss_mask,
            context_indices=context_indices,
            target_indices=target_indices,
            diagnostics=diagnostics,
        )

    def _sample_seed(self, epoch: int, step: int, batch_idx: int, time_idx: int) -> int:
        return int(self.seed + int(epoch) * 1_000_003 + int(step) * 9_176 + batch_idx * 97 + time_idx * 13)

    def _sample_target_indices(
        self,
        count: int,
        *,
        num_tokens: int,
        grid_size: tuple[int, int],
        gps: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if self.mode == "random":
            return torch.randperm(num_tokens, generator=generator)[:count]
        weights = self._gps_angle_weights(num_tokens, grid_size=grid_size, gps=gps)
        return torch.multinomial(weights, num_samples=count, replacement=False, generator=generator)

    def _gps_angle_weights(self, num_tokens: int, *, grid_size: tuple[int, int], gps: torch.Tensor) -> torch.Tensor:
        rows, cols = grid_size
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, rows),
            torch.linspace(-1.0, 1.0, cols),
            indexing="ij",
        )
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)[:num_tokens]
        angle_index = min(max(self.angle_feature_index, 0), max(int(gps.numel()) - 1, 0))
        angle = float(gps.reshape(-1)[angle_index].item()) if gps.numel() else 0.0
        direction = torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float32)
        weights = torch.exp(self.angle_concentration * (coords @ direction))
        return weights.clamp_min(1e-6)


class GpsConditioner(nn.Module):
    def __init__(
        self,
        *,
        conditioning_type: str = "film",
        gps_input_size: int = 3,
        latent_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.conditioning_type = str(conditioning_type).strip().lower()
        self.gps_input_size = int(gps_input_size)
        self.latent_dim = int(latent_dim)
        hidden = int(hidden_dim)
        if self.conditioning_type == "film":
            self.net = nn.Sequential(nn.Linear(self.gps_input_size, hidden), nn.GELU(), nn.Linear(hidden, 2 * self.latent_dim))
        elif self.conditioning_type == "concat_mlp":
            self.net = nn.Sequential(
                nn.Linear(self.gps_input_size + self.latent_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.latent_dim),
            )
        else:
            raise ValueError("JEPA GPS conditioner type must be film or concat_mlp.")

    def forward(self, context_latent: torch.Tensor, gps_batch: torch.Tensor) -> torch.Tensor:
        if gps_batch.shape[-1] != self.gps_input_size:
            raise ValueError(
                f"GPS-conditioned JEPA expected GPS feature dim {self.gps_input_size}, got {gps_batch.shape[-1]}."
            )
        gps = gps_batch.to(device=context_latent.device, dtype=context_latent.dtype)
        if self.conditioning_type == "film":
            gamma_beta = self.net(gps).unsqueeze(2)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            return context_latent * (1.0 + gamma) + beta
        expanded = gps.unsqueeze(2).expand(*context_latent.shape[:-1], gps.shape[-1])
        return self.net(torch.cat([context_latent, expanded], dim=-1))


class TargetLatentPredictor(nn.Module):
    def __init__(self, *, latent_dim: int = 64, hidden_dim: int = 128, max_tokens: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.target_pos_embed = nn.Embedding(int(max_tokens), int(latent_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(int(latent_dim)),
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(latent_dim)),
        )

    def forward(self, context_latent: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        summary = context_latent.mean(dim=2, keepdim=True)
        position = self.target_pos_embed(target_indices.clamp_min(0).clamp_max(self.target_pos_embed.num_embeddings - 1))
        return self.net(summary + position)


@ENCODERS.register("jepa_context_image")
@MODELS.register("jepa_context_image")
class JepaContextImageEncoder(nn.Module):
    expected_image_profile = "rgb_imagenet"
    input_channels = 3

    def __init__(
        self,
        *,
        checkpoint_path: str | None = None,
        checkpoint: str | None = None,
        output_dim: int | None = None,
        latent_dim: int = 64,
        image_channels: int = 3,
        image_profile: str | None = "rgb_imagenet",
        visual_encoder: dict[str, Any] | None = None,
        freeze_encoder: bool = False,
        strict: bool = True,
        state_dict_prefix: str = "context_encoder",
        pooling: str = "mean",
        pooler: dict[str, Any] | str | None = None,
        adapter: dict[str, Any] | str | None = None,
        gps_query_pool: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim if output_dim is not None else self.latent_dim)
        self.checkpoint_path = str(checkpoint_path or checkpoint or "")
        self.freeze_encoder = bool(freeze_encoder)
        self.strict = bool(strict)
        self.state_dict_prefix = str(state_dict_prefix).strip().rstrip(".") or "context_encoder"
        self.pooler_config = normalize_jepa_downstream_pooler_config(
            pooler=pooler,
            pooling=pooling,
            gps_query_pool=gps_query_pool,
            latent_dim=self.latent_dim,
        )
        self.adapter_config = normalize_jepa_downstream_adapter_config(
            adapter=adapter,
            latent_dim=self.latent_dim,
            output_dim=self.output_dim,
        )
        self.pooling = str(self.pooler_config.get("type", "mean")).strip().lower()
        if self.output_dim != self.latent_dim:
            raise ValueError(
                "jepa_context_image requires output_dim to equal latent_dim because it reuses the JEPA "
                f"context encoder projection directly; got output_dim={self.output_dim}, latent_dim={self.latent_dim}."
            )
        self.pooler = build_jepa_downstream_pooler(self.pooler_config)
        self.adapter = build_jepa_downstream_adapter(self.adapter_config)
        self.required_context_modalities = tuple(getattr(self.pooler, "required_context_modalities", ()))
        self.context_feature_source = str(getattr(self.pooler, "context_feature_source", "none"))
        raw_kwargs = getattr(self.pooler, "context_feature_kwargs", {})
        self.context_feature_kwargs = dict(raw_kwargs) if isinstance(raw_kwargs, dict) else {}
        self.gps_query_pool_config: dict[str, Any] = {}
        object.__setattr__(
            self,
            "gps_query_pool",
            self.pooler if isinstance(self.pooler, GPSQueryPool) else None,
        )
        self.last_attention_map: torch.Tensor | None = None
        if self.pooling == "gps_query_attention":
            self.gps_query_pool_config = {
                "latent_dim": getattr(self.pooler, "latent_dim", self.latent_dim),
                "condition_dim": getattr(self.pooler, "condition_dim", self.latent_dim),
                "k_queries": getattr(self.pooler, "k_queries", None),
                "num_heads": getattr(self.pooler, "num_heads", None),
                "dropout": self.pooler_config.get("dropout", 0.0),
                "return_attention": getattr(self.pooler, "return_attention", False),
                "condition_source": getattr(self.pooler, "condition_source", "projected_gps"),
            }
        encoder_cfg = dict(visual_encoder or {})
        encoder_cfg.setdefault("image_channels", image_channels)
        encoder_cfg.setdefault("latent_dim", self.latent_dim)
        encoder_cfg.setdefault("image_profile", image_profile)
        self.context_encoder = VisualPatchTokenEncoder(**encoder_cfg)
        if self.checkpoint_path:
            _load_context_encoder_state(
                Path(self.checkpoint_path),
                self.context_encoder,
                prefix=self.state_dict_prefix,
                strict=self.strict,
            )
        if self.freeze_encoder:
            for param in self.context_encoder.parameters():
                param.requires_grad_(False)

    def forward(self, image_batch: torch.Tensor, gps_condition_features: torch.Tensor | None = None) -> torch.Tensor:
        tokens, _ = self.context_encoder(image_batch)
        if self.required_context_modalities and gps_condition_features is None:
            if self.pooling == "gps_query_attention":
                raise ValueError("jepa_context_image GPS-query pooling requires GPS condition feature.")
            raise ValueError(f"jepa_context_image pooler {self.pooling!r} requires condition features.")
        result = self.pooler(tokens, condition_features=gps_condition_features)
        if isinstance(result, tuple):
            pooled, attention_map = result
            self.last_attention_map = attention_map
            return self.adapter(pooled)
        self.last_attention_map = getattr(self.pooler, "last_attention_map", None)
        return self.adapter(result)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        legacy_prefix = f"{prefix}gps_query_pool."
        pooler_prefix = f"{prefix}pooler."
        if any(key.startswith(legacy_prefix) for key in state_dict) and not any(
            key.startswith(pooler_prefix) for key in state_dict
        ):
            for key, value in list(state_dict.items()):
                if key.startswith(legacy_prefix):
                    state_dict[f"{pooler_prefix}{key[len(legacy_prefix):]}"] = value
                    state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def training_strategy_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "encoder": "jepa_context_image",
            "checkpoint_path": self.checkpoint_path,
            "state_dict_prefix": self.state_dict_prefix,
            "freeze_encoder": self.freeze_encoder,
            "pooling": self.pooling,
            "pooler_type": self.pooling,
            "adapter_type": str(self.adapter_config.get("type", "identity")),
            "condition_source": self.gps_query_pool_config.get("condition_source"),
            "attention_diagnostics": bool(self.gps_query_pool_config.get("return_attention", False)),
            "latent_dim": self.latent_dim,
        }
        pooler_metadata = (
            self.pooler.training_strategy_metadata()
            if hasattr(self.pooler, "training_strategy_metadata")
            else {"type": self.pooling}
        )
        adapter_metadata = (
            self.adapter.training_strategy_metadata()
            if hasattr(self.adapter, "training_strategy_metadata")
            else {"type": self.adapter_config.get("type", "identity")}
        )
        metadata["pooler"] = pooler_metadata
        metadata["adapter"] = adapter_metadata
        if self.pooling == "gps_query_attention":
            metadata["gps_query_pooling_enabled"] = True
            metadata["gps_query_pool"] = dict(self.gps_query_pool_config)
            metadata["required_context_modalities"] = list(self.required_context_modalities)
            metadata["context_feature_source"] = self.context_feature_source
        else:
            metadata["gps_query_pooling_enabled"] = False
        return metadata


@MODELS.register("gps_conditioned_jepa")
class GPSConditionedJEPA(nn.Module):
    supports_modality_kwargs = True

    def __init__(
        self,
        *,
        latent_dim: int = 64,
        image_channels: int = 3,
        gps_input_size: int = 3,
        image_profile: str | None = None,
        visual_encoder: dict[str, Any] | None = None,
        conditioning: dict[str, Any] | None = None,
        predictor: dict[str, Any] | None = None,
        mask_sampler: dict[str, Any] | None = None,
        ema_decay: float = 0.996,
        num_classes: int = 1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.gps_input_size = int(gps_input_size)
        self.ema_decay = float(ema_decay)
        self.num_classes = int(num_classes)
        encoder_cfg = dict(visual_encoder or {})
        encoder_cfg.setdefault("image_channels", image_channels)
        encoder_cfg.setdefault("latent_dim", self.latent_dim)
        encoder_cfg.setdefault("image_profile", image_profile)
        self.context_encoder = VisualPatchTokenEncoder(**encoder_cfg)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)
        conditioner_cfg = dict(conditioning or {})
        conditioner_cfg.setdefault("conditioning_type", conditioner_cfg.pop("type", "film"))
        conditioner_cfg.setdefault("gps_input_size", self.gps_input_size)
        conditioner_cfg.setdefault("latent_dim", self.latent_dim)
        self.gps_conditioner = GpsConditioner(**conditioner_cfg)
        predictor_cfg = dict(predictor or {})
        predictor_cfg.setdefault("latent_dim", self.latent_dim)
        predictor_cfg.setdefault("max_tokens", encoder_cfg.get("max_tokens", 256))
        self.predictor = TargetLatentPredictor(**predictor_cfg)
        sampler_cfg = dict(mask_sampler or {})
        self.mask_sampler = JepaMaskSampler(**sampler_cfg)

    def forward(
        self,
        *,
        image_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        jepa_epoch: int = 0,
        jepa_step: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        if image_batch is None:
            raise ValueError("gps_conditioned_jepa objective requires image input in 'image_batch'.")
        if gps_batch is None:
            raise ValueError("GPS-conditioned JEPA requires GPS-Rel-Polar input in 'gps_batch'.")
        if gps_batch.ndim != 3:
            raise ValueError(f"GPS-conditioned JEPA expects gps_batch shape [B, T, F], got {tuple(gps_batch.shape)}.")
        if gps_batch.shape[-1] != self.gps_input_size:
            raise ValueError(
                f"GPS-conditioned JEPA expected GPS feature dim {self.gps_input_size}, got {gps_batch.shape[-1]}."
            )
        context_tokens, grid_size = self.context_encoder(image_batch)
        with torch.no_grad():
            target_tokens, _ = self.target_encoder(image_batch)
        batch_size, seq_len, num_tokens, _ = context_tokens.shape
        masks = self.mask_sampler.sample(
            batch_size=batch_size,
            seq_len=seq_len,
            num_tokens=num_tokens,
            grid_size=grid_size,
            gps_batch=gps_batch,
            epoch=int(jepa_epoch),
            step=int(jepa_step),
            device=context_tokens.device,
        )
        context_latent = _gather_tokens(context_tokens, masks.context_indices)
        target_latent = _gather_tokens(target_tokens, masks.target_indices).detach()
        conditioned_context = self.gps_conditioner(context_latent, gps_batch)
        predicted = self.predictor(conditioned_context, masks.target_indices)
        logits = predicted.mean(dim=(2, 3), keepdim=False).unsqueeze(-1).expand(-1, -1, self.num_classes)
        diagnostics: dict[str, Any] = {
            "predicted_target_latent": predicted,
            "target_latent": target_latent,
            "context_mask": masks.context_mask,
            "target_mask": masks.target_mask,
            "loss_mask": masks.loss_mask,
            "ema_decay": float(self.ema_decay),
            "jepa/ema_decay": float(self.ema_decay),
            "jepa/latent_norm": float(predicted.detach().norm(dim=-1).mean().cpu().item()),
            **masks.diagnostics,
        }
        return {
            "logits": logits,
            "input_features": context_latent,
            "output_features": predicted,
            **diagnostics,
        }

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        decay = float(self.ema_decay)
        for target_param, context_param in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target_param.data.mul_(decay).add_(context_param.data, alpha=1.0 - decay)
        for target_buffer, context_buffer in zip(self.target_encoder.buffers(), self.context_encoder.buffers()):
            target_buffer.copy_(context_buffer)


def _ratio(value: float, name: str) -> float:
    ratio = float(value)
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError(f"JEPA mask {name} must be in (0, 1), got {value}.")
    return ratio


def _indices_to_mask(indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
    mask = torch.zeros(*indices.shape[:-1], int(num_tokens), dtype=torch.bool, device=indices.device)
    return mask.scatter(-1, indices, True)


def _gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    expanded = indices.unsqueeze(-1).expand(*indices.shape, tokens.shape[-1])
    return torch.gather(tokens, dim=2, index=expanded)


def _load_context_encoder_state(path: Path, encoder: nn.Module, *, prefix: str, strict: bool) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise CheckpointLoadError(f"JEPA checkpoint payload must be a state dict, got {type(state_dict).__name__}.")
    extracted = _extract_prefixed_state(
        state_dict,
        prefixes=(
            f"{prefix}.",
            f"model.primary.{prefix}.",
            f"primary.{prefix}.",
            f"module.{prefix}.",
        ),
    )
    if not extracted:
        available = sorted(str(key) for key in state_dict.keys())[:20]
        raise CheckpointLoadError(
            f"Could not find JEPA context encoder prefix '{prefix}' in {path}. "
            f"First checkpoint keys: {available}."
        )
    incompatible = encoder.load_state_dict(extracted, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    if strict and (missing or unexpected):
        raise CheckpointLoadError(
            f"Checkpoint mismatch while loading JEPA context encoder from {path}. "
            f"Missing keys: {missing}. Unexpected keys: {unexpected}."
        )


def _extract_prefixed_state(state_dict: dict[str, Any], *, prefixes: tuple[str, ...]) -> dict[str, Any]:
    for prefix in prefixes:
        extracted = {str(key)[len(prefix) :]: value for key, value in state_dict.items() if str(key).startswith(prefix)}
        if extracted:
            return extracted
    return {}


__all__ = [
    "GPSConditionedJEPA",
    "GPSQueryPool",
    "GpsConditioner",
    "JepaContextImageEncoder",
    "JepaMaskSample",
    "JepaMaskSampler",
    "TargetLatentPredictor",
    "VisualPatchTokenEncoder",
]
