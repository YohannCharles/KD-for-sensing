from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.registries import MODELS
from kd_sensing.utils.checkpoint import CheckpointLoadError


PAPER_MODALITIES: tuple[str, ...] = ("image", "radar", "lidar", "gps", "rf")
PAPER_MODALITY_TO_ID = {name: index for index, name in enumerate(PAPER_MODALITIES)}


@dataclass(frozen=True)
class JepaMsacTokenBatch:
    tokens: torch.Tensor
    time_index: torch.Tensor
    modality_index: torch.Tensor
    intra_frame_index: torch.Tensor
    token_ranges: dict[str, tuple[int, int]]
    token_counts: dict[str, int]
    total_frames: int

    def metadata(self) -> dict[str, Any]:
        return {
            "paper_modalities": list(PAPER_MODALITIES),
            "token_ranges": {key: list(value) for key, value in self.token_ranges.items()},
            "token_counts": dict(self.token_counts),
            "total_tokens": int(self.tokens.shape[1]),
            "total_frames": int(self.total_frames),
        }


@dataclass(frozen=True)
class JepaMsacMaskSample:
    keep_mask: torch.Tensor
    mask: torch.Tensor
    keep_indices: torch.Tensor
    mask_indices: torch.Tensor
    diagnostics: dict[str, Any]


class GridModalityTokenizer(nn.Module):
    def __init__(self, *, input_channels: int, latent_dim: int, token_count: int, modality_name: str) -> None:
        super().__init__()
        grid = int(round(float(token_count) ** 0.5))
        if grid * grid != int(token_count):
            raise ValueError(f"{modality_name} token_count must be a square grid, got {token_count}.")
        self.input_channels = int(input_channels)
        self.latent_dim = int(latent_dim)
        self.token_count = int(token_count)
        self.modality_name = str(modality_name)
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.projection = nn.Linear(self.input_channels, self.latent_dim)
        self.norm = nn.LayerNorm(self.latent_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        tensor = value
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(2)
        if tensor.ndim != 5:
            raise ValueError(
                f"{self.modality_name} tokenizer expects [B,T,C,H,W] or [B,T,H,W], got {tuple(value.shape)}."
            )
        batch_size, steps, channels, height, width = tensor.shape
        if int(channels) != self.input_channels:
            raise ValueError(
                f"{self.modality_name} tokenizer expected {self.input_channels} channels, got {int(channels)} "
                f"for input shape {tuple(value.shape)}."
            )
        frames = tensor.reshape(batch_size * steps, channels, height, width)
        pooled = self.pool(frames).flatten(2).transpose(1, 2)
        projected = self.norm(self.projection(pooled))
        return projected.reshape(batch_size, steps, self.token_count, self.latent_dim)


class StateModalityTokenizer(nn.Module):
    def __init__(self, *, input_dim: int, latent_dim: int, modality_name: str) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.modality_name = str(modality_name)
        self.projection = nn.Linear(self.input_dim, self.latent_dim)
        self.norm = nn.LayerNorm(self.latent_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3:
            raise ValueError(f"{self.modality_name} tokenizer expects [B,T,F], got {tuple(value.shape)}.")
        if int(value.shape[-1]) != self.input_dim:
            raise ValueError(
                f"{self.modality_name} tokenizer expected feature dim {self.input_dim}, got {int(value.shape[-1])}."
            )
        return self.norm(self.projection(value)).unsqueeze(2)


class FactorizedJepaMsacPositionEmbedding(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        max_frames: int = 13,
        max_modalities: int = 5,
        max_tokens_per_frame: int = 16,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.max_frames = int(max_frames)
        self.max_modalities = int(max_modalities)
        self.max_tokens_per_frame = int(max_tokens_per_frame)
        self.time = nn.Embedding(self.max_frames, self.latent_dim)
        self.modality = nn.Embedding(self.max_modalities, self.latent_dim)
        self.intra_frame = nn.Embedding(self.max_tokens_per_frame, self.latent_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        time_index: torch.Tensor,
        modality_index: torch.Tensor,
        intra_frame_index: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_index(time_index, self.max_frames, "time", tokens)
        self._validate_index(modality_index, self.max_modalities, "modality", tokens)
        self._validate_index(intra_frame_index, self.max_tokens_per_frame, "intra-frame token", tokens)
        position = (
            self.time(time_index)
            + self.modality(modality_index)
            + self.intra_frame(intra_frame_index)
        ).to(device=tokens.device, dtype=tokens.dtype)
        return tokens + position.unsqueeze(0)

    @staticmethod
    def _validate_index(index: torch.Tensor, limit: int, name: str, tokens: torch.Tensor) -> None:
        if index.numel() == 0:
            raise ValueError(f"JEPA-MSAC {name} index is empty for token shape {tuple(tokens.shape)}.")
        actual = int(index.max().detach().cpu().item())
        if actual >= int(limit):
            raise ValueError(
                f"JEPA-MSAC {name} positional embedding index {actual} exceeds configured max {int(limit) - 1}; "
                f"token shape is {tuple(tokens.shape)}."
            )


class TemporalBlockMaskSampler(nn.Module):
    def __init__(self, *, rho: float = 0.5, pattern: str = "random", seed: int = 0) -> None:
        super().__init__()
        self.rho = float(rho)
        if not 0.0 < self.rho < 1.0:
            raise ValueError(f"JEPA-MSAC mask ratio rho must be in (0, 1), got {rho}.")
        self.pattern = str(pattern).strip().lower()
        if self.pattern not in {"random", "checkerboard", "ablation"}:
            raise ValueError("JEPA-MSAC mask pattern must be random, checkerboard, or ablation.")
        self.seed = int(seed)

    def sample(
        self,
        *,
        time_index: torch.Tensor,
        modality_index: torch.Tensor,
        total_frames: int,
        epoch: int = 0,
        step: int = 0,
    ) -> JepaMsacMaskSample:
        device = time_index.device
        mask = torch.zeros_like(time_index, dtype=torch.bool, device=device)
        block_len = max(1, int(torch.floor(torch.tensor(float(total_frames) * self.rho)).item()))
        for modality_id in sorted(int(item) for item in torch.unique(modality_index).detach().cpu().tolist()):
            selected_frames = self._selected_frames(
                total_frames=int(total_frames),
                block_len=block_len,
                modality_id=modality_id,
                epoch=int(epoch),
                step=int(step),
            )
            modality_mask = modality_index == modality_id
            frame_mask = torch.zeros_like(mask)
            for frame in selected_frames:
                frame_mask |= time_index == int(frame)
            mask |= modality_mask & frame_mask
        keep_mask = ~mask
        if torch.any(keep_mask & mask):
            raise RuntimeError("JEPA-MSAC temporal mask produced overlapping keep and masked tokens.")
        keep_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
        mask_indices = torch.nonzero(mask, as_tuple=False).flatten()
        diagnostics = {
            "mask_pattern": self.pattern,
            "mask_ratio": float(mask.float().mean().detach().cpu().item()),
            "target_token_count": int(mask_indices.numel()),
            "keep_token_count": int(keep_indices.numel()),
            "block_length": int(block_len),
            "rho": float(self.rho),
        }
        return JepaMsacMaskSample(
            keep_mask=keep_mask,
            mask=mask,
            keep_indices=keep_indices,
            mask_indices=mask_indices,
            diagnostics=diagnostics,
        )

    def _selected_frames(
        self,
        *,
        total_frames: int,
        block_len: int,
        modality_id: int,
        epoch: int,
        step: int,
    ) -> list[int]:
        if self.pattern == "checkerboard":
            frames = [frame for frame in range(total_frames) if (frame + modality_id) % 2 == 0]
            return frames[:block_len] or [0]
        if self.pattern == "ablation":
            return list(range(min(block_len, total_frames)))
        gen = torch.Generator().manual_seed(self.seed + epoch * 1_000_003 + step * 9_176 + modality_id * 101)
        max_start = max(total_frames - block_len, 0)
        start = int(torch.randint(max_start + 1, (1,), generator=gen).item()) if max_start else 0
        return list(range(start, min(start + block_len, total_frames)))


class JepaMsacLocalizationHead(nn.Module):
    def __init__(self, *, latent_dim: int, hidden_dim: int = 64, use_constant_velocity: bool = True) -> None:
        super().__init__()
        self.use_constant_velocity = bool(use_constant_velocity)
        self.residual = nn.Sequential(
            nn.LayerNorm(int(latent_dim)),
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, latent: torch.Tensor, gps_history: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        residual = self.residual(latent)
        coarse = torch.zeros_like(residual)
        strategy = "zero"
        if self.use_constant_velocity and gps_history is not None:
            coarse = _constant_velocity(gps_history, steps=int(latent.shape[1])).to(dtype=residual.dtype, device=residual.device)
            strategy = "constant_velocity"
        return coarse + residual, {"coarse_strategy": strategy}


class JepaMsacBeamHead(nn.Module):
    def __init__(self, *, latent_dim: int, num_beams: int = 64, hidden_dim: int = 64, localization_guidance: bool = True) -> None:
        super().__init__()
        self.localization_guidance = bool(localization_guidance)
        input_dim = int(latent_dim) + 2
        self.projection = nn.Linear(input_dim, int(hidden_dim))
        self.gru = nn.GRU(int(hidden_dim), int(hidden_dim), batch_first=True)
        self.output = nn.Linear(int(hidden_dim), int(num_beams))

    def forward(self, latent: torch.Tensor, predicted_location: torch.Tensor | None = None) -> torch.Tensor:
        if self.localization_guidance and predicted_location is not None:
            features = torch.cat([latent, predicted_location.to(dtype=latent.dtype, device=latent.device)], dim=-1)
        else:
            zeros = torch.zeros(*latent.shape[:-1], 2, dtype=latent.dtype, device=latent.device)
            features = torch.cat([latent, zeros], dim=-1)
        hidden, _ = self.gru(torch.tanh(self.projection(features)))
        return self.output(hidden)


class JepaMsacRssiHead(nn.Module):
    def __init__(self, *, latent_dim: int, num_beams: int = 64, hidden_dim: int = 64, localization_guidance: bool = True) -> None:
        super().__init__()
        self.localization_guidance = bool(localization_guidance)
        input_dim = int(latent_dim) + 2
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(num_beams) + 1),
        )
        self.num_beams = int(num_beams)

    def forward(self, latent: torch.Tensor, predicted_location: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.localization_guidance and predicted_location is not None:
            features = torch.cat([latent, predicted_location.to(dtype=latent.dtype, device=latent.device)], dim=-1)
        else:
            zeros = torch.zeros(*latent.shape[:-1], 2, dtype=latent.dtype, device=latent.device)
            features = torch.cat([latent, zeros], dim=-1)
        raw = self.net(features)
        return {
            "rssi_profile": raw[..., : self.num_beams],
            "scalar_rssi": raw[..., self.num_beams],
        }


@MODELS.register("jepa_msac")
class JepaMsacModel(nn.Module):
    supports_modality_kwargs = True

    def __init__(
        self,
        *,
        latent_dim: int = 64,
        t_hist: int = 8,
        t_pred: int = 5,
        num_beams: int = 64,
        image_channels: int = 3,
        radar_channels: int = 1,
        lidar_channels: int = 1,
        gps_input_size: int = 2,
        rf_input_size: int = 64,
        image_tokens: int = 9,
        radar_tokens: int = 16,
        lidar_tokens: int = 16,
        max_frames: int = 13,
        max_tokens_per_frame: int = 16,
        transformer_depth: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        ema_momentum: float = 0.996,
        mask_ratio: float = 0.5,
        mask_pattern: str = "random",
        mask_seed: int = 0,
        localization_guidance: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.t_hist = int(t_hist)
        self.t_pred = int(t_pred)
        self.num_beams = int(num_beams)
        self.ema_momentum = float(ema_momentum)
        self.localization_guidance = bool(localization_guidance)
        self.image_tokenizer = GridModalityTokenizer(
            input_channels=int(image_channels),
            latent_dim=self.latent_dim,
            token_count=int(image_tokens),
            modality_name="Image",
        )
        self.radar_tokenizer = GridModalityTokenizer(
            input_channels=int(radar_channels),
            latent_dim=self.latent_dim,
            token_count=int(radar_tokens),
            modality_name="Radar",
        )
        self.lidar_tokenizer = GridModalityTokenizer(
            input_channels=int(lidar_channels),
            latent_dim=self.latent_dim,
            token_count=int(lidar_tokens),
            modality_name="LiDAR",
        )
        self.gps_tokenizer = StateModalityTokenizer(
            input_dim=int(gps_input_size),
            latent_dim=self.latent_dim,
            modality_name="GPS",
        )
        self.rf_tokenizer = StateModalityTokenizer(
            input_dim=int(rf_input_size),
            latent_dim=self.latent_dim,
            modality_name="RF",
        )
        self.position = FactorizedJepaMsacPositionEmbedding(
            latent_dim=self.latent_dim,
            max_frames=int(max_frames),
            max_modalities=len(PAPER_MODALITIES),
            max_tokens_per_frame=int(max_tokens_per_frame),
        )
        self.context_encoder = _transformer_stack(
            latent_dim=self.latent_dim,
            depth=int(transformer_depth),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.predictor = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, max(self.latent_dim, int(self.latent_dim * mlp_ratio))),
            nn.GELU(),
            nn.Linear(max(self.latent_dim, int(self.latent_dim * mlp_ratio)), self.latent_dim),
        )
        self.mask_sampler = TemporalBlockMaskSampler(rho=float(mask_ratio), pattern=mask_pattern, seed=int(mask_seed))
        self.future_position = nn.Embedding(self.t_pred, self.latent_dim)
        self.future_predictor = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.pretrain_probe = nn.Linear(self.latent_dim, self.num_beams)
        self.localization_head = JepaMsacLocalizationHead(latent_dim=self.latent_dim)
        self.beam_head = JepaMsacBeamHead(
            latent_dim=self.latent_dim,
            num_beams=self.num_beams,
            localization_guidance=self.localization_guidance,
        )
        self.rssi_head = JepaMsacRssiHead(
            latent_dim=self.latent_dim,
            num_beams=self.num_beams,
            localization_guidance=self.localization_guidance,
        )

    def tokenize(
        self,
        *,
        image_batch: torch.Tensor | None = None,
        radar_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        rf_history: torch.Tensor | None = None,
        rf_batch: torch.Tensor | None = None,
        mmwave_batch: torch.Tensor | None = None,
    ) -> JepaMsacTokenBatch:
        inputs = {
            "image": image_batch,
            "radar": radar_batch,
            "lidar": lidar_batch,
            "gps": gps_batch,
            "rf": rf_history if rf_history is not None else rf_batch if rf_batch is not None else mmwave_batch,
        }
        missing = [name for name, value in inputs.items() if value is None]
        if missing:
            raise ValueError(f"JEPA-MSAC requires paper modalities {list(PAPER_MODALITIES)}; missing {missing}.")
        pieces = {
            "image": self.image_tokenizer(inputs["image"]),
            "radar": self.radar_tokenizer(inputs["radar"]),
            "lidar": self.lidar_tokenizer(inputs["lidar"]),
            "gps": self.gps_tokenizer(inputs["gps"]),
            "rf": self.rf_tokenizer(_normalize_rf_history(inputs["rf"])),
        }
        return _concat_token_pieces(pieces)

    def forward(self, *, stage: str = "pretrain", jepa_epoch: int = 0, jepa_step: int = 0, **batch: Any) -> dict[str, Any]:
        if str(stage).strip().lower() in {"heads", "stage2", "downstream"}:
            return self.forward_stage2(**batch)
        schema = self.tokenize(**batch)
        positioned = self.position(
            schema.tokens,
            time_index=schema.time_index,
            modality_index=schema.modality_index,
            intra_frame_index=schema.intra_frame_index,
        )
        masks = self.mask_sampler.sample(
            time_index=schema.time_index,
            modality_index=schema.modality_index,
            total_frames=schema.total_frames,
            epoch=int(jepa_epoch),
            step=int(jepa_step),
        )
        context_input = positioned.clone()
        context_input[:, masks.mask, :] = self.mask_token.to(dtype=positioned.dtype, device=positioned.device)
        context_latent = self.context_encoder(context_input)
        with torch.no_grad():
            target_latent_full = self.target_encoder(positioned).detach()
        predicted_full = self.predictor(context_latent)
        predicted = predicted_full[:, masks.mask_indices, :]
        target = target_latent_full[:, masks.mask_indices, :].detach()
        frame_latent = _frame_pool(predicted_full, schema.time_index, schema.total_frames)
        logits = self.pretrain_probe(frame_latent)
        diagnostics = {
            "predicted_target_latent": predicted,
            "target_latent": target,
            "loss_mask": torch.ones(predicted.shape[:2], dtype=torch.bool, device=predicted.device),
            "context_mask": masks.keep_mask,
            "target_mask": masks.mask,
            "keep_indices": masks.keep_indices,
            "mask_indices": masks.mask_indices,
            "concat_index_metadata": schema.metadata(),
            "ema_momentum": float(self.ema_momentum),
            "jepa_msac/latent_norm": float(predicted.detach().norm(dim=-1).mean().cpu().item()),
            **{f"jepa_msac/{key}": value for key, value in masks.diagnostics.items()},
        }
        return {
            "logits": logits,
            "input_features": context_latent,
            "output_features": predicted_full,
            **diagnostics,
        }

    def future_latent_inference(self, **batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        schema = self.tokenize(**batch)
        positioned = self.position(
            schema.tokens,
            time_index=schema.time_index,
            modality_index=schema.modality_index,
            intra_frame_index=schema.intra_frame_index,
        )
        latent = self.context_encoder(positioned)
        history_summary = latent.mean(dim=1, keepdim=True)
        offsets = torch.arange(self.t_pred, device=latent.device)
        future = history_summary + self.future_position(offsets).to(dtype=latent.dtype).unsqueeze(0)
        predicted = self.future_predictor(future)
        return predicted, {
            "future_mask_slots": int(self.t_pred),
            "pooling_strategy": "global_history_mean_plus_future_position",
            "target_modalities": list(PAPER_MODALITIES),
        }

    def forward_stage2(self, **batch: Any) -> dict[str, Any]:
        s_pred, inference_metadata = self.future_latent_inference(**batch)
        gps_history = batch.get("gps_batch")
        location, loc_metadata = self.localization_head(s_pred, gps_history=gps_history)
        beam_logits = self.beam_head(s_pred, location if self.localization_guidance else None)
        rssi = self.rssi_head(s_pred, location if self.localization_guidance else None)
        return {
            "logits": beam_logits,
            "input_features": s_pred,
            "output_features": s_pred,
            "S_pred": s_pred,
            "predicted_location": location,
            "beam_logits": beam_logits,
            **rssi,
            "stage2_metadata": {
                **inference_metadata,
                **loc_metadata,
                "localization_guidance": bool(self.localization_guidance),
            },
        }

    @torch.no_grad()
    def update_target_encoder_ema(self, momentum: float | None = None) -> None:
        decay = float(self.ema_momentum if momentum is None else momentum)
        for target_param, context_param in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target_param.data.mul_(decay).add_(context_param.data, alpha=1.0 - decay)
        for target_buffer, context_buffer in zip(self.target_encoder.buffers(), self.context_encoder.buffers()):
            target_buffer.copy_(context_buffer)

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "workflow_family": "jepa_msac",
            "paper_workflow_baseline": True,
            "stages": ["pretrain", "heads", "evaluate", "report"],
            "pretraining_metric": "val_jepa_msac_loss",
            "stage1": {
                "objective": "temporal_block_masked_jepa",
                "ema_momentum": float(self.ema_momentum),
                "mask_ratio": float(self.mask_sampler.rho),
                "mask_pattern": self.mask_sampler.pattern,
            },
            "stage2": {
                "freeze_policy": "freeze_context_target_predictor_by_default",
                "task_heads": ["localization", "beam", "rssi"],
                "localization_guidance": bool(self.localization_guidance),
            },
        }


def freeze_jepa_msac_backbone(model: JepaMsacModel) -> dict[str, Any]:
    frozen_prefixes = (
        "image_tokenizer",
        "radar_tokenizer",
        "lidar_tokenizer",
        "gps_tokenizer",
        "rf_tokenizer",
        "position",
        "context_encoder",
        "target_encoder",
        "mask_token",
        "predictor",
        "future_position",
        "future_predictor",
        "pretrain_probe",
    )
    for name, param in model.named_parameters():
        if name.startswith(frozen_prefixes):
            param.requires_grad_(False)
    trainable = [(name, int(param.numel())) for name, param in model.named_parameters() if param.requires_grad]
    frozen = [(name, int(param.numel())) for name, param in model.named_parameters() if not param.requires_grad]
    return {
        "freeze_policy": "stage2_default_freeze_backbone",
        "trainable_parameter_count": int(sum(count for _, count in trainable)),
        "frozen_parameter_count": int(sum(count for _, count in frozen)),
        "trainable_parameter_names": [name for name, _ in trainable],
        "frozen_module_prefixes": list(frozen_prefixes),
    }


def build_frozen_jepa_msac_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_config: Mapping[str, Any] | None = None,
    strict: bool = False,
) -> tuple[JepaMsacModel, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise CheckpointLoadError(f"JEPA-MSAC checkpoint not found: {path}")
    model = JepaMsacModel(**dict(model_config or {}))
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise CheckpointLoadError(f"JEPA-MSAC checkpoint payload must be a state dict, got {type(state_dict).__name__}.")
    incompatible = model.load_state_dict(state_dict, strict=False)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise CheckpointLoadError(
            f"JEPA-MSAC checkpoint mismatch. Missing keys: {sorted(incompatible.missing_keys)}; "
            f"unexpected keys: {sorted(incompatible.unexpected_keys)}."
        )
    metadata = freeze_jepa_msac_backbone(model)
    metadata.update(
        {
            "checkpoint_path": str(path),
            "missing_keys": sorted(incompatible.missing_keys),
            "unexpected_keys": sorted(incompatible.unexpected_keys),
        }
    )
    return model, metadata


def stage2_optimizer_parameters(model: JepaMsacModel) -> list[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def _transformer_stack(*, latent_dim: int, depth: int, num_heads: int, mlp_ratio: float) -> nn.Module:
    if depth <= 0:
        return nn.Identity()
    layer = nn.TransformerEncoderLayer(
        d_model=int(latent_dim),
        nhead=int(num_heads),
        dim_feedforward=max(int(latent_dim), int(latent_dim * mlp_ratio)),
        dropout=0.0,
        batch_first=True,
        activation="gelu",
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=int(depth))


def _normalize_rf_history(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 4 and int(value.shape[2]) == 1:
        value = value.squeeze(2)
    if value.ndim != 3:
        raise ValueError(f"RF history expects [B,T,K] or [B,T,1,K], got {tuple(value.shape)}.")
    return value


def _concat_token_pieces(pieces: Mapping[str, torch.Tensor]) -> JepaMsacTokenBatch:
    flat_tokens: list[torch.Tensor] = []
    time_indices: list[torch.Tensor] = []
    modality_indices: list[torch.Tensor] = []
    intra_indices: list[torch.Tensor] = []
    token_ranges: dict[str, tuple[int, int]] = {}
    token_counts: dict[str, int] = {}
    cursor = 0
    total_frames = None
    device = next(iter(pieces.values())).device
    for modality_name in PAPER_MODALITIES:
        tokens = pieces[modality_name]
        if tokens.ndim != 4:
            raise ValueError(f"{modality_name} tokenizer output must be [B,T,N,D], got {tuple(tokens.shape)}.")
        steps = int(tokens.shape[1])
        token_count = int(tokens.shape[2])
        if total_frames is None:
            total_frames = steps
        elif total_frames != steps:
            raise ValueError(f"JEPA-MSAC modality frame mismatch: expected {total_frames}, got {steps} for {modality_name}.")
        flat = tokens.reshape(tokens.shape[0], steps * token_count, tokens.shape[-1])
        flat_tokens.append(flat)
        time_indices.append(torch.arange(steps, device=device).repeat_interleave(token_count))
        modality_indices.append(torch.full((steps * token_count,), PAPER_MODALITY_TO_ID[modality_name], dtype=torch.long, device=device))
        intra_indices.append(torch.arange(token_count, device=device).repeat(steps))
        token_ranges[modality_name] = (cursor, cursor + steps * token_count)
        token_counts[modality_name] = token_count
        cursor += steps * token_count
    return JepaMsacTokenBatch(
        tokens=torch.cat(flat_tokens, dim=1),
        time_index=torch.cat(time_indices),
        modality_index=torch.cat(modality_indices),
        intra_frame_index=torch.cat(intra_indices),
        token_ranges=token_ranges,
        token_counts=token_counts,
        total_frames=int(total_frames or 0),
    )


def _frame_pool(tokens: torch.Tensor, time_index: torch.Tensor, total_frames: int) -> torch.Tensor:
    pooled = []
    for frame in range(int(total_frames)):
        selected = tokens[:, time_index == frame, :]
        pooled.append(selected.mean(dim=1))
    return torch.stack(pooled, dim=1)


def _constant_velocity(gps_history: torch.Tensor, *, steps: int) -> torch.Tensor:
    if gps_history.ndim != 3 or gps_history.shape[-1] < 2:
        raise ValueError(f"GPS history for constant velocity must have shape [B,T,F>=2], got {tuple(gps_history.shape)}.")
    xy = gps_history[..., :2]
    last = xy[:, -1, :]
    prev = xy[:, -2, :] if xy.shape[1] > 1 else xy[:, -1, :]
    velocity = last - prev
    offsets = torch.arange(1, int(steps) + 1, device=xy.device, dtype=xy.dtype).view(1, steps, 1)
    return last.unsqueeze(1) + offsets * velocity.unsqueeze(1)


__all__ = [
    "PAPER_MODALITIES",
    "JepaMsacModel",
    "JepaMsacTokenBatch",
    "JepaMsacMaskSample",
    "GridModalityTokenizer",
    "StateModalityTokenizer",
    "FactorizedJepaMsacPositionEmbedding",
    "TemporalBlockMaskSampler",
    "JepaMsacLocalizationHead",
    "JepaMsacBeamHead",
    "JepaMsacRssiHead",
    "freeze_jepa_msac_backbone",
    "build_frozen_jepa_msac_from_checkpoint",
    "stage2_optimizer_parameters",
]
