from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.modalities import normalize_modalities
from kd_sensing.registries import MODELS


SUPPORTED_MODALITIES = ("image", "radar", "gps", "lidar")
GPS_PATHWAYS = {"dual_path", "spatial_only", "global_only", "none"}
TEMPORAL_CORES = {"transformer", "single_frame", "mean_pool_temporal"}
FUSION_CORES = {"bev_spatial", "one_d_fusion"}


@MODELS.register("bev_fusion_2604")
class BEVFusion2604Net(nn.Module):
    supports_force_modality_mask = True

    def __init__(
        self,
        modalities: list[str] | tuple[str, ...] | None = None,
        *,
        num_classes: int = 64,
        num_pred: int = 1,
        d_model: int = 256,
        bev_size: list[int] | tuple[int, int] = (128, 128),
        image_channels: int = 3,
        radar_channels: int = 2,
        lidar_channels: int = 3,
        gps_input_size: int = 3,
        camera_backbone: str | dict[str, Any] | None = None,
        camera_to_bev: dict[str, Any] | None = None,
        gps_spatial: dict[str, Any] | None = None,
        gps_pathway: str | dict[str, Any] = "dual_path",
        fusion_core: str | dict[str, Any] = "bev_spatial",
        temporal_core: str | dict[str, Any] = "transformer",
        dropout: float = 0.1,
        ablation_name: str | None = None,
        paper_approximation: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.modalities = _normalize_bev_modalities(modalities)
        self.num_classes = int(num_classes)
        self.num_pred = int(num_pred)
        self.d_model = int(d_model)
        self.bev_size = _as_hw(bev_size, name="bev_size")
        self.gps_pathway = _gps_pathway_mode(gps_pathway)
        self.fusion_core = _core_mode(fusion_core, FUSION_CORES, "fusion_core")
        self.temporal_core = _core_mode(temporal_core, TEMPORAL_CORES, "temporal_core")
        self.ablation_name = ablation_name
        self.paper_approximation = bool(paper_approximation)

        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if self.num_pred <= 0:
            raise ValueError(f"num_pred must be positive, got {num_pred}.")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if "gps" not in self.modalities and self.gps_pathway != "none":
            self.gps_pathway = "none"

        camera_cfg = _mapping_from_config(camera_backbone, default_type="lightweight_cnn")
        cross_cfg = dict(camera_to_bev or {})
        gps_spatial_cfg = dict(gps_spatial or {})
        temporal_cfg = _mapping_from_config(temporal_core, default_type=self.temporal_core)
        fusion_cfg = _mapping_from_config(fusion_core, default_type=self.fusion_core)

        self.branch_names = self._spatial_branch_names()
        self.camera_backbone_name = str(camera_cfg.get("type", "lightweight_cnn"))
        self.camera_attention_layers = int(cross_cfg.get("num_layers", cross_cfg.get("layers", 3)))
        self.camera_attention_heads = int(cross_cfg.get("num_heads", cross_cfg.get("heads", 4)))
        if "image" in self.modalities:
            self.camera_backbone = _build_camera_backbone(
                camera_cfg,
                in_channels=int(image_channels),
                d_model=self.d_model,
            )
            self.camera_to_bev = CameraToBEV(
                d_model=self.d_model,
                bev_size=self.bev_size,
                num_layers=self.camera_attention_layers,
                num_heads=self.camera_attention_heads,
                dropout=float(cross_cfg.get("dropout", dropout)),
            )

        if "radar" in self.modalities:
            self.radar_projection = SpatialProjection(int(radar_channels), self.d_model)
        if "lidar" in self.modalities:
            self.lidar_projection = SpatialProjection(int(lidar_channels), self.d_model)
        if self._uses_gps_spatial:
            self.gps_spatial_projection = SpatialProjection(1, self.d_model)

        self.gps_roi = _as_roi(gps_spatial_cfg.get("roi", gps_spatial_cfg.get("bounds", (-60.0, 60.0, -60.0, 60.0))))
        self.gps_gaussian_sigma = float(gps_spatial_cfg.get("gaussian_sigma_cells", gps_spatial_cfg.get("sigma_cells", 1.5)))
        self.gps_clip_out_of_bounds = bool(gps_spatial_cfg.get("clip_out_of_bounds", True))
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.bev_size[0], dtype=torch.float32),
            torch.arange(self.bev_size[1], dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("_bev_grid_x", grid_x, persistent=False)
        self.register_buffer("_bev_grid_y", grid_y, persistent=False)

        spatial_branch_count = max(len(self.branch_names), 1)
        self.bev_fusion = BEVFusionBlock(
            in_channels=self.d_model * spatial_branch_count,
            d_model=self.d_model,
            dropout=float(fusion_cfg.get("dropout", dropout)),
        )
        if self.fusion_core == "one_d_fusion":
            self.one_d_fusion = nn.Sequential(
                nn.Linear(self.d_model * spatial_branch_count, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Dropout(float(fusion_cfg.get("dropout", dropout))),
            )

        if self._uses_gps_global:
            self.gps_global_mlp = nn.Sequential(
                nn.Linear(int(gps_input_size), self.d_model),
                nn.LayerNorm(self.d_model),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.d_model, self.d_model),
            )
            self.gps_global_projection = nn.Linear(self.d_model, self.d_model)
            self.gps_global_gate = nn.Parameter(torch.tensor(0.0))

        max_seq_len = int(temporal_cfg.get("max_seq_len", 16))
        self.time_embedding = nn.Parameter(torch.zeros(max_seq_len, self.d_model))
        if self.temporal_core == "transformer":
            num_heads = int(temporal_cfg.get("num_heads", temporal_cfg.get("heads", 4)))
            num_layers = int(temporal_cfg.get("num_layers", temporal_cfg.get("layers", 4)))
            if self.d_model % num_heads != 0:
                raise ValueError(f"d_model ({self.d_model}) must be divisible by temporal num_heads ({num_heads}).")
            layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=num_heads,
                dim_feedforward=int(temporal_cfg.get("dim_feedforward", self.d_model * 4)),
                dropout=float(temporal_cfg.get("dropout", dropout)),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal_transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.temporal_layers = num_layers
            self.temporal_heads = num_heads
        else:
            self.temporal_layers = 0
            self.temporal_heads = 0

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, self.num_pred * self.num_classes),
        )

    @property
    def _uses_gps_spatial(self) -> bool:
        return "gps" in self.modalities and self.gps_pathway in {"dual_path", "spatial_only"}

    @property
    def _uses_gps_global(self) -> bool:
        return "gps" in self.modalities and self.gps_pathway in {"dual_path", "global_only"}

    def forward(
        self,
        image_batch: torch.Tensor | None = None,
        radar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        gps_bev_xy_batch: torch.Tensor | None = None,
        force_modality_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        active_modalities = self._active_modalities(force_modality_mask)
        spatial_features: dict[str, torch.Tensor] = {}
        pooled_features: dict[str, torch.Tensor] = {}
        diagnostics: dict[str, Any] = {
            "modalities": self.modalities,
            "effective_modalities": active_modalities,
            "bev_size": self.bev_size,
            "d_model": self.d_model,
            "gps_pathway": self.gps_pathway,
            "fusion_core": self.fusion_core,
            "temporal_core": self.temporal_core,
            "ablation_name": self.ablation_name,
            "paper_approximation": self.paper_approximation,
        }
        batch_size = None
        seq_len = None

        if "image" in active_modalities:
            image_batch = self._require_tensor(image_batch, "image", active_modalities)
            batch_size, seq_len = _check_sequence_dims(image_batch, "image", batch_size, seq_len, expected_ndim=5)
            camera_tokens = self.camera_backbone(image_batch)
            camera_bev = self.camera_to_bev(camera_tokens, batch_size=batch_size, seq_len=seq_len)
            spatial_features["image"] = camera_bev
            pooled_features["image"] = _pool_bev_sequence(camera_bev)
            diagnostics["camera_bev_shape"] = tuple(int(dim) for dim in camera_bev.shape)

        if "radar" in active_modalities:
            radar_batch = self._require_tensor(radar_batch, "radar", active_modalities)
            batch_size, seq_len = _check_sequence_dims(radar_batch, "radar", batch_size, seq_len, expected_ndim=5)
            radar_bev = self._project_spatial_sequence(self.radar_projection, radar_batch, "radar")
            spatial_features["radar"] = radar_bev
            pooled_features["radar"] = _pool_bev_sequence(radar_bev)
            diagnostics["radar_mapping_profile"] = {
                "input_shape": tuple(int(dim) for dim in radar_batch.shape),
                "output_shape": tuple(int(dim) for dim in radar_bev.shape),
                "projection": "cnn_interpolate_to_bev",
            }

        if "lidar" in active_modalities:
            lidar_batch = self._require_tensor(lidar_batch, "lidar", active_modalities)
            if lidar_batch.ndim == 6:
                lidar_batch = lidar_batch.mean(dim=3)
            batch_size, seq_len = _check_sequence_dims(lidar_batch, "lidar", batch_size, seq_len, expected_ndim=5)
            lidar_bev = self._project_spatial_sequence(self.lidar_projection, lidar_batch, "lidar")
            spatial_features["lidar"] = lidar_bev
            pooled_features["lidar"] = _pool_bev_sequence(lidar_bev)
            diagnostics["lidar_original_shape"] = tuple(int(dim) for dim in lidar_batch.shape)
            diagnostics["lidar_aligned_shape"] = tuple(int(dim) for dim in lidar_bev.shape)

        gps_sequence_features = None
        gps_global_embedding = None
        if "gps" in active_modalities and self._uses_gps_global:
            gps_batch = self._require_tensor(gps_batch, "gps", active_modalities)
            batch_size, seq_len = _check_sequence_dims(gps_batch, "gps", batch_size, seq_len, expected_ndim=3)
            gps_sequence_features = self.gps_global_mlp(gps_batch.to(dtype=self._module_dtype()))
            gps_global_embedding = gps_sequence_features.mean(dim=1)
            diagnostics["gps_global_shape"] = tuple(int(dim) for dim in gps_global_embedding.shape)

        if "gps" in active_modalities and self._uses_gps_spatial:
            gps_bev_xy_batch = self._require_gps_bev_xy(gps_bev_xy_batch, active_modalities)
            batch_size, seq_len = _check_sequence_dims(gps_bev_xy_batch, "gps_bev_xy", batch_size, seq_len, expected_ndim=3)
            gps_mask, gps_diag = self._gps_xy_to_mask(gps_bev_xy_batch)
            gps_bev = self._project_spatial_sequence(self.gps_spatial_projection, gps_mask, "gps_spatial")
            spatial_features["gps"] = gps_bev
            pooled_features["gps"] = _pool_bev_sequence(gps_bev)
            diagnostics["gps_spatial"] = gps_diag
            diagnostics["gps_spatial_bev_shape"] = tuple(int(dim) for dim in gps_bev.shape)

        if batch_size is None or seq_len is None:
            raise ValueError("bev_fusion_2604 received no usable active modalities.")

        if self.fusion_core == "one_d_fusion":
            fused_sequence, fused_bev = self._one_d_fusion_sequence(
                pooled_features,
                gps_sequence_features=gps_sequence_features,
                batch_size=batch_size,
                seq_len=seq_len,
            )
        else:
            fused_bev = self._bev_fusion_sequence(spatial_features, batch_size=batch_size, seq_len=seq_len)
            fused_sequence = _pool_bev_sequence(fused_bev)

        temporal_output, final_representation = self._temporal_aggregate(fused_sequence)
        if gps_global_embedding is not None:
            gate = torch.tanh(self.gps_global_gate)
            final_representation = final_representation + gate * self.gps_global_projection(gps_global_embedding)
            diagnostics["gps_global_gate"] = float(gate.detach().cpu().item())

        logits = self.classifier(final_representation).view(batch_size, self.num_pred, self.num_classes)
        diagnostics["bev_feature_shape"] = tuple(int(dim) for dim in fused_bev.shape)
        diagnostics["input_feature_shape"] = tuple(int(dim) for dim in fused_sequence.shape)
        diagnostics["output_feature_shape"] = tuple(int(dim) for dim in temporal_output.shape)
        diagnostics["effective_modalities"] = active_modalities
        diagnostics["modality_bev_shapes"] = {
            name: tuple(int(dim) for dim in value.shape)
            for name, value in spatial_features.items()
        }
        return {
            "logits": logits,
            "input_features": fused_sequence,
            "output_features": temporal_output,
            "bev_features": fused_bev,
            **diagnostics,
        }

    def _active_modalities(self, force_modality_mask: torch.Tensor | None) -> tuple[str, ...]:
        if force_modality_mask is None:
            return self.modalities
        mask = force_modality_mask.detach().cpu().to(torch.bool).flatten()
        if int(mask.numel()) != len(self.modalities):
            raise ValueError(
                f"force_modality_mask length ({int(mask.numel())}) must match enabled modalities "
                f"({len(self.modalities)}): {list(self.modalities)}."
            )
        active = tuple(name for name, keep in zip(self.modalities, mask.tolist()) if keep)
        if not active:
            raise ValueError("force_modality_mask leaves no active modalities for bev_fusion_2604.")
        return active

    def _spatial_branch_names(self) -> tuple[str, ...]:
        names = [name for name in self.modalities if name in {"image", "radar", "lidar"}]
        if self._uses_gps_spatial:
            names.append("gps")
        if not names and self._uses_gps_global:
            names.append("gps_global")
        return tuple(names)

    def _require_tensor(
        self,
        value: torch.Tensor | None,
        modality: str,
        active_modalities: tuple[str, ...],
    ) -> torch.Tensor:
        if value is None:
            raise ValueError(
                f"bev_fusion_2604 requires '{modality}' input because it is enabled. "
                f"Enabled modalities: {list(self.modalities)}; active modalities: {list(active_modalities)}."
            )
        return value

    def _require_gps_bev_xy(
        self,
        value: torch.Tensor | None,
        active_modalities: tuple[str, ...],
    ) -> torch.Tensor:
        if value is None:
            raise ValueError(
                "bev_fusion_2604 GPS spatial pathway requires 'gps_bev_xy_batch'. "
                "Enable data.dataset.use_gps_bev_xy=true or use gps_pathway='global_only'. "
                f"Enabled modalities: {list(self.modalities)}; active modalities: {list(active_modalities)}."
            )
        return value

    def _project_spatial_sequence(self, projection: nn.Module, tensor: torch.Tensor, name: str) -> torch.Tensor:
        batch_size, seq_len, channels, height, width = tensor.shape
        frames = tensor.reshape(batch_size * seq_len, channels, height, width).to(dtype=self._module_dtype())
        projected = projection(frames)
        if tuple(projected.shape[-2:]) != self.bev_size:
            projected = F.interpolate(projected, size=self.bev_size, mode="bilinear", align_corners=False)
        return projected.reshape(batch_size, seq_len, self.d_model, self.bev_size[0], self.bev_size[1])

    def _gps_xy_to_mask(self, xy: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if xy.ndim != 3 or int(xy.shape[-1]) != 2:
            raise ValueError(f"gps_bev_xy_batch must have shape [B, T, 2], got {tuple(xy.shape)}.")
        xy = xy.to(dtype=self._module_dtype())
        x_min, x_max, y_min, y_max = self.gps_roi
        span_x = max(x_max - x_min, 1e-6)
        span_y = max(y_max - y_min, 1e-6)
        valid = torch.isfinite(xy).all(dim=-1)
        gx = (xy[..., 0] - x_min) / span_x * float(self.bev_size[1] - 1)
        gy = (xy[..., 1] - y_min) / span_y * float(self.bev_size[0] - 1)
        inside = valid & gx.ge(0) & gx.le(self.bev_size[1] - 1) & gy.ge(0) & gy.le(self.bev_size[0] - 1)
        if bool((valid & ~inside).any().item()) and not self.gps_clip_out_of_bounds:
            raise ValueError(
                "gps_bev_xy_batch contains coordinates outside the configured GPS BEV ROI. "
                "Expand gps_spatial.roi or set gps_spatial.clip_out_of_bounds=true."
            )
        gx = gx.clamp(0, self.bev_size[1] - 1)
        gy = gy.clamp(0, self.bev_size[0] - 1)
        if self.gps_gaussian_sigma <= 0:
            mask = torch.zeros(*xy.shape[:2], self.bev_size[0], self.bev_size[1], dtype=xy.dtype, device=xy.device)
            linear = gy.round().to(torch.long) * self.bev_size[1] + gx.round().to(torch.long)
            flat = mask.reshape(*xy.shape[:2], -1)
            flat.scatter_(-1, linear.unsqueeze(-1), inside.to(dtype=xy.dtype).unsqueeze(-1))
        else:
            dx = self._bev_grid_x.to(device=xy.device, dtype=xy.dtype).view(1, 1, *self.bev_size) - gx.unsqueeze(-1).unsqueeze(-1)
            dy = self._bev_grid_y.to(device=xy.device, dtype=xy.dtype).view(1, 1, *self.bev_size) - gy.unsqueeze(-1).unsqueeze(-1)
            sigma2 = max(float(self.gps_gaussian_sigma) ** 2, 1e-6)
            mask = torch.exp(-(dx * dx + dy * dy) / (2.0 * sigma2))
            mask = mask * valid.to(dtype=xy.dtype).unsqueeze(-1).unsqueeze(-1)
        diagnostics = {
            "input_shape": tuple(int(dim) for dim in xy.shape),
            "mask_shape": tuple(int(dim) for dim in mask.unsqueeze(2).shape),
            "roi": [float(value) for value in self.gps_roi],
            "gaussian_sigma_cells": float(self.gps_gaussian_sigma),
            "clip_out_of_bounds": bool(self.gps_clip_out_of_bounds),
            "clipped_points": int((valid & ~inside).detach().sum().cpu().item()),
            "invalid_points": int((~valid).detach().sum().cpu().item()),
        }
        return mask.unsqueeze(2), diagnostics

    def _bev_fusion_sequence(
        self,
        spatial_features: dict[str, torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        if not spatial_features:
            return torch.zeros(
                batch_size,
                seq_len,
                self.d_model,
                self.bev_size[0],
                self.bev_size[1],
                dtype=self._module_dtype(),
                device=next(self.parameters()).device,
            )
        ordered = [spatial_features[name] for name in self.branch_names if name in spatial_features]
        fused_input = torch.cat(ordered, dim=2)
        frames = fused_input.reshape(batch_size * seq_len, -1, self.bev_size[0], self.bev_size[1])
        fused = self.bev_fusion(frames)
        return fused.reshape(batch_size, seq_len, self.d_model, self.bev_size[0], self.bev_size[1])

    def _one_d_fusion_sequence(
        self,
        pooled_features: dict[str, torch.Tensor],
        *,
        gps_sequence_features: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ordered = [pooled_features[name] for name in self.branch_names if name in pooled_features]
        if not ordered and gps_sequence_features is not None:
            ordered = [gps_sequence_features]
        if not ordered:
            raise ValueError("one_d_fusion requires at least one active feature sequence.")
        while len(ordered) < len(self.branch_names):
            ordered.append(torch.zeros_like(ordered[0]))
        fused = self.one_d_fusion(torch.cat(ordered, dim=-1))
        bev = fused.view(batch_size, seq_len, self.d_model, 1, 1).expand(
            batch_size,
            seq_len,
            self.d_model,
            self.bev_size[0],
            self.bev_size[1],
        )
        return fused, bev

    def _temporal_aggregate(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3:
            raise ValueError(f"Temporal input must have shape [B, T, D], got {tuple(sequence.shape)}.")
        seq_len = int(sequence.shape[1])
        if seq_len > int(self.time_embedding.shape[0]):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {int(self.time_embedding.shape[0])} "
                "for bev_fusion_2604 time embedding."
            )
        x = sequence + self.time_embedding[:seq_len].to(device=sequence.device, dtype=sequence.dtype).unsqueeze(0)
        if self.temporal_core == "transformer":
            temporal = self.temporal_transformer(x)
            return temporal, temporal[:, -1, :]
        if self.temporal_core == "single_frame":
            temporal = x[:, -1:, :]
            return temporal, temporal[:, -1, :]
        temporal = x.mean(dim=1, keepdim=True)
        return temporal, temporal[:, 0, :]

    def _module_dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "bev_fusion_2604",
            "modalities": list(self.modalities),
            "bev_size": list(self.bev_size),
            "d_model": self.d_model,
            "num_classes": self.num_classes,
            "num_pred": self.num_pred,
            "camera_backbone": self.camera_backbone_name,
            "camera_to_bev_layers": self.camera_attention_layers,
            "camera_to_bev_heads": self.camera_attention_heads,
            "temporal_core": self.temporal_core,
            "temporal_layers": self.temporal_layers,
            "temporal_heads": self.temporal_heads,
            "fusion_core": self.fusion_core,
            "gps_pathway": self.gps_pathway,
            "gps_roi": [float(value) for value in self.gps_roi],
            "ablation_name": self.ablation_name,
            "paper_approximation": self.paper_approximation,
        }


class CameraToBEV(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        bev_size: tuple[int, int],
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % int(num_heads) != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by camera attention heads ({num_heads}).")
        self.d_model = int(d_model)
        self.bev_size = bev_size
        self.query_count = int(bev_size[0] * bev_size[1])
        self.bev_queries = nn.Parameter(torch.randn(self.query_count, self.d_model) * 0.02)
        self.layers = nn.ModuleList(
            CrossAttentionBlock(
                d_model=self.d_model,
                num_heads=int(num_heads),
                dropout=float(dropout),
            )
            for _ in range(max(int(num_layers), 1))
        )

    def forward(self, feature_map: torch.Tensor, *, batch_size: int, seq_len: int) -> torch.Tensor:
        if feature_map.ndim != 4:
            raise ValueError(f"Camera backbone must return [B*T, D, H, W], got {tuple(feature_map.shape)}.")
        if int(feature_map.shape[0]) != int(batch_size * seq_len):
            raise ValueError("Camera feature map batch dimension must equal B*T.")
        if int(feature_map.shape[1]) != self.d_model:
            raise ValueError(
                f"Camera feature channel count must equal d_model={self.d_model}, got {int(feature_map.shape[1])}."
            )
        tokens = feature_map.flatten(2).transpose(1, 2)
        queries = self.bev_queries.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, tokens)
        bev = queries.transpose(1, 2).reshape(batch_size, seq_len, self.d_model, self.bev_size[0], self.bev_size[1])
        return bev


class CrossAttentionBlock(nn.Module):
    def __init__(self, *, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.query_norm(queries), self.context_norm(context), self.context_norm(context), need_weights=False)
        queries = queries + attn_out
        return queries + self.ffn(queries)


class SpatialProjection(nn.Module):
    def __init__(self, in_channels: int, d_model: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(d_model), d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(d_model), d_model),
            nn.GELU(),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if int(tensor.shape[1]) != self.in_channels:
            raise ValueError(
                f"Spatial projection expected {self.in_channels} channels, got {int(tensor.shape[1])}."
            )
        return self.net(tensor)


class BEVFusionBlock(nn.Module):
    def __init__(self, *, in_channels: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), int(d_model), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(int(d_model)), int(d_model)),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(int(d_model), int(d_model), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(int(d_model)), int(d_model)),
            nn.GELU(),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.net(tensor)


class LightweightCameraBackbone(nn.Module):
    def __init__(self, *, in_channels: int, d_model: int, width: int = 48) -> None:
        super().__init__()
        hidden = max(int(width), min(int(d_model), 64))
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), hidden, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, int(d_model), kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(int(d_model)), int(d_model)),
            nn.GELU(),
        )

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, height, width = image_batch.shape
        frames = image_batch.reshape(batch_size * seq_len, channels, height, width).to(dtype=next(self.parameters()).dtype)
        return self.net(frames)


class ResNetCameraBackbone(nn.Module):
    def __init__(self, *, name: str, in_channels: int, d_model: int, pretrained: bool, weights: str | None, freeze_backbone: bool) -> None:
        super().__init__()
        if int(in_channels) != 3:
            raise ValueError("ResNet camera backbones require RGB image input with 3 channels.")
        try:
            import torchvision.models as tv_models
        except Exception as exc:  # pragma: no cover - environment-dependent.
            raise RuntimeError("ResNet camera backbone requires torchvision in the kd_mm_beam environment.") from exc
        if name == "resnet34":
            enum = getattr(tv_models, "ResNet34_Weights", None)
            builder = tv_models.resnet34
        elif name == "resnet18":
            enum = getattr(tv_models, "ResNet18_Weights", None)
            builder = tv_models.resnet18
        else:
            raise ValueError("camera_backbone.type must be 'lightweight_cnn', 'resnet18', or 'resnet34'.")
        weights_obj = None
        if pretrained:
            if enum is None:
                model = builder(pretrained=True)
            else:
                weights_obj = enum.DEFAULT if weights in (None, "", "DEFAULT", "default") else getattr(enum, str(weights))
                model = builder(weights=weights_obj)
        else:
            model = builder(weights=None)
        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        if freeze_backbone:
            for param in self.stem.parameters():
                param.requires_grad = False
        self.projection = nn.Conv2d(int(model.fc.in_features), int(d_model), kernel_size=1)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, height, width = image_batch.shape
        frames = image_batch.reshape(batch_size * seq_len, channels, height, width).to(dtype=next(self.parameters()).dtype)
        features = self.stem(frames)
        return self.projection(features)


def _build_camera_backbone(cfg: dict[str, Any], *, in_channels: int, d_model: int) -> nn.Module:
    kind = str(cfg.get("type", "lightweight_cnn")).strip().lower()
    if kind in {"lightweight", "lightweight_cnn", "smoke_cnn"}:
        return LightweightCameraBackbone(
            in_channels=in_channels,
            d_model=d_model,
            width=int(cfg.get("width", 48)),
        )
    if kind in {"resnet18", "resnet34"}:
        return ResNetCameraBackbone(
            name=kind,
            in_channels=in_channels,
            d_model=d_model,
            pretrained=bool(cfg.get("pretrained", False)),
            weights=cfg.get("weights"),
            freeze_backbone=bool(cfg.get("freeze_backbone", False)),
        )
    raise ValueError("camera_backbone.type must be 'lightweight_cnn', 'resnet18', or 'resnet34'.")


def _pool_bev_sequence(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.mean(dim=(-2, -1))


def _check_sequence_dims(
    tensor: torch.Tensor,
    name: str,
    batch_size: int | None,
    seq_len: int | None,
    *,
    expected_ndim: int,
) -> tuple[int, int]:
    if tensor.ndim != expected_ndim:
        raise ValueError(f"{name} input must have shape with {expected_ndim} dims, got {tuple(tensor.shape)}.")
    current_batch, current_seq = int(tensor.shape[0]), int(tensor.shape[1])
    if batch_size is not None and (current_batch != batch_size or current_seq != seq_len):
        raise ValueError("Enabled bev_fusion_2604 modalities must share batch and sequence dimensions.")
    return current_batch, current_seq


def _normalize_bev_modalities(modalities: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    selected = normalize_modalities(tuple(modalities or SUPPORTED_MODALITIES), context="bev_fusion_2604 modalities")
    unsupported = [name for name in selected if name not in SUPPORTED_MODALITIES]
    if unsupported:
        raise ValueError(f"bev_fusion_2604 supports only {list(SUPPORTED_MODALITIES)}, got {unsupported}.")
    return selected


def _as_hw(value: list[int] | tuple[int, int], *, name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain [height, width].")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} values must be positive, got {value}.")
    return height, width


def _as_roi(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("GPS spatial roi must contain [x_min, x_max, y_min, y_max].")
    x_min, x_max, y_min, y_max = (float(item) for item in value)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("GPS spatial roi max bounds must be greater than min bounds.")
    return x_min, x_max, y_min, y_max


def _mapping_from_config(raw: Any, *, default_type: str) -> dict[str, Any]:
    if isinstance(raw, str):
        return {"type": raw}
    if isinstance(raw, dict):
        result = dict(raw)
        result.setdefault("type", default_type)
        return result
    return {"type": default_type}


def _gps_pathway_mode(raw: str | dict[str, Any]) -> str:
    if isinstance(raw, dict):
        raw = raw.get("mode", raw.get("type", "dual_path"))
    mode = str(raw or "dual_path").strip().lower()
    if mode in {"gps_dual_path", "dual"}:
        mode = "dual_path"
    if mode in {"spatial", "gps_spatial_only"}:
        mode = "spatial_only"
    if mode in {"global", "gps_global_only"}:
        mode = "global_only"
    if mode not in GPS_PATHWAYS:
        raise ValueError(f"gps_pathway must be one of {sorted(GPS_PATHWAYS)}, got '{raw}'.")
    return mode


def _core_mode(raw: str | dict[str, Any], allowed: set[str], name: str) -> str:
    if isinstance(raw, dict):
        raw = raw.get("type", raw.get("mode", next(iter(allowed))))
    mode = str(raw).strip().lower()
    if mode not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got '{raw}'.")
    return mode


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


__all__ = ["BEVFusion2604Net"]
