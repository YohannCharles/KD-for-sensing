from typing import Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank
from kd_sensing.modalities import MODALITY_ORDER
from kd_sensing.models.reliability_biased_missing_attention import ReliabilityBiasedMissingAwareAttention
from kd_sensing.registries import ENCODERS, MODELS


DEFAULT_MODALITIES = ("image", "radar", "lidar", "gps")


class ModalityReliabilityHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(int(d_model)), nn.Linear(int(d_model), int(d_model) * 2))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.net(z).chunk(2, dim=-1)
        return mu, logvar


class FullModalTeacher(nn.Module):
    def __init__(self, d_model: int, num_beams: int, num_heads: int = 4, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, int(d_model)))
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(num_heads),
            dim_feedforward=max(int(d_model) * 4, 64),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.head = BeamPredictionHead(int(d_model), int(num_beams), dropout=dropout)

    def forward(self, tokens: torch.Tensor, modality_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(tokens.shape[0])
        query = self.query.expand(batch_size, -1, -1)
        memory = self.encoder(torch.cat([query, tokens + modality_embedding.unsqueeze(0)], dim=1))
        u_star = memory[:, 0, :]
        return u_star, self.head(u_star)


class SetContextEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, int(d_model)))
        self.reliability_projection = nn.Linear(1, int(d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(num_heads),
            dim_feedforward=max(int(d_model) * 4, 64),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        reliability: torch.Tensor,
        modality_embedding: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(tokens.shape[0])
        query = self.query.expand(batch_size, -1, -1)
        token_input = tokens + modality_embedding.unsqueeze(0) + self.reliability_projection(reliability)
        padding = torch.cat([torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device), ~mask], dim=1)
        memory = self.encoder(torch.cat([query, token_input], dim=1), src_key_padding_mask=padding)
        return memory[:, 0, :]


class GaussianJEPAPredictor(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int | None = None, logvar_min: float = -6.0, logvar_max: float = 2.0):
        super().__init__()
        hidden = int(hidden_dim or max(int(d_model) * 2, 64))
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.net = nn.Sequential(nn.LayerNorm(int(d_model)), nn.Linear(int(d_model), hidden), nn.GELU())
        self.mu = nn.Linear(hidden, int(d_model))
        self.logvar = nn.Linear(hidden, int(d_model))

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(context)
        return self.mu(hidden), self.logvar(hidden).clamp(self.logvar_min, self.logvar_max)


class ReliabilityGatedCrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int, beta: float = 1.0, eps: float = 1e-6, dropout: float = 0.0):
        super().__init__()
        self.beta = float(beta)
        self.eps = float(eps)
        self.query = nn.Parameter(torch.zeros(int(d_model)))
        self.key = nn.Linear(int(d_model), int(d_model))
        self.value = nn.Linear(int(d_model), int(d_model))
        self.out = nn.Sequential(nn.LayerNorm(int(d_model)), nn.Dropout(float(dropout)), nn.Linear(int(d_model), int(d_model)))

    def forward(
        self,
        tokens: torch.Tensor,
        mu_token: torch.Tensor,
        reliability: torch.Tensor,
        global_reliability: torch.Tensor,
    ) -> torch.Tensor:
        all_tokens = torch.cat([tokens, mu_token.unsqueeze(1)], dim=1)
        all_reliability = torch.cat([reliability.squeeze(-1), global_reliability.view(-1, 1)], dim=1)
        keys = self.key(all_tokens)
        scores = (keys * self.query.view(1, 1, -1)).sum(dim=-1) / (keys.shape[-1] ** 0.5)
        scores = scores + self.beta * all_reliability.clamp_min(self.eps).log()
        scores = scores.masked_fill(all_reliability <= 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        fused = (weights.unsqueeze(-1) * self.value(all_tokens)).sum(dim=1)
        return self.out(fused)


class BeamPredictionHead(nn.Module):
    def __init__(self, d_model: int, num_beams: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(int(d_model)), nn.Dropout(float(dropout)), nn.Linear(int(d_model), int(num_beams)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@MODELS.register("u_mask_beam_jepa")
class UMaskBeamJEPA(nn.Module):
    supports_modality_kwargs = True
    supports_force_modality_mask = True

    def __init__(
        self,
        *,
        modalities: list[str] | tuple[str, ...] | None = None,
        d_model: int = 64,
        num_classes: int = 64,
        num_pred: int = 1,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        image_channels: int = 3,
        radar_channels: int = 2,
        lidar_channels: int = 3,
        gps_input_size: int = 3,
        fusion_type: str = "reliability_gated_cross_attention",
        context_type: str = "set_transformer_simplified",
        use_teacher: bool = True,
        use_jepa_loss: bool = True,
        use_modality_uncertainty: bool = True,
        use_global_uncertainty: bool = True,
        logvar_min: float = -6.0,
        logvar_max: float = 2.0,
        eval_missing_pattern: dict[str, Any] | None = None,
        beta: float = 1.0,
        eps: float = 1e-6,
        use_beam_prototype_alignment: bool = False,
        beam_proto_temperature: float = 0.2,
        tau_proto: float | None = None,
        use_full_to_partial_kd: bool = False,
        kd_teacher_mode: str = "disabled",
        mask_sampler: str | None = None,
        ablation_id: str | None = None,
        encoders: dict[str, Any] | None = None,
        encoder_checkpoint_paths: dict[str, str] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.modalities = _validate_modalities(DEFAULT_MODALITIES if modalities is None else modalities)
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.num_pred = int(num_pred)
        self.fusion_type = str(fusion_type)
        self.context_type = str(context_type)
        self.use_teacher = bool(use_teacher)
        self.use_jepa_loss = bool(use_jepa_loss)
        self.use_modality_uncertainty = bool(use_modality_uncertainty)
        self.use_global_uncertainty = bool(use_global_uncertainty)
        self.use_beam_prototype_alignment = bool(use_beam_prototype_alignment)
        self.use_full_to_partial_kd = bool(use_full_to_partial_kd)
        self.kd_teacher_mode = str(kd_teacher_mode)
        self.mask_sampler = mask_sampler
        self.ablation_id = ablation_id
        self.eval_missing_pattern = dict(eval_missing_pattern or {})
        _validate_context_type(self.context_type)
        if self.fusion_type not in {
            "reliability_gated_cross_attention",
            "reliability_biased_missing_attention",
            "concat_mlp",
            "weighted_sum",
        }:
            raise ValueError(
                "fusion_type must be reliability_gated_cross_attention, reliability_biased_missing_attention, "
                "concat_mlp, or weighted_sum."
            )
        if self.d_model <= 0 or self.num_classes <= 0 or self.num_pred <= 0:
            raise ValueError("d_model, num_classes, and num_pred must be positive.")

        self.encoder_configs = {name: dict((encoders or {}).get(name, {})) for name in self.modalities}
        missing_encoder_cfg = [name for name in self.modalities if not self.encoder_configs[name]]
        if missing_encoder_cfg:
            raise ValueError(f"u_mask_beam_jepa requires model.primary.encoders for modalities {missing_encoder_cfg}.")
        self.use_registry_encoders = True
        self.encoders = nn.ModuleDict()
        self.encoder_projections = nn.ModuleDict()
        for name in self.modalities:
            cfg = dict(self.encoder_configs[name])
            cfg.setdefault("output_dim", self.d_model)
            if name == "image":
                cfg.setdefault("image_channels", image_channels)
            elif name == "radar":
                cfg.setdefault("radar_channels", radar_channels)
            elif name == "lidar":
                cfg.setdefault("lidar_channels", lidar_channels)
            elif name == "gps":
                cfg.setdefault("gps_input_size", gps_input_size)
            encoder = ENCODERS.build(cfg)
            self.encoders[name] = encoder
            output_dim = int(getattr(encoder, "output_dim", cfg.get("output_dim", self.d_model)))
            self.encoder_projections[name] = (
                nn.Identity() if output_dim == self.d_model else nn.Linear(output_dim, self.d_model)
            )
            self.encoder_configs[name] = cfg
        self.encoder_checkpoint_loads = self._load_encoder_checkpoints(encoder_checkpoint_paths or {})
        self.modality_embedding = nn.Parameter(torch.zeros(len(self.modalities), self.d_model))
        self.reliability_heads = nn.ModuleDict(
            {name: ModalityReliabilityHead(self.d_model) for name in self.modalities}
        )
        self.teacher = FullModalTeacher(self.d_model, self.num_classes, num_heads, num_layers, dropout)
        self.context_encoder = SetContextEncoder(self.d_model, num_heads, num_layers, dropout)
        self.predictor = GaussianJEPAPredictor(self.d_model, logvar_min=logvar_min, logvar_max=logvar_max)
        self.cross_attention_fusion = ReliabilityGatedCrossAttentionFusion(self.d_model, beta=beta, eps=eps, dropout=dropout)
        self.rbma_fusion = ReliabilityBiasedMissingAwareAttention(
            self.d_model,
            len(self.modalities),
            num_heads=num_heads,
            beta_reliability=beta,
            eps=eps,
            dropout=dropout,
        )
        self.concat_fusion = nn.Sequential(nn.LayerNorm(self.d_model * 2), nn.Linear(self.d_model * 2, self.d_model), nn.GELU())
        self.beam_head = BeamPredictionHead(self.d_model, self.num_classes, dropout=dropout)
        self.prototype_bank = BeamPrototypeBank(
            self.d_model,
            self.num_classes,
            temperature=beam_proto_temperature if tau_proto is None else float(tau_proto),
        )

    def forward(
        self,
        *,
        image_batch: torch.Tensor | None = None,
        radar_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        missing_mask: torch.Tensor | None = None,
        force_modality_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        inputs = {"image": image_batch, "radar": radar_batch, "lidar": lidar_batch, "gps": gps_batch}
        latent = torch.stack([self._encode(name, inputs[name]) for name in self.modalities], dim=1)
        mask = self._resolve_mask(
            missing_mask if missing_mask is not None else force_modality_mask,
            latent,
            allow_all_missing=self.fusion_type in {"reliability_biased_missing_attention", "weighted_sum"},
        )
        reliability, modality_mu_b, modality_logvar_b = self._modality_reliability(latent, mask)
        u_star, teacher_logits = self.teacher(latent, self.modality_embedding)
        c_a = self.context_encoder(latent, mask, reliability, self.modality_embedding)
        mu_b, logvar_b = self.predictor(c_a)
        global_reliability = (
            torch.exp(-F.softplus(logvar_b).mean(dim=-1)) if self.use_global_uncertainty else torch.ones_like(mu_b[:, 0])
        )
        fused, fusion_diagnostics = self._fuse(latent, mask, reliability, mu_b, global_reliability)
        logits = self.beam_head(fused).unsqueeze(1).expand(-1, self.num_pred, -1)
        teacher_logits = teacher_logits.unsqueeze(1).expand(-1, self.num_pred, -1)
        return {
            "logits": logits,
            "input_features": latent,
            "output_features": fused,
            "teacher_logits": teacher_logits,
            "u_star": u_star.detach(),
            "mu_B": mu_b,
            "logvar_B": logvar_b,
            "modality_mu_B": modality_mu_b,
            "modality_logvar_B": modality_logvar_b,
            "modality_reliability": reliability,
            "global_reliability": global_reliability,
            "missing_mask": mask,
            "modality_features": latent,
            "student_feature": fused,
            **fusion_diagnostics,
            "metadata": self.training_strategy_metadata(),
        }

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "u_mask_beam_jepa",
            "architecture_category": "whole_model_exception",
            "enabled_modalities": list(self.modalities),
            "modalities": list(self.modalities),
            "consumes_missing_mask": True,
            "consumes_missing_modality_metadata": False,
            "consumes_reliability_metadata": self.fusion_type == "reliability_biased_missing_attention",
            "reliability_metadata_consumption": "internal_modality_uncertainty"
            if self.fusion_type == "reliability_biased_missing_attention"
            else "none",
            "same_model_full_modal_teacher_auxiliary": True,
            "use_teacher": self.use_teacher,
            "use_jepa_loss": self.use_jepa_loss,
            "use_beam_prototype_alignment": self.use_beam_prototype_alignment,
            "use_full_to_partial_kd": self.use_full_to_partial_kd,
            "kd_teacher_mode": self.kd_teacher_mode,
            "teacher_checkpoint_pending_reason": "checkpoint teacher is not implemented"
            if self.kd_teacher_mode == "checkpoint"
            else None,
            "mask_sampler": self.mask_sampler,
            "ablation_id": self.ablation_id,
            "use_modality_uncertainty": self.use_modality_uncertainty,
            "use_global_uncertainty": self.use_global_uncertainty,
            "fusion_type": self.fusion_type,
            "context_type": self.context_type,
            "use_registry_encoders": self.use_registry_encoders,
            "encoder_configs": self.encoder_configs,
            "encoder_checkpoint_loads": self.encoder_checkpoint_loads,
        }

    def _encode(self, modality: str, value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            raise ValueError(f"u_mask_beam_jepa requires {modality}_batch for enabled modalities {list(self.modalities)}.")
        features = self.encoders[modality](value)
        if features.ndim == 3:
            features = features.mean(dim=1)
        elif features.ndim != 2:
            raise ValueError(f"{modality} encoder must return [B,T,D] or [B,D], got {tuple(features.shape)}.")
        return self.encoder_projections[modality](features)

    def _load_encoder_checkpoints(self, paths: dict[str, str]) -> dict[str, Any]:
        if not paths:
            return {}
        loads: dict[str, Any] = {}
        for modality, raw_path in paths.items():
            if modality not in self.encoders:
                raise ValueError(f"encoder checkpoint configured for disabled modality '{modality}'.")
            checkpoint_path = Path(str(raw_path))
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if not isinstance(state_dict, dict):
                raise ValueError(f"Encoder checkpoint {checkpoint_path} does not contain a state dict.")
            prefix = f"encoders.{modality}."
            target = self.encoders[modality].state_dict()
            matched = {}
            skipped = []
            for key, tensor in state_dict.items():
                if not key.startswith(prefix):
                    continue
                local_key = key[len(prefix):]
                if local_key in target and tuple(target[local_key].shape) == tuple(tensor.shape):
                    matched[local_key] = tensor
                else:
                    skipped.append(local_key)
            if not matched:
                raise ValueError(f"No matching encoder weights for '{modality}' in {checkpoint_path}.")
            incompatible = self.encoders[modality].load_state_dict(matched, strict=False)
            loads[modality] = {
                "path": str(checkpoint_path),
                "loaded_keys": len(matched),
                "missing_keys": sorted(incompatible.missing_keys),
                "unexpected_keys": sorted(incompatible.unexpected_keys),
                "skipped_keys": sorted(skipped),
            }
        return loads

    def _resolve_mask(
        self,
        missing_mask: torch.Tensor | None,
        latent: torch.Tensor,
        *,
        allow_all_missing: bool = False,
    ) -> torch.Tensor:
        if missing_mask is None and (not self.training) and self.eval_missing_pattern:
            from kd_sensing.data.missing_mask import make_pattern_mask

            missing_mask = make_pattern_mask(
                int(latent.shape[0]),
                self.modalities,
                available_modalities=self.eval_missing_pattern.get("available_modalities"),
                pattern_mask=self.eval_missing_pattern.get("pattern_mask"),
                device=latent.device,
            )
        if missing_mask is None:
            mask = torch.ones(latent.shape[:2], dtype=torch.bool, device=latent.device)
        else:
            mask = missing_mask.to(device=latent.device, dtype=torch.bool)
        expected = (int(latent.shape[0]), len(self.modalities))
        if tuple(mask.shape) != expected:
            raise ValueError(f"missing_mask must have shape {expected}, got {tuple(mask.shape)}.")
        empty = (~mask.any(dim=1)).nonzero(as_tuple=False).flatten()
        if int(empty.numel()) > 0 and not allow_all_missing:
            raise ValueError(f"missing_mask has no available modalities for sample indices {empty.detach().cpu().tolist()}.")
        return mask

    def _modality_reliability(self, latent: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = [self.reliability_heads[name](latent[:, index, :]) for index, name in enumerate(self.modalities)]
        modality_mu_b = torch.stack([item[0] for item in outputs], dim=1)
        modality_logvar_b = torch.stack([item[1] for item in outputs], dim=1)
        mask_values = mask.unsqueeze(-1).to(dtype=latent.dtype)
        if not self.use_modality_uncertainty:
            return mask_values, modality_mu_b, modality_logvar_b
        reliability = torch.exp(-F.softplus(modality_logvar_b).mean(dim=-1, keepdim=True))
        return reliability * mask_values, modality_mu_b, modality_logvar_b

    def _fuse(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
        reliability: torch.Tensor,
        mu_b: torch.Tensor,
        global_reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.fusion_type == "reliability_biased_missing_attention":
            result = self.rbma_fusion(
                latent,
                mask,
                reliability,
                global_token=mu_b if self.use_jepa_loss else None,
                global_reliability=global_reliability if self.use_jepa_loss else None,
            )
            diagnostics = dict(result["diagnostics"])
            diagnostics["rbma_mask_provenance"] = "missing_mask"
            diagnostics["rbma_modality_reliability_mean"] = reliability.detach().mean(dim=(1, 2))
            return result["fused"], diagnostics
        if self.fusion_type == "reliability_gated_cross_attention":
            return self.cross_attention_fusion(latent, mu_b, reliability, global_reliability), {}
        weights = reliability.squeeze(-1)
        weight_sum = weights.sum(dim=1, keepdim=True)
        weights = weights / weight_sum.clamp_min(1e-6)
        pooled = (latent * weights.unsqueeze(-1)).sum(dim=1)
        pooled = torch.where(weight_sum.gt(0), pooled, mu_b)
        if self.fusion_type == "weighted_sum":
            return 0.5 * (pooled + mu_b), {}
        return self.concat_fusion(torch.cat([pooled, mu_b], dim=-1)), {}


def _validate_modalities(modalities: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(str(item) for item in modalities)
    if not values:
        raise ValueError(f"u_mask_beam_jepa modalities must be non-empty. Available canonical modalities: {list(DEFAULT_MODALITIES)}.")
    duplicates = sorted({item for item in values if values.count(item) > 1})
    invalid = [item for item in values if item not in DEFAULT_MODALITIES]
    if duplicates or invalid:
        raise ValueError(
            "Invalid u_mask_beam_jepa modalities. "
            f"duplicates={duplicates}, invalid={invalid}, available canonical modalities={list(DEFAULT_MODALITIES)}, "
            f"project canonical order={list(MODALITY_ORDER)}."
        )
    return values


def _validate_context_type(value: str) -> None:
    if value in {"set_transformer_simplified", "beam_query_transformer"}:
        return
    if value == "mask_transformer":
        raise ValueError("context_type='mask_transformer' is not implemented for u_mask_beam_jepa.")
    raise ValueError("context_type must be set_transformer_simplified or beam_query_transformer.")
