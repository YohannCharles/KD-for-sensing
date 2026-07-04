
from typing import Any, Mapping

import torch
import torch.nn as nn

from kd_sensing.registries import MODELS


_MODALITIES = ("image", "lidar", "gps")


class _ImageEncoder(nn.Module):
    def __init__(self, *, in_channels: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class _LidarEncoder(nn.Module):
    def __init__(self, *, input_features: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.net = nn.Sequential(
            nn.Conv1d(self.input_features, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value.transpose(1, 2).contiguous())


class _GpsEncoder(nn.Module):
    def __init__(self, *, input_features: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_features, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


@MODELS.register("amr_net")
class AMRNet(nn.Module):
    supports_modality_kwargs = True
    supports_force_modality_mask = True

    def __init__(
        self,
        *,
        modalities: list[str] | tuple[str, ...] | None = None,
        num_classes: int = 64,
        num_pred: int = 1,
        image_channels: int = 1,
        image_feature_dim: int = 128,
        lidar_input_features: int = 2,
        lidar_feature_dim: int = 512,
        gps_input_size: int = 2,
        gps_feature_dim: int = 512,
        latent_dim: int = 128,
        dropout: float = 0.1,
        deterministic_inference: bool = True,
        stochastic_eval: bool = False,
        logvar_min: float = -8.0,
        logvar_max: float = 8.0,
        cuaf: Mapping[str, Any] | None = None,
        loss: Mapping[str, Any] | None = None,
        consumes_reliability_metadata: bool = False,
        paper_approximation: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.modalities = tuple(str(item) for item in (modalities or _MODALITIES))
        invalid = sorted(set(self.modalities) - set(_MODALITIES))
        if invalid:
            raise ValueError(f"amr_net supports modalities {list(_MODALITIES)}, got unsupported {invalid}.")
        self.num_classes = int(num_classes)
        self.num_pred = int(num_pred)
        self.latent_dim = int(latent_dim)
        self.deterministic_inference = bool(deterministic_inference)
        self.stochastic_eval = bool(stochastic_eval)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.consumes_reliability_metadata = bool(consumes_reliability_metadata)
        self.paper_approximation = bool(paper_approximation)
        self.cuaf_cfg = {"enabled": True, "eps": 1e-8, **dict(cuaf or {})}
        self.loss_cfg = {
            "alpha": 0.01,
            "beta": 1.0,
            "pre_enabled": True,
            "pre_samples": 2,
            "temperature": 0.1,
            **dict(loss or {}),
        }
        if self.num_classes <= 0 or self.num_pred <= 0 or self.latent_dim <= 0:
            raise ValueError("amr_net num_classes, num_pred and latent_dim must be positive.")
        self.image_channels = int(image_channels)

        encoder_dims = {
            "image": int(image_feature_dim),
            "lidar": int(lidar_feature_dim),
            "gps": int(gps_feature_dim),
        }
        self.encoder_dims = encoder_dims
        encoders: dict[str, nn.Module] = {}
        if "image" in self.modalities:
            encoders["image"] = _ImageEncoder(in_channels=self.image_channels, output_dim=encoder_dims["image"], dropout=float(dropout))
        if "lidar" in self.modalities:
            encoders["lidar"] = _LidarEncoder(
                input_features=int(lidar_input_features),
                output_dim=encoder_dims["lidar"],
                dropout=float(dropout),
            )
        if "gps" in self.modalities:
            encoders["gps"] = _GpsEncoder(input_features=int(gps_input_size), output_dim=encoder_dims["gps"], dropout=float(dropout))
        self.encoders = nn.ModuleDict(encoders)
        self.mu_heads = nn.ModuleDict({name: nn.Linear(encoder_dims[name], self.latent_dim) for name in self.modalities})
        self.logvar_heads = nn.ModuleDict({name: nn.Linear(encoder_dims[name], self.latent_dim) for name in self.modalities})
        self.classifiers = nn.ModuleDict({name: nn.Linear(self.latent_dim, self.num_classes) for name in self.modalities})

    def forward(
        self,
        *,
        image_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        force_modality_mask: torch.Tensor | None = None,
        modality_availability: Mapping[str, torch.Tensor] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        batches = {"image": image_batch, "lidar": lidar_batch, "gps": gps_batch}
        features: dict[str, torch.Tensor] = {}
        mu: dict[str, torch.Tensor] = {}
        logvar: dict[str, torch.Tensor] = {}
        z: dict[str, torch.Tensor] = {}
        modality_logits: dict[str, torch.Tensor] = {}
        for name in self.modalities:
            snapshot = self._snapshot(batches[name], name=name)
            encoded = self.encoders[name](snapshot)
            features[name] = encoded
            mu[name] = self.mu_heads[name](encoded)
            logvar[name] = self.logvar_heads[name](encoded).clamp(self.logvar_min, self.logvar_max)
            z[name] = self._sample(mu[name], logvar[name])
            modality_logits[name] = self.classifiers[name](z[name]).unsqueeze(1).expand(-1, self.num_pred, -1)

        fused = self._cuaf(modality_logits, force_modality_mask=force_modality_mask, modality_availability=modality_availability)
        return {
            "logits": fused["logits"],
            "input_features": torch.cat([features[name] for name in self.modalities], dim=-1),
            "output_features": torch.stack([z[name] for name in self.modalities], dim=1).mean(dim=1),
            "modality_logits": modality_logits,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "features": features,
            "cuaf_weights": fused["weights"],
            "cuaf_entropy": fused["entropy"],
            "cuaf_entropy_score": fused["entropy_score"],
            "cuaf_kl_consistency": fused["kl_consistency"],
            "cuaf_pairwise_kl_score": fused["pairwise_kl_score"],
            "cuaf_topk_margin": fused["topk_margin"],
            "cuaf_top_t_margin_score": fused["top_t_margin_score"],
            "cuaf_criterion_weights": fused["criterion_weights"],
            "cuaf_available": fused["available"],
            "amr": {
                "modality_logits": modality_logits,
                "mu": mu,
                "logvar": logvar,
                "z": z,
                "cuaf": fused,
                "modalities": list(self.modalities),
            },
            "metadata": self.training_strategy_metadata(),
        }

    def _snapshot(self, value: torch.Tensor | None, *, name: str) -> torch.Tensor:
        if value is None:
            raise ValueError(f"amr_net requires {name}_batch for enabled modality '{name}'.")
        expected_rank = 5 if name == "image" else 4 if name == "lidar" else 3
        if value.ndim != expected_rank:
            raise ValueError(f"amr_net expects {name}_batch rank {expected_rank}, got shape {tuple(value.shape)}.")
        if int(value.shape[1]) <= 0:
            raise ValueError(f"amr_net requires {name}_batch to contain at least one time step, got shape {tuple(value.shape)}.")
        snapshot = value.to(dtype=torch.float32).mean(dim=1)
        if name == "image" and self.image_channels == 1 and int(snapshot.shape[1]) == 3:
            weights = torch.tensor((0.2989, 0.5870, 0.1140), dtype=snapshot.dtype, device=snapshot.device).view(1, 3, 1, 1)
            return (snapshot * weights).sum(dim=1, keepdim=True)
        return snapshot

    def _sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training and (self.deterministic_inference and not self.stochastic_eval):
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def _cuaf(
        self,
        modality_logits: Mapping[str, torch.Tensor],
        *,
        force_modality_mask: torch.Tensor | None,
        modality_availability: Mapping[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        names = list(self.modalities)
        logits = torch.stack([modality_logits[name] for name in names], dim=2)
        probs = torch.softmax(logits, dim=-1)
        eps = float(self.cuaf_cfg.get("eps", 1e-8))
        class_count = max(int(logits.shape[-1]), 2)
        available = torch.ones_like(probs[..., 0], dtype=torch.bool)
        if modality_availability:
            for index, name in enumerate(names):
                mask = modality_availability.get(name)
                if torch.is_tensor(mask):
                    available[:, :, index] = _prediction_mask(mask, available[:, :, index], name=name)
        if force_modality_mask is not None:
            available = available & _force_mask_for_cuaf(force_modality_mask, available)

        entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)
        entropy_score = 1.0 / entropy.clamp_min(eps)
        log_probs = probs.clamp_min(eps).log()
        pairwise_kl = (probs.unsqueeze(3) * (log_probs.unsqueeze(3) - log_probs.unsqueeze(2))).sum(dim=-1)
        peer_mask = available.unsqueeze(2) & available.unsqueeze(3)
        eye = torch.eye(len(names), dtype=torch.bool, device=available.device).view(1, 1, len(names), len(names))
        peer_mask = peer_mask & ~eye
        peer_count = peer_mask.to(dtype=probs.dtype).sum(dim=3).clamp_min(1.0)
        avg_kl = pairwise_kl.masked_fill(~peer_mask, 0.0).sum(dim=3) / peer_count
        kl_score = 1.0 / avg_kl.clamp_min(eps)
        top_t = max(int(self.cuaf_cfg.get("top_t", self.cuaf_cfg.get("top_k_margin", 5))), 1)
        top_values = torch.topk(probs, k=min(top_t + 1, int(class_count)), dim=-1).values
        if top_values.shape[-1] > 1:
            margin = top_values[..., 0] - top_values[..., 1:].mean(dim=-1)
        else:
            margin = top_values[..., 0]

        entropy_weight = _masked_modality_softmax(entropy_score, available)
        kl_weight = _masked_modality_softmax(kl_score, available)
        margin_weight = _masked_modality_softmax(margin, available)
        scores = (entropy_weight + kl_weight + margin_weight).masked_fill(~available, 0.0)
        denom = scores.sum(dim=2, keepdim=True)
        fallback = available.to(dtype=scores.dtype) / available.to(dtype=scores.dtype).sum(dim=2, keepdim=True).clamp_min(1.0)
        weights = torch.where(denom > 0.0, scores / denom.clamp_min(eps), fallback)
        fused_prob = (weights.unsqueeze(-1) * probs).sum(dim=2).clamp_min(eps)
        fused_prob = fused_prob / fused_prob.sum(dim=-1, keepdim=True).clamp_min(eps)
        return {
            "logits": fused_prob.log(),
            "probabilities": fused_prob,
            "weights": weights,
            "entropy": entropy,
            "entropy_score": entropy_score,
            "kl_consistency": kl_score,
            "pairwise_kl_score": kl_score,
            "topk_margin": margin,
            "top_t_margin_score": margin,
            "criterion_weights": {
                "entropy": entropy_weight,
                "pairwise_kl": kl_weight,
                "top_t_margin": margin_weight,
            },
            "available": available,
            "finite": torch.isfinite(fused_prob).all(dim=-1) & torch.isfinite(weights).all(dim=-1),
        }

    def training_strategy_metadata(self) -> dict[str, Any]:
        return {
            "type": "amr_net",
            "registry_type": "amr_net",
            "architecture_category": "whole_model_exception",
            "enabled_modalities": list(self.modalities),
            "modalities": list(self.modalities),
            "encoder_dims": dict(self.encoder_dims),
            "image_channels": self.image_channels,
            "latent_dim": self.latent_dim,
            "num_classes": self.num_classes,
            "num_pred": self.num_pred,
            "cuaf_enabled": bool(self.cuaf_cfg.get("enabled", True)),
            "cuaf": dict(self.cuaf_cfg),
            "cuaf_formula": "entropy_pairwise_kl_top_t_margin",
            "supports_variable_input_length": True,
            "temporal_pooling": "mean",
            "loss": dict(self.loss_cfg),
            "deterministic_inference": self.deterministic_inference,
            "consumes_reliability_metadata": self.consumes_reliability_metadata,
            "checkpoint_policy": "standard_state_dict",
            "freeze_policy": "none",
            "paper_approximation": self.paper_approximation,
        }


def _masked_modality_softmax(scores: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    masked = scores.masked_fill(~available, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked, dim=2)
    return weights.masked_fill(~available, 0.0)


def _prediction_mask(mask: torch.Tensor, target: torch.Tensor, *, name: str) -> torch.Tensor:
    value = mask.to(device=target.device, dtype=torch.bool)
    batch_size, pred_steps = target.shape
    if value.ndim == 0 or int(value.shape[0]) != batch_size:
        raise ValueError(
            f"amr_net modality availability for {name} must start with batch size {batch_size}, got shape {tuple(value.shape)}."
        )
    if value.ndim == 1:
        return value.view(batch_size, 1).expand(-1, pred_steps)
    if tuple(value.shape) == tuple(target.shape):
        return value
    return value.reshape(batch_size, -1).any(dim=1, keepdim=True).expand(-1, pred_steps)


def _force_mask_for_cuaf(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    value = mask.to(device=target.device, dtype=torch.bool)
    batch_size, pred_steps, modality_count = target.shape
    if value.ndim == 1:
        if int(value.shape[0]) != modality_count:
            raise ValueError(f"force_modality_mask shape must end with K={modality_count}, got {tuple(value.shape)}.")
        return value.view(1, 1, modality_count).expand(batch_size, pred_steps, -1)
    if value.ndim == 2:
        if tuple(value.shape) == (batch_size, modality_count):
            return value.view(batch_size, 1, modality_count).expand(-1, pred_steps, -1)
        if tuple(value.shape) == (pred_steps, modality_count):
            return value.view(1, pred_steps, modality_count).expand(batch_size, -1, -1)
    if value.ndim >= 3 and int(value.shape[0]) == batch_size and int(value.shape[-1]) >= modality_count:
        trimmed = value[..., :modality_count]
        if trimmed.ndim == 3 and int(trimmed.shape[1]) == pred_steps:
            return trimmed
        return trimmed.reshape(batch_size, -1, modality_count).any(dim=1, keepdim=True).expand(-1, pred_steps, -1)
    raise ValueError(
        f"force_modality_mask shape must be [K], [B,K], [P,K] or [B,T,K], got {tuple(value.shape)}."
    )
