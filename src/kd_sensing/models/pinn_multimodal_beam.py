
from typing import Any

import torch
import torch.nn as nn

from kd_sensing.models.physics.beam_scoring import beam_logits_from_channel
from kd_sensing.models.physics.channel_synthesizer import synthesize_ula_channel
from kd_sensing.registries import ENCODERS, MODELS


@MODELS.register("pinn_multimodal_beam")
class PINNMultimodalBeamModel(nn.Module):
    def __init__(
        self,
        *,
        modalities: list[str] | tuple[str, ...] | None = None,
        hidden_dim: int = 64,
        num_classes: int = 64,
        num_pred: int = 3,
        num_paths: int = 3,
        num_subcarriers: int = 32,
        num_antennas: int = 16,
        carrier_frequency_hz: float = 60e9,
        antenna_spacing_ratio: float = 0.5,
        use_direct_head: bool = True,
        use_physics_head: bool = True,
        physics_beta: float = 0.5,
        frontend: dict[str, Any] | str | None = None,
        encoders: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.supports_modality_kwargs = True
        self.modalities = tuple(modalities or ("image", "csi"))
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.num_pred = int(num_pred)
        self.num_paths = int(num_paths)
        self.num_subcarriers = int(num_subcarriers)
        self.num_antennas = int(num_antennas)
        self.carrier_frequency_hz = float(carrier_frequency_hz)
        self.antenna_spacing_ratio = float(antenna_spacing_ratio)
        self.use_direct_head = bool(use_direct_head)
        self.use_physics_head = bool(use_physics_head)
        self.physics_beta = float(physics_beta)
        self.frontend_config = _normalize_frontend_config(frontend)
        self.frontend_type = str(self.frontend_config.get("type", "stats"))
        self.formal_experiment_eligible = bool(self.frontend_config.get("formal_experiment_eligible", True))
        self.local_token_count = int(self.frontend_config.get("local_token_count", 1))
        if self.frontend_type == "paper_modal_tokenizers":
            tokenizer_cfgs = dict(encoders or self.frontend_config.get("encoders") or {})
            image_cfg = tokenizer_cfgs.get("image", _default_tokenizer_config("image", self.hidden_dim))
            if "image" in self.modalities and not _has_checkpoint(image_cfg) and self.formal_experiment_eligible:
                raise ValueError(
                    "formal paper_modal_tokenizers image runs require a jepa_context_image checkpoint; "
                    "set frontend.formal_experiment_eligible=false for debug/smoke."
                )
            self.encoders = nn.ModuleDict(
                {
                    name: _ModalityTokenizer(name, self.hidden_dim, tokenizer_cfgs.get(name))
                    for name in self.modalities
                }
            )
            self.modality_embeddings = nn.ParameterDict(
                {name: nn.Parameter(torch.zeros(self.hidden_dim)) for name in self.modalities}
            )
            self.time_embedding = nn.Embedding(int(self.frontend_config.get("max_time_steps", 64)), self.hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(self.frontend_config.get("num_heads", 4)),
                dim_feedforward=int(self.frontend_config.get("ffn_dim", self.hidden_dim * 4)),
                dropout=float(self.frontend_config.get("dropout", 0.0)),
                batch_first=True,
                norm_first=True,
            )
            self.fusion = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(self.frontend_config.get("num_layers", 1)),
            )
            self.horizon_adapter = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.encoders = nn.ModuleDict({name: _SmallSequenceEncoder(self.hidden_dim) for name in self.modalities})
            self.fusion = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_dim),
            )
            self.modality_embeddings = nn.ParameterDict()
            self.time_embedding = None
            self.horizon_adapter = None
        self.path_head = nn.Linear(self.hidden_dim, self.num_paths * 5)
        self.direct_head = nn.Linear(self.hidden_dim, self.num_classes)
        self.physics_logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        image_batch: torch.Tensor | None = None,
        radar_batch: torch.Tensor | None = None,
        gps_batch: torch.Tensor | None = None,
        lidar_batch: torch.Tensor | None = None,
        mmwave_batch: torch.Tensor | None = None,
        csi_batch: torch.Tensor | None = None,
        csi_input: torch.Tensor | None = None,
        csi_observation_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if kwargs.get("csi_target") is not None:
            raise ValueError("pinn_multimodal_beam refuses csi_target as model input; pass restricted csi_input instead.")
        raw = {
            "image": image_batch,
            "radar": radar_batch,
            "gps": gps_batch,
            "lidar": lidar_batch,
            "mmwave": mmwave_batch,
            "csi": csi_input if csi_input is not None else csi_batch,
        }
        if self.frontend_type == "paper_modal_tokenizers":
            latent, token_metadata = self._paper_frontend(raw)
        else:
            encoded = [self.encoders[name](raw[name]) for name in self.modalities if raw.get(name) is not None]
            if not encoded:
                raise ValueError(f"pinn_multimodal_beam requires one enabled modality from {list(self.modalities)}.")
            horizon = min(tensor.shape[1] for tensor in encoded)
            latent = torch.stack([tensor[:, -horizon:, :] for tensor in encoded], dim=0).mean(dim=0)
            latent = self.fusion(latent)
            token_metadata = {"frontend_type": self.frontend_type}
        if latent.shape[1] < self.num_pred:
            pad = latent[:, -1:, :].expand(latent.shape[0], self.num_pred - latent.shape[1], latent.shape[2])
            latent = torch.cat([latent, pad], dim=1)
        latent = latent[:, -self.num_pred :, :]
        direct_logits = self.direct_head(latent)
        path_hat = self.path_head(latent).view(latent.shape[0], latent.shape[1], self.num_paths, 5)
        path_hat = _normalize_path_hat(path_hat)
        h_hat = synthesize_ula_channel(
            path_hat,
            num_subcarriers=self.num_subcarriers,
            num_antennas=self.num_antennas,
            carrier_frequency_hz=self.carrier_frequency_hz,
            spacing_ratio=self.antenna_spacing_ratio,
        )
        physics_logits, scoring = beam_logits_from_channel(h_hat, num_beams=self.num_classes)
        physics_logits = physics_logits * self.physics_logit_scale
        if self.use_direct_head and self.use_physics_head:
            beta = max(0.0, min(1.0, self.physics_beta))
            logits = (1.0 - beta) * direct_logits + beta * physics_logits
        elif self.use_physics_head:
            logits = physics_logits
        else:
            logits = direct_logits
        return {
            "logits": logits,
            "output_features": latent,
            "direct_logits": direct_logits,
            "physics_logits": physics_logits,
            "h_hat": h_hat,
            "path_hat": path_hat,
            "latent": latent,
            "shape_metadata": {
                "num_subcarriers": self.num_subcarriers,
                "num_antennas": self.num_antennas,
                "num_paths": self.num_paths,
                "num_beams": self.num_classes,
                "channel_target_scope": _channel_target_scope(self.num_subcarriers),
                "csi_observation_mask": (
                    tuple(csi_observation_mask.shape) if torch.is_tensor(csi_observation_mask) else None
                ),
                **token_metadata,
            },
            **scoring,
        }

    def _paper_frontend(self, raw: dict[str, torch.Tensor | None]) -> tuple[torch.Tensor, dict[str, Any]]:
        tokens_by_modality: list[torch.Tensor] = []
        metadata: dict[str, Any] = {"frontend_type": self.frontend_type, "tokenizers": {}}
        for name in self.modalities:
            value = raw.get(name)
            if value is None:
                continue
            tokens = self.encoders[name](value)
            if tokens.ndim == 3:
                tokens = tokens.unsqueeze(2)
            if tokens.ndim != 4:
                raise ValueError(f"{name} tokenizer must return [B, T, D] or [B, T, K, D], got {tuple(tokens.shape)}.")
            batch, steps, local_tokens, dim = tokens.shape
            if dim != self.hidden_dim:
                raise ValueError(f"{name} tokenizer hidden dim must be {self.hidden_dim}, got {dim}.")
            if self.time_embedding is None or self.horizon_adapter is None:
                raise RuntimeError("paper_modal_tokenizers frontend is missing token fusion modules.")
            time_ids = torch.arange(steps, device=tokens.device).clamp_max(self.time_embedding.num_embeddings - 1)
            tokens = tokens + self.modality_embeddings[name].view(1, 1, 1, -1)
            tokens = tokens + self.time_embedding(time_ids).view(1, steps, 1, -1)
            tokens_by_modality.append(tokens.reshape(batch, steps * local_tokens, dim))
            metadata["tokenizers"][name] = self.encoders[name].metadata()
        if not tokens_by_modality:
            raise ValueError(f"pinn_multimodal_beam requires one enabled modality from {list(self.modalities)}.")
        fused = self.fusion(torch.cat(tokens_by_modality, dim=1))
        pooled = fused.mean(dim=1, keepdim=True)
        latent = self.horizon_adapter(pooled).expand(fused.shape[0], self.num_pred, self.hidden_dim)
        metadata["shared_transformer_layers"] = int(getattr(self.fusion, "num_layers", 0))
        metadata["hidden_dim"] = self.hidden_dim
        return latent, metadata

    def training_strategy_metadata(self) -> dict[str, Any]:
        used_csi = "csi" in self.modalities
        tokenizers = {
            name: encoder.metadata()
            for name, encoder in self.encoders.items()
            if hasattr(encoder, "metadata")
        }
        return {
            "registry_name": "pinn_multimodal_beam",
            "architecture_category": "whole_model_exception",
            "exception_reason": "path head, differentiable channel synthesis, physics logits and diagnostics are coupled inside forward",
            "enabled_modalities": list(self.modalities),
            "frontend_type": self.frontend_type,
            "tokenizers": tokenizers,
            "shared_transformer_layers": (
                int(getattr(self.fusion, "num_layers", 0)) if self.frontend_type == "paper_modal_tokenizers" else 0
            ),
            "hidden_dim": self.hidden_dim,
            "physics_branch": bool(self.use_physics_head),
            "array_type": "ula",
            "codebook_source": "ula_dft_fallback",
            "loss_weights": {},
            "used_csi_as_input": used_csi,
            "restricted_wireless_input": used_csi,
            "oracle_upper_bound": False,
            "channel_target_scope": _channel_target_scope(self.num_subcarriers),
            "formal_experiment_eligible": self.formal_experiment_eligible,
            "used_path_label_for_training": False,
            "used_beam_power_for_training": False,
            "used_reliability_metadata": False,
            "main_conclusion_eligible": not used_csi,
        }


class _SmallSequenceEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 2:
            value = value.unsqueeze(1)
        if value.ndim < 3:
            raise ValueError(f"modality tensor must have shape [B, T, ...], got {tuple(value.shape)}.")
        flat = value.to(torch.float32).flatten(start_dim=2)
        stats = torch.stack(
            (
                flat.mean(dim=-1),
                flat.std(dim=-1, unbiased=False),
                flat.amax(dim=-1),
                flat.amin(dim=-1),
            ),
            dim=-1,
        )
        return self.net(stats)


class _ModalityTokenizer(nn.Module):
    def __init__(self, name: str, hidden_dim: int, cfg: Any = None) -> None:
        super().__init__()
        self.name = name
        self.hidden_dim = int(hidden_dim)
        cfg = _default_tokenizer_config(name, self.hidden_dim) if cfg is None else cfg
        if isinstance(cfg, str):
            cfg = {"type": cfg}
        self.cfg = dict(cfg)
        self.encoder = ENCODERS.build({**self.cfg, "output_dim": self.hidden_dim})
        self.type = str(self.cfg.get("type", "linear"))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.encoder(value)

    def metadata(self) -> dict[str, Any]:
        encoder_meta = (
            self.encoder.training_strategy_metadata() if hasattr(self.encoder, "training_strategy_metadata") else {}
        )
        return {
            "type": self.type,
            "checkpoint_path": getattr(self.encoder, "checkpoint_path", self.cfg.get("checkpoint_path", "")),
            "freeze_policy": bool(getattr(self.encoder, "freeze_encoder", self.cfg.get("freeze_encoder", False))),
            "uses_gps_context": bool(getattr(self.encoder, "required_context_modalities", ())),
            **(encoder_meta if isinstance(encoder_meta, dict) else {}),
        }


@ENCODERS.register("linear_sequence_tokenizer")
class LinearSequenceTokenizer(nn.Module):
    def __init__(self, output_dim: int = 64, input_dim: int | None = None, **_: Any) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.input_dim = None if input_dim is None else int(input_dim)
        self.proj = (
            nn.LazyLinear(self.output_dim) if self.input_dim is None else nn.Linear(self.input_dim, self.output_dim)
        )
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 2:
            value = value.unsqueeze(1)
        if value.ndim < 3:
            raise ValueError(f"tokenizer input must have shape [B, T, ...], got {tuple(value.shape)}.")
        return self.norm(self.proj(value.to(torch.float32).flatten(start_dim=2)))


def _normalize_frontend_config(frontend: dict[str, Any] | str | None) -> dict[str, Any]:
    if frontend is None:
        return {"type": "stats"}
    if isinstance(frontend, str):
        return {"type": frontend}
    return dict(frontend)


def _default_tokenizer_config(name: str, hidden_dim: int) -> dict[str, Any]:
    if name == "image":
        return {"type": "jepa_context_image", "latent_dim": hidden_dim, "pooling": "mean"}
    if name == "csi":
        return {"type": "pilot_dual_view_csi", "view_fusion": "freq_only", "use_internal_gru": False}
    if name == "radar":
        return {"type": "radar_cnn"}
    if name == "lidar":
        return {"type": "lidar_cnn"}
    if name == "gps":
        return {"type": "gps_mlp"}
    return {"type": "linear_sequence_tokenizer"}


def _has_checkpoint(cfg: Any) -> bool:
    if isinstance(cfg, str):
        return False
    return isinstance(cfg, dict) and bool(cfg.get("checkpoint_path") or cfg.get("checkpoint"))


def _channel_target_scope(num_subcarriers: int) -> str:
    return "narrowband_array_channel" if int(num_subcarriers) == 1 else "array_channel"


def _normalize_path_hat(path_hat: torch.Tensor) -> torch.Tensor:
    angles = torch.pi * torch.tanh(path_hat[..., 0:2])
    delay = torch.nn.functional.softplus(path_hat[..., 2:3]) * 1e-9
    gain = path_hat[..., 3:5]
    return torch.cat((angles, delay, gain), dim=-1)
