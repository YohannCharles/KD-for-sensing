from __future__ import annotations

import torch
from torch import nn


class SparsePilotEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_candidate_patterns: int = 32,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        quality_dim: int = 16,
        include_index_embeddings: bool = False,
        num_frequency_indices: int = 16,
        maximum_time_steps: int = 5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.quality_dim = int(quality_dim)
        self.include_index_embeddings = bool(include_index_embeddings)
        if int(num_layers) < 0:
            raise ValueError("num_layers must be non-negative.")
        if self.include_index_embeddings and (int(num_frequency_indices) <= 0 or int(maximum_time_steps) <= 0):
            raise ValueError("Index-embedding frequency and time dimensions must be positive.")
        self.token_projection = nn.Linear(5, self.hidden_dim)
        self.pattern_embedding = nn.Embedding(int(num_candidate_patterns), self.hidden_dim)
        self.frequency_embedding: nn.Embedding | None = None
        self.time_embedding: nn.Embedding | None = None
        self.validity_embedding: nn.Embedding | None = None
        if self.include_index_embeddings:
            self.frequency_embedding = nn.Embedding(int(num_frequency_indices), self.hidden_dim)
            self.time_embedding = nn.Embedding(int(maximum_time_steps), self.hidden_dim)
            self.validity_embedding = nn.Embedding(2, self.hidden_dim)
        self.encoder = None
        if int(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(num_heads),
                dim_feedforward=self.hidden_dim * 2,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=int(num_layers),
                enable_nested_tensor=False,
            )
        self.pool_score = nn.Linear(self.hidden_dim, 1)
        self.quality_projection = nn.Sequential(
            nn.Linear(3, self.quality_dim),
            nn.GELU(),
            nn.Linear(self.quality_dim, self.quality_dim),
        )

    def forward(
        self,
        pilot_observations: torch.Tensor,
        pattern_ids: torch.Tensor,
        frequency_positions: torch.Tensor,
        pilot_mask: torch.Tensor,
        snr_db: torch.Tensor | float | None = None,
        *,
        frequency_ids: torch.Tensor | None = None,
        time_ids: torch.Tensor | int | None = None,
    ) -> dict[str, torch.Tensor]:
        values = torch.as_tensor(pilot_observations)
        if not torch.is_complex(values) or values.ndim != 3:
            raise ValueError("pilot_observations must be complex [B,M,K].")
        batch, patterns, frequencies = values.shape
        ids = torch.as_tensor(pattern_ids, device=values.device, dtype=torch.long)
        valid = torch.as_tensor(pilot_mask, device=values.device, dtype=torch.bool)
        if tuple(ids.shape) != (batch, patterns) or tuple(valid.shape) != tuple(values.shape):
            raise ValueError("pattern_ids and pilot_mask must have shapes [B,M] and [B,M,K].")
        frequency = torch.as_tensor(frequency_positions, device=values.device, dtype=values.real.dtype)
        if frequency.ndim == 1:
            frequency = frequency[None, None, :].expand(batch, patterns, -1)
        elif frequency.ndim == 2:
            frequency = frequency[:, None, :].expand(-1, patterns, -1)
        if tuple(frequency.shape) != (batch, patterns, frequencies):
            raise ValueError("frequency_positions must have shape [K] or [B,K].")

        count = valid.sum(dim=(1, 2)).clamp_min(1).to(values.real.dtype)
        power = (values.abs().square() * valid).sum(dim=(1, 2)) / count
        rms = power.clamp_min(torch.finfo(values.real.dtype).tiny).sqrt()
        normalized = values / rms[:, None, None]
        frequency_scale = frequency.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1.0)
        normalized_frequency = frequency / frequency_scale
        token_features = torch.stack(
            (
                normalized.real,
                normalized.imag,
                torch.log(normalized.abs().clamp_min(1e-8)),
                valid.to(values.real.dtype),
                normalized_frequency,
            ),
            dim=-1,
        ).reshape(batch, patterns * frequencies, 5)
        token_ids = ids[:, :, None].expand(-1, -1, frequencies).reshape(batch, -1)
        flat_valid = valid.reshape(batch, -1)
        safe_valid = flat_valid.clone()
        empty = ~safe_valid.any(dim=1)
        safe_valid[empty, 0] = True
        tokens = self.token_projection(token_features) + self.pattern_embedding(token_ids)
        if self.include_index_embeddings:
            assert self.frequency_embedding is not None
            assert self.time_embedding is not None
            assert self.validity_embedding is not None
            if frequency_ids is None or time_ids is None:
                raise ValueError("frequency_ids and time_ids are required when include_index_embeddings=true.")
            frequency_index = torch.as_tensor(frequency_ids, device=values.device, dtype=torch.long)
            if frequency_index.ndim == 1:
                if tuple(frequency_index.shape) != (frequencies,):
                    raise ValueError("frequency_ids must have shape [K], [B,K], or [B,M,K].")
                frequency_index = frequency_index[None, None, :].expand(batch, patterns, -1)
            elif frequency_index.ndim == 2:
                if tuple(frequency_index.shape) != (batch, frequencies):
                    raise ValueError("Batched frequency_ids must have shape [B,K].")
                frequency_index = frequency_index[:, None, :].expand(-1, patterns, -1)
            elif tuple(frequency_index.shape) != (batch, patterns, frequencies):
                raise ValueError("frequency_ids must have shape [K], [B,K], or [B,M,K].")
            if bool(((frequency_index < 0) | (frequency_index >= self.frequency_embedding.num_embeddings)).any()):
                raise ValueError("frequency_ids contains an out-of-range index.")
            time_index = torch.as_tensor(time_ids, device=values.device, dtype=torch.long).reshape(-1)
            if time_index.numel() == 1:
                time_index = time_index.expand(batch)
            if time_index.numel() != batch or bool(
                ((time_index < 0) | (time_index >= self.time_embedding.num_embeddings)).any()
            ):
                raise ValueError("time_ids must be scalar or [B] within the configured range.")
            tokens = tokens + self.frequency_embedding(frequency_index.reshape(batch, -1))
            tokens = tokens + self.time_embedding(time_index)[:, None]
            tokens = tokens + self.validity_embedding(flat_valid.long())
        encoded = tokens if self.encoder is None else self.encoder(tokens, src_key_padding_mask=~safe_valid)
        scores = self.pool_score(encoded).squeeze(-1).masked_fill(~flat_valid, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(empty[:, None], torch.zeros_like(weights), weights)
        feature = (weights.unsqueeze(-1) * encoded).sum(dim=1)

        if snr_db is None:
            snr = torch.zeros(batch, device=values.device, dtype=values.real.dtype)
            snr_available = torch.zeros(batch, device=values.device, dtype=torch.bool)
        else:
            snr = torch.as_tensor(snr_db, device=values.device, dtype=values.real.dtype).reshape(-1)
            if snr.numel() == 1:
                snr = snr.expand(batch)
            if snr.numel() != batch or not bool(torch.isfinite(snr).all().item()):
                raise ValueError("snr_db must be a finite scalar or shape [B].")
            snr_available = torch.ones(batch, device=values.device, dtype=torch.bool)
        valid_ratio = valid.to(values.real.dtype).mean(dim=(1, 2))
        log_rms = torch.log(rms.clamp_min(1e-8))
        quality_scalars = torch.stack((snr / 30.0, valid_ratio, log_rms), dim=-1)
        learned_quality = self.quality_projection(quality_scalars)
        quality_confidence = torch.where(
            snr_available,
            torch.sigmoid((snr + 5.0) / 2.0) * valid_ratio,
            valid_ratio,
        )
        quality = torch.cat((learned_quality, quality_scalars, quality_confidence[:, None]), dim=-1)
        return {
            "csi_feature": feature,
            "csi_quality": quality,
            "quality_confidence": quality_confidence,
            "valid_ratio": valid_ratio,
            "log_rms": log_rms,
            "csi_available": ~empty,
            "snr_available": snr_available,
        }


__all__ = ["SparsePilotEncoder"]
