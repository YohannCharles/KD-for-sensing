"""Deterministic statistics for a categorical beam posterior."""

from __future__ import annotations

import math

import torch


def beam_posterior_statistics(
    probabilities: torch.Tensor,
    *,
    num_beams: int = 64,
    topology_positions: torch.Tensor | None = None,
    top_l: int = 7,
    row_tolerance: float = 1e-4,
    resultant_epsilon: float = 1e-7,
) -> dict[str, torch.Tensor]:
    """Return stable circular statistics without adding a trainable path.

    ``topology_positions[label]`` is the audited circular coordinate of a beam
    label. For the ULA-DFT phase cycle it is simply ``0..63``. The returned
    tensors are detached because these values are policy/evaluation metadata,
    never optimization targets.
    """
    if not torch.is_tensor(probabilities) or probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [B, C].")
    batch, classes = (int(probabilities.shape[0]), int(probabilities.shape[1]))
    expected_beams = int(num_beams)
    if expected_beams <= 1 or classes != expected_beams:
        raise ValueError(f"probabilities must have exactly {expected_beams} beams.")
    if batch <= 0:
        raise ValueError("probabilities must contain a non-empty batch.")
    limit = int(top_l)
    if limit <= 0 or limit > classes:
        raise ValueError(f"top_l must be in [1, {classes}], got {top_l}.")
    tolerance = float(row_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("row_tolerance must be finite and non-negative.")
    epsilon = float(resultant_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("resultant_epsilon must be finite and positive.")

    values = probabilities.detach().float()
    if not bool(torch.isfinite(values).all().item()) or bool((values < 0).any().item()):
        raise ValueError("probabilities must be finite and non-negative.")
    row_sum = values.sum(dim=-1)
    if not bool(torch.allclose(row_sum, torch.ones_like(row_sum), atol=tolerance, rtol=0.0)):
        raise ValueError("probabilities rows must sum to one.")

    if topology_positions is None:
        positions = torch.arange(classes, device=values.device, dtype=torch.float32)
    else:
        positions = torch.as_tensor(topology_positions, device=values.device, dtype=torch.float32).reshape(-1)
        if int(positions.numel()) != classes:
            raise ValueError("topology_positions must contain one coordinate per beam class.")
    if not bool(torch.isfinite(positions).all().item()):
        raise ValueError("topology_positions must be finite.")
    if bool((positions < 0).any().item()) or bool((positions >= classes).any().item()):
        raise ValueError("topology_positions must lie in [0, num_beams).")
    if int(torch.unique(positions).numel()) != classes:
        raise ValueError("topology_positions must be a permutation of circular coordinates.")

    center = torch.argmax(values, dim=-1)
    center_position = positions[center]
    distance = (positions.unsqueeze(0) - center_position.unsqueeze(-1)).abs()
    distance = torch.minimum(distance, float(classes) - distance)

    phase = positions * (2.0 * math.pi / float(classes))
    cos_resultant = (values * phase.cos().unsqueeze(0)).sum(dim=-1)
    sin_resultant = (values * phase.sin().unsqueeze(0)).sum(dim=-1)
    resultant = torch.sqrt(cos_resultant.square() + sin_resultant.square())
    circular_mean_phase = torch.remainder(torch.atan2(sin_resultant, cos_resultant), 2.0 * math.pi)
    circular_mean_position = circular_mean_phase * (float(classes) / (2.0 * math.pi))
    circular_mean_position = torch.where(
        resultant <= epsilon,
        center_position,
        circular_mean_position,
    )
    mean_distance = (positions.unsqueeze(0) - circular_mean_position.unsqueeze(-1)).abs()
    mean_distance = torch.minimum(mean_distance, float(classes) - mean_distance)
    circular_mean_beam = mean_distance.argmin(dim=-1)

    beam_variance = (values * distance.square()).sum(dim=-1)
    beam_spread = beam_variance.clamp_min(0).sqrt()
    circular_variance = (1.0 - resultant).clamp(0.0, 1.0)
    entropy = -(values.clamp_min(torch.finfo(torch.float32).tiny) * values.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(dim=-1)
    normalized_entropy = (entropy / math.log(float(classes))).clamp(0.0, 1.0)
    top_indices = torch.argsort(values, dim=-1, descending=True, stable=True)[..., :limit]
    top_probability = values.gather(dim=-1, index=top_indices)

    return {
        "beam_map": center.detach(),
        "beam_circular_mean": circular_mean_position.detach(),
        "beam_circular_mean_label": circular_mean_beam.detach(),
        "beam_resultant_length": resultant.detach(),
        "beam_circular_variance": circular_variance.detach(),
        "beam_variance": beam_variance.detach(),
        "beam_spread": beam_spread.detach(),
        "beam_normalized_entropy": normalized_entropy.detach(),
        "beam_top_indices": top_indices.detach(),
        "beam_top_probabilities": top_probability.detach(),
    }


__all__ = ["beam_posterior_statistics"]
