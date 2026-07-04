
from collections import defaultdict
from typing import Any, Iterable, Mapping

import torch

from kd_sensing.models.physics.complex_utils import abs_square, ri_to_complex


def csi_nmse(pred: torch.Tensor, target_ri: torch.Tensor) -> torch.Tensor:
    target = ri_to_complex(target_ri.to(pred.device)) if not torch.is_complex(target_ri) else target_ri.to(pred.device)
    return abs_square(pred - target).mean() / abs_square(target).mean().clamp_min(1e-12)


def path_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred[..., :5] - target.to(pred.device, dtype=pred.dtype)[..., :5]).abs().mean()


def gain_nmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = pred[..., 3].square() + pred[..., 4].square()
    t = target.to(pred.device, dtype=pred.dtype)
    truth = t[..., 3].square() + t[..., 4].square()
    return (p - truth).square().mean() / truth.square().mean().clamp_min(1e-12)


def normalized_beamforming_gain(predicted_beam: torch.Tensor, beam_power: torch.Tensor) -> torch.Tensor:
    power = beam_power.to(predicted_beam.device, dtype=torch.float32)
    chosen = power.gather(-1, predicted_beam.to(torch.long).unsqueeze(-1)).squeeze(-1)
    best = power.max(dim=-1).values.clamp_min(1e-12)
    return chosen / best


def grouped_report(rows: Iterable[Mapping[str, Any]], group_keys: tuple[str, ...] = ("condition", "town", "scene")) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        metric = row.get("value")
        if metric is None:
            continue
        key = "|".join(str(row.get(item, "unknown") or "unknown") for item in group_keys)
        buckets[key].append(float(metric))
    return {key: {"mean": sum(values) / len(values), "count": float(len(values))} for key, values in buckets.items()}
