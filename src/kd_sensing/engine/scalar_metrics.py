from typing import Any, Mapping

import torch


MetricTerms = Mapping[str, tuple[torch.Tensor, torch.Tensor]]


def mean_metric_term(value: torch.Tensor, observations: torch.Tensor | int | float) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = torch.as_tensor(observations, device=value.device, dtype=torch.float32).reshape(())
    return value.detach().to(dtype=torch.float32).reshape(()) * denominator, denominator.detach()


def materialize_batch_scalars(
    metric_terms: MetricTerms,
    diagnostics: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    entries: list[tuple[str, str]] = []
    tensors: list[torch.Tensor] = []
    reference_device: torch.device | None = None

    for name, (numerator, denominator) in metric_terms.items():
        for kind, value in (("numerator", numerator), ("denominator", denominator)):
            if not torch.is_tensor(value) or value.numel() != 1:
                raise ValueError(f"Metric '{name}' {kind} must be a scalar tensor.")
            reference_device = reference_device or value.device
            entries.append((kind, name))
            tensors.append(value.detach().reshape(()))

    materialized_diagnostics: dict[str, Any] = {}
    for name, value in (diagnostics or {}).items():
        if torch.is_tensor(value) and value.numel() == 1:
            reference_device = reference_device or value.device
            entries.append(("diagnostic", name))
            tensors.append(value.detach().reshape(()))
        elif isinstance(value, (int, float)):
            materialized_diagnostics[name] = float(value)
        else:
            materialized_diagnostics[name] = value

    values: list[float] = []
    if tensors:
        device = reference_device or torch.device("cpu")
        packed = torch.stack([value.to(device=device, dtype=torch.float64) for value in tensors])
        values = [float(value) for value in packed.cpu().tolist()]

    numerators: dict[str, float] = {}
    denominators: dict[str, float] = {}
    for (kind, name), value in zip(entries, values):
        if kind == "numerator":
            numerators[name] = value
        elif kind == "denominator":
            denominators[name] = value
        else:
            materialized_diagnostics[name] = value
    return numerators, denominators, materialized_diagnostics


__all__ = ["MetricTerms", "materialize_batch_scalars", "mean_metric_term"]
