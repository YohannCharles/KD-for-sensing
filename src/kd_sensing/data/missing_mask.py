import torch


def sample_missing_mask(
    batch_size: int,
    num_modalities: int,
    p_missing: float | list[float] | tuple[float, ...] = 0.25,
    *,
    always_available_indices: list[int] | tuple[int, ...] | None = None,
    ensure_at_least_one: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if batch_size <= 0 or num_modalities <= 0:
        raise ValueError("batch_size and num_modalities must be positive.")
    probabilities = torch.as_tensor(p_missing, dtype=torch.float32, device=device)
    if probabilities.ndim == 0:
        probabilities = probabilities.expand(num_modalities)
    if tuple(probabilities.shape) != (num_modalities,) or torch.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("p_missing must be a probability or one probability per modality.")
    mask = torch.rand(batch_size, num_modalities, device=device, generator=generator).ge(probabilities)
    fixed = tuple(int(index) for index in always_available_indices or ())
    if any(index < 0 or index >= num_modalities for index in fixed):
        raise ValueError("always_available_indices contains an invalid modality index.")
    if fixed:
        mask[:, list(fixed)] = True
    if ensure_at_least_one:
        mask[~mask.any(dim=1), fixed[0] if fixed else 0] = True
    return mask.to(dtype=dtype)


__all__ = ["sample_missing_mask"]
