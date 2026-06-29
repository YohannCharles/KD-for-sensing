import torch


def dft_codebook(num_antennas: int, num_beams: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if num_antennas <= 0 or num_beams <= 0:
        raise ValueError(f"num_antennas and num_beams must be positive, got {num_antennas}, {num_beams}.")
    antennas = torch.arange(num_antennas, device=device, dtype=dtype).unsqueeze(-1)
    beams = torch.arange(num_beams, device=device, dtype=dtype).unsqueeze(0)
    codebook = torch.exp(-2j * torch.pi * antennas * beams / float(num_beams))
    return codebook / torch.linalg.norm(codebook, dim=0, keepdim=True).clamp_min(1e-12)


def beam_logits_from_channel(
    channel: torch.Tensor,
    *,
    num_beams: int = 64,
    codebook: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, str]]:
    if not torch.is_complex(channel):
        raise TypeError("beam_logits_from_channel expects a complex channel tensor.")
    num_antennas = int(channel.shape[-1])
    source = "provided"
    if codebook is None:
        codebook = dft_codebook(num_antennas, int(num_beams), device=channel.device, dtype=channel.real.dtype)
        source = "ula_dft_fallback"
    cb = codebook.to(device=channel.device)
    projected = torch.einsum("...ka,ab->...kb", channel, cb.conj())
    logits = projected.abs().square().mean(dim=-2)
    return logits.to(torch.float32), {"codebook_source": source}


__all__ = ["beam_logits_from_channel", "dft_codebook"]
