import torch

from kd_sensing.losses.beam_prototype_alignment import prototype_alignment_loss


def add_prototype_alignment_losses(
    loss: torch.Tensor,
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    prototype_bank: torch.nn.Module | None,
    enabled: bool,
    lambda_proto: float,
    lambda_modality_proto: float,
    beam_label_sigma: float,
    prototype_target_circular: bool,
    prototype_topology_id: str | None = None,
    prototype_topology_permutation: list[int] | tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not enabled or prototype_bank is None:
        return loss, {}
    prototype_loss, diagnostics = prototype_alignment_loss(
        prototype_bank,
        labels,
        fused_features=output["output_features"],
        modality_features=output.get("modality_features"),
        mask=output.get("missing_mask"),
        beam_label_sigma=beam_label_sigma,
        circular=prototype_target_circular,
        topology_id=prototype_topology_id,
        topology_permutation=prototype_topology_permutation,
        lambda_proto=lambda_proto,
        lambda_modality_proto=lambda_modality_proto,
    )
    return loss + prototype_loss, diagnostics
