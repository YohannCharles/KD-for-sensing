from typing import Any

import torch

from kd_sensing.losses.beam_prototype_alignment import prototype_alignment_loss, supervised_contrastive_loss


def add_prototype_alignment_losses(
    loss: torch.Tensor,
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    teacher_output: dict[str, torch.Tensor] | None,
    prototype_bank: torch.nn.Module | None,
    use_beam_prototype_alignment: bool,
    lambda_proto: float,
    lambda_modality_proto: float,
    lambda_supcon: float,
    lambda_teacher_proto: float,
    beam_label_sigma: float,
    beam_label_circular: bool,
    prototype_target_circular: bool | None,
    proto_target_type: str,
    tau_beam: float,
    circular_beam_distance: bool | None,
    btapa_include_fusion: bool,
    btapa_include_modalities: bool,
    btapa_fusion_weight: float,
    btapa_modality_weight: float | None,
    use_adba_aware_proto: bool,
    lambda_adba_proto: float,
    adba_margin: int,
    use_pattern_conditional_btapa: bool,
    pattern_names: list[str] | None,
    btapa_apply_patterns: list[str] | tuple[str, ...] | None,
    btapa_disable_on_patterns: list[str] | tuple[str, ...] | None,
    btapa_fallback_to_ordinary_proto: bool,
    ordinary_proto_target_type: str,
    proto_sample_weights: torch.Tensor | None,
    kd_temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    diagnostics: dict[str, float] = {}
    if not use_beam_prototype_alignment or prototype_bank is None:
        return loss, diagnostics
    proto_loss, proto_diag = prototype_alignment_loss(
        prototype_bank,
        labels,
        fused_features=output["output_features"],
        modality_features=output.get("modality_features"),
        mask=output.get("missing_mask"),
        teacher_features=(teacher_output or {}).get("output_features"),
        beam_label_sigma=beam_label_sigma,
        beam_label_circular=(
            beam_label_circular if prototype_target_circular is None else bool(prototype_target_circular)
        ),
        proto_target_type=proto_target_type,
        tau_beam=tau_beam,
        circular_beam_distance=circular_beam_distance,
        lambda_proto=lambda_proto,
        lambda_modality_proto=lambda_modality_proto,
        lambda_teacher_proto=lambda_teacher_proto,
        btapa_include_fusion=btapa_include_fusion,
        btapa_include_modalities=btapa_include_modalities,
        btapa_fusion_weight=btapa_fusion_weight,
        btapa_modality_weight=btapa_modality_weight,
        use_adba_aware_proto=use_adba_aware_proto,
        lambda_adba_proto=lambda_adba_proto,
        adba_margin=adba_margin,
        use_pattern_conditional_btapa=use_pattern_conditional_btapa,
        pattern_names=pattern_names,
        btapa_apply_patterns=btapa_apply_patterns,
        btapa_disable_on_patterns=btapa_disable_on_patterns,
        btapa_fallback_to_ordinary_proto=btapa_fallback_to_ordinary_proto,
        ordinary_proto_target_type=ordinary_proto_target_type,
        sample_weights=proto_sample_weights,
    )
    loss = loss + proto_loss
    diagnostics.update(proto_diag)
    if float(lambda_supcon) != 0.0:
        supcon, supcon_diag = supervised_contrastive_loss(output["output_features"], labels[:, 0], temperature=kd_temperature)
        loss = loss + float(lambda_supcon) * supcon
        diagnostics.update(supcon_diag)
    return loss, diagnostics
