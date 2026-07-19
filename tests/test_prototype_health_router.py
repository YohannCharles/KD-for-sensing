import pytest
import torch

from kd_sensing.losses.beam_prototype_alignment import beam_topology_positions, make_soft_beam_labels
from kd_sensing.models.prototype_health_router import (
    CONSENSUS_FEATURE_DIM,
    H2R_FEATURE_DIM,
    TEMPORAL_FEATURE_DIM,
    PrototypeReliabilityRouter,
    leave_one_out_consensus_features,
)


def _inputs(
    *,
    batch: int = 2,
    steps: int = 3,
    modalities: int = 3,
    features: int = 4,
    classes: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    latent = torch.randn(batch, steps, modalities, features, generator=generator)
    logits = torch.randn(batch, steps, modalities, classes, generator=generator)
    reliability = torch.sigmoid(torch.randn(batch, steps, modalities, 1, generator=generator))
    mask = torch.ones(batch, steps, modalities, dtype=torch.bool)
    mask[0, -1, 0] = False
    mask[1, :, -1] = False
    base = torch.randn(batch, modalities, 5, generator=generator)
    return latent, logits, reliability, mask, base


def _router(variant: str) -> PrototypeReliabilityRouter:
    return PrototypeReliabilityRouter(
        variant=variant,
        modality_count=3,
        num_classes=8,
        base_feature_dim=5,
        prior_weights=[0.6, 0.3, 0.1],
        topology_id="cyclic_index_v1",
        circular=True,
        dropout=0.0,
    )


def test_public_topology_positions_are_reused_by_soft_labels() -> None:
    positions = beam_topology_positions(4, topology_id="permuted_index_v1", topology_permutation=[0, 2, 3, 1])
    assert torch.equal(positions, torch.tensor([0.0, 2.0, 3.0, 1.0]))

    target = make_soft_beam_labels(
        torch.tensor([0]),
        4,
        0.5,
        circular=True,
        topology_id="permuted_index_v1",
        topology_permutation=[0, 2, 3, 1],
    )
    assert target[0, 3] > target[0, 1]
    with pytest.raises(ValueError, match="bijection"):
        beam_topology_positions(4, topology_id="permuted_index_v1", topology_permutation=[0, 0, 2, 3])


def test_patr_zero_initialization_falls_back_to_masked_prior_and_mean_pool() -> None:
    latent, logits, reliability, mask, base = _inputs()
    router = _router("patr")
    evidence = router.prepare(latent, logits, reliability, mask)
    temporal = router.temporal_pool(latent, evidence)
    pooled_logits = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(9))
    output = router.route(base, pooled_logits, evidence, temporal)

    weights = mask.to(dtype=latent.dtype).unsqueeze(-1)
    expected_pool = (latent * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    assert torch.allclose(temporal.features, expected_pool)
    assert torch.equal(temporal.weights.masked_select(~mask), torch.zeros_like(temporal.weights.masked_select(~mask)))
    assert torch.allclose(output.weights, output.prior_weights)
    assert torch.equal(output.residual_logits, torch.zeros_like(output.residual_logits))
    assert torch.allclose(output.effective_cell_weights.sum(dim=(1, 2)), torch.ones(2))
    assert output.modality_features.shape[-1] == 5 + TEMPORAL_FEATURE_DIM
    assert not evidence.frame_probabilities.requires_grad

    expected_second = torch.tensor([2.0 / 3.0, 1.0 / 3.0, 0.0])
    assert torch.allclose(output.weights[1], expected_second)


def test_h2r_masks_empty_modalities_and_health_head_receives_gradient() -> None:
    latent, logits, reliability, mask, base = _inputs()
    latent[:, :, 0, 0] = torch.tensor([[-2.0, 0.0, 4.0], [1.0, 3.0, 8.0]])
    router = _router("h2r")
    evidence = router.prepare(latent, logits, reliability, mask)
    temporal = router.temporal_pool(latent, evidence)
    pooled_logits = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(11))
    output = router.route(base, pooled_logits, evidence, temporal)

    assert router.frame_health_head is not None
    assert torch.equal(temporal.weights.masked_select(~mask), torch.zeros_like(temporal.weights.masked_select(~mask)))
    assert torch.allclose(temporal.weights.sum(dim=1)[0], torch.ones(3))
    assert torch.allclose(temporal.weights.sum(dim=1)[1], torch.tensor([1.0, 1.0, 0.0]))
    assert output.modality_features.shape[-1] == 5 + H2R_FEATURE_DIM

    temporal.features[:, 0, 0].sum().backward()
    gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in router.frame_health_head.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0.0


def test_core_marks_an_outlier_and_safely_disables_single_modality_consensus() -> None:
    probabilities = torch.full((2, 4, 8), 1.0e-4)
    probabilities[0, 0, 0] = 1.0
    probabilities[0, 1, 0] = 1.0
    probabilities[0, 2, 0] = 1.0
    probabilities[0, 3, 4] = 1.0
    probabilities[1, 0, 2] = 1.0
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    available = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
    features = leave_one_out_consensus_features(
        probabilities,
        available,
        torch.arange(8, dtype=torch.float32),
        circular=True,
        top_k=1,
    )

    assert features.shape == (2, 4, CONSENSUS_FEATURE_DIM)
    assert features[0, 3, 0] > features[0, 0, 0]
    assert features[0, 3, 1] > features[0, 0, 1]
    assert features[0, 3, 2] < features[0, 0, 2]
    assert torch.equal(features[1], torch.zeros_like(features[1]))

    latent, logits, reliability, mask, base = _inputs()
    mask[1, :, 1:] = False
    router = _router("core")
    evidence = router.prepare(latent, logits, reliability, mask)
    temporal = router.temporal_pool(latent, evidence)
    output = router.route(base, torch.randn(2, 3, 8), evidence, temporal)
    assert torch.equal(output.weights[1], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(output.consensus_features[1], torch.zeros_like(output.consensus_features[1]))


def test_unified_hpr_uses_one_shared_route_and_all_evidence_groups() -> None:
    latent, logits, reliability, mask, base = _inputs()
    router = _router("unified_hpr")
    evidence = router.prepare(latent, logits, reliability, mask)
    temporal = router.temporal_pool(latent, evidence)
    output = router.route(base, torch.randn(2, 3, 8), evidence, temporal)

    expected_features = 5 + TEMPORAL_FEATURE_DIM + H2R_FEATURE_DIM + CONSENSUS_FEATURE_DIM
    assert output.modality_features.shape == (2, 3, expected_features)
    assert output.weights.shape == (2, 3)
    assert output.effective_cell_weights.shape == (2, 3, 3)
    assert torch.allclose(output.weights.sum(dim=1), torch.ones(2))
    assert torch.allclose(output.effective_cell_weights.sum(dim=(1, 2)), torch.ones(2))
    assert set(name for name, _ in router.named_parameters()) == {
        "prior_logits",
        "frame_health_head.0.weight",
        "frame_health_head.0.bias",
        "frame_health_head.1.weight",
        "frame_health_head.1.bias",
        "frame_health_head.4.weight",
        "frame_health_head.4.bias",
        "modality_residual_head.0.weight",
        "modality_residual_head.0.bias",
        "modality_residual_head.1.weight",
        "modality_residual_head.1.bias",
        "modality_residual_head.4.weight",
        "modality_residual_head.4.bias",
    }


def test_router_rejects_unknown_variants_and_incompatible_topology() -> None:
    with pytest.raises(ValueError, match="router variant"):
        PrototypeReliabilityRouter(
            variant="unknown",
            modality_count=3,
            num_classes=8,
            base_feature_dim=5,
        )
    with pytest.raises(ValueError, match="conflicts"):
        PrototypeReliabilityRouter(
            variant="patr",
            modality_count=3,
            num_classes=8,
            base_feature_dim=5,
            topology_id="linear_index_v1",
            circular=True,
        )
