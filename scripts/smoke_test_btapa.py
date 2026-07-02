#!/usr/bin/env python3
import torch

from kd_sensing.losses.beam_prototype_alignment import (
    BeamPrototypeBank,
    make_beam_topology_soft_targets,
    prototype_alignment_loss,
)


def main() -> int:
    torch.manual_seed(7)
    batch_size, num_modalities, hidden_dim, num_beams = 4, 4, 16, 64
    labels = torch.tensor([0, 7, 31, 63])
    fused = torch.randn(batch_size, hidden_dim, requires_grad=True)
    modality = torch.randn(batch_size, num_modalities, hidden_dim, requires_grad=True)
    available = torch.tensor(
        [
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.bool,
    )
    bank = BeamPrototypeBank(hidden_dim, num_beams, temperature=0.1)

    target_linear = make_beam_topology_soft_targets(labels, num_beams, 2.0, circular=False)
    target_circular = make_beam_topology_soft_targets(labels, num_beams, 2.0, circular=True)
    assert target_linear.shape == (batch_size, num_beams)
    assert torch.isfinite(target_linear).all()
    assert torch.allclose(target_linear.sum(dim=1), torch.ones(batch_size), atol=1e-6)
    assert target_circular[0, -1] > target_linear[0, -1]

    loss, diagnostics = prototype_alignment_loss(
        bank,
        labels,
        fused_features=fused,
        modality_features=modality,
        mask=available,
        proto_target_type="beam_soft",
        tau_beam=2.0,
        circular_beam_distance=False,
        lambda_proto=0.2,
        btapa_include_fusion=True,
        btapa_include_modalities=True,
        btapa_fusion_weight=1.0,
        btapa_modality_weight=0.5,
        use_adba_aware_proto=True,
        lambda_adba_proto=0.05,
        adba_margin=3,
    )
    assert torch.isfinite(loss)
    assert diagnostics["prototype/modality_sample_count"] == float(available.sum())
    loss.backward()
    assert fused.grad is not None and torch.isfinite(fused.grad).all()
    assert modality.grad is not None and torch.isfinite(modality.grad).all()
    assert modality.grad[~available].abs().sum().item() == 0.0

    circ_loss, _ = prototype_alignment_loss(
        bank,
        labels,
        fused_features=torch.randn(batch_size, hidden_dim, requires_grad=True),
        proto_target_type="beam_soft",
        tau_beam=2.0,
        circular_beam_distance=True,
    )
    assert torch.isfinite(circ_loss)
    print("BTAPA smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
