import hashlib

import torch
import torch.nn as nn

from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank
from kd_sensing.models.missing_decision_adapter import (
    FrozenU0DecisionAdapter,
    MissingDecisionAdapter,
    is_full_mask,
)
from kd_sensing.modalities import MODALITY_ORDER


class _ToyU0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)
        self.modalities = MODALITY_ORDER
        self.calls = 0

    def forward(self, *, features: torch.Tensor, missing_mask: torch.Tensor | None = None, **_: object) -> dict:
        self.calls += 1
        h_proto = self.projection(features)
        logits = h_proto.unsqueeze(1)
        batch = features.shape[0]
        assignment = torch.softmax(h_proto, dim=-1)
        return {
            "logits": logits,
            "output_features": h_proto,
            "prototype_state": {
                "assignment": assignment,
                "nearest_id": assignment.argmax(dim=-1),
                "nearest_distance": h_proto[:, 0],
                "distance_margin": h_proto[:, 1],
                "entropy": h_proto[:, 2],
                "restoration_residual_norm": h_proto[:, 3],
            },
            "missing_mask": torch.ones(batch, 4, dtype=torch.bool, device=features.device)
            if missing_mask is None
            else missing_mask,
        }


def _state_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def test_prototype_describe_is_label_free_and_preserves_forward_logits() -> None:
    bank = BeamPrototypeBank(4, 4)
    features = torch.randn(3, 4, requires_grad=True)

    logits = bank(features)
    state = bank.describe(features)

    assert torch.equal(logits, bank(features))
    assert state["assignment"].shape == (3, 4)
    assert state["nearest_id"].shape == (3,)
    assert set(state) == {
        "assignment",
        "nearest_id",
        "nearest_distance",
        "distance_margin",
        "entropy",
        "restoration_residual_norm",
    }


def test_full_path_never_calls_adapter_and_is_exactly_equivalent() -> None:
    base = _ToyU0()
    adapter = MissingDecisionAdapter(4, num_classes=4, rank=2, variant="proto_uncertainty_lora", prototype_dim=4)
    model = FrozenU0DecisionAdapter(base, adapter)
    features = torch.randn(3, 4)
    full = torch.ones(3, 4, dtype=torch.bool)

    expected = base(features=features, missing_mask=full)["logits"]
    output = model(features=features, missing_mask=full)

    assert output["adapter_called"] is False
    assert torch.equal(output["base_logits"], expected)
    assert torch.equal(output["logits"], expected)
    assert output["logits"].data_ptr() == output["base_logits"].data_ptr()
    assert float(output["delta_logits"].abs().max()) == 0.0


def test_zero_initialization_matches_missing_u0_and_only_adapter_updates() -> None:
    base = _ToyU0()
    adapter = MissingDecisionAdapter(4, num_classes=4, rank=2, variant="proto_lora", prototype_dim=4)
    model = FrozenU0DecisionAdapter(base, adapter)
    features = torch.randn(2, 4)
    missing = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
    before = _state_digest(base)

    output = model(features=features, missing_mask=missing)
    loss = output["logits"].sum()
    loss.backward()

    assert output["adapter_called"] is True
    assert torch.equal(output["logits"], output["base_logits"])
    assert _state_digest(base) == before
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert all(parameter.grad is None for parameter in base.parameters())
    assert any(parameter.grad is not None for parameter in adapter.parameters())
    assert float(adapter.up.weight.grad.abs().sum()) > 0


def test_fixed_modality_order_and_radar_gps_mask_are_enforced() -> None:
    assert MODALITY_ORDER == ("image", "radar", "gps", "lidar")
    radar_gps = torch.tensor([[0, 1, 1, 0]], dtype=torch.bool)
    assert not bool(is_full_mask(radar_gps).item())
    assert bool(is_full_mask(torch.ones(1, 4, dtype=torch.bool)).item())


def test_mixed_precision_adapter_output_is_cast_to_base_dtype() -> None:
    class _BFloatResidual(nn.Module):
        def forward(self, h_proto, mask, proto_state):
            batch = h_proto.shape[0]
            return torch.zeros(batch, 4, dtype=torch.bfloat16), torch.zeros(batch, 2, dtype=torch.bfloat16)

    model = FrozenU0DecisionAdapter(_ToyU0(), _BFloatResidual())
    output = model(
        features=torch.randn(2, 4),
        missing_mask=torch.tensor([[0, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool),
    )

    assert output["logits"].dtype == torch.float32
    assert output["delta_logits"].dtype == torch.float32
    assert output["adapter_alpha"].dtype == torch.float32


def test_simple_bias_variants_are_zero_initialized_and_factorized_is_compositional() -> None:
    masks = torch.tensor([[0, 1, 1, 1], [1, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.bool)
    features = torch.zeros(3, 4)
    for variant, count in (("global_bias", 4), ("mask_lookup", 64), ("factorized_bias", 20)):
        adapter = MissingDecisionAdapter(4, num_classes=4, rank=2, variant=variant, prototype_dim=4)
        update, _ = adapter(features, masks)
        assert torch.equal(update, torch.zeros_like(update))
        assert adapter.parameter_count() == count

    factorized = MissingDecisionAdapter(4, num_classes=4, rank=2, variant="factorized_bias")
    with torch.no_grad():
        factorized.factorized.weight.copy_(torch.eye(4))
    update, _ = factorized(features, masks)
    assert torch.equal(update[2], update[0] + update[1])


def test_circular_transport_preserves_mass_wraps_and_composes_missing_modalities() -> None:
    adapter = MissingDecisionAdapter(64, num_classes=64, rank=2, variant="circular_transport", transport_radius=3)
    with torch.no_grad():
        adapter.transport_kernel_logits.fill_(-80.0)
        adapter.transport_kernel_logits[0, 4] = 80.0  # Image missing: +1 circular shift.
        adapter.transport_kernel_logits[1, 5] = 80.0  # Radar missing: +2 circular shift.

    base = torch.full((2, 64), -80.0)
    base[0, 63] = 80.0
    base[1, 60] = 80.0
    masks = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]], dtype=torch.bool)
    transported, alpha = adapter.transport_logits(base, masks)
    probabilities = torch.softmax(transported, dim=-1)
    composed = adapter.composed_transport_kernel(masks)

    assert adapter.parameter_count() == 4 * 7
    assert alpha.shape == (2, 4 * 7)
    assert torch.all(probabilities >= 0)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2), atol=1e-7)
    assert torch.allclose(composed.sum(dim=-1), torch.ones(2), atol=1e-7)
    assert probabilities[0].argmax().item() == 0  # 63 + 1 wraps to 0.
    assert probabilities[1].argmax().item() == 63  # 60 + 1 + 2 wraps to 63.

    kernels = adapter.transport_kernel()
    expected = torch.zeros(64)
    expected[0] = 1.0
    for kernel in (kernels[0], kernels[1]):
        expected = sum(
            kernel[index] * torch.roll(expected, shifts=shift, dims=-1)
            for index, shift in enumerate(range(-3, 4))
        )
    assert torch.allclose(composed[1], expected, atol=1e-7)


def test_circular_transport_full_path_is_exact_and_kernels_receive_gradients() -> None:
    adapter = MissingDecisionAdapter(4, num_classes=4, rank=2, variant="circular_transport", transport_radius=1)
    model = FrozenU0DecisionAdapter(_ToyU0(), adapter)
    features = torch.randn(2, 4)
    full = torch.ones(2, 4, dtype=torch.bool)
    full_output = model(features=features, missing_mask=full)

    assert full_output["adapter_called"] is False
    assert full_output["logits"].data_ptr() == full_output["base_logits"].data_ptr()
    assert torch.equal(full_output["logits"], full_output["base_logits"])

    missing = torch.tensor([[0, 1, 1, 1], [1, 0, 1, 1]], dtype=torch.bool)
    output = model(features=features, missing_mask=missing)
    loss = torch.nn.functional.cross_entropy(output["logits"][:, 0, :], torch.tensor([0, 1]))
    loss.backward()

    assert torch.isfinite(output["logits"]).all()
    assert adapter.transport_kernel_logits is not None
    assert adapter.transport_kernel_logits.grad is not None
    assert float(adapter.transport_kernel_logits.grad.abs().sum()) > 0.0
