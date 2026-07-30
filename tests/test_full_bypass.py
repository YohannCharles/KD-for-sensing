import torch

from kd_sensing.models.missing_sensing_prototype_adapter import MissingSensingPrototypeAdapter
from kd_sensing.models.propagation_mode_fusion import full_path_bypass


def test_full_path_bypass_is_bitwise_exact() -> None:
    original = torch.randn(8, 64)
    candidate = torch.randn(8, 64)
    output = full_path_bypass(original, candidate, torch.ones(8, dtype=torch.bool))
    assert torch.equal(output, original)


def test_adapter_is_only_enabled_for_missing_rows() -> None:
    adapter = MissingSensingPrototypeAdapter(embedding_dim=4, bottleneck_dim=2)
    with torch.no_grad():
        adapter.mix.fill_(1.0)
        adapter.up.weight.fill_(0.1)
    feature = torch.randn(2, 4)
    output = adapter(feature, torch.tensor([False, True]))
    assert torch.equal(output[0], feature[0])
    assert not torch.equal(output[1], feature[1])


def test_zero_initialized_adapter_is_exactly_m4_equivalent() -> None:
    adapter = MissingSensingPrototypeAdapter(embedding_dim=64, bottleneck_dim=16)
    feature = torch.randn(11, 64)
    assert torch.equal(adapter(feature, True), feature)
