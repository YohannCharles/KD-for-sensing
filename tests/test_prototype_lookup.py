import torch
import pytest

from kd_sensing.models.prototype_pilot_selector import (
    PrototypePilotSelector,
    load_prototype_pilot_lookup,
    select_from_lookup,
)


def test_lookup_exports_and_only_selected_patterns_reach_encoder(tmp_path):
    selector = PrototypePilotSelector(3, 8, num_selected_patterns=2)
    with torch.no_grad():
        selector.pilot_logits.copy_(torch.arange(24, dtype=torch.float32).reshape(3, 8))
    path = selector.export_lookup(tmp_path / "lookup.json", metadata={"split": "train"})
    lookup = load_prototype_pilot_lookup(path, num_prototypes=3, num_selected_patterns=2, num_candidate_patterns=8)
    candidates = torch.complex(torch.arange(2 * 8 * 4).reshape(2, 8, 4).float(), torch.zeros(2, 8, 4))
    selected = select_from_lookup(torch.tensor([0, 2]), candidates, lookup)

    assert selected["selected_y"].shape == (2, 2, 4)
    assert selected["pattern_ids"].tolist() == [[7, 6], [7, 6]]
    assert torch.equal(selected["selected_y"][0, 0], candidates[0, 7])


def test_training_selector_has_gradient_and_distinct_topk():
    selector = PrototypePilotSelector(4, 8, num_selected_patterns=3)
    selector.train()
    candidates = torch.randn(5, 8, 4, dtype=torch.complex64)
    result = selector(torch.tensor([0, 1, 2, 3, 0]), candidates, generator=torch.Generator().manual_seed(9))
    result["selected_y"].abs().mean().backward()
    assert selector.pilot_logits.grad is not None
    assert all(torch.unique(row).numel() == 3 for row in result["pattern_ids"])


def test_selector_supports_runtime_dense_to_sparse_budget():
    selector = PrototypePilotSelector(4, 8, num_selected_patterns=2).eval()
    with torch.no_grad():
        selector.pilot_logits.copy_(torch.arange(32, dtype=torch.float32).reshape(4, 8))
    candidates = torch.randn(3, 8, 6, dtype=torch.complex64)
    dense = selector(torch.tensor([0, 1, 2]), candidates, num_selected_patterns=8)
    sparse = selector(torch.tensor([0, 1, 2]), candidates, num_selected_patterns=4)
    assert dense["selected_y"].shape == (3, 8, 6)
    assert sparse["selected_y"].shape == (3, 4, 6)
    assert torch.equal(selector.lookup(8)[:, :4], selector.lookup(4))
    with pytest.raises(ValueError, match="candidate pattern budget"):
        selector(torch.tensor([0, 1, 2]), candidates, num_selected_patterns=9)


def test_selector_exports_runtime_lookup_width(tmp_path):
    selector = PrototypePilotSelector(4, 8, num_selected_patterns=2)
    path = selector.export_lookup(
        tmp_path / "dense.json",
        metadata={"source_split": "train"},
        num_selected_patterns=8,
    )
    lookup = load_prototype_pilot_lookup(path, num_prototypes=4, num_selected_patterns=8, num_candidate_patterns=8)
    assert lookup.shape == (4, 8)


def test_lookup_metadata_records_train_only_origin(tmp_path):
    selector = PrototypePilotSelector(3, 8, num_selected_patterns=2)
    path = selector.export_lookup(tmp_path / "lookup.json", metadata={"source_split": "full_pool_train"})
    assert "full_pool_train" in path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="train-only"):
        selector.export_lookup(tmp_path / "invalid.json", metadata={"source_split": "validation"})
