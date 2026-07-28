import pytest
import torch

from kd_sensing.baselines.sparse_pilot_transition import SparsePilotInformationClassifier
from tools.run_sparse_pilot_trajectory_recovery import (
    _fallback_probabilities,
    _initialize_model,
    _task_records,
    nested_frequency_indices,
    parse_budget,
)


def test_nested_budget_is_deterministic_subset_of_mother_observation():
    assert parse_budget("8x8") == (8, 8)
    assert nested_frequency_indices(16, 8).tolist() == [0, 2, 4, 6, 9, 11, 13, 15]
    with pytest.raises(ValueError, match="exceeds|maximum|count"):
        nested_frequency_indices(16, 17)


def test_task_records_slice_budget_and_repeat_only_requested_masks():
    count = 3
    records = {
        "candidate_history": torch.randn(count, 5, 32, 16, dtype=torch.complex64),
        "labels_current": torch.arange(count),
        "labels_future": torch.arange(count) + 1,
        "current_beam_power": torch.rand(count, 64),
        "future_beam_power": torch.rand(count, 64),
        "z_image_only": torch.rand(count, 64),
        "z_full": torch.rand(count, 64),
        "p0_image_only": torch.softmax(torch.rand(count, 64), dim=-1),
        "p0_full": torch.softmax(torch.rand(count, 64), dim=-1),
    }
    selected = _task_records(
        records,
        "I5",
        (8, 8),
        max_frequencies=16,
        masks=("image_only", "full"),
    )
    assert selected["candidate_history"].shape == (6, 5, 8, 8)
    assert selected["mask_names"] == ("image_only", "full")
    assert selected["is_full"].tolist() == [False, False, False, True, True, True]


def test_residual_initialization_transfers_encoder_but_keeps_zero_output(tmp_path):
    source = SparsePilotInformationClassifier(
        history_length=5,
        sensing_dim=64,
        hidden_dim=16,
        encoder_layers=0,
    )
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "model_state": source.state_dict(),
            "epoch": 7,
            "selection_metric": "single_worst",
        },
        checkpoint,
    )
    residual = SparsePilotInformationClassifier(
        history_length=5,
        sensing_dim=64,
        hidden_dim=16,
        encoder_layers=0,
        fusion_mode="residual",
    )
    provenance = _initialize_model(residual, checkpoint, fusion_mode="residual")
    assert provenance["source_epoch"] == 7
    assert torch.equal(
        residual.csi_encoder.token_projection.weight,
        source.csi_encoder.token_projection.weight,
    )
    assert residual.classifier[-1].weight.count_nonzero() == 0
    assert residual.classifier[-1].bias.count_nonzero() == 0


def test_hard_fallback_uses_csi_only_for_at_most_two_sensing_modalities():
    csi = torch.tensor([[0.8, 0.2]])
    base = torch.tensor([[0.1, 0.9]])
    assert _fallback_probabilities(csi, base, "image_only") is csi
    assert _fallback_probabilities(csi, base, "missing_image_lidar") is csi
    assert _fallback_probabilities(csi, base, "missing_image") is base
    assert _fallback_probabilities(csi, base, "full") is base
