from pathlib import Path

import pytest
import torch

from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.baselines.sparse_pilot_transition import SparsePilotInformationClassifier
from tools.run_sparse_pilot_trajectory_recovery import (
    _conditional_expert_metrics,
    _fallback_probabilities,
    _initialize_model,
    _select_equal_re_shape,
    _select_low_re_candidate,
    _task_records,
    nested_frequency_indices,
    parse_budget,
    pilot_resource_accounting,
)


def test_nested_budget_is_deterministic_subset_of_mother_observation():
    assert parse_budget("8x8") == (8, 8)
    assert nested_frequency_indices(16, 8).tolist() == [0, 2, 4, 6, 9, 11, 13, 15]
    with pytest.raises(ValueError, match="exceeds|maximum|count"):
        nested_frequency_indices(16, 17)


def test_low_re_round_registers_nested_budget_and_shape_control():
    config = safe_load_yaml(Path("tools/configs/sparse_pilot_trajectory_recovery.yaml").read_text(encoding="utf-8"))
    assert config["rounds"]["round4_low_re"]["budgets"] == ["4x4", "2x4", "4x2", "2x2"]
    assert [parse_budget(value) for value in config["rounds"]["round4_low_re"]["budgets"]] == [
        (4, 4),
        (2, 4),
        (4, 2),
        (2, 2),
    ]


def test_single_frame_round_is_exactly_four_re_per_window():
    config = safe_load_yaml(Path("tools/configs/sparse_pilot_trajectory_recovery.yaml").read_text(encoding="utf-8"))
    assert config["rounds"]["round5_single_frame_4re"] == {
        "budgets": ["2x2"],
        "tasks": ["I2"],
        "fusion_mode": "replace",
    }
    assert pilot_resource_accounting("I2", (2, 2)) == {
        "pilot_re": 4,
        "pilot_re_per_frame": 4,
        "pilot_history_frames": 1,
        "pilot_re_total": 4,
    }
    assert pilot_resource_accounting("I3", (2, 2))["pilot_re_total"] == 20


def test_single_frame_total_twenty_re_round_has_equal_shape_controls():
    config = safe_load_yaml(Path("tools/configs/sparse_pilot_trajectory_recovery.yaml").read_text(encoding="utf-8"))
    assert config["rounds"]["round6_single_frame_20re"] == {
        "budgets": ["5x4", "4x5"],
        "tasks": ["I2"],
        "fusion_mode": "replace",
    }
    candidates = [
        {"budget": "5x4", "pilot_re_total": 20, "csi_top1": 0.601},
        {"budget": "4x5", "pilot_re_total": 20, "csi_top1": 0.597},
    ]
    assert all(pilot_resource_accounting("I2", parse_budget(row["budget"]))["pilot_re_total"] == 20 for row in candidates)
    assert _select_equal_re_shape(candidates)["budget"] == "4x5"


def test_low_re_selection_requires_weak_but_complementary_csi():
    baseline = {"all14_macro": 0.60, "all14_worst": 0.10}
    selected, assessed = _select_low_re_candidate(
        [
            {"budget": "4x4", "pilot_re": 16, "csi_top1": 0.70, "oracle_gain": 0.20},
            {"budget": "2x4", "pilot_re": 8, "csi_top1": 0.58, "oracle_gain": 0.04},
            {"budget": "4x2", "pilot_re": 8, "csi_top1": 0.57, "oracle_gain": 0.08},
            {"budget": "2x2", "pilot_re": 4, "csi_top1": 0.566, "oracle_gain": 0.07},
        ],
        baseline,
    )
    assert [row["eligible"] for row in assessed] == [False, False, True, True]
    assert selected is not None
    assert selected["budget"] == "2x2"


def test_conditional_expert_metrics_separate_fix_harm_and_oracle():
    labels = torch.tensor([0, 0, 1, 1])
    sensing = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.8, 0.2]])
    csi = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8]])
    metrics = _conditional_expert_metrics(csi, sensing, labels)
    assert metrics == {
        "sensing_top1": 0.5,
        "csi_top1": 0.75,
        "conditional_fix": 1.0,
        "conditional_harm": 0.5,
        "oracle_top1": 1.0,
        "error_overlap": 0.0,
    }


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


def test_i2_total_four_re_uses_only_last_history_frame():
    candidates = torch.arange(2 * 5 * 32 * 16).reshape(2, 5, 32, 16).to(torch.complex64)
    records = {
        "candidate_history": candidates,
        "labels_current": torch.tensor([1, 2]),
        "labels_future": torch.tensor([3, 4]),
        "current_beam_power": torch.rand(2, 64),
        "future_beam_power": torch.rand(2, 64),
        "p0_full": torch.softmax(torch.rand(2, 64), dim=-1),
    }
    selected = _task_records(records, "I2", (2, 2), max_frequencies=16)
    expected = candidates[:, -1:, :2].index_select(-1, torch.tensor([0, 15]))
    assert selected["candidate_history"].shape == (2, 1, 2, 2)
    assert torch.equal(selected["candidate_history"], expected)
    assert torch.equal(selected["labels"], records["labels_future"])


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
