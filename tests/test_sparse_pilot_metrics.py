from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, TensorDataset

from kd_sensing.config.parsing import safe_load_yaml
from tools.run_sparse_pilot_transition import (
    _aggregate_evaluation_chunks,
    apply_budget_arm,
    balanced_subset_indices,
    missing_mask_schedule,
    nested_frequency_indices,
    noisy_observations,
    prediction_metrics,
    resolve_budget_curriculum,
    resolve_methods,
    slice_record_frequencies,
)


def test_sparse_pilot_metrics_fix_harm_and_resource_gain():
    labels = torch.tensor([0, 1, 2, 3])
    base = torch.nn.functional.one_hot(torch.tensor([0, 0, 2, 2]), 64).float()
    final = torch.nn.functional.one_hot(torch.tensor([0, 1, 1, 3]), 64).float()
    power = torch.ones(4, 64)
    metrics = prediction_metrics(final, labels, base, power)
    assert metrics["top1"] == 0.75
    assert metrics["fix_rate"] == 1.0
    assert metrics["harm_rate"] == 0.5
    assert metrics["normalized_beamforming_gain"] == 1.0


def test_noise_and_dropout_are_reproducible():
    selected = torch.ones(2, 4, 8, dtype=torch.complex64)
    snr = torch.tensor([0.0, 10.0])
    first = noisy_observations(selected, snr, dropout=0.2, generator=torch.Generator().manual_seed(4))
    second = noisy_observations(selected, snr, dropout=0.2, generator=torch.Generator().manual_seed(4))
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_dense_to_sparse_frequency_budgets_are_nested_and_monotonic():
    config = {
        "pilot_codebook": {"num_candidate_patterns": 32},
        "channel": {"pilot_subcarriers": 16},
        "training": {
            "budget_curriculum": [
                {"name": "D32x16", "num_selected_patterns": 32, "num_pilot_subcarriers": 16, "epochs": 2},
                {"name": "D16x16", "num_selected_patterns": 16, "num_pilot_subcarriers": 16, "epochs": 2},
                {"name": "S8x8", "num_selected_patterns": 8, "num_pilot_subcarriers": 8, "epochs": 2},
                {"name": "T4x8", "num_selected_patterns": 4, "num_pilot_subcarriers": 8, "epochs": 2},
            ]
        },
        "evaluation": {"primary_num_selected_patterns": 4, "primary_num_pilot_subcarriers": 8},
    }
    stages = resolve_budget_curriculum(config, fallback_epochs=1)
    assert [stage["frequency_token_indices"].tolist() for stage in stages] == [
        list(range(16)),
        list(range(16)),
        list(range(0, 16, 2)),
        list(range(0, 16, 2)),
    ]
    assert nested_frequency_indices(16, 4).tolist() == [0, 4, 8, 12]
    records = {
        "candidate_g": torch.zeros(2, 32, 16, dtype=torch.complex64),
        "candidate_multice": torch.zeros(2, 32, 16, dtype=torch.complex64),
        "labels": torch.arange(2),
    }
    sliced = slice_record_frequencies(records, stages[-1]["frequency_token_indices"])
    assert sliced["candidate_g"].shape == (2, 32, 8)
    assert sliced["candidate_multice"].shape == (2, 32, 8)
    assert sliced["labels"] is records["labels"]


def test_dense_to_sparse_curriculum_rejects_budget_growth():
    config = {
        "pilot_codebook": {"num_candidate_patterns": 32},
        "channel": {"pilot_subcarriers": 16},
        "training": {
            "budget_curriculum": [
                {"name": "sparse", "num_selected_patterns": 4, "num_pilot_subcarriers": 8, "epochs": 1},
                {"name": "dense", "num_selected_patterns": 8, "num_pilot_subcarriers": 16, "epochs": 1},
            ]
        },
        "evaluation": {"primary_num_selected_patterns": 8, "primary_num_pilot_subcarriers": 16},
    }
    try:
        resolve_budget_curriculum(config, fallback_epochs=1)
    except ValueError as error:
        assert "monotonic" in str(error)
    else:
        raise AssertionError("budget growth must fail closed")


def test_dense_to_sparse_local_config_resolves_without_outputs():
    path = Path("tools/configs/sparse_pilot_dense_to_sparse.yaml")
    config = safe_load_yaml(path.read_text(encoding="utf-8"))
    stages = resolve_budget_curriculum(config, fallback_epochs=99)
    assert [
        (stage["num_selected_patterns"], stage["num_pilot_subcarriers"], stage["epochs"])
        for stage in stages
    ] == [(32, 16, 2), (16, 16, 2), (8, 8, 2), (4, 8, 2)]
    assert config["output"]["root"] == "outputs/sparse_pilot_transition_dense_to_sparse"


def test_matched_update_budget_arms_resolve_to_eight_epochs():
    path = Path("tools/configs/sparse_pilot_dense_to_sparse.yaml")
    expected = {
        "dense32x16": [(32, 16, 8)],
        "mid16x16": [(16, 16, 8)],
        "mid8x16": [(8, 16, 8)],
        "mid16x8": [(16, 8, 8)],
        "mid8x8": [(8, 8, 8)],
        "spatial4x16": [(4, 16, 8)],
        "target4x8": [(4, 8, 8)],
        "curriculum": [(32, 16, 2), (16, 16, 2), (8, 8, 2), (4, 8, 2)],
    }
    for arm, budgets in expected.items():
        config = safe_load_yaml(path.read_text(encoding="utf-8"))
        stages = resolve_budget_curriculum(apply_budget_arm(config, arm), fallback_epochs=99)
        assert [(s["num_selected_patterns"], s["num_pilot_subcarriers"], s["epochs"]) for s in stages] == budgets
        assert sum(stage["epochs"] for stage in stages) == 8


def test_scale_up_arm_epochs_and_balanced_subset_are_deterministic():
    config = safe_load_yaml(Path("tools/configs/sparse_pilot_dense_to_sparse.yaml").read_text(encoding="utf-8"))
    stages = resolve_budget_curriculum(apply_budget_arm(config, "curriculum", total_epochs=40), fallback_epochs=1)
    assert [stage["epochs"] for stage in stages] == [10, 10, 10, 10]

    dataset = ConcatDataset([TensorDataset(torch.arange(9)), TensorDataset(torch.arange(12)), TensorDataset(torch.arange(15))])
    indices = balanced_subset_indices(dataset, 12)
    assert indices == balanced_subset_indices(dataset, 12)
    assert len(indices) == len(set(indices)) == 12
    assert [sum(start <= index < stop for index in indices) for start, stop in ((0, 9), (9, 21), (21, 36))] == [4, 4, 4]


def test_missing_fallback_schedule_emphasizes_one_and_two_modality_cases():
    weights = {1: 0.50, 2: 0.35, 3: 0.10, 4: 0.05}
    schedule, audit = missing_mask_schedule(2000, weights, seed=2026)
    repeated, repeated_audit = missing_mask_schedule(2000, weights, seed=2026)
    assert torch.equal(schedule, repeated)
    assert audit == repeated_audit
    assert audit["cardinality_counts"] == {"1": 1000, "2": 700, "3": 200, "4": 100}
    assert torch.bincount(schedule.sum(dim=1), minlength=5).tolist() == [0, 1000, 700, 200, 100]
    assert max(audit["pattern_counts"].values()) - min(
        value for key, value in audit["pattern_counts"].items() if key != "full"
    ) <= 225


def test_matched_update_method_subset_requires_c0_and_c5():
    config = {"evaluation": {"methods": ["C0", "C5"]}}
    assert resolve_methods(config, "C0,C5") == ("C0", "C5")
    try:
        resolve_methods(config, "C5")
    except ValueError as error:
        assert "require C0 and C5" in str(error)
    else:
        raise AssertionError("matched diagnostic must retain its baseline")


def test_batched_evaluation_aggregation_preserves_fix_harm_denominators():
    first = {
        "sample_count": 8.0,
        "base_correct_count": 2.0,
        "base_incorrect_count": 6.0,
        "top1": 0.5,
        "fix_rate": 0.5,
        "harm_rate": 0.0,
        "fallback_max_abs_error": 0.0,
        "snr_db": 10.0,
        "csi_parameters": 3.0,
        "pilot_soundings": 4.0,
        "pilot_resource_elements": 32.0,
    }
    second = first | {
        "sample_count": 2.0,
        "base_correct_count": 1.0,
        "base_incorrect_count": 1.0,
        "top1": 1.0,
        "fix_rate": 1.0,
        "harm_rate": 1.0,
    }
    result = _aggregate_evaluation_chunks([first, second])
    assert result["top1"] == 0.6
    assert result["fix_rate"] == 4.0 / 7.0
    assert result["harm_rate"] == 1.0 / 3.0
