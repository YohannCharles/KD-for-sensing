import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_csi_complementary_direction_diagnostic import (  # noqa: E402
    RidgeWorkspace,
    _cross_entropy_values,
    _direction_rank,
    _probe_scope_success,
    _write_json,
    _target_rank,
    benjamini_hochberg,
    bootstrap_weights,
    delete_directions,
    fisher_scores,
    fit_bin_edges,
    fixed_bin_mutual_information,
    project_directions,
    prototype_basis,
    reconstruct_directions,
    replace_directions,
    require_inner_split,
    ridge_predict,
    validate_historical_identity,
    validate_sample_alignment,
)


def test_shared_prototype_basis_is_orthonormal_and_has_expected_shape() -> None:
    generator = torch.Generator().manual_seed(1)
    prototype = torch.randn(64, 64, generator=generator)
    singular, directions = prototype_basis(prototype)
    assert singular.shape == (64,)
    assert directions.shape == (64, 64)
    assert torch.allclose(directions.T @ directions, torch.eye(64), atol=1e-5)
    assert bool((singular[:-1] >= singular[1:]).all())


def test_direction_projection_reconstructs_without_loss() -> None:
    generator = torch.Generator().manual_seed(2)
    prototype = torch.randn(64, 64, generator=generator)
    _, directions = prototype_basis(prototype)
    feature = torch.randn(7, 64, generator=generator)
    reconstructed = reconstruct_directions(project_directions(feature, directions), directions)
    assert torch.allclose(feature, reconstructed, atol=1e-5)


def test_replacement_and_deletion_touch_only_selected_coefficients() -> None:
    directions = torch.eye(4)
    sensing = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    radio = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
    assert torch.equal(
        replace_directions(sensing, radio, directions, [1, 3]),
        torch.tensor([[1.0, 6.0, 3.0, 8.0]]),
    )
    assert torch.equal(
        delete_directions(radio, directions, [0, 2]),
        torch.tensor([[0.0, 6.0, 0.0, 8.0]]),
    )


def test_direction_selection_api_has_no_validation_input() -> None:
    parameters = inspect.signature(_direction_rank).parameters
    assert not any("validation" in name for name in parameters)
    assert "labels" in parameters


def test_outer_test_role_fails_closed() -> None:
    assert require_inner_split("train") == "train"
    assert require_inner_split("validation") == "validation"
    with pytest.raises(ValueError, match="outer test remains sealed"):
        require_inner_split("test")


@pytest.mark.parametrize(
    "override,match",
    [
        ({"history_frames": 4}, "five-frame"),
        ({"re_per_frame": 5}, "five-frame"),
        ({"re_window": 21}, "five-frame"),
        ({"future_channel_used": True}, "Future"),
        ({"outer_test_accessed": True}, "Outer"),
    ],
)
def test_historical_identity_rejects_future_outer_or_wrong_re(
    override: dict[str, object], match: str
) -> None:
    identity: dict[str, object] = {
        "history_frames": 5,
        "re_per_frame": 4,
        "re_window": 20,
        "future_channel_used": False,
        "outer_test_accessed": False,
    }
    identity.update(override)
    with pytest.raises(ValueError, match=match):
        validate_historical_identity(identity)


def test_sample_alignment_is_fail_closed() -> None:
    reference = {
        "sample_ids": ["a", "b"],
        "trajectory_ids": ["t0", "t1"],
        "mask_names": ["m0"],
        "target": torch.tensor([1, 2]),
    }
    validate_sample_alignment(reference, dict(reference))
    for field, replacement in (
        ("sample_ids", ["b", "a"]),
        ("trajectory_ids", ["t1", "t0"]),
        ("mask_names", ["other"]),
        ("target", torch.tensor([2, 1])),
    ):
        changed = dict(reference)
        changed[field] = replacement
        with pytest.raises(ValueError, match="not aligned"):
            validate_sample_alignment(reference, changed)


def test_bootstrap_and_random_controls_are_reproducible() -> None:
    first = bootstrap_weights(40, 1000, 7)
    second = bootstrap_weights(40, 1000, 7)
    other = bootstrap_weights(40, 1000, 8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.allclose(first.sum(axis=1), 1.0)


def test_benjamini_hochberg_is_monotone_in_ranked_order() -> None:
    p_values = np.array([0.04, 0.001, 0.03, 0.2])
    adjusted = benjamini_hochberg(p_values)
    order = np.argsort(p_values)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)
    assert np.all((adjusted >= 0) & (adjusted <= 1))


def test_train_fitted_mi_detects_signal_and_label_permutation_control() -> None:
    rng = np.random.default_rng(9)
    labels = np.repeat(np.arange(4), 100)
    values = np.stack((labels + 0.05 * rng.standard_normal(len(labels)), rng.standard_normal(len(labels))), axis=1)
    edges = fit_bin_edges(values, 8)
    aligned = fixed_bin_mutual_information(values, labels, edges, num_classes=4)
    permuted = fixed_bin_mutual_information(
        values, labels[rng.permutation(len(labels))], edges, num_classes=4
    )
    assert aligned[0] > aligned[1]
    assert aligned[0] > permuted[0]
    assert fisher_scores(values, labels, num_classes=4)[0] > fisher_scores(values, labels, num_classes=4)[1]


def test_closed_form_ridge_probe_learns_without_checkpoint() -> None:
    rng = np.random.default_rng(10)
    labels = np.repeat(np.arange(64), 4)
    features = np.eye(64)[labels] + 0.01 * rng.standard_normal((len(labels), 64))
    workspace = RidgeWorkspace(features, labels)
    model = workspace.fit(np.arange(64), 1e-4)
    prediction = ridge_predict(model, features).argmax(axis=1)
    assert (prediction == labels).mean() > 0.99


def test_target_rank_and_ce_find_sensing_and_radio_sample_axes() -> None:
    target = torch.tensor([0, 1, 2])
    sensing = torch.zeros(3, 2, 4)
    radio = torch.zeros(2, 3, 4)
    sensing[torch.arange(3), :, target] = 2.0
    radio[:, torch.arange(3), target] = 2.0
    assert torch.equal(_target_rank(sensing, target), torch.ones(3, 2, dtype=torch.long))
    assert torch.equal(_target_rank(radio, target), torch.ones(2, 3, dtype=torch.long))
    assert _cross_entropy_values(sensing, target).shape == (3, 2)
    assert _cross_entropy_values(radio, target).shape == (2, 3)


def test_json_writer_handles_numpy_and_torch_values(tmp_path: Path) -> None:
    destination = tmp_path / "summary.json"
    _write_json(
        destination,
        {
            "array": np.array([1, 2]),
            "scalar": np.float64(0.5),
            "tensor": torch.tensor([3]),
        },
    )
    assert destination.read_text(encoding="utf-8").startswith("{")


def test_probe_gate_uses_preregistered_any_scope_rule() -> None:
    scopes = {
        "all14": {
            "gain": 0.017,
            "p3_top8": 0.572,
            "p4_random_mean": 0.5703,
            "p4_random_std": 0.0012,
            "gain_ci_low": 0.015,
        },
        "worst": {
            "gain": 0.006,
            "p3_top8": 0.101,
            "p4_random_mean": 0.099,
            "p4_random_std": 0.0014,
            "gain_ci_low": 0.004,
        },
        "missing_lidar": {
            "gain": 0.011,
            "p3_top8": 0.232,
            "p4_random_mean": 0.230,
            "p4_random_std": 0.0022,
            "gain_ci_low": 0.008,
        },
    }
    passed = _probe_scope_success(
        scopes, {"all14": 0.01, "worst": 0.02, "missing_lidar": 0.02}
    )
    assert passed == {"all14": True, "worst": False, "missing_lidar": False}
    assert any(passed.values())
