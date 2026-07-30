import numpy as np
import pytest
import torch

from kd_sensing.baselines.mmw_trajectory import ABTC_METHOD, TrajectoryBaselineModel
from kd_sensing.diagnostics.paired_geometry import (
    binary_probe_metrics,
    classification_groups,
    cosine_knn,
    decision_decomposition,
    fit_logistic_probe,
    linear_cka,
    minimal_interpolation_alpha,
    nonempty_subset_utilities,
    predict_logistic_probe,
    predictive_statistics,
    representation_spectrum,
    signed_cycle_offset,
    validate_pair_alignment,
    validate_safety_contract,
    validate_train_only_selection,
)
from kd_sensing.diagnostics.prototype_deformation import MASKS


def test_group_definition_covers_all_full_missing_outcomes() -> None:
    target = torch.tensor([1, 1, 1, 1])
    full = torch.tensor([1, 1, 0, 0])
    missing = torch.tensor([1, 0, 0, 1])
    assert classification_groups(full, missing, target).squeeze(1).tolist() == [0, 1, 2, 3]


def test_signed_cycle_offset_uses_audited_label_order() -> None:
    order = (2, 0, 3, 1)
    target = torch.tensor([2, 2, 0, 1])
    prediction = torch.tensor([0, 1, 2, 3])
    assert signed_cycle_offset(target, prediction, order).tolist() == [1, -1, -1, -1]


def test_predictive_statistics_uses_target_vs_best_other_margin() -> None:
    logits = torch.full((2, 64), -5.0)
    logits[0, 3] = 2.0
    logits[0, 4] = 1.25
    logits[1, 7] = 0.5
    logits[1, 8] = 1.5
    values = predictive_statistics(logits, torch.tensor([3, 7]))
    assert values["target_margin"].tolist() == pytest.approx([0.75, -1.0])
    assert values["target_rank"].tolist() == [1, 2]


def test_decision_projection_identity_matches_scaled_cosine_scores() -> None:
    generator = torch.Generator().manual_seed(11)
    prototypes = torch.randn(64, 8, generator=generator)
    full = torch.randn(6, 8, generator=generator)
    missing = full + 0.2 * torch.randn(6, 8, generator=generator)
    bank = torch.nn.functional.normalize(prototypes)
    full_logits = torch.nn.functional.normalize(full) @ bank.t() / 0.1
    missing_logits = torch.nn.functional.normalize(missing) @ bank.t() / 0.1
    target = full_logits.argmax(dim=1)
    values = decision_decomposition(full, missing, full_logits, missing_logits, target, prototypes)
    assert float(values["identity_absolute_error"].max()) < 2e-5
    noisy = decision_decomposition(full, missing, full_logits, missing_logits.round(decimals=1), target, prototypes)
    assert float(noisy["identity_absolute_error"].max()) < 2e-5
    assert float(noisy["production_identity_absolute_error"].max()) > 1e-3
    assert torch.allclose(
        values["parallel_energy_ratio"] + values["orthogonal_vector"].square().sum(dim=1) / (
            torch.nn.functional.normalize(missing).sub(torch.nn.functional.normalize(full)).square().sum(dim=1).clamp_min(1e-12)
        ),
        torch.ones(6),
        atol=1e-5,
    )


def test_effective_rank_distinguishes_line_and_isotropic_features() -> None:
    line = torch.arange(1, 65, dtype=torch.float32)[:, None] * torch.tensor([[1.0, 2.0, -1.0]])
    isotropic = torch.eye(8).repeat(8, 1)
    assert representation_spectrum(line)["effective_rank"] == pytest.approx(1.0, abs=1e-4)
    assert representation_spectrum(isotropic)["effective_rank"] > 6.5


def test_linear_cka_is_one_for_orthogonal_feature_rotation() -> None:
    generator = torch.Generator().manual_seed(12)
    features = torch.randn(32, 5, generator=generator)
    rotation, _ = torch.linalg.qr(torch.randn(5, 5, generator=generator))
    assert linear_cka(features, features @ rotation) == pytest.approx(1.0, abs=1e-5)


def test_counterfactual_interpolation_finds_first_recovery_step() -> None:
    prototypes = torch.eye(4)
    missing = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    full = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    alpha = minimal_interpolation_alpha(missing, full, torch.tensor([0]), prototypes, steps=100)
    assert alpha.item() == pytest.approx(0.5)


def test_cosine_knn_excludes_self() -> None:
    values = torch.eye(4)
    scores, indices = cosine_knn(values, values, k=1, exclude_self=True)
    assert not torch.any(indices.squeeze(1).eq(torch.arange(4)))
    assert torch.allclose(scores, torch.zeros_like(scores))


def test_logistic_probe_is_train_only_and_predictive() -> None:
    rng = np.random.default_rng(13)
    features = rng.normal(size=(300, 3)).astype(np.float32)
    labels = (features[:, 0] + 0.3 * features[:, 1] > 0).astype(np.int8)
    state = fit_logistic_probe(features, labels, epochs=8, batch_size=64, seed=13)
    _, probability = predict_logistic_probe(features, state)
    metrics = binary_probe_metrics(labels, probability)
    assert metrics["roc_auc"] > 0.95
    assert metrics["pr_auc"] > labels.mean()


def test_formal_selection_rejects_validation_inputs() -> None:
    validate_train_only_selection({"source_roles": ["train"], "validation_leakage_oracle": False})
    with pytest.raises(ValueError, match="non-train"):
        validate_train_only_selection({"source_roles": ["train", "validation"], "validation_leakage_oracle": False})
    with pytest.raises(ValueError, match="validation_leakage_oracle"):
        validate_train_only_selection({"source_roles": ["train"], "validation_leakage_oracle": True})


def test_pair_and_safety_guards_fail_closed() -> None:
    validate_pair_alignment(["a", "b"], torch.zeros(2, 15, 64))
    with pytest.raises(ValueError):
        validate_pair_alignment(["a", "a"], torch.zeros(2, 15, 64))
    validate_safety_contract(
        {
            "csi_used": False,
            "channel_input_used": False,
            "f1_used": False,
            "outer_test_accessed": False,
            "future_beam_power_role": "label_side_evaluation_metric_only",
        }
    )
    with pytest.raises(ValueError):
        validate_safety_contract({"csi_used": True})


def test_nonempty_subset_utility_never_invents_empty_mask() -> None:
    bits = torch.tensor(list(MASKS.values()))
    marginal, interaction = nonempty_subset_utilities(torch.zeros(3, 15), bits)
    assert len(marginal) == 28
    assert interaction
    assert all(bits[row["base"]].any() for row in marginal + interaction)


def test_fusion_hooks_do_not_change_frozen_output() -> None:
    torch.manual_seed(7)
    model = TrajectoryBaselineModel(ABTC_METHOD, dropout=0.0).eval()
    tokens = {name: torch.randn(2, 5, 64) for name in ("image", "lidar", "radar", "gps")}
    availability = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 0]], dtype=torch.bool)
    reference = model.forward_tokens(tokens, availability=availability)["logits"]
    captured: list[torch.Tensor] = []
    handle = model.fusion[4].register_forward_hook(lambda _module, _inputs, output: captured.append(output.detach()))
    hooked = model.forward_tokens(tokens, availability=availability)["logits"]
    handle.remove()
    assert captured[0].shape == (2, 512)
    torch.testing.assert_close(hooked, reference, rtol=0, atol=0)
