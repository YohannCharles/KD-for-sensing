import math

import pytest
import torch
import torch.nn as nn

from kd_sensing.engine.batch import model_cfg_consumes_missing_modality_metadata
from kd_sensing.engine.model_output import adapt_model_output
from kd_sensing.models.u_mask_beam_jepa import (
    TEMPORAL_MASK_STATISTIC_NAMES,
    TEMPORAL_SCORER_STATISTIC_NAMES,
    _temporal_scorer_statistics,
    temporal_mask_statistics,
)
from kd_sensing.registries import ENCODERS, MODELS, import_default_components


@ENCODERS.register("s1_lightweight_temporal_test_encoder", force=True)
class S1LightweightTemporalTestEncoder(nn.Module):
    def __init__(self, output_dim: int = 4, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch_size, steps = batch.shape[:2]
        scalar = batch.float().reshape(batch_size, steps, -1).mean(dim=-1, keepdim=True)
        return scalar.expand(batch_size, steps, self.output_dim)


def _config(*, modalities: tuple[str, ...] = ("image", "radar"), **overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "type": "u_mask_beam_jepa",
        "modalities": list(modalities),
        "d_model": 4,
        "num_classes": 4,
        "num_pred": 1,
        "num_heads": 1,
        "num_layers": 1,
        "dropout": 0.0,
        "fusion_type": "supervised_router",
        "head_type": "classifier",
        "bprr_dropout": 0.0,
        "use_teacher": False,
        "use_jepa_loss": False,
        "encoders": {
            name: {"type": "s1_lightweight_temporal_test_encoder", "output_dim": 4}
            for name in modalities
        },
    }
    config.update(overrides)
    return config


def _build(*, modalities: tuple[str, ...] = ("image", "radar"), **overrides: object):
    import_default_components()
    return MODELS.build(_config(modalities=modalities, **overrides))


def _batch(values: dict[str, list[float]]) -> dict[str, torch.Tensor]:
    return {f"{name}_batch": torch.tensor(series, dtype=torch.float32).view(1, -1, 1) for name, series in values.items()}


def _pooling(kind: str, **overrides: object) -> dict[str, object]:
    return {"enabled": True, "type": kind, **overrides}


def test_disabled_temporal_pooling_preserves_state_dict_and_forward() -> None:
    torch.manual_seed(10)
    baseline = _build()
    torch.manual_seed(10)
    disabled = _build(
        temporal_pooling={"enabled": False, "type": "gap_aware_residual"},
        coverage_shrinkage={"enabled": False},
    )
    masked_mean = _build(temporal_pooling=_pooling("masked_mean"))
    assert list(baseline.state_dict()) == list(disabled.state_dict())
    assert list(baseline.state_dict()) == list(masked_mean.state_dict())
    assert not any("temporal_" in key or "coverage_shrinkage" in key for key in disabled.state_dict())

    inputs = _batch({"image": [1, 2, 3], "radar": [4, 5, 6]})
    mask = torch.tensor([[True, False]])
    baseline.eval()
    disabled.eval()
    with torch.no_grad():
        baseline_output = baseline(**inputs, missing_mask=mask)
        disabled_output = disabled(**inputs, missing_mask=mask)
    for key in ("logits", "input_features", "output_features", "modality_reliability"):
        assert torch.equal(baseline_output[key], disabled_output[key])
    assert "temporal_pooling_type" not in disabled_output


@pytest.mark.parametrize("value", [None, "", "none", "NONE"])
def test_retired_temporal_router_none_remains_disabled_equivalent(value: object) -> None:
    baseline = _build()
    compatible = _build(temporal_router_type=value)
    assert list(baseline.state_dict()) == list(compatible.state_dict())
    compatible.load_state_dict(baseline.state_dict())
    baseline.eval()
    compatible.eval()
    inputs = _batch({"image": [1, 2], "radar": [3, 4]})
    with torch.no_grad():
        expected = baseline(**inputs)
        actual = compatible(**inputs)
    assert torch.equal(actual["logits"], expected["logits"])
    assert torch.equal(actual["input_features"], expected["input_features"])


@pytest.mark.parametrize(
    "value",
    ["s1_temporalagg_modality", "s2_pertime_modality", "s3_two_level", "s4_global"],
)
def test_retired_temporal_router_routes_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="temporal_router_type is retired"):
        _build(temporal_router_type=value)


def test_masked_mean_uses_modality_temporal_mask_and_preserves_model_output() -> None:
    model = _build(temporal_pooling=_pooling("masked_mean"))
    inputs = _batch({"image": [1, 2, 3, 4, 5], "radar": [10, 20, 30, 40, 50]})
    mask = torch.tensor(
        [[[1, 0], [0, 1], [1, 0], [0, 0], [1, 1]]],
        dtype=torch.bool,
    )
    output = model(**inputs, modality_temporal_mask=mask)
    expected = torch.tensor([[[3.0] * 4, [35.0] * 4]])

    assert torch.allclose(output["input_features"], expected)
    assert torch.equal(output["modality_temporal_mask"], mask)
    assert torch.equal(output["temporal_mask"], mask.any(dim=2))
    assert torch.equal(output["available_modalities"], mask.any(dim=1))
    assert output["temporal_pooling_type"] == "masked_mean"
    assert output["temporal_pooling_param_count"] == 0
    expected_teacher = model._head_logits(output["output_features"]).unsqueeze(1)
    assert torch.allclose(output["teacher_logits"], expected_teacher)
    assert torch.equal(output["u_star"], output["output_features"])
    assert torch.equal(output["mu_B"], output["output_features"])
    assert torch.equal(output["logvar_B"], torch.zeros_like(output["output_features"]))
    assert torch.equal(output["modality_mu_B"], output["input_features"])
    assert torch.equal(output["modality_logvar_B"], torch.zeros_like(output["input_features"]))
    assert torch.equal(output["global_reliability"], torch.ones(1))
    adapted = adapt_model_output(output)
    assert adapted.logits.shape == (1, 1, 4)
    assert adapted.diagnostics["temporal_mask_statistics"].shape == (1, 2, 5)
    metadata = output["metadata"]
    assert metadata["consumes_missing_modality_metadata"] is True
    assert metadata["temporal_pooling_type"] == "masked_mean"
    assert metadata["total_params"] >= metadata["trainable_params"] > 0


def test_temporal_pooling_config_alone_opts_into_shared_runtime_metadata() -> None:
    assert model_cfg_consumes_missing_modality_metadata(
        {"temporal_pooling": {"enabled": True, "type": "masked_mean"}}
    )
    assert not model_cfg_consumes_missing_modality_metadata(
        {"temporal_pooling": {"enabled": False, "type": "masked_mean"}}
    )


def test_temporal_mask_statistics_exact_gap_and_single_step_boundaries() -> None:
    mask = torch.tensor(
        [
            [
                [1, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 1, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
            ]
        ],
        dtype=torch.bool,
    )
    stats = temporal_mask_statistics(mask)[0]
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.2, 0.0, 2.0 / 3.0],
            [0.6, 0.5, 0.4, 0.4, 1.0 / 3.0],
            [0.0, 1.0, 1.0, 1.0, 1.0 / 3.0],
        ]
    )
    assert torch.allclose(stats, expected)
    assert stats[1, 0] == stats[2, 0]
    assert not torch.equal(stats[1, 2:], stats[2, 2:])
    one_step = temporal_mask_statistics(torch.tensor([[[True, False]]]))[0]
    assert torch.equal(one_step[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert torch.equal(one_step[1], torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0]))
    assert torch.isfinite(stats).all() and torch.isfinite(one_step).all()


def test_fixed_recency_is_parameter_free_hard_masked_and_single_cell_exact() -> None:
    decay = math.log(2.0)
    model = _build(temporal_pooling=_pooling("fixed_recency", recency_decay=decay))
    inputs = _batch({"image": [1, 2, 3], "radar": [10, 20, 30]})
    mask = torch.tensor([[[1, 1], [1, 0], [1, 0]]], dtype=torch.bool)
    output = model(**inputs, modality_temporal_mask=mask)
    expected_image = (0.25 * 1 + 0.5 * 2 + 3) / 1.75

    assert torch.allclose(output["input_features"][0, 0], torch.full((4,), expected_image))
    assert torch.equal(output["input_features"][0, 1], torch.full((4,), 10.0))
    assert torch.all(output["temporal_pooling_weights"][~mask] == 0)
    assert torch.allclose(output["temporal_pooling_weights"].sum(dim=1), torch.ones(1, 2))
    assert output["temporal_pooling_param_count"] == 0
    assert output["temporal_recency_decay"] == pytest.approx(decay)
    model.temporal_pooling_config["recency_decay"] = 1_000.0
    extreme = model(**inputs, modality_temporal_mask=mask)
    assert torch.isfinite(extreme["temporal_pooling_weights"]).all()
    assert torch.equal(extreme["input_features"][0, 1], torch.full((4,), 10.0))


def test_gap_residual_starts_at_masked_mean_has_finite_backward_and_small_budget() -> None:
    model = _build(temporal_pooling=_pooling("gap_aware_residual", hidden_dim=8))
    inputs = _batch({"image": [1, 2, 7, 4], "radar": [10, 20, 30, 40]})
    mask = torch.tensor([[[1, 0], [0, 0], [1, 0], [1, 1]]], dtype=torch.bool)
    output = model(**inputs, modality_temporal_mask=mask)
    expected = torch.tensor([[[4.0] * 4, [40.0] * 4]])

    assert torch.allclose(output["input_features"], expected, atol=0, rtol=0)
    assert torch.all(output["temporal_pooling_weights"][~mask] == 0)
    assert torch.equal(output["temporal_residual_gate"], torch.zeros(2))
    assert 0 < output["temporal_pooling_param_count"] < 30_000
    assert output["metadata"]["temporal_pooling_param_count"] == output["temporal_pooling_param_count"]
    output["logits"].sum().backward()
    for parameter in (
        model.temporal_content_projection.weight,
        model.temporal_statistics_projection.weight,
        model.temporal_score_projection.weight,
        model.temporal_residual_gate,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    empty = torch.zeros_like(mask)
    with pytest.raises(ValueError, match="no available cells"):
        model(**inputs, modality_temporal_mask=empty)


def test_gap_scorer_is_order_and_recency_aware() -> None:
    model = _build(
        modalities=("image",),
        temporal_pooling=_pooling("gap_aware_residual", hidden_dim=1),
    )
    with torch.no_grad():
        model.temporal_content_projection.weight.zero_()
        model.temporal_statistics_projection.weight.zero_()
        model.temporal_statistics_projection.bias.zero_()
        relative_age_index = TEMPORAL_SCORER_STATISTIC_NAMES.index("relative_age")
        model.temporal_statistics_projection.weight[0, relative_age_index] = 2.0
        model.temporal_score_projection.weight.fill_(1.0)
        model.temporal_residual_gate.fill_(1.0)
    mask = torch.ones(1, 3, 1, dtype=torch.bool)
    stats = temporal_mask_statistics(mask)
    ascending = torch.tensor([1.0, 2.0, 8.0]).view(1, 3, 1, 1).expand(-1, -1, -1, 4)
    descending = ascending.flip(1)
    first, first_weights = model._pool_temporal_sequence(ascending, mask, stats)
    second, second_weights = model._pool_temporal_sequence(descending, mask, stats)

    assert torch.equal(first_weights, second_weights)
    assert first_weights[0, 0, 0] > first_weights[0, 2, 0]
    assert not torch.allclose(first, second)
    assert model.temporal_statistics_projection.in_features == 7

    gapped_mask = torch.tensor([[[1], [0], [0], [1], [1]]], dtype=torch.bool)
    cell_stats = _temporal_scorer_statistics(gapped_mask, temporal_mask_statistics(gapped_mask))
    relative_age_index = TEMPORAL_SCORER_STATISTIC_NAMES.index("relative_age")
    previous_gap_index = TEMPORAL_SCORER_STATISTIC_NAMES.index("distance_since_previous_valid")
    assert cell_stats[0, 0, 0, relative_age_index].item() == 1.0
    assert cell_stats[0, 4, 0, relative_age_index].item() == 0.0
    assert cell_stats[0, 3, 0, previous_gap_index].item() == pytest.approx(0.5)
    assert cell_stats[0, 4, 0, previous_gap_index].item() == 0.0


def test_mask_statistics_expand_only_opt_in_supervised_router_features() -> None:
    plain = _build(temporal_pooling=_pooling("masked_mean"))
    stats_model = _build(temporal_pooling=_pooling("masked_mean"), use_mask_statistics=True)
    inputs = _batch({"image": [1, 2, 3], "radar": [10, 20, 30]})
    mask = torch.tensor([[[1, 1], [0, 1], [1, 0]]], dtype=torch.bool)
    plain_output = plain(**inputs, modality_temporal_mask=mask)
    output = stats_model(**inputs, modality_temporal_mask=mask)

    assert plain.bprr_router.feature_dim == 8
    assert stats_model.bprr_router.feature_dim == 13
    assert torch.equal(plain_output["input_features"], output["input_features"])
    features = output["supervised_router_reliability_features"]
    assert features.shape == (1, 2, 13)
    assert torch.allclose(features[..., -5:], output["temporal_mask_statistics"])
    assert output["supervised_router_feature_names"][-5:] == tuple(
        f"temporal_{name}" for name in TEMPORAL_MASK_STATISTIC_NAMES
    )
    assert output["metadata"]["router_mask_statistic_features"] == list(TEMPORAL_MASK_STATISTIC_NAMES)


def test_coverage_shrinkage_is_bounded_normalized_and_clean_safe() -> None:
    model = _build(
        modalities=("image", "radar", "gps"),
        temporal_pooling=_pooling("masked_mean"),
        coverage_shrinkage={"enabled": True, "rho_max": 0.4, "hidden_dim": 4},
    )
    values = {
        "image": [1, 2, 3],
        "radar": [4, 5, 6],
        "gps": [7, 8, 9],
    }
    inputs = {key: value.expand(3, -1, -1).clone() for key, value in _batch(values).items()}
    mask = torch.tensor(
        [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 0], [0, 1, 0], [1, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
        ],
        dtype=torch.bool,
    )
    output = model(**inputs, modality_temporal_mask=mask)
    pre = output["coverage_shrinkage_pre_weights"]
    final = output["supervised_router_gate_weights"]
    uniform = output["coverage_shrinkage_uniform_weights"]
    rho = output["coverage_shrinkage_rho"]
    available = mask.any(dim=1)
    expected = (1.0 - rho.unsqueeze(-1)) * pre + rho.unsqueeze(-1) * uniform

    assert rho[0].item() == 0.0
    assert torch.equal(final[0], pre[0])
    assert torch.all((rho >= 0) & (rho <= 0.4))
    assert torch.all(final[~available] == 0)
    assert torch.allclose(final.sum(dim=1), torch.ones(3))
    assert torch.allclose(final, expected)
    assert final[2, 0].item() == 1.0
    assert torch.all(final[2, 1:] == 0)
    assert output["coverage_shrinkage_mean_coverage"][0].item() == 1.0
    assert output["metadata"]["coverage_shrinkage_rho_max"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"temporal_pooling": _pooling("masked_mean"), "fusion_type": "average"}, "requires fusion_type"),
        ({"use_mask_statistics": True}, "requires temporal_pooling"),
        ({"coverage_shrinkage": {"enabled": True}}, "requires temporal_pooling"),
        ({"temporal_pooling": _pooling("fixed_recency", recency_decay=-1)}, "non-negative"),
        ({"coverage_shrinkage": {"enabled": True, "rho_max": 1.1}}, r"within \[0, 1\]"),
    ],
)
def test_temporal_configuration_rejects_invalid_combinations(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _build(**overrides)
