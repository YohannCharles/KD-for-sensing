from copy import deepcopy

import pytest

from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config


PANEL_SHA256 = "a" * 64
VARIANT_COMPONENTS = {
    "patr": (True, False, False),
    "h2r": (False, True, False),
    "core": (False, False, True),
    "unified_hpr": (True, True, True),
}


def _config(variant: str = "current", supervision: str = "label_topology") -> dict:
    primary = {
        "fusion_type": "supervised_router",
        "head_type": "prototype",
    }
    loss = {
        "enabled": True,
        "use_beam_prototype_alignment": True,
        "router_oracle_weight": 0.1,
    }
    dataset = {"include_router_utility_targets": False}
    if variant != "current":
        primary["router_variant"] = variant
        loss["router_oracle_weight"] = 0.0
        loss["dynamic_router"] = {
            "supervision": supervision,
            "utility_temperature": 0.5,
            "quality_regression_weight": 0.1,
            "fused_utility_weight": 0.2,
            "paired_joint": {
                "enabled": True,
                "panel_path": "outputs/cache/dynamic_router_joint/panel.json",
                "panel_sha256": PANEL_SHA256,
                "monotonic_weight": 0.05,
                "quality_drop_epsilon": 0.01,
            },
        }
        dataset["include_router_utility_targets"] = supervision == "beam_power"
    return {
        "model": {"primary": primary},
        "data": {"dataset": dataset},
        "loss": {"u_mask_beam_jepa": loss},
    }


def test_current_omission_and_explicit_variant_resolve_identically() -> None:
    implicit = _config()
    explicit = deepcopy(implicit)
    explicit["model"]["primary"]["router_variant"] = "current"

    implicit_resolved = u_mask_beam_jepa_config(implicit)
    explicit_resolved = u_mask_beam_jepa_config(explicit)

    assert implicit_resolved == explicit_resolved
    assert implicit_resolved["router_oracle_weight"] == pytest.approx(0.1)
    assert implicit_resolved["router_variant"] == "current"
    assert implicit_resolved["dynamic_router"]["enabled"] is False
    assert not any(implicit_resolved["dynamic_router"]["components"].values())


@pytest.mark.parametrize("variant", VARIANT_COMPONENTS)
@pytest.mark.parametrize("supervision", ("label_topology", "beam_power"))
def test_candidate_matrix_resolves_fixed_components_and_supervision(variant: str, supervision: str) -> None:
    resolved = u_mask_beam_jepa_config(_config(variant, supervision))
    dynamic = resolved["dynamic_router"]
    window, hierarchical, consensus = VARIANT_COMPONENTS[variant]

    assert resolved["router_variant"] == variant
    assert dynamic["enabled"] is True
    assert dynamic["supervision"] == supervision
    assert dynamic["requires_beam_power"] is (supervision == "beam_power")
    assert dynamic["components"] == {
        "prior_anchored_residual": True,
        "window_temporal_evidence": window,
        "hierarchical_cell_gate": hierarchical,
        "consensus_evidence": consensus,
    }
    assert dynamic["frame_rank_weight"] == pytest.approx(0.1 if hierarchical else 0.0)
    assert dynamic["residual_anchor_weight"] == pytest.approx(0.01)
    assert dynamic["paired_joint"] == {
        "enabled": True,
        "panel_path": "outputs/cache/dynamic_router_joint/panel.json",
        "panel_sha256": PANEL_SHA256,
        "corruption_seed": 20260719,
        "monotonic_weight": pytest.approx(0.05),
        "monotonic_margin_scale": pytest.approx(0.25),
        "quality_drop_epsilon": pytest.approx(0.01),
    }


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("unknown_variant", "router_variant"),
        ("current_with_candidate", "must not declare"),
        ("missing_candidate", "dynamic_router mapping"),
        ("classifier", "head_type=prototype"),
        ("uniform", "fusion_type=supervised_router"),
        ("bpa_off", "use_beam_prototype_alignment=true"),
        ("oracle_enabled", "router_oracle_weight=0"),
        ("legacy_pairing", "legacy router_quality_pairing"),
        ("component_boolean", "selected only by"),
        ("label_with_power", "label_topology.*false"),
        ("power_without_power", "beam_power.*true"),
    ),
)
def test_candidate_routes_are_strictly_mutually_exclusive(case: str, match: str) -> None:
    cfg = _config("patr")
    primary = cfg["model"]["primary"]
    loss = cfg["loss"]["u_mask_beam_jepa"]
    dataset = cfg["data"]["dataset"]

    if case == "unknown_variant":
        primary["router_variant"] = "other"
    elif case == "current_with_candidate":
        primary["router_variant"] = "current"
    elif case == "missing_candidate":
        loss.pop("dynamic_router")
    elif case == "classifier":
        primary["head_type"] = "classifier"
    elif case == "uniform":
        primary["fusion_type"] = "uniform_mean"
    elif case == "bpa_off":
        loss["use_beam_prototype_alignment"] = False
    elif case == "oracle_enabled":
        loss["router_oracle_weight"] = 0.1
    elif case == "legacy_pairing":
        loss["router_quality_pairing"] = {"enabled": True, "utility_weight": 0.1}
        loss["dynamic_router"]["supervision"] = "beam_power"
        dataset["include_router_utility_targets"] = True
    elif case == "component_boolean":
        primary["router_use_consensus_evidence"] = True
    elif case == "label_with_power":
        dataset["include_router_utility_targets"] = True
    elif case == "power_without_power":
        loss["dynamic_router"]["supervision"] = "beam_power"

    with pytest.raises(ValueError, match=match):
        u_mask_beam_jepa_config(cfg)


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("unknown_dynamic", "Unknown .*dynamic_router fields"),
        ("unknown_paired", "Unknown .*paired_joint fields"),
        ("bad_supervision", "supervision"),
        ("bad_sha", "panel_sha256"),
        ("paired_disabled", "paired_joint.enabled=true"),
        ("zero_monotonic", "monotonic_weight.*positive"),
        ("zero_epsilon", "quality_drop_epsilon.*positive"),
        ("nonfinite_temperature", "utility_temperature.*finite"),
        ("zero_utility_losses", "positive quality or fused"),
        ("missing_frame_rank", "positive .*frame_rank_weight"),
        ("unexpected_frame_rank", "only valid for H2R"),
        ("bad_corruption_seed", "corruption_seed"),
        ("boolean_weight", "quality_regression_weight.*finite"),
        ("non_string_panel_path", "panel_path"),
        ("non_boolean_power_flag", "include_router_utility_targets must be boolean"),
    ),
)
def test_candidate_nested_config_fails_closed(case: str, match: str) -> None:
    cfg = _config("unified_hpr")
    dynamic = cfg["loss"]["u_mask_beam_jepa"]["dynamic_router"]
    paired = dynamic["paired_joint"]

    if case == "unknown_dynamic":
        dynamic["use_consensus"] = True
    elif case == "unknown_paired":
        paired["condition_id"] = "leak"
    elif case == "bad_supervision":
        dynamic["supervision"] = ["label_topology", "beam_power"]
    elif case == "bad_sha":
        paired["panel_sha256"] = "abc"
    elif case == "paired_disabled":
        paired["enabled"] = False
    elif case == "zero_monotonic":
        paired["monotonic_weight"] = 0.0
    elif case == "zero_epsilon":
        paired["quality_drop_epsilon"] = 0.0
    elif case == "nonfinite_temperature":
        dynamic["utility_temperature"] = float("nan")
    elif case == "zero_utility_losses":
        dynamic["quality_regression_weight"] = 0.0
        dynamic["fused_utility_weight"] = 0.0
    elif case == "missing_frame_rank":
        dynamic["frame_rank_weight"] = 0.0
    elif case == "unexpected_frame_rank":
        cfg["model"]["primary"]["router_variant"] = "patr"
        dynamic["frame_rank_weight"] = 0.1
    elif case == "bad_corruption_seed":
        paired["corruption_seed"] = -1
    elif case == "boolean_weight":
        dynamic["quality_regression_weight"] = True
    elif case == "non_string_panel_path":
        paired["panel_path"] = ["panel.json"]
    elif case == "non_boolean_power_flag":
        cfg["data"]["dataset"]["include_router_utility_targets"] = "false"

    with pytest.raises(ValueError, match=match):
        u_mask_beam_jepa_config(cfg)
