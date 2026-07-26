"""Protocol tests for the frozen-U0 Router observability screen.

The decisive property is that the cached representations replay the frozen U0
forward exactly: encoders are mask independent, so one clean pass per sample can
serve every mask.  These tests build a synthetic U0 that reproduces the audited
hook module paths, so they exercise the real contract without the real
checkpoint or dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from kd_sensing.baselines import router_observability as ro
from kd_sensing.modalities import MODALITY_ORDER
from kd_sensing.models.router_quality_branch import (
    ARMS,
    PROTOTYPE_STATE_KEYS,
    RouterObservabilityModel,
    uses_quality_branch,
)
from kd_sensing.registries import ENCODERS, MODELS

import kd_sensing.models.u_mask_beam_jepa  # noqa: F401


PREPROJECTION_DIMS = {"image": 20, "radar": 12, "gps": 12, "lidar": 20}
D_MODEL = 8
NUM_CLASSES = 8
STEPS = 3


class _Body(nn.Module):
    """Flatten [B,T,...] into [B*T, D_pre] so the final layer matches U0's hooks."""

    def __init__(self, input_dim: int, preprojection_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(input_dim), int(preprojection_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, steps = value.shape[:2]
        return self.linear(value.reshape(batch * steps, -1))


@ENCODERS.register("router_test_projection", force=True)
class _ProjectionEncoder(nn.Module):
    """Mirrors TinyViT: the final linear sits at ``projection.1``."""

    def __init__(self, *, output_dim: int = D_MODEL, input_dim: int = 3 * 4 * 4, preprojection_dim: int = 20, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.body = _Body(input_dim, preprojection_dim)
        self.projection = nn.Sequential(nn.LayerNorm(int(preprojection_dim)), nn.Linear(int(preprojection_dim), self.output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, steps = value.shape[:2]
        return self.projection(self.body(value)).view(batch, steps, self.output_dim)


@ENCODERS.register("router_test_fc_layer", force=True)
class _FcLayerEncoder(nn.Module):
    """Mirrors the radar CNN: the final linear sits at ``fc_layer.9``."""

    def __init__(self, *, output_dim: int = D_MODEL, input_dim: int = 2 * 4 * 4, preprojection_dim: int = 12, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.body = _Body(input_dim, preprojection_dim)
        layers: list[nn.Module] = []
        for _index in range(3):
            layers += [nn.Linear(int(preprojection_dim), int(preprojection_dim)), nn.ReLU(), nn.Dropout(0.0)]
        layers.append(nn.Linear(int(preprojection_dim), self.output_dim))
        self.fc_layer = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, steps = value.shape[:2]
        return self.fc_layer(self.body(value)).view(batch, steps, self.output_dim)


@ENCODERS.register("router_test_net", force=True)
class _NetEncoder(nn.Module):
    """Mirrors the GPS MLP: the final linear sits at ``net.4``."""

    def __init__(self, *, output_dim: int = D_MODEL, input_dim: int = 3, preprojection_dim: int = 12, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.body = _Body(input_dim, preprojection_dim)
        self.net = nn.Sequential(
            nn.Linear(int(preprojection_dim), int(preprojection_dim)),
            nn.LayerNorm(int(preprojection_dim)),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(int(preprojection_dim), self.output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, steps = value.shape[:2]
        return self.net(self.body(value)).view(batch, steps, self.output_dim)


def _build_model() -> nn.Module:
    torch.manual_seed(0)
    model = MODELS.build(
        {
            "type": "u_mask_beam_jepa",
            "modalities": list(MODALITY_ORDER),
            "d_model": D_MODEL,
            "num_classes": NUM_CLASSES,
            "num_pred": 1,
            "seq_length": STEPS,
            "dropout": 0.0,
            "fusion_type": "supervised_router",
            "head_type": "prototype",
            "temporal_pooling": {"enabled": True, "type": "masked_mean"},
            "encoders": {
                "image": {"type": "router_test_projection", "output_dim": D_MODEL, "input_dim": 48, "preprojection_dim": 20},
                "radar": {"type": "router_test_fc_layer", "output_dim": D_MODEL, "input_dim": 32, "preprojection_dim": 12},
                "gps": {"type": "router_test_net", "output_dim": D_MODEL, "input_dim": 3, "preprojection_dim": 12},
                "lidar": {"type": "router_test_projection", "output_dim": D_MODEL, "input_dim": 48, "preprojection_dim": 20},
            },
        }
    )
    return model.eval()


def _inputs(batch: int = 6) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    return {
        "image_batch": torch.randn(batch, STEPS, 3, 4, 4),
        "radar_batch": torch.randn(batch, STEPS, 2, 4, 4),
        "gps_batch": torch.randn(batch, STEPS, 3),
        "lidar_batch": torch.randn(batch, STEPS, 3, 4, 4),
    }


def _capture(model: nn.Module, inputs: dict[str, torch.Tensor], batch: int):
    with ro.EncoderCapture(model) as capture, torch.no_grad():
        model(**inputs)
        return capture.collect(batch, STEPS)


# -- hook contract ---------------------------------------------------------


def test_hook_targets_resolve_to_linear_layers_with_the_audited_widths() -> None:
    model = _build_model()
    modules = ro.resolve_hook_modules(model)
    assert set(modules) == set(MODALITY_ORDER)
    assert ro.preprojection_dims(model) == PREPROJECTION_DIMS
    for module in modules.values():
        assert isinstance(module, nn.Linear)
        assert module.out_features == D_MODEL


def test_hook_resolution_fails_closed_when_a_target_is_not_linear() -> None:
    model = _build_model()
    model.encoders["gps"].net[4] = nn.Identity()
    with pytest.raises(ValueError, match="is not an nn.Linear"):
        ro.resolve_hook_modules(model)


def test_captured_features_are_mask_independent() -> None:
    model = _build_model()
    inputs = _inputs()
    latent_clean, preproj_clean = _capture(model, inputs, 6)
    masked = dict(inputs)
    with ro.EncoderCapture(model) as capture, torch.no_grad():
        model(**masked, missing_mask=ro.expand_mask((1, 0, 1, 0), 6, torch.device("cpu")))
        latent_masked, preproj_masked = capture.collect(6, STEPS)
    assert torch.allclose(latent_clean, latent_masked, atol=0.0)
    for name in MODALITY_ORDER:
        assert torch.allclose(preproj_clean[name], preproj_masked[name], atol=0.0)


# -- cache storage precision -----------------------------------------------


def test_cache_storage_round_trips_without_changing_a_single_bit() -> None:
    """float32 storage is what makes the equivalence gate an exact check."""
    value = torch.randn(64, 5, 32)
    packed = ro.pack_cache_array(value)
    assert packed.dtype == np.float32
    restored = ro.unpack_cache_array(packed, torch.device("cpu"))
    assert torch.equal(restored, value)
    assert torch.equal(ro.quantize_for_cache(value), value)


def test_cache_storage_survives_magnitudes_reduced_precision_would_destroy() -> None:
    """float16 saturates at 65504 and bfloat16 keeps only 7 mantissa bits."""
    value = torch.tensor([[1e5, 7e4, 1e-5, -3e5, 1.0000001]], dtype=torch.float32)
    restored = ro.unpack_cache_array(ro.pack_cache_array(value), torch.device("cpu"))
    assert torch.equal(restored, value)
    assert not torch.isfinite(value.to(torch.float16)).all()
    assert not torch.equal(value.to(torch.bfloat16).float(), value)


# -- the decisive equivalence property -------------------------------------


@pytest.mark.parametrize("pattern", [pattern for _, pattern in ro.MASK_PATTERNS])
def test_cached_replay_reproduces_the_live_frozen_forward(pattern: tuple[int, ...]) -> None:
    model = _build_model()
    inputs = _inputs()
    latent, preprojection = _capture(model, inputs, 6)
    head = ro.FrozenU0Head(model)
    mask = ro.expand_mask(pattern, 6, torch.device("cpu"))
    with torch.no_grad():
        live = model(**inputs, missing_mask=mask)
    replayed = head(latent, preprojection, mask)
    reference = head.reference_logits(replayed)
    assert torch.allclose(reference, live["logits"][:, 0, :], atol=1e-5)
    assert torch.allclose(replayed["latent"], live["modality_features"], atol=1e-5)
    assert torch.allclose(replayed["unimodal_logits"], live["unimodal_logits"], atol=1e-5)
    assert torch.allclose(replayed["scalars"], live["supervised_router_reliability_features"], atol=1e-5)


def test_replay_exposes_the_prototype_state_and_pooled_preprojection() -> None:
    model = _build_model()
    latent, preprojection = _capture(model, _inputs(), 6)
    head = ro.FrozenU0Head(model)
    replayed = head(latent, preprojection, ro.expand_mask((1, 1, 1, 1), 6, torch.device("cpu")))
    for key in PROTOTYPE_STATE_KEYS:
        assert replayed["prototype_state"][key].shape == (6, len(MODALITY_ORDER))
    for name in MODALITY_ORDER:
        assert replayed["preprojection"][name].shape == (6, PREPROJECTION_DIMS[name])


def test_dense_temporal_guard_accepts_the_adapter_style_batch() -> None:
    ro.assert_dense_temporal_inputs(_inputs())
    ro.assert_dense_temporal_inputs({**_inputs(), "available_modalities": torch.ones(6, 4, dtype=torch.bool)})


def test_dense_temporal_guard_fails_closed_on_a_dataset_supplied_temporal_mask() -> None:
    mask = torch.ones(6, STEPS, len(MODALITY_ORDER), dtype=torch.bool)
    mask[0, 1, 2] = False
    with pytest.raises(ValueError, match="dense temporal inputs"):
        ro.assert_dense_temporal_inputs({**_inputs(), "modality_temporal_mask": mask})


def test_replay_fails_closed_when_a_sample_has_no_available_modality() -> None:
    model = _build_model()
    latent, preprojection = _capture(model, _inputs(), 6)
    head = ro.FrozenU0Head(model)
    empty = torch.zeros(6, len(MODALITY_ORDER), dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one available temporal cell"):
        head(latent, preprojection, empty)


# -- arm construction ------------------------------------------------------


def _arm_model(arm: str) -> RouterObservabilityModel:
    torch.manual_seed(0)
    return RouterObservabilityModel(arm, scalar_feature_count=5, preprojection_dims=PREPROJECTION_DIMS)


def test_arms_form_a_strictly_nested_input_ladder() -> None:
    counts = {arm: _arm_model(arm).feature_count for arm in ARMS}
    assert counts["q0"] < counts["q1"] < counts["q2"]
    assert counts["q1"] + 8 == counts["q2"]
    assert counts["q2"] == counts["q3"]


def test_q3_has_exactly_the_same_parameter_count_as_q2() -> None:
    # Strict equality: this is what removes the "q2 only won because it has more
    # parameters" explanation.
    assert _arm_model("q2").parameter_count() == _arm_model("q3").parameter_count()
    assert _arm_model("q0").parameter_count() < _arm_model("q2").parameter_count()


def test_only_q2_and_q3_build_a_quality_branch() -> None:
    for arm in ARMS:
        model = _arm_model(arm)
        assert (model.quality_branch is not None) == uses_quality_branch(arm)


def test_q3_permutation_changes_the_router_input_but_not_the_parameters() -> None:
    model = _build_model()
    latent, preprojection = _capture(model, _inputs(), 6)
    head = ro.FrozenU0Head(model)
    replayed = head(latent, preprojection, ro.expand_mask((1, 1, 1, 1), 6, torch.device("cpu")))
    arm = _arm_model("q3")
    identity = torch.arange(6)
    permuted = torch.as_tensor(ro.cross_sample_permutation(6, seed=1, tag="unit"))
    with torch.no_grad():
        straight = arm.router_features(replayed["scalars"], replayed["prototype_state"], replayed["preprojection"], permutation=identity)
        shuffled = arm.router_features(replayed["scalars"], replayed["prototype_state"], replayed["preprojection"], permutation=permuted)
    assert straight.shape == shuffled.shape
    assert not torch.allclose(straight, shuffled)


def test_q3_requires_an_explicit_permutation() -> None:
    model = _build_model()
    latent, preprojection = _capture(model, _inputs(), 6)
    replayed = ro.FrozenU0Head(model)(latent, preprojection, ro.expand_mask((1, 1, 1, 1), 6, torch.device("cpu")))
    with pytest.raises(ValueError, match="explicit cross-sample permutation"):
        _arm_model("q3").router_features(replayed["scalars"], replayed["prototype_state"], replayed["preprojection"])


def test_inference_ablation_requires_a_fitted_train_only_mean() -> None:
    model = _build_model()
    latent, preprojection = _capture(model, _inputs(), 6)
    replayed = ro.FrozenU0Head(model)(latent, preprojection, ro.expand_mask((1, 1, 1, 1), 6, torch.device("cpu")))
    arm = _arm_model("q2")
    with pytest.raises(ValueError, match="train-only mean embedding"):
        arm.router_features(replayed["scalars"], replayed["prototype_state"], replayed["preprojection"], ablate_quality=True)
    arm.set_quality_mean(torch.zeros_like(arm.quality_mean))
    ablated = arm.router_features(replayed["scalars"], replayed["prototype_state"], replayed["preprojection"], ablate_quality=True)
    assert ablated.shape[-1] == arm.feature_count
    # Ablation replaces the per-sample embedding with one shared vector.
    assert torch.allclose(ablated[:, :, -8:], ablated[0:1, :, -8:].expand(6, -1, -1))


# -- deterministic schedules ----------------------------------------------


def test_mask_schedule_is_deterministic_and_covers_every_canonical_mask() -> None:
    ids = [f"sample:{index:05d}" for index in range(4000)]
    first = ro.mask_schedule(ids, 0)
    assert np.array_equal(first, ro.mask_schedule(ids, 0))
    assert not np.array_equal(first, ro.mask_schedule(ids, 1))
    assert sorted(set(first.tolist())) == list(range(len(ro.MASK_PATTERNS)))
    assert len(ro.MASK_PATTERNS) == 15 and len(ro.NON_FULL_KEYS) == 14


def test_condition_draw_is_deterministic_and_covers_the_table() -> None:
    ids = [f"sample:{index:05d}" for index in range(6000)]
    drawn = ro.draw_conditions(ids)
    assert drawn == ro.draw_conditions(ids)
    assert len(set(drawn)) == 45


def test_cross_sample_permutation_leaves_no_fixed_point() -> None:
    permutation = ro.cross_sample_permutation(64, seed=3, tag="unit")
    assert sorted(permutation.tolist()) == list(range(64))
    assert not np.any(permutation == np.arange(64))


def test_cross_sample_permutation_stays_a_bijection_over_many_tags() -> None:
    """One tag is not enough: the fixed-point repair only collides on some draws.

    A q3 batch that duplicates one sample and drops another silently stops being a
    capacity control, so this sweeps the tags the training loop actually uses.
    """
    reference = np.arange(256)
    for epoch in range(40):
        for start in range(0, 5000, 256):
            permutation = ro.cross_sample_permutation(256, seed=1, tag=f"train:{epoch}:{start}")
            assert sorted(permutation.tolist()) == list(range(256)), f"train:{epoch}:{start} is not a bijection"
            assert not np.any(permutation == reference)


@pytest.mark.parametrize("count", [2, 3, 4, 5, 8, 33])
def test_cross_sample_permutation_holds_for_small_and_odd_batches(count: int) -> None:
    for seed in range(60):
        permutation = ro.cross_sample_permutation(count, seed=seed, tag="tail-batch")
        assert sorted(permutation.tolist()) == list(range(count))
        assert not np.any(permutation == np.arange(count))


def test_cross_sample_permutation_is_the_identity_only_for_a_single_row() -> None:
    assert ro.cross_sample_permutation(1, seed=0, tag="unit").tolist() == [0]


# -- gates -----------------------------------------------------------------


def _summaries(q2: list[float], q1: list[float], q3: list[float]) -> dict[str, list[dict[str, float]]]:
    def rows(values: list[float]) -> list[dict[str, float]]:
        return [
            {"full_top1": value, "all14_top1": value, "all14_within3": value, "all14_mae": 1.0 - value}
            for value in values
        ]

    return {"q0": rows(q1), "q1": rows(q1), "q2": rows(q2), "q3": rows(q3)}


def test_gates_pass_only_when_q2_separates_from_both_controls() -> None:
    gates = ro.evaluate_gates(_summaries([0.30, 0.31, 0.32], [0.20, 0.21, 0.22], [0.10, 0.11, 0.12]))
    assert ro.direction_survives(gates)
    assert {row["gate"] for row in gates} >= {"q2_beats_q1_full_top1", "q2_beats_q3_all14_top1"}


def test_gates_fail_when_seed_ranges_overlap() -> None:
    gates = ro.evaluate_gates(_summaries([0.30, 0.21, 0.32], [0.20, 0.25, 0.22], [0.10, 0.11, 0.12]))
    assert not ro.direction_survives(gates)


def test_gates_fail_when_the_capacity_control_matches_the_treatment() -> None:
    gates = ro.evaluate_gates(_summaries([0.30, 0.31, 0.32], [0.20, 0.21, 0.22], [0.30, 0.31, 0.32]))
    assert not ro.direction_survives(gates)


def test_non_regression_gate_rejects_a_worse_mae() -> None:
    summaries = _summaries([0.30, 0.31, 0.32], [0.20, 0.21, 0.22], [0.10, 0.11, 0.12])
    for row in summaries["q2"]:
        row["all14_mae"] = 9.0
    gates = ro.evaluate_gates(summaries)
    assert not ro.direction_survives(gates)
    assert any(row["gate"] == "q2_no_regression_all14_mae" and not row["passed"] for row in gates)


def test_run_inventory_is_two_settings_by_four_arms_by_three_seeds() -> None:
    runs = ro.all_runs()
    assert len(runs) == 2 * 4 * 3
    assert len({run.key for run in runs}) == len(runs)
    assert ro.ROUTER_SEEDS == (1, 2, 3)


# --------------------------------------------------------------------------
# corruption-forced removals (setting C only)
# --------------------------------------------------------------------------


def _synthetic_cache(forced: np.ndarray) -> ro.RepresentationCache:
    count = forced.shape[0]
    payload: dict[str, np.ndarray] = {
        "sample_id": np.array([f"s{i:03d}" for i in range(count)]),
        "domain": np.array(["clear"] * count),
        "weather": np.array(["clear"] * count),
        "condition": np.array(["none"] * count),
        "label": np.arange(count, dtype=np.int64),
        "latent_sequence": np.zeros((count, 5, len(MODALITY_ORDER), 8), dtype=np.float32),
        "forced_missing": forced,
    }
    for name, dim in PREPROJECTION_DIMS.items():
        payload[f"preprojection_{name}"] = np.zeros((count, 5, dim), dtype=np.float32)
    return ro.RepresentationCache(payload, torch.device("cpu"))


def test_available_never_returns_an_all_zero_mask() -> None:
    """Pooling over an empty mask is undefined, so setting C must not produce one."""
    forced = np.array(
        [[bool(index >> shift & 1) for shift in range(len(MODALITY_ORDER))] for index in range(16)]
    )
    cache = _synthetic_cache(forced)
    index = torch.arange(len(cache))
    for _, pattern in ro.MASK_PATTERNS:
        available = cache.available(index, pattern)
        assert bool(available.any(dim=1).all()), f"mask {pattern} left a sample with nothing"


def test_available_restores_only_modalities_the_schedule_allowed() -> None:
    """The rescue must stay inside the pattern.

    Reviving a hard-masked modality would give setting C an easier mask
    distribution than setting N, and the two settings are only comparable
    because they share one schedule.
    """
    forced = np.array(
        [[bool(index >> shift & 1) for shift in range(len(MODALITY_ORDER))] for index in range(16)]
    )
    cache = _synthetic_cache(forced)
    index = torch.arange(len(cache))
    for _, pattern in ro.MASK_PATTERNS:
        scheduled = ro.expand_mask(pattern, len(cache), torch.device("cpu"))
        available = cache.available(index, pattern)
        assert bool((available & ~scheduled).sum() == 0), f"mask {pattern} revived a masked modality"


def test_available_removes_corrupted_modalities_whenever_something_survives() -> None:
    """The rescue is a last resort, not a blanket restoration."""
    forced = np.array([[True, False, False, False]])
    cache = _synthetic_cache(forced)
    available = cache.available(torch.arange(1), (1, 1, 0, 0))
    assert available.tolist() == [[False, True, False, False]]


def test_available_accepts_a_per_sample_pattern_tensor() -> None:
    """The training loop passes one row per sample, not one shared pattern."""
    forced = np.array([[True, True, False, False], [False, False, False, False]])
    cache = _synthetic_cache(forced)
    patterns = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.bool)
    available = cache.available(torch.arange(2), patterns)
    assert available[0].tolist() == [True, False, False, False]
    assert available[1].tolist() == [False, False, True, False]
