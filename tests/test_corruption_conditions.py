"""Protocol tests for the restored 45-condition corruption table."""

from __future__ import annotations

import pytest
import torch

from kd_sensing.data.corruption_conditions import (
    CONDITION_IDS,
    CONDITIONS,
    CONDITIONS_BY_ID,
    SEVERITY_SCALARS,
    apply_batch_conditions,
    apply_condition,
    condition_table,
)
from kd_sensing.modalities import MODALITY_ORDER


def _inputs(batch: int = 4, steps: int = 5) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "image_batch": torch.randn(batch, steps, 3, 224, 224, generator=generator),
        "radar_batch": torch.randn(batch, steps, 2, 128, 64, generator=generator),
        "gps_batch": torch.randn(batch, steps, 3, generator=generator),
        "lidar_batch": torch.randn(batch, steps, 3, 224, 224, generator=generator),
    }


def test_condition_inventory_matches_the_restored_manifest() -> None:
    assert len(CONDITIONS) == 45
    assert len(set(CONDITION_IDS)) == 45
    assert CONDITIONS[0].condition_id == "clean"
    for modality in ("image", "radar", "gps", "lidar"):
        entries = [item for item in CONDITIONS if item.modality == modality]
        # three graded families x three levels, plus one_step_stale l2 and missing l4
        assert len(entries) == 11
        assert sum(1 for item in entries if item.corruption_type == "one_step_stale") == 1
        assert sum(1 for item in entries if item.corruption_type == "missing") == 1
        assert sorted({item.severity for item in entries}) == [1, 2, 3, 4]


def test_severity_scalars_match_the_recorded_ladder() -> None:
    assert SEVERITY_SCALARS == {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
    for row in condition_table():
        assert row["severity_scalar"] == SEVERITY_SCALARS[row["severity"]]


@pytest.mark.parametrize("condition_id", [item for item in CONDITION_IDS if item != "clean"])
def test_every_condition_changes_only_its_own_modality(condition_id: str) -> None:
    condition = CONDITIONS_BY_ID[condition_id]
    inputs = _inputs()
    result, forced = apply_batch_conditions(inputs, [condition_id] * 4, seed=7)
    for name in MODALITY_ORDER:
        key = f"{name}_batch"
        if name == condition.modality:
            assert not torch.equal(result[key], inputs[key]), f"{condition_id} left {name} untouched"
        else:
            assert torch.equal(result[key], inputs[key]), f"{condition_id} altered {name}"
    assert torch.isfinite(result[f"{condition.modality}_batch"]).all()
    assert bool(forced[:, MODALITY_ORDER.index(condition.modality)].all()) == condition.forces_missing


def test_clean_is_an_exact_identity() -> None:
    inputs = _inputs()
    result, forced = apply_batch_conditions(inputs, ["clean"] * 4, seed=7)
    for key, value in inputs.items():
        assert torch.equal(result[key], value)
    assert not bool(forced.any())


def test_application_is_reproducible_for_a_fixed_assignment() -> None:
    """The cache is built once over a fixed ordering, so this is the guarantee we rely on."""
    inputs = _inputs()
    ids = ["image_gaussian_blur_l2", "gps_sudden_jump_l3", "image_gaussian_blur_l2", "clean"]
    first, first_forced = apply_batch_conditions(inputs, ids, seed=11)
    second, second_forced = apply_batch_conditions(inputs, ids, seed=11)
    for key in inputs:
        assert torch.equal(first[key], second[key])
    assert torch.equal(first_forced, second_forced)


def test_stochastic_operators_follow_the_seed_and_deterministic_ones_do_not() -> None:
    """Blur is a fixed kernel by design; only sampling operators may track the seed."""
    inputs = _inputs()
    ids = ["image_gaussian_blur_l2", "gps_sudden_jump_l3", "image_gaussian_blur_l2", "clean"]
    first, _ = apply_batch_conditions(inputs, ids, seed=11)
    other, _ = apply_batch_conditions(inputs, ids, seed=12)
    assert not torch.equal(other["gps_batch"], first["gps_batch"])
    assert torch.equal(other["image_batch"], first["image_batch"])


def test_clean_rows_are_untouched_by_other_rows_conditions() -> None:
    inputs = _inputs()
    ids = ["image_gaussian_blur_l2", "gps_sudden_jump_l3", "clean", "clean"]
    result, _ = apply_batch_conditions(inputs, ids, seed=11)
    for key, value in inputs.items():
        assert torch.equal(result[key][2:], value[2:])


def test_one_step_stale_delays_the_sequence() -> None:
    value = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    stale = apply_condition(value, CONDITIONS_BY_ID["gps_one_step_stale_l2"], torch.Generator())
    assert torch.equal(stale[:, 0], value[:, 0])
    assert torch.equal(stale[:, 1:], value[:, :-1])


def test_missing_zeroes_the_modality_and_reports_the_flag() -> None:
    inputs = _inputs()
    result, forced = apply_batch_conditions(inputs, ["radar_missing_l4"] * 4, seed=3)
    assert torch.count_nonzero(result["radar_batch"]) == 0
    assert bool(forced[:, MODALITY_ORDER.index("radar")].all())
    assert not bool(forced[:, MODALITY_ORDER.index("image")].any())


def test_severity_increases_the_perturbation_for_graded_families() -> None:
    inputs = _inputs()
    for prefix in ("image_gaussian_blur", "radar_detection_dropout", "gps_white_position_noise", "lidar_point_dropout"):
        magnitudes = []
        for level in (1, 2, 3):
            condition_id = f"{prefix}_l{level}"
            key = f"{CONDITIONS_BY_ID[condition_id].modality}_batch"
            result, _ = apply_batch_conditions(inputs, [condition_id] * 4, seed=5)
            magnitudes.append(float((result[key] - inputs[key]).abs().mean()))
        assert magnitudes == sorted(magnitudes), f"{prefix} is not monotone in severity: {magnitudes}"


def test_unknown_condition_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown corruption condition"):
        apply_batch_conditions(_inputs(), ["image_not_a_real_condition_l1"] * 4, seed=1)
