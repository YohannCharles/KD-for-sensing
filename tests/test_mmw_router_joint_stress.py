import json
from pathlib import Path

import pytest

from kd_sensing.data.mmw.twc_router_joint_stress import (
    JOINT_RATES,
    MASKS_PER_RATE,
    STATE_CORRUPT,
    STATE_DROP,
    build_router_joint_stress_cache,
    load_router_joint_stress_cache,
    prepare_router_joint_stress_cache,
)


def test_joint_stress_cache_is_deterministic_unique_and_exactly_balanced():
    cache = build_router_joint_stress_cache(seed=20260719)
    replay = build_router_joint_stress_cache(seed=20260719)

    assert cache == replay
    assert cache["checksum"] == replay["checksum"]
    assert len(cache["conditions"]) == 1 + len(JOINT_RATES) * MASKS_PER_RATE == 81
    assert len({item["state_digest"] for item in cache["conditions"]}) == 81

    clean, *stressed_conditions = cache["conditions"]
    assert clean["state_counts"] == {"clean": 20, "drop": 0, "corrupt": 0}

    audits = {item["requested_stress_rate"]: item for item in cache["rate_balance_audit"]}
    for rate in JOINT_RATES:
        conditions = [item for item in stressed_conditions if item["requested_stress_rate"] == rate]
        state_count = int(round(20 * rate / 2))
        expected_counts = {
            "clean": 20 - 2 * state_count,
            "drop": state_count,
            "corrupt": state_count,
        }
        assert len(conditions) == MASKS_PER_RATE
        for condition in conditions:
            matrix = condition["state_matrix"]
            assert len(matrix) == 5
            assert all(len(row) == 4 for row in matrix)
            assert condition["state_counts"] == expected_counts
            assert [value != STATE_DROP for row in matrix for value in row] == [
                value for row in condition["modality_temporal_mask"] for value in row
            ]

        matrices = [item["state_matrix"] for item in conditions]
        per_cell_drop = [
            sum(matrix[time][modality] == STATE_DROP for matrix in matrices)
            for time in range(5)
            for modality in range(4)
        ]
        per_cell_corrupt = [
            sum(matrix[time][modality] == STATE_CORRUPT for matrix in matrices)
            for time in range(5)
            for modality in range(4)
        ]
        per_cell_clean = [
            MASKS_PER_RATE - drop - corrupt for drop, corrupt in zip(per_cell_drop, per_cell_corrupt)
        ]
        per_modality_drop = [sum(per_cell_drop[modality::4]) for modality in range(4)]
        per_modality_corrupt = [sum(per_cell_corrupt[modality::4]) for modality in range(4)]
        per_frame_drop = [sum(per_cell_drop[time * 4 : (time + 1) * 4]) for time in range(5)]
        per_frame_corrupt = [sum(per_cell_corrupt[time * 4 : (time + 1) * 4]) for time in range(5)]

        assert set(per_cell_drop) == set(per_cell_corrupt) == {state_count}
        assert set(per_cell_clean) == {MASKS_PER_RATE - 2 * state_count}
        assert set(per_modality_drop) == set(per_modality_corrupt) == {state_count * 5}
        assert set(per_frame_drop) == set(per_frame_corrupt) == {state_count * 4}
        assert audits[rate]["per_cell_drop_counts"] == per_cell_drop
        assert audits[rate]["per_cell_corrupt_counts"] == per_cell_corrupt
        assert audits[rate]["per_cell_clean_counts"] == per_cell_clean
        assert audits[rate]["per_modality_drop_counts"] == per_modality_drop
        assert audits[rate]["per_modality_corrupt_counts"] == per_modality_corrupt
        assert audits[rate]["per_frame_drop_counts"] == per_frame_drop
        assert audits[rate]["per_frame_corrupt_counts"] == per_frame_corrupt


def test_joint_stress_cache_load_rejects_tampering(tmp_path: Path):
    path = tmp_path / "fixed_state_cache.json"
    cache = prepare_router_joint_stress_cache(path, seed=20260719)
    original = cache["conditions"][1]["state_matrix"][0][0]
    cache["conditions"][1]["state_matrix"][0][0] = (original + 1) % 3
    path.write_text(json.dumps(cache), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_router_joint_stress_cache(path)


def test_joint_stress_cache_path_is_immutable_across_seeds(tmp_path: Path):
    path = tmp_path / "fixed_state_cache.json"
    first = prepare_router_joint_stress_cache(path, seed=20260719)

    assert prepare_router_joint_stress_cache(path, seed=20260719) == first
    with pytest.raises(ValueError, match="differs from the frozen request"):
        prepare_router_joint_stress_cache(path, seed=20260720)
