import json
from pathlib import Path

import pytest

from kd_sensing.data.mmw.twc_router_joint_stress import STATE_CLEAN, STATE_CORRUPT, STATE_DROP
from kd_sensing.data.mmw.twc_router_joint_training import (
    CORRUPTION_SEVERITIES,
    JOINT_RATES,
    MASKS_PER_RATE,
    MASKS_PER_RATE_SEVERITY,
    PANEL_SIZE,
    build_router_joint_training_panel,
    condition_for_global_row,
    load_router_joint_training_panel,
    prepare_router_joint_training_panel,
)


def test_router_joint_training_panel_is_deterministic_and_exactly_balanced() -> None:
    panel = build_router_joint_training_panel()
    replay = build_router_joint_training_panel()

    assert panel == replay
    assert panel["checksum"] == replay["checksum"]
    assert len(panel["conditions"]) == PANEL_SIZE == 240

    for rate in JOINT_RATES:
        rate_conditions = [
            item for item in panel["conditions"] if item["requested_stress_rate"] == rate
        ]
        assert len(rate_conditions) == MASKS_PER_RATE == 60
        assert sorted(item["rate_mask_index"] for item in rate_conditions) == list(range(MASKS_PER_RATE))
        state_per_condition = int(round(20 * rate / 2.0))

        for severity in CORRUPTION_SEVERITIES:
            conditions = [
                item for item in rate_conditions if item["corruption_severity"] == severity
            ]
            assert len(conditions) == MASKS_PER_RATE_SEVERITY == 20
            drop_counts = [0] * 20
            corrupt_counts = [0] * 20
            for condition in conditions:
                assert condition["state_counts"] == {
                    "clean": 20 - 2 * state_per_condition,
                    "drop": state_per_condition,
                    "corrupt": state_per_condition,
                }
                flat = [value for row in condition["state_matrix"] for value in row]
                severity_flat = [value for row in condition["severity_matrix"] for value in row]
                for index, (state, cell_severity) in enumerate(zip(flat, severity_flat)):
                    drop_counts[index] += state == STATE_DROP
                    corrupt_counts[index] += state == STATE_CORRUPT
                    assert cell_severity == (severity if state == STATE_CORRUPT else 0)
                    assert state in {STATE_CLEAN, STATE_DROP, STATE_CORRUPT}
            assert drop_counts == [state_per_condition] * 20
            assert corrupt_counts == [state_per_condition] * 20


def test_router_joint_training_condition_uses_global_row_modulo_panel() -> None:
    panel = build_router_joint_training_panel()

    assert condition_for_global_row(panel, 0) == condition_for_global_row(panel, PANEL_SIZE)
    assert condition_for_global_row(panel, PANEL_SIZE - 1) == condition_for_global_row(
        panel, 2 * PANEL_SIZE - 1
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        condition_for_global_row(panel, -1)
    with pytest.raises(ValueError, match="non-negative integer"):
        condition_for_global_row(panel, 1.5)


def test_router_joint_training_panel_is_immutable_and_checksum_validated(tmp_path: Path) -> None:
    path = tmp_path / "router_joint_training.json"
    panel = prepare_router_joint_training_panel(path)

    assert load_router_joint_training_panel(path) == panel
    assert prepare_router_joint_training_panel(path) == panel
    with pytest.raises(ValueError, match="differs from the frozen request"):
        prepare_router_joint_training_panel(path, seed=20260720)

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["conditions"][0]["state_matrix"][0][0] = STATE_DROP
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_router_joint_training_panel(path)
