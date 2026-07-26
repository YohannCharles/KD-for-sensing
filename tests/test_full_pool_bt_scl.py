import json
import importlib.util
from pathlib import Path

import pytest
import torch

from kd_sensing.baselines.full_pool_bt_scl import (
    BTSCLModel,
    PATTERNS,
    check_missing_token_invariance,
    coarse_to_fine_loss,
    generate_nested_schedule,
    hierarchical_sector_loss,
    load_audited_topology,
    monotonicity_loss,
    schedule_masks,
    stochastic_dominance_loss,
)


_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_full_pool_bt_scl",
    Path(__file__).parents[1] / "tools/run_full_pool_bt_scl.py",
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(_RUNNER)


def _topology(tmp_path: Path):
    manifest = tmp_path / "topology_manifest.json"
    table = tmp_path / "topology_table.csv"
    edges = tmp_path / "topology_edges.csv"
    table.write_text("label,phase_coordinate\n" + "\n".join(f"{index},{index / 64}" for index in range(64)) + "\n")
    edges.write_text("left_label,right_label\n" + "\n".join(f"{index},{(index + 1) % 64}" for index in range(64)) + "\n")
    manifest.write_text(json.dumps({"descriptor": {"topology_id": "ula_dft_phase_cycle_v1", "num_beams": 64, "num_antennas": 64}, "descriptor_sha256": "a" * 64, "domain_count": 15, "metadata_consistent": True, "domains": [{"metadata_status": "verified"} for _ in range(15)]}))
    return load_audited_topology(manifest)


def _model_inputs():
    return {
        "image": torch.randn(2, 2, 3, 32, 32),
        "lidar": torch.randn(2, 2, 3, 32, 32),
        "radar": torch.randn(2, 2, 2, 32, 32),
        "gps": torch.randn(2, 2, 3),
    }


def test_nested_schedule_balances_and_builds_strict_views():
    ids = [f"sample-{index}" for index in range(48)]
    schedule = generate_nested_schedule(ids, seed=2026, split="train")
    masks = schedule_masks(ids[:2], schedule, torch.device("cpu"))
    assert masks.shape == (2, 4, 4)
    assert masks.sum(dim=-1).tolist() == [[1, 2, 3, 4], [1, 2, 3, 4]]
    assert all(value == 12 for value in schedule["balance_counts"]["single_start"].values())
    assert len(schedule["balance_counts"]["double_subset"]) == 6
    assert len(schedule["balance_counts"]["triple_subset"]) == 4


def test_model_handles_all_patterns_and_strict_missing_invariance():
    model = BTSCLModel(d_model=32, seq_len=2)
    inputs = _model_inputs()
    tokens = model.encode(inputs)
    for values in PATTERNS.values():
        availability = torch.tensor(values, dtype=torch.bool).expand(2, -1)
        logits, masked = model.logits_from_tokens(tokens, availability)
        assert logits.shape == (2, 64)
        assert torch.isfinite(logits).all()
        assert torch.equal(masked[:, :, ~availability[0]], torch.zeros_like(masked[:, :, ~availability[0]]))
    check_missing_token_invariance(model, inputs, torch.tensor(PATTERNS["missing_image"], dtype=torch.bool).expand(2, -1))
    with pytest.raises(ValueError, match="all-missing"):
        model.logits_from_tokens(tokens, torch.zeros((2, 4), dtype=torch.bool))


def test_topology_losses_use_phase_cycle_and_stop_at_zero_when_satisfied(tmp_path: Path):
    topology = _topology(tmp_path)
    assert topology.distance[0, 63].item() == 1
    logits = torch.zeros(2, 4, 64)
    labels = torch.tensor([0, 1])
    assert monotonicity_loss(logits, labels, topology).item() == 0.0
    c2f, detail = coarse_to_fine_loss(logits, labels, topology)
    assert torch.isfinite(c2f)
    assert set(detail) == {"sector_4", "sector_8", "sector_16", "local"}


def test_r6_losses_are_label_anchored_and_measure_nested_dominance(tmp_path: Path):
    topology = _topology(tmp_path)
    labels = torch.tensor([0, 1])
    uniform = torch.zeros(2, 4, 64)
    hierarchy_uniform, _ = hierarchical_sector_loss(uniform, labels, topology)
    anchored = uniform.clone()
    anchored[:, :3].scatter_(2, labels[:, None, None].expand(-1, 3, 1), 12.0)
    hierarchy_anchored, detail = hierarchical_sector_loss(anchored, labels, topology)
    assert hierarchy_anchored < hierarchy_uniform
    assert set(detail) == {"sector_4", "sector_8", "sector_16"}

    dominance_equal, _ = stochastic_dominance_loss(uniform, labels, topology)
    degraded = uniform.clone()
    degraded[:, 0].scatter_(1, labels[:, None], 12.0)
    dominance_degraded, radius_detail = stochastic_dominance_loss(degraded, labels, topology)
    assert dominance_equal.item() == pytest.approx(0.0)
    assert dominance_degraded > dominance_equal
    assert set(radius_detail) == {"radius_0", "radius_3", "radius_5"}


def test_topology_manifest_requires_verified_complete_cycle(tmp_path: Path):
    topology = _topology(tmp_path)
    assert topology.num_beams == 64
    manifest = Path(topology.manifest_path)
    payload = json.loads(manifest.read_text())
    payload["domains"][0]["metadata_status"] = "failed"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unverified"):
        load_audited_topology(manifest)


def test_selection_loss_preserves_equal_missing_group_weighting():
    values = {
        "full": 1.0,
        "missing_image": 2.0,
        "missing_lidar": 2.0,
        "missing_radar": 2.0,
        "missing_gps": 2.0,
        "missing_image_lidar": 3.0,
        "missing_image_radar": 3.0,
        "missing_image_gps": 3.0,
        "missing_lidar_radar": 3.0,
        "missing_lidar_gps": 3.0,
        "missing_radar_gps": 3.0,
        "only_image": 4.0,
        "only_lidar": 4.0,
        "only_radar": 4.0,
        "only_gps": 4.0,
    }
    patterns = {name: {"ce_loss": value} for name, value in values.items()}
    assert _RUNNER._selection_loss(patterns) == pytest.approx(2.5)


def test_vectorized_metric_groups_sum_to_full_batch(tmp_path: Path):
    topology = _topology(tmp_path)
    logits = torch.randn(6, 64)
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    values = _RUNNER._metric_values(logits, labels, topology)
    full = _RUNNER._new_bucket()
    left = _RUNNER._new_bucket()
    right = _RUNNER._new_bucket()
    _RUNNER._metric_update_values(full, values)
    _RUNNER._metric_update_values(left, values[:2])
    _RUNNER._metric_update_values(right, values[2:])
    assert full["count"] == left["count"] + right["count"]
    for key in full:
        if key != "count":
            assert full[key] == pytest.approx(left[key] + right[key])
