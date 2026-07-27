import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import torch

from kd_sensing.baselines.mmw_trajectory import (
    ABTC_METHOD,
    M4_BALANCED_METHOD,
    M4_GENERIC_KL_METHOD,
    M4_UNIFORM_METHOD,
    METHODS,
    TrajectoryBaselineModel,
    abtc_loss,
    availability_balanced_assignment,
    baseline_loss,
    paired_missing_loss,
    random_balanced_assignment,
    topology_smoothed_consistency_loss,
    uniform_mask_assignment,
)
from kd_sensing.data.mmw.trajectory_protocol import (
    AUDIT_IDENTITIES,
    assign_trajectory_groups,
    audit_trajectory_splits,
    protocol_dataset_domains,
    reconstruct_trajectory_groups,
    split_group_counts,
)
from tools.run_mmw_trajectory_baselines import ALL_PATTERNS, _missing_rate_rows


def _row(
    *,
    node: str,
    agent: str,
    frame: int,
    radar: str,
    bs_gps: str,
    group: str = "",
    split: str = "",
) -> dict[str, object]:
    domain = "sunny/scene"
    return {
        "domain_id": domain,
        "condition": "sunny",
        "sensor_scenario": "scene",
        "raw_trajectory_id": node,
        "trajectory_group_id": group,
        "scenario_execution_id": f"{domain}:{node}",
        "agent": agent,
        "window_start_frame": frame,
        "window_end_frame": frame + 5,
        "source_row_sha256": f"row-{node}-{frame}",
        "sample_id": f"sample-{node}-{frame}",
        "target_sample_id": f"target-{node}-{frame}",
        "history_frame_ids_json": json.dumps([frame + offset for offset in range(5)]),
        "future_frame_ids_json": json.dumps([frame + 5]),
        "camera1": f"camera/{node}/{frame}.png",
        "lidar1": f"lidar/{node}/{frame}.npy",
        "radar1": radar,
        "gps1": f"gps/{node}/{frame}.yaml",
        "bs_gps1": bs_gps,
        "future_csi1": f"channel/{node}/{frame}.txt",
        "future_beam_label1": frame % 64,
        "split": split,
    }


def test_shared_rsu_resources_merge_cavs_into_one_trajectory_group() -> None:
    frame = pd.DataFrame(
        [
            _row(node="cav1", agent="1", frame=10, radar="rsu/shared.npy", bs_gps="rsu/shared.yaml"),
            _row(node="cav2", agent="2", frame=20, radar="rsu/shared.npy", bs_gps="rsu/shared.yaml"),
        ]
    )

    groups, annotated = reconstruct_trajectory_groups(frame)

    assert len(groups) == 1
    assert groups[0]["cav_ids"] == "1|2"
    assert annotated["trajectory_group_id"].nunique() == 1


def test_group_count_rules_and_assignment_are_deterministic() -> None:
    assert split_group_counts(50) == (40, 5, 5)
    assert split_group_counts(15) == (12, 2, 1)
    groups = [
        {
            "trajectory_group_id": f"tg_{index}",
            "weather": ("sunny", "rainy", "foggy")[index % 3],
            "scenario": f"scene_{index % 5}",
            "window_count": 100 + index,
        }
        for index in range(15)
    ]
    first, _ = assign_trajectory_groups(groups)
    second, _ = assign_trajectory_groups(groups)
    assert first == second
    assert {role: list(first.values()).count(role) for role in ("train", "validation", "test")} == {
        "train": 12,
        "validation": 2,
        "test": 1,
    }


def test_pairwise_audit_covers_every_identity_and_detects_shared_resource() -> None:
    frames = {
        "train": pd.DataFrame([_row(node="a", agent="1", frame=0, radar="rsu/a.npy", bs_gps="rsu/a.yaml", group="ga")]),
        "validation": pd.DataFrame([_row(node="b", agent="2", frame=20, radar="rsu/b.npy", bs_gps="rsu/b.yaml", group="gb")]),
        "test": pd.DataFrame([_row(node="c", agent="3", frame=40, radar="rsu/c.npy", bs_gps="rsu/c.yaml", group="gc")]),
    }
    passed = audit_trajectory_splits(frames)
    assert passed["status"] == "passed"
    assert all(set(values) == set(AUDIT_IDENTITIES) for values in passed["pairwise_overlaps"].values())

    frames["validation"].loc[0, "radar1"] = frames["train"].loc[0, "radar1"]
    failed = audit_trajectory_splits(frames)
    assert failed["status"] == "failed"
    assert failed["pairwise_overlaps"]["train_vs_validation"]["radar_resource"]["count"] == 1


def test_test_csv_is_sealed_by_default() -> None:
    protocol = {
        "domains": [
            {
                "id": "sunny/train",
                "condition": "sunny",
                "scene": "train",
                "data_root": "/data/sunny",
                "train_split": "/splits/train.csv",
            },
            {
                "id": "rainy/test",
                "condition": "rainy",
                "scene": "test",
                "data_root": "/data/rainy",
                "test_split": "/splits/test.csv",
            },
        ]
    }
    ordinary = protocol_dataset_domains(protocol)
    authorized = protocol_dataset_domains(protocol, allow_test_evaluation=True)
    assert not any("test_csv_name" in domain for domain in ordinary)
    assert authorized[1]["test_csv_name"] == "/splits/test.csv"


@pytest.mark.parametrize("method", METHODS)
def test_m0_m3_heads_and_losses_are_finite(method: str) -> None:
    torch.manual_seed(2026)
    model = TrajectoryBaselineModel(method, d_model=8)
    tokens = {name: torch.randn(2, 5, 8) for name in ("image", "lidar", "radar", "gps")}
    output = model.forward_tokens(tokens)
    loss, _ = baseline_loss(model, output, torch.tensor([1, 2]))
    assert output["logits"].shape == (2, 64)
    assert bool(torch.isfinite(loss))
    assert not hasattr(model, "motion")


def test_random_balanced_assignment_is_exact_and_stable() -> None:
    sample_ids = [f"sample-{index}" for index in range(40)]
    first = random_balanced_assignment(sample_ids)
    second = random_balanced_assignment(sample_ids)
    assert first == second
    assert np.bincount(list(first.values()), minlength=4).tolist() == [10, 10, 10, 10]


def test_all_mask_patterns_and_missing_rate_summary_cover_four_modalities() -> None:
    assert len(ALL_PATTERNS) == 15
    assert len(set(ALL_PATTERNS.values())) == 15
    assert all(any(mask) and len(mask) == 4 for mask in ALL_PATTERNS.values())
    rows = [
        {
            "available_count": sum(mask),
            "sample_count": 10,
            "top1": sum(mask) / 4,
            "adba": 0.5 + sum(mask) / 8,
        }
        for mask in ALL_PATTERNS.values()
    ]

    summary = _missing_rate_rows(METHODS[0], rows)

    assert [row["missing_rate"] for row in summary] == [0.0, 0.25, 0.5, 0.75]
    assert [row["pattern_count"] for row in summary] == [1, 4, 6, 4]
    assert [row["top1_macro"] for row in summary] == pytest.approx([1.0, 0.75, 0.5, 0.25])


def test_abtc_assignment_balances_levels_and_combinations_per_epoch() -> None:
    sample_ids = [f"sample-{index}" for index in range(120)]
    first = availability_balanced_assignment(sample_ids, epoch=1)
    second = availability_balanced_assignment(sample_ids, epoch=2)

    assert Counter(sum(mask) for mask in first.values()) == {1: 40, 2: 40, 3: 40}
    for available_count in (1, 2, 3):
        counts = Counter(mask for mask in first.values() if sum(mask) == available_count)
        assert max(counts.values()) - min(counts.values()) <= 1
    assert first != second
    assert availability_balanced_assignment(sample_ids, epoch=1) == first


def test_m4a_assignment_is_uniform_over_all_14_masks() -> None:
    assigned = uniform_mask_assignment([f"sample-{index}" for index in range(140)], epoch=1)
    counts = Counter(assigned.values())

    assert len(counts) == 14
    assert set(counts.values()) == {10}
    assert Counter(sum(mask) for mask in assigned.values()) == {1: 40, 2: 60, 3: 40}


def test_abtc_topology_consistency_is_detached_and_distance_aware() -> None:
    positions = torch.arange(8)
    direct = (positions[:, None] - positions[None, :]).abs()
    distance = torch.minimum(direct, 8 - direct)
    teacher = torch.full((1, 8), -8.0, requires_grad=True)
    teacher.data[0, 0] = 8.0
    same = teacher.detach().clone().requires_grad_(True)
    near = torch.roll(teacher.detach(), 1, dims=-1)
    far = torch.roll(teacher.detach(), 4, dims=-1)

    same_loss = topology_smoothed_consistency_loss(same, teacher, distance, sigma=0.5)
    near_loss = topology_smoothed_consistency_loss(near, teacher, distance, sigma=0.5)
    far_loss = topology_smoothed_consistency_loss(far, teacher, distance, sigma=0.5)
    same_loss.backward()

    assert abs(float(same_loss.detach())) < 1e-6
    assert float(near_loss.detach()) < float(far_loss.detach())
    assert same.grad is not None
    assert teacher.grad is None


def test_abtc_paired_forward_encodes_once_and_has_finite_loss() -> None:
    torch.manual_seed(2026)
    model = TrajectoryBaselineModel(ABTC_METHOD, d_model=8, dropout=0.0)
    tokens = {name: torch.randn(2, 5, 8) for name in ("image", "lidar", "radar", "gps")}
    calls = 0

    def encode_once(_: object) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        return tokens

    model.encode = encode_once  # type: ignore[method-assign]
    availability = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bool)
    full, masked = model.forward_paired({}, availability)
    positions = torch.arange(64)
    direct = (positions[:, None] - positions[None, :]).abs()
    distance = torch.minimum(direct, 64 - direct)
    loss, report = abtc_loss(model, full, masked, torch.tensor([1, 2]), distance)

    assert calls == 1
    assert full["logits"].shape == masked["logits"].shape == (2, 64)
    assert bool(torch.isfinite(loss))
    assert report["topology_consistency"] >= 0


@pytest.mark.parametrize(
    ("method", "uses_consistency"),
    (
        (M4_UNIFORM_METHOD, False),
        (M4_BALANCED_METHOD, False),
        (M4_GENERIC_KL_METHOD, True),
        (ABTC_METHOD, True),
    ),
)
def test_m4_family_paired_losses_are_finite(method: str, uses_consistency: bool) -> None:
    torch.manual_seed(2026)
    model = TrajectoryBaselineModel(method, d_model=8, dropout=0.0)
    tokens = {name: torch.randn(2, 5, 8) for name in ("image", "lidar", "radar", "gps")}
    full = model.forward_tokens(tokens)
    masked = model.forward_tokens(tokens, availability=torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]]))
    positions = torch.arange(64)
    direct = (positions[:, None] - positions[None, :]).abs()
    loss, report = paired_missing_loss(
        model,
        full,
        masked,
        torch.tensor([1, 2]),
        torch.minimum(direct, 64 - direct),
    )

    assert bool(torch.isfinite(loss))
    assert (report["topology_consistency"] > 0) is uses_consistency
