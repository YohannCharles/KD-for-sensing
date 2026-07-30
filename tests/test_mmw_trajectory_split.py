import json

import pandas as pd
from kd_sensing.data.mmw.trajectory_protocol import (
    AUDIT_IDENTITIES,
    assign_trajectory_groups,
    audit_trajectory_splits,
    protocol_dataset_domains,
    reconstruct_trajectory_groups,
    split_group_counts,
)


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
