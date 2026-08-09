import json
from pathlib import Path

import pandas as pd
import pytest

from kd_sensing.config import load_config
from kd_sensing.data.mmw import trajectory_protocol as protocol_module
from kd_sensing.data.mmw.trajectory_protocol import (
    ASSIGNMENT_ALGORITHM,
    CACHE_IDENTITY_FIELDS,
    DEFAULT_BLOCK_SIZE,
    EXPECTED_WEATHERS,
    SPLIT_RATIOS,
    SPLIT_ROLES,
    TRAJECTORY_PROTOCOL_MODE,
    TRAJECTORY_MANIFEST_VERSION,
    assign_mmw_blocks_stratified,
    bind_trajectory_config,
    build_trajectory_protocol,
    load_trajectory_protocol,
    protocol_dataset_domains,
    trajectory_manifest_path,
    validate_mmw_id_block_split,
    validate_split_cache_identity,
    validate_trajectory_protocol,
)


SCENES = tuple(f"scene_{index}" for index in range(5))
CAVS = ("cav_1", "cav_2")
BLOCK_SIZE = 8
BASE_FRAMES = BLOCK_SIZE * 6
HISTORY_SPAN = 5
SAMPLE_SPAN = HISTORY_SPAN + 1


def _label(scene_index: int, cav_index: int, base_index: int) -> int:
    return (7 * (base_index // BLOCK_SIZE) + 3 * scene_index + cav_index) % 16


def _write_source_indexes(dataset_root: Path) -> None:
    weather_offsets = {"foggy": 1_000, "rainy": 3_000, "sunny": 5_000}
    for weather in EXPECTED_WEATHERS:
        for scene_index, scene in enumerate(SCENES):
            sequence_rows = []
            frame_rows = []
            for cav_index, cav in enumerate(CAVS):
                physical_start = weather_offsets[weather] + scene_index * 500 + cav_index * 100
                source_sequence_start = cav_index * (BASE_FRAMES - SAMPLE_SPAN + 1)
                for base_index in range(BASE_FRAMES):
                    frame_id = physical_start + base_index
                    frame_rows.append(
                        {
                            "agent": cav,
                            "condition": weather,
                            "sensor_scenario": scene,
                            "frame_id": frame_id,
                            "beam_label": _label(scene_index, cav_index, base_index),
                            "sample_id": f"{weather}:{scene}:{cav}:{frame_id}",
                        }
                    )
                for window_start in range(BASE_FRAMES - SAMPLE_SPAN + 1):
                    sequence_rows.append(
                        _source_row(
                            weather,
                            scene,
                            scene_index,
                            cav,
                            cav_index,
                            physical_start,
                            source_sequence_start,
                            window_start,
                        )
                    )
            sequence_path = (
                dataset_root
                / weather
                / "Prepared"
                / scene
                / "splits"
                / "h5p1_strict_v2"
                / "all_sequences.csv"
            )
            sequence_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(sequence_rows).to_csv(sequence_path, index=False)
            frame_path = dataset_root / weather / "Prepared" / scene / "manifests" / "frame_manifest.csv"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(frame_rows).to_csv(frame_path, index=False)


def _source_row(
    weather: str,
    scene: str,
    scene_index: int,
    cav: str,
    cav_index: int,
    physical_start: int,
    source_sequence_start: int,
    window_start: int,
) -> dict[str, object]:
    history = [physical_start + window_start + offset for offset in range(HISTORY_SPAN)]
    future = [physical_start + window_start + HISTORY_SPAN]
    row: dict[str, object] = {
        "agent": cav,
        "condition": weather,
        "sensor_scenario": scene,
        "contiguous_segment_id": f"{weather}:{scene}:{cav}:segment_0000",
        "seq_index": source_sequence_start + window_start,
        "sample_id": f"{weather}:{scene}:{cav}:{history[-1]}",
        "target_sample_id": f"{weather}:{scene}:{cav}:{future[0]}",
        "history_frame_ids_json": json.dumps(history),
        "future_frame_ids_json": json.dumps(future),
        "future_beam_label1": _label(scene_index, cav_index, window_start + HISTORY_SPAN),
    }
    for offset, frame_id in enumerate(history, start=1):
        root = f"Sensor_Data/{scene}/{cav}/{frame_id:06d}"
        row[f"camera{offset}"] = f"{root}_camera0.png"
        row[f"lidar{offset}"] = f"{root}.pcd"
        row[f"gps{offset}"] = f"{root}.yaml"
        row[f"csi{offset}"] = f"Channel_Data/{scene}/{cav}/{frame_id:06d}_paths.npz"
        row[f"beam{offset}"] = f"Prepared/{scene}/beam_power/{cav}/{frame_id:06d}.txt"
    return row


@pytest.fixture(scope="module")
def built_protocol(tmp_path_factory) -> tuple[Path, dict[str, object]]:
    root = tmp_path_factory.mktemp("mmw-id-block")
    dataset_root = root / "MMW"
    output_root = root / "outputs"
    _write_source_indexes(dataset_root)
    protocol = build_trajectory_protocol(
        output_root,
        dataset_root=dataset_root,
        split_seed=0,
        block_size=BLOCK_SIZE,
    )
    return trajectory_manifest_path(output_root, 0), protocol


def _all_records(protocol: dict[str, object]) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_csv(domain[f"{role}_split"], na_values="").fillna("")
            for domain in protocol["domains"]
            for role in SPLIT_ROLES
        ],
        ignore_index=True,
    )


def _blocks(protocol: dict[str, object]) -> list[dict[str, object]]:
    return [block for role in SPLIT_ROLES for block in protocol[f"{role}_blocks"]]


def test_seed_zero_manifest_is_byte_identical_and_order_independent(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "MMW"
    output_root = tmp_path / "outputs"
    _write_source_indexes(dataset_root)
    build_trajectory_protocol(output_root, dataset_root=dataset_root, split_seed=0, block_size=BLOCK_SIZE)
    path = trajectory_manifest_path(output_root, 0)
    first = path.read_bytes()
    original_discover = protocol_module._discover_sources

    def reversed_discover(root: Path):
        sequences, frames = original_discover(root)
        return list(reversed(sequences)), list(reversed(frames))

    monkeypatch.setattr(protocol_module, "_discover_sources", reversed_discover)
    build_trajectory_protocol(
        output_root,
        dataset_root=dataset_root,
        split_seed=0,
        block_size=BLOCK_SIZE,
        regenerate=True,
    )

    assert path.read_bytes() == first


def test_different_split_seeds_change_assignment_and_keep_hard_constraints(built_protocol) -> None:
    _path, protocol = built_protocol
    blocks = _blocks(protocol)
    assignments = [assign_mmw_blocks_stratified(blocks, split_seed=seed)[0] for seed in (0, 1)]

    assert assignments[0] != assignments[1]
    for assignment in assignments:
        counts = {role: list(assignment.values()).count(role) for role in SPLIT_ROLES}
        assert counts["train"] > max(counts["validation"], counts["test"])
        for scene in SCENES:
            for cav in CAVS:
                ids = [block["block_id"] for block in blocks if block["scene_id"] == scene and block["cav_id"] == cav]
                assert {assignment[block_id] for block_id in ids} == set(SPLIT_ROLES)


def test_block_base_weather_raw_frame_and_window_invariants_pass(built_protocol) -> None:
    _path, protocol = built_protocol
    audit = validate_mmw_id_block_split(protocol, _all_records(protocol))

    assert audit["status"] == "passed"
    assert all(audit["checks"].values())
    assert all(value == 0 for value in audit["raw_frame_overlap_counts"].values())
    assert audit["window_errors"] == []


def test_every_split_covers_all_scenes_and_trajectories(built_protocol) -> None:
    _path, protocol = built_protocol
    for role in SPLIT_ROLES:
        blocks = protocol[f"{role}_blocks"]
        assert {block["scene_id"] for block in blocks} == set(SCENES)
        assert {(block["scene_id"], block["cav_id"]) for block in blocks} == {
            (scene, cav) for scene in SCENES for cav in CAVS
        }


def test_split_ratios_are_close_and_stratification_beats_contiguous_baseline(built_protocol) -> None:
    _path, protocol = built_protocol
    statistics = protocol["statistics"]

    for measure in ("blocks", "base_frames", "weather_samples", "windows"):
        ratios = statistics["ratios"][measure]
        assert abs(ratios["train_ratio"] - SPLIT_RATIOS["train"]) <= 0.05
        assert abs(ratios["validation_ratio"] - SPLIT_RATIOS["validation"]) <= 0.05
        assert abs(ratios["test_ratio"] - SPLIT_RATIOS["test"]) <= 0.05
    assert statistics["stratified_not_worse_than_contiguous"] is True
    assert statistics["label_tv_sum"] < statistics["simple_contiguous_label_tv_sum"]
    assert statistics["conditional_label_distribution"]["per_domain"]["group_count"] == len(EXPECTED_WEATHERS) * len(SCENES)
    assert statistics["conditional_label_distribution"]["per_scene"]["group_count"] == len(SCENES)
    assert statistics["conditional_label_distribution"]["per_trajectory"]["group_count"] == len(SCENES) * len(CAVS)
    assert protocol["manifest_version"] == TRAJECTORY_MANIFEST_VERSION
    assert protocol["assignment_algorithm"] == ASSIGNMENT_ALGORITHM


def test_assignment_objective_detects_conditional_shift_hidden_by_global_histogram() -> None:
    blocks = []
    for scene, cav in (("scene_a", "cav_1"), ("scene_b", "cav_1")):
        for index, label in enumerate((0, 0, 0, 1, 1, 1)):
            histogram = [0] * 64
            histogram[label] = 10
            blocks.append(
                {
                    "block_id": f"{scene}::{cav}::{index}",
                    "scene_id": scene,
                    "cav_id": cav,
                    "block_start_base_index": index * 8,
                    "num_windows_estimated": 10,
                    "window_beam_histogram": histogram,
                }
            )

    def assignment(scene_a_roles, scene_b_roles):
        return {
            f"{scene}::cav_1::{index}": role
            for scene, roles in (("scene_a", scene_a_roles), ("scene_b", scene_b_roles))
            for index, role in enumerate(roles)
        }

    balanced = assignment(
        ("train", "train", "validation", "train", "train", "test"),
        ("train", "train", "test", "train", "train", "validation"),
    )
    shifted = assignment(
        ("train", "train", "train", "train", "validation", "test"),
        ("train", "validation", "test", "train", "train", "train"),
    )

    balanced_score = protocol_module._assignment_objective(blocks, balanced)
    shifted_score = protocol_module._assignment_objective(blocks, shifted)

    assert balanced_score["label_distribution_error"] == pytest.approx(
        shifted_score["label_distribution_error"]
    )
    assert balanced_score["per_trajectory_label_distribution_error"] < shifted_score[
        "per_trajectory_label_distribution_error"
    ]
    assert balanced_score["objective"] < shifted_score["objective"]


def test_weather_and_window_corruption_are_rejected(built_protocol) -> None:
    _path, protocol = built_protocol
    records = _all_records(protocol)
    broken_window = records.copy()
    broken_window.loc[0, "future_frame_ids_json"] = json.dumps([99_999])
    with pytest.raises(ValueError, match="window_crossing_or_order"):
        validate_mmw_id_block_split(protocol, broken_window)

    broken_role = records.copy()
    broken_role.loc[0, "split"] = "validation" if records.loc[0, "split"] == "train" else "train"
    with pytest.raises(ValueError, match="record_role"):
        validate_mmw_id_block_split(protocol, broken_role)


def test_legacy_manifest_and_cache_are_rejected(built_protocol) -> None:
    _path, protocol = built_protocol
    with pytest.raises(ValueError, match="Legacy manifests must be regenerated"):
        validate_trajectory_protocol(
            {"protocol": "mmw_trajectory_disjoint", "mode": "mmw_trajectory_disjoint"}
        )
    stale_manifest = dict(protocol)
    stale_manifest["manifest_version"] = TRAJECTORY_MANIFEST_VERSION - 1
    with pytest.raises(ValueError, match="manifest version mismatch"):
        validate_trajectory_protocol(stale_manifest, verify_sources=False)
    stale_assignment = dict(protocol)
    stale_assignment["assignment_algorithm"] = "deterministic_multistart_greedy_swap_v1"
    with pytest.raises(ValueError, match="assignment algorithm mismatch"):
        validate_trajectory_protocol(stale_assignment, verify_sources=False)
    with pytest.raises(ValueError, match="Legacy or stale"):
        validate_split_cache_identity({}, protocol)
    identity = protocol_module.split_cache_identity(protocol)
    assert set(identity) == set(CACHE_IDENTITY_FIELDS)
    validate_split_cache_identity(identity, protocol)


def test_changed_source_requires_explicit_regeneration(tmp_path: Path) -> None:
    dataset_root = tmp_path / "MMW"
    output_root = tmp_path / "outputs"
    _write_source_indexes(dataset_root)
    build_trajectory_protocol(output_root, dataset_root=dataset_root, split_seed=0, block_size=BLOCK_SIZE)
    source = next(dataset_root.glob("*/Prepared/*/manifests/frame_manifest.csv"))
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing or changed"):
        build_trajectory_protocol(output_root, dataset_root=dataset_root, split_seed=0, block_size=BLOCK_SIZE)


def test_binding_defaults_to_train_validation_without_loading_test(built_protocol, monkeypatch) -> None:
    path, _protocol = built_protocol
    observed_roles = []
    original = protocol_module._load_materialized_splits

    def observe_roles(protocol, *, roles=SPLIT_ROLES):
        observed_roles.append(tuple(roles))
        return original(protocol, roles=roles)

    monkeypatch.setattr(protocol_module, "_load_materialized_splits", observe_roles)
    cfg = {
        "experiment": {"seed": 0, "train_seed": 0},
        "runtime": {"evaluate_test_requested": False},
        "data": {
            "split_protocol": TRAJECTORY_PROTOCOL_MODE,
            "split_seed": 0,
            "block_size": BLOCK_SIZE,
            "split_ratios": dict(SPLIT_RATIOS),
            "dataset": {},
        },
        "training": {},
    }

    bind_trajectory_config(cfg, path)

    assert observed_roles and all(roles == ("train", "validation") for roles in observed_roles)
    assert cfg["training"]["final_test"] == {"enabled": False}
    assert cfg["data_protocol"]["test_evaluated"] is False
    assert cfg["data_protocol"]["outer_test_enabled"] is False
    assert cfg["data_protocol"]["allow_confirmation_train"] is False
    assert all("test_csv_name" in domain for domain in protocol_dataset_domains(_protocol))


def test_explicit_test_binding_loads_test_and_records_authorization(built_protocol, monkeypatch) -> None:
    path, _protocol = built_protocol
    observed_roles = []
    original = protocol_module._load_materialized_splits

    def observe_roles(protocol, *, roles=SPLIT_ROLES):
        observed_roles.append(tuple(roles))
        return original(protocol, roles=roles)

    monkeypatch.setattr(protocol_module, "_load_materialized_splits", observe_roles)
    cfg = {
        "experiment": {"seed": 0, "train_seed": 0},
        "runtime": {"evaluate_test_requested": True},
        "data": {
            "split_protocol": TRAJECTORY_PROTOCOL_MODE,
            "split_seed": 0,
            "block_size": BLOCK_SIZE,
            "split_ratios": dict(SPLIT_RATIOS),
            "dataset": {},
        },
        "training": {},
    }

    bind_trajectory_config(cfg, path)

    assert observed_roles and all(roles == SPLIT_ROLES for roles in observed_roles)
    assert cfg["training"]["final_test"] == {"enabled": True}
    assert cfg["data_protocol"]["test_evaluated"] is True
    assert cfg["data_protocol"]["outer_test_enabled"] is True
    assert cfg["data_protocol"]["allow_confirmation_train"] is False


def test_split_seed_and_train_seed_are_independent(built_protocol) -> None:
    path, protocol = built_protocol
    fingerprints = []
    for train_seed in (0, 17):
        cfg = {
            "experiment": {"seed": train_seed, "train_seed": train_seed},
            "runtime": {"evaluate_test_requested": False},
            "data": {
                "split_protocol": TRAJECTORY_PROTOCOL_MODE,
                "split_seed": 0,
                "block_size": BLOCK_SIZE,
                "split_ratios": dict(SPLIT_RATIOS),
                "dataset": {},
            },
            "training": {},
        }
        bind_trajectory_config(cfg, path)
        fingerprints.append(cfg["data_protocol"]["protocol_fingerprint"])
        assert cfg["data_protocol"]["train_seed"] == train_seed

    assert fingerprints == [protocol["protocol_fingerprint"]] * 2


def test_too_few_blocks_and_weather_misalignment_fail_without_fallback(tmp_path: Path) -> None:
    short_blocks = [
        {
            "block_id": f"scene_0::cav_1::{index}",
            "scene_id": "scene_0",
            "cav_id": "cav_1",
            "block_start_base_index": index * 8,
            "num_windows_estimated": 3,
            "window_beam_histogram": [3] + [0] * 63,
        }
        for index in range(2)
    ]
    with pytest.raises(ValueError, match="at least three blocks.*scene_0/cav_1=2"):
        assign_mmw_blocks_stratified(short_blocks, split_seed=0)

    dataset_root = tmp_path / "MMW"
    _write_source_indexes(dataset_root)
    frame_path = dataset_root / "rainy" / "Prepared" / SCENES[0] / "manifests" / "frame_manifest.csv"
    frame = pd.read_csv(frame_path)
    frame.loc[(frame["agent"] == CAVS[0]) & (frame.index == 0), "beam_label"] = 63
    frame.to_csv(frame_path, index=False)
    with pytest.raises(ValueError, match="weather copies disagree on beam labels"):
        build_trajectory_protocol(
            tmp_path / "outputs",
            dataset_root=dataset_root,
            split_seed=0,
            block_size=BLOCK_SIZE,
        )


def test_canonical_config_uses_seed_zero_block_protocol() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/mmw/u0.yaml")

    assert cfg["data"]["split_protocol"] == TRAJECTORY_PROTOCOL_MODE
    assert cfg["data"]["split_seed"] == cfg["experiment"]["train_seed"] == 0
    assert cfg["data"]["block_size"] == DEFAULT_BLOCK_SIZE == 32
    assert cfg["data"]["split_ratios"] == SPLIT_RATIOS
    assert cfg["training"]["final_test"] == {"enabled": False}


def test_manifest_reuse_validates_all_sources_and_splits(built_protocol) -> None:
    path, protocol = built_protocol

    loaded = load_trajectory_protocol(path)

    assert loaded["protocol_fingerprint"] == protocol["protocol_fingerprint"]
