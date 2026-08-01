import hashlib
import importlib.util
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _analysis_module():
    path = ROOT / "tools" / "analyze_mmw_trajectory_dataset.py"
    spec = importlib.util.spec_from_file_location("analyze_mmw_trajectory_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYSIS = _analysis_module()


def _frame(role: str, labels: list[int], group: str, *, domain_id: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "domain_id": domain_id or f"domain-{role}",
            "sample_id": [f"{role}-{index}" for index in range(len(labels))],
            "trajectory_group_id": group,
            "split": role,
            "future_beam_label1": labels,
            "beam_label": labels,
            "agent": "cav-1",
            "seq_index": range(len(labels)),
            "window_frame_ids_json": [json.dumps(list(range(index, index + 6))) for index in range(len(labels))],
        }
    )


def _write_bound_protocol(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    domain_id = "domain"
    trajectory_id = "scene-a::cav-1"
    train = _frame("train", [63, 0, 0], trajectory_id, domain_id=domain_id)
    validation = _frame("validation", [4, 4], trajectory_id, domain_id=domain_id)
    test = _frame("test", [7], trajectory_id, domain_id=domain_id)
    paths = {
        "train": tmp_path / "train.csv",
        "validation": tmp_path / "validation.csv",
        "test": tmp_path / "test.csv",
    }
    train.to_csv(paths["train"], index=False)
    validation.to_csv(paths["validation"], index=False)
    test.to_csv(paths["test"], index=False)
    source_indexes: list[dict[str, object]] = []
    window_config = {"history_span": 5, "future_span": 1, "sample_span": 6}
    role_frames = {"train": train, "validation": validation, "test": test}
    blocks = {
        role: [
            {
                "block_id": f"scene-a::cav-1::block-{role}",
                "scene_id": "scene-a",
                "cav_id": "cav-1",
                "block_start_base_index": index * 128,
                "block_end_base_index": index * 128 + 127,
                "num_base_samples": 128,
                "num_windows_actual": len(role_frames[role]),
                "split": role,
            }
        ]
        for index, role in enumerate(("train", "validation", "test"))
    }
    protocol = {
        "dataset": "MMW",
        "protocol": ANALYSIS.PROTOCOL_ID,
        "mode": ANALYSIS.TRAJECTORY_PROTOCOL_MODE,
        "protocol_id": ANALYSIS.PROTOCOL_ID,
        "protocol_version": ANALYSIS.TRAJECTORY_PROTOCOL_VERSION,
        "manifest_version": ANALYSIS.TRAJECTORY_MANIFEST_VERSION,
        "split_seed": ANALYSIS.TRAJECTORY_SPLIT_SEED,
        "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "block_size": 128,
        "trajectory_key": ["scene_id", "cav_id"],
        "base_sample_key": ["scene_id", "cav_id", "base_frame_index"],
        "weather_binding": True,
        "expected_weathers": ["foggy", "rainy", "sunny"],
        "dataset_root": str(tmp_path / "data"),
        "source_indexes": source_indexes,
        "data_source_hash": hashlib.sha256(b"[]").hexdigest(),
        "window_config": window_config,
        "window_config_hash": hashlib.sha256(
            json.dumps(window_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "train_blocks": blocks["train"],
        "validation_blocks": blocks["validation"],
        "test_blocks": blocks["test"],
        "train_block_count": 1,
        "validation_block_count": 1,
        "test_block_count": 1,
        "train_role": "train",
        "validation_role": "validation",
        "test_role": "test",
        "test_evaluated": False,
        "candidate_window_count": len(train) + len(validation) + len(test),
        "materialized_window_count": len(train) + len(validation) + len(test),
        "train_window_count": len(train),
        "validation_window_count": len(validation),
        "test_window_count": len(test),
        "split_hashes": {
            "train": ANALYSIS._split_sample_id_hash(train),
            "validation": ANALYSIS._split_sample_id_hash(validation),
            "test": ANALYSIS._split_sample_id_hash(test),
        },
        "domains": [
            {
                "id": domain_id,
                "data_root": str(tmp_path / "data"),
                "train_split": str(paths["train"]),
                "train_csv_sha256": hashlib.sha256(paths["train"].read_bytes()).hexdigest(),
                "train_sample_count": len(train),
                "validation_split": str(paths["validation"]),
                "validation_csv_sha256": hashlib.sha256(paths["validation"].read_bytes()).hexdigest(),
                "validation_sample_count": len(validation),
                "test_split": str(paths["test"]),
                "test_csv_sha256": hashlib.sha256(paths["test"].read_bytes()).hexdigest(),
                "test_sample_count": len(test),
            },
        ],
    }
    fingerprint = ANALYSIS._protocol_fingerprint(protocol)
    protocol["protocol_fingerprint"] = fingerprint
    protocol_path = (
        tmp_path
        / "splits"
        / ANALYSIS.TRAJECTORY_PROTOCOL_MODE
        / f"seed_{ANALYSIS.TRAJECTORY_SPLIT_SEED}.json"
    )
    audit_path = protocol_path.with_name(f"{protocol_path.stem}_audit.json")
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    audit = {
        "schema_version": 1,
        "audit_id": ANALYSIS.AUDIT_ID,
        "status": "passed",
        "failures": [],
        "checks": {"block_overlap": True, "weather_copy_overlap": True, "window_crossing": True},
        "protocol": ANALYSIS.PROTOCOL_ID,
        "protocol_version": ANALYSIS.TRAJECTORY_PROTOCOL_VERSION,
        "manifest_version": ANALYSIS.TRAJECTORY_MANIFEST_VERSION,
        "split_seed": ANALYSIS.TRAJECTORY_SPLIT_SEED,
        "block_size": 128,
        "protocol_fingerprint": fingerprint,
        "split_manifest_hash": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "data_source_hash": protocol["data_source_hash"],
        "window_config_hash": protocol["window_config_hash"],
        "weather_binding": True,
        "test_evaluated": False,
        "train_sample_count": len(train),
        "validation_sample_count": len(validation),
        "test_sample_count": len(test),
        "train_sample_id_hash": ANALYSIS._sample_id_hash(train),
        "validation_sample_id_hash": ANALYSIS._sample_id_hash(validation),
        "test_sample_id_hash": ANALYSIS._sample_id_hash(test),
        "block_counts": {"train": 1, "validation": 1, "test": 1},
        "trajectory_counts": {"train": 1, "validation": 1, "test": 1},
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return protocol_path, audit_path, paths


def _rewrite_protocol(protocol_path: Path, audit_path: Path, protocol: dict[str, object]) -> None:
    protocol["protocol_fingerprint"] = ANALYSIS._protocol_fingerprint(protocol)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["protocol_fingerprint"] = protocol["protocol_fingerprint"]
    audit["split_manifest_hash"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    audit_path.write_text(json.dumps(audit), encoding="utf-8")


def test_load_development_frames_reads_only_declared_train_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, audit_path, paths = _write_bound_protocol(tmp_path)
    hashed: list[Path] = []
    opened: list[Path] = []
    original_hash = ANALYSIS._sha256_file
    original_read_csv = ANALYSIS.pd.read_csv

    def tracked_hash(path: Path) -> str:
        hashed.append(Path(path).resolve())
        return original_hash(path)

    def tracked_read_csv(path: Path, *args, **kwargs) -> pd.DataFrame:
        opened.append(Path(path).resolve())
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(ANALYSIS, "_sha256_file", tracked_hash)
    monkeypatch.setattr(ANALYSIS.pd, "read_csv", tracked_read_csv)

    binding, frames = ANALYSIS.load_development_frames(protocol_path, audit_path)

    development_paths = {paths["train"].resolve(), paths["validation"].resolve()}
    assert set(hashed) == development_paths | {protocol_path.resolve(), audit_path.resolve()}
    assert set(opened) == development_paths
    assert set(frames) == {"train", "validation"}
    assert binding["outer_test_accessed"] is False


def test_load_development_frames_rejects_missing_sealed_test_binding(tmp_path: Path) -> None:
    protocol_path, audit_path, _ = _write_bound_protocol(tmp_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    del protocol["domains"][0]["test_csv_sha256"]
    _rewrite_protocol(protocol_path, audit_path, protocol)

    with pytest.raises(ValueError, match="must bind a sealed test index"):
        ANALYSIS.load_development_frames(protocol_path, audit_path)


@pytest.mark.parametrize(
    ("document", "field", "value", "message"),
    [
        ("protocol", "train_csv_sha256", "f" * 64, "CSV SHA256 mismatch"),
        ("protocol", "train_window_count", 4, "total count mismatch"),
        ("audit", "train_sample_id_hash", "f" * 64, "sample identity hash mismatch"),
        ("audit", "train_sample_count", 4, "total count mismatch"),
    ],
)
def test_load_development_frames_rejects_hash_and_count_mismatches(
    tmp_path: Path, document: str, field: str, value: object, message: str
) -> None:
    protocol_path, audit_path, _ = _write_bound_protocol(tmp_path)
    path = protocol_path if document == "protocol" else audit_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if field == "train_csv_sha256":
        payload["domains"][0][field] = value
    else:
        payload[field] = value
    if document == "protocol":
        _rewrite_protocol(protocol_path, audit_path, payload)
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        ANALYSIS.load_development_frames(protocol_path, audit_path)


def test_circular_distance_wraps_between_last_and_first_beam() -> None:
    np.testing.assert_array_equal(
        ANALYSIS.circular_distance([63, 0, 1], [0, 63, 63]),
        [1, 1, 2],
    )


def test_temporal_analysis_reports_prepared_target_alias_and_circular_change() -> None:
    train = _frame("train", [63, 0, 0], "train-group")
    validation = _frame("validation", [4, 4], "validation-group")
    validation.loc[1, "beam_label"] = 5

    rows, summary = ANALYSIS.analyze_temporal({"train": train, "validation": validation})

    train_row = rows[rows["role"] == "train"].iloc[0]
    assert train_row["prepared_label_target_alias_rate"] == pytest.approx(1.0)
    assert train_row["adjacent_target_same_rate"] == pytest.approx(0.5)
    assert train_row["adjacent_target_circular_change_mean"] == pytest.approx(0.5)
    assert train_row["window_frame_exposures"] == 18
    assert train_row["unique_window_frames"] == 8
    assert summary["train"]["prepared_label_target_alias_rate"] == pytest.approx(1.0)
    assert summary["validation"]["prepared_label_target_alias_rate"] == pytest.approx(0.5)


def test_scan_resources_records_missing_resource_and_strict_mode_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resource_key = "logical/beam/missing.txt"
    scan_path = tmp_path / "missing-beam.txt"
    arguments = {
        "modality": "beam",
        "key_matrix": np.asarray([[resource_key]]),
        "scan_paths": {resource_key: str(scan_path)},
        "worker_name": "beam",
        "feature_names": ANALYSIS._beam_feature_names(),
        "workers": 1,
        "force": True,
    }
    monkeypatch.setattr(ANALYSIS, "ProcessPoolExecutor", ThreadPoolExecutor)

    result = ANALYSIS._scan_resources(
        **arguments,
        cache_path=tmp_path / "non-strict-cache.npz",
        strict_resources=False,
    )

    assert result["metadata"]["failed_count"] == 1
    assert np.isnan(result["features"]).all()
    assert result["errors"] == [
        {
            "modality": "beam",
            "resource_key": resource_key,
            "scan_path": str(scan_path),
            "error": result["errors"][0]["error"],
        }
    ]
    assert result["errors"][0]["error"].startswith("FileNotFoundError:")
    assert str(scan_path) in result["errors"][0]["error"]

    with pytest.raises(ValueError, match="beam scan failed for 1 resources"):
        ANALYSIS._scan_resources(
            **arguments,
            cache_path=tmp_path / "strict-cache.npz",
            strict_resources=True,
        )

    reuse_arguments = {**arguments, "force": False}
    with pytest.raises(ValueError, match="beam scan failed for 1 resources"):
        ANALYSIS._scan_resources(
            **reuse_arguments,
            cache_path=tmp_path / "non-strict-cache.npz",
            strict_resources=True,
        )


def test_feature_space_shift_marks_duplicate_nn_degeneracy_and_is_deterministic() -> None:
    features = np.vstack(
        [
            np.zeros((2, 40)),
            np.ones((2, 40)),
            np.full((1, 40), 0.25),
            np.full((1, 40), 2.0),
        ]
    )
    roles = np.asarray(["train"] * 4 + ["validation"] * 2)
    groups = np.asarray([f"group-{index}" for index in range(len(features))])

    first = ANALYSIS._feature_space_shift(features, roles, groups, seed=2026)
    second = ANALYSIS._feature_space_shift(features, roles, groups, seed=2026)

    assert first["nn_duplicate_degenerate"] is True
    assert not np.isfinite(first["nn_distance_ratio"])
    assert first["train_self_nn_median"] == pytest.approx(0.0)
    assert {key: value for key, value in first.items() if key != "nn_distance_ratio"} == {
        key: value for key, value in second.items() if key != "nn_distance_ratio"
    }


def test_linear_probe_fits_statistics_on_train_rows_only_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [0.5, 1.5], [1.5, 0.5], [100.0, -100.0], [200.0, -200.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    fit_mask = np.asarray([True, True, True, True, False, False])
    evaluation_mask = ~fit_mask
    groups = np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int64)
    observed_shapes: list[tuple[int, ...]] = []
    original_nanmean = ANALYSIS.np.nanmean

    def tracked_nanmean(values, *args, **kwargs):
        observed_shapes.append(np.asarray(values).shape)
        return original_nanmean(values, *args, **kwargs)

    monkeypatch.setattr(ANALYSIS.np, "nanmean", tracked_nanmean)
    arguments = {
        "features": features,
        "labels": labels,
        "fit_mask": fit_mask,
        "evaluation_mask": evaluation_mask,
        "group_codes": groups,
        "device": ANALYSIS.torch.device("cpu"),
        "epochs": 2,
        "seed": 2026,
    }

    first = ANALYSIS._fit_linear_probe(**arguments)
    second = ANALYSIS._fit_linear_probe(**arguments)

    assert observed_shapes == [(4, 2), (4, 2)]
    assert second == pytest.approx(first)


def test_cross_weather_alignment_exposes_route_content_clones() -> None:
    rows = []
    for condition in ("rainy", "sunny", "foggy"):
        role = "validation" if condition == "foggy" else "train"
        for seq_index, label in enumerate((7, 8)):
            row = {
                "domain_id": f"{condition}/route-a",
                "condition": condition,
                "sensor_scenario": "route-a",
                "agent": "cav-1",
                "seq_index": seq_index,
                "future_beam_label1": label,
                "trajectory_group_id": f"group-{condition}",
            }
            row.update({f"geometry{step}": json.dumps({"relative_x": seq_index}) for step in range(1, 6)})
            rows.append((role, row))
    frames = {role: pd.DataFrame([row for row_role, row in rows if row_role == role]) for role in ("train", "validation")}

    pairs, table, summary = ANALYSIS.analyze_cross_role_alignment(frames)
    shortcut = ANALYSIS._alignment_lookup_shortcut(frames, pairs)

    assert len(pairs) == 4
    assert len(table) == 2
    assert summary["validation_sample_coverage"] == pytest.approx(1.0)
    assert summary["counterparts_per_validation_median"] == pytest.approx(2.0)
    assert summary["target_pair_match_rate"] == pytest.approx(1.0)
    assert summary["geometry_pair_exact_match_rate"] == pytest.approx(1.0)
    assert summary["cross_weather_route_content_overlap_detected"] is True
    assert shortcut.iloc[0]["top1"] == pytest.approx(1.0)
    assert shortcut.iloc[0]["diagnostic_only"] == np.True_


def test_paired_signatures_report_exact_and_non_exact_modalities() -> None:
    rows = []
    for condition in ("rainy", "sunny", "foggy"):
        role = "validation" if condition == "foggy" else "train"
        rows.append(
            (
                role,
                {
                    "domain_id": f"{condition}/route-a",
                    "condition": condition,
                    "sensor_scenario": "route-a",
                    "agent": "cav-1",
                    "seq_index": 0,
                    "future_beam_label1": 3,
                    "trajectory_group_id": f"group-{condition}",
                    **{f"geometry{step}": "{}" for step in range(1, 6)},
                },
            )
        )
    frames = {role: pd.DataFrame([row for row_role, row in rows if row_role == role]) for role in ("train", "validation")}
    pairs, _, _ = ANALYSIS.analyze_cross_role_alignment(frames)
    features = {
        "exact": np.asarray([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=np.float32),
        "shifted": np.asarray([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]], dtype=np.float32),
    }

    result = ANALYSIS.analyze_paired_signatures(
        pairs,
        features,
        train_count=2,
        train_group_ids=frames["train"]["trajectory_group_id"],
    )
    overall = result[result["scope"] == "all"].set_index("modality")

    assert overall.loc["exact", "validation_any_exact_share"] == pytest.approx(1.0)
    assert overall.loc["shifted", "validation_any_exact_share"] == pytest.approx(0.0)
    assert overall.loc["shifted", "standardization_unit"] == "trajectory_group"
    assert overall.loc["shifted", "standardized_rmse_min_p50"] == pytest.approx(2.0)


def test_load_development_frames_rejects_fractional_labels(tmp_path: Path) -> None:
    protocol_path, audit_path, paths = _write_bound_protocol(tmp_path)
    train = pd.read_csv(paths["train"])
    train["future_beam_label1"] = train["future_beam_label1"].astype(np.float64)
    train.loc[0, "future_beam_label1"] = 1.5
    train.to_csv(paths["train"], index=False)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["domains"][0]["train_csv_sha256"] = hashlib.sha256(paths["train"].read_bytes()).hexdigest()
    _rewrite_protocol(protocol_path, audit_path, protocol)

    with pytest.raises(ValueError, match="must be integer labels"):
        ANALYSIS.load_development_frames(protocol_path, audit_path)


def test_radar_inventory_digest_tracks_da_companion(tmp_path: Path) -> None:
    ra_path = tmp_path / "frame_RA.npy"
    da_path = tmp_path / "frame_DA.npy"
    ra_path.write_bytes(b"ra")
    da_path.write_bytes(b"da-before")
    key = "logical-radar-frame"

    before = ANALYSIS._resource_inventory_digest([key], {key: str(ra_path)})
    da_path.write_bytes(b"da-after-with-different-size")
    after = ANALYSIS._resource_inventory_digest([key], {key: str(ra_path)})

    assert before != after


def test_csi_packed_cache_rejects_extra_development_path_before_values_are_used(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    fingerprint = "a" * 64
    rows = []
    expected_paths = []
    for role in ("train", "validation"):
        row = {"_role": role, "_data_root": str(data_root)}
        for step in range(1, 6):
            reference = f"{role}/channel_{step}.npy"
            row[f"csi{step}"] = reference
            expected_paths.append(str((data_root / reference).resolve()))
        rows.append(row)
    all_rows = pd.DataFrame(rows)
    channel_paths = np.asarray([*expected_paths, str((data_root / "sealed-test/channel.npy").resolve())])
    metadata = {
        "schema_version": ANALYSIS.PCPF_SPARSE_CSI_PACKED_CACHE_SCHEMA_VERSION,
        "status": "passed",
        "protocol_id": ANALYSIS.PROTOCOL_ID,
        "protocol_fingerprint": fingerprint,
        "manifest_version": ANALYSIS.TRAJECTORY_MANIFEST_VERSION,
        "split_seed": ANALYSIS.TRAJECTORY_SPLIT_SEED,
        "outer_test_accessed": False,
        "entry_count": len(channel_paths),
        "roles": {
            "train": {"sample_count": 1, "unique_channel_count": 5},
            "validation": {"sample_count": 1, "unique_channel_count": 5},
        },
        "cache_spec_sha256": "b" * 64,
        "codebook_file_sha256": "c" * 64,
        "codebook_hash": "d" * 64,
        "selection_sha256": ANALYSIS.PCPF_SPARSE_CSI_SELECTION_SHA256,
    }
    packed = tmp_path / "packed.npz"
    np.savez(
        packed,
        metadata_json=np.asarray(json.dumps(metadata)),
        channel_paths=channel_paths,
        cache_keys=np.asarray([f"key-{index}" for index in range(len(channel_paths))]),
        selected_g=np.zeros((len(channel_paths), 2, 2), dtype=np.complex64),
    )

    with pytest.raises(ValueError, match="path set mismatch.*extra=1"):
        ANALYSIS._analyze_csi(
            all_rows,
            packed_cache_path=packed,
            expected_packed_cache_sha256=hashlib.sha256(packed.read_bytes()).hexdigest(),
            protocol_fingerprint=fingerprint,
            seed=2026,
        )


def test_load_development_frames_recomputes_manifest_fingerprint(tmp_path: Path) -> None:
    protocol_path, audit_path, _ = _write_bound_protocol(tmp_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["claim_eligible"] = True
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest fingerprint mismatch"):
        ANALYSIS.load_development_frames(protocol_path, audit_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("audit_id", "must declare audit_id"),
        ("failures", "failures must be an empty list"),
        ("check", "pass every structural"),
    ],
)
def test_load_development_frames_rejects_invalid_audit_contract(tmp_path: Path, mutation: str, message: str) -> None:
    protocol_path, audit_path, _ = _write_bound_protocol(tmp_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if mutation == "audit_id":
        audit["audit_id"] = "wrong-audit"
    elif mutation == "failures":
        audit["failures"] = ["weather_binding"]
    else:
        audit["checks"]["weather_binding"] = False
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        ANALYSIS.load_development_frames(protocol_path, audit_path)


def test_formal_analysis_binding_requires_exact_development_budget(tmp_path: Path) -> None:
    protocol_path, audit_path, _ = _write_bound_protocol(tmp_path)
    binding, _ = ANALYSIS.load_development_frames(protocol_path, audit_path)

    with pytest.raises(ValueError, match="Formal trajectory analysis binding mismatch"):
        ANALYSIS._require_formal_analysis_binding(binding)


def test_artifact_inventory_hashes_only_declared_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    allowed_root = tmp_path / "allowed"
    output_dir = allowed_root / "run"
    output_dir.mkdir(parents=True)
    expected = {
        "report.md",
        "summary.json",
        "tables/conditional_label_shift.csv",
        "tables/cross_weather_alignment.csv",
        "tables/cross_weather_signature_overlap.csv",
        "tables/geometry_shift.csv",
        "tables/label_distribution.csv",
        "tables/resource_errors.csv",
        "tables/resource_quality.csv",
        "tables/resource_reuse.csv",
        "tables/sample_diagnostics.csv",
        "tables/shortcut_baselines.csv",
        "tables/signal_shift.csv",
        "tables/signal_shortcut_baselines.csv",
        "tables/split_composition.csv",
        "tables/split_sensitivity.csv",
        "tables/temporal_profile.csv",
        "tables/trajectory_group_profile.csv",
    }
    for relative in expected:
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    sealed = output_dir / "splits" / "test" / "sealed.csv"
    sealed.parent.mkdir(parents=True)
    sealed.write_text("secret", encoding="utf-8")
    hashed: list[Path] = []
    original_hash = ANALYSIS._sha256_file

    def tracked_hash(path: Path) -> str:
        hashed.append(Path(path))
        return original_hash(path)

    monkeypatch.setattr(ANALYSIS, "DEFAULT_OUTPUT", allowed_root)
    monkeypatch.setattr(ANALYSIS, "_sha256_file", tracked_hash)

    assert ANALYSIS._validated_output_dir(output_dir) == output_dir.resolve()
    with pytest.raises(ValueError, match="must stay under"):
        ANALYSIS._validated_output_dir(tmp_path / "outside")
    inventory = ANALYSIS._artifact_inventory(
        output_dir,
        active_probe_names=set(),
        probes_enabled=False,
        modalities=set(),
        figure_paths=set(),
    )

    assert [item["path"] for item in inventory] == sorted(expected)
    assert hashed == [output_dir / relative for relative in sorted(expected)]
    assert sealed not in hashed

    (output_dir / "report.md").unlink()
    with pytest.raises(FileNotFoundError, match="Expected analysis artifact is missing"):
        ANALYSIS._artifact_inventory(
            output_dir,
            active_probe_names=set(),
            probes_enabled=False,
            modalities=set(),
            figure_paths=set(),
        )


def test_diagnostic_probes_reject_geometry_and_unknown_roles(tmp_path: Path) -> None:
    rows = pd.concat(
        [
            _frame("train", [0, 1], "train-group"),
            _frame("validation", [0], "validation-group"),
        ],
        ignore_index=True,
    )
    rows["_role"] = rows["split"]
    rows["sensor_scenario"] = "scenario-a"
    rows["condition"] = ["rainy", "sunny", "foggy"]
    features = np.zeros((len(rows), 2), dtype=np.float32)

    with pytest.raises(ValueError, match="forbidden or unknown features.*geometry"):
        ANALYSIS.run_diagnostic_probes(
            rows,
            {"geometry": features, "image": features},
            cache_dir=tmp_path,
            devices=["cpu"],
            epochs=1,
            folds=2,
            seed=2026,
        )

    rows.loc[0, "_role"] = "test"
    with pytest.raises(ValueError, match="exactly train/validation roles.*test"):
        ANALYSIS.run_diagnostic_probes(
            rows,
            {"image": features},
            cache_dir=tmp_path,
            devices=["cpu"],
            epochs=1,
            folds=2,
            seed=2026,
        )


def test_probe_summary_ranks_equal_group_macro_before_window_micro() -> None:
    rows = pd.DataFrame(
        [
            {
                "probe": probe,
                "evaluation": evaluation,
                "feature_dimensions": 2,
                "top1": micro,
                "group_macro_top1": macro,
                "group_worst_top1": macro / 2,
            }
            for probe, micro, macro in (
                ("micro_winner", 0.9, 0.2),
                ("macro_winner", 0.4, 0.8),
            )
            for evaluation in ("scenario_leave_one_out", "trajectory_validation")
        ]
    )

    summary = ANALYSIS.summarize_probes(rows)

    assert summary.iloc[0]["probe"] == "macro_winner"
    assert summary.iloc[0]["scenario_loo_group_macro_top1_mean"] == pytest.approx(0.8)
    assert summary.iloc[1]["scenario_loo_top1_mean"] == pytest.approx(0.9)


def test_probe_worker_rejects_unknown_role_codes(tmp_path: Path) -> None:
    features_path = tmp_path / "features.npy"
    metadata_path = tmp_path / "metadata.npz"
    np.save(features_path, np.ones((3, 2), dtype=np.float32))
    np.savez(
        metadata_path,
        role_codes=np.asarray([0, 2, 1], dtype=np.int8),
        labels=np.asarray([0, 1, 2], dtype=np.int64),
        group_codes=np.asarray([0, 1, 2], dtype=np.int64),
        group_names=np.asarray(["g0", "g1", "g2"]),
        scenario_codes=np.asarray([0, 1, 2], dtype=np.int64),
        scenario_names=np.asarray(["s0", "s1", "s2"]),
        weather_codes=np.asarray([0, 1, 2], dtype=np.int64),
        weather_names=np.asarray(["w0", "w1", "w2"]),
    )

    with pytest.raises(ValueError, match="metadata arrays, roles, labels, or feature rows are invalid"):
        ANALYSIS._run_probe_task(
            {
                "name": "image",
                "feature_path": str(features_path),
                "metadata_path": str(metadata_path),
                "device": "cpu",
                "epochs": 1,
                "folds": 2,
                "seed": 2026,
            }
        )


def test_probe_worker_uses_train_only_fit_masks_for_every_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    features_path = tmp_path / "features.npy"
    metadata_path = tmp_path / "metadata.npz"
    np.save(features_path, np.arange(16, dtype=np.float32).reshape(8, 2))
    role_codes = np.asarray([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
    scenario_codes = np.asarray([0, 0, 1, 1, 2, 2, 0, 1], dtype=np.int64)
    weather_codes = np.asarray([0, 1, 0, 1, 0, 1, 2, 2], dtype=np.int64)
    np.savez(
        metadata_path,
        role_codes=role_codes,
        labels=np.arange(8, dtype=np.int64),
        group_codes=np.arange(8, dtype=np.int64),
        group_names=np.asarray([f"g{index}" for index in range(8)]),
        scenario_codes=scenario_codes,
        scenario_names=np.asarray(["s0", "s1", "s2"]),
        weather_codes=weather_codes,
        weather_names=np.asarray(["w0", "w1", "w2"]),
    )
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_fit(features, labels, fit_mask, evaluation_mask, group_codes, **kwargs):
        del features, labels, group_codes, kwargs
        calls.append((np.flatnonzero(fit_mask), np.flatnonzero(evaluation_mask)))
        return {"top1": 0.5, "group_macro_top1": 0.5, "group_worst_top1": 0.5}

    monkeypatch.setattr(ANALYSIS, "_fit_linear_probe", fake_fit)
    rows = ANALYSIS._run_probe_task(
        {
            "name": "image",
            "feature_path": str(features_path),
            "metadata_path": str(metadata_path),
            "device": "cpu",
            "epochs": 2,
            "folds": 3,
            "seed": 2026,
        }
    )

    assert len(rows) == len(calls)
    for row, (fit_indices, evaluation_indices) in zip(rows, calls, strict=True):
        assert set(fit_indices).issubset(range(6))
        if row["evaluation"] == "trajectory_validation":
            assert set(evaluation_indices) == {6, 7}
        else:
            assert set(evaluation_indices).issubset(range(6))
        assert row["optimizer_steps"] == row["epochs"] * int(np.ceil(row["fit_samples"] / row["batch_size"]))


def test_probe_collection_is_sorted_after_out_of_order_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = pd.DataFrame(
        {
            "_role": ["train", "train", "validation"],
            "future_beam_label1": [0, 1, 0],
            "trajectory_group_id": ["g0", "g1", "g2"],
            "sensor_scenario": ["s0", "s1", "s0"],
            "condition": ["rainy", "sunny", "foggy"],
        }
    )
    submitted: list[dict[str, object]] = []

    class ImmediateExecutor:
        def __init__(self, **kwargs):
            del kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

        def submit(self, function, task):
            del function
            submitted.append(dict(task))
            future = Future()
            future.set_result(
                [
                    {
                        "probe": task["name"],
                        "evaluation": "trajectory_validation",
                        "fold": -1,
                        "diagnostic_only": True,
                    }
                ]
            )
            return future

    monkeypatch.setattr(ANALYSIS, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(ANALYSIS, "as_completed", lambda futures: reversed(list(futures)))
    result = ANALYSIS.run_diagnostic_probes(
        rows,
        {
            "radar": np.ones((3, 2), dtype=np.float32),
            "image": np.zeros((3, 2), dtype=np.float32),
        },
        cache_dir=tmp_path,
        devices=["cuda:1", "cuda:2", "cuda:3"],
        epochs=2,
        folds=2,
        seed=2026,
    )

    assert result["probe"].tolist() == sorted(result["probe"].tolist())
    assert {task["seed"] for task in submitted} == {2026}
    assert [task["device"] for task in submitted] == ["cuda:1", "cuda:2", "cuda:3"]
    assert not (tmp_path / "probe_features_geometry.npy").exists()


def test_csi_packed_cache_rejects_wrong_sha_and_selection(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    fingerprint = "a" * 64
    rows = []
    channel_paths = []
    for role in ANALYSIS.ROLES:
        row = {"_role": role, "_data_root": str(data_root)}
        for step in range(1, 6):
            reference = f"{role}/channel_{step}.npy"
            row[f"csi{step}"] = reference
            channel_paths.append(str((data_root / reference).resolve()))
        rows.append(row)
    all_rows = pd.DataFrame(rows)
    metadata = {
        "schema_version": ANALYSIS.PCPF_SPARSE_CSI_PACKED_CACHE_SCHEMA_VERSION,
        "status": "passed",
        "protocol_id": ANALYSIS.PROTOCOL_ID,
        "protocol_fingerprint": fingerprint,
        "manifest_version": ANALYSIS.TRAJECTORY_MANIFEST_VERSION,
        "split_seed": ANALYSIS.TRAJECTORY_SPLIT_SEED,
        "outer_test_accessed": False,
        "entry_count": len(channel_paths),
        "roles": {role: {"sample_count": 1, "unique_channel_count": 5} for role in ANALYSIS.ROLES},
        "cache_spec_sha256": "b" * 64,
        "codebook_file_sha256": "c" * 64,
        "codebook_hash": "d" * 64,
        "selection_sha256": "e" * 64,
    }
    packed = tmp_path / "packed.npz"
    np.savez(
        packed,
        metadata_json=np.asarray(json.dumps(metadata)),
        channel_paths=np.asarray(channel_paths),
        cache_keys=np.asarray([f"key-{index}" for index in range(len(channel_paths))]),
        selected_g=np.zeros((len(channel_paths), 2, 2), dtype=np.complex64),
    )
    actual_sha = hashlib.sha256(packed.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="packed cache SHA256 mismatch"):
        ANALYSIS._analyze_csi(
            all_rows,
            packed_cache_path=packed,
            expected_packed_cache_sha256="f" * 64,
            protocol_fingerprint=fingerprint,
            seed=2026,
        )
    with pytest.raises(ValueError, match="fixed TSPC-V2 2x2 selection"):
        ANALYSIS._analyze_csi(
            all_rows,
            packed_cache_path=packed,
            expected_packed_cache_sha256=actual_sha,
            protocol_fingerprint=fingerprint,
            seed=2026,
        )


def test_resource_inventory_digest_tracks_same_size_content_change_with_restored_mtime(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resource.bin"
    path.write_bytes(b"before")
    original = path.stat()
    before = ANALYSIS._resource_inventory_digest(["resource"], {"resource": str(path)})

    path.write_bytes(b"differ")
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    after = ANALYSIS._resource_inventory_digest(["resource"], {"resource": str(path)})

    assert path.stat().st_size == original.st_size
    assert path.stat().st_mtime_ns == original.st_mtime_ns
    assert before != after


def test_paired_signatures_never_count_missing_values_as_exact() -> None:
    alignment_pairs = pd.DataFrame(
        {
            "_train_index": [0],
            "_validation_index": [0],
            "validation_domain_id": ["validation-domain"],
            "validation_trajectory_group_id": ["validation-group"],
        }
    )
    features = {"image": np.full((2, 3), np.nan, dtype=np.float32)}

    result = ANALYSIS.analyze_paired_signatures(
        alignment_pairs,
        features,
        train_count=1,
        train_group_ids=["train-group"],
    )
    row = result.iloc[0]

    assert row["valid_pair_share"] == pytest.approx(0.0)
    assert row["exact_pair_share"] == pytest.approx(0.0)
    assert row["validation_group_macro_any_exact_share"] == pytest.approx(0.0)


def test_paired_signature_scale_weights_train_trajectory_groups_equally() -> None:
    def analyze(large_group_size: int) -> pd.Series:
        train_count = large_group_size + 1
        pairs = pd.DataFrame(
            {
                "_train_index": np.arange(train_count),
                "_validation_index": np.zeros(train_count, dtype=np.int64),
                "validation_domain_id": "validation-domain",
                "validation_trajectory_group_id": "validation-group",
            }
        )
        train_features = np.asarray([[0.0], *([[2.0]] * large_group_size)], dtype=np.float32)
        features = {"image": np.concatenate([train_features, np.asarray([[4.0]], dtype=np.float32)])}
        result = ANALYSIS.analyze_paired_signatures(
            pairs,
            features,
            train_count=train_count,
            train_group_ids=["small", *(["large"] * large_group_size)],
        )
        return result.iloc[0]

    balanced = analyze(1)
    imbalanced = analyze(100)

    assert balanced["standardization_unit"] == "trajectory_group"
    assert balanced["standardized_rmse_min_p50"] == pytest.approx(2.0)
    assert imbalanced["standardized_rmse_min_p50"] == pytest.approx(balanced["standardized_rmse_min_p50"])


def test_continuous_shift_uses_equal_group_means_as_primary_statistic() -> None:
    frame = pd.DataFrame(
        {
            "role": ["train"] * 10 + ["validation"] * 2,
            "trajectory_group_id": ["small"] + ["large"] * 9 + ["v0", "v1"],
            "value": [0.0] + [10.0] * 9 + [4.0, 6.0],
        }
    )

    result = ANALYSIS._continuous_shift_table(
        frame,
        role_column="role",
        group_column="trajectory_group_id",
        columns=["value"],
        modality="synthetic",
    ).iloc[0]

    assert result["independent_unit"] == "trajectory_group"
    assert result["train_mean"] == pytest.approx(5.0)
    assert result["train_window_micro_mean"] == pytest.approx(9.0)
    assert result["validation_mean"] == pytest.approx(5.0)


def test_cross_weather_alignment_excludes_same_weather_and_canonicalizes_geometry() -> None:
    train_rows = []
    for condition, domain in (("rainy", "train-rainy"), ("sunny", "train-sunny")):
        row = {
            "domain_id": domain,
            "condition": condition,
            "sensor_scenario": "route-a",
            "agent": "cav-1",
            "seq_index": 0,
            "future_beam_label1": 7,
            "trajectory_group_id": f"group-{condition}",
        }
        row.update({f"geometry{step}": '{"a":1,"b":2}' for step in range(1, 6)})
        train_rows.append(row)
    validation_row = {
        "domain_id": "validation-rainy",
        "condition": "rainy",
        "sensor_scenario": "route-a",
        "agent": "cav-1",
        "seq_index": 0,
        "future_beam_label1": 7,
        "trajectory_group_id": "validation-group",
        **{f"geometry{step}": '{ "b": 2, "a": 1 }' for step in range(1, 6)},
    }

    pairs, _, summary = ANALYSIS.analyze_cross_role_alignment(
        {"train": pd.DataFrame(train_rows), "validation": pd.DataFrame([validation_row])}
    )

    assert len(pairs) == 1
    assert pairs.iloc[0]["train_condition"] == "sunny"
    assert pairs.iloc[0]["geometry_sequence_exact_match"] == np.True_
    assert summary["same_weather_pairs_excluded"] == 1
    assert summary["validation_group_macro_coverage"] == pytest.approx(1.0)


def test_load_development_frames_locates_invalid_diagnostic_json(tmp_path: Path) -> None:
    protocol_path, audit_path, paths = _write_bound_protocol(tmp_path)
    train = pd.read_csv(paths["train"])
    train.loc[0, "window_frame_ids_json"] = "not-json"
    train.to_csv(paths["train"], index=False)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["domains"][0]["train_csv_sha256"] = hashlib.sha256(paths["train"].read_bytes()).hexdigest()
    _rewrite_protocol(protocol_path, audit_path, protocol)

    with pytest.raises(ValueError, match="column=window_frame_ids_json, sample_id=train-0"):
        ANALYSIS.load_development_frames(protocol_path, audit_path)


def test_output_writers_reject_symbolic_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    target = tmp_path / "target.json"
    target.write_text("unchanged", encoding="utf-8")
    linked = allowed_root / "summary.json"
    linked.symlink_to(target)
    monkeypatch.setattr(ANALYSIS, "DEFAULT_OUTPUT", allowed_root)

    with pytest.raises(ValueError, match="must not contain symbolic links"):
        ANALYSIS._write_json(linked, {"status": "passed"})

    assert target.read_text(encoding="utf-8") == "unchanged"
