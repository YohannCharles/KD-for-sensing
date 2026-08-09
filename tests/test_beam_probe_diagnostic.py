import csv
import gzip
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kd_sensing.eval.beam_probe_diagnostic import (
    ADAPTIVE_OFFSETS,
    ADAPTIVE_SPACINGS,
    BeamProbeSimulator,
    ProbeEvidence,
    SEVERE_SINGLE_PATTERNS,
    TBCP_DIAGONAL_METHOD,
    build_adaptive_local_candidates,
    build_local_candidates,
    build_oracle_local_candidates,
    build_posterior_topk_candidates,
    build_uniform_candidates,
    build_validation_power_index,
    load_probe_evidence,
    run_probe_diagnostic,
    run_tbcp_probe_diagnostic,
    run_tbcp_robustness_sensitivity,
    select_probe_evidence_by_hash,
    summarize_tbcp_robustness_replays,
    summarize_tbcp_replays,
    uniform_offsets,
)
from kd_sensing.eval.beam_topology_likelihood import fit_topology_likelihood, save_topology_likelihood
from kd_sensing.utils.checkpoint import checkpoint_file_digest


FOUR_SENSING_MASKS = {
    "missing_csi": (1, 1, 1, 1, 0),
    "missing_image_csi": (0, 1, 1, 1, 0),
    "missing_radar_csi": (1, 0, 1, 1, 0),
    "missing_gps_csi": (1, 1, 0, 1, 0),
    "missing_lidar_csi": (1, 1, 1, 0, 0),
    "missing_image_radar_csi": (0, 0, 1, 1, 0),
    "missing_image_gps_csi": (0, 1, 0, 1, 0),
    "missing_image_lidar_csi": (0, 1, 1, 0, 0),
    "missing_radar_gps_csi": (1, 0, 0, 1, 0),
    "missing_radar_lidar_csi": (1, 0, 1, 0, 0),
    "missing_gps_lidar_csi": (1, 1, 0, 0, 0),
    "image_only": (1, 0, 0, 0, 0),
    "radar_only": (0, 1, 0, 0, 0),
    "gps_only": (0, 0, 1, 0, 0),
    "lidar_only": (0, 0, 0, 1, 0),
}


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _synthetic_likelihood(tmp_path: Path):
    train = tmp_path / "train"
    train.mkdir(parents=True)
    paths = {}
    labels = {}
    protocol_ids = {}
    for index in range(8):
        label = (index * 7) % 64
        delta = np.arange(64)
        distance = np.minimum(delta, 64 - delta)
        aligned = np.exp(-np.square(distance) / (2.0 * (1.5 + 0.1 * index) ** 2)) + 1e-5
        aligned = np.minimum(aligned, 0.98)
        aligned[0] = 1.0
        power = np.empty(64)
        power[(label + delta) % 64] = aligned
        path = train / f"{index}.txt"
        np.savetxt(path, power)
        paths[f"stable-{index}"] = path
        labels[f"stable-{index}"] = label
        protocol_ids[f"stable-{index}"] = f"source-{index}"
    manifest = train / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    source = train / "source.csv"
    source.write_text("synthetic\n", encoding="utf-8")
    provenance = {
        "fit_split": "train",
        "source_split": "train",
        "protocol_id": "mmw_id_stratified_block_v1",
        "protocol_version": 1,
        "protocol_fingerprint": "a" * 64,
        "split_manifest": str(manifest.resolve()),
        "split_manifest_hash": manifest_sha256,
        "split_manifest_file_sha256": manifest_sha256,
        "data_source_hash": "b" * 64,
        "window_config_hash": "c" * 64,
        "split_seed": 0,
        "block_size": 32,
        "weather_binding": True,
        "train_sample_count": len(paths),
        "train_sample_id_hash": _sha256_lines(list(protocol_ids.values())),
        "train_stable_sample_id_hash": _sha256_lines(list(paths)),
        "topology_id": "ula_dft_phase_cycle_v1",
        "topology_descriptor_sha256": "d" * 64,
        "topology_audit_sha256": "e" * 64,
        "test_evaluated": False,
        "outer_test_accessed": False,
        "source_components": [
            {
                "domain_id": "synthetic/domain",
                "path": str(source.resolve()),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "sample_count": len(paths),
            }
        ],
    }
    return fit_topology_likelihood(paths, labels, protocol_ids, provenance=provenance)


def test_probe_candidate_policies_are_oracle_separated_and_exact_budget() -> None:
    assert build_local_candidates(0, 7) == (61, 62, 63, 0, 1, 2, 3)
    assert build_uniform_candidates(7, 0) == (0, 9, 18, 27, 36, 45, 54)
    assert build_oracle_local_candidates(63, 3) == (62, 63, 0)
    assert uniform_offsets(7) == tuple(range(64))
    counts = np.bincount(
        [beam for offset in uniform_offsets(7) for beam in build_uniform_candidates(7, offset)],
        minlength=64,
    )
    assert counts.tolist() == [7] * 64
    assert tuple(inspect.signature(build_local_candidates).parameters) == ("pred_beam", "k", "num_beams")
    assert tuple(inspect.signature(build_uniform_candidates).parameters) == ("k", "offset", "num_beams")
    assert tuple(inspect.signature(build_oracle_local_candidates).parameters) == ("gt_beam", "k", "num_beams")
    assert tuple(inspect.signature(build_adaptive_local_candidates).parameters) == ("pred_prob", "num_beams")
    assert tuple(inspect.signature(build_posterior_topk_candidates).parameters) == ("pred_prob", "k", "num_beams")


def test_synthetic_probe_noise_is_matched_by_sample_and_beam(tmp_path: Path) -> None:
    power = np.linspace(0.1, 1.0, 64, dtype=np.float64)
    path = tmp_path / "power.txt"
    np.savetxt(path, power)
    simulator = BeamProbeSimulator({"sample": path}, require_strict_positive=True)

    assert simulator.probe("sample", (2, 17), measurement_error_std_db=0.0) == pytest.approx(
        (power[2], power[17]), abs=0.0, rel=0.0
    )
    forward = simulator.probe(
        "sample", (2, 17), measurement_error_std_db=3.0, noise_seed=7, noise_replica=1
    )
    reverse = simulator.probe(
        "sample", (17, 2), measurement_error_std_db=3.0, noise_seed=7, noise_replica=1
    )
    assert forward == pytest.approx(tuple(reversed(reverse)), abs=0.0, rel=0.0)
    assert forward != simulator.probe(
        "sample", (2, 17), measurement_error_std_db=3.0, noise_seed=7, noise_replica=2
    )
    with pytest.raises(ValueError, match="finite and non-negative"):
        simulator.probe("sample", (2,), measurement_error_std_db=float("nan"))


def test_adaptive_local7_preserves_core_wrap_and_uses_narrowest_tie() -> None:
    probability = np.zeros(64, dtype=np.float64)
    probability[0] = 1.0

    candidates, spacing = build_adaptive_local_candidates(probability)

    assert spacing == 1
    assert candidates == (61, 62, 63, 0, 1, 2, 3)
    assert {63, 0, 1}.issubset(candidates)
    assert len(candidates) == len(set(candidates)) == 7


def test_adaptive_local7_expands_for_a_distant_posterior_mode() -> None:
    probability = np.zeros(64, dtype=np.float64)
    probability[[0, 16, 48, 10]] = [0.30, 0.25, 0.25, 0.20]

    candidates, spacing = build_adaptive_local_candidates(probability)

    assert spacing == 8
    assert candidates == (48, 56, 63, 0, 1, 8, 16)
    assert set(ADAPTIVE_SPACINGS) == set(ADAPTIVE_OFFSETS)
    assert {63, 0, 1}.issubset(candidates)


def test_posterior_top7_uses_stable_lower_label_ties() -> None:
    probability = np.full(64, 1.0 / 64.0, dtype=np.float64)

    assert build_posterior_topk_candidates(probability) == (0, 1, 2, 3, 4, 5, 6)


def test_posterior_policies_reject_invalid_probability_without_oracle_inputs() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        build_adaptive_local_candidates(np.zeros(64))
    with pytest.raises(ValueError, match="exactly 64"):
        build_posterior_topk_candidates(np.ones(63) / 63.0)


def test_beam_probe_simulator_reveals_requested_measurements_only(tmp_path: Path) -> None:
    power = np.arange(1, 65, dtype=np.float64)
    path = tmp_path / "power.txt"
    np.savetxt(path, power)
    simulator = BeamProbeSimulator({"sample": path})

    assert simulator.probe("sample", (0, 7, 63)) == (1.0, 8.0, 64.0)
    assert simulator.normalized_gain("sample", 7) == pytest.approx(8.0 / 64.0)
    with pytest.raises(ValueError, match="unique"):
        simulator.probe("sample", (1, 1))


def test_probe_diagnostic_writes_complete_bounded_artifacts(tmp_path: Path) -> None:
    sample_ids = ("sample-a", "sample-b")
    labels = {"sample-a": 0, "sample-b": 32}
    paths = {}
    for sample_id, label in labels.items():
        power = np.full(64, 0.01, dtype=np.float64)
        power[label] = 1.0
        power[(label + 1) % 64] = 0.8
        path = tmp_path / f"{sample_id}.txt"
        np.savetxt(path, power)
        paths[sample_id] = path

    row_ids = tuple(sample_id for pattern in SEVERE_SINGLE_PATTERNS for sample_id in sample_ids)
    patterns = tuple(pattern for pattern in SEVERE_SINGLE_PATTERNS for _ in sample_ids)
    gt = np.asarray([labels[sample_id] for sample_id in row_ids], dtype=np.int64)
    pred = np.asarray([(value + 1) % 64 for value in gt], dtype=np.int64)
    probability = np.full((len(row_ids), 64), 1e-6, dtype=np.float32)
    probability[np.arange(len(row_ids)), pred] = 1.0
    probability /= probability.sum(axis=1, keepdims=True)
    evidence = ProbeEvidence(
        sample_id=row_ids,
        pattern=patterns,
        gt_beam=gt,
        pred_beam=pred,
        pred_prob=probability,
        pred_logits=np.log(probability),
        source={
            "checkpoint_path": "checkpoint.pth",
            "checkpoint_sha256": "a" * 64,
            "experiment_seed": 1,
            "data_protocol": {"protocol_id": "mmw_id_stratified_block_v1"},
            "bounded_evaluation": False,
        },
    )

    output = tmp_path / "diagnostic"
    result = run_probe_diagnostic(
        evidence,
        power_paths=paths,
        indexed_labels=labels,
        output_dir=output,
    )

    assert result["config"]["outer_test_accessed"] is False
    assert result["config"]["model_trained_or_updated"] is False
    assert result["config"]["probing_policy"]["budget"] == 7
    assert result["config"]["probing_policy"]["adaptive_spacings"] == [1, 2, 4, 8]
    for name in (
        "config.json",
        "summary.csv",
        "per_mask_summary.csv",
        "uniform_offset_summary.csv",
        "adaptive_spacing_summary.csv",
        "per_sample_results.csv.gz",
        "sensing_predictions.csv.gz",
        "fig_top1_vs_budget.png",
        "fig_gain_vs_budget.png",
        "fig_coverage_vs_budget.png",
        "diagnostic_report.md",
    ):
        assert (output / name).is_file()
    with gzip.open(output / "per_sample_results.csv.gz", "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    local3 = [row for row in rows if row["strategy"] == "Local Scan" and row["K"] == "3"]
    adaptive7 = [row for row in rows if row["strategy"] == "Adaptive Local" and row["K"] == "7"]
    posterior7 = [row for row in rows if row["strategy"] == "Posterior Top-K" and row["K"] == "7"]
    assert len(local3) == len(row_ids)
    assert len(adaptive7) == len(posterior7) == len(row_ids)
    assert all(row["correct"] == row["gt_covered"] == "1" for row in local3)
    assert all(row["adaptive_spacing"] == "1" for row in adaptive7)
    assert all(row["beam_spread"] and row["beam_normalized_entropy"] for row in adaptive7)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        run_probe_diagnostic(
            evidence,
            power_paths=paths,
            indexed_labels=labels,
            output_dir=output,
        )


def test_probe_diagnostic_rejects_label_drift(tmp_path: Path) -> None:
    probability = np.zeros((4, 64), dtype=np.float32)
    probability[:, 0] = 1.0
    evidence = ProbeEvidence(
        sample_id=("sample",) * 4,
        pattern=SEVERE_SINGLE_PATTERNS,
        gt_beam=np.zeros(4, dtype=np.int64),
        pred_beam=np.zeros(4, dtype=np.int64),
        pred_prob=probability,
        pred_logits=np.log(np.clip(probability, np.finfo(np.float32).tiny, 1.0)),
        source={"checkpoint_path": "checkpoint.pth", "checkpoint_sha256": "a" * 64},
    )
    path = tmp_path / "power.txt"
    np.savetxt(path, np.ones(64))
    with pytest.raises(ValueError, match="label mismatch"):
        run_probe_diagnostic(
            evidence,
            power_paths={"sample": path},
            indexed_labels={"sample": 1},
            output_dir=tmp_path / "output",
        )


def test_probe_diagnostic_rejects_incomplete_unbounded_evidence(tmp_path: Path) -> None:
    probability = np.zeros((4, 64), dtype=np.float32)
    probability[:, 0] = 1.0
    evidence = ProbeEvidence(
        sample_id=("sample",) * 4,
        pattern=SEVERE_SINGLE_PATTERNS,
        gt_beam=np.zeros(4, dtype=np.int64),
        pred_beam=np.zeros(4, dtype=np.int64),
        pred_prob=probability,
        pred_logits=np.log(np.clip(probability, np.finfo(np.float32).tiny, 1.0)),
        source={"bounded_evaluation": False},
    )
    path = tmp_path / "power.txt"
    np.savetxt(path, np.arange(1, 65))
    with pytest.raises(ValueError, match="exactly match"):
        run_probe_diagnostic(
            evidence,
            power_paths={"sample": path, "missing-from-evidence": path},
            indexed_labels={"sample": 0, "missing-from-evidence": 63},
            output_dir=tmp_path / "output",
        )


def test_probe_evidence_selects_all_15_four_sensing_masks_with_csi_sealed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pth"
    torch.save(
        {
            "checkpoint_role": "validation_best",
            "model_metadata": {"training_stage": "stage3_fusion"},
        },
        checkpoint,
    )
    checkpoint_sha256, _ = checkpoint_file_digest(checkpoint)
    sample_ids = ("sample-a", "sample-b")
    patterns = tuple(pattern for pattern in FOUR_SENSING_MASKS for _ in sample_ids)
    row_ids = tuple(sample for _pattern in FOUR_SENSING_MASKS for sample in sample_ids)
    available = torch.as_tensor(
        [FOUR_SENSING_MASKS[pattern] for pattern in FOUR_SENSING_MASKS for _ in sample_ids],
        dtype=torch.bool,
    )
    probability = torch.full((len(patterns), 64), 1e-6, dtype=torch.float32)
    probability[:, 3] = 1.0
    probability /= probability.sum(dim=-1, keepdim=True)
    protocol = {
        "protocol_id": "mmw_id_stratified_block_v1",
        "protocol_fingerprint": "a" * 64,
        "split_manifest_hash": "b" * 64,
        "data_source_hash": "c" * 64,
        "window_config_hash": "d" * 64,
        "split_seed": 0,
        "block_size": 32,
        "weather_binding": True,
        "validation_sample_id_hash": "b" * 64,
        "validation_sample_count": len(sample_ids),
        "test_evaluated": False,
    }
    topology = {
        "id": "ula_dft_phase_cycle_v1",
        "descriptor_sha256": "e" * 64,
        "audit_sha256": "f" * 64,
    }
    evidence_path = tmp_path / "records.pt"
    torch.save(
        {
            "bounded_evaluation": False,
            "evidence_binding": {
                "checkpoint_sha256": checkpoint_sha256,
                "experiment_seed": 1,
                "data_protocol": protocol,
                "prototype_topology": topology,
            },
            "modalities": ("image", "radar", "gps", "lidar", "csi"),
            "pattern": patterns,
            "sample_id": row_ids,
            "labels": torch.full((len(patterns),), 3, dtype=torch.long),
            "fused_probability": probability,
            "final_prediction": probability.argmax(dim=-1),
            "available": available,
        },
        evidence_path,
    )
    evidence_sha256, evidence_size = checkpoint_file_digest(evidence_path)
    report = tmp_path / "matrix.json"
    report.write_text(
        json.dumps(
            {
                "claim_ineligible": True,
                "outer_test_accessed": False,
                "provenance": {
                    "checkpoint": {"sha256": checkpoint_sha256},
                    "sample_evidence": {
                        "path": str(evidence_path),
                        "sha256": evidence_sha256,
                        "size_bytes": evidence_size,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "experiment": {"seed": 1},
        "data_protocol": protocol,
        "loss": {"pcpf_temporal_risk": {"prototype_topology": topology}},
    }

    evidence = load_probe_evidence(matrix_report=report, checkpoint=checkpoint, cfg=cfg)

    assert len(evidence.sample_id) == len(FOUR_SENSING_MASKS) * len(sample_ids)
    assert set(evidence.pattern) == set(FOUR_SENSING_MASKS)
    assert evidence.source["pattern_available_sensing_count"]["missing_csi"] == 4
    assert evidence.source["pattern_available_sensing_count"]["radar_only"] == 1
    assert evidence.source["bounded_evaluation"] is False

    cfg["loss"]["pcpf_temporal_risk"]["prototype_topology"]["audit_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="topology identity"):
        load_probe_evidence(matrix_report=report, checkpoint=checkpoint, cfg=cfg)


def test_validation_power_index_is_bound_to_manifest_csv_and_sample_identity(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    rows = []
    for index, label in enumerate((3, 9)):
        power = np.full(64, 0.01)
        power[label] = 1.0
        path = root / f"power-{index}.txt"
        np.savetxt(path, power)
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "future_beam1": path.name,
                "future_beam_label1": label,
                "split": "validation",
            }
        )
    validation_csv = tmp_path / "validation.csv"
    with validation_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    validation_csv_sha256 = hashlib.sha256(validation_csv.read_bytes()).hexdigest()
    manifest = {
        "protocol_id": "mmw_id_stratified_block_v1",
        "protocol_version": 1,
        "protocol_fingerprint": "1" * 64,
        "data_source_hash": "2" * 64,
        "window_config_hash": "3" * 64,
        "split_seed": 0,
        "block_size": 32,
        "weather_binding": True,
        "domains": [
            {
                "id": "sunny/scene",
                "condition": "sunny",
                "scene": "scene",
                "data_root": str(root),
                "validation_split": str(validation_csv),
                "validation_csv_sha256": validation_csv_sha256,
                "validation_sample_count": len(rows),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    cfg = {
        "data_protocol": {
            **{key: manifest[key] for key in (
                "protocol_id",
                "protocol_version",
                "protocol_fingerprint",
                "data_source_hash",
                "window_config_hash",
                "split_seed",
                "block_size",
                "weather_binding",
            )},
            "split_manifest": str(manifest_path),
            "split_manifest_hash": manifest_sha256,
            "validation_role": "validation",
            "validation_sample_count": len(rows),
            "validation_sample_id_hash": _sha256_lines([row["sample_id"] for row in rows]),
            "test_evaluated": False,
            "outer_test_accessed": False,
            "outer_test_enabled": False,
        }
    }

    paths, labels = build_validation_power_index(cfg)

    assert len(paths) == len(labels) == 2
    assert labels["mmw:sunny:scene:validation:sample-1"] == 9
    validation_csv.write_text(validation_csv.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV SHA256 mismatch"):
        build_validation_power_index(cfg)


def test_tbcp_diagnostic_writes_15_mask_closed_loop_trace(tmp_path: Path) -> None:
    sample_ids = ("sample-a", "sample-b")
    labels = {"sample-a": 4, "sample-b": 60}
    power_paths = {}
    for sample_id, label in labels.items():
        delta = np.arange(64)
        distance = np.minimum((delta - label) % 64, (label - delta) % 64)
        power = np.exp(-np.square(distance) / 8.0) + 1e-5
        path = tmp_path / f"{sample_id}.txt"
        np.savetxt(path, power)
        power_paths[sample_id] = path
    patterns = tuple(pattern for pattern in FOUR_SENSING_MASKS for _ in sample_ids)
    row_ids = tuple(sample for _pattern in FOUR_SENSING_MASKS for sample in sample_ids)
    gt = np.asarray([labels[sample] for sample in row_ids], dtype=np.int64)
    pred = (gt + 1) % 64
    probability = np.full((len(row_ids), 64), 1e-6, dtype=np.float64)
    probability[np.arange(len(row_ids)), pred] = 1.0
    probability /= probability.sum(axis=-1, keepdims=True)
    pattern_counts = {name: int(sum(mask[:4])) for name, mask in FOUR_SENSING_MASKS.items()}
    likelihood = _synthetic_likelihood(tmp_path)
    likelihood_provenance = likelihood.metadata["provenance"]
    likelihood_path = tmp_path / "likelihood.npz"
    likelihood_record = save_topology_likelihood(likelihood, likelihood_path)
    evidence = ProbeEvidence(
        sample_id=row_ids,
        pattern=patterns,
        gt_beam=gt,
        pred_beam=pred,
        pred_prob=probability,
        pred_logits=np.log(probability),
        source={
            "bounded_evaluation": False,
            "pattern_available_sensing_count": pattern_counts,
            "checkpoint_path": "checkpoint.pth",
            "checkpoint_sha256": "a" * 64,
            "data_protocol": {
                key: likelihood_provenance[key]
                for key in (
                    "protocol_id",
                    "protocol_fingerprint",
                    "data_source_hash",
                    "window_config_hash",
                    "split_seed",
                    "block_size",
                    "weather_binding",
                )
            }
            | {
                "validation_sample_id_hash": "f" * 64,
                "validation_sample_count": len(sample_ids),
            },
            "prototype_topology": {
                "id": likelihood_provenance["topology_id"],
                "descriptor_sha256": likelihood_provenance["topology_descriptor_sha256"],
                "audit_sha256": likelihood_provenance["topology_audit_sha256"],
            },
        },
    )
    output = tmp_path / "tbcp"
    likelihood_source = {
        "path": str(likelihood_path),
        "sha256": likelihood_record["artifact_sha256"],
        "size_bytes": likelihood_path.stat().st_size,
        "artifact_fingerprint": likelihood.metadata["artifact_fingerprint"],
        "fit_split": "train",
    }

    result = run_tbcp_probe_diagnostic(
        evidence,
        power_paths=power_paths,
        indexed_labels=labels,
        likelihood=likelihood,
        likelihood_source=likelihood_source,
        output_dir=output,
        batch_size=7,
        include_diagonal_covariance_ablation=True,
        include_defense_experiments=True,
        include_batch_feedback_experiments=True,
    )

    assert result["config"]["primary_policy"]["name"] == "TBCP-7"
    assert result["config"]["outer_test_accessed"] is False
    groups = {row["group"] for row in result["group_summary"]}
    assert {"Full", "Drop-1 Worst", "Drop-2 Worst", "Single Worst"}.issubset(groups)
    with gzip.open(output / "per_sample_results.csv.gz", "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tbcp_rows = [row for row in rows if row["method"] == "TBCP-7"]
    diagonal_rows = [row for row in rows if row["method"] == TBCP_DIAGONAL_METHOD]
    assert len(tbcp_rows) == len(row_ids)
    assert len(diagonal_rows) == len(row_ids)
    assert len([row for row in rows if row["method"] == "Topology Open-loop Gain-7"]) == len(row_ids)
    assert len([row for row in rows if row["method"] == "TBCP-7"]) == len(row_ids)
    for budget in (3, 5, 9):
        for method in (f"TBCP-{budget}", f"Topology Open-loop Gain-{budget}", f"Posterior Top-{budget}"):
            selected = [row for row in rows if row["method"] == method]
            assert len(selected) == len(row_ids)
            assert all(len(json.loads(row["probe_indices"])) == budget for row in selected)
    assert all(len(json.loads(row["probe_indices"])) == 7 for row in tbcp_rows)
    assert all(len(json.loads(row["probe_measurements"])) == 7 for row in tbcp_rows)
    assert all(len(json.loads(row["posterior_map_trace"])) == 8 for row in tbcp_rows)
    assert result["config"]["covariance_ablation"]["modes"] == ["full", "diagonal"]
    assert result["config"]["defense_experiments"]["budgets"] == [3, 5, 7, 9]
    assert result["config"]["defense_experiments"]["primary_budget_remains_frozen"] == 7
    assert result["config"]["batch_feedback_experiments"]["enabled"] is True
    assert result["config"]["batch_feedback_experiments"]["schedules"]["Batch-TBCP-2+2+3"]["measurement_rounds"] == 3
    assert result["config"]["batch_feedback_experiments"]["schedules"]["Batch-TBCP-2+5"]["feedback_updates"] == 1
    assert result["config"]["batch_feedback_experiments"]["schedules"]["Batch-TBCP-3+4"] == {
        "batch_schedule": [3, 4],
        "probe_k": 7,
        "measurement_rounds": 2,
        "feedback_updates": 1,
    }
    batch_rows = [row for row in rows if row["method"] == "Batch-TBCP-2+2+3"]
    assert len(batch_rows) == len(row_ids)
    assert all(json.loads(row["batch_schedule"]) == [2, 2, 3] for row in batch_rows)
    assert all(row["measurement_rounds"] == "3" and row["feedback_rounds"] == "2" for row in batch_rows)
    assert all(len(json.loads(row["probe_indices"])) == 7 for row in batch_rows)
    balanced_rows = [row for row in rows if row["method"] == "Batch-TBCP-3+4"]
    assert len(balanced_rows) == len(row_ids)
    assert all(json.loads(row["batch_schedule"]) == [3, 4] for row in balanced_rows)
    assert all(row["measurement_rounds"] == "2" and row["feedback_rounds"] == "1" for row in balanced_rows)
    assert all(len(json.loads(row["probe_indices"])) == 7 for row in balanced_rows)
    assert "Batch-TBCP-3+4" in Path(result["report"]).read_text(encoding="utf-8")
    with gzip.open(output / "tbcp_trace.csv.gz", "rt", encoding="utf-8") as handle:
        trace_rows = list(csv.DictReader(handle))
    assert len(trace_rows) == len(row_ids) * 7

    run_paths = {}
    for seed in (1, 2, 3):
        checkpoint = tmp_path / f"checkpoint-seed{seed}.pth"
        torch.save({"seed": seed}, checkpoint)
        checkpoint_sha256, _ = checkpoint_file_digest(checkpoint)
        matrix_report = tmp_path / f"matrix-seed{seed}.json"
        matrix_report.write_text(
            json.dumps(
                {
                    "experiment_seed": seed,
                    "claim_ineligible": True,
                    "outer_test_accessed": False,
                    "provenance": {"checkpoint": {"sha256": checkpoint_sha256}},
                }
            ),
            encoding="utf-8",
        )
        payload = json.loads(json.dumps(result))
        payload["config"]["source"]["matrix_report"] = str(matrix_report)
        payload["config"]["source"]["checkpoint_path"] = str(checkpoint)
        payload["config"]["source"]["checkpoint_sha256"] = checkpoint_sha256
        run_path = tmp_path / f"result-seed{seed}.json"
        run_path.write_text(json.dumps(payload), encoding="utf-8")
        run_paths[seed] = run_path
    summary = summarize_tbcp_replays(run_paths, output_dir=tmp_path / "three-seed")
    assert summary["config"]["seeds"] == [1, 2, 3]
    assert summary["stability"]["Posterior5+Hill2"]["pattern_count"] == 15
    assert summary["stability"][TBCP_DIAGONAL_METHOD]["pattern_count"] == 15
    assert summary["stability"]["Topology Open-loop Gain-7"]["pattern_count"] == 15
    assert summary["stability"]["Batch-TBCP-2+2+3"]["pattern_count"] == 15
    assert summary["stability"]["Batch-TBCP-3+4"]["pattern_count"] == 15
    assert Path(summary["report"]).is_file()
    assert "Defense Budget Deltas" in Path(summary["report"]).read_text(encoding="utf-8")
    drift_payload = json.loads(run_paths[3].read_text(encoding="utf-8"))
    drift_payload["config"]["defense_experiments"]["budgets"] = [3, 5, 7]
    drift_path = tmp_path / "result-seed3-defense-drift.json"
    drift_path.write_text(json.dumps(drift_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="do not share one policy"):
        summarize_tbcp_replays(
            {1: run_paths[1], 2: run_paths[2], 3: drift_path},
            output_dir=tmp_path / "three-seed-defense-drift",
        )

    selected = select_probe_evidence_by_hash(evidence, samples_per_pattern=1)
    assert len(selected.sample_id) == 15
    assert len(set(selected.sample_id)) == 1
    with pytest.raises(ValueError, match="fewer than 3"):
        select_probe_evidence_by_hash(evidence, samples_per_pattern=3)
    robustness = run_tbcp_robustness_sensitivity(
        evidence,
        power_paths=power_paths,
        indexed_labels=labels,
        likelihood=likelihood,
        likelihood_source=likelihood_source,
        output_dir=tmp_path / "robustness",
        samples_per_pattern=2,
        batch_size=7,
    )
    assert len(robustness["config"]["scenario_grid"]) == 7
    assert {row["measurement_error_std_db"] for row in robustness["noise_summary"]} == {
        0.0,
        3.0,
        6.0,
    }
    assert all(row["condition_count"] == (1 if row["measurement_error_std_db"] == 0 else 3) for row in robustness["noise_summary"])
    trace_paths = [
        tmp_path / "robustness" / "scenarios" / name / "tbcp_trace.csv.gz"
        for name in ("sigma_00db_replica_0", "sigma_06db_replica_0")
    ]
    traces = []
    for path in trace_paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            traces.append(list(csv.DictReader(handle)))
    assert any(clean["measurement"] != noisy["measurement"] for clean, noisy in zip(*traces, strict=True))
    assert any(clean["probe_beam"] != noisy["probe_beam"] for clean, noisy in zip(*traces, strict=True))

    robustness_runs = {}
    for seed in (1, 2, 3):
        checkpoint = tmp_path / f"robustness-checkpoint-seed{seed}.pth"
        torch.save({"robustness_seed": seed}, checkpoint)
        checkpoint_sha256, _ = checkpoint_file_digest(checkpoint)
        payload = json.loads(json.dumps(robustness))
        payload["config"]["experiment_seed"] = seed
        payload["config"]["source"]["experiment_seed"] = seed
        payload["config"]["source"]["checkpoint_path"] = str(checkpoint)
        payload["config"]["source"]["checkpoint_sha256"] = checkpoint_sha256
        for row in payload["scenario_group_summary"]:
            row["experiment_seed"] = seed
        run_path = tmp_path / f"robustness-result-seed{seed}.json"
        run_path.write_text(json.dumps(payload), encoding="utf-8")
        robustness_runs[seed] = run_path
    robustness_summary = summarize_tbcp_robustness_replays(
        robustness_runs,
        output_dir=tmp_path / "robustness-three-seed",
    )
    assert robustness_summary["config"]["checkpoint_seeds"] == [1, 2, 3]
    assert robustness_summary["stability"]["sigma_6db_vs_Posterior5+Hill2"]["condition_count"] == 9
    assert Path(robustness_summary["report"]).is_file()
