import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from kd_sensing.eval.beam_topology_likelihood import (
    BATCH_TBCP_SCHEDULES,
    COVARIANCE_JITTER_DB2,
    COVARIANCE_MODE_DIAGONAL,
    COVARIANCE_MODE_FULL,
    TBCP_BUDGET,
    build_topology_open_loop_candidates,
    fit_topology_likelihood,
    load_topology_likelihood,
    run_tbcp_batch,
    run_batched_tbcp,
    save_topology_likelihood,
    train_power_content_sha256,
    update_beam_belief,
)


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _fit_inputs(tmp_path: Path, count: int = 8):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    labels: dict[str, int] = {}
    protocol_ids: dict[str, str] = {}
    for index in range(count):
        label = (index * 7) % 64
        delta = np.arange(64)
        distance = np.minimum(delta, 64 - delta)
        width = 1.6 + 0.15 * index
        aligned = np.exp(-np.square(distance) / (2.0 * width**2)) + 1e-5
        aligned *= 1.0 + 0.015 * np.cos((index + 1) * delta * 2.0 * np.pi / 64.0)
        aligned = np.minimum(aligned, 0.98)
        aligned[0] = 1.0
        power = np.empty(64, dtype=np.float64)
        power[(label + delta) % 64] = aligned
        stable_id = f"stable-{index}"
        source_id = f"source-{index}"
        path = tmp_path / f"power-{index}.txt"
        np.savetxt(path, power)
        paths[stable_id] = path
        labels[stable_id] = label
        protocol_ids[stable_id] = source_id
    component = tmp_path / "train.csv"
    component.write_text("synthetic\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    provenance = {
        "fit_split": "train",
        "source_split": "train",
        "protocol_id": "mmw_id_stratified_block_v1",
        "protocol_version": 1,
        "protocol_fingerprint": "a" * 64,
        "split_manifest": str(manifest.resolve()),
        "split_manifest_hash": manifest_sha256,
        "split_manifest_file_sha256": manifest_sha256,
        "data_source_hash": "c" * 64,
        "window_config_hash": "d" * 64,
        "split_seed": 0,
        "block_size": 32,
        "weather_binding": True,
        "train_sample_count": count,
        "train_sample_id_hash": _sha256_lines(list(protocol_ids.values())),
        "train_stable_sample_id_hash": _sha256_lines(list(paths)),
        "topology_id": "ula_dft_phase_cycle_v1",
        "topology_descriptor_sha256": "e" * 64,
        "topology_audit_sha256": "f" * 64,
        "test_evaluated": False,
        "outer_test_accessed": False,
        "source_components": [
            {
                "domain_id": "synthetic/domain",
                "path": str(component.resolve()),
                "sha256": hashlib.sha256(component.read_bytes()).hexdigest(),
                "sample_count": count,
            }
        ],
    }
    return paths, labels, protocol_ids, provenance


def _artifact(tmp_path: Path):
    paths, labels, protocol_ids, provenance = _fit_inputs(tmp_path)
    return fit_topology_likelihood(paths, labels, protocol_ids, provenance=provenance), provenance


def test_train_topology_likelihood_round_trips_with_bound_provenance(tmp_path: Path) -> None:
    paths, labels, protocol_ids, provenance = _fit_inputs(tmp_path)
    artifact = fit_topology_likelihood(paths, labels, protocol_ids, provenance=provenance)
    output = tmp_path / "likelihood.npz"

    record = save_topology_likelihood(artifact, output)
    loaded = load_topology_likelihood(
        output,
        expected_provenance=provenance,
        expected_train_power_content_sha256=train_power_content_sha256(paths),
    )

    assert record["metadata"]["provenance"]["fit_split"] == "train"
    assert record["metadata"]["train_power_content_sha256"]
    assert np.array_equal(loaded.mean_db, artifact.mean_db)
    assert np.array_equal(loaded.covariance_db2, artifact.covariance_db2)
    assert np.array_equal(loaded.gain_kernel, artifact.gain_kernel)
    assert loaded.gain_kernel[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="overwrite"):
        save_topology_likelihood(artifact, output)


def test_topology_likelihood_rejects_non_train_before_reading_power(tmp_path: Path) -> None:
    paths, labels, protocol_ids, provenance = _fit_inputs(tmp_path)
    paths[next(iter(paths))] = tmp_path / "does-not-exist.txt"
    provenance["fit_split"] = "validation"

    with pytest.raises(ValueError, match="train role only"):
        fit_topology_likelihood(paths, labels, protocol_ids, provenance=provenance)


def test_topology_likelihood_rejects_label_and_artifact_drift(tmp_path: Path) -> None:
    paths, labels, protocol_ids, provenance = _fit_inputs(tmp_path)
    labels[next(iter(labels))] = 63
    with pytest.raises(ValueError, match="label/power argmax mismatch"):
        fit_topology_likelihood(paths, labels, protocol_ids, provenance=provenance)

    artifact, provenance = _artifact(tmp_path / "fresh")
    output = tmp_path / "likelihood.npz"
    save_topology_likelihood(artifact, output)
    sidecar = output.with_suffix(".npz.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["artifact_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        load_topology_likelihood(
            output,
            expected_provenance=provenance,
            expected_train_power_content_sha256=artifact.metadata["train_power_content_sha256"],
        )

def test_topology_likelihood_rejects_current_train_power_content_drift(tmp_path: Path) -> None:
    paths, labels, protocol_ids, provenance = _fit_inputs(tmp_path)
    artifact = fit_topology_likelihood(paths, labels, protocol_ids, provenance=provenance)
    output = tmp_path / "likelihood.npz"
    save_topology_likelihood(artifact, output)
    drifted = np.loadtxt(next(iter(paths.values())))
    drifted[1] *= 0.9
    np.savetxt(next(iter(paths.values())), drifted)

    with pytest.raises(ValueError, match="train power content"):
        load_topology_likelihood(
            output,
            expected_provenance=provenance,
            expected_train_power_content_sha256=train_power_content_sha256(paths),
        )


def test_joint_relative_db_update_matches_manual_correlated_gaussian(tmp_path: Path) -> None:
    artifact, _ = _artifact(tmp_path)
    prior = np.linspace(1.0, 2.0, 64, dtype=np.float64)
    prior /= prior.sum()
    indices = np.asarray([0, 3, 10], dtype=np.int64)
    true_beam = 5
    offsets = (indices - true_beam) % 64
    relative_mean = artifact.mean_db[offsets[1:]] - artifact.mean_db[offsets[0]]
    measurements = np.concatenate(([1.0], np.power(10.0, relative_mean / 10.0)))

    observed = update_beam_belief(prior, indices, measurements, artifact)

    log_likelihood = []
    covariance = artifact.covariance_db2
    z = 10.0 * np.log10(measurements[1:] / measurements[0])
    for hypothesis in range(64):
        local = (indices - hypothesis) % 64
        mean = artifact.mean_db[local[1:]] - artifact.mean_db[local[0]]
        joint = (
            covariance[np.ix_(local[1:], local[1:])]
            - covariance[local[1:], local[0]][:, None]
            - covariance[local[0], local[1:]][None, :]
            + covariance[local[0], local[0]]
        )
        joint += np.eye(2) * COVARIANCE_JITTER_DB2
        residual = z - mean
        sign, logdet = np.linalg.slogdet(joint)
        assert sign > 0
        log_likelihood.append(
            -0.5 * (residual @ np.linalg.solve(joint, residual) + logdet + 2 * np.log(2 * np.pi))
        )
    expected_log = np.log(prior) + np.asarray(log_likelihood)
    expected = np.exp(expected_log - expected_log.max())
    expected /= expected.sum()
    assert np.allclose(observed, expected, atol=1e-10, rtol=1e-10)


def test_diagonal_covariance_update_preserves_shared_reference_term(tmp_path: Path) -> None:
    artifact, _ = _artifact(tmp_path)
    prior = np.linspace(1.0, 2.0, 64, dtype=np.float64)
    prior /= prior.sum()
    indices = np.asarray([0, 3, 10], dtype=np.int64)
    measurements = np.asarray([1.0, 0.4, 0.2], dtype=np.float64)

    observed = update_beam_belief(
        prior,
        indices,
        measurements,
        artifact,
        covariance_mode=COVARIANCE_MODE_DIAGONAL,
    )

    covariance = np.diag(np.diagonal(artifact.covariance_db2))
    z = 10.0 * np.log10(measurements[1:] / measurements[0])
    log_likelihood = []
    for hypothesis in range(64):
        local = (indices - hypothesis) % 64
        mean = artifact.mean_db[local[1:]] - artifact.mean_db[local[0]]
        joint = (
            covariance[np.ix_(local[1:], local[1:])]
            - covariance[local[1:], local[0]][:, None]
            - covariance[local[0], local[1:]][None, :]
            + covariance[local[0], local[0]]
        )
        joint += np.eye(2) * COVARIANCE_JITTER_DB2
        residual = z - mean
        sign, logdet = np.linalg.slogdet(joint)
        assert sign > 0
        log_likelihood.append(
            -0.5 * (residual @ np.linalg.solve(joint, residual) + logdet + 2 * np.log(2 * np.pi))
        )
    expected_log = np.log(prior) + np.asarray(log_likelihood)
    expected = np.exp(expected_log - expected_log.max())
    expected /= expected.sum()
    assert np.allclose(observed, expected, atol=1e-10, rtol=1e-10)
    assert np.array_equal(
        update_beam_belief(prior, indices, measurements, artifact),
        update_beam_belief(
            prior,
            indices,
            measurements,
            artifact,
            covariance_mode=COVARIANCE_MODE_FULL,
        ),
    )
    with pytest.raises(ValueError, match="covariance_mode"):
        update_beam_belief(prior, indices, measurements, artifact, covariance_mode="independent")


def test_joint_relative_db_update_adds_shared_reference_measurement_error(tmp_path: Path) -> None:
    artifact, _ = _artifact(tmp_path)
    prior = np.full(64, 1.0 / 64.0, dtype=np.float64)
    indices = np.asarray([0, 3, 10], dtype=np.int64)
    measurements = np.asarray([1.0, 0.4, 0.2], dtype=np.float64)
    sigma_db = 2.0

    observed = update_beam_belief(
        prior,
        indices,
        measurements,
        artifact,
        measurement_error_std_db=sigma_db,
    )

    covariance = artifact.covariance_db2
    z = 10.0 * np.log10(measurements[1:] / measurements[0])
    log_likelihood = []
    for hypothesis in range(64):
        local = (indices - hypothesis) % 64
        mean = artifact.mean_db[local[1:]] - artifact.mean_db[local[0]]
        joint = (
            covariance[np.ix_(local[1:], local[1:])]
            - covariance[local[1:], local[0]][:, None]
            - covariance[local[0], local[1:]][None, :]
            + covariance[local[0], local[0]]
        )
        joint += sigma_db**2
        joint += np.eye(2) * (sigma_db**2 + COVARIANCE_JITTER_DB2)
        residual = z - mean
        sign, logdet = np.linalg.slogdet(joint)
        assert sign > 0
        log_likelihood.append(
            -0.5 * (residual @ np.linalg.solve(joint, residual) + logdet + 2 * np.log(2 * np.pi))
        )
    expected_log = np.log(prior) + np.asarray(log_likelihood)
    expected = np.exp(expected_log - expected_log.max())
    expected /= expected.sum()
    assert np.allclose(observed, expected, atol=1e-10, rtol=1e-10)


def test_tbcp7_queries_one_requested_beam_per_step_and_selects_only_measured(tmp_path: Path) -> None:
    artifact, _ = _artifact(tmp_path)
    prior = np.full((2, 64), 1e-4, dtype=np.float64)
    prior[:, 0] = 1.0
    prior /= prior.sum(axis=-1, keepdims=True)
    oracle = np.full((2, 64), 1e-4, dtype=np.float64)
    oracle[0, 4] = 1.0
    oracle[1, 60] = 1.0
    calls: list[np.ndarray] = []

    def probe(candidate: np.ndarray) -> np.ndarray:
        calls.append(candidate.copy())
        return oracle[np.arange(2), candidate]

    trace = run_tbcp_batch(prior, probe, artifact)

    assert len(calls) == TBCP_BUDGET
    assert trace.probe_indices.shape == (2, TBCP_BUDGET)
    assert trace.posterior_map.shape == (2, TBCP_BUDGET + 1)
    assert trace.posterior_entropy.shape == (2, TBCP_BUDGET + 1)
    assert all(len(set(row.tolist())) == TBCP_BUDGET for row in trace.probe_indices)
    expected_final = trace.probe_indices[np.arange(2), np.argmax(trace.measurements, axis=-1)]
    assert np.array_equal(trace.final_beam, expected_final)
    assert np.isfinite(trace.final_posterior).all()
    assert np.allclose(trace.final_posterior.sum(axis=-1), 1.0)


def test_tbcp_budget_and_open_loop_control_share_prefix_without_feedback(tmp_path: Path) -> None:
    artifact, _ = _artifact(tmp_path)
    prior = np.full((2, 64), 1e-4, dtype=np.float64)
    prior[0, 0] = 1.0
    prior[1, 63] = 1.0
    prior /= prior.sum(axis=-1, keepdims=True)
    oracle = np.full((2, 64), 1e-4, dtype=np.float64)
    oracle[0, 4] = 1.0
    oracle[1, 58] = 1.0

    def probe(candidate: np.ndarray) -> np.ndarray:
        return oracle[np.arange(2), candidate]

    trace9 = run_tbcp_batch(prior, probe, artifact, budget=9)
    trace7 = run_tbcp_batch(prior, probe, artifact, budget=7)
    open_loop = build_topology_open_loop_candidates(prior, artifact.gain_kernel, budget=9)

    assert np.array_equal(trace9.probe_indices[:, :7], trace7.probe_indices)
    assert np.allclose(trace9.measurements[:, :7], trace7.measurements)
    assert np.array_equal(trace9.posterior_map[:, :8], trace7.posterior_map)
    assert np.array_equal(open_loop[:, 0], np.argmax(prior, axis=-1))
    assert np.array_equal(open_loop[:, :2], trace9.probe_indices[:, :2])
    assert all(len(set(row.tolist())) == 9 for row in open_loop)


@pytest.mark.parametrize("schedule", BATCH_TBCP_SCHEDULES)
def test_batched_tbcp_observes_only_at_registered_batch_boundaries(
    tmp_path: Path, schedule: tuple[int, ...]
) -> None:
    artifact, _ = _artifact(tmp_path)
    prior = np.full((2, 64), 1e-4, dtype=np.float64)
    prior[0, 0] = 1.0
    prior[1, 63] = 1.0
    prior /= prior.sum(axis=-1, keepdims=True)
    oracle = np.full((2, 64), 1e-4, dtype=np.float64)
    oracle[0, 4] = 1.0
    oracle[1, 58] = 1.0
    calls: list[np.ndarray] = []

    def probe(candidates: np.ndarray) -> np.ndarray:
        calls.append(candidates.copy())
        return oracle[np.arange(2)[:, None], candidates]

    trace = run_batched_tbcp(prior, probe, artifact, batch_schedule=schedule)
    open_loop = build_topology_open_loop_candidates(prior, artifact.gain_kernel, budget=7)

    assert [call.shape for call in calls] == [(2, width) for width in schedule]
    budget = sum(schedule)
    assert trace.probe_indices.shape == (2, budget)
    assert trace.measurements.shape == (2, budget)
    assert trace.posterior_map.shape == (2, len(schedule) + 1)
    assert trace.posterior_entropy.shape == (2, len(schedule) + 1)
    assert np.array_equal(trace.probe_indices[:, :2], open_loop[:, :2])
    assert all(len(set(row.tolist())) == budget for row in trace.probe_indices)
    expected_final = trace.probe_indices[np.arange(2), np.argmax(trace.measurements, axis=-1)]
    assert np.array_equal(trace.final_beam, expected_final)


@pytest.mark.parametrize("schedule", ((7,), (2, 1, 4), (2, 2, 2), (2, 0, 5)))
def test_batched_tbcp_rejects_unregistered_schedule(
    tmp_path: Path, schedule: tuple[int, ...]
) -> None:
    artifact, _ = _artifact(tmp_path)
    prior = np.full(64, 1.0 / 64.0)
    with pytest.raises(ValueError, match="preregistered schedules"):
        run_batched_tbcp(
            prior,
            lambda candidates: np.ones(candidates.shape, dtype=np.float64),
            artifact,
            batch_schedule=schedule,
        )


@pytest.mark.parametrize("budget", (0, 65, 1.5, True))
def test_tbcp_and_open_loop_reject_invalid_budget(tmp_path: Path, budget: object) -> None:
    artifact, _ = _artifact(tmp_path)
    prior = np.full(64, 1.0 / 64.0)
    with pytest.raises(ValueError, match="Probe budget"):
        build_topology_open_loop_candidates(prior, artifact.gain_kernel, budget=budget)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Probe budget"):
        run_tbcp_batch(
            prior,
            lambda candidate: np.ones(candidate.shape[0], dtype=np.float64),
            artifact,
            budget=budget,  # type: ignore[arg-type]
        )
