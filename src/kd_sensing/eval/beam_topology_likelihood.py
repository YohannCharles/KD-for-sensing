from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


NUM_BEAMS = 64
TBCP_BUDGET = 7
TBCP_POLICY_VERSION = "topology_bayesian_closed_loop_gain_v1"
BATCH_TBCP_POLICY_VERSION = "topology_bayesian_batch_closed_loop_gain_v1"
# Fixed schedules are deliberately preregistered rather than validation-tuned.
# The balanced 3+4 split is a single additional feedback-round control.
BATCH_TBCP_SCHEDULES = ((2, 2, 3), (2, 5), (3, 4))
LIKELIHOOD_SCHEMA_VERSION = 1
COVARIANCE_JITTER_DB2 = 1e-6
COVARIANCE_MODE_FULL = "full"
COVARIANCE_MODE_DIAGONAL = "diagonal"
COVARIANCE_MODES = (COVARIANCE_MODE_FULL, COVARIANCE_MODE_DIAGONAL)
CALIBRATION_CONFIG = {
    "num_beams": NUM_BEAMS,
    "relative_db_multiplier": 10.0,
    "covariance_ddof": 1,
    "covariance_jitter_db2": COVARIANCE_JITTER_DB2,
    "reference": "official_train_argmax_beam",
    "strict_positive_power": True,
    "fitted_measurement_noise": False,
}

_REQUIRED_PROVENANCE = (
    "fit_split",
    "source_split",
    "protocol_id",
    "protocol_version",
    "protocol_fingerprint",
    "split_manifest",
    "split_manifest_hash",
    "split_manifest_file_sha256",
    "data_source_hash",
    "window_config_hash",
    "split_seed",
    "block_size",
    "weather_binding",
    "train_sample_count",
    "train_sample_id_hash",
    "train_stable_sample_id_hash",
    "topology_id",
    "topology_descriptor_sha256",
    "topology_audit_sha256",
    "test_evaluated",
    "outer_test_accessed",
    "source_components",
)


@dataclass(frozen=True)
class TopologyLikelihood:
    mean_db: np.ndarray
    covariance_db2: np.ndarray
    gain_kernel: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TBCPTrace:
    probe_indices: np.ndarray
    measurements: np.ndarray
    posterior_map: np.ndarray
    posterior_entropy: np.ndarray
    final_posterior: np.ndarray
    final_beam: np.ndarray


def validate_beam_probability(values: Sequence[float] | np.ndarray, num_beams: int = NUM_BEAMS) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim not in (1, 2) or probability.shape[-1] != int(num_beams):
        raise ValueError(f"pred_prob must have final dimension exactly {int(num_beams)}.")
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("pred_prob must be finite and non-negative.")
    if not np.allclose(probability.sum(axis=-1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("pred_prob must sum to one.")
    return probability


def fit_topology_likelihood(
    power_paths: Mapping[str, str | Path],
    labels: Mapping[str, int],
    protocol_sample_ids: Mapping[str, str],
    *,
    provenance: Mapping[str, Any],
    num_beams: int = NUM_BEAMS,
) -> TopologyLikelihood:
    beams = int(num_beams)
    if beams != NUM_BEAMS:
        raise ValueError(f"Topology likelihood is preregistered for {NUM_BEAMS} beams.")
    recorded = _validate_provenance(provenance)
    keys = set(power_paths)
    if not keys or keys != set(labels) or keys != set(protocol_sample_ids):
        raise ValueError("Train power, label, and protocol sample identities must have identical non-empty keys.")
    if len(keys) != int(recorded["train_sample_count"]):
        raise ValueError("Train topology likelihood sample count does not match protocol provenance.")
    observed_hash = _sha256_lines(protocol_sample_ids[key] for key in keys)
    if observed_hash != recorded["train_sample_id_hash"]:
        raise ValueError("Train topology likelihood sample identity does not match protocol provenance.")

    offsets = np.arange(beams, dtype=np.int64)
    aligned_gain: list[np.ndarray] = []
    power_digest = hashlib.sha256()
    for sample_id in sorted(keys):
        path = Path(power_paths[sample_id]).resolve()
        power = _load_train_power(path, beams)
        label = int(labels[sample_id])
        if label < 0 or label >= beams:
            raise ValueError(f"Train beam label must be in [0, {beams - 1}]: {sample_id}")
        if int(np.argmax(power)) != label:
            raise ValueError(f"Train beam label/power argmax mismatch: {sample_id}")
        power_digest.update(sample_id.encode("utf-8"))
        power_digest.update(np.ascontiguousarray(power, dtype=np.float64).tobytes())
        aligned_gain.append(power[(label + offsets) % beams] / float(power[label]))
    gain = np.stack(aligned_gain, axis=0)
    log_gain = 10.0 * np.log10(gain)
    mean_db = log_gain.mean(axis=0, dtype=np.float64)
    covariance_db2 = np.cov(log_gain, rowvar=False, ddof=1)
    gain_kernel = gain.mean(axis=0, dtype=np.float64)
    metadata = _artifact_metadata(
        mean_db=mean_db,
        covariance_db2=covariance_db2,
        gain_kernel=gain_kernel,
        provenance=recorded,
        train_power_content_sha256=power_digest.hexdigest(),
    )
    return _validated_likelihood(mean_db, covariance_db2, gain_kernel, metadata)


def train_power_content_sha256(
    power_paths: Mapping[str, str | Path],
    *,
    num_beams: int = NUM_BEAMS,
) -> str:
    """Hash the current train power values using the calibration artifact ordering."""
    beams = int(num_beams)
    if beams != NUM_BEAMS or not power_paths:
        raise ValueError(f"Train power content hashing requires non-empty {NUM_BEAMS}-beam inputs.")
    digest = hashlib.sha256()
    for sample_id in sorted(str(value) for value in power_paths):
        power = _load_train_power(Path(power_paths[sample_id]).resolve(), beams)
        digest.update(sample_id.encode("utf-8"))
        digest.update(np.ascontiguousarray(power, dtype=np.float64).tobytes())
    return digest.hexdigest()


def save_topology_likelihood(artifact: TopologyLikelihood, path: str | Path) -> dict[str, Any]:
    validated = _validated_likelihood(
        artifact.mean_db,
        artifact.covariance_db2,
        artifact.gain_kernel,
        artifact.metadata,
    )
    output = Path(path).resolve()
    if output.suffix != ".npz":
        raise ValueError("Topology likelihood artifact path must end in .npz.")
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise ValueError(f"Refusing to overwrite topology likelihood artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(validated.metadata, sort_keys=True, separators=(",", ":"))
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".npz", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            mean_db=validated.mean_db,
            covariance_db2=validated.covariance_db2,
            gain_kernel=validated.gain_kernel,
            metadata_json=np.asarray(metadata_json),
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = {
        "artifact": str(output),
        "artifact_sha256": _sha256_file(output),
        "metadata": validated.metadata,
    }
    sidecar_path = output.with_suffix(output.suffix + ".json")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{sidecar_path.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(sidecar_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(output), "sidecar": str(sidecar_path), **sidecar}


def load_topology_likelihood(
    path: str | Path,
    *,
    expected_provenance: Mapping[str, Any],
    expected_train_power_content_sha256: str,
) -> TopologyLikelihood:
    artifact_path = Path(path).resolve()
    sidecar_path = artifact_path.with_suffix(artifact_path.suffix + ".json")
    if not artifact_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"Topology likelihood artifact or sidecar not found: {artifact_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict) or not isinstance(sidecar.get("metadata"), dict):
        raise ValueError("Topology likelihood sidecar must contain metadata.")
    if sidecar.get("artifact") != str(artifact_path) or sidecar.get("artifact_sha256") != _sha256_file(artifact_path):
        raise ValueError("Topology likelihood artifact path or SHA256 does not match its sidecar.")
    try:
        with np.load(artifact_path, allow_pickle=False) as payload:
            mean_db = np.asarray(payload["mean_db"], dtype=np.float64)
            covariance_db2 = np.asarray(payload["covariance_db2"], dtype=np.float64)
            gain_kernel = np.asarray(payload["gain_kernel"], dtype=np.float64)
            raw_metadata = np.asarray(payload["metadata_json"]).item()
    except Exception as exc:
        raise ValueError(f"Failed to load topology likelihood artifact {artifact_path}: {exc}") from exc
    metadata = json.loads(str(raw_metadata))
    if metadata != sidecar["metadata"]:
        raise ValueError("Topology likelihood embedded metadata does not match its sidecar.")
    validated = _validated_likelihood(mean_db, covariance_db2, gain_kernel, metadata)
    expected = _validate_provenance(expected_provenance)
    if validated.metadata["provenance"] != expected:
        raise ValueError("Topology likelihood provenance does not match the current train protocol/topology binding.")
    expected_power_digest = str(expected_train_power_content_sha256).strip().lower()
    if (
        len(expected_power_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_power_digest)
        or validated.metadata["train_power_content_sha256"] != expected_power_digest
    ):
        raise ValueError("Topology likelihood train power content does not match the current train source.")
    return validated


def update_beam_belief(
    prior: Sequence[float] | np.ndarray,
    probe_indices: Sequence[int] | np.ndarray,
    measurements: Sequence[float] | np.ndarray,
    likelihood: TopologyLikelihood,
    *,
    measurement_error_std_db: float = 0.0,
    covariance_mode: str = COVARIANCE_MODE_FULL,
) -> np.ndarray:
    artifact = _validated_likelihood(
        likelihood.mean_db,
        likelihood.covariance_db2,
        likelihood.gain_kernel,
        likelihood.metadata,
    )
    return _update_beam_belief_validated(
        prior,
        probe_indices,
        measurements,
        artifact,
        measurement_error_std_db=measurement_error_std_db,
        covariance_mode=_validate_covariance_mode(covariance_mode),
    )


def _update_beam_belief_validated(
    prior: Sequence[float] | np.ndarray,
    probe_indices: Sequence[int] | np.ndarray,
    measurements: Sequence[float] | np.ndarray,
    artifact: TopologyLikelihood,
    *,
    measurement_error_std_db: float,
    covariance_mode: str,
) -> np.ndarray:
    probability = validate_beam_probability(prior)
    scalar = probability.ndim == 1
    batched_prior = probability[None, :] if scalar else probability
    indices = np.asarray(probe_indices, dtype=np.int64)
    power = np.asarray(measurements, dtype=np.float64)
    if scalar:
        indices = indices[None, :] if indices.ndim == 1 else indices
        power = power[None, :] if power.ndim == 1 else power
    if indices.ndim != 2 or power.shape != indices.shape or indices.shape[0] != batched_prior.shape[0]:
        raise ValueError("probe_indices and measurements must align with the prior batch.")
    if indices.shape[1] < 2:
        raise ValueError("Joint relative-gain likelihood requires at least two requested measurements.")
    if np.any(indices < 0) or np.any(indices >= NUM_BEAMS):
        raise ValueError(f"probe_indices must be in [0, {NUM_BEAMS - 1}].")
    if any(len(set(row.tolist())) != len(row) for row in indices):
        raise ValueError("probe_indices must be unique within each sample.")
    if not np.isfinite(power).all() or np.any(power <= 0.0):
        raise ValueError("requested measurements must be finite and strictly positive.")
    sigma_db = float(measurement_error_std_db)
    if not np.isfinite(sigma_db) or sigma_db < 0.0:
        raise ValueError("measurement_error_std_db must be finite and non-negative.")

    hypotheses = np.arange(NUM_BEAMS, dtype=np.int64)
    offsets = (indices[:, None, :] - hypotheses[None, :, None]) % NUM_BEAMS
    relative_mean = artifact.mean_db[offsets[..., 1:]] - artifact.mean_db[offsets[..., :1]]
    observed = 10.0 * np.log10(power[:, 1:] / power[:, :1])
    residual = observed[:, None, :] - relative_mean

    other = offsets[..., 1:]
    reference = offsets[..., 0]
    covariance = artifact.covariance_db2
    if covariance_mode == COVARIANCE_MODE_DIAGONAL:
        covariance = np.diag(np.diagonal(covariance))
    joint = covariance[other[..., :, None], other[..., None, :]]
    joint -= covariance[other, reference[..., None]][..., :, None]
    joint -= covariance[reference[..., None], other][..., None, :]
    joint += covariance[reference, reference][..., None, None]
    dimension = indices.shape[1] - 1
    if sigma_db > 0.0:
        joint += sigma_db**2
        diagonal = np.arange(dimension)
        joint[..., diagonal, diagonal] += sigma_db**2
    diagonal = np.arange(dimension)
    joint[..., diagonal, diagonal] += COVARIANCE_JITTER_DB2
    try:
        cholesky = np.linalg.cholesky(joint)
        whitened = np.linalg.solve(cholesky, residual[..., None])[..., 0]
    except np.linalg.LinAlgError as exc:
        raise ValueError("Relative-gain joint covariance is not positive definite after fixed jitter.") from exc
    log_determinant = 2.0 * np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)).sum(axis=-1)
    log_likelihood = -0.5 * (
        np.square(whitened).sum(axis=-1)
        + log_determinant
        + dimension * np.log(2.0 * np.pi)
    )
    log_posterior = np.log(np.clip(batched_prior, np.finfo(np.float64).tiny, 1.0)) + log_likelihood
    log_posterior -= log_posterior.max(axis=-1, keepdims=True)
    posterior = np.exp(log_posterior)
    posterior /= posterior.sum(axis=-1, keepdims=True)
    if not np.isfinite(posterior).all():
        raise ValueError("TBCP posterior update produced non-finite probability.")
    return posterior[0] if scalar else posterior


def select_expected_gain_candidates(
    posterior: Sequence[float] | np.ndarray,
    selected_indices: Sequence[int] | np.ndarray,
    gain_kernel: Sequence[float] | np.ndarray,
) -> int | np.ndarray:
    probability = validate_beam_probability(posterior)
    scalar = probability.ndim == 1
    batched = probability[None, :] if scalar else probability
    selected = np.asarray(selected_indices, dtype=np.int64)
    if scalar and selected.ndim == 1:
        selected = selected[None, :]
    if selected.ndim != 2 or selected.shape[0] != batched.shape[0] or not 0 < selected.shape[1] < NUM_BEAMS:
        raise ValueError("selected_indices must be a non-empty [B,T] array aligned with posterior.")
    if np.any(selected < 0) or np.any(selected >= NUM_BEAMS):
        raise ValueError(f"selected_indices must be in [0, {NUM_BEAMS - 1}].")
    if any(len(set(row.tolist())) != len(row) for row in selected):
        raise ValueError("selected_indices must be unique within each sample.")
    kernel = np.asarray(gain_kernel, dtype=np.float64)
    if kernel.shape != (NUM_BEAMS,) or not np.isfinite(kernel).all() or np.any(kernel < 0.0):
        raise ValueError(f"gain_kernel must contain {NUM_BEAMS} finite non-negative values.")

    hypotheses = np.arange(NUM_BEAMS, dtype=np.int64)
    current_offsets = (selected[:, None, :] - hypotheses[None, :, None]) % NUM_BEAMS
    current_gain = kernel[current_offsets].max(axis=-1)
    candidates = np.arange(NUM_BEAMS, dtype=np.int64)
    candidate_gain = kernel[(candidates[:, None] - hypotheses[None, :]) % NUM_BEAMS]
    terminal_gain = np.maximum(current_gain[:, None, :], candidate_gain[None, :, :])
    utility = np.einsum("bh,bch->bc", batched, terminal_gain, optimize=True)
    rows = np.arange(selected.shape[0])[:, None]
    utility[rows, selected] = -np.inf
    choice = np.argmax(utility, axis=-1).astype(np.int64)
    return int(choice[0]) if scalar else choice


def build_topology_open_loop_candidates(
    prior: Sequence[float] | np.ndarray,
    gain_kernel: Sequence[float] | np.ndarray,
    *,
    budget: int = TBCP_BUDGET,
) -> np.ndarray:
    probability = validate_beam_probability(prior)
    scalar = probability.ndim == 1
    batched_prior = probability[None, :] if scalar else probability
    probe_budget = _validate_probe_budget(budget)
    selected = np.argmax(batched_prior, axis=-1).astype(np.int64)[:, None]
    for _step in range(1, probe_budget):
        candidate = np.asarray(
            select_expected_gain_candidates(batched_prior, selected, gain_kernel),
            dtype=np.int64,
        ).reshape(batched_prior.shape[0])
        selected = np.concatenate((selected, candidate[:, None]), axis=1)
    return selected[0] if scalar else selected


def run_tbcp_batch(
    prior: Sequence[float] | np.ndarray,
    probe: Callable[[np.ndarray], Sequence[float] | np.ndarray],
    likelihood: TopologyLikelihood,
    *,
    budget: int = TBCP_BUDGET,
    measurement_error_std_db: float = 0.0,
    covariance_mode: str = COVARIANCE_MODE_FULL,
) -> TBCPTrace:
    artifact = _validated_likelihood(
        likelihood.mean_db,
        likelihood.covariance_db2,
        likelihood.gain_kernel,
        likelihood.metadata,
    )
    probability = validate_beam_probability(prior)
    mode = _validate_covariance_mode(covariance_mode)
    probe_budget = _validate_probe_budget(budget)
    batched_prior = probability[None, :] if probability.ndim == 1 else probability
    batch_size = batched_prior.shape[0]
    selected = np.empty((batch_size, 0), dtype=np.int64)
    measured = np.empty((batch_size, 0), dtype=np.float64)
    posterior = batched_prior.copy()
    map_trace = [np.argmax(posterior, axis=-1).astype(np.int64)]
    entropy_trace = [_normalized_entropy(posterior)]

    for step in range(probe_budget):
        if step == 0:
            candidate = np.argmax(batched_prior, axis=-1).astype(np.int64)
        else:
            candidate = np.asarray(
                select_expected_gain_candidates(posterior, selected, artifact.gain_kernel),
                dtype=np.int64,
            ).reshape(batch_size)
        response = np.asarray(probe(candidate.copy()), dtype=np.float64)
        if response.shape != (batch_size,) or not np.isfinite(response).all() or np.any(response <= 0.0):
            raise ValueError("probe callback must return one finite strictly positive requested measurement per sample.")
        selected = np.concatenate((selected, candidate[:, None]), axis=1)
        measured = np.concatenate((measured, response[:, None]), axis=1)
        if selected.shape[1] >= 2:
            posterior = _update_beam_belief_validated(
                batched_prior,
                selected,
                measured,
                artifact,
                measurement_error_std_db=measurement_error_std_db,
                covariance_mode=mode,
            )
        map_trace.append(np.argmax(posterior, axis=-1).astype(np.int64))
        entropy_trace.append(_normalized_entropy(posterior))
    final_beam = selected[np.arange(batch_size), np.argmax(measured, axis=-1)]
    return TBCPTrace(
        probe_indices=selected,
        measurements=measured,
        posterior_map=np.stack(map_trace, axis=1),
        posterior_entropy=np.stack(entropy_trace, axis=1),
        final_posterior=posterior,
        final_beam=final_beam,
    )


def run_batched_tbcp(
    prior: Sequence[float] | np.ndarray,
    probe: Callable[[np.ndarray], Sequence[float] | np.ndarray],
    likelihood: TopologyLikelihood,
    *,
    batch_schedule: Sequence[int],
    measurement_error_std_db: float = 0.0,
    covariance_mode: str = COVARIANCE_MODE_FULL,
) -> TBCPTrace:
    """Run a fixed batch-feedback approximation to TBCP-7.

    Candidates inside one batch are selected without observing that batch's
    measurements.  The posterior is recomputed once after the whole batch,
    so the only information barriers are the registered batch boundaries.
    """
    artifact = _validated_likelihood(
        likelihood.mean_db,
        likelihood.covariance_db2,
        likelihood.gain_kernel,
        likelihood.metadata,
    )
    probability = validate_beam_probability(prior)
    mode = _validate_covariance_mode(covariance_mode)
    schedule = tuple(batch_schedule)
    if schedule not in BATCH_TBCP_SCHEDULES:
        raise ValueError(
            "batch_schedule must be one of the preregistered schedules "
            f"{BATCH_TBCP_SCHEDULES}."
        )
    if sum(schedule) != TBCP_BUDGET:
        raise ValueError(f"batch_schedule must contain exactly {TBCP_BUDGET} probes.")
    if any(isinstance(size, bool) or int(size) != size or int(size) <= 0 for size in schedule):
        raise ValueError("batch_schedule entries must be positive integers.")
    batched_prior = probability[None, :] if probability.ndim == 1 else probability
    batch_size = batched_prior.shape[0]
    selected = np.empty((batch_size, 0), dtype=np.int64)
    measured = np.empty((batch_size, 0), dtype=np.float64)
    posterior = batched_prior.copy()
    map_trace = [np.argmax(posterior, axis=-1).astype(np.int64)]
    entropy_trace = [_normalized_entropy(posterior)]

    for width in schedule:
        pending = np.empty((batch_size, 0), dtype=np.int64)
        for _ in range(int(width)):
            available = np.concatenate((selected, pending), axis=1)
            if available.shape[1] == 0:
                candidate = np.argmax(batched_prior, axis=-1).astype(np.int64)
            else:
                candidate = np.asarray(
                    select_expected_gain_candidates(posterior, available, artifact.gain_kernel),
                    dtype=np.int64,
                ).reshape(batch_size)
            pending = np.concatenate((pending, candidate[:, None]), axis=1)
        response = np.asarray(probe(pending.copy()), dtype=np.float64)
        if response.shape != pending.shape or not np.isfinite(response).all() or np.any(response <= 0.0):
            raise ValueError(
                "batched probe callback must return one finite strictly positive "
                "requested measurement per sample and batch slot."
            )
        selected = np.concatenate((selected, pending), axis=1)
        measured = np.concatenate((measured, response), axis=1)
        posterior = _update_beam_belief_validated(
            batched_prior,
            selected,
            measured,
            artifact,
            measurement_error_std_db=measurement_error_std_db,
            covariance_mode=mode,
        )
        map_trace.append(np.argmax(posterior, axis=-1).astype(np.int64))
        entropy_trace.append(_normalized_entropy(posterior))

    final_beam = selected[np.arange(batch_size), np.argmax(measured, axis=-1)]
    return TBCPTrace(
        probe_indices=selected,
        measurements=measured,
        posterior_map=np.stack(map_trace, axis=1),
        posterior_entropy=np.stack(entropy_trace, axis=1),
        final_posterior=posterior,
        final_beam=final_beam,
    )


def _validate_probe_budget(budget: int) -> int:
    if isinstance(budget, bool):
        raise ValueError(f"Probe budget must be an integer in [1, {NUM_BEAMS}].")
    value = int(budget)
    if value != budget or value < 1 or value > NUM_BEAMS:
        raise ValueError(f"Probe budget must be an integer in [1, {NUM_BEAMS}].")
    return value


def build_posterior5_hill2_candidates(
    initial_indices: Sequence[int] | np.ndarray,
    initial_measurements: Sequence[float] | np.ndarray,
) -> np.ndarray:
    indices = np.asarray(initial_indices, dtype=np.int64)
    power = np.asarray(initial_measurements, dtype=np.float64)
    scalar = indices.ndim == 1
    if scalar:
        indices = indices[None, :]
        power = power[None, :]
    if indices.ndim != 2 or indices.shape[1] != 5 or power.shape != indices.shape:
        raise ValueError("Posterior5+Hill2 requires aligned [B,5] requested indices and measurements.")
    if np.any(indices < 0) or np.any(indices >= NUM_BEAMS) or any(
        len(set(row.tolist())) != 5 for row in indices
    ):
        raise ValueError("Posterior5+Hill2 initial indices must be five unique valid beams.")
    if not np.isfinite(power).all() or np.any(power <= 0.0):
        raise ValueError("Posterior5+Hill2 measurements must be finite and strictly positive.")
    result = np.empty((indices.shape[0], 2), dtype=np.int64)
    for row_index, (row_indices, row_power) in enumerate(zip(indices, power, strict=True)):
        center = int(row_indices[int(np.argmax(row_power))])
        used = set(int(value) for value in row_indices)
        additions: list[int] = []
        for distance in range(1, NUM_BEAMS):
            for delta in (-distance, distance):
                candidate = (center + delta) % NUM_BEAMS
                if candidate not in used:
                    additions.append(candidate)
                    used.add(candidate)
                if len(additions) == 2:
                    break
            if len(additions) == 2:
                break
        result[row_index] = additions
    return result[0] if scalar else result


def _artifact_metadata(
    *,
    mean_db: np.ndarray,
    covariance_db2: np.ndarray,
    gain_kernel: np.ndarray,
    provenance: Mapping[str, Any],
    train_power_content_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": LIKELIHOOD_SCHEMA_VERSION,
        "artifact_type": "train_only_ula_dft_relative_gain_likelihood",
        "policy_version": TBCP_POLICY_VERSION,
        "num_beams": NUM_BEAMS,
        "covariance_jitter_db2": COVARIANCE_JITTER_DB2,
        "relative_gain_reference": "official_train_argmax_beam",
        "train_power_content_sha256": train_power_content_sha256,
        "calibration_config": CALIBRATION_CONFIG,
        "calibration_config_sha256": _sha256_json(CALIBRATION_CONFIG),
        "provenance": dict(provenance),
        "array_sha256": _array_digest(mean_db, covariance_db2, gain_kernel),
    }
    return {**payload, "artifact_fingerprint": _sha256_json(payload)}


def _validated_likelihood(
    mean_db: Sequence[float] | np.ndarray,
    covariance_db2: Sequence[Sequence[float]] | np.ndarray,
    gain_kernel: Sequence[float] | np.ndarray,
    metadata: Mapping[str, Any],
) -> TopologyLikelihood:
    mean = np.asarray(mean_db, dtype=np.float64)
    covariance = np.asarray(covariance_db2, dtype=np.float64)
    kernel = np.asarray(gain_kernel, dtype=np.float64)
    if mean.shape != (NUM_BEAMS,) or covariance.shape != (NUM_BEAMS, NUM_BEAMS) or kernel.shape != (NUM_BEAMS,):
        raise ValueError("Topology likelihood arrays must have shapes [64], [64,64], and [64].")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all() or not np.isfinite(kernel).all():
        raise ValueError("Topology likelihood arrays must be finite.")
    if not np.allclose(covariance, covariance.T, atol=1e-8, rtol=0.0):
        raise ValueError("Topology likelihood covariance must be symmetric.")
    if float(np.linalg.eigvalsh(covariance).min()) < -1e-7:
        raise ValueError("Topology likelihood covariance must be positive semidefinite.")
    if np.any(kernel < 0.0) or np.any(kernel > 1.0 + 1e-9) or not np.isclose(kernel[0], 1.0, atol=1e-9):
        raise ValueError("Topology likelihood gain kernel must be normalized to one at offset zero.")
    recorded = dict(metadata)
    if (
        recorded.get("schema_version") != LIKELIHOOD_SCHEMA_VERSION
        or recorded.get("artifact_type") != "train_only_ula_dft_relative_gain_likelihood"
        or recorded.get("policy_version") != TBCP_POLICY_VERSION
        or recorded.get("num_beams") != NUM_BEAMS
        or recorded.get("covariance_jitter_db2") != COVARIANCE_JITTER_DB2
        or not isinstance(recorded.get("provenance"), dict)
        or len(str(recorded.get("train_power_content_sha256", ""))) != 64
        or recorded.get("calibration_config") != CALIBRATION_CONFIG
        or recorded.get("calibration_config_sha256") != _sha256_json(CALIBRATION_CONFIG)
    ):
        raise ValueError("Topology likelihood metadata schema or policy binding is invalid.")
    _validate_provenance(recorded["provenance"])
    payload = {key: value for key, value in recorded.items() if key != "artifact_fingerprint"}
    if recorded.get("array_sha256") != _array_digest(mean, covariance, kernel):
        raise ValueError("Topology likelihood array fingerprint does not match metadata.")
    if recorded.get("artifact_fingerprint") != _sha256_json(payload):
        raise ValueError("Topology likelihood metadata fingerprint is invalid.")
    mean.setflags(write=False)
    covariance.setflags(write=False)
    kernel.setflags(write=False)
    return TopologyLikelihood(mean_db=mean, covariance_db2=covariance, gain_kernel=kernel, metadata=recorded)


def _validate_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Topology likelihood provenance must be a mapping.")
    result = dict(value)
    missing = [key for key in _REQUIRED_PROVENANCE if key not in result]
    if missing:
        raise ValueError(f"Topology likelihood provenance is missing: {', '.join(missing)}.")
    if result["fit_split"] != "train" or result["source_split"] != "train":
        raise ValueError("Topology likelihood provenance must be fitted from the train role only.")
    if result["protocol_id"] != "mmw_id_stratified_block_v1":
        raise ValueError("Topology likelihood requires the formal MMW ID-block protocol.")
    if result["topology_id"] != "ula_dft_phase_cycle_v1":
        raise ValueError("Topology likelihood requires the audited ULA-DFT phase-cycle topology.")
    if result["test_evaluated"] is not False or result["outer_test_accessed"] is not False:
        raise ValueError("Topology likelihood provenance requires test and outer-test access to remain sealed.")
    for key in (
        "protocol_fingerprint",
        "split_manifest_hash",
        "split_manifest_file_sha256",
        "data_source_hash",
        "window_config_hash",
        "train_sample_id_hash",
        "train_stable_sample_id_hash",
        "topology_descriptor_sha256",
        "topology_audit_sha256",
    ):
        text = str(result[key]).strip().lower()
        if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
            raise ValueError(f"Topology likelihood provenance {key} must be a SHA256 digest.")
        result[key] = text
    if result["split_manifest_hash"] != result["split_manifest_file_sha256"]:
        raise ValueError("Topology likelihood split manifest content SHA256 does not match its binding.")
    if int(result["train_sample_count"]) <= 1:
        raise ValueError("Topology likelihood requires at least two train samples.")
    result["train_sample_count"] = int(result["train_sample_count"])
    result["protocol_version"] = int(result["protocol_version"])
    result["split_seed"] = int(result["split_seed"])
    result["block_size"] = int(result["block_size"])
    if result["protocol_version"] <= 0 or result["split_seed"] < 0 or result["block_size"] <= 0:
        raise ValueError("Topology likelihood protocol version, seed, and block size are invalid.")
    if result["weather_binding"] is not True:
        raise ValueError("Topology likelihood requires the formal weather binding.")
    manifest_path = Path(str(result["split_manifest"]))
    if not manifest_path.is_absolute():
        raise ValueError("Topology likelihood split_manifest provenance must be an absolute path.")
    if not manifest_path.is_file() or _sha256_file(manifest_path) != result["split_manifest_file_sha256"]:
        raise ValueError("Topology likelihood split manifest file does not match provenance.")
    components = result["source_components"]
    if not isinstance(components, list) or not components:
        raise ValueError("Topology likelihood provenance requires train source components.")
    component_paths: set[str] = set()
    component_domains: set[str] = set()
    component_count = 0
    for component in components:
        component_path = Path(str(component.get("path", ""))) if isinstance(component, dict) else Path()
        component_sha256 = str(component.get("sha256", "")).strip().lower() if isinstance(component, dict) else ""
        if (
            not isinstance(component, dict)
            or not str(component.get("domain_id", ""))
            or not component_path.is_absolute()
            or int(component.get("sample_count", 0)) <= 0
            or len(component_sha256) != 64
            or any(character not in "0123456789abcdef" for character in component_sha256)
            or not component_path.is_file()
            or _sha256_file(component_path) != component_sha256
        ):
            raise ValueError("Topology likelihood train source component provenance is invalid.")
        component["sha256"] = component_sha256
        component_count += int(component["sample_count"])
        component_paths.add(str(component_path))
        component_domains.add(str(component["domain_id"]))
    if (
        component_count != result["train_sample_count"]
        or len(component_paths) != len(components)
        or len(component_domains) != len(components)
    ):
        raise ValueError("Topology likelihood train source components do not match the train sample inventory.")
    return result


def _normalized_entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, np.finfo(np.float64).tiny, 1.0)
    return -(probability * np.log(clipped)).sum(axis=-1) / np.log(NUM_BEAMS)


def _validate_covariance_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in COVARIANCE_MODES:
        raise ValueError(f"covariance_mode must be one of {COVARIANCE_MODES}.")
    return mode


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _sha256_lines(values: Sequence[str] | Any) -> str:
    return hashlib.sha256("\n".join(sorted(str(value) for value in values)).encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_train_power(path: Path, num_beams: int) -> np.ndarray:
    try:
        power = np.asarray(np.loadtxt(path), dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise ValueError(f"Failed to load train beam power {path}: {exc}") from exc
    if power.size != num_beams or not np.isfinite(power).all() or np.any(power <= 0.0):
        raise ValueError(f"Train beam power must contain {num_beams} finite strictly positive values: {path}")
    return power


__all__ = [
    "BATCH_TBCP_POLICY_VERSION",
    "BATCH_TBCP_SCHEDULES",
    "COVARIANCE_JITTER_DB2",
    "COVARIANCE_MODE_DIAGONAL",
    "COVARIANCE_MODE_FULL",
    "COVARIANCE_MODES",
    "CALIBRATION_CONFIG",
    "LIKELIHOOD_SCHEMA_VERSION",
    "NUM_BEAMS",
    "TBCP_BUDGET",
    "TBCP_POLICY_VERSION",
    "TBCPTrace",
    "TopologyLikelihood",
    "build_posterior5_hill2_candidates",
    "build_topology_open_loop_candidates",
    "fit_topology_likelihood",
    "load_topology_likelihood",
    "run_tbcp_batch",
    "run_batched_tbcp",
    "save_topology_likelihood",
    "select_expected_gain_candidates",
    "train_power_content_sha256",
    "update_beam_belief",
    "validate_beam_probability",
]
