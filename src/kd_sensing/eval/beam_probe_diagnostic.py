from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.models.beam_posterior import beam_posterior_statistics
from kd_sensing.eval.beam_topology_likelihood import (
    BATCH_TBCP_POLICY_VERSION,
    BATCH_TBCP_SCHEDULES,
    COVARIANCE_MODE_DIAGONAL,
    COVARIANCE_MODE_FULL,
    TBCP_BUDGET,
    TBCP_POLICY_VERSION,
    TopologyLikelihood,
    build_posterior5_hill2_candidates,
    build_topology_open_loop_candidates,
    run_batched_tbcp,
    run_tbcp_batch,
    validate_beam_probability,
)
from kd_sensing.evaluation.metrics import COMMUNICATION_SNR_DB
from kd_sensing.utils.checkpoint import (
    checkpoint_file_digest,
    load_torch_payload,
    validate_checkpoint_publication,
)


NUM_BEAMS = 64
DEFAULT_BUDGETS = (3, 5, 7, 9)
ADAPTIVE_BUDGET = 7
ADAPTIVE_SPACINGS = (1, 2, 4, 8)
ADAPTIVE_OFFSETS = {
    1: (-3, -2, -1, 0, 1, 2, 3),
    2: (-4, -2, -1, 0, 1, 2, 4),
    4: (-8, -4, -1, 0, 1, 4, 8),
    8: (-16, -8, -1, 0, 1, 8, 16),
}
PROBING_POLICY_VERSION = "sensing_guided_local7_posterior_mass_v1"
SEVERE_SINGLE_PATTERNS = ("image_only", "radar_only", "gps_only", "lidar_only")
EXPECTED_MODALITIES = ("image", "radar", "gps", "lidar")
EXPECTED_TOPOLOGY = "ula_dft_phase_cycle_v1"
EXPECTED_PROTOCOL = "mmw_id_stratified_block_v1"
ROBUSTNESS_VERSION = "tbcp7_synthetic_measurement_error_v1"
ROBUSTNESS_NOISE_MODEL_VERSION = "matched_lognormal_db_sha256_box_muller_v1"
ROBUSTNESS_NOISE_STD_DB = (0.0, 3.0, 6.0)
ROBUSTNESS_NOISE_REPLICAS = (0, 1, 2)
ROBUSTNESS_NOISE_SEED = 20260809
ROBUSTNESS_SAMPLE_COUNT = 512
ROBUSTNESS_FEEDBACK_OVERHEAD = (0.0, 0.005, 0.01, 0.02, 0.05)
TBCP_DIAGONAL_METHOD = "TBCP-7 Diagonal Covariance"
DEFENSE_EXPERIMENT_VERSION = "tbcp_open_loop_budget_curve_v1"
BATCH_FEEDBACK_EXPERIMENT_VERSION = "tbcp7_fixed_batch_feedback_v2"
BATCH_TBCP_METHODS = {
    "Batch-TBCP-2+2+3": BATCH_TBCP_SCHEDULES[0],
    "Batch-TBCP-2+5": BATCH_TBCP_SCHEDULES[1],
    "Batch-TBCP-3+4": BATCH_TBCP_SCHEDULES[2],
}
METHOD_FEEDBACK_UPDATES = {
    "Direct Prediction": 0,
    "Local-7": 0,
    "Adaptive Local-7": 0,
    "Posterior Top-7": 0,
    "Posterior5+Hill2": 1,
    "TBCP-7": 5,
    "Batch-TBCP-2+2+3": 2,
    "Batch-TBCP-2+5": 1,
    "Batch-TBCP-3+4": 1,
}


def build_local_candidates(pred_beam: int, k: int, num_beams: int = NUM_BEAMS) -> tuple[int, ...]:
    """Build a circular local scan using prediction and codebook metadata only."""
    beams, budget = _validate_scan_request(num_beams, k)
    center = _validate_beam(pred_beam, beams, "pred_beam")
    radius = budget // 2
    return _validate_candidates(((center + delta) % beams for delta in range(-radius, radius + 1)), budget, beams)


def build_uniform_candidates(k: int, offset: int, num_beams: int = NUM_BEAMS) -> tuple[int, ...]:
    """Build one translated, approximately uniform circular grid without sample information."""
    beams, budget = _validate_scan_request(num_beams, k)
    start = int(offset)
    if start < 0 or start >= beams:
        raise ValueError(f"offset must be in [0, {beams - 1}] for K={budget}, got {offset}.")
    return _validate_candidates(
        (((index * beams) // budget + start) % beams for index in range(budget)),
        budget,
        beams,
    )


def build_adaptive_local_candidates(
    pred_prob: Sequence[float],
    num_beams: int = NUM_BEAMS,
) -> tuple[tuple[int, ...], int]:
    """Choose a preregistered Local-7 spacing by predicted posterior mass."""
    probability = validate_beam_probability(pred_prob, num_beams)
    if probability.ndim != 1:
        raise ValueError("pred_prob must be a single beam posterior.")
    center = int(np.argmax(probability))
    best_candidates: tuple[int, ...] | None = None
    best_spacing = 0
    best_mass = -1.0
    for spacing in ADAPTIVE_SPACINGS:
        offsets = ADAPTIVE_OFFSETS[spacing]
        candidates = _validate_candidates(
            ((center + delta) % int(num_beams) for delta in offsets),
            ADAPTIVE_BUDGET,
            int(num_beams),
        )
        mass = float(probability[np.asarray(candidates, dtype=np.int64)].sum())
        if mass > best_mass:
            best_candidates = candidates
            best_spacing = spacing
            best_mass = mass
    if best_candidates is None:
        raise RuntimeError("Adaptive Local-7 candidate library is empty.")
    return best_candidates, best_spacing


def build_posterior_topk_candidates(
    pred_prob: Sequence[float],
    k: int = ADAPTIVE_BUDGET,
    num_beams: int = NUM_BEAMS,
) -> tuple[int, ...]:
    """Select the highest predicted beam probabilities with stable label ties."""
    beams, budget = _validate_scan_request(num_beams, k)
    probability = validate_beam_probability(pred_prob, beams)
    if probability.ndim != 1:
        raise ValueError("pred_prob must be a single beam posterior.")
    ranking = np.argsort(-probability, kind="stable")[:budget]
    return _validate_candidates(ranking.tolist(), budget, beams)


def build_oracle_local_candidates(gt_beam: int, k: int, num_beams: int = NUM_BEAMS) -> tuple[int, ...]:
    """Claim-ineligible upper bound; unlike Local-K, this explicitly centers on GT."""
    beams, budget = _validate_scan_request(num_beams, k)
    center = _validate_beam(gt_beam, beams, "gt_beam")
    radius = budget // 2
    return _validate_candidates(((center + delta) % beams for delta in range(-radius, radius + 1)), budget, beams)


def uniform_offsets(k: int, num_beams: int = NUM_BEAMS) -> tuple[int, ...]:
    beams, budget = _validate_scan_request(num_beams, k)
    del budget
    return tuple(range(beams))


class BeamProbeSimulator:
    """Offline radio ground truth whose public probe API reveals requested beams only.

    Full 64-beam vectors may be cached privately for I/O efficiency. Candidate
    policies never receive this object or its cache; they see only their allowed
    scalar inputs. The evaluation-only normalized-gain method cannot influence
    candidate selection.
    """

    def __init__(
        self,
        power_paths: Mapping[str, str | Path],
        *,
        num_beams: int = NUM_BEAMS,
        require_strict_positive: bool = False,
    ) -> None:
        self._num_beams = int(num_beams)
        if self._num_beams <= 0:
            raise ValueError("num_beams must be positive.")
        self._power_paths = {str(key): Path(value).resolve() for key, value in power_paths.items()}
        if not self._power_paths:
            raise ValueError("BeamProbeSimulator requires at least one validation sample.")
        self._require_strict_positive = bool(require_strict_positive)
        self._cache: dict[str, np.ndarray] = {}

    def probe(
        self,
        sample_id: str,
        beam_indices: Sequence[int],
        *,
        measurement_error_std_db: float = 0.0,
        noise_seed: int = ROBUSTNESS_NOISE_SEED,
        noise_replica: int = 0,
    ) -> tuple[float, ...]:
        indices = _validate_candidates(beam_indices, len(beam_indices), self._num_beams)
        power = self._radio_ground_truth(sample_id)
        sigma = float(measurement_error_std_db)
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError("measurement_error_std_db must be finite and non-negative.")
        if int(noise_seed) < 0 or int(noise_replica) < 0:
            raise ValueError("noise_seed and noise_replica must be non-negative integers.")
        observed = np.asarray([float(power[index]) for index in indices], dtype=np.float64)
        if sigma > 0.0:
            if np.any(observed <= 0.0):
                raise ValueError(f"Noisy probing requires strictly positive requested power: {sample_id}")
            error_db = sigma * np.asarray(
                [
                    _matched_standard_normal(
                        sample_id=str(sample_id),
                        beam_index=index,
                        noise_seed=int(noise_seed),
                        noise_replica=int(noise_replica),
                    )
                    for index in indices
                ],
                dtype=np.float64,
            )
            observed *= np.power(10.0, error_db / 10.0)
            if not np.isfinite(observed).all() or np.any(observed <= 0.0):
                raise ValueError("Synthetic dB measurement error produced invalid requested power.")
        return tuple(float(value) for value in observed)

    def normalized_gain(self, sample_id: str, selected_beam: int) -> float:
        """Evaluation-only metric; the full-vector denominator is never exposed."""
        index = _validate_beam(selected_beam, self._num_beams, "selected_beam")
        power = self._radio_ground_truth(sample_id)
        best = float(power.max())
        if best <= 0.0:
            raise ValueError(f"Beam-power vector has no positive oracle power: {sample_id}")
        return float(np.clip(float(power[index]) / best, np.finfo(np.float32).tiny, 1.0))

    def _radio_ground_truth(self, sample_id: str) -> np.ndarray:
        key = str(sample_id)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        path = self._power_paths.get(key)
        if path is None:
            raise KeyError(f"Unknown validation sample_id: {key}")
        try:
            values = np.asarray(np.loadtxt(path), dtype=np.float64).reshape(-1)
        except Exception as exc:
            raise ValueError(f"Failed to load radio ground truth {path}: {exc}") from exc
        if values.size != self._num_beams or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Radio ground truth must contain {self._num_beams} finite non-negative values: {path}")
        if self._require_strict_positive and np.any(values <= 0.0):
            raise ValueError(f"Noisy probing requires strictly positive beam power: {path}")
        values.setflags(write=False)
        self._cache[key] = values
        return values


@dataclass(frozen=True)
class ProbeEvidence:
    sample_id: tuple[str, ...]
    pattern: tuple[str, ...]
    gt_beam: np.ndarray
    pred_beam: np.ndarray
    pred_prob: np.ndarray
    pred_logits: np.ndarray
    source: dict[str, Any]


@dataclass
class _MetricAccumulator:
    count: int = 0
    correct: float = 0.0
    normalized_gain: float = 0.0
    coverage: float = 0.0
    coverage_count: int = 0
    covered_correct: float = 0.0
    spectral_efficiency_ratio: dict[int, float] = field(
        default_factory=lambda: {int(value): 0.0 for value in COMMUNICATION_SNR_DB}
    )

    def add(self, *, correct: bool, normalized_gain: float, covered: bool | None) -> None:
        self.count += 1
        self.correct += float(correct)
        self.normalized_gain += float(normalized_gain)
        for snr_db in self.spectral_efficiency_ratio:
            self.spectral_efficiency_ratio[snr_db] += _spectral_efficiency_ratio(
                float(normalized_gain), snr_db
            )
        if covered is not None:
            self.coverage += float(covered)
            self.coverage_count += 1
            self.covered_correct += float(bool(covered) and bool(correct))

    def summary(self) -> dict[str, Any]:
        if self.count <= 0:
            raise ValueError("Cannot summarize an empty probe accumulator.")
        result = {
            "sample_count": self.count,
            "top1": self.correct / self.count,
            "normalized_gain": self.normalized_gain / self.count,
            "gt_coverage": self.coverage / self.coverage_count if self.coverage_count else None,
            "selection_accuracy_given_coverage": (
                self.covered_correct / self.coverage if self.coverage > 0.0 else None
            ),
        }
        result.update(
            {
                f"spectral_efficiency_ratio_{snr_db}db": total / self.count
                for snr_db, total in self.spectral_efficiency_ratio.items()
            }
        )
        return result


def load_probe_evidence(
    *,
    matrix_report: str | Path,
    checkpoint: str | Path,
    cfg: Mapping[str, Any],
    max_samples_per_pattern: int | None = None,
) -> ProbeEvidence:
    report_path = Path(matrix_report).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Topology-predictor matrix report must be a mapping.")
    if report.get("claim_ineligible") is not True or report.get("outer_test_accessed") is not False:
        raise ValueError("Probe diagnostic requires claim-ineligible validation evidence with outer test sealed.")

    checkpoint_sha256, checkpoint_size = checkpoint_file_digest(checkpoint_path)
    checkpoint_payload = load_torch_payload(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint_payload, Mapping) or checkpoint_payload.get("checkpoint_role") != "validation_best":
        raise ValueError("Probe diagnostic requires a validation_best checkpoint.")
    validate_checkpoint_publication(checkpoint_path, payload=checkpoint_payload)
    model_metadata = checkpoint_payload.get("model_metadata")
    if (
        not isinstance(model_metadata, Mapping)
        or model_metadata.get("type") != "four_modal_topology_predictor"
        or tuple(model_metadata.get("modalities", ())) != EXPECTED_MODALITIES
    ):
        raise ValueError("Probe diagnostic requires a native four-modal topology-predictor checkpoint.")

    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Topology-predictor matrix report is missing provenance.")
    recorded_checkpoint = provenance.get("checkpoint")
    if not isinstance(recorded_checkpoint, Mapping) or recorded_checkpoint.get("sha256") != checkpoint_sha256:
        raise ValueError("Matrix report does not bind the requested checkpoint SHA256.")
    evidence_record = provenance.get("sample_evidence")
    if not isinstance(evidence_record, Mapping):
        raise ValueError("Topology-predictor matrix report is missing sample evidence provenance.")
    evidence_path = Path(str(evidence_record.get("path", ""))).resolve()
    evidence_sha256, evidence_size = checkpoint_file_digest(evidence_path)
    if evidence_sha256 != evidence_record.get("sha256") or evidence_size != int(evidence_record.get("size_bytes", -1)):
        raise ValueError("Topology-predictor sample evidence hash or size does not match the matrix report.")
    records = load_torch_payload(evidence_path, map_location="cpu")
    if not isinstance(records, Mapping) or records.get("bounded_evaluation") is not False:
        raise ValueError("Probe diagnostic requires an unbounded topology-predictor observation cache.")

    binding = records.get("evidence_binding")
    protocol = cfg.get("data_protocol")
    if not isinstance(binding, Mapping) or not isinstance(protocol, Mapping):
        raise ValueError("Probe evidence or config is missing data-protocol binding.")
    bound_protocol = binding.get("data_protocol")
    if (
        binding.get("checkpoint_sha256") != checkpoint_sha256
        or binding.get("experiment_seed") != int(cfg.get("experiment", {}).get("seed", -1))
        or not isinstance(bound_protocol, Mapping)
        or bound_protocol.get("protocol_id") != EXPECTED_PROTOCOL
        or bound_protocol.get("protocol_fingerprint") != protocol.get("protocol_fingerprint")
        or bound_protocol.get("split_manifest_hash") != protocol.get("split_manifest_hash")
        or bound_protocol.get("data_source_hash") != protocol.get("data_source_hash")
        or bound_protocol.get("window_config_hash") != protocol.get("window_config_hash")
        or bound_protocol.get("split_seed") != protocol.get("split_seed")
        or bound_protocol.get("block_size") != protocol.get("block_size")
        or bound_protocol.get("weather_binding") != protocol.get("weather_binding")
        or bound_protocol.get("validation_sample_id_hash") != protocol.get("validation_sample_id_hash")
        or bound_protocol.get("validation_sample_count") != protocol.get("validation_sample_count")
        or bound_protocol.get("test_evaluated") is not False
    ):
        raise ValueError("Probe evidence checkpoint, seed, protocol, or validation identity does not match config.")
    topology = binding.get("prototype_topology")
    expected_topology = _topology_binding(cfg)
    if not isinstance(topology, Mapping) or any(
        topology.get(key) != expected_topology[key]
        for key in ("id", "descriptor_sha256", "audit_sha256")
    ):
        raise ValueError("Probe evidence does not match the configured audited ULA-DFT topology identity.")
    if tuple(records.get("modalities", ())) != EXPECTED_MODALITIES:
        raise ValueError("Probe diagnostic requires the native four-modal evidence schema.")

    patterns = tuple(str(value) for value in records.get("pattern", ()))
    sample_ids = tuple(str(value) for value in records.get("sample_id", ()))
    labels = torch.as_tensor(records.get("labels"), dtype=torch.long)
    probability = torch.as_tensor(records.get("fused_probability"), dtype=torch.float32)
    final_prediction = torch.as_tensor(records.get("final_prediction"), dtype=torch.long)
    available = torch.as_tensor(records.get("available"), dtype=torch.bool)
    row_count = len(patterns)
    if not (
        row_count == len(sample_ids) == labels.numel() == probability.shape[0] == final_prediction.numel() == available.shape[0]
        and probability.ndim == 2
        and probability.shape[1] == NUM_BEAMS
        and available.shape[1] == len(EXPECTED_MODALITIES)
    ):
        raise ValueError("Probe evidence tensors and identity rows have inconsistent shapes.")
    if not bool(torch.isfinite(probability).all()) or bool((probability < 0).any()):
        raise ValueError("Probe evidence probabilities must be finite and non-negative.")
    if not torch.allclose(probability.sum(dim=-1), torch.ones(row_count), atol=1e-5, rtol=0.0):
        raise ValueError("Probe evidence probabilities are not normalized.")
    predictions = probability.argmax(dim=-1)
    if not torch.equal(predictions, final_prediction):
        raise ValueError("Probe evidence pred_beam does not match fused_probability argmax.")

    expected_pattern_count = int(bound_protocol.get("validation_sample_count", -1))
    if expected_pattern_count <= 0:
        raise ValueError("Probe evidence protocol has no valid validation sample count.")
    pattern_masks: dict[str, tuple[bool, ...]] = {}
    for pattern_name in dict.fromkeys(patterns):
        rows = [index for index, value in enumerate(patterns) if value == pattern_name]
        masks = {tuple(bool(value) for value in available[index].tolist()) for index in rows}
        if len(masks) != 1:
            raise ValueError(f"Evidence pattern {pattern_name} has inconsistent availability masks.")
        pattern_masks[pattern_name] = next(iter(masks))
    expected_masks = {
        tuple(bool(bits & (1 << index)) for index in range(4))
        for bits in range(1, 1 << 4)
    }
    selected_pattern_masks = {name: mask for name, mask in pattern_masks.items() if any(mask)}
    if set(selected_pattern_masks.values()) != expected_masks or len(selected_pattern_masks) != len(expected_masks):
        raise ValueError("Probe evidence must contain exactly 15 non-empty native four-sensing masks.")
    selected_pattern_names = tuple(
        sorted(selected_pattern_masks, key=lambda name: (-sum(selected_pattern_masks[name]), name))
    )

    selected_rows: list[int] = []
    reference_ids: tuple[str, ...] | None = None
    for pattern_name in selected_pattern_names:
        rows = [index for index, value in enumerate(patterns) if value == pattern_name]
        expected_mask = torch.as_tensor(selected_pattern_masks[pattern_name], dtype=torch.bool)
        if not rows or not bool(available[rows].eq(expected_mask).all()):
            raise ValueError(f"Evidence pattern {pattern_name} does not match its four-sensing availability mask.")
        pattern_ids = tuple(sample_ids[index] for index in rows)
        if len(pattern_ids) != expected_pattern_count or len(set(pattern_ids)) != expected_pattern_count:
            raise ValueError(
                f"Evidence pattern {pattern_name} does not contain the complete unique validation sample set."
            )
        if reference_ids is None:
            reference_ids = pattern_ids
        elif pattern_ids != reference_ids:
            raise ValueError("Four-sensing patterns do not share identical validation sample identity/order.")
        if max_samples_per_pattern is not None:
            if int(max_samples_per_pattern) <= 0:
                raise ValueError("max_samples_per_pattern must be positive.")
            rows = rows[: int(max_samples_per_pattern)]
        selected_rows.extend(rows)

    row_index = torch.as_tensor(selected_rows, dtype=torch.long)
    selected_probability = probability[row_index].numpy()
    return ProbeEvidence(
        sample_id=tuple(sample_ids[index] for index in selected_rows),
        pattern=tuple(patterns[index] for index in selected_rows),
        gt_beam=labels[row_index].numpy(),
        pred_beam=predictions[row_index].numpy(),
        pred_prob=selected_probability,
        pred_logits=np.log(np.clip(selected_probability, np.finfo(np.float32).tiny, 1.0)),
        source={
            "matrix_report": str(report_path),
            "evidence_path": str(evidence_path),
            "evidence_sha256": evidence_sha256,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_size_bytes": checkpoint_size,
            "experiment_seed": int(cfg.get("experiment", {}).get("seed", -1)),
            "data_protocol": dict(bound_protocol),
            "prototype_topology": dict(topology),
            "bounded_evaluation": max_samples_per_pattern is not None,
            "validation_stable_sample_id_hash": _sha256_lines(reference_ids or ()),
            "validation_sample_order_sha256": _sha256_ordered_lines(reference_ids or ()),
            "patterns": list(selected_pattern_names),
            "pattern_available_sensing_count": {
                name: int(sum(selected_pattern_masks[name])) for name in selected_pattern_names
            },
        },
    )


def select_probe_evidence_by_hash(
    evidence: ProbeEvidence,
    *,
    samples_per_pattern: int = ROBUSTNESS_SAMPLE_COUNT,
) -> ProbeEvidence:
    count = int(samples_per_pattern)
    patterns = tuple(dict.fromkeys(evidence.pattern))
    if count <= 0 or not patterns:
        raise ValueError("Hash-selected probe evidence requires a positive count and non-empty patterns.")
    by_pattern: dict[str, dict[str, int]] = {}
    reference_ids: set[str] | None = None
    for pattern in patterns:
        rows = {
            sample_id: index
            for index, (sample_id, observed_pattern) in enumerate(
                zip(evidence.sample_id, evidence.pattern, strict=True)
            )
            if observed_pattern == pattern
        }
        if len(rows) < count:
            raise ValueError(f"Pattern {pattern} has fewer than {count} unique samples.")
        if reference_ids is None:
            reference_ids = set(rows)
        elif set(rows) != reference_ids:
            raise ValueError("Hash-selected patterns do not share one complete sample identity set.")
        by_pattern[pattern] = rows
    ranked_ids = tuple(
        sorted(
            reference_ids or (),
            key=lambda sample_id: (hashlib.sha256(sample_id.encode("utf-8")).digest(), sample_id),
        )[:count]
    )
    selected_rows = [by_pattern[pattern][sample_id] for pattern in patterns for sample_id in ranked_ids]
    row_index = np.asarray(selected_rows, dtype=np.int64)
    source = dict(evidence.source)
    source.update(
        {
            "bounded_evaluation": True,
            "selection_method": "lowest_sha256_stable_sample_id_v1",
            "samples_per_pattern": count,
            "selected_stable_sample_id_hash": _sha256_lines(ranked_ids),
            "selected_sample_order_sha256": _sha256_ordered_lines(ranked_ids),
        }
    )
    return ProbeEvidence(
        sample_id=tuple(evidence.sample_id[index] for index in selected_rows),
        pattern=tuple(evidence.pattern[index] for index in selected_rows),
        gt_beam=np.asarray(evidence.gt_beam)[row_index],
        pred_beam=np.asarray(evidence.pred_beam)[row_index],
        pred_prob=np.asarray(evidence.pred_prob)[row_index],
        pred_logits=np.asarray(evidence.pred_logits)[row_index],
        source=source,
    )


def build_train_power_index(
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, int], dict[str, str], dict[str, Any]]:
    paths, labels, protocol_sample_ids, source = _build_protocol_power_index(cfg, role="train")
    protocol = source["protocol"]
    topology = _topology_binding(cfg)
    provenance = {
        "fit_split": "train",
        "source_split": "train",
        "protocol_id": protocol["protocol_id"],
        "protocol_version": int(protocol["protocol_version"]),
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "split_manifest": source["split_manifest"],
        "split_manifest_hash": protocol["split_manifest_hash"],
        "split_manifest_file_sha256": source["split_manifest_file_sha256"],
        "data_source_hash": protocol["data_source_hash"],
        "window_config_hash": protocol["window_config_hash"],
        "split_seed": int(protocol["split_seed"]),
        "block_size": int(protocol["block_size"]),
        "weather_binding": bool(protocol["weather_binding"]),
        "train_sample_count": source["sample_count"],
        "train_sample_id_hash": protocol["train_sample_id_hash"],
        "train_stable_sample_id_hash": source["stable_sample_id_hash"],
        "topology_id": topology["id"],
        "topology_descriptor_sha256": topology["descriptor_sha256"],
        "topology_audit_sha256": topology.get("audit_sha256"),
        "test_evaluated": False,
        "outer_test_accessed": False,
        "source_components": source["source_components"],
    }
    return paths, labels, protocol_sample_ids, provenance


def build_validation_power_index(cfg: Mapping[str, Any]) -> tuple[dict[str, Path], dict[str, int]]:
    paths, labels, _protocol_sample_ids, _source = _build_protocol_power_index(cfg, role="validation")
    return paths, labels


def _build_protocol_power_index(
    cfg: Mapping[str, Any],
    *,
    role: str,
) -> tuple[dict[str, Path], dict[str, int], dict[str, str], dict[str, Any]]:
    if role not in {"train", "validation"}:
        raise ValueError("Beam-power indexing is restricted to train or validation; test remains sealed.")
    protocol = cfg.get("data_protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("protocol_id") != EXPECTED_PROTOCOL
        or protocol.get(f"{role}_role") != role
        or protocol.get("test_evaluated") is not False
        or protocol.get("outer_test_accessed", False) is not False
        or protocol.get("outer_test_enabled", False) is not False
    ):
        raise ValueError(f"Beam-power indexing requires the sealed ID-block {role} protocol.")
    manifest_path = Path(str(protocol.get("split_manifest", protocol.get("path", "")))).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MMW split manifest not found: {manifest_path}")
    manifest_sha256 = _sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, Mapping)
        or manifest_sha256 != protocol.get("split_manifest_hash")
        or manifest.get("protocol_id") != protocol.get("protocol_id")
        or int(manifest.get("protocol_version", -1)) != int(protocol.get("protocol_version", -2))
        or manifest.get("protocol_fingerprint") != protocol.get("protocol_fingerprint")
        or manifest.get("data_source_hash") != protocol.get("data_source_hash")
        or manifest.get("window_config_hash") != protocol.get("window_config_hash")
        or int(manifest.get("split_seed", -1)) != int(protocol.get("split_seed", -2))
        or int(manifest.get("block_size", -1)) != int(protocol.get("block_size", -2))
        or manifest.get("weather_binding") is not protocol.get("weather_binding")
    ):
        raise ValueError(f"MMW {role} manifest does not match the configured protocol identity.")
    domains = manifest.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError(f"MMW {role} manifest has no domain bindings.")

    paths: dict[str, Path] = {}
    labels: dict[str, int] = {}
    protocol_sample_ids: dict[str, str] = {}
    source_components: list[dict[str, Any]] = []
    for domain in domains:
        if not isinstance(domain, Mapping):
            raise ValueError(f"MMW {role} manifest domains must be mappings.")
        condition = str(domain.get("condition", ""))
        scene = str(domain.get("scene", ""))
        root = Path(str(domain.get("data_root", ""))).resolve()
        csv_path = Path(str(domain.get(f"{role}_split", ""))).resolve()
        expected_sha256 = str(domain.get(f"{role}_csv_sha256", ""))
        expected_count = int(domain.get(f"{role}_sample_count", -1))
        if not condition or not scene or not root.is_dir() or not csv_path.is_file():
            raise ValueError(f"Incomplete {role} domain binding: {domain.get('id', '<unknown>')}")
        actual_sha256 = _sha256_file(csv_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"MMW {role} CSV SHA256 mismatch: {csv_path}")
        frame = pd.read_csv(csv_path, usecols=["sample_id", "future_beam1", "future_beam_label1", "split"])
        if len(frame) != expected_count or set(frame["split"].astype(str)) != {role}:
            raise ValueError(f"MMW {role} CSV count or split role mismatch: {csv_path}")
        source_components.append(
            {
                "domain_id": str(domain.get("id", "")),
                "path": str(csv_path),
                "sha256": actual_sha256,
                "sample_count": len(frame),
            }
        )
        for row in frame.itertuples(index=False):
            source_sample_id = str(row.sample_id)
            stable_sample_id = f"mmw:{condition}:{scene}:{role}:{source_sample_id}"
            if stable_sample_id in paths:
                raise ValueError(f"Duplicate {role} sample identity: {stable_sample_id}")
            label = int(row.future_beam_label1)
            _validate_beam(label, NUM_BEAMS, "future_beam_label1")
            power_path = joined_resource(root, str(row.future_beam1)).resolve()
            if not power_path.is_file():
                raise FileNotFoundError(f"MMW {role} beam-power source not found: {power_path}")
            paths[stable_sample_id] = power_path
            labels[stable_sample_id] = label
            protocol_sample_ids[stable_sample_id] = source_sample_id
    expected_total = int(protocol.get(f"{role}_sample_count", -1))
    expected_sample_hash = str(protocol.get(f"{role}_sample_id_hash", ""))
    if len(paths) != expected_total or _sha256_lines(protocol_sample_ids.values()) != expected_sample_hash:
        raise ValueError(f"{role.capitalize()} power index count or sample identity does not match protocol audit.")
    return paths, labels, protocol_sample_ids, {
        "protocol": protocol,
        "split_manifest": str(manifest_path),
        "split_manifest_file_sha256": manifest_sha256,
        "sample_count": expected_total,
        "stable_sample_id_hash": _sha256_lines(paths),
        "stable_sample_order_sha256": _sha256_ordered_lines(paths),
        "source_components": source_components,
    }


def run_probe_diagnostic(
    evidence: ProbeEvidence,
    *,
    power_paths: Mapping[str, str | Path],
    indexed_labels: Mapping[str, int],
    output_dir: str | Path,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite a non-empty diagnostic directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    scan_budgets = tuple(int(value) for value in budgets)
    if scan_budgets != DEFAULT_BUDGETS:
        raise ValueError(f"Diagnostic budgets are preregistered as {DEFAULT_BUDGETS}.")
    evidence_ids = set(evidence.sample_id)
    power_ids = set(power_paths)
    if evidence_ids - power_ids:
        raise ValueError("Prediction evidence contains samples absent from the validation power index.")
    if evidence.source.get("bounded_evaluation") is not True and evidence_ids != power_ids:
        raise ValueError("Unbounded prediction evidence does not exactly match the validation power index.")
    for sample_id, label in zip(evidence.sample_id, evidence.gt_beam, strict=True):
        if int(indexed_labels[sample_id]) != int(label):
            raise ValueError(f"Evidence/validation label mismatch for {sample_id}.")

    posterior_tensors = beam_posterior_statistics(
        torch.as_tensor(evidence.pred_prob, dtype=torch.float32),
        num_beams=NUM_BEAMS,
        top_l=ADAPTIVE_BUDGET,
    )
    if not np.array_equal(posterior_tensors["beam_map"].cpu().numpy(), evidence.pred_beam):
        raise ValueError("Posterior statistics MAP does not match prediction evidence.")
    posterior_arrays = {name: value.cpu().numpy() for name, value in posterior_tensors.items()}

    simulator = BeamProbeSimulator(power_paths)
    accumulators: dict[tuple[str, str, int, int | None], _MetricAccumulator] = defaultdict(_MetricAccumulator)
    distance_counts = {pattern: np.zeros(5, dtype=np.int64) for pattern in SEVERE_SINGLE_PATTERNS}
    pattern_counts = {pattern: 0 for pattern in SEVERE_SINGLE_PATTERNS}
    ledger_path = output / "per_sample_results.csv.gz"
    prediction_path = output / "sensing_predictions.csv.gz"
    spacing_rows: list[dict[str, Any]] = []
    ledger_fields = (
        "sample_id",
        "missing_pattern",
        "gt_beam",
        "pred_beam",
        "strategy",
        "K",
        "uniform_offset",
        "adaptive_spacing",
        "selected_posterior_mass",
        "beam_circular_mean",
        "beam_circular_variance",
        "beam_variance",
        "beam_spread",
        "beam_normalized_entropy",
        "probe_indices",
        "final_beam",
        "correct",
        "normalized_gain",
        "gt_covered",
    )
    with gzip.open(ledger_path, "wt", encoding="utf-8", newline="") as ledger_handle, gzip.open(
        prediction_path, "wt", encoding="utf-8", newline=""
    ) as prediction_handle:
        ledger = csv.DictWriter(ledger_handle, fieldnames=ledger_fields)
        ledger.writeheader()
        prediction = csv.DictWriter(
            prediction_handle,
            fieldnames=(
                "sample_id",
                "missing_pattern",
                "gt_beam",
                "pred_beam",
                "beam_circular_mean",
                "beam_resultant_length",
                "beam_circular_variance",
                "beam_variance",
                "beam_spread",
                "beam_normalized_entropy",
                "beam_top_indices",
                "pred_logits",
                "pred_prob",
            ),
        )
        prediction.writeheader()
        for index, (sample_id, pattern, raw_gt, raw_pred) in enumerate(
            zip(evidence.sample_id, evidence.pattern, evidence.gt_beam, evidence.pred_beam, strict=True)
        ):
            gt_beam, pred_beam = int(raw_gt), int(raw_pred)
            posterior_row = {
                "beam_circular_mean": float(posterior_arrays["beam_circular_mean"][index]),
                "beam_resultant_length": float(posterior_arrays["beam_resultant_length"][index]),
                "beam_circular_variance": float(posterior_arrays["beam_circular_variance"][index]),
                "beam_variance": float(posterior_arrays["beam_variance"][index]),
                "beam_spread": float(posterior_arrays["beam_spread"][index]),
                "beam_normalized_entropy": float(posterior_arrays["beam_normalized_entropy"][index]),
            }
            prediction.writerow(
                {
                    "sample_id": sample_id,
                    "missing_pattern": pattern,
                    "gt_beam": gt_beam,
                    "pred_beam": pred_beam,
                    **posterior_row,
                    "beam_top_indices": json.dumps(
                        posterior_arrays["beam_top_indices"][index].tolist(),
                        separators=(",", ":"),
                    ),
                    "pred_logits": json.dumps(evidence.pred_logits[index].tolist(), separators=(",", ":")),
                    "pred_prob": json.dumps(evidence.pred_prob[index].tolist(), separators=(",", ":")),
                }
            )
            distance = min(abs(pred_beam - gt_beam), NUM_BEAMS - abs(pred_beam - gt_beam))
            pattern_counts[pattern] += 1
            for threshold in range(1, 6):
                distance_counts[pattern][threshold - 1] += int(distance <= threshold)

            _record_strategy(
                ledger,
                accumulators,
                simulator,
                sample_id=sample_id,
                pattern=pattern,
                gt_beam=gt_beam,
                pred_beam=pred_beam,
                strategy="Direct Prediction",
                k=0,
                candidates=(),
                offset=None,
                final_beam=pred_beam,
                covered=None,
                posterior_statistics=posterior_row,
            )
            for budget in scan_budgets:
                local = build_local_candidates(pred_beam, budget)
                _probe_and_record(
                    ledger,
                    accumulators,
                    simulator,
                    sample_id=sample_id,
                    pattern=pattern,
                    gt_beam=gt_beam,
                    pred_beam=pred_beam,
                    strategy="Local Scan",
                    k=budget,
                    candidates=local,
                    offset=None,
                    posterior_statistics=posterior_row,
                )
                if budget == ADAPTIVE_BUDGET:
                    adaptive, spacing = build_adaptive_local_candidates(evidence.pred_prob[index])
                    selected_mass = float(evidence.pred_prob[index][np.asarray(adaptive, dtype=np.int64)].sum())
                    spacing_rows.append(
                        {
                            "group": pattern,
                            "adaptive_spacing": spacing,
                            "beam_spread": posterior_row["beam_spread"],
                            "beam_normalized_entropy": posterior_row["beam_normalized_entropy"],
                            "selected_posterior_mass": selected_mass,
                        }
                    )
                    _probe_and_record(
                        ledger,
                        accumulators,
                        simulator,
                        sample_id=sample_id,
                        pattern=pattern,
                        gt_beam=gt_beam,
                        pred_beam=pred_beam,
                        strategy="Adaptive Local",
                        k=budget,
                        candidates=adaptive,
                        offset=None,
                        adaptive_spacing=spacing,
                        selected_posterior_mass=selected_mass,
                        posterior_statistics=posterior_row,
                    )
                    posterior_top7 = build_posterior_topk_candidates(evidence.pred_prob[index], budget)
                    _probe_and_record(
                        ledger,
                        accumulators,
                        simulator,
                        sample_id=sample_id,
                        pattern=pattern,
                        gt_beam=gt_beam,
                        pred_beam=pred_beam,
                        strategy="Posterior Top-K",
                        k=budget,
                        candidates=posterior_top7,
                        offset=None,
                        posterior_statistics=posterior_row,
                    )
                oracle = build_oracle_local_candidates(gt_beam, budget)
                _probe_and_record(
                    ledger,
                    accumulators,
                    simulator,
                    sample_id=sample_id,
                    pattern=pattern,
                    gt_beam=gt_beam,
                    pred_beam=pred_beam,
                    strategy="Oracle Local",
                    k=budget,
                    candidates=oracle,
                    offset=None,
                    posterior_statistics=posterior_row,
                )
                for offset in uniform_offsets(budget):
                    uniform = build_uniform_candidates(budget, offset)
                    _probe_and_record(
                        ledger,
                        accumulators,
                        simulator,
                        sample_id=sample_id,
                        pattern=pattern,
                        gt_beam=gt_beam,
                        pred_beam=pred_beam,
                        strategy="Global Uniform",
                        k=budget,
                        candidates=uniform,
                        offset=offset,
                        posterior_statistics=posterior_row,
                    )
            full = tuple(range(NUM_BEAMS))
            _probe_and_record(
                ledger,
                accumulators,
                simulator,
                sample_id=sample_id,
                pattern=pattern,
                gt_beam=gt_beam,
                pred_beam=pred_beam,
                strategy="Full Sweep",
                k=NUM_BEAMS,
                candidates=full,
                offset=None,
                posterior_statistics=posterior_row,
            )

    per_mask, uniform_summary = _summarize_accumulators(accumulators, scan_budgets)
    summary_rows = _aggregate_single_rows(per_mask, scan_budgets)
    _write_csv(output / "per_mask_summary.csv", per_mask)
    _write_csv(output / "summary.csv", summary_rows)
    _write_csv(output / "uniform_offset_summary.csv", uniform_summary)
    spacing_summary = _summarize_adaptive_spacing(spacing_rows)
    _write_csv(output / "adaptive_spacing_summary.csv", spacing_summary)

    sanity = _distance_sanity(distance_counts, pattern_counts)
    (output / "distance_sanity.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    config = {
        "schema_version": 2,
        "evaluation_scope": "validation_only_sensing_guided_probe_feasibility",
        "claim_ineligible": True,
        "outer_test_accessed": False,
        "model_trained_or_updated": False,
        "prediction_evidence_reused": True,
        "patterns": list(SEVERE_SINGLE_PATTERNS),
        "budgets": list(scan_budgets),
        "num_beams": NUM_BEAMS,
        "local_topology": "audited_ula_dft_phase_cycle_modulo_64_not_world_azimuth_ring",
        "probing_policy": {
            "version": PROBING_POLICY_VERSION,
            "budget": ADAPTIVE_BUDGET,
            "center": "argmax_predicted_probability",
            "adaptive_selection": "maximum_candidate_posterior_mass_tie_break_smallest_spacing",
            "adaptive_spacings": list(ADAPTIVE_SPACINGS),
            "adaptive_offsets": {str(key): list(value) for key, value in ADAPTIVE_OFFSETS.items()},
            "posterior_topk_tie_break": "lower_beam_index",
            "train_or_validation_fitted_parameters": False,
        },
        "uniform_offsets": {str(k): list(uniform_offsets(k)) for k in scan_budgets},
        "normalized_gain": "selected_power/max(all_64_power)",
        "oracle_claim_ineligible": True,
        "leakage_contract": {
            "local_selection_reads_gt": False,
            "local_selection_reads_csi_or_channel": False,
            "local_selection_reads_full_gain": False,
            "adaptive_selection_reads_predicted_probability_only": True,
            "posterior_topk_reads_predicted_probability_only": True,
            "uniform_selection_reads_sample_data": False,
            "policy_only_selects_from_probed_beams": True,
            "full_gain_private_to_radio_ground_truth_simulator": True,
        },
        "source": evidence.source,
        "sample_rows": len(evidence.sample_id),
        "unique_validation_samples": len(set(evidence.sample_id)),
        "artifacts": {
            "per_sample_results": str(ledger_path),
            "sensing_predictions": str(prediction_path),
            "adaptive_spacing_summary": str(output / "adaptive_spacing_summary.csv"),
        },
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    figures = _plot_results(output, summary_rows, uniform_summary)
    decision = _feasibility_decision(per_mask, summary_rows)
    report = _write_report(output, config, summary_rows, per_mask, spacing_summary, sanity, decision)
    result = {
        "output_dir": str(output),
        "config": config,
        "decision": decision,
        "summary": summary_rows,
        "adaptive_spacing_summary": spacing_summary,
        "figures": figures,
        "report": str(report),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_tbcp_probe_diagnostic(
    evidence: ProbeEvidence,
    *,
    power_paths: Mapping[str, str | Path],
    indexed_labels: Mapping[str, int],
    likelihood: TopologyLikelihood,
    likelihood_source: Mapping[str, Any],
    output_dir: str | Path,
    batch_size: int = 256,
    measurement_error_std_db: float = 0.0,
    noise_seed: int = ROBUSTNESS_NOISE_SEED,
    noise_replica: int = 0,
    include_reference_baselines: bool = True,
    include_diagonal_covariance_ablation: bool = False,
    include_defense_experiments: bool = False,
    include_batch_feedback_experiments: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite a non-empty diagnostic directory: {output}")
    if int(batch_size) <= 0:
        raise ValueError("TBCP diagnostic batch_size must be positive.")
    measurement_sigma = float(measurement_error_std_db)
    if not np.isfinite(measurement_sigma) or measurement_sigma < 0.0:
        raise ValueError("measurement_error_std_db must be finite and non-negative.")
    if int(noise_seed) < 0 or int(noise_replica) < 0:
        raise ValueError("noise_seed and noise_replica must be non-negative integers.")
    if include_diagonal_covariance_ablation and measurement_sigma != 0.0:
        raise ValueError("The diagonal covariance ablation is preregistered only for noiseless probing.")
    if include_defense_experiments and measurement_sigma != 0.0:
        raise ValueError("The open-loop and budget defense experiments are preregistered only for noiseless probing.")
    if include_batch_feedback_experiments and measurement_sigma != 0.0:
        raise ValueError("The batch-feedback experiments are preregistered only for noiseless probing.")
    patterns = tuple(dict.fromkeys(evidence.pattern))
    pattern_available = evidence.source.get("pattern_available_sensing_count")
    if not isinstance(pattern_available, Mapping) or set(pattern_available) != set(patterns):
        raise ValueError("TBCP evidence is missing complete four-sensing pattern availability metadata.")
    counts = [int(pattern_available[pattern]) for pattern in patterns]
    if len(patterns) != 15 or sorted(counts) != [1] * 4 + [2] * 6 + [3] * 4 + [4]:
        raise ValueError("TBCP diagnostic requires all 15 non-empty four-sensing masks.")
    rows_per_pattern = {pattern: evidence.pattern.count(pattern) for pattern in patterns}
    if len(set(rows_per_pattern.values())) != 1 or next(iter(rows_per_pattern.values())) <= 0:
        raise ValueError("TBCP patterns must contain the same positive validation sample count.")
    reference_ids: tuple[str, ...] | None = None
    for pattern in patterns:
        pattern_ids = tuple(
            sample_id
            for sample_id, observed_pattern in zip(evidence.sample_id, evidence.pattern, strict=True)
            if observed_pattern == pattern
        )
        if len(set(pattern_ids)) != len(pattern_ids):
            raise ValueError(f"TBCP pattern contains duplicate sample identity: {pattern}")
        if reference_ids is None:
            reference_ids = pattern_ids
        elif pattern_ids != reference_ids:
            raise ValueError("TBCP patterns do not share identical validation sample identity/order.")
    if (
        not isinstance(likelihood_source, Mapping)
        or likelihood_source.get("artifact_fingerprint") != likelihood.metadata.get("artifact_fingerprint")
        or likelihood_source.get("fit_split") != "train"
    ):
        raise ValueError("TBCP likelihood source does not bind the loaded train-only artifact.")
    likelihood_provenance = likelihood.metadata.get("provenance")
    evidence_topology = evidence.source.get("prototype_topology")
    evidence_protocol = evidence.source.get("data_protocol")
    if (
        not isinstance(likelihood_provenance, Mapping)
        or not isinstance(evidence_topology, Mapping)
        or not isinstance(evidence_protocol, Mapping)
        or likelihood_provenance.get("topology_id") != evidence_topology.get("id")
        or likelihood_provenance.get("topology_descriptor_sha256")
        != evidence_topology.get("descriptor_sha256")
        or likelihood_provenance.get("topology_audit_sha256") != evidence_topology.get("audit_sha256")
        or any(
            likelihood_provenance.get(key) != evidence_protocol.get(key)
            for key in (
                "protocol_id",
                "protocol_fingerprint",
                "data_source_hash",
                "window_config_hash",
                "split_seed",
                "block_size",
                "weather_binding",
            )
        )
    ):
        raise ValueError("TBCP evidence and train-only likelihood do not share one protocol/topology binding.")
    evidence_ids = set(evidence.sample_id)
    power_ids = set(power_paths)
    if evidence_ids - power_ids:
        raise ValueError("Prediction evidence contains samples absent from the validation power index.")
    if evidence.source.get("bounded_evaluation") is not True and evidence_ids != power_ids:
        raise ValueError("Unbounded prediction evidence does not exactly match the validation power index.")
    for sample_id, label in zip(evidence.sample_id, evidence.gt_beam, strict=True):
        if int(indexed_labels[sample_id]) != int(label):
            raise ValueError(f"Evidence/validation label mismatch for {sample_id}.")

    output.mkdir(parents=True, exist_ok=True)
    probability = validate_beam_probability(evidence.pred_prob)
    if probability.ndim != 2 or probability.shape[0] != len(evidence.sample_id):
        raise ValueError("TBCP prediction probability must align with evidence rows.")
    posterior_tensors = beam_posterior_statistics(
        torch.as_tensor(probability, dtype=torch.float32),
        num_beams=NUM_BEAMS,
        top_l=TBCP_BUDGET,
    )
    posterior_arrays = {name: value.cpu().numpy() for name, value in posterior_tensors.items()}
    if not np.array_equal(posterior_arrays["beam_map"], evidence.pred_beam):
        raise ValueError("TBCP posterior MAP does not match prediction evidence.")

    simulator = BeamProbeSimulator(power_paths, require_strict_positive=measurement_sigma > 0.0)

    def probe_measurements(sample_id: str, candidates: Sequence[int]) -> tuple[float, ...]:
        return simulator.probe(
            sample_id,
            candidates,
            measurement_error_std_db=measurement_sigma,
            noise_seed=int(noise_seed),
            noise_replica=int(noise_replica),
        )

    accumulators: dict[tuple[str, str, int, int | None], _MetricAccumulator] = defaultdict(_MetricAccumulator)
    ledger_path = output / "per_sample_results.csv.gz"
    trace_path = output / "tbcp_trace.csv.gz"
    ledger_fields = (
        "sample_id",
        "missing_pattern",
        "available_sensing_count",
        "gt_beam",
        "pred_beam",
        "method",
        "probe_k",
        "probe_indices",
        "probe_measurements",
        "final_beam",
        "correct",
        "normalized_gain",
        "gt_covered",
        "feedback_rounds",
        "measurement_rounds",
        "batch_schedule",
        "posterior_map_trace",
        "posterior_entropy_trace",
    )
    trace_fields = (
        "sample_id",
        "missing_pattern",
        "step",
        "probe_beam",
        "measurement",
        "posterior_map_after_step",
        "posterior_entropy_after_step",
    )
    with gzip.open(ledger_path, "wt", encoding="utf-8", newline="") as ledger_handle, gzip.open(
        trace_path, "wt", encoding="utf-8", newline=""
    ) as trace_handle:
        ledger = csv.DictWriter(ledger_handle, fieldnames=ledger_fields)
        trace_writer = csv.DictWriter(trace_handle, fieldnames=trace_fields)
        ledger.writeheader()
        trace_writer.writeheader()
        for start in range(0, len(evidence.sample_id), int(batch_size)):
            stop = min(start + int(batch_size), len(evidence.sample_id))
            batch_ids = evidence.sample_id[start:stop]
            batch_prior = probability[start:stop]

            def probe_one(candidate: np.ndarray) -> np.ndarray:
                if candidate.shape != (len(batch_ids),):
                    raise ValueError("TBCP requested candidate batch has the wrong shape.")
                return np.asarray(
                    [
                        probe_measurements(sample_id, (int(beam),))[0]
                        for sample_id, beam in zip(batch_ids, candidate, strict=True)
                    ],
                    dtype=np.float64,
                )

            def probe_batch(candidates: np.ndarray) -> np.ndarray:
                if candidates.ndim != 2 or candidates.shape[0] != len(batch_ids):
                    raise ValueError("Batch-TBCP requested candidate batch has the wrong shape.")
                return np.asarray(
                    [
                        probe_measurements(sample_id, tuple(int(beam) for beam in row))
                        for sample_id, row in zip(batch_ids, candidates, strict=True)
                    ],
                    dtype=np.float64,
                )

            max_tbcp_budget = max(DEFAULT_BUDGETS) if include_defense_experiments else TBCP_BUDGET
            tbcp_trace = run_tbcp_batch(
                batch_prior,
                probe_one,
                likelihood,
                budget=max_tbcp_budget,
                measurement_error_std_db=measurement_sigma,
                covariance_mode=COVARIANCE_MODE_FULL,
            )
            diagonal_trace = (
                run_tbcp_batch(
                    batch_prior,
                    probe_one,
                    likelihood,
                    budget=TBCP_BUDGET,
                    measurement_error_std_db=measurement_sigma,
                    covariance_mode=COVARIANCE_MODE_DIAGONAL,
                )
                if include_diagonal_covariance_ablation
                else None
            )
            batch_traces = (
                {
                    method: run_batched_tbcp(
                        batch_prior,
                        probe_batch,
                        likelihood,
                        batch_schedule=schedule,
                        measurement_error_std_db=measurement_sigma,
                        covariance_mode=COVARIANCE_MODE_FULL,
                    )
                    for method, schedule in BATCH_TBCP_METHODS.items()
                }
                if include_batch_feedback_experiments
                else {}
            )
            batch_open_loop = (
                build_topology_open_loop_candidates(
                    batch_prior,
                    likelihood.gain_kernel,
                    budget=TBCP_BUDGET,
                )
                if include_batch_feedback_experiments and not include_defense_experiments
                else None
            )
            batch_open_loop_measurements = (
                probe_batch(batch_open_loop)
                if batch_open_loop is not None
                else None
            )
            defense_open_loop = (
                build_topology_open_loop_candidates(
                    batch_prior,
                    likelihood.gain_kernel,
                    budget=max(DEFAULT_BUDGETS),
                )
                if include_defense_experiments
                else None
            )
            defense_open_loop_measurements = (
                np.stack(
                    [probe_one(defense_open_loop[:, column]) for column in range(defense_open_loop.shape[1])],
                    axis=1,
                )
                if defense_open_loop is not None
                else None
            )
            defense_posterior = (
                np.argsort(-batch_prior, axis=-1, kind="stable")[:, : max(DEFAULT_BUDGETS)]
                if include_defense_experiments
                else None
            )
            defense_posterior_measurements = (
                np.stack(
                    [probe_one(defense_posterior[:, column]) for column in range(defense_posterior.shape[1])],
                    axis=1,
                )
                if defense_posterior is not None
                else None
            )
            posterior5 = np.argsort(-batch_prior, axis=-1, kind="stable")[:, :5]
            posterior5_measurements = np.stack(
                [probe_one(posterior5[:, column]) for column in range(posterior5.shape[1])],
                axis=1,
            )
            hill2 = build_posterior5_hill2_candidates(posterior5, posterior5_measurements)
            hill2_measurements = np.stack(
                [probe_one(hill2[:, column]) for column in range(hill2.shape[1])],
                axis=1,
            )
            hill_indices = np.concatenate((posterior5, hill2), axis=1)
            hill_measurements = np.concatenate((posterior5_measurements, hill2_measurements), axis=1)

            for local_index, global_index in enumerate(range(start, stop)):
                sample_id = evidence.sample_id[global_index]
                pattern = evidence.pattern[global_index]
                gt_beam = int(evidence.gt_beam[global_index])
                pred_beam = int(evidence.pred_beam[global_index])
                available_count = int(pattern_available[pattern])
                _write_tbcp_result(
                    ledger,
                    accumulators,
                    simulator,
                    sample_id=sample_id,
                    pattern=pattern,
                    available_sensing_count=available_count,
                    gt_beam=gt_beam,
                    pred_beam=pred_beam,
                    method="Direct Prediction",
                    candidates=(),
                    measurements=(),
                    final_beam=pred_beam,
                    feedback_rounds=0,
                    require_coverage_equivalence=measurement_sigma == 0.0,
                )
                for method, candidates in (
                    ("Local-7", build_local_candidates(pred_beam, TBCP_BUDGET)),
                    ("Adaptive Local-7", build_adaptive_local_candidates(probability[global_index])[0]),
                    ("Posterior Top-7", build_posterior_topk_candidates(probability[global_index], TBCP_BUDGET)),
                ):
                    measurements = probe_measurements(sample_id, candidates)
                    _write_tbcp_result(
                        ledger,
                        accumulators,
                        simulator,
                        sample_id=sample_id,
                        pattern=pattern,
                        available_sensing_count=available_count,
                        gt_beam=gt_beam,
                        pred_beam=pred_beam,
                        method=method,
                        candidates=candidates,
                        measurements=measurements,
                        final_beam=candidates[int(np.argmax(measurements))],
                        feedback_rounds=0,
                        measurement_rounds=1,
                        batch_schedule=(TBCP_BUDGET,),
                        require_coverage_equivalence=measurement_sigma == 0.0,
                    )
                hill_row = tuple(int(value) for value in hill_indices[local_index])
                hill_power = tuple(float(value) for value in hill_measurements[local_index])
                _write_tbcp_result(
                    ledger,
                    accumulators,
                    simulator,
                    sample_id=sample_id,
                    pattern=pattern,
                    available_sensing_count=available_count,
                    gt_beam=gt_beam,
                    pred_beam=pred_beam,
                    method="Posterior5+Hill2",
                    candidates=hill_row,
                    measurements=hill_power,
                    final_beam=hill_row[int(np.argmax(hill_power))],
                    feedback_rounds=1,
                    measurement_rounds=2,
                    batch_schedule=(5, 2),
                    require_coverage_equivalence=measurement_sigma == 0.0,
                )
                tbcp_indices = tuple(
                    int(value) for value in tbcp_trace.probe_indices[local_index, :TBCP_BUDGET]
                )
                tbcp_measurements = tuple(
                    float(value) for value in tbcp_trace.measurements[local_index, :TBCP_BUDGET]
                )
                map_trace = tuple(
                    int(value) for value in tbcp_trace.posterior_map[local_index, : TBCP_BUDGET + 1]
                )
                entropy_trace = tuple(
                    float(value)
                    for value in tbcp_trace.posterior_entropy[local_index, : TBCP_BUDGET + 1]
                )
                _write_tbcp_result(
                    ledger,
                    accumulators,
                    simulator,
                    sample_id=sample_id,
                    pattern=pattern,
                    available_sensing_count=available_count,
                    gt_beam=gt_beam,
                    pred_beam=pred_beam,
                    method="TBCP-7",
                    candidates=tbcp_indices,
                    measurements=tbcp_measurements,
                    final_beam=tbcp_indices[int(np.argmax(tbcp_measurements))],
                    feedback_rounds=5,
                    measurement_rounds=6,
                    batch_schedule=(2, 1, 1, 1, 1, 1),
                    posterior_map_trace=map_trace,
                    posterior_entropy_trace=entropy_trace,
                    require_coverage_equivalence=measurement_sigma == 0.0,
                )
                if include_batch_feedback_experiments:
                    for method, schedule in BATCH_TBCP_METHODS.items():
                        batch_trace = batch_traces[method]
                        batch_indices = tuple(
                            int(value) for value in batch_trace.probe_indices[local_index]
                        )
                        batch_power = tuple(
                            float(value) for value in batch_trace.measurements[local_index]
                        )
                        _write_tbcp_result(
                            ledger,
                            accumulators,
                            simulator,
                            sample_id=sample_id,
                            pattern=pattern,
                            available_sensing_count=available_count,
                            gt_beam=gt_beam,
                            pred_beam=pred_beam,
                            method=method,
                            candidates=batch_indices,
                            measurements=batch_power,
                            final_beam=int(batch_trace.final_beam[local_index]),
                            feedback_rounds=len(schedule) - 1,
                            measurement_rounds=len(schedule),
                            batch_schedule=schedule,
                            posterior_map_trace=tuple(
                                int(value) for value in batch_trace.posterior_map[local_index]
                            ),
                            posterior_entropy_trace=tuple(
                                float(value) for value in batch_trace.posterior_entropy[local_index]
                            ),
                            require_coverage_equivalence=True,
                        )
                    if not include_defense_experiments:
                        if batch_open_loop is None or batch_open_loop_measurements is None:
                            raise RuntimeError("Batch-feedback open-loop control was not initialized.")
                        open_indices = tuple(int(value) for value in batch_open_loop[local_index])
                        open_power = tuple(
                            float(value) for value in batch_open_loop_measurements[local_index]
                        )
                        _write_tbcp_result(
                            ledger,
                            accumulators,
                            simulator,
                            sample_id=sample_id,
                            pattern=pattern,
                            available_sensing_count=available_count,
                            gt_beam=gt_beam,
                            pred_beam=pred_beam,
                            method="Topology Open-loop Gain-7",
                            candidates=open_indices,
                            measurements=open_power,
                            final_beam=open_indices[int(np.argmax(open_power))],
                            feedback_rounds=0,
                            measurement_rounds=1,
                            batch_schedule=(TBCP_BUDGET,),
                        )
                if include_defense_experiments:
                    if defense_open_loop is None or defense_open_loop_measurements is None:
                        raise RuntimeError("Defense open-loop candidates were not initialized.")
                    if defense_posterior is None or defense_posterior_measurements is None:
                        raise RuntimeError("Defense posterior candidates were not initialized.")
                    for budget in DEFAULT_BUDGETS:
                        open_indices = tuple(
                            int(value) for value in defense_open_loop[local_index, :budget]
                        )
                        open_power = tuple(
                            float(value)
                            for value in defense_open_loop_measurements[local_index, :budget]
                        )
                        _write_tbcp_result(
                            ledger,
                            accumulators,
                            simulator,
                            sample_id=sample_id,
                            pattern=pattern,
                            available_sensing_count=available_count,
                            gt_beam=gt_beam,
                            pred_beam=pred_beam,
                            method=f"Topology Open-loop Gain-{budget}",
                            candidates=open_indices,
                            measurements=open_power,
                            final_beam=open_indices[int(np.argmax(open_power))],
                            feedback_rounds=0,
                        )
                        if budget == TBCP_BUDGET:
                            continue
                        posterior_indices = tuple(
                            int(value) for value in defense_posterior[local_index, :budget]
                        )
                        posterior_power = tuple(
                            float(value)
                            for value in defense_posterior_measurements[local_index, :budget]
                        )
                        _write_tbcp_result(
                            ledger,
                            accumulators,
                            simulator,
                            sample_id=sample_id,
                            pattern=pattern,
                            available_sensing_count=available_count,
                            gt_beam=gt_beam,
                            pred_beam=pred_beam,
                            method=f"Posterior Top-{budget}",
                            candidates=posterior_indices,
                            measurements=posterior_power,
                            final_beam=posterior_indices[int(np.argmax(posterior_power))],
                            feedback_rounds=0,
                        )
                        curve_tbcp_indices = tuple(
                            int(value) for value in tbcp_trace.probe_indices[local_index, :budget]
                        )
                        curve_tbcp_power = tuple(
                            float(value) for value in tbcp_trace.measurements[local_index, :budget]
                        )
                        _write_tbcp_result(
                            ledger,
                            accumulators,
                            simulator,
                            sample_id=sample_id,
                            pattern=pattern,
                            available_sensing_count=available_count,
                            gt_beam=gt_beam,
                            pred_beam=pred_beam,
                            method=f"TBCP-{budget}",
                            candidates=curve_tbcp_indices,
                            measurements=curve_tbcp_power,
                            final_beam=curve_tbcp_indices[int(np.argmax(curve_tbcp_power))],
                            feedback_rounds=max(0, budget - 2),
                            posterior_map_trace=tuple(
                                int(value)
                                for value in tbcp_trace.posterior_map[local_index, : budget + 1]
                            ),
                            posterior_entropy_trace=tuple(
                                float(value)
                                for value in tbcp_trace.posterior_entropy[local_index, : budget + 1]
                            ),
                        )
                if diagonal_trace is not None:
                    diagonal_indices = tuple(
                        int(value) for value in diagonal_trace.probe_indices[local_index]
                    )
                    diagonal_measurements = tuple(
                        float(value) for value in diagonal_trace.measurements[local_index]
                    )
                    _write_tbcp_result(
                        ledger,
                        accumulators,
                        simulator,
                        sample_id=sample_id,
                        pattern=pattern,
                        available_sensing_count=available_count,
                        gt_beam=gt_beam,
                        pred_beam=pred_beam,
                        method=TBCP_DIAGONAL_METHOD,
                        candidates=diagonal_indices,
                        measurements=diagonal_measurements,
                        final_beam=int(diagonal_trace.final_beam[local_index]),
                        feedback_rounds=5,
                        posterior_map_trace=tuple(
                            int(value) for value in diagonal_trace.posterior_map[local_index]
                        ),
                        posterior_entropy_trace=tuple(
                            float(value) for value in diagonal_trace.posterior_entropy[local_index]
                        ),
                        require_coverage_equivalence=True,
                    )
                for step, (beam, measurement) in enumerate(
                    zip(tbcp_indices, tbcp_measurements, strict=True), start=1
                ):
                    trace_writer.writerow(
                        {
                            "sample_id": sample_id,
                            "missing_pattern": pattern,
                            "step": step,
                            "probe_beam": beam,
                            "measurement": f"{measurement:.12g}",
                            "posterior_map_after_step": map_trace[step],
                            "posterior_entropy_after_step": f"{entropy_trace[step]:.12g}",
                        }
                    )
                if include_reference_baselines:
                    oracle = build_oracle_local_candidates(gt_beam, TBCP_BUDGET)
                    oracle_power = probe_measurements(sample_id, oracle)
                    _write_tbcp_result(
                        ledger,
                        accumulators,
                        simulator,
                        sample_id=sample_id,
                        pattern=pattern,
                        available_sensing_count=available_count,
                        gt_beam=gt_beam,
                        pred_beam=pred_beam,
                        method="Oracle Local-7",
                        candidates=oracle,
                        measurements=oracle_power,
                        final_beam=oracle[int(np.argmax(oracle_power))],
                        feedback_rounds=0,
                        require_coverage_equivalence=measurement_sigma == 0.0,
                    )
                    full = tuple(range(NUM_BEAMS))
                    full_power = probe_measurements(sample_id, full)
                    _write_tbcp_result(
                        ledger,
                        accumulators,
                        simulator,
                        sample_id=sample_id,
                        pattern=pattern,
                        available_sensing_count=available_count,
                        gt_beam=gt_beam,
                        pred_beam=pred_beam,
                        method="Full-64",
                        candidates=full,
                        measurements=full_power,
                        final_beam=int(np.argmax(full_power)),
                        feedback_rounds=0,
                        require_coverage_equivalence=measurement_sigma == 0.0,
                    )
                    for offset in uniform_offsets(TBCP_BUDGET):
                        uniform = build_uniform_candidates(TBCP_BUDGET, offset)
                        uniform_power = probe_measurements(sample_id, uniform)
                        final_beam = uniform[int(np.argmax(uniform_power))]
                        covered = gt_beam in uniform
                        if measurement_sigma == 0.0 and (final_beam == gt_beam) != covered:
                            raise ValueError(
                                f"Noiseless probe correct/coverage mismatch for {sample_id}, "
                                f"method=Uniform-7, offset={offset}."
                            )
                        accumulators[(pattern, "Uniform-7", TBCP_BUDGET, offset)].add(
                            correct=final_beam == gt_beam,
                            normalized_gain=simulator.normalized_gain(sample_id, final_beam),
                            covered=covered,
                        )

    per_pattern, uniform_offsets_rows = _summarize_tbcp_patterns(
        accumulators,
        patterns,
        include_reference_baselines=include_reference_baselines,
        include_diagonal_covariance_ablation=include_diagonal_covariance_ablation,
        include_defense_experiments=include_defense_experiments,
        include_batch_feedback_experiments=include_batch_feedback_experiments,
    )
    group_summary = _summarize_tbcp_groups(per_pattern, pattern_available)
    _write_csv(output / "per_pattern_summary.csv", per_pattern)
    _write_csv(output / "group_summary.csv", group_summary)
    if uniform_offsets_rows:
        _write_csv(output / "uniform_offset_summary.csv", uniform_offsets_rows)
    baseline_names = [
        "Direct Prediction",
        "Local-7",
        "Adaptive Local-7",
        "Posterior Top-7",
        "Posterior5+Hill2",
    ]
    if include_reference_baselines:
        baseline_names.extend(("Uniform-7 offset mean", "Oracle Local-7", "Full-64"))
    if include_diagonal_covariance_ablation:
        baseline_names.append(TBCP_DIAGONAL_METHOD)
    if include_defense_experiments:
        baseline_names.extend(
            method
            for budget in DEFAULT_BUDGETS
            for method in (
                f"Topology Open-loop Gain-{budget}",
                *(() if budget == TBCP_BUDGET else (f"Posterior Top-{budget}", f"TBCP-{budget}")),
            )
        )
    if include_batch_feedback_experiments:
        baseline_names.extend(BATCH_TBCP_METHODS)
        if not include_defense_experiments:
            baseline_names.append("Topology Open-loop Gain-7")
    config = {
        "schema_version": 1,
        "evaluation_scope": (
            "validation_only_tbcp_batch_feedback_diagnostic"
            if include_batch_feedback_experiments
            else (
                "validation_only_tbcp_covariance_ablation"
                if include_diagonal_covariance_ablation
                else (
                    "validation_only_tbcp_open_loop_budget_defense"
                    if include_defense_experiments
                    else (
                        "validation_only_topology_bayesian_closed_loop_probe"
                        if measurement_sigma == 0.0 and include_reference_baselines
                        else "bounded_synthetic_measurement_error_sensitivity"
                    )
                )
            )
        ),
        "claim_ineligible": True,
        "outer_test_accessed": False,
        "model_trained_or_updated": False,
        "prediction_evidence_reused": True,
        "patterns": list(patterns),
        "pattern_available_sensing_count": {key: int(value) for key, value in pattern_available.items()},
        "budget": TBCP_BUDGET,
        "num_beams": NUM_BEAMS,
        "primary_policy": {
            "name": "TBCP-7",
            "version": TBCP_POLICY_VERSION,
            "first_probe": "sensing_posterior_map",
            "subsequent_probe": "maximum_posterior_expected_terminal_normalized_gain",
            "belief_update": "joint_relative_db_gaussian_from_original_sensing_prior",
            "covariance_mode": COVARIANCE_MODE_FULL,
            "measurement_error_std_db": measurement_sigma,
            "feedback_dependent_decisions": 5,
            "tie_break": "lower_beam_index",
        },
        "baselines": baseline_names,
        "covariance_ablation": {
            "enabled": bool(include_diagonal_covariance_ablation),
            "modes": [
                COVARIANCE_MODE_FULL,
                *([COVARIANCE_MODE_DIAGONAL] if include_diagonal_covariance_ablation else []),
            ],
            "diagonal_transform": "diag(diag(train_covariance_db2))",
            "shared_reference_correlation_preserved": True,
        },
        "defense_experiments": {
            "enabled": bool(include_defense_experiments),
            "version": DEFENSE_EXPERIMENT_VERSION,
            "budgets": list(DEFAULT_BUDGETS) if include_defense_experiments else [],
            "primary_budget_remains_frozen": TBCP_BUDGET,
            "methods": ["TBCP-K", "Topology Open-loop Gain-K", "Posterior Top-K"],
            "measurement_error_std_db": 0.0,
            "open_loop_reads_measurements_during_acquisition": False,
            "validation_result_may_select_primary_budget": False,
        },
        "batch_feedback_experiments": {
            "enabled": bool(include_batch_feedback_experiments),
            "version": BATCH_FEEDBACK_EXPERIMENT_VERSION,
            "policy_version": BATCH_TBCP_POLICY_VERSION,
            "schedules": {
                method: {
                    "batch_schedule": list(schedule),
                    "probe_k": TBCP_BUDGET,
                    "measurement_rounds": len(schedule),
                    "feedback_updates": len(schedule) - 1,
                }
                for method, schedule in BATCH_TBCP_METHODS.items()
            },
            "selection": "current_posterior_expected_terminal_gain_without_intra_batch_measurements",
            "total_measurement_slots": TBCP_BUDGET,
            "validation_result_may_select_schedule": False,
        },
        "measurement_model": {
            "version": ROBUSTNESS_NOISE_MODEL_VERSION,
            "distribution": "independent_zero_mean_gaussian_log_power_error_db",
            "measurement_error_std_db": measurement_sigma,
            "noise_seed": int(noise_seed),
            "noise_replica": int(noise_replica),
            "common_random_key": "noise_seed/replica/stable_sample_id/beam_index",
            "probing_snr_available": False,
            "clean_metric_denominator": True,
        },
        "likelihood": {
            "source": dict(likelihood_source),
            "metadata": likelihood.metadata,
        },
        "source": evidence.source,
        "sample_rows": len(evidence.sample_id),
        "unique_validation_samples": len(set(evidence.sample_id)),
        "leakage_contract": {
            "train_likelihood_reads_train_radio_ground_truth_only": True,
            "policy_reads_gt": False,
            "policy_reads_channel_or_unrequested_csi": False,
            "policy_reads_full_validation_gain": False,
            "policy_reads_requested_measurements_only": True,
            "final_beam_selected_from_seven_measurements_only": True,
            "defense_final_beam_selected_from_requested_measurements_only": True,
            "batch_policy_reads_only_completed_batch_measurements": True,
        },
        "artifacts": {
            "per_sample_results": str(ledger_path),
            "tbcp_trace": str(trace_path),
            "per_pattern_summary": str(output / "per_pattern_summary.csv"),
            "group_summary": str(output / "group_summary.csv"),
            **(
                {"uniform_offset_summary": str(output / "uniform_offset_summary.csv")}
                if uniform_offsets_rows
                else {}
            ),
        },
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    report_path = _write_tbcp_report(output, config, group_summary)
    result = {
        "output_dir": str(output),
        "config": config,
        "group_summary": group_summary,
        "report": str(report_path),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_tbcp_robustness_sensitivity(
    evidence: ProbeEvidence,
    *,
    power_paths: Mapping[str, str | Path],
    indexed_labels: Mapping[str, int],
    likelihood: TopologyLikelihood,
    likelihood_source: Mapping[str, Any],
    output_dir: str | Path,
    samples_per_pattern: int = ROBUSTNESS_SAMPLE_COUNT,
    batch_size: int = 256,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite a non-empty robustness directory: {output}")
    selected = select_probe_evidence_by_hash(evidence, samples_per_pattern=samples_per_pattern)
    output.mkdir(parents=True, exist_ok=True)
    scenario_rows: list[dict[str, Any]] = []
    scenario_artifacts: list[dict[str, Any]] = []
    for sigma_db, replica in _robustness_scenarios():
        scenario_dir = output / "scenarios" / f"sigma_{int(sigma_db):02d}db_replica_{replica}"
        scenario = run_tbcp_probe_diagnostic(
            selected,
            power_paths=power_paths,
            indexed_labels=indexed_labels,
            likelihood=likelihood,
            likelihood_source=likelihood_source,
            output_dir=scenario_dir,
            batch_size=batch_size,
            measurement_error_std_db=sigma_db,
            noise_seed=ROBUSTNESS_NOISE_SEED,
            noise_replica=replica,
            include_reference_baselines=False,
        )
        scenario_result = scenario_dir / "result.json"
        scenario_artifacts.append(
            {
                "measurement_error_std_db": sigma_db,
                "noise_replica": replica,
                "result": str(scenario_result),
                "result_sha256": _sha256_file(scenario_result),
            }
        )
        for row in scenario["group_summary"]:
            scenario_rows.append(
                {
                    "experiment_seed": int(selected.source.get("experiment_seed", -1)),
                    "measurement_error_std_db": sigma_db,
                    "noise_replica": replica,
                    **row,
                }
            )
    noise_summary = _aggregate_robustness_rows(scenario_rows)
    latency_summary = _robustness_latency_rows(noise_summary)
    break_even = _robustness_break_even_rows(noise_summary)
    _write_csv(output / "scenario_group_summary.csv", scenario_rows)
    _write_csv(output / "noise_summary.csv", noise_summary)
    _write_csv(output / "feedback_overhead_summary.csv", latency_summary)
    _write_csv(output / "feedback_break_even.csv", break_even)
    config = {
        "schema_version": 1,
        "robustness_version": ROBUSTNESS_VERSION,
        "claim_ineligible": True,
        "outer_test_accessed": False,
        "model_trained_or_updated": False,
        "experiment_seed": int(selected.source.get("experiment_seed", -1)),
        "measurement_error_std_db": list(ROBUSTNESS_NOISE_STD_DB),
        "noise_replicas": list(ROBUSTNESS_NOISE_REPLICAS),
        "noise_seed": ROBUSTNESS_NOISE_SEED,
        "noise_model_version": ROBUSTNESS_NOISE_MODEL_VERSION,
        "scenario_grid": [
            {"measurement_error_std_db": sigma_db, "noise_replica": replica}
            for sigma_db, replica in _robustness_scenarios()
        ],
        "samples_per_pattern": int(samples_per_pattern),
        "sample_selection": {
            key: selected.source[key]
            for key in (
                "selection_method",
                "selected_stable_sample_id_hash",
                "selected_sample_order_sha256",
            )
        },
        "communication_reference_snr_db": list(COMMUNICATION_SNR_DB),
        "probing_measurement_snr_available": False,
        "feedback_overhead_fraction_per_update": list(ROBUSTNESS_FEEDBACK_OVERHEAD),
        "method_feedback_updates": METHOD_FEEDBACK_UPDATES,
        "common_measurement_slots": TBCP_BUDGET,
        "source": selected.source,
        "likelihood": {
            "source": dict(likelihood_source),
            "artifact_fingerprint": likelihood.metadata["artifact_fingerprint"],
        },
        "scenario_artifacts": scenario_artifacts,
        "artifacts": {
            "scenario_group_summary": str(output / "scenario_group_summary.csv"),
            "noise_summary": str(output / "noise_summary.csv"),
            "feedback_overhead_summary": str(output / "feedback_overhead_summary.csv"),
            "feedback_break_even": str(output / "feedback_break_even.csv"),
        },
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    report = _write_robustness_report(output, config, noise_summary, break_even)
    result = {
        "output_dir": str(output),
        "config": config,
        "scenario_group_summary": scenario_rows,
        "noise_summary": noise_summary,
        "feedback_break_even": break_even,
        "report": str(report),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def summarize_tbcp_robustness_replays(
    runs: Mapping[int, str | Path],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    if set(int(seed) for seed in runs) != {1, 2, 3} or len(runs) != 3:
        raise ValueError("Robustness summary requires exactly checkpoint seeds 1, 2, and 3.")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite a non-empty robustness summary directory: {output}")
    combined_rows: list[dict[str, Any]] = []
    run_artifacts: dict[str, Any] = {}
    reference_binding: dict[str, Any] | None = None
    checkpoint_digests: set[str] = set()
    for seed in (1, 2, 3):
        path = Path(runs[seed]).resolve()
        result = json.loads(path.read_text(encoding="utf-8"))
        config = result.get("config") if isinstance(result, Mapping) else None
        if (
            not isinstance(config, Mapping)
            or config.get("robustness_version") != ROBUSTNESS_VERSION
            or config.get("claim_ineligible") is not True
            or config.get("outer_test_accessed") is not False
            or config.get("model_trained_or_updated") is not False
            or int(config.get("experiment_seed", -1)) != seed
            or tuple(float(value) for value in config.get("measurement_error_std_db", ()))
            != ROBUSTNESS_NOISE_STD_DB
            or tuple(int(value) for value in config.get("noise_replicas", ())) != ROBUSTNESS_NOISE_REPLICAS
            or int(config.get("noise_seed", -1)) != ROBUSTNESS_NOISE_SEED
            or config.get("noise_model_version") != ROBUSTNESS_NOISE_MODEL_VERSION
        ):
            raise ValueError(f"Robustness seed {seed} result does not match the preregistered sealed grid.")
        source = config.get("source", {})
        protocol = source.get("data_protocol", {}) if isinstance(source, Mapping) else {}
        topology = source.get("prototype_topology", {}) if isinstance(source, Mapping) else {}
        selection = config.get("sample_selection", {})
        binding = {
            "samples_per_pattern": config.get("samples_per_pattern"),
            "selection_method": selection.get("selection_method"),
            "selected_stable_sample_id_hash": selection.get("selected_stable_sample_id_hash"),
            "selected_sample_order_sha256": selection.get("selected_sample_order_sha256"),
            "likelihood_fingerprint": config.get("likelihood", {}).get("artifact_fingerprint"),
            "protocol_fingerprint": protocol.get("protocol_fingerprint"),
            "validation_sample_id_hash": protocol.get("validation_sample_id_hash"),
            "topology_id": topology.get("id"),
            "topology_descriptor_sha256": topology.get("descriptor_sha256"),
            "topology_audit_sha256": topology.get("audit_sha256"),
        }
        if binding["topology_id"] != EXPECTED_TOPOLOGY or any(
            not _is_sha256(binding[key])
            for key in (
                "selected_stable_sample_id_hash",
                "selected_sample_order_sha256",
                "likelihood_fingerprint",
                "protocol_fingerprint",
                "validation_sample_id_hash",
                "topology_descriptor_sha256",
                "topology_audit_sha256",
            )
        ):
            raise ValueError(f"Robustness seed {seed} has incomplete sample/protocol/topology provenance.")
        if reference_binding is None:
            reference_binding = binding
        elif binding != reference_binding:
            raise ValueError("Robustness seed replays do not share one sample/protocol/topology binding.")
        checkpoint_path = Path(str(source.get("checkpoint_path", ""))).resolve()
        checkpoint_sha256, _checkpoint_size = checkpoint_file_digest(checkpoint_path)
        if checkpoint_sha256 != source.get("checkpoint_sha256"):
            raise ValueError(f"Robustness seed {seed} checkpoint digest does not match result provenance.")
        checkpoint_digests.add(checkpoint_sha256)
        for artifact in config.get("scenario_artifacts", ()):
            scenario_path = Path(str(artifact.get("result", ""))).resolve()
            if _sha256_file(scenario_path) != artifact.get("result_sha256"):
                raise ValueError(f"Robustness seed {seed} scenario artifact digest mismatch.")
        rows = result.get("scenario_group_summary")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Robustness seed {seed} result has no scenario group rows.")
        expected_conditions = set(_robustness_scenarios())
        observed_conditions = {
            (float(row["measurement_error_std_db"]), int(row["noise_replica"])) for row in rows
        }
        if observed_conditions != expected_conditions:
            raise ValueError(f"Robustness seed {seed} scenario rows do not cover the fixed grid.")
        combined_rows.extend(dict(row) for row in rows)
        run_artifacts[str(seed)] = {"path": str(path), "sha256": _sha256_file(path)}
    if len(checkpoint_digests) != 3:
        raise ValueError("Robustness summary requires three distinct checkpoint lineages.")

    noise_summary = _aggregate_robustness_rows(combined_rows)
    latency_summary = _robustness_latency_rows(noise_summary)
    break_even = _robustness_break_even_rows(noise_summary)
    stability = _robustness_stability(combined_rows)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "combined_scenario_group_summary.csv", combined_rows)
    _write_csv(output / "three_seed_noise_summary.csv", noise_summary)
    _write_csv(output / "three_seed_feedback_overhead.csv", latency_summary)
    _write_csv(output / "three_seed_feedback_break_even.csv", break_even)
    (output / "stability.json").write_text(json.dumps(stability, indent=2) + "\n", encoding="utf-8")
    config = {
        "schema_version": 1,
        "robustness_version": ROBUSTNESS_VERSION,
        "claim_ineligible": True,
        "outer_test_accessed": False,
        "model_trained_or_updated": False,
        "checkpoint_seeds": [1, 2, 3],
        "binding": reference_binding,
        "measurement_error_std_db": list(ROBUSTNESS_NOISE_STD_DB),
        "noise_replicas": list(ROBUSTNESS_NOISE_REPLICAS),
        "noise_seed": ROBUSTNESS_NOISE_SEED,
        "noise_model_version": ROBUSTNESS_NOISE_MODEL_VERSION,
        "run_artifacts": run_artifacts,
        "artifacts": {
            "scenario_summary": str(output / "combined_scenario_group_summary.csv"),
            "noise_summary": str(output / "three_seed_noise_summary.csv"),
            "feedback_overhead": str(output / "three_seed_feedback_overhead.csv"),
            "feedback_break_even": str(output / "three_seed_feedback_break_even.csv"),
            "stability": str(output / "stability.json"),
        },
    }
    report = _write_robustness_report(output, {**config, "samples_per_pattern": reference_binding["samples_per_pattern"]}, noise_summary, break_even)
    result = {
        "output_dir": str(output),
        "config": config,
        "noise_summary": noise_summary,
        "feedback_break_even": break_even,
        "stability": stability,
        "report": str(report),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _robustness_stability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for sigma_db in ROBUSTNESS_NOISE_STD_DB:
        for baseline in ("Posterior Top-7", "Posterior5+Hill2"):
            condition_deltas: list[tuple[float, float, float]] = []
            conditions = sorted(
                {
                    (int(row["experiment_seed"]), int(row["noise_replica"]))
                    for row in rows
                    if float(row["measurement_error_std_db"]) == sigma_db
                    and row["group"] == "All-15 Macro"
                }
            )
            for seed, replica in conditions:
                selected = [
                    row
                    for row in rows
                    if float(row["measurement_error_std_db"]) == sigma_db
                    and int(row["experiment_seed"]) == seed
                    and int(row["noise_replica"]) == replica
                    and row["group"] == "All-15 Macro"
                ]
                by_method = {str(row["method"]): row for row in selected}
                if "TBCP-7" not in by_method or baseline not in by_method:
                    raise ValueError("Robustness stability rows are incomplete.")
                condition_deltas.append(
                    (
                        float(by_method["TBCP-7"]["top1"]) - float(by_method[baseline]["top1"]),
                        float(by_method["TBCP-7"]["normalized_gain"])
                        - float(by_method[baseline]["normalized_gain"]),
                        float(by_method["TBCP-7"]["spectral_efficiency_ratio_10db"])
                        - float(by_method[baseline]["spectral_efficiency_ratio_10db"]),
                    )
                )
            array = np.asarray(condition_deltas, dtype=np.float64)
            result[f"sigma_{int(sigma_db)}db_vs_{baseline}"] = {
                "condition_count": len(condition_deltas),
                "positive_top1_conditions": int((array[:, 0] > 0.0).sum()),
                "positive_gain_conditions": int((array[:, 1] > 0.0).sum()),
                "positive_rate_10db_conditions": int((array[:, 2] > 0.0).sum()),
                "mean_top1_delta": float(array[:, 0].mean()),
                "minimum_top1_delta": float(array[:, 0].min()),
                "mean_normalized_gain_delta": float(array[:, 1].mean()),
                "minimum_normalized_gain_delta": float(array[:, 1].min()),
            }
    return result


def _robustness_scenarios() -> tuple[tuple[float, int], ...]:
    return (
        (0.0, 0),
        *((sigma_db, replica) for sigma_db in (3.0, 6.0) for replica in ROBUSTNESS_NOISE_REPLICAS),
    )


def _aggregate_robustness_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (float(row["measurement_error_std_db"]), str(row["group"]), str(row["method"]))
            for row in rows
        }
    )
    result: list[dict[str, Any]] = []
    for sigma_db, group, method in keys:
        selected = [
            row
            for row in rows
            if float(row["measurement_error_std_db"]) == sigma_db
            and str(row["group"]) == group
            and str(row["method"]) == method
        ]
        if not selected:
            raise ValueError("Robustness aggregation received an empty condition.")
        aggregate: dict[str, Any] = {
            "measurement_error_std_db": sigma_db,
            "group": group,
            "method": method,
            "probe_k": int(selected[0]["probe_k"]),
            "condition_count": len(selected),
            "checkpoint_seed_count": len({int(row["experiment_seed"]) for row in selected}),
            "noise_replica_count": len({int(row["noise_replica"]) for row in selected}),
        }
        for metric in _probe_metric_names():
            values = np.asarray(
                [float(row[metric]) for row in selected if row.get(metric) is not None],
                dtype=np.float64,
            )
            aggregate[f"mean_{metric}"] = float(values.mean()) if values.size else None
            aggregate[f"std_{metric}"] = (
                float(values.std(ddof=1)) if values.size > 1 else 0.0 if values.size else None
            )
        result.append(aggregate)
    return result


def _robustness_latency_rows(noise_summary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in noise_summary:
        method = str(summary["method"])
        feedback_updates = METHOD_FEEDBACK_UPDATES.get(method)
        if feedback_updates is None:
            continue
        for overhead in ROBUSTNESS_FEEDBACK_OVERHEAD:
            payload_fraction = max(0.0, 1.0 - feedback_updates * overhead)
            for snr_db in COMMUNICATION_SNR_DB:
                rate = float(summary[f"mean_spectral_efficiency_ratio_{int(snr_db)}db"])
                rows.append(
                    {
                        "measurement_error_std_db": summary["measurement_error_std_db"],
                        "group": summary["group"],
                        "method": method,
                        "communication_reference_snr_db": int(snr_db),
                        "feedback_updates": feedback_updates,
                        "feedback_overhead_fraction_per_update": overhead,
                        "effective_spectral_efficiency_ratio": rate * payload_fraction,
                    }
                )
    return rows


def _robustness_break_even_rows(noise_summary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sigma_db in ROBUSTNESS_NOISE_STD_DB:
        groups = sorted({str(row["group"]) for row in noise_summary if row["measurement_error_std_db"] == sigma_db})
        for group in groups:
            by_method = {
                str(row["method"]): row
                for row in noise_summary
                if row["measurement_error_std_db"] == sigma_db and row["group"] == group
            }
            tbcp = by_method.get("TBCP-7")
            if tbcp is None:
                raise ValueError(f"Robustness summary is missing TBCP-7 for {sigma_db}/{group}.")
            for baseline in ("Posterior Top-7", "Posterior5+Hill2"):
                control = by_method.get(baseline)
                if control is None:
                    raise ValueError(f"Robustness summary is missing {baseline} for {sigma_db}/{group}.")
                for snr_db in COMMUNICATION_SNR_DB:
                    metric = f"mean_spectral_efficiency_ratio_{int(snr_db)}db"
                    tbcp_rate = float(tbcp[metric])
                    control_rate = float(control[metric])
                    denominator = 5.0 * tbcp_rate - METHOD_FEEDBACK_UPDATES[baseline] * control_rate
                    threshold = (tbcp_rate - control_rate) / denominator if denominator > 0.0 else None
                    rows.append(
                        {
                            "measurement_error_std_db": sigma_db,
                            "group": group,
                            "baseline": baseline,
                            "communication_reference_snr_db": int(snr_db),
                            "tbcp_rate_ratio": tbcp_rate,
                            "baseline_rate_ratio": control_rate,
                            "tbcp_advantage_at_zero_overhead": tbcp_rate > control_rate,
                            "break_even_feedback_overhead_fraction_per_update": (
                                threshold if threshold is not None and threshold >= 0.0 else None
                            ),
                        }
                    )
    return rows


def _write_robustness_report(
    output: Path,
    config: Mapping[str, Any],
    noise_summary: Sequence[Mapping[str, Any]],
    break_even: Sequence[Mapping[str, Any]],
) -> Path:
    lines = [
        "# TBCP-7 Synthetic Robustness Sensitivity",
        "",
        f"- Samples: {config['samples_per_pattern']} hash-selected validation samples per each of 15 masks",
        f"- Measurement error sigma: {config['measurement_error_std_db']} dB",
        "- This is matched synthetic log-power error, not measured probing SNR",
        "- Communication SNR is oracle-beam reference SNR; latency is a normalized feedback-cost proxy",
        "- Claim boundary: bounded validation-only, claim-ineligible, outer test not accessed",
        "",
    ]
    for group in ("All-15 Macro", "Single Macro"):
        lines.extend(
            [
                f"## {group}",
                "",
                "| sigma dB | Method | Top-1 | Gain | Rate@10dB | Select given coverage |",
                "| ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for sigma_db in ROBUSTNESS_NOISE_STD_DB:
            for method in ("Posterior Top-7", "Posterior5+Hill2", "TBCP-7"):
                row = next(
                    value
                    for value in noise_summary
                    if value["measurement_error_std_db"] == sigma_db
                    and value["group"] == group
                    and value["method"] == method
                )
                selection = row["mean_selection_accuracy_given_coverage"]
                lines.append(
                    f"| {sigma_db:.0f} | {method} | {100 * float(row['mean_top1']):.2f}% | "
                    f"{100 * float(row['mean_normalized_gain']):.2f}% | "
                    f"{100 * float(row['mean_spectral_efficiency_ratio_10db']):.2f}% | "
                    f"{'-' if selection is None else f'{100 * float(selection):.2f}%'} |"
                )
        lines.append("")
    lines.extend(["## Feedback Break-even", "", "10 dB communication reference SNR, All-15 Macro:", ""])
    for sigma_db in ROBUSTNESS_NOISE_STD_DB:
        for baseline in ("Posterior Top-7", "Posterior5+Hill2"):
            row = next(
                value
                for value in break_even
                if value["measurement_error_std_db"] == sigma_db
                and value["group"] == "All-15 Macro"
                and value["baseline"] == baseline
                and value["communication_reference_snr_db"] == 10
            )
            threshold = row["break_even_feedback_overhead_fraction_per_update"]
            lines.append(
                f"- sigma={sigma_db:.0f} dB vs {baseline}: "
                + ("no zero-overhead advantage" if threshold is None else f"rho={100 * float(threshold):.2f}% per update")
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "The feedback threshold is normalized to a post-probing payload opportunity. It is not milliseconds, "
            "does not model real beam-switch hardware, and must not be presented as measured latency.",
        ]
    )
    path = output / "robustness_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_tbcp_result(
    ledger: csv.DictWriter,
    accumulators: Mapping[tuple[str, str, int, int | None], _MetricAccumulator],
    simulator: BeamProbeSimulator,
    *,
    sample_id: str,
    pattern: str,
    available_sensing_count: int,
    gt_beam: int,
    pred_beam: int,
    method: str,
    candidates: Sequence[int],
    measurements: Sequence[float],
    final_beam: int,
    feedback_rounds: int,
    measurement_rounds: int | None = None,
    batch_schedule: Sequence[int] = (),
    posterior_map_trace: Sequence[int] = (),
    posterior_entropy_trace: Sequence[float] = (),
    require_coverage_equivalence: bool = True,
) -> None:
    candidate_tuple = tuple(int(value) for value in candidates)
    measurement_tuple = tuple(float(value) for value in measurements)
    if len(candidate_tuple) != len(measurement_tuple):
        raise ValueError(f"{method} candidates and requested measurements are misaligned.")
    schedule = tuple(int(value) for value in batch_schedule)
    if schedule and (any(value <= 0 for value in schedule) or sum(schedule) != len(candidate_tuple)):
        raise ValueError(f"{method} batch_schedule must contain positive rounds summing to probe_k.")
    rounds = int(measurement_rounds) if measurement_rounds is not None else len(schedule)
    if rounds < 0 or (schedule and rounds != len(schedule)):
        raise ValueError(f"{method} measurement_rounds does not match batch_schedule.")
    covered = None if not candidate_tuple else gt_beam in candidate_tuple
    if require_coverage_equivalence and covered is not None and (int(final_beam) == gt_beam) != covered:
        raise ValueError(
            f"Noiseless probe correct/coverage mismatch for {sample_id}, method={method}; "
            "check label/power ties or protocol drift."
        )
    gain = simulator.normalized_gain(sample_id, int(final_beam))
    correct = int(final_beam) == gt_beam
    ledger.writerow(
        {
            "sample_id": sample_id,
            "missing_pattern": pattern,
            "available_sensing_count": available_sensing_count,
            "gt_beam": gt_beam,
            "pred_beam": pred_beam,
            "method": method,
            "probe_k": len(candidate_tuple),
            "probe_indices": json.dumps(candidate_tuple, separators=(",", ":")),
            "probe_measurements": json.dumps(measurement_tuple, separators=(",", ":")),
            "final_beam": int(final_beam),
            "correct": int(correct),
            "normalized_gain": f"{gain:.12g}",
            "gt_covered": "" if covered is None else int(covered),
            "feedback_rounds": int(feedback_rounds),
            "measurement_rounds": rounds,
            "batch_schedule": json.dumps(schedule, separators=(",", ":")),
            "posterior_map_trace": json.dumps(tuple(posterior_map_trace), separators=(",", ":")),
            "posterior_entropy_trace": json.dumps(
                tuple(float(value) for value in posterior_entropy_trace), separators=(",", ":")
            ),
        }
    )
    accumulators[(pattern, method, len(candidate_tuple), None)].add(
        correct=correct,
        normalized_gain=gain,
        covered=covered,
    )


def _summarize_tbcp_patterns(
    accumulators: Mapping[tuple[str, str, int, int | None], _MetricAccumulator],
    patterns: Sequence[str],
    *,
    include_reference_baselines: bool = True,
    include_diagonal_covariance_ablation: bool = False,
    include_defense_experiments: bool = False,
    include_batch_feedback_experiments: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    methods: tuple[tuple[str, int], ...] = (
        ("Direct Prediction", 0),
        ("Local-7", TBCP_BUDGET),
        ("Adaptive Local-7", TBCP_BUDGET),
        ("Posterior Top-7", TBCP_BUDGET),
        ("Posterior5+Hill2", TBCP_BUDGET),
        ("TBCP-7", TBCP_BUDGET),
    )
    if include_diagonal_covariance_ablation:
        methods += ((TBCP_DIAGONAL_METHOD, TBCP_BUDGET),)
    if include_defense_experiments:
        for budget in DEFAULT_BUDGETS:
            methods += ((f"Topology Open-loop Gain-{budget}", budget),)
            if budget != TBCP_BUDGET:
                methods += ((f"Posterior Top-{budget}", budget), (f"TBCP-{budget}", budget))
    if include_batch_feedback_experiments:
        methods += tuple((method, TBCP_BUDGET) for method in BATCH_TBCP_METHODS)
        if not include_defense_experiments:
            methods += (("Topology Open-loop Gain-7", TBCP_BUDGET),)
    if include_reference_baselines:
        methods += (("Oracle Local-7", TBCP_BUDGET), ("Full-64", NUM_BEAMS))
    rows: list[dict[str, Any]] = []
    uniform_rows: list[dict[str, Any]] = []
    for pattern in patterns:
        for method, budget in methods:
            rows.append(
                {
                    "group": pattern,
                    "method": method,
                    "probe_k": budget,
                    "offset_count": 0,
                    **accumulators[(pattern, method, budget, None)].summary(),
                }
            )
        if not include_reference_baselines:
            continue
        offset_metrics = [
            accumulators[(pattern, "Uniform-7", TBCP_BUDGET, offset)].summary()
            for offset in uniform_offsets(TBCP_BUDGET)
        ]
        uniform_row: dict[str, Any] = {
            "group": pattern,
            "method": "Uniform-7 Offset Mean",
            "probe_k": TBCP_BUDGET,
            "offset_count": NUM_BEAMS,
            "sample_count": offset_metrics[0]["sample_count"],
        }
        for metric in _probe_metric_names():
            available = [float(row[metric]) for row in offset_metrics if row.get(metric) is not None]
            uniform_row[metric] = float(np.mean(available)) if available else None
            uniform_row[f"{metric}_offset_std"] = float(np.std(available, ddof=0)) if available else None
        rows.append(uniform_row)
        for offset, metric in zip(uniform_offsets(TBCP_BUDGET), offset_metrics, strict=True):
            uniform_rows.append({"group": pattern, "offset": offset, **metric})
    return rows, uniform_rows


def _summarize_tbcp_groups(
    per_pattern: Sequence[Mapping[str, Any]],
    pattern_available: Mapping[str, Any],
) -> list[dict[str, Any]]:
    methods = tuple(dict.fromkeys(str(row["method"]) for row in per_pattern))
    groups: list[tuple[str, list[str], Any]] = [("All-15 Macro", list(pattern_available), np.mean)]
    for available, name in ((4, "Full"), (3, "Drop-1"), (2, "Drop-2"), (1, "Single")):
        patterns = [key for key, value in pattern_available.items() if int(value) == available]
        groups.append((name if available == 4 else f"{name} Macro", patterns, np.mean))
        if available != 4:
            groups.append((f"{name} Worst", patterns, np.min))
    result: list[dict[str, Any]] = []
    for group_name, patterns, reducer in groups:
        for method in methods:
            selected = [
                row for row in per_pattern if row["group"] in patterns and row["method"] == method
            ]
            if len(selected) != len(patterns):
                raise ValueError(f"TBCP summary is missing pattern rows for {group_name}/{method}.")
            result.append(
                _aggregate_probe_rows(
                    selected,
                    reducer=reducer,
                    prefix={
                    "group": group_name,
                    "method": method,
                    "probe_k": int(selected[0]["probe_k"]),
                    "pattern_count": len(patterns),
                    "sample_count": sum(int(row["sample_count"]) for row in selected),
                    },
                )
            )
    return result


def _probe_metric_names() -> tuple[str, ...]:
    return (
        "top1",
        "normalized_gain",
        "gt_coverage",
        "selection_accuracy_given_coverage",
        *(f"spectral_efficiency_ratio_{int(value)}db" for value in COMMUNICATION_SNR_DB),
    )


def _aggregate_probe_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    reducer: Any,
    prefix: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(prefix)
    for metric in _probe_metric_names():
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        result[metric] = float(reducer(values)) if values else None
    return result


def _write_tbcp_report(
    output: Path,
    config: Mapping[str, Any],
    group_summary: Sequence[Mapping[str, Any]],
) -> Path:
    lines = [
        "# TBCP-7 Validation Diagnostic",
        "",
        "- Scope: all 15 non-empty image/radar/GPS/LiDAR masks",
        "- Budget: K=7; no model training or checkpoint update",
        "- Likelihood: train-only aligned ULA-DFT relative log-gain mean/covariance",
        "- Claim boundary: validation-only, claim-ineligible, outer test not accessed",
        f"- Synthetic measurement error: sigma={config['measurement_model']['measurement_error_std_db']:.1f} dB; "
        f"replica={config['measurement_model']['noise_replica']}",
        f"- Policy version: `{config['primary_policy']['version']}`",
        f"- Likelihood fingerprint: `{config['likelihood']['metadata']['artifact_fingerprint']}`",
        "",
    ]
    if config.get("defense_experiments", {}).get("enabled") is True:
        lines[6:6] = [
            "- Defense controls: topology open-loop acquisition and preregistered K={3,5,7,9} curve",
            "- K=7 remains the frozen primary setting; the validation curve cannot select a new K",
        ]
    if config.get("batch_feedback_experiments", {}).get("enabled") is True:
        lines[6:6] = [
            "- Batch controls: fixed schedules `(2,2,3)`, `(2,5)`, and `(3,4)`; total slots remain K=7",
            "- Batch rounds are controller barriers, not measured hardware milliseconds",
        ]
    for group in (
        "All-15 Macro",
        "Full",
        "Drop-1 Macro",
        "Drop-1 Worst",
        "Drop-2 Macro",
        "Drop-2 Worst",
        "Single Macro",
        "Single Worst",
    ):
        rows = [row for row in group_summary if row["group"] == group]
        lines.extend(
            [
                f"## {group}",
                "",
                "| Method | K | Top-1 | Normalized gain | Rate@10dB | GT coverage | Select given coverage |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            coverage = "-" if row["gt_coverage"] is None else f"{100 * float(row['gt_coverage']):.2f}%"
            selection = (
                "-"
                if row["selection_accuracy_given_coverage"] is None
                else f"{100 * float(row['selection_accuracy_given_coverage']):.2f}%"
            )
            lines.append(
                f"| {row['method']} | {row['probe_k']} | {100 * float(row['top1']):.2f}% | "
                f"{100 * float(row['normalized_gain']):.2f}% | "
                f"{100 * float(row['spectral_efficiency_ratio_10db']):.2f}% | {coverage} | {selection} |"
            )
        tbcp = next(row for row in rows if row["method"] == "TBCP-7")
        lines.extend(["", "TBCP-7 paired aggregate deltas:"])
        baselines = ("Posterior Top-7", "Local-7", "Adaptive Local-7", "Posterior5+Hill2")
        if any(row["method"] == "Topology Open-loop Gain-7" for row in rows):
            baselines += ("Topology Open-loop Gain-7",)
        if any(row["method"] == TBCP_DIAGONAL_METHOD for row in rows):
            baselines += (TBCP_DIAGONAL_METHOD,)
        for batch_method in BATCH_TBCP_METHODS:
            if any(row["method"] == batch_method for row in rows):
                baselines += (batch_method,)
        for baseline in baselines:
            row = next(value for value in rows if value["method"] == baseline)
            lines.append(
                f"- vs {baseline}: Top-1 {100 * (float(tbcp['top1']) - float(row['top1'])):+.2f} pp, "
                f"gain {100 * (float(tbcp['normalized_gain']) - float(row['normalized_gain'])):+.2f} pp"
            )
        lines.append("")
    if config.get("batch_feedback_experiments", {}).get("enabled") is True:
        lines.extend(
            [
                "## Feedback Round Trade-off",
                "",
                "| Method | Slots | Measurement rounds | Feedback updates |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        round_rows = {
            "TBCP-7": {"slots": 7, "rounds": 6, "updates": 5},
            "Batch-TBCP-2+2+3": {"slots": 7, "rounds": 3, "updates": 2},
            "Batch-TBCP-2+5": {"slots": 7, "rounds": 2, "updates": 1},
            "Batch-TBCP-3+4": {"slots": 7, "rounds": 2, "updates": 1},
            "Topology Open-loop Gain-7": {"slots": 7, "rounds": 1, "updates": 0},
        }
        for method, values in round_rows.items():
            if any(row["method"] == method for row in group_summary):
                lines.append(
                    f"| {method} | {values['slots']} | {values['rounds']} | {values['updates']} |"
                )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "TBCP-7 uses only the sensing posterior, a train-only topology likelihood, and already requested RF "
            "measurements. The full validation power vector remains private to the simulator and is used only for "
            "the final metric/oracle checks. Synthetic dB error is a controlled stress test, not measured RF SNR; "
            "the replay does not establish performance under real switching latency or hardware constraints.",
        ]
    )
    path = output / "diagnostic_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def summarize_tbcp_replays(
    runs: Mapping[int, str | Path],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    if set(int(seed) for seed in runs) != {1, 2, 3} or len(runs) != 3:
        raise ValueError("TBCP formal development summary requires exactly seeds 1, 2, and 3.")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite a non-empty summary directory: {output}")
    loaded: list[tuple[int, Path, dict[str, Any], list[dict[str, str]]]] = []
    reference_binding: dict[str, Any] | None = None
    checkpoint_digests: set[str] = set()
    for seed in (1, 2, 3):
        result_path = Path(runs[seed]).resolve()
        result = json.loads(result_path.read_text(encoding="utf-8"))
        config = result.get("config") if isinstance(result, Mapping) else None
        if (
            not isinstance(config, Mapping)
            or config.get("claim_ineligible") is not True
            or config.get("outer_test_accessed") is not False
            or config.get("model_trained_or_updated") is not False
            or config.get("sample_rows") != 15 * int(config.get("unique_validation_samples", -1))
            or config.get("source", {}).get("bounded_evaluation") is not False
        ):
            raise ValueError(f"TBCP seed {seed} result is not a complete sealed validation replay.")
        patterns = config.get("patterns")
        if not isinstance(patterns, list) or len(patterns) != 15 or len(set(patterns)) != 15:
            raise ValueError(f"TBCP seed {seed} result does not contain exactly 15 unique sensing masks.")
        source = config["source"]
        checkpoint_path = Path(str(source.get("checkpoint_path", ""))).resolve()
        checkpoint_sha256, _checkpoint_size = checkpoint_file_digest(checkpoint_path)
        if checkpoint_sha256 != source.get("checkpoint_sha256"):
            raise ValueError(f"TBCP seed {seed} checkpoint file does not match result provenance.")
        likelihood_source = config.get("likelihood", {}).get("source", {})
        likelihood_path = Path(str(likelihood_source.get("path", ""))).resolve()
        likelihood_sha256, likelihood_size = checkpoint_file_digest(likelihood_path)
        if (
            likelihood_sha256 != likelihood_source.get("sha256")
            or likelihood_size != int(likelihood_source.get("size_bytes", -1))
        ):
            raise ValueError(f"TBCP seed {seed} likelihood file does not match result provenance.")
        matrix_report = Path(str(config["source"].get("matrix_report", ""))).resolve()
        matrix = json.loads(matrix_report.read_text(encoding="utf-8"))
        matrix_checkpoint = matrix.get("provenance", {}).get("checkpoint", {})
        if (
            int(matrix.get("experiment_seed", -1)) != seed
            or matrix.get("claim_ineligible") is not True
            or matrix.get("outer_test_accessed") is not False
            or matrix_checkpoint.get("sha256") != checkpoint_sha256
        ):
            raise ValueError(f"TBCP seed {seed} matrix report does not match the requested seed or claim boundary.")
        protocol = config["source"].get("data_protocol", {})
        topology = config["source"].get("prototype_topology", {})
        binding = {
            "policy_version": config.get("primary_policy", {}).get("version"),
            "likelihood_fingerprint": config.get("likelihood", {}).get("metadata", {}).get(
                "artifact_fingerprint"
            ),
            "protocol_fingerprint": protocol.get("protocol_fingerprint"),
            "validation_sample_id_hash": protocol.get("validation_sample_id_hash"),
            "validation_sample_count": protocol.get("validation_sample_count"),
            "validation_sample_order_sha256": config["source"].get(
                "validation_sample_order_sha256", protocol.get("validation_sample_id_hash")
            ),
            "topology_id": topology.get("id"),
            "topology_descriptor_sha256": topology.get("descriptor_sha256"),
            "topology_audit_sha256": topology.get("audit_sha256"),
            "patterns": config.get("patterns"),
            "sample_rows": config.get("sample_rows"),
            "budget": config.get("budget"),
            "covariance_modes": config.get("covariance_ablation", {}).get("modes"),
            "defense_experiments": config.get("defense_experiments"),
            "batch_feedback_experiments": config.get("batch_feedback_experiments"),
        }
        if binding["topology_id"] != EXPECTED_TOPOLOGY or any(
            not _is_sha256(binding[key])
            for key in (
                "likelihood_fingerprint",
                "protocol_fingerprint",
                "validation_sample_id_hash",
                "validation_sample_order_sha256",
                "topology_descriptor_sha256",
                "topology_audit_sha256",
            )
        ):
            raise ValueError(f"TBCP seed {seed} summary binding is incomplete or invalid.")
        if reference_binding is None:
            reference_binding = binding
        elif binding != reference_binding:
            raise ValueError("TBCP seed replays do not share one policy/likelihood/protocol/topology binding.")
        checkpoint_sha256 = str(config["source"].get("checkpoint_sha256", ""))
        if len(checkpoint_sha256) != 64:
            raise ValueError(f"TBCP seed {seed} result is missing checkpoint SHA256.")
        checkpoint_digests.add(checkpoint_sha256)
        pattern_summary = Path(str(config["artifacts"].get("per_pattern_summary", ""))).resolve()
        with pattern_summary.open(newline="", encoding="utf-8") as handle:
            pattern_rows = list(csv.DictReader(handle))
        loaded.append((seed, result_path, result, pattern_rows))
    if len(checkpoint_digests) != 3:
        raise ValueError("TBCP three-seed summary requires three distinct checkpoint lineages.")

    group_keys = [
        (str(row["group"]), str(row["method"]))
        for row in loaded[0][2]["group_summary"]
    ]
    pattern_keys = [
        (str(row["group"]), str(row["method"]))
        for row in loaded[0][3]
    ]
    group_rows = _aggregate_tbcp_seed_rows(loaded, group_keys, source="group")
    pattern_rows = _aggregate_tbcp_seed_rows(loaded, pattern_keys, source="pattern")
    stability = _tbcp_stability(pattern_rows)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "three_seed_group_summary.csv", group_rows)
    _write_csv(output / "three_seed_pattern_summary.csv", pattern_rows)
    (output / "stability.json").write_text(json.dumps(stability, indent=2) + "\n", encoding="utf-8")
    config = {
        "schema_version": 1,
        "claim_ineligible": True,
        "outer_test_accessed": False,
        "model_trained_or_updated": False,
        "seeds": [1, 2, 3],
        "binding": reference_binding,
        "runs": {str(seed): str(path) for seed, path, _result, _rows in loaded},
        "run_artifacts": {
            str(seed): {
                "result_sha256": _sha256_file(path),
                "per_pattern_summary_sha256": _sha256_file(
                    Path(str(result["config"]["artifacts"]["per_pattern_summary"])).resolve()
                ),
            }
            for seed, path, result, _rows in loaded
        },
        "artifacts": {
            "group_summary": str(output / "three_seed_group_summary.csv"),
            "pattern_summary": str(output / "three_seed_pattern_summary.csv"),
            "stability": str(output / "stability.json"),
        },
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    report = _write_tbcp_seed_report(output, config, group_rows, stability)
    result = {
        "output_dir": str(output),
        "config": config,
        "group_summary": group_rows,
        "stability": stability,
        "report": str(report),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _aggregate_tbcp_seed_rows(
    loaded: Sequence[tuple[int, Path, dict[str, Any], list[dict[str, str]]]],
    keys: Sequence[tuple[str, str]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for group, method in keys:
        selected: list[Mapping[str, Any]] = []
        for _seed, _path, run, pattern_rows in loaded:
            rows: Sequence[Mapping[str, Any]] = run["group_summary"] if source == "group" else pattern_rows
            matches = [row for row in rows if row["group"] == group and row["method"] == method]
            if len(matches) != 1:
                raise ValueError(f"TBCP seed summary has missing or duplicate row for {group}/{method}.")
            selected.append(matches[0])
        row: dict[str, Any] = {
            "group": group,
            "method": method,
            "probe_k": int(selected[0]["probe_k"]),
        }
        for metric in ("top1", "normalized_gain", "gt_coverage"):
            values = [
                None if item.get(metric) in (None, "", "None") else float(item[metric])
                for item in selected
            ]
            for seed, value in zip((1, 2, 3), values, strict=True):
                row[f"seed{seed}_{metric}"] = value
            present = np.asarray([value for value in values if value is not None], dtype=np.float64)
            row[f"mean_{metric}"] = None if present.size == 0 else float(present.mean())
            row[f"std_{metric}"] = None if present.size <= 1 else float(present.std(ddof=1))
        result.append(row)
    return result


def _tbcp_stability(pattern_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    patterns = tuple(dict.fromkeys(str(row["group"]) for row in pattern_rows))
    result: dict[str, Any] = {}
    baselines = ["Posterior Top-7", "Local-7", "Adaptive Local-7", "Posterior5+Hill2"]
    if any(row["method"] == "Topology Open-loop Gain-7" for row in pattern_rows):
        baselines.append("Topology Open-loop Gain-7")
    if any(row["method"] == TBCP_DIAGONAL_METHOD for row in pattern_rows):
        baselines.append(TBCP_DIAGONAL_METHOD)
    for method in BATCH_TBCP_METHODS:
        if any(row["method"] == method for row in pattern_rows):
            baselines.append(method)
    for baseline in baselines:
        top_deltas: list[list[float]] = []
        gain_deltas: list[list[float]] = []
        for pattern in patterns:
            tbcp = next(row for row in pattern_rows if row["group"] == pattern and row["method"] == "TBCP-7")
            control = next(row for row in pattern_rows if row["group"] == pattern and row["method"] == baseline)
            top_deltas.append(
                [float(tbcp[f"seed{seed}_top1"]) - float(control[f"seed{seed}_top1"]) for seed in (1, 2, 3)]
            )
            gain_deltas.append(
                [
                    float(tbcp[f"seed{seed}_normalized_gain"])
                    - float(control[f"seed{seed}_normalized_gain"])
                    for seed in (1, 2, 3)
                ]
            )
        result[baseline] = {
            "pattern_count": len(patterns),
            "top1_positive_all_three_seed_patterns": sum(min(values) > 0.0 for values in top_deltas),
            "gain_positive_all_three_seed_patterns": sum(min(values) > 0.0 for values in gain_deltas),
            "mean_top1_delta": float(np.mean(top_deltas)),
            "minimum_top1_delta": float(np.min(top_deltas)),
            "mean_normalized_gain_delta": float(np.mean(gain_deltas)),
            "minimum_normalized_gain_delta": float(np.min(gain_deltas)),
        }
    return result


def _write_tbcp_seed_report(
    output: Path,
    config: Mapping[str, Any],
    group_rows: Sequence[Mapping[str, Any]],
    stability: Mapping[str, Any],
) -> Path:
    lines = [
        "# TBCP-7 Three-seed Validation Summary",
        "",
        "- Seeds: 1, 2, 3; all runs use one train-only likelihood and one validation identity",
        "- Scope: all 15 non-empty four-sensing masks",
        "- Claim boundary: validation-only, claim-ineligible, outer test not accessed",
        f"- Policy: `{config['binding']['policy_version']}`",
        f"- Likelihood: `{config['binding']['likelihood_fingerprint']}`",
        "",
    ]
    for group in (
        "All-15 Macro",
        "Full",
        "Drop-1 Macro",
        "Drop-1 Worst",
        "Drop-2 Macro",
        "Drop-2 Worst",
        "Single Macro",
        "Single Worst",
    ):
        rows = [row for row in group_rows if row["group"] == group]
        lines.extend(
            [
                f"## {group}",
                "",
                "| Method | Top-1 mean +/- std | Gain mean +/- std |",
                "| --- | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['method']} | {100 * float(row['mean_top1']):.3f} +/- "
                f"{100 * float(row['std_top1'] or 0.0):.3f} | "
                f"{100 * float(row['mean_normalized_gain']):.3f} +/- "
                f"{100 * float(row['std_normalized_gain'] or 0.0):.3f} |"
            )
        lines.append("")
    defense = config["binding"].get("defense_experiments")
    if isinstance(defense, Mapping) and defense.get("enabled") is True:
        lines.extend(
            [
                "## Defense Budget Deltas",
                "",
                "| Group | K | TBCP - Open-loop Top-1 | TBCP - Open-loop gain | "
                "TBCP - Posterior Top-K Top-1 | TBCP - Posterior Top-K gain |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for group in ("All-15 Macro", "Single Worst"):
            for budget in DEFAULT_BUDGETS:
                tbcp = next(
                    row
                    for row in group_rows
                    if row["group"] == group and row["method"] == f"TBCP-{budget}"
                )
                open_loop = next(
                    row
                    for row in group_rows
                    if row["group"] == group
                    and row["method"] == f"Topology Open-loop Gain-{budget}"
                )
                posterior = next(
                    row
                    for row in group_rows
                    if row["group"] == group and row["method"] == f"Posterior Top-{budget}"
                )
                lines.append(
                    f"| {group} | {budget} | "
                    f"{100 * (float(tbcp['mean_top1']) - float(open_loop['mean_top1'])):+.3f} pp | "
                    f"{100 * (float(tbcp['mean_normalized_gain']) - float(open_loop['mean_normalized_gain'])):+.3f} pp | "
                    f"{100 * (float(tbcp['mean_top1']) - float(posterior['mean_top1'])):+.3f} pp | "
                    f"{100 * (float(tbcp['mean_normalized_gain']) - float(posterior['mean_normalized_gain'])):+.3f} pp |"
                )
        lines.append("")
    lines.extend(["## Pattern Stability", ""])
    for baseline, values in stability.items():
        lines.append(
            f"- vs {baseline}: Top-1 positive on {values['top1_positive_all_three_seed_patterns']}/"
            f"{values['pattern_count']} patterns for every seed; gain positive on "
            f"{values['gain_positive_all_three_seed_patterns']}/{values['pattern_count']}; mean/min deltas "
            f"{100 * values['mean_top1_delta']:+.3f}/{100 * values['minimum_top1_delta']:+.3f} pp Top-1 and "
            f"{100 * values['mean_normalized_gain_delta']:+.3f}/"
            f"{100 * values['minimum_normalized_gain_delta']:+.3f} pp gain."
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "These are deterministic validation replays for method selection, not outer-test or hardware-noise claims.",
        ]
    )
    path = output / "three_seed_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _probe_and_record(
    ledger: csv.DictWriter,
    accumulators: Mapping[tuple[str, str, int, int | None], _MetricAccumulator],
    simulator: BeamProbeSimulator,
    *,
    sample_id: str,
    pattern: str,
    gt_beam: int,
    pred_beam: int,
    strategy: str,
    k: int,
    candidates: tuple[int, ...],
    offset: int | None,
    adaptive_spacing: int | None = None,
    selected_posterior_mass: float | None = None,
    posterior_statistics: Mapping[str, float] | None = None,
) -> None:
    measured = simulator.probe(sample_id, candidates)
    final_beam = candidates[int(np.argmax(measured))]
    covered = gt_beam in candidates
    if (final_beam == gt_beam) != covered:
        raise ValueError(
            f"Noiseless probe correct/coverage mismatch for {sample_id}, strategy={strategy}, K={k}; "
            "check label/power ties or protocol drift."
        )
    _record_strategy(
        ledger,
        accumulators,
        simulator,
        sample_id=sample_id,
        pattern=pattern,
        gt_beam=gt_beam,
        pred_beam=pred_beam,
        strategy=strategy,
        k=k,
        candidates=candidates,
        offset=offset,
        final_beam=final_beam,
        covered=covered,
        adaptive_spacing=adaptive_spacing,
        selected_posterior_mass=selected_posterior_mass,
        posterior_statistics=posterior_statistics,
    )


def _record_strategy(
    ledger: csv.DictWriter,
    accumulators: Mapping[tuple[str, str, int, int | None], _MetricAccumulator],
    simulator: BeamProbeSimulator,
    *,
    sample_id: str,
    pattern: str,
    gt_beam: int,
    pred_beam: int,
    strategy: str,
    k: int,
    candidates: Sequence[int],
    offset: int | None,
    final_beam: int,
    covered: bool | None,
    adaptive_spacing: int | None = None,
    selected_posterior_mass: float | None = None,
    posterior_statistics: Mapping[str, float] | None = None,
) -> None:
    gain = simulator.normalized_gain(sample_id, final_beam)
    correct = final_beam == gt_beam
    statistics = dict(posterior_statistics or {})
    ledger.writerow(
        {
            "sample_id": sample_id,
            "missing_pattern": pattern,
            "gt_beam": gt_beam,
            "pred_beam": pred_beam,
            "strategy": strategy,
            "K": k,
            "uniform_offset": "" if offset is None else offset,
            "adaptive_spacing": "" if adaptive_spacing is None else adaptive_spacing,
            "selected_posterior_mass": (
                "" if selected_posterior_mass is None else f"{float(selected_posterior_mass):.12g}"
            ),
            "beam_circular_mean": statistics.get("beam_circular_mean", ""),
            "beam_circular_variance": statistics.get("beam_circular_variance", ""),
            "beam_variance": statistics.get("beam_variance", ""),
            "beam_spread": statistics.get("beam_spread", ""),
            "beam_normalized_entropy": statistics.get("beam_normalized_entropy", ""),
            "probe_indices": json.dumps(list(candidates), separators=(",", ":")),
            "final_beam": final_beam,
            "correct": int(correct),
            "normalized_gain": f"{gain:.12g}",
            "gt_covered": "" if covered is None else int(covered),
        }
    )
    accumulators[(pattern, strategy, k, offset)].add(correct=correct, normalized_gain=gain, covered=covered)


def _summarize_accumulators(
    accumulators: Mapping[tuple[str, str, int, int | None], _MetricAccumulator],
    budgets: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    uniform_rows: list[dict[str, Any]] = []
    for pattern in SEVERE_SINGLE_PATTERNS:
        for strategy, budget in _strategy_budget_keys(budgets):
            if strategy != "Global Uniform":
                metric = accumulators[(pattern, strategy, budget, None)].summary()
                rows.append({"group": pattern, "method": strategy, "probe_k": budget, "offset_count": 0, **metric})
                continue
            offsets = uniform_offsets(budget)
            metrics = [accumulators[(pattern, strategy, budget, offset)].summary() for offset in offsets]
            row: dict[str, Any] = {
                "group": pattern,
                "method": strategy,
                "probe_k": budget,
                "offset_count": len(offsets),
                "sample_count": metrics[0]["sample_count"] * len(offsets),
            }
            for name in ("top1", "normalized_gain", "gt_coverage"):
                values = np.asarray([float(metric[name]) for metric in metrics], dtype=np.float64)
                row[name] = float(values.mean())
                row[f"{name}_offset_std"] = float(values.std(ddof=0))
                row[f"{name}_best_offset"] = int(offsets[int(values.argmax())])
                row[f"{name}_worst_offset"] = int(offsets[int(values.argmin())])
            rows.append(row)
            for offset, metric in zip(offsets, metrics, strict=True):
                uniform_rows.append(
                    {
                        "group": pattern,
                        "probe_k": budget,
                        "offset": offset,
                        **metric,
                        "top1_offset_mean": row["top1"],
                        "top1_offset_std": row["top1_offset_std"],
                        "top1_best_offset": row["top1_best_offset"],
                        "top1_worst_offset": row["top1_worst_offset"],
                        "normalized_gain_offset_mean": row["normalized_gain"],
                        "normalized_gain_offset_std": row["normalized_gain_offset_std"],
                        "gt_coverage_offset_mean": row["gt_coverage"],
                        "gt_coverage_offset_std": row["gt_coverage_offset_std"],
                    }
                )
    return rows, uniform_rows


def _aggregate_single_rows(per_mask: list[dict[str, Any]], budgets: Sequence[int]) -> list[dict[str, Any]]:
    rows = list(per_mask)
    for method, budget in _strategy_budget_keys(budgets):
        selected = [row for row in per_mask if row["method"] == method and row["probe_k"] == budget]
        if len(selected) != len(SEVERE_SINGLE_PATTERNS):
            raise ValueError(f"Missing per-mask summary rows for {method}, K={budget}.")
        for group, reducer in (("Single Macro", np.mean), ("Single Worst", np.min)):
            row: dict[str, Any] = {
                "group": group,
                "method": method,
                "probe_k": budget,
                "offset_count": selected[0].get("offset_count", 0),
                "sample_count": sum(int(value["sample_count"]) for value in selected),
            }
            for metric in ("top1", "normalized_gain", "gt_coverage"):
                values = [value[metric] for value in selected if value.get(metric) is not None]
                row[metric] = float(reducer(values)) if values else None
            if method == "Global Uniform":
                for metric in ("top1", "normalized_gain", "gt_coverage"):
                    row[f"{metric}_offset_std"] = float(np.mean([value[f"{metric}_offset_std"] for value in selected]))
            rows.append(row)
    return rows


def _strategy_budget_keys(budgets: Sequence[int]) -> list[tuple[str, int]]:
    keys = [("Direct Prediction", 0)]
    for budget in budgets:
        keys.extend((method, int(budget)) for method in ("Global Uniform", "Local Scan"))
        if int(budget) == ADAPTIVE_BUDGET:
            keys.extend(
                (
                    ("Adaptive Local", ADAPTIVE_BUDGET),
                    ("Posterior Top-K", ADAPTIVE_BUDGET),
                )
            )
        keys.append(("Oracle Local", int(budget)))
    keys.append(("Full Sweep", NUM_BEAMS))
    return keys


def _summarize_adaptive_spacing(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Adaptive Local-7 produced no spacing records.")
    summary: list[dict[str, Any]] = []
    for group in (*SEVERE_SINGLE_PATTERNS, "Single Macro"):
        group_rows = list(rows) if group == "Single Macro" else [row for row in rows if row["group"] == group]
        if not group_rows:
            raise ValueError(f"Adaptive Local-7 has no rows for {group}.")
        for spacing in ADAPTIVE_SPACINGS:
            selected = [row for row in group_rows if int(row["adaptive_spacing"]) == spacing]
            summary.append(
                {
                    "group": group,
                    "adaptive_spacing": spacing,
                    "sample_count": len(selected),
                    "fraction": len(selected) / len(group_rows),
                    "beam_spread_mean": (
                        float(np.mean([float(row["beam_spread"]) for row in selected])) if selected else None
                    ),
                    "beam_normalized_entropy_mean": (
                        float(np.mean([float(row["beam_normalized_entropy"]) for row in selected]))
                        if selected
                        else None
                    ),
                    "selected_posterior_mass_mean": (
                        float(np.mean([float(row["selected_posterior_mass"]) for row in selected]))
                        if selected
                        else None
                    ),
                }
            )
    return summary


def _distance_sanity(
    counts: Mapping[str, np.ndarray],
    totals: Mapping[str, int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for pattern in SEVERE_SINGLE_PATTERNS:
        total = int(totals[pattern])
        result[pattern] = {f"distance_le_{threshold}": float(counts[pattern][threshold - 1] / total) for threshold in range(1, 6)}
    result["Single Macro"] = {
        key: float(np.mean([result[pattern][key] for pattern in SEVERE_SINGLE_PATTERNS]))
        for key in result[SEVERE_SINGLE_PATTERNS[0]]
    }
    result["Single Worst"] = {
        key: float(np.min([result[pattern][key] for pattern in SEVERE_SINGLE_PATTERNS]))
        for key in result[SEVERE_SINGLE_PATTERNS[0]]
    }
    return result


def _feasibility_decision(per_mask: list[dict[str, Any]], summary: list[dict[str, Any]]) -> dict[str, Any]:
    macro = {(row["method"], row["probe_k"]): row for row in summary if row["group"] == "Single Macro"}
    deltas: dict[str, dict[str, float | int]] = {}
    positive_pattern_comparisons = 0
    for budget in (5, 7, 9):
        local = macro[("Local Scan", budget)]
        uniform = macro[("Global Uniform", budget)]
        positive_masks = 0
        for pattern in SEVERE_SINGLE_PATTERNS:
            local_row = next(row for row in per_mask if row["group"] == pattern and row["method"] == "Local Scan" and row["probe_k"] == budget)
            uniform_row = next(row for row in per_mask if row["group"] == pattern and row["method"] == "Global Uniform" and row["probe_k"] == budget)
            positive_masks += int(local_row["top1"] > uniform_row["top1"])
        positive_pattern_comparisons += positive_masks
        deltas[str(budget)] = {
            "delta_top1": float(local["top1"] - uniform["top1"]),
            "delta_normalized_gain": float(local["normalized_gain"] - uniform["normalized_gain"]),
            "delta_coverage": float(local["gt_coverage"] - uniform["gt_coverage"]),
            "positive_top1_pattern_count": positive_masks,
        }
    passing_k = sum(
        int(value["delta_top1"] > 0 and value["delta_normalized_gain"] > 0 and value["positive_top1_pattern_count"] >= 3)
        for value in deltas.values()
    )
    positive_macro_k = sum(int(value["delta_top1"] > 0 and value["delta_normalized_gain"] > 0) for value in deltas.values())
    if passing_k == 3:
        verdict = "PASS"
        reason = "Sensing-guided local beam probing has experimental feasibility."
    elif positive_macro_k >= 2 or positive_pattern_comparisons >= 6:
        verdict = "PARTIAL"
        reason = "Need to inspect sensing localization quality before method expansion."
    else:
        verdict = "FAIL"
        reason = "Current sensing prediction does not provide a useful local search prior."
    return {
        "verdict": verdict,
        "reason": reason,
        "preregistered_rule": "PASS requires K=5/7/9 positive Macro Top1 and gain deltas plus >=3/4 positive masks per K.",
        "deltas": deltas,
        "positive_pattern_budget_comparisons": positive_pattern_comparisons,
    }


def _plot_results(
    output: Path,
    summary: list[dict[str, Any]],
    uniform_summary: list[dict[str, Any]],
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    macro = [row for row in summary if row["group"] == "Single Macro"]
    paths: dict[str, str] = {}
    for metric, filename, ylabel in (
        ("top1", "fig_top1_vs_budget.png", "Final Top-1"),
        ("normalized_gain", "fig_gain_vs_budget.png", "Normalized Beamforming Gain"),
        ("gt_coverage", "fig_coverage_vs_budget.png", "GT Coverage"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        colors = {"Global Uniform": "#2878B5", "Local Scan": "#D95319", "Oracle Local": "#2A9D62"}
        for method in ("Global Uniform", "Local Scan", "Oracle Local"):
            rows = sorted((row for row in macro if row["method"] == method), key=lambda row: row["probe_k"])
            x = [row["probe_k"] for row in rows]
            y = [row[metric] for row in rows]
            ax.plot(x, y, marker="o", linewidth=2, color=colors[method], label=method)
            if method == "Global Uniform":
                std = [row.get(f"{metric}_offset_std", 0.0) for row in rows]
                ax.fill_between(x, np.asarray(y) - np.asarray(std), np.asarray(y) + np.asarray(std), color=colors[method], alpha=0.14)
        for method, color, marker in (
            ("Adaptive Local", "#7A3E9D", "s"),
            ("Posterior Top-K", "#8C6D31", "^"),
        ):
            row = next(value for value in macro if value["method"] == method and value["probe_k"] == ADAPTIVE_BUDGET)
            ax.scatter(
                [ADAPTIVE_BUDGET],
                [row[metric]],
                color=color,
                marker=marker,
                s=55,
                label=f"{method} (K={ADAPTIVE_BUDGET})",
            )
        full = next(row for row in macro if row["method"] == "Full Sweep")
        ax.axhline(full[metric], color="#555555", linestyle="--", linewidth=1.4, label="Full Sweep (K=64)")
        if metric != "gt_coverage":
            direct = next(row for row in macro if row["method"] == "Direct Prediction")
            ax.scatter([0], [direct[metric]], color="#7A3E9D", marker="D", s=45, label="Direct Prediction (K=0)")
        ax.set_xlabel("Number of Probed Beams")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 3, 5, 7, 9])
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        path = output / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths[metric] = str(path)
    return paths


def _write_report(
    output: Path,
    config: Mapping[str, Any],
    summary: list[dict[str, Any]],
    per_mask: list[dict[str, Any]],
    spacing_summary: list[dict[str, Any]],
    sanity: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> Path:
    macro = {(row["method"], row["probe_k"]): row for row in summary if row["group"] == "Single Macro"}
    full = macro[("Full Sweep", NUM_BEAMS)]
    local7 = macro[("Local Scan", ADAPTIVE_BUDGET)]
    adaptive7 = macro[("Adaptive Local", ADAPTIVE_BUDGET)]
    posterior7 = macro[("Posterior Top-K", ADAPTIVE_BUDGET)]
    deltas = decision["deltas"]
    best_pattern = max(
        (
            (
                local["top1"] - uniform["top1"],
                pattern,
                budget,
            )
            for budget in DEFAULT_BUDGETS
            for pattern in SEVERE_SINGLE_PATTERNS
            for local in per_mask
            if local["group"] == pattern and local["method"] == "Local Scan" and local["probe_k"] == budget
            for uniform in per_mask
            if uniform["group"] == pattern and uniform["method"] == "Global Uniform" and uniform["probe_k"] == budget
        ),
        key=lambda value: value[0],
    )
    useful_k = [
        budget
        for budget in DEFAULT_BUDGETS
        if macro[("Local Scan", budget)]["top1"] > macro[("Global Uniform", budget)]["top1"]
        and macro[("Local Scan", budget)]["normalized_gain"] > macro[("Global Uniform", budget)]["normalized_gain"]
    ]
    lines = [
        "# Sensing-guided Uncertainty-adaptive Local Beam Probing Diagnostic",
        "",
        f"- Verdict: **{decision['verdict']}** - {decision['reason']}",
        f"- Checkpoint: `{config['source']['checkpoint_path']}`",
        f"- Checkpoint SHA256: `{config['source']['checkpoint_sha256']}`",
        f"- Split: `{EXPECTED_PROTOCOL}` seed 0 validation; source has "
        f"{int(config['source']['data_protocol'].get('validation_sample_count', config['unique_validation_samples'])):,} "
        "windows, this run evaluated "
        f"{config['unique_validation_samples']:,} unique windows",
        "- Evidence: frozen checkpoint-bound validation predictions; no training or model update",
        f"- Adaptive policy: `{config['probing_policy']['version']}`, fixed K={ADAPTIVE_BUDGET}, "
        "posterior-mass selection over spacings 1/2/4/8",
        "- Topology: audited ULA-DFT phase cycle with modulo-64 adjacency; not a world-azimuth ring",
        "- Claim boundary: validation-only, claim-ineligible, outer test not accessed",
        "",
        "## Core Results - Single Macro",
        "",
        "| Method | Probe K | Top-1 | Norm Gain | GT Coverage |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    ordered = [macro[("Direct Prediction", 0)]]
    for budget in DEFAULT_BUDGETS:
        ordered.extend(macro[(method, budget)] for method in ("Global Uniform", "Local Scan"))
        if budget == ADAPTIVE_BUDGET:
            ordered.extend(macro[(method, budget)] for method in ("Adaptive Local", "Posterior Top-K"))
        ordered.append(macro[("Oracle Local", budget)])
    ordered.append(full)
    for row in ordered:
        coverage = "-" if row["gt_coverage"] is None else f"{100 * row['gt_coverage']:.2f}%"
        lines.append(
            f"| {row['method']} | {row['probe_k']} | {100 * row['top1']:.2f}% | "
            f"{100 * row['normalized_gain']:.2f}% | {coverage} |"
        )
    lines.extend(["", "## Key Deltas", ""])
    for budget in (5, 7, 9):
        value = deltas[str(budget)]
        lines.append(
            f"- K={budget}: Delta Top-1 {100 * value['delta_top1']:+.2f} pp, "
            f"Delta gain {100 * value['delta_normalized_gain']:+.2f} pp, "
            f"Delta coverage {100 * value['delta_coverage']:+.2f} pp; "
            f"Local wins on {value['positive_top1_pattern_count']}/4 masks."
        )
    lines.extend(
        [
            "",
            "## K=7 Uncertainty Policy",
            "",
            f"- Adaptive vs fixed Local7: Top-1 {100 * (adaptive7['top1'] - local7['top1']):+.2f} pp, "
            f"normalized gain {100 * (adaptive7['normalized_gain'] - local7['normalized_gain']):+.2f} pp。",
            f"- Posterior Top7 vs fixed Local7: Top-1 {100 * (posterior7['top1'] - local7['top1']):+.2f} pp, "
            f"normalized gain {100 * (posterior7['normalized_gain'] - local7['normalized_gain']):+.2f} pp。",
            "",
            "| Pattern | Fixed Local7 Top-1 | Adaptive Local7 Top-1 | Posterior Top7 Top-1 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for pattern in SEVERE_SINGLE_PATTERNS:
        values = {
            row["method"]: row
            for row in per_mask
            if row["group"] == pattern and row["probe_k"] == ADAPTIVE_BUDGET
        }
        lines.append(
            f"| {pattern} | {100 * values['Local Scan']['top1']:.2f}% | "
            f"{100 * values['Adaptive Local']['top1']:.2f}% | "
            f"{100 * values['Posterior Top-K']['top1']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "| Adaptive spacing | Fraction | Mean spread | Mean entropy | Mean selected mass |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in spacing_summary:
        if row["group"] != "Single Macro":
            continue
        spread = "-" if row["beam_spread_mean"] is None else f"{row['beam_spread_mean']:.3f}"
        entropy = (
            "-" if row["beam_normalized_entropy_mean"] is None else f"{row['beam_normalized_entropy_mean']:.3f}"
        )
        mass = (
            "-" if row["selected_posterior_mass_mean"] is None else f"{row['selected_posterior_mass_mean']:.3f}"
        )
        lines.append(
            f"| {row['adaptive_spacing']} | {100 * row['fraction']:.2f}% | {spread} | {entropy} | {mass} |"
        )
    lines.extend(
        [
            "",
            "## Q1-Q6",
            "",
            f"**Q1. Local-K 是否稳定优于 Uniform-K？** {decision['verdict']}: {decision['reason']}",
            "",
            f"**Q2. 优势最明显的 severe pattern？** {best_pattern[1]} at K={best_pattern[2]}, "
            f"Top-1 delta {100 * best_pattern[0]:+.2f} pp。",
            "",
            f"**Q3. 最小多少 K 获得 Macro Top-1 与 gain 同时提升？** {min(useful_k) if useful_k else '无'}。",
            "",
            f"**Q4. Local-K 与 Full-64 差距？** K=9 的 Top-1/Norm Gain 距 Full-64 分别为 "
            f"{100 * (full['top1'] - macro[('Local Scan', 9)]['top1']):.2f}/"
            f"{100 * (full['normalized_gain'] - macro[('Local Scan', 9)]['normalized_gain']):.2f} pp。",
            "",
            f"**Q5. Oracle 与 Actual Local 差距？** K=9 的 Top-1/Norm Gain gap 为 "
            f"{100 * (macro[('Oracle Local', 9)]['top1'] - macro[('Local Scan', 9)]['top1']):.2f}/"
            f"{100 * (macro[('Oracle Local', 9)]['normalized_gain'] - macro[('Local Scan', 9)]['normalized_gain']):.2f} pp。",
            "",
            "**Q6. 是否满足无泄漏/真实 K-beam query？ YES.** Local 只使用 pred_beam/K/topology；Adaptive/Posterior Top7 "
            "只额外使用预测 p[64]；Uniform 只使用 K/offset/codebook size。它们均不读取 GT、CSI/channel 或完整 gain。完整 power 仅由 simulator 私有持有，"
            "final beam 只从 probe 返回的 K 个 measurement 中选择。Oracle 单独标记 claim-ineligible。",
            "",
            "## Distance Sanity - Single Macro",
            "",
        ]
    )
    for key, value in sanity["Single Macro"].items():
        lines.append(f"- `{key}`: {100 * value:.2f}%")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "本数据的 GT 是同一无噪声 64-beam power vector 的 argmax，因此 scan Top-1 应等于 GT coverage。"
            "Normalized gain 仍用于衡量未覆盖 GT 时邻近 beam 的通信保真度。该结果只判断 feasibility，不是 test 或论文级 claim。",
        ]
    )
    path = output / "diagnostic_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate_scan_request(num_beams: int, k: int) -> tuple[int, int]:
    beams, budget = int(num_beams), int(k)
    if beams <= 0 or budget <= 0 or budget > beams or budget % 2 == 0:
        raise ValueError("Local probing requires an odd K in [1, num_beams].")
    return beams, budget


def _validate_beam(value: int, num_beams: int, name: str) -> int:
    beam = int(value)
    if beam < 0 or beam >= int(num_beams):
        raise ValueError(f"{name} must be in [0, {int(num_beams) - 1}], got {value}.")
    return beam


def _validate_candidates(values: Sequence[int] | Any, expected: int, num_beams: int) -> tuple[int, ...]:
    candidates = tuple(int(value) for value in values)
    if len(candidates) != int(expected) or len(set(candidates)) != len(candidates):
        raise ValueError(f"candidate_beams must contain exactly {expected} unique beams.")
    if any(value < 0 or value >= int(num_beams) for value in candidates):
        raise ValueError(f"candidate_beams must be in [0, {int(num_beams) - 1}].")
    return candidates


def _topology_binding(cfg: Mapping[str, Any]) -> dict[str, Any]:
    loss = cfg.get("loss", {})
    topology_loss = loss.get("four_modal_topology", {}) if isinstance(loss, Mapping) else {}
    topology = topology_loss.get("prototype_topology") if isinstance(topology_loss, Mapping) else None
    if not isinstance(topology, Mapping):
        raise ValueError("Topology likelihood requires four-modal prototype topology provenance.")
    result = {
        "id": str(topology.get("id", "")),
        "descriptor_sha256": str(topology.get("descriptor_sha256", "")),
        "audit_sha256": str(topology.get("audit_sha256", "")),
    }
    if result["id"] != EXPECTED_TOPOLOGY or any(
        len(result[key]) != 64 or any(character not in "0123456789abcdef" for character in result[key].lower())
        for key in ("descriptor_sha256", "audit_sha256")
    ):
        raise ValueError("Topology likelihood requires the audited ULA-DFT topology identity and SHA256.")
    runtime = cfg.get("runtime", {})
    resolver = runtime.get("topology_predictor_resolver", {}) if isinstance(runtime, Mapping) else {}
    runtime_topology = resolver.get("prototype_topology") if isinstance(resolver, Mapping) else None
    if isinstance(runtime_topology, Mapping) and any(
        runtime_topology.get(key) != result[key] for key in ("id", "descriptor_sha256", "audit_sha256")
    ):
        raise ValueError("Topology-predictor runtime and loss topology provenance disagree.")
    return result


def _sha256_lines(values: Sequence[str] | Any) -> str:
    return hashlib.sha256("\n".join(sorted(str(value) for value in values)).encode("utf-8")).hexdigest()


def _sha256_ordered_lines(values: Sequence[str] | Any) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def _matched_standard_normal(
    *,
    sample_id: str,
    beam_index: int,
    noise_seed: int,
    noise_replica: int,
) -> float:
    payload = "\x1f".join(
        (
            ROBUSTNESS_NOISE_MODEL_VERSION,
            str(int(noise_seed)),
            str(int(noise_replica)),
            str(sample_id),
            str(int(beam_index)),
        )
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    scale = float(1 << 64)
    uniform_1 = (int.from_bytes(digest[:8], "big") + 0.5) / scale
    uniform_2 = (int.from_bytes(digest[8:16], "big") + 0.5) / scale
    return float(np.sqrt(-2.0 * np.log(uniform_1)) * np.cos(2.0 * np.pi * uniform_2))


def _spectral_efficiency_ratio(normalized_gain: float, snr_db: int) -> float:
    gain = float(normalized_gain)
    if not np.isfinite(gain) or gain <= 0.0 or gain > 1.0:
        raise ValueError("normalized_gain must be finite in (0, 1].")
    reference_snr = 10.0 ** (float(snr_db) / 10.0)
    return float(np.log2(1.0 + reference_snr * gain) / np.log2(1.0 + reference_snr))


def _is_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "BATCH_FEEDBACK_EXPERIMENT_VERSION",
    "BATCH_TBCP_METHODS",
    "ADAPTIVE_BUDGET",
    "ADAPTIVE_OFFSETS",
    "ADAPTIVE_SPACINGS",
    "BeamProbeSimulator",
    "DEFAULT_BUDGETS",
    "DEFENSE_EXPERIMENT_VERSION",
    "ProbeEvidence",
    "SEVERE_SINGLE_PATTERNS",
    "TBCP_DIAGONAL_METHOD",
    "build_adaptive_local_candidates",
    "build_local_candidates",
    "build_oracle_local_candidates",
    "build_posterior_topk_candidates",
    "build_train_power_index",
    "build_uniform_candidates",
    "build_validation_power_index",
    "load_probe_evidence",
    "run_probe_diagnostic",
    "run_tbcp_probe_diagnostic",
    "run_tbcp_robustness_sensitivity",
    "select_probe_evidence_by_hash",
    "summarize_tbcp_robustness_replays",
    "summarize_tbcp_replays",
    "uniform_offsets",
]
