#!/usr/bin/env python3
"""MMW final-test seal preflight.

This tool intentionally stops before the outer test is opened.  It validates the
frozen validation artifacts which are needed by the later, one-shot final-test
panel and writes an immutable seal manifest.  In particular, it does *not*
call the MMW protocol binder, build a dataloader, read a split CSV, or inspect
test labels/radio power.

The later test evaluator is deliberately not implemented here.  Keeping the
preflight as a small, read-only surface makes it possible to review the exact
inputs before granting the separate ``--evaluate-test`` authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from kd_sensing.utils.checkpoint import load_torch_payload, validate_checkpoint_publication


PROTOCOL_ID = "mmw_id_stratified_block_v1"
TOPOLOGY_ID = "ula_dft_phase_cycle_v1"
EXPECTED_MODALITIES = ("image", "radar", "gps", "lidar")
EXPECTED_PATTERN_MASKS: dict[str, tuple[int, ...]] = {
    "full": (1, 1, 1, 1),
    "missing_image": (0, 1, 1, 1),
    "missing_radar": (1, 0, 1, 1),
    "missing_gps": (1, 1, 0, 1),
    "missing_lidar": (1, 1, 1, 0),
    "missing_image_radar": (0, 0, 1, 1),
    "missing_image_gps": (0, 1, 0, 1),
    "missing_image_lidar": (0, 1, 1, 0),
    "missing_radar_gps": (1, 0, 0, 1),
    "missing_radar_lidar": (1, 0, 1, 0),
    "missing_gps_lidar": (1, 1, 0, 0),
    "image_only": (1, 0, 0, 0),
    "radar_only": (0, 1, 0, 0),
    "gps_only": (0, 0, 1, 0),
    "lidar_only": (0, 0, 0, 1),
}
EXPECTED_PATTERN_NAMES = tuple(EXPECTED_PATTERN_MASKS)
PROTOCOL_FIELDS = (
    "mode",
    "protocol_id",
    "protocol_version",
    "split_protocol_version",
    "manifest_version",
    "assignment_algorithm",
    "protocol_fingerprint",
    "audit_id",
    "audit_sha256",
    "split_seed",
    "block_size",
    "split_manifest_hash",
    "data_source_hash",
    "window_config_hash",
    "weather_binding",
    "train_role",
    "validation_role",
    "test_role",
    "train_sample_count",
    "validation_sample_count",
    "test_sample_count",
    "train_sample_id_hash",
    "validation_sample_id_hash",
    "test_sample_id_hash",
    "test_evaluated",
)
TRAIN_ONLY_NORMALIZATION_FIELDS = (
    "fit_split",
    "source_split",
    "sample_id_hash",
    "protocol_fingerprint",
    "protocol_version",
    "split_seed",
    "block_size",
    "split_manifest_hash",
    "data_source_hash",
    "window_config_hash",
    "weather_binding",
)


@dataclass(frozen=True)
class CandidateSpec:
    """Paths for one frozen method/seed validation bundle."""

    method: str
    family: str
    seed: int
    config: str
    checkpoint: str
    checkpoint_sidecar: str
    matrix_report: str
    evidence: str


def _candidate_specs() -> tuple[CandidateSpec, ...]:
    specs: list[CandidateSpec] = []
    roots = {
        "Prototype-only": (
            "prototype",
            "outputs/four_modal_topology_predictor_masked_feature_fusion/"
            "masked_feature_fusion_prototype_only_seed{seed}",
        ),
        "Hard": (
            "hard",
            "outputs/four_modal_topology_predictor_masked_feature_fusion/"
            "masked_feature_fusion_off_seed{seed}",
        ),
        "RMBP-MM-local": (
            "rmbp_mm",
            "outputs/mmw_sensing_baselines_no_history_v2/rmbp_mm/train_seed{seed}",
        ),
        "AMBER-Full-local": (
            "amber_full",
            "outputs/mmw_sensing_baselines_no_history_v2/amber_full/train_seed{seed}",
        ),
    }
    for method, (family, root_template) in roots.items():
        for seed in (1, 2, 3):
            run_root = root_template.format(seed=seed)
            if method in {"Prototype-only", "Hard"}:
                report_root = "outputs/four_modal_topology_predictor_masked_feature_fusion/reports"
                stem = (
                    "masked_feature_fusion_prototype_only_seed{seed}"
                    if method == "Prototype-only"
                    else "masked_feature_fusion_off_seed{seed}"
                ).format(seed=seed)
                matrix_report = f"{report_root}/{stem}_matrix.json"
                evidence = f"{report_root}/{stem}_matrix_sample_records.pt"
            else:
                report_root = f"outputs/fair_ablation_baseline_panel/baseline_evaluations/{family}_seed{seed}"
                matrix_report = f"{report_root}/baseline_matrix_report.json"
                evidence = f"{report_root}/baseline_sample_records.pt"
            specs.append(
                CandidateSpec(
                    method=method,
                    family=family,
                    seed=seed,
                    config=f"{run_root}/resolved_config.yaml",
                    checkpoint=f"{run_root}/checkpoints/best.pth",
                    checkpoint_sidecar=f"{run_root}/checkpoints/best.pth.json",
                    matrix_report=matrix_report,
                    evidence=evidence,
                )
            )
    # Keep the method order stable in the manifest and in review diffs.
    return tuple(specs)


DEFAULT_LIKELIHOOD = "outputs/tbcp7_probe_calibration/mmw_id_block_seed0_train_likelihood_v1.npz"
SOURCE_FILES = (
    "tools/mmw_final_test_panel.py",
    "src/kd_sensing/eval/beam_probe_diagnostic.py",
    "src/kd_sensing/eval/beam_topology_likelihood.py",
    "src/kd_sensing/eval/topology_predictor.py",
    "src/kd_sensing/eval/sensing_baseline.py",
    "src/kd_sensing/engine/data_factory.py",
    "src/kd_sensing/data/mmw/trajectory_protocol.py",
    "src/kd_sensing/utils/checkpoint.py",
)


class PreflightError(RuntimeError):
    """Raised when the frozen validation panel is not seal-ready."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lines(values: Sequence[str], *, sort: bool = False) -> str:
    items = list(values)
    if sort:
        items.sort()
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def _path(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _assert_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise PreflightError(f"{label} is missing: {path}")


def _assert_not_test_resource(path: Path, label: str) -> None:
    """Guard explicit test split paths, without treating ``tests/`` as data.

    This is only a last-resort path guard.  The primary guarantee is that the
    preflight never calls a dataset/protocol loader and that every protocol
    record is still sealed (``test_evaluated=false``).
    """

    normalized = path.as_posix().lower()
    explicit_test_tokens = (
        "/test/",
        "/test_split/",
        "/test_windows/",
        "/test.csv",
        "__test.csv",
        "/test_",
    )
    if any(token in normalized for token in explicit_test_tokens):
        raise PreflightError(f"Refusing to open a test resource during preflight ({label}): {path}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _assert_not_test_resource(path, label)
    _assert_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{label} must contain a JSON object: {path}")
    return value


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    _assert_not_test_resource(path, label)
    _assert_file(path, label)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PreflightError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{label} must contain a YAML mapping: {path}")
    return value


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise PreflightError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _protocol_view(protocol: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(protocol, Mapping):
        raise PreflightError(f"{label} is missing data_protocol mapping")
    missing = [key for key in PROTOCOL_FIELDS if key not in protocol]
    if missing:
        raise PreflightError(f"{label} is missing immutable protocol fields: {missing}")
    view = {key: protocol[key] for key in PROTOCOL_FIELDS}
    _require_equal(f"{label}.protocol_id", view["protocol_id"], PROTOCOL_ID)
    _require_equal(f"{label}.test_evaluated", view["test_evaluated"], False)
    _require_equal(f"{label}.test_role", view["test_role"], "test")
    _require_equal(f"{label}.train_role", view["train_role"], "train")
    _require_equal(f"{label}.validation_role", view["validation_role"], "validation")
    return view


def _config_protocol(cfg: Mapping[str, Any], label: str) -> dict[str, Any]:
    data = cfg.get("data_protocol")
    return _protocol_view(data, f"{label}.config")


def _checkpoint_protocol(sidecar: Mapping[str, Any], label: str) -> dict[str, Any]:
    return _protocol_view(sidecar.get("data_protocol"), f"{label}.checkpoint")


def _validate_sealed_runtime(cfg: Mapping[str, Any], label: str) -> None:
    training = cfg.get("training") if isinstance(cfg.get("training"), Mapping) else {}
    final_test = training.get("final_test", {})
    enabled = final_test if isinstance(final_test, bool) else bool(final_test.get("enabled", False))
    _require_equal(f"{label}.training.final_test.enabled", enabled, False)
    runtime = cfg.get("runtime") if isinstance(cfg.get("runtime"), Mapping) else {}
    _require_equal(f"{label}.runtime.evaluate_test_requested", bool(runtime.get("evaluate_test_requested", False)), False)
    dataset = cfg.get("data") if isinstance(cfg.get("data"), Mapping) else {}
    dataset_protocol = dataset.get("split_protocol")
    if dataset_protocol is not None:
        _require_equal(f"{label}.data.split_protocol", dataset_protocol, PROTOCOL_ID)


def _extract_topology(value: Mapping[str, Any], label: str, *, required: bool = True) -> dict[str, str]:
    """Extract topology identity from config/report/checkpoint-shaped mappings."""

    candidates: list[Mapping[str, Any]] = []
    for key in ("prototype_topology", "topology"):
        item = value.get(key)
        if isinstance(item, Mapping):
            candidates.append(item)
    model = value.get("model")
    if isinstance(model, Mapping):
        primary = model.get("primary")
        if isinstance(primary, Mapping):
            candidates.append(primary)
            item = primary.get("prototype_topology")
            if isinstance(item, Mapping):
                candidates.append(item)
    metadata = value.get("model_metadata")
    if isinstance(metadata, Mapping):
        item = metadata.get("prototype_topology")
        if isinstance(item, Mapping):
            candidates.append(item)
    loss = value.get("loss")
    if isinstance(loss, Mapping):
        item = loss.get("four_modal_topology")
        if isinstance(item, Mapping):
            item = item.get("prototype_topology")
            if isinstance(item, Mapping):
                candidates.append(item)
    provenance = value.get("provenance")
    if isinstance(provenance, Mapping):
        item = provenance.get("prototype_topology")
        if isinstance(item, Mapping):
            candidates.append(item)
    # Baseline matrix reports expose the binding under `provenance`; the
    # report itself is the authoritative topology source for that adapter.
    report_provenance = value.get("provenance")
    if isinstance(report_provenance, Mapping):
        item = report_provenance.get("prototype_topology")
        if isinstance(item, Mapping):
            candidates.append(item)
    result: dict[str, str] = {}
    for item in candidates:
        if item.get("id") is not None:
            result["id"] = str(item["id"])
        if item.get("descriptor_sha256") is not None:
            result["descriptor_sha256"] = str(item["descriptor_sha256"])
        if item.get("audit_sha256") is not None:
            result["audit_sha256"] = str(item["audit_sha256"])
        # Resolved configs flatten topology fields under model.primary.
        for source, target in (
            ("prototype_topology_id", "id"),
            ("prototype_topology_descriptor_sha256", "descriptor_sha256"),
            ("prototype_topology_audit_sha256", "audit_sha256"),
        ):
            if item.get(source) is not None:
                result[target] = str(item[source])
    if not result and not required:
        return {}
    if result.get("id") != TOPOLOGY_ID:
        raise PreflightError(f"{label} does not bind topology {TOPOLOGY_ID!r}: {result}")
    for key in ("descriptor_sha256", "audit_sha256"):
        value = result.get(key, "")
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
            raise PreflightError(f"{label} has invalid topology {key}: {value!r}")
    return result


def _normalization_record(sidecar: Mapping[str, Any], root: Path, label: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    normalization = sidecar.get("normalization_artifacts")
    if not isinstance(normalization, Mapping):
        raise PreflightError(f"{label} checkpoint has no normalization_artifacts")
    metadata = normalization.get("metadata")
    if not isinstance(metadata, Mapping):
        raise PreflightError(f"{label} checkpoint normalization metadata is missing")
    for key in TRAIN_ONLY_NORMALIZATION_FIELDS:
        if key not in metadata:
            raise PreflightError(f"{label} normalization metadata missing {key}")
    _require_equal(f"{label}.normalization.fit_split", metadata["fit_split"], "train")
    _require_equal(f"{label}.normalization.source_split", metadata["source_split"], "train")
    _require_equal(f"{label}.normalization.sample_id_hash", metadata["sample_id_hash"], protocol["train_sample_id_hash"])
    for key in ("protocol_fingerprint", "protocol_version", "split_seed", "block_size", "split_manifest_hash", "data_source_hash", "window_config_hash", "weather_binding"):
        _require_equal(f"{label}.normalization.{key}", metadata[key], protocol[key] if key != "protocol_version" else protocol[key])
    _require_equal(f"{label}.normalization_modalities", metadata.get("normalization_modalities"), ["gps"])
    _require_equal(f"{label}.gps_feature_mode", metadata.get("gps_feature_mode"), "relative_polar")
    scaler = normalization.get("gps_scaler")
    sidecar_path_value = normalization.get("metadata_sidecar")
    if not scaler or not sidecar_path_value:
        raise PreflightError(f"{label} normalization artifact paths are incomplete")
    scaler_path = _path(root, str(scaler))
    scaler_sidecar_path = _path(root, str(sidecar_path_value))
    _assert_not_test_resource(scaler_path, f"{label} GPS scaler")
    _assert_not_test_resource(scaler_sidecar_path, f"{label} GPS scaler sidecar")
    scaler_sidecar = _read_json(scaler_sidecar_path, f"{label} GPS scaler sidecar")
    _require_equal(f"{label}.normalization_sidecar.fit_split", scaler_sidecar.get("fit_split"), "train")
    _require_equal(f"{label}.normalization_sidecar.sample_id_hash", scaler_sidecar.get("sample_id_hash"), protocol["train_sample_id_hash"])
    _require_equal(f"{label}.normalization_sidecar.artifact", _path(root, str(scaler_sidecar.get("artifact", ""))), scaler_path)
    artifact_sha = scaler_sidecar.get("artifact_sha256")
    if not isinstance(artifact_sha, str) or len(artifact_sha) != 64:
        raise PreflightError(f"{label} normalization sidecar has no artifact SHA256")
    _assert_file(scaler_path, f"{label} GPS scaler artifact")
    actual_sha = _sha256_file(scaler_path)
    _require_equal(f"{label}.normalization_artifact_sha256", actual_sha, artifact_sha)
    return {
        "metadata": {key: metadata[key] for key in TRAIN_ONLY_NORMALIZATION_FIELDS},
        "metadata_sidecar": str(scaler_sidecar_path),
        "metadata_sidecar_sha256": _sha256_file(scaler_sidecar_path),
        "artifact": str(scaler_path),
        "artifact_sha256": actual_sha,
        "effective_sample_count": metadata.get("effective_sample_count"),
        "normalization_fingerprint": metadata.get("normalization_fingerprint"),
    }


def _load_evidence(path: Path, label: str) -> dict[str, Any]:
    _assert_not_test_resource(path, label)
    _assert_file(path, label)
    try:
        records = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - exact torch exception differs by version
        raise PreflightError(f"Cannot read validation evidence {path}: {exc}") from exc
    if not isinstance(records, dict):
        raise PreflightError(f"Validation evidence must be a mapping: {path}")
    return records


def _validate_evidence(path: Path, report: Mapping[str, Any], label: str, protocol: Mapping[str, Any], checkpoint_sha: str) -> dict[str, Any]:
    records = _load_evidence(path, label)
    patterns = records.get("pattern")
    sample_ids = records.get("sample_id")
    labels = records.get("labels")
    available = records.get("available")
    if not isinstance(patterns, list) or not isinstance(sample_ids, list):
        raise PreflightError(f"{label} evidence has no pattern/sample_id rows")
    if not isinstance(labels, torch.Tensor) or not isinstance(available, torch.Tensor):
        raise PreflightError(f"{label} evidence labels/available are not tensors")
    expected_count = int(protocol["validation_sample_count"])
    row_count = expected_count * len(EXPECTED_PATTERN_NAMES)
    if len(patterns) != row_count or len(sample_ids) != row_count or int(labels.numel()) != row_count:
        raise PreflightError(f"{label} evidence row count must be {row_count}")
    if tuple(available.shape) != (row_count, 4):
        raise PreflightError(f"{label} evidence available shape must be {(row_count, 4)}, got {tuple(available.shape)}")
    ids_by_pattern: dict[str, tuple[str, ...]] = {}
    masks_by_pattern: dict[str, tuple[int, ...]] = {}
    for name in EXPECTED_PATTERN_NAMES:
        indices = [index for index, value in enumerate(patterns) if str(value) == name]
        if len(indices) != expected_count:
            raise PreflightError(f"{label} pattern {name} has {len(indices)} rows, expected {expected_count}")
        ids = tuple(str(sample_ids[index]) for index in indices)
        if len(set(ids)) != expected_count:
            raise PreflightError(f"{label} pattern {name} has duplicate validation sample IDs")
        if any(":test:" in value or ":test/" in value for value in ids):
            raise PreflightError(f"{label} evidence contains test sample identity")
        if any(":validation:" not in value for value in ids):
            raise PreflightError(f"{label} evidence contains a non-validation sample identity")
        masks = {tuple(int(value) for value in available[index].tolist()) for index in indices}
        if masks != {EXPECTED_PATTERN_MASKS[name]}:
            raise PreflightError(f"{label} pattern {name} availability mask mismatch: {masks}")
        ids_by_pattern[name] = ids
        masks_by_pattern[name] = next(iter(masks))
    reference_ids = ids_by_pattern[EXPECTED_PATTERN_NAMES[0]]
    reference_set = set(reference_ids)
    for name, ids in ids_by_pattern.items():
        if set(ids) != reference_set:
            raise PreflightError(f"{label} pattern {name} does not cover the same validation identity set")
    if report.get("outer_test_accessed") is not False or report.get("claim_ineligible") is not True:
        raise PreflightError(f"{label} matrix report is not sealed validation-only evidence")
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PreflightError(f"{label} matrix report has no provenance")
    checkpoint = provenance.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("sha256") != checkpoint_sha:
        raise PreflightError(f"{label} matrix report checkpoint SHA does not match checkpoint sidecar")
    evidence_meta = provenance.get("sample_evidence")
    if not isinstance(evidence_meta, Mapping):
        raise PreflightError(f"{label} matrix report has no sample-evidence provenance")
    actual_sha = _sha256_file(path)
    if evidence_meta.get("sha256") != actual_sha:
        raise PreflightError(f"{label} evidence SHA mismatch in matrix report")
    if _path(path.parent.parent.parent, str(evidence_meta.get("path", ""))) != path:
        # Provenance paths are absolute in production.  For a relative test
        # fixture, accept a path relative to the repository root below.
        if Path(str(evidence_meta.get("path", ""))).resolve() != path:
            raise PreflightError(f"{label} evidence provenance path does not match evidence file")
    report_protocol = provenance.get("data_protocol")
    if isinstance(report_protocol, Mapping):
        _protocol_view(report_protocol, f"{label}.matrix_report")
    return {
        "path": str(path),
        "sha256": actual_sha,
        "rows": row_count,
        "validation_sample_count": expected_count,
        "pattern_count": len(EXPECTED_PATTERN_NAMES),
        "patterns": list(EXPECTED_PATTERN_NAMES),
        "validation_sample_id_hash": _sha256_lines(reference_ids, sort=True),
        "validation_sample_order_sha256": _sha256_lines(reference_ids),
        "domain_count": len(set(str(value) for value in records.get("domain", []))),
        "trajectory_group_count": len(set(str(value) for value in records.get("group_id", []))),
        "model_type": records.get("model_type"),
        "available_masks": {name: list(mask) for name, mask in masks_by_pattern.items()},
    }


def _strategy_snapshot() -> dict[str, Any]:
    # Importing these modules is safe: this path contains no data-loader or
    # protocol binding calls.  The explicit assertions make a changed policy
    # fail closed instead of silently changing the sealed experiment.
    from kd_sensing.eval.beam_probe_diagnostic import (
        BATCH_TBCP_METHODS,
        COVARIANCE_MODE_FULL,
        METHOD_FEEDBACK_UPDATES,
        PRIMARY_BATCH_METHOD,
        PRIMARY_POSTERIOR_METHOD,
        PRIMARY_TBCP_METHOD,
    )
    from kd_sensing.eval.beam_topology_likelihood import BATCH_TBCP_SCHEDULES, TBCP_BUDGET, TBCP_POLICY_VERSION

    expected_schedule = (2, 1)
    _require_equal("TBCP budget", int(TBCP_BUDGET), 3)
    _require_equal("TBCP covariance", str(COVARIANCE_MODE_FULL), "full")
    _require_equal("TBCP primary method", str(PRIMARY_TBCP_METHOD), "TBCP-3")
    _require_equal("Posterior baseline", str(PRIMARY_POSTERIOR_METHOD), "Posterior Top-3")
    _require_equal("Batch primary method", str(PRIMARY_BATCH_METHOD), "Batch-TBCP-2+1")
    _require_equal("TBCP policy version", str(TBCP_POLICY_VERSION), "topology_bayesian_closed_loop_gain_v1")
    _require_equal("Batch schedule", tuple(BATCH_TBCP_SCHEDULES[0]), expected_schedule)
    _require_equal("Batch schedule registry", tuple(BATCH_TBCP_METHODS[PRIMARY_BATCH_METHOD]), expected_schedule)
    _require_equal("TBCP feedback updates", int(METHOD_FEEDBACK_UPDATES[PRIMARY_TBCP_METHOD]), 1)
    return {
        "budget": int(TBCP_BUDGET),
        "covariance_mode": str(COVARIANCE_MODE_FULL),
        "primary_method": str(PRIMARY_TBCP_METHOD),
        "posterior_baseline": str(PRIMARY_POSTERIOR_METHOD),
        "policy_version": str(TBCP_POLICY_VERSION),
        "batch_method": str(PRIMARY_BATCH_METHOD),
        "batch_schedule": list(expected_schedule),
        "measurement_rounds": 2,
        "feedback_updates": 1,
        "full_sweep_beams": 64,
        "primary_probe_slots": 3,
        "missing_pattern_count": len(EXPECTED_PATTERN_NAMES),
    }


def _likelihood_record(root: Path, path: Path, protocol: Mapping[str, Any], topology: Mapping[str, str]) -> dict[str, Any]:
    sidecar_path = Path(str(path) + ".json")
    sidecar = _read_json(sidecar_path, "train-only topology likelihood sidecar")
    metadata = sidecar.get("metadata")
    if not isinstance(metadata, Mapping):
        raise PreflightError("Train-only topology likelihood sidecar has no metadata")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PreflightError("Train-only topology likelihood has no provenance")
    _require_equal("likelihood.fit_split", provenance.get("fit_split"), "train")
    _require_equal("likelihood.source_split", provenance.get("source_split"), "train")
    _require_equal("likelihood.test_evaluated", provenance.get("test_evaluated"), False)
    _require_equal("likelihood.outer_test_accessed", provenance.get("outer_test_accessed"), False)
    for key in ("protocol_id", "protocol_version", "protocol_fingerprint", "split_manifest_hash", "data_source_hash", "window_config_hash", "split_seed", "block_size", "weather_binding", "train_sample_count", "train_sample_id_hash"):
        _require_equal(f"likelihood.provenance.{key}", provenance.get(key), protocol[key] if key != "train_sample_count" else protocol[key])
    for key, expected in (("topology_id", topology["id"]), ("topology_descriptor_sha256", topology["descriptor_sha256"]), ("topology_audit_sha256", topology["audit_sha256"])):
        _require_equal(f"likelihood.provenance.{key}", provenance.get(key), expected)
    _require_equal("likelihood.artifact_type", metadata.get("artifact_type"), "train_only_ula_dft_relative_gain_likelihood")
    _require_equal("likelihood.num_beams", int(metadata.get("num_beams", -1)), 64)
    calibration = metadata.get("calibration_config")
    if not isinstance(calibration, Mapping) or calibration.get("fitted_measurement_noise") is not False:
        raise PreflightError("Likelihood calibration is not the frozen train-only/no-measurement-noise artifact")
    _assert_not_test_resource(path, "train-only topology likelihood")
    _assert_file(path, "train-only topology likelihood")
    actual_sha = _sha256_file(path)
    _require_equal("likelihood.artifact_sha256", sidecar.get("artifact_sha256"), actual_sha)
    return {
        "artifact": str(path),
        "artifact_sha256": actual_sha,
        "sidecar": str(sidecar_path),
        "sidecar_sha256": _sha256_file(sidecar_path),
        "artifact_fingerprint": metadata.get("artifact_fingerprint"),
        "fit_split": "train",
        "train_sample_count": int(provenance["train_sample_count"]),
        "train_sample_id_hash": provenance["train_sample_id_hash"],
        "topology": dict(topology),
    }


def _candidate_preflight(root: Path, spec: CandidateSpec, expected_protocol: Mapping[str, Any], likelihood_topology: Mapping[str, str]) -> dict[str, Any]:
    label = f"{spec.method}/seed{spec.seed}"
    config_path = _path(root, spec.config)
    checkpoint_path = _path(root, spec.checkpoint)
    checkpoint_sidecar_path = _path(root, spec.checkpoint_sidecar)
    matrix_path = _path(root, spec.matrix_report)
    evidence_path = _path(root, spec.evidence)
    for path, role in ((config_path, "config"), (checkpoint_path, "checkpoint"), (checkpoint_sidecar_path, "checkpoint sidecar"), (matrix_path, "matrix report"), (evidence_path, "validation evidence")):
        _assert_not_test_resource(path, f"{label} {role}")
        _assert_file(path, f"{label} {role}")
    cfg = _read_yaml(config_path, f"{label} config")
    sidecar = _read_json(checkpoint_sidecar_path, f"{label} checkpoint sidecar")
    try:
        payload = load_torch_payload(checkpoint_path, map_location="cpu")
        publication = validate_checkpoint_publication(checkpoint_path, payload=payload)
    except Exception as exc:
        raise PreflightError(f"{label} checkpoint publication validation failed: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PreflightError(f"{label} checkpoint payload is not a mapping")
    _require_equal(f"{label}.payload.checkpoint_role", payload.get("checkpoint_role"), "validation_best")
    _require_equal(f"{label}.payload.experiment_seed", int(payload.get("experiment_seed", -1)), int(spec.seed))
    _require_equal(f"{label}.publication.integrity_verified", publication.get("integrity_verified"), True)
    _require_equal(f"{label}.checkpoint_role", sidecar.get("checkpoint_role"), "validation_best")
    _require_equal(f"{label}.checkpoint_policy", sidecar.get("checkpoint_policy"), "best_validation_loss")
    _require_equal(f"{label}.publish_complete", sidecar.get("publish_complete"), True)
    _require_equal(f"{label}.experiment_seed", int(sidecar.get("experiment_seed", -1)), int(spec.seed))
    if sidecar.get("path"):
        _require_equal(f"{label}.checkpoint_path", _path(root, str(sidecar["path"])), checkpoint_path)
    checkpoint_sha = _sha256_file(checkpoint_path)
    _require_equal(f"{label}.checkpoint_sha256", sidecar.get("checkpoint_sha256"), checkpoint_sha)
    cfg_protocol = _config_protocol(cfg, label)
    checkpoint_protocol = _checkpoint_protocol(sidecar, label)
    payload_protocol = _checkpoint_protocol(payload, label)
    if checkpoint_protocol != payload_protocol:
        raise PreflightError(f"{label} sidecar/payload immutable protocol differs")
    if cfg_protocol != checkpoint_protocol:
        raise PreflightError(f"{label} config/checkpoint immutable protocol differs")
    if expected_protocol and checkpoint_protocol != expected_protocol:
        raise PreflightError(f"{label} protocol differs from the frozen panel protocol")
    _require_equal(f"{label}.config experiment.seed", int(cfg.get("experiment", {}).get("seed", -1)), int(spec.seed))
    _validate_sealed_runtime(cfg, label)
    enabled = cfg.get("model", {}).get("primary", {}).get("modalities")
    _require_equal(f"{label}.modalities", tuple(enabled or ()), EXPECTED_MODALITIES)
    # Baseline recipes intentionally do not carry the native topology resolver;
    # their matrix provenance is the binding source.  Native topology configs
    # must still carry the identity themselves.
    try:
        topology = _extract_topology(cfg, f"{label}.config")
    except PreflightError:
        topology = {}
    sidecar_topology = _extract_topology(sidecar, f"{label}.checkpoint", required=False)
    if sidecar_topology and sidecar_topology != topology:
        raise PreflightError(f"{label} checkpoint topology differs from config")
    report = _read_json(matrix_path, f"{label} validation matrix report")
    report_topology = _extract_topology(report, f"{label}.matrix_report")
    if not topology:
        topology = report_topology
    if report_topology != topology:
        raise PreflightError(f"{label} matrix topology differs from config")
    if topology != likelihood_topology:
        raise PreflightError(f"{label} topology differs from train-only likelihood")
    payload_metadata = payload.get("model_metadata")
    if not isinstance(payload_metadata, Mapping):
        # The retained sensing baselines predate explicit model metadata.  The
        # resume config is still part of the published payload and is checked
        # for the same four sensing streams and no history-beam input.
        resume = payload.get("resume_contract")
        recorded_cfg = resume.get("config") if isinstance(resume, Mapping) else None
        payload_metadata = recorded_cfg.get("model", {}).get("primary", {}) if isinstance(recorded_cfg, Mapping) else {}
    if not isinstance(payload_metadata, Mapping):
        raise PreflightError(f"{label} checkpoint has no model metadata/config")
    payload_modalities = payload_metadata.get("modalities")
    _require_equal(f"{label}.payload.modalities", tuple(payload_modalities or ()), EXPECTED_MODALITIES)
    payload_type = payload_metadata.get("type")
    expected_type = "four_modal_topology_predictor" if spec.method in {"Prototype-only", "Hard"} else "modular_sequence"
    _require_equal(f"{label}.payload.model_type", payload_type, expected_type)
    forbidden_fields = ("history_beam", "history_beam_index", "current_beam", "previous_beam", "beam_history")
    for field in forbidden_fields:
        if bool(payload_metadata.get(field, False)):
            raise PreflightError(f"{label} payload declares forbidden history-beam field {field!r}")
    payload_topology = _extract_topology(payload, f"{label}.payload", required=False)
    if payload_topology and payload_topology != topology:
        raise PreflightError(f"{label} payload topology differs from config/report")
    normalization = _normalization_record(sidecar, root, label, checkpoint_protocol)
    evidence = _validate_evidence(evidence_path, report, label, checkpoint_protocol, checkpoint_sha)
    if report.get("provenance", {}).get("sample_evidence", {}).get("path"):
        evidence_meta_path = Path(str(report["provenance"]["sample_evidence"]["path"])).resolve()
        if evidence_meta_path != evidence_path:
            raise PreflightError(f"{label} matrix evidence path mismatch")
    return {
        "method": spec.method,
        "family": spec.family,
        "seed": int(spec.seed),
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha, "role": "validation_best", "policy": "best_validation_loss"},
        "checkpoint_sidecar": {"path": str(checkpoint_sidecar_path), "sha256": _sha256_file(checkpoint_sidecar_path)},
        "matrix_report": {"path": str(matrix_path), "sha256": _sha256_file(matrix_path)},
        "normalization": normalization,
        "topology": topology,
        "validation_evidence": evidence,
    }


def run_preflight(
    *,
    repo_root: str | Path | None = None,
    output: str | Path | None = None,
    likelihood: str | Path | None = None,
    candidates: Sequence[CandidateSpec] | None = None,
) -> dict[str, Any]:
    """Validate the frozen validation panel and write a non-overwritable seal.

    No dataset or split CSV is opened by this function.  The only tensor file
    loaded is the already materialized validation 15-mask evidence cache.
    """

    root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    output_path = _path(root, output or "outputs/final_test_seal/mmw_final_test_seal_v1.json")
    if output_path.exists():
        raise PreflightError(f"Refusing to overwrite an existing seal manifest: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    specs = tuple(candidates or _candidate_specs())
    if len(specs) != 12:
        raise PreflightError(f"Final MMW panel requires 12 candidates, got {len(specs)}")
    methods = [(spec.method, int(spec.seed)) for spec in specs]
    if len(set(methods)) != 12:
        raise PreflightError("Final MMW panel candidate method/seed identities are not unique")
    strategy = _strategy_snapshot()

    # Establish the immutable protocol/topology from the first checkpoint,
    # then require every candidate to match.  This reads only sidecar JSON and
    # never opens the manifest or any split CSV.
    first_sidecar_path = _path(root, specs[0].checkpoint_sidecar)
    first_sidecar = _read_json(first_sidecar_path, "first checkpoint sidecar")
    expected_protocol = _checkpoint_protocol(first_sidecar, f"{specs[0].method}/seed{specs[0].seed}")
    first_cfg = _read_yaml(_path(root, specs[0].config), "first candidate config")
    topology = _extract_topology(first_cfg, "first candidate config")
    likelihood_path = _path(root, likelihood or DEFAULT_LIKELIHOOD)
    likelihood_record = _likelihood_record(root, likelihood_path, expected_protocol, topology)

    candidate_records = [
        _candidate_preflight(root, spec, expected_protocol, topology)
        for spec in specs
    ]
    validation_hashes = {
        record["validation_evidence"]["validation_sample_order_sha256"] for record in candidate_records
    }
    if len(validation_hashes) != 1:
        raise PreflightError("Candidates do not share one validation sample identity/order")
    protocol_hashes = {record["validation_evidence"]["validation_sample_id_hash"] for record in candidate_records}
    if len(protocol_hashes) != 1:
        raise PreflightError("Candidates do not share one validation sample identity set")

    source_hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        source_path = _path(root, relative)
        _assert_not_test_resource(source_path, "source file")
        _assert_file(source_path, "source file")
        source_hashes[relative] = _sha256_file(source_path)

    manifest = {
        "schema_version": 1,
        "seal_type": "mmw_final_test_panel_preflight_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "preflight_only": True,
        "test_unlock_authorized": False,
        "test_accessed": False,
        "outer_test_accessed": False,
        "test_loader_constructed": False,
        "test_csv_read": False,
        "test_label_read": False,
        "test_power_read": False,
        "candidate_count": len(candidate_records),
        "candidate_identities": [{"method": spec.method, "seed": int(spec.seed)} for spec in specs],
        "protocol": expected_protocol,
        "topology": topology,
        "strategy": strategy,
        "train_only_likelihood": likelihood_record,
        "validation_panel": {
            "patterns": list(EXPECTED_PATTERN_NAMES),
            "pattern_count": len(EXPECTED_PATTERN_NAMES),
            "sample_count": int(expected_protocol["validation_sample_count"]),
            "sample_rows_per_candidate": int(expected_protocol["validation_sample_count"]) * len(EXPECTED_PATTERN_NAMES),
            "sample_identity_set_sha256": next(iter(protocol_hashes)),
            "sample_identity_order_sha256": next(iter(validation_hashes)),
            "candidate_count": len(candidate_records),
        },
        "candidates": candidate_records,
        "source_files_sha256": source_hashes,
        "preflight_checks": {
            "candidate_checkpoints_validation_best": True,
            "candidate_configs_test_disabled": True,
            "immutable_protocol_match": True,
            "topology_match": True,
            "train_only_normalization": True,
            "train_only_likelihood": True,
            "validation_15_mask_evidence": True,
            "strategy_constants_frozen": True,
            "test_artifacts_opened": False,
        },
    }
    # Write once.  The existence guard above and exclusive create prevent a
    # concurrent second preflight from replacing the first seal.
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError as exc:  # pragma: no cover - race protection
        raise PreflightError(f"Refusing to overwrite an existing seal manifest: {output_path}") from exc
    manifest["seal_manifest"] = {"path": str(output_path), "sha256": _sha256_file(output_path)}
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight the frozen MMW final-test panel without opening test.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    preflight = subparsers.add_parser("preflight", help="Validate validation artifacts and write an immutable seal manifest.")
    preflight.add_argument("--repo-root", default=None)
    preflight.add_argument("--output", default=None)
    preflight.add_argument("--likelihood", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.action != "preflight":
        raise PreflightError(f"Unsupported action: {args.action}")
    try:
        result = run_preflight(repo_root=args.repo_root, output=args.output, likelihood=args.likelihood)
    except PreflightError as exc:
        print(f"mmw_final_test_panel preflight failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"seal_manifest": result["seal_manifest"], "candidate_count": result["candidate_count"], "test_accessed": False}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
