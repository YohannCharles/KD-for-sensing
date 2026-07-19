#!/usr/bin/env python3
"""Summarize the fixed-checkpoint MMW Router joint Drop+Corrupt screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from kd_sensing.data.mmw.twc_router_joint_stress import (
    BALANCE_POLICY,
    CORRUPTION_SEVERITY,
    GENERATOR,
    JOINT_RATES,
    MASKS_PER_RATE,
    MASK_SEED,
    PROTOCOL_ID,
    load_router_joint_stress_cache,
)
from kd_sensing.evaluation.corruptions import CORRUPTION_PARAMETERS

from launch_mmw_router_joint_stress import BRANCH_ALGORITHM, CORRUPTION_SEED, EVALUATOR_ALGORITHM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs/mmw_router_joint_stress_v1"
DEFAULT_CACHE = ROOT / "outputs/cache/mmw_router_joint_stress_v1/fixed_state_cache.json"
FUSIONS = ("uniform", "learned", "oracle")
DYNAMIC_FUSIONS = (
    *FUSIONS,
    "train_fit_static_prior",
    "frozen_current_router",
    "post_health_uniform",
    "post_health_static_prior",
)
METRICS = (
    "adba",
    "top1",
    "normalized_gain",
    "spectral_efficiency_ratio_0db",
    "spectral_efficiency_ratio_10db",
    "spectral_efficiency_ratio_20db",
)
GATE_RATES = (0.4, 0.6, 0.8)
GATE_METRICS = ("adba", "normalized_gain")
REGRET_FIELDS = ("router_soft_oracle_regret", "router_selection_oracle_regret")
ROUTER_WEIGHT_FIELDS = tuple(f"router_weight_{name}" for name in ("image", "radar", "gps", "lidar"))
DYNAMIC_DIAGNOSTIC_FIELDS = (
    "router_residual_abs_mean",
    "corrupted_cell_weight_mass",
    "corrupted_cell_available_share",
    "corrupted_cell_weight_response_ratio",
    "corrupted_cell_static_prior_mass",
    "corrupted_cell_weight_vs_static_ratio",
    "corrupted_cell_downweight_vs_static_rate",
)
CLEAN_ADBA_NONINFERIORITY_MARGIN = 0.002
COMBINED_MIN_DELTA = {"adba": 0.002, "normalized_gain": 0.005}
MIN_CORRUPTED_CELL_DOWNWEIGHT_RATE = 0.60
MAX_CORRUPTED_CELL_WEIGHT_VS_STATIC_RATIO = 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    if int(args.bootstrap_iterations) <= 0:
        parser.error("--bootstrap-iterations must be positive.")
    if not 0.0 < float(args.confidence) < 1.0:
        parser.error("--confidence must be in (0, 1).")
    try:
        summarize(
            Path(args.root).resolve(),
            Path(args.cache).resolve(),
            bootstrap_iterations=int(args.bootstrap_iterations),
            bootstrap_seed=int(args.bootstrap_seed),
            confidence=float(args.confidence),
        )
    except Exception as exc:  # noqa: BLE001 - CLI must fail closed with context.
        parser.error(f"{type(exc).__name__}: {exc}")
    return 0


def summarize(
    root: Path,
    cache_path: Path,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any]:
    cache = load_router_joint_stress_cache(cache_path)
    manifest = _validate_manifest(root, cache_path, cache)
    conditions = {str(item["pattern"]): dict(item) for item in cache["conditions"]}
    if len(conditions) != 1 + len(JOINT_RATES) * MASKS_PER_RATE:
        raise ValueError("Joint-stress cache does not contain exactly 81 unique conditions.")

    fusions = _resolve_fusions(root, conditions, manifest)
    raw_rows = _load_condition_rows(root, conditions, manifest, fusions=fusions)
    condition_rows = _condition_summary(raw_rows, conditions, fusions=fusions)
    domain_rate_rows = _domain_rate_summary(raw_rows, conditions, fusions=fusions)
    rate_rows = _rate_summary(domain_rate_rows, condition_rows, fusions=fusions)
    bootstrap_rows = _bootstrap_rows(
        domain_rate_rows,
        controls=_bootstrap_controls(fusions),
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    gate = _gate_decision(rate_rows, bootstrap_rows, fusions=fusions)
    provenance = {
        "protocol": PROTOCOL_ID,
        "claim_eligible": False,
        "split": "frozen_inner_validation_only",
        "condition_count": len(conditions),
        "domain_count": 15,
        "fusion_branches": list(fusions),
        "cache": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "cache_checksum": str(cache["checksum"]),
        "evaluation_manifest": str((root / "evaluation_manifest.json").resolve()),
        "evaluation_request_sha256": str(manifest["request_sha256"]),
        "checkpoint": str(manifest["request"]["checkpoint"]),
        "checkpoint_sha256": str(manifest["request"]["checkpoint_sha256"]),
        "corruption_seed": int(manifest["request"]["corruption_seed"]),
        "corruption_severity": int(manifest["request"]["corruption_severity"]),
        "corruption_parameters": manifest["request"]["corruption_parameters"],
        "evaluator_algorithm": str(manifest["request"]["evaluator_algorithm"]),
        "source_sha256": {
            key: str(manifest["request"][key])
            for key in (
                "evaluator_sha256",
                "oracle_helper_sha256",
                "corruption_runtime_sha256",
                "joint_cache_runtime_sha256",
                "summary_sha256",
            )
        },
        "branch_algorithm": str(manifest["request"]["branch_algorithm"]),
        "router_regret_scope": (
            "learned_dynamic_branch_only" if fusions == DYNAMIC_FUSIONS else "learned_router_branch_only"
        ),
        "reference_control_decomposition": {
            "closure_field_prefix": "learned_reference_control_closure_",
            "modality_residual_contrast_prefix": "learned_minus_post_health_static_prior_",
            "uniform_and_oracle_experts": "frozen_reference_masked_mean_experts",
            "learned_experts": "candidate_post_health_experts",
            "interpretation": "diagnostic_mixed_expert_decomposition_not_pure_router_oracle_gap",
        },
        "bootstrap": {
            "unit": "paired_domain_after_equal_mask_average",
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
            "confidence": confidence,
        },
        "gate_thresholds": {
            "joint_rate_delta_static": 0.0,
            "combined_min_delta": dict(COMBINED_MIN_DELTA),
            "combined_control_ci_low": 0.0,
            "clean_adba_noninferiority_margin": CLEAN_ADBA_NONINFERIORITY_MARGIN,
            "corrupted_cell_downweight_vs_static_rate": MIN_CORRUPTED_CELL_DOWNWEIGHT_RATE,
            "corrupted_cell_weight_vs_static_ratio_exclusive_max": MAX_CORRUPTED_CELL_WEIGHT_VS_STATIC_RATIO,
        },
    }
    payload = {
        "provenance": provenance,
        "gate": gate,
        "rate_summary": rate_rows,
        "paired_domain_bootstrap": bootstrap_rows,
    }
    _write_csv(root / "joint_condition_summary.csv", condition_rows)
    _write_csv(root / "joint_domain_rate_summary.csv", domain_rate_rows)
    _write_csv(root / "joint_rate_summary.csv", rate_rows)
    _write_csv(root / "joint_paired_domain_bootstrap.csv", bootstrap_rows)
    _write_json(root / "joint_summary.json", payload)
    _write_json(root / "joint_gate_decision.json", gate)
    _plot(root, rate_rows, fusions=fusions)
    _write_text(root / "README.md", _markdown(provenance, rate_rows, bootstrap_rows, gate))
    return payload


def _validate_manifest(root: Path, cache_path: Path, cache: Mapping[str, Any]) -> dict[str, Any]:
    path = root / "evaluation_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing joint-stress evaluation manifest: {path}")
    manifest = _read_json(path)
    request = manifest.get("request", {})
    if (
        manifest.get("status") != "complete"
        or request.get("protocol") != PROTOCOL_ID
        or manifest.get("request_sha256") != _payload_sha256(request)
    ):
        raise ValueError("Joint-stress manifest is not complete and protocol-matched.")
    if str(request.get("cache_checksum")) != str(cache["checksum"]):
        raise ValueError("Joint-stress cache checksum differs from the evaluation manifest.")
    if str(request.get("cache_sha256")) != _sha256(cache_path):
        raise ValueError("Joint-stress cache file SHA256 differs from the evaluation manifest.")
    if int(request.get("condition_count", -1)) != len(cache["conditions"]):
        raise ValueError("Joint-stress manifest condition count mismatch.")
    expected_recipe = _joint_stress_request_identity()
    if any(request.get(key) != value for key, value in expected_recipe.items()):
        raise ValueError("Joint-stress manifest immutable mask/corruption recipe mismatch.")
    for key, label in (("config", "config"), ("checkpoint", "checkpoint")):
        source = Path(str(request.get(key, ""))).resolve()
        if not source.is_file() or _sha256(source) != str(request.get(f"{key}_sha256")):
            raise ValueError(f"Joint-stress manifest {label} identity mismatch.")
    if (
        request.get("evaluator_algorithm") != EVALUATOR_ALGORITHM
        or request.get("branch_algorithm") != BRANCH_ALGORITHM
        or request.get("split") != "frozen_inner_validation_only"
        or request.get("claim_eligible") is not False
    ):
        raise ValueError("Joint-stress manifest evaluator/branch evidence boundary mismatch.")
    source_paths = {
        "evaluator_sha256": ROOT / "scripts/eval_mmw_router_joint_stress.py",
        "oracle_helper_sha256": ROOT / "scripts/eval_mmw_router_oracle_gap.py",
        "corruption_runtime_sha256": ROOT / "src/kd_sensing/evaluation/corruptions.py",
        "joint_cache_runtime_sha256": ROOT / "src/kd_sensing/data/mmw/twc_router_joint_stress.py",
        "summary_sha256": ROOT / "scripts/summarize_mmw_router_joint_stress.py",
    }
    if any(request.get(key) != _sha256(source) for key, source in source_paths.items()):
        raise ValueError("Joint-stress manifest source-code identity mismatch.")
    jobs = manifest.get("jobs", ())
    if len(jobs) != 8 or any(job.get("status") != "complete" for job in jobs):
        raise ValueError("All eight joint-stress evaluator shards must be complete before summary.")
    return manifest


def _joint_stress_request_identity() -> dict[str, Any]:
    return {
        "mask_seed": MASK_SEED,
        "mask_generator": GENERATOR,
        "mask_balance_policy": BALANCE_POLICY,
        "corruption_seed": CORRUPTION_SEED,
        "corruption_severity": CORRUPTION_SEVERITY,
        "corruption_parameters": {
            name: {
                "unit": CORRUPTION_PARAMETERS[name]["unit"],
                "value": CORRUPTION_PARAMETERS[name]["values"][CORRUPTION_SEVERITY - 1],
            }
            for name in ("image_occlusion", "radar_noise", "lidar_sparsify", "gps_noise")
        },
        "condition_count": 1 + len(JOINT_RATES) * MASKS_PER_RATE,
        "joint_rates": [float(value) for value in JOINT_RATES],
        "masks_per_rate": MASKS_PER_RATE,
        "batch_size": 64,
    }


def _resolve_fusions(
    root: Path,
    conditions: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    first_pattern = next(iter(conditions), None)
    if first_pattern is None:
        raise ValueError("Cannot resolve fusion branches from an empty condition inventory.")
    path = root / first_pattern / "domain_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing joint-stress metrics used to resolve fusion branches: {path}")
    observed = {str(row.get("fusion", "")) for row in _read_csv(path)}
    declared = tuple(manifest.get("request", {}).get("fusion_branches", FUSIONS))
    if declared not in (FUSIONS, DYNAMIC_FUSIONS):
        raise ValueError(f"Unsupported manifest fusion branch inventory: {list(declared)}.")
    if observed == set(declared):
        return declared
    raise ValueError(
        "Joint-stress fusion inventory differs from the immutable manifest: "
        f"declared={list(declared)}, observed={sorted(observed)}."
    )


def _comparison_controls(fusions: tuple[str, ...]) -> tuple[str, ...]:
    return (
        (
            "uniform",
            "train_fit_static_prior",
            "frozen_current_router",
            "post_health_static_prior",
        )
        if fusions == DYNAMIC_FUSIONS
        else ("uniform",)
    )


def _bootstrap_controls(fusions: tuple[str, ...]) -> tuple[str, ...]:
    return (
        ("train_fit_static_prior", "frozen_current_router", "uniform")
        if fusions == DYNAMIC_FUSIONS
        else ("uniform",)
    )


def _load_condition_rows(
    root: Path,
    conditions: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    fusions: tuple[str, ...],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    expected_domains: set[str] | None = None
    request = manifest["request"]
    marker_identity = {
        "protocol": PROTOCOL_ID,
        "cache_checksum": str(request["cache_checksum"]),
        "config_sha256": str(request["config_sha256"]),
        "checkpoint_sha256": str(request["checkpoint_sha256"]),
        "request_sha256": str(manifest["request_sha256"]),
    }
    for pattern, condition in conditions.items():
        directory = root / pattern
        complete_path = directory / "complete.json"
        metrics_path = directory / "domain_metrics.csv"
        if not complete_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"Incomplete joint-stress condition: {pattern}")
        complete = _read_json(complete_path)
        if complete.get("status") != "complete" or bool(complete.get("partial", True)):
            raise ValueError(f"Invalid completion marker for joint-stress condition: {pattern}")
        if any(str(complete.get(key, "")) != value for key, value in marker_identity.items()):
            raise ValueError(f"Completion identity mismatch for joint-stress condition: {pattern}")
        recorded_condition = complete.get("condition", {})
        if (
            complete.get("pattern") != pattern
            or recorded_condition.get("state_digest") != condition["state_digest"]
            or int(recorded_condition.get("condition_index", -1)) != int(condition["condition_index"])
        ):
            raise ValueError(f"Completion condition metadata mismatch for: {pattern}")
        rows = _read_csv(metrics_path)
        expected_rows = 15 * len(fusions)
        if len(rows) != expected_rows:
            raise ValueError(f"{pattern} expected {expected_rows} domain/fusion rows, got {len(rows)}.")
        seen: set[tuple[str, str]] = set()
        domains: set[str] = set()
        for source in rows:
            fusion = str(source.get("fusion", ""))
            domain_id = str(source.get("domain_id", ""))
            if fusion not in fusions or not domain_id:
                raise ValueError(f"Invalid domain/fusion identity in {metrics_path}.")
            if (
                source.get("protocol") != PROTOCOL_ID
                or source.get("cache_checksum") != str(request["cache_checksum"])
                or source.get("condition") != pattern
                or source.get("state_digest") != str(condition["state_digest"])
                or int(source.get("condition_index", -1)) != int(condition["condition_index"])
                or int(source.get("mask_set_index", -1)) != int(condition["mask_set_index"])
                or abs(float(source.get("requested_stress_rate", -1.0)) - float(condition["requested_stress_rate"]))
                > 1.0e-12
            ):
                raise ValueError(f"Metric-row evidence identity mismatch in {metrics_path}.")
            if fusion == "learned":
                expected_regret_scope = (
                    "learned_dynamic_branch_only"
                    if fusions == DYNAMIC_FUSIONS
                    else "learned_router_branch_only"
                )
                if source.get("router_regret_scope") != expected_regret_scope:
                    raise ValueError(f"Router regret scope is ambiguous in {metrics_path}.")
            elif source.get("router_regret_scope") not in {
                "not_applicable_control_branch",
                "",
                None,
            }:
                raise ValueError(f"Control-branch regret scope is invalid in {metrics_path}.")
            identity = (domain_id, fusion)
            if identity in seen:
                raise ValueError(f"Duplicate domain/fusion row in {metrics_path}: {identity}")
            seen.add(identity)
            domains.add(domain_id)
            row: dict[str, Any] = {
                "pattern": pattern,
                "requested_stress_rate": float(condition["requested_stress_rate"]),
                "mask_set_index": int(condition["mask_set_index"]),
                "state_digest": str(condition["state_digest"]),
                "domain_id": domain_id,
                "fusion": fusion,
                "sample_count": int(source["sample_count"]),
            }
            for metric in METRICS:
                row[metric] = _finite_float(source, metric, metrics_path)
            for field in REGRET_FIELDS:
                if fusion == "learned":
                    row[field] = _finite_float(source, field, metrics_path)
                else:
                    # Control branches deliberately do not carry Router
                    # regret; keep a nullable column for CSV compatibility.
                    raw_value = source.get(field, "")
                    if str(raw_value).strip():
                        raise ValueError(f"Control branch unexpectedly carries {field} in {metrics_path}.")
                    row[field] = None
            for field in ROUTER_WEIGHT_FIELDS:
                row[field] = _finite_float(source, field, metrics_path)
            if fusions == DYNAMIC_FUSIONS:
                for field in DYNAMIC_DIAGNOSTIC_FIELDS:
                    row[field] = _finite_float(source, field, metrics_path)
            result.append(row)
        if len(domains) != 15:
            raise ValueError(f"{pattern} expected 15 domains, got {len(domains)}.")
        if expected_domains is None:
            expected_domains = domains
        elif domains != expected_domains:
            raise ValueError(f"Domain identity differs for joint-stress condition {pattern}.")
    return result


def _condition_summary(
    raw_rows: Iterable[Mapping[str, Any]],
    conditions: Mapping[str, Mapping[str, Any]],
    *,
    fusions: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["pattern"]), str(row["fusion"]))].append(row)
    result: list[dict[str, Any]] = []
    for pattern, condition in conditions.items():
        branch = {fusion: grouped[(pattern, fusion)] for fusion in fusions}
        if any(len(rows) != 15 for rows in branch.values()):
            raise ValueError(f"Incomplete condition summary inputs for {pattern}.")
        branch_metrics = {
            fusion: {metric: _mean(float(row[metric]) for row in rows) for metric in METRICS}
            for fusion, rows in branch.items()
        }
        for fusion in fusions:
            row: dict[str, Any] = {
                "pattern": pattern,
                "requested_stress_rate": float(condition["requested_stress_rate"]),
                "mask_set_index": int(condition["mask_set_index"]),
                "state_digest": str(condition["state_digest"]),
                "fusion": fusion,
                "domain_count": 15,
                **branch_metrics[fusion],
            }
            for metric in METRICS:
                for control in _comparison_controls(fusions):
                    row[f"learned_minus_{control}_{metric}"] = (
                        branch_metrics["learned"][metric] - branch_metrics[control][metric]
                    )
                row[f"learned_reference_control_closure_{metric}"] = _gap_closure(
                    branch_metrics["learned"][metric],
                    branch_metrics["uniform"][metric],
                    branch_metrics["oracle"][metric],
                )
            for field in REGRET_FIELDS:
                row[field] = (
                    _mean(float(item[field]) for item in branch["learned"])
                    if fusion == "learned"
                    else None
                )
            for field in ROUTER_WEIGHT_FIELDS:
                row[field] = _mean(float(item[field]) for item in branch["learned"])
            if fusions == DYNAMIC_FUSIONS:
                for field in DYNAMIC_DIAGNOSTIC_FIELDS:
                    row[field] = _mean(float(item[field]) for item in branch["learned"])
            result.append(row)
    return result


def _domain_rate_summary(
    raw_rows: Iterable[Mapping[str, Any]],
    conditions: Mapping[str, Mapping[str, Any]],
    *,
    fusions: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(float(row["requested_stress_rate"]), str(row["domain_id"]), str(row["fusion"]))].append(row)
    domains = sorted({str(row["domain_id"]) for row in raw_rows})
    result = []
    for rate in (0.0, *JOINT_RATES):
        expected_masks = 1 if rate == 0.0 else MASKS_PER_RATE
        for domain_id in domains:
            branch: dict[str, dict[str, float]] = {}
            for fusion in fusions:
                selected = grouped[(float(rate), domain_id, fusion)]
                if len(selected) != expected_masks:
                    raise ValueError(
                        f"rate={rate:g}/{domain_id}/{fusion} expected {expected_masks} masks, got {len(selected)}."
                    )
                branch[fusion] = {metric: _mean(float(row[metric]) for row in selected) for metric in METRICS}
            for fusion in fusions:
                item: dict[str, Any] = {
                    "requested_stress_rate": float(rate),
                    "domain_id": domain_id,
                    "fusion": fusion,
                    "mask_count": expected_masks,
                    **branch[fusion],
                }
                for metric in METRICS:
                    for control in _comparison_controls(fusions):
                        item[f"learned_minus_{control}_{metric}"] = (
                            branch["learned"][metric] - branch[control][metric]
                        )
                learned_rows = grouped[(float(rate), domain_id, "learned")]
                for field in REGRET_FIELDS:
                    item[field] = (
                        _mean(float(row[field]) for row in learned_rows)
                        if fusion == "learned"
                        else None
                    )
                for field in ROUTER_WEIGHT_FIELDS:
                    item[field] = _mean(float(row[field]) for row in learned_rows)
                if fusions == DYNAMIC_FUSIONS:
                    for field in DYNAMIC_DIAGNOSTIC_FIELDS:
                        item[field] = _mean(float(row[field]) for row in learned_rows)
                result.append(item)
    return result


def _rate_summary(
    domain_rows: Iterable[Mapping[str, Any]],
    condition_rows: Iterable[Mapping[str, Any]],
    *,
    fusions: tuple[str, ...],
) -> list[dict[str, Any]]:
    by_rate_fusion: dict[tuple[float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in domain_rows:
        by_rate_fusion[(float(row["requested_stress_rate"]), str(row["fusion"]))].append(row)
    by_rate_condition: dict[tuple[float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in condition_rows:
        by_rate_condition[(float(row["requested_stress_rate"]), str(row["fusion"]))].append(row)
    result: list[dict[str, Any]] = []
    for rate in (0.0, *JOINT_RATES):
        branches = {fusion: by_rate_fusion[(float(rate), fusion)] for fusion in fusions}
        if any(len(rows) != 15 for rows in branches.values()):
            raise ValueError(f"rate={rate:g} does not have exactly 15 domain aggregates per fusion.")
        values = {
            fusion: {metric: _mean(float(row[metric]) for row in rows) for metric in METRICS}
            for fusion, rows in branches.items()
        }
        row: dict[str, Any] = {
            "cell": "Clean" if rate == 0.0 else f"Joint{int(round(rate * 100))}",
            "requested_stress_rate": float(rate),
            "mask_count": 1 if rate == 0.0 else MASKS_PER_RATE,
            "domain_count": 15,
        }
        for metric in METRICS:
            for fusion in fusions:
                row[f"{fusion}_{metric}"] = values[fusion][metric]
            for control in _comparison_controls(fusions):
                row[f"learned_minus_{control}_{metric}"] = (
                    values["learned"][metric] - values[control][metric]
                )
            row[f"learned_reference_control_closure_{metric}"] = _gap_closure(
                values["learned"][metric], values["uniform"][metric], values["oracle"][metric]
            )
            masks = by_rate_condition[(float(rate), "learned")]
            row[f"learned_mask_win_rate_{metric}"] = (
                None
                if rate == 0.0
                else _mean(float(item[f"learned_minus_uniform_{metric}"]) > 0.0 for item in masks)
            )
        for field in REGRET_FIELDS:
            row[field] = _mean(float(item[field]) for item in branches["learned"])
        for field in ROUTER_WEIGHT_FIELDS:
            row[field] = _mean(float(item[field]) for item in branches["learned"])
        if fusions == DYNAMIC_FUSIONS:
            for field in DYNAMIC_DIAGNOSTIC_FIELDS:
                row[field] = _mean(float(item[field]) for item in branches["learned"])
        result.append(row)
    return result


def _bootstrap_rows(
    domain_rows: Iterable[Mapping[str, Any]],
    *,
    controls: tuple[str, ...],
    iterations: int,
    seed: int,
    confidence: float,
) -> list[dict[str, Any]]:
    indexed = {
        (float(row["requested_stress_rate"]), str(row["domain_id"]), str(row["fusion"])): row
        for row in domain_rows
    }
    learned = {
        (rate, domain): row
        for (rate, domain, fusion), row in indexed.items()
        if fusion == "learned"
    }
    domains = sorted({domain for _, domain in learned})
    if len(domains) != 15:
        raise ValueError("Paired joint-stress bootstrap requires exactly 15 domains.")
    rows: list[dict[str, Any]] = []
    for control in controls:
        control_keys = {
            (rate, domain)
            for rate, domain, fusion in indexed
            if fusion == control
        }
        if learned.keys() != control_keys:
            raise ValueError(f"Learned and {control} domain/rate identities differ.")
        for metric in GATE_METRICS:
            for rate in GATE_RATES:
                deltas = np.asarray(
                    [
                        float(learned[(rate, domain)][metric])
                        - float(indexed[(rate, domain, control)][metric])
                        for domain in domains
                    ],
                    dtype=np.float64,
                )
                low, high = _paired_domain_bootstrap(
                    deltas,
                    iterations=iterations,
                    seed=_stable_seed(seed, control, metric, f"{rate:.1f}"),
                    confidence=confidence,
                )
                rows.append(
                    _bootstrap_row(
                        scope=f"Joint{int(round(rate * 100))}",
                        control=control,
                        metric=metric,
                        deltas=deltas,
                        low=low,
                        high=high,
                        confidence=confidence,
                        iterations=iterations,
                        unit="paired_domain_after_equal_mask_average",
                    )
                )
            combined = np.asarray(
                [
                    _mean(
                        float(learned[(rate, domain)][metric])
                        - float(indexed[(rate, domain, control)][metric])
                        for rate in GATE_RATES
                    )
                    for domain in domains
                ],
                dtype=np.float64,
            )
            low, high = _paired_domain_bootstrap(
                combined,
                iterations=iterations,
                seed=_stable_seed(seed, control, metric, "combined_40_80"),
                confidence=confidence,
            )
            rows.append(
                _bootstrap_row(
                    scope="Joint40_60_80Combined",
                    control=control,
                    metric=metric,
                    deltas=combined,
                    low=low,
                    high=high,
                    confidence=confidence,
                    iterations=iterations,
                    unit="paired_domain_after_equal_mask_and_rate_average",
                )
            )
    return rows


def _bootstrap_row(
    *,
    scope: str,
    control: str,
    metric: str,
    deltas: np.ndarray,
    low: float,
    high: float,
    confidence: float,
    iterations: int,
    unit: str,
) -> dict[str, Any]:
    mean_delta = float(deltas.mean())
    return {
        "scope": scope,
        "control": control,
        "contrast": f"learned_minus_{control}",
        "metric": metric,
        "mean_delta": mean_delta,
        "mean_learned_minus_control": mean_delta,
        "mean_learned_minus_uniform": mean_delta if control == "uniform" else None,
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "bootstrap_iterations": iterations,
        "paired_domain_count": int(deltas.size),
        "bootstrap_unit": unit,
    }


def _gate_decision(
    rate_rows: Iterable[Mapping[str, Any]],
    bootstrap_rows: Iterable[Mapping[str, Any]],
    *,
    fusions: tuple[str, ...],
) -> dict[str, Any]:
    rates = {float(row["requested_stress_rate"]): row for row in rate_rows}
    dynamic = fusions == DYNAMIC_FUSIONS
    control = "train_fit_static_prior" if dynamic else "uniform"
    intervals = {
        (str(row["scope"]), str(row["metric"]), str(row["control"])): row
        for row in bootstrap_rows
    }
    checks = []
    for rate in GATE_RATES:
        for metric in GATE_METRICS:
            delta = float(rates[rate][f"learned_minus_{control}_{metric}"])
            checks.append(
                {
                    "check": f"Joint{int(round(rate * 100))}_{metric}_learned_minus_{control}_positive",
                    "value": delta,
                    "threshold": 0.0,
                    "passed": delta > 0.0,
                }
            )
    for metric in GATE_METRICS:
        interval = intervals[("Joint40_60_80Combined", metric, control)]
        mean_delta = float(interval["mean_delta"])
        minimum_delta = COMBINED_MIN_DELTA[metric]
        if dynamic:
            checks.append(
                {
                    "check": f"Joint40_60_80Combined_{metric}_learned_minus_{control}_minimum_effect",
                    "value": mean_delta,
                    "threshold": minimum_delta,
                    "passed": mean_delta >= minimum_delta,
                }
            )
        low = float(interval["ci_low"])
        checks.append(
            {
                "check": f"Joint40_60_80Combined_{metric}_learned_minus_{control}_bootstrap_low_positive",
                "value": low,
                "threshold": 0.0,
                "passed": low > 0.0,
            }
        )
    if dynamic:
        for combined_control in ("frozen_current_router", "uniform"):
            for metric in GATE_METRICS:
                low = float(intervals[("Joint40_60_80Combined", metric, combined_control)]["ci_low"])
                checks.append(
                    {
                        "check": f"Joint40_60_80Combined_{metric}_learned_minus_{combined_control}_bootstrap_low_positive",
                        "value": low,
                        "threshold": 0.0,
                        "passed": low > 0.0,
                    }
                )
        for clean_control in ("train_fit_static_prior", "frozen_current_router"):
            delta = float(rates[0.0][f"learned_minus_{clean_control}_adba"])
            checks.append(
                {
                    "check": f"Clean_adba_learned_minus_{clean_control}_noninferior",
                    "value": delta,
                    "threshold": -CLEAN_ADBA_NONINFERIORITY_MARGIN,
                    "passed": delta >= -CLEAN_ADBA_NONINFERIORITY_MARGIN,
                }
            )
        response_rates = [rates[rate] for rate in GATE_RATES]
        downweight_rate = _mean(
            float(row["corrupted_cell_downweight_vs_static_rate"]) for row in response_rates
        )
        response_ratio = _mean(
            float(row["corrupted_cell_weight_vs_static_ratio"]) for row in response_rates
        )
        checks.extend(
            [
                {
                    "check": "Joint40_60_80_corrupted_cell_downweight_vs_static_rate",
                    "value": downweight_rate,
                    "threshold": MIN_CORRUPTED_CELL_DOWNWEIGHT_RATE,
                    "passed": downweight_rate >= MIN_CORRUPTED_CELL_DOWNWEIGHT_RATE,
                },
                {
                    "check": "Joint40_60_80_corrupted_cell_weight_vs_static_ratio",
                    "value": response_ratio,
                    "threshold": MAX_CORRUPTED_CELL_WEIGHT_VS_STATIC_RATIO,
                    "comparison": "exclusive_upper_bound",
                    "passed": response_ratio < MAX_CORRUPTED_CELL_WEIGHT_VS_STATIC_RATIO,
                },
            ]
        )
    passed = all(bool(item["passed"]) for item in checks)
    if dynamic:
        return {
            "passed": passed,
            "mechanism_gate_passed": passed,
            "claim_eligible": False,
            "decision": "advance_to_pure_drop" if passed else "candidate_rejected_by_inner_gate",
            "primary_control": control,
            "thresholds": {
                "combined_min_delta": dict(COMBINED_MIN_DELTA),
                "clean_adba_noninferiority_margin": CLEAN_ADBA_NONINFERIORITY_MARGIN,
                "corrupted_cell_downweight_vs_static_rate": MIN_CORRUPTED_CELL_DOWNWEIGHT_RATE,
                "corrupted_cell_weight_vs_static_ratio_exclusive_max": MAX_CORRUPTED_CELL_WEIGHT_VS_STATIC_RATIO,
            },
            "pure_drop_protection": "pending_separate_fixed_mask_evaluation",
            "checks": checks,
            "policy": (
                "Joint40/60/80 的 ADBA 与 normalized gain 均须 Dynamic>train-fit static prior，"
                "合并 ΔADBA>=0.002、Δgain>=0.005，且相对 static/current/uniform 的合并 paired-domain "
                "bootstrap 下界均须大于0；Clean ADBA 相对 static/current 下降不得超过0.002；"
                "Corrupt cell相对static prior的降权率须>=0.60且权重比<1。纯Drop仍须独立验证。"
            ),
        }
    return {
        "passed": passed,
        "decision": "keep_current_router" if passed else "diagnostic_failed_no_model_change",
        "claim_eligible": False,
        "primary_control": control,
        "checks": checks,
        "policy": (
            "Joint40/60/80 的 ADBA 与 normalized gain 均须 Learned>Uniform，且两项指标的"
            "三档合并 paired-domain bootstrap 下界均须大于 0。失败只触发后续独立设计，"
            "本诊断不会自动修改主线模型。"
        ),
    }


def _paired_domain_bootstrap(
    deltas: np.ndarray,
    *,
    iterations: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    if deltas.shape != (15,) or not np.isfinite(deltas).all():
        raise ValueError("Paired-domain bootstrap requires 15 finite domain deltas.")
    rng = np.random.default_rng(seed)
    draws = deltas[rng.integers(0, len(deltas), size=(iterations, len(deltas)))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _plot(
    root: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fusions: tuple[str, ...],
) -> None:
    import matplotlib.pyplot as plt

    selected = sorted(rows, key=lambda item: float(item["requested_stress_rate"]))
    x = [100.0 * float(row["requested_stress_rate"]) for row in selected]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    for axis, metric, label in zip(axes, ("adba", "normalized_gain"), ("ADBA", "Normalized gain")):
        markers = ("o", "s", "^", "D", "v", "P", "X")
        for fusion, marker in zip(fusions, markers, strict=True):
            axis.plot(x, [float(row[f"{fusion}_{metric}"]) for row in selected], marker=marker, label=fusion.title())
        axis.set_xlabel("Joint stressed cells (%)")
        axis.set_ylabel(label)
        axis.set_xticks(x)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(root / "joint_router_stress_curve.png", dpi=220)
    plt.close(fig)


def _markdown(
    provenance: Mapping[str, Any],
    rate_rows: Iterable[Mapping[str, Any]],
    bootstrap_rows: Iterable[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> str:
    dynamic = tuple(provenance["fusion_branches"]) == DYNAMIC_FUSIONS
    lines = [
        "# MMW Router Joint Drop+Corrupt 机制诊断",
        "",
        "固定同一个 CurrentControl seed1 inner-validation checkpoint，对 20 个模态时间块施加互斥的 Clean/Drop/Corrupt 三态压力。该结果 `claim_eligible=false`，用于决定是否需要继续改 Router，不替代正式 outer 多 seed 证据。",
        "",
        "## 结果",
        "",
        "| Cell | Uniform ADBA | Learned ADBA | Δ ADBA | Uniform gain | Learned gain | Δ gain | Reference oracle gain | Reference-control closure | Mask win (gain) | Learned soft regret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rate_rows:
        lines.append(
            f"| {row['cell']} | {float(row['uniform_adba']):.4f} | {float(row['learned_adba']):.4f} | "
            f"{float(row['learned_minus_uniform_adba']):+.4f} | {float(row['uniform_normalized_gain']):.4f} | "
            f"{float(row['learned_normalized_gain']):.4f} | {float(row['learned_minus_uniform_normalized_gain']):+.4f} | "
            f"{float(row['oracle_normalized_gain']):.4f} | {_format_optional(row['learned_reference_control_closure_normalized_gain'])} | "
            f"{_format_optional(row['learned_mask_win_rate_normalized_gain'])} | {float(row['router_soft_oracle_regret']):.4f} |"
        )
    if dynamic:
        lines.extend(
            [
                "",
                "## 动态 Router 对照",
                "",
                "`Uniform` 和 `Reference oracle` 使用冻结 reference experts，`Learned` 使用 candidate post-health experts。因此 `Reference-control closure` 仅是混合 expert 的诊断分解，不是纯 Router Oracle-gap，也不作为 Gate 条件；`Learned soft regret` 只对应 candidate learned Router 分支。",
                "",
                "| Cell | Uniform ADBA | Static ADBA | Current ADBA | Post-health uniform | Post-health static | Dynamic ADBA | Δ Static ADBA | Δ Current ADBA | Δ Post-health static ADBA |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rate_rows:
            lines.append(
                f"| {row['cell']} | {float(row['uniform_adba']):.4f} | "
                f"{float(row['train_fit_static_prior_adba']):.4f} | "
                f"{float(row['frozen_current_router_adba']):.4f} | "
                f"{float(row['post_health_uniform_adba']):.4f} | "
                f"{float(row['post_health_static_prior_adba']):.4f} | "
                f"{float(row['learned_adba']):.4f} | "
                f"{float(row['learned_minus_train_fit_static_prior_adba']):+.4f} | "
                f"{float(row['learned_minus_frozen_current_router_adba']):+.4f} | "
                f"{float(row['learned_minus_post_health_static_prior_adba']):+.4f} |"
            )
    lines.extend(
        [
            "",
            "## Top-1 与频谱效率比",
            "",
            "| Cell | Uniform Top1 | Learned Top1 | Δ Top1 | Uniform R@0dB | Learned R@0dB | Uniform R@10dB | Learned R@10dB | Uniform R@20dB | Learned R@20dB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rate_rows:
        lines.append(
            f"| {row['cell']} | {float(row['uniform_top1']):.4f} | {float(row['learned_top1']):.4f} | "
            f"{float(row['learned_minus_uniform_top1']):+.4f} | "
            f"{float(row['uniform_spectral_efficiency_ratio_0db']):.4f} | "
            f"{float(row['learned_spectral_efficiency_ratio_0db']):.4f} | "
            f"{float(row['uniform_spectral_efficiency_ratio_10db']):.4f} | "
            f"{float(row['learned_spectral_efficiency_ratio_10db']):.4f} | "
            f"{float(row['uniform_spectral_efficiency_ratio_20db']):.4f} | "
            f"{float(row['learned_spectral_efficiency_ratio_20db']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Learned Router 平均权重",
            "",
            "| Cell | Image | Radar | GPS | LiDAR |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rate_rows:
        lines.append(
            f"| {row['cell']} | {float(row['router_weight_image']):.4f} | "
            f"{float(row['router_weight_radar']):.4f} | {float(row['router_weight_gps']):.4f} | "
            f"{float(row['router_weight_lidar']):.4f} |"
        )
    if dynamic:
        lines.extend(
            [
                "",
                "## 动态响应审计",
                "",
                "`corrupt weight ratio < 1` 表示 Router 给 Corrupt cell 的总权重低于其在可用 cell 中的数量占比。",
                "",
                "| Cell | Mean abs(residual) | Corrupt cell share | Dynamic mass | Static-prior mass | vs-uniform ratio | vs-static ratio | Downweight rate |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rate_rows:
            lines.append(
                f"| {row['cell']} | {float(row['router_residual_abs_mean']):.4f} | "
                f"{float(row['corrupted_cell_available_share']):.4f} | "
                f"{float(row['corrupted_cell_weight_mass']):.4f} | "
                f"{float(row['corrupted_cell_static_prior_mass']):.4f} | "
                f"{float(row['corrupted_cell_weight_response_ratio']):.4f} | "
                f"{float(row['corrupted_cell_weight_vs_static_ratio']):.4f} | "
                f"{float(row['corrupted_cell_downweight_vs_static_rate']):.4f} |"
            )
    lines.extend(
        [
            "",
            "## Paired-domain bootstrap",
            "",
            "先在每个域内等权平均固定 mask，再以 15 个域为配对重采样单位。",
            "",
            "| Scope | Control | Metric | Dynamic-Control | CI low | CI high |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in bootstrap_rows:
        lines.append(
            f"| {row['scope']} | {row['control']} | {row['metric']} | {float(row['mean_delta']):+.4f} | "
            f"{float(row['ci_low']):+.4f} | {float(row['ci_high']):+.4f} |"
        )
    lines.extend(
        [
            "",
            "## 决策",
            "",
            f"- Gate：`{'PASS' if gate['passed'] else 'FAIL'}`",
            f"- 动作：`{gate['decision']}`",
            f"- 规则：{gate['policy']}",
            "",
            "## 复现身份",
            "",
            f"- Protocol：`{provenance['protocol']}`",
            f"- Cache SHA256：`{provenance['cache_sha256']}`",
            f"- Cache checksum：`{provenance['cache_checksum']}`",
            f"- Checkpoint SHA256：`{provenance['checkpoint_sha256']}`",
            f"- Corruption seed：`{provenance['corruption_seed']}`；固定 S2 参数：`{json.dumps(provenance['corruption_parameters'], ensure_ascii=False, sort_keys=True)}`",
            f"- Bootstrap：`{provenance['bootstrap']['iterations']}` 次，seed `{provenance['bootstrap']['seed']}`",
            f"- Gate thresholds：`{json.dumps(provenance['gate_thresholds'], ensure_ascii=False, sort_keys=True)}`",
            "",
        ]
    )
    return "\n".join(lines)


def _finite_float(row: Mapping[str, str], key: str, path: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {key} in {path}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key} in {path}.")
    return value


def _gap_closure(learned: float, uniform: float, oracle: float) -> float | None:
    denominator = oracle - uniform
    if denominator <= 0.0:
        return None
    return (learned - uniform) / denominator


def _stable_seed(base: int, *parts: str) -> int:
    payload = "::".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _format_optional(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("Cannot average an empty collection.")
    return float(sum(items) / len(items))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
