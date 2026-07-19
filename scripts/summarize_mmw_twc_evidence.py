#!/usr/bin/env python3
"""Summarize complete MMW TWC post-selection confirmation evidence.

This is deliberately separate from the historical all-weather summary.  It
accepts only the immutable ``mmw_twc_outer_v1`` protocol and refuses partial
or identity-mismatched evidence before writing a claim-facing artifact.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


METHODS = ("T2", "S1", "masktrain_cls", "amber_full", "rmbp_mm", "amr_net_4m")
MAIN_COMPARISON_METHODS = METHODS
SEEDS = (1, 2, 3, 4, 5)
METRICS = (
    "adba", "top1", "top3", "top5", "within_1", "within_3", "mae",
    "normalized_gain", "gain_loss_db",
    "spectral_efficiency_ratio_0db", "spectral_efficiency_loss_0db",
    "spectral_efficiency_ratio_10db", "spectral_efficiency_loss_10db",
    "spectral_efficiency_ratio_20db", "spectral_efficiency_loss_20db",
)
PRIMARY_METRIC = "adba"
SECONDARY_METRIC = "top1"
ADBA_DELTA = 5.0
ADBA_DISTANCE_MODE = "circular"
HIGHER_IS_BETTER = frozenset(
    ("top1", "top3", "top5", "within_1", "within_3", "adba", "normalized_gain")
    + tuple(metric for metric in METRICS if metric.startswith("spectral_efficiency_ratio_"))
)
PROTOCOL_ID = "mmw_twc_outer_v1"
PROTOCOL_KIND = "post_selection_confirmation_not_historical_blind_test"
OUTER_ROLE = "outer_evidence"
TRAINING_ROLE = "confirmation_train"
TEMPORAL_RATES = (0.2, 0.4, 0.6, 0.8)
TEMPORAL_MASK_TYPES = ("modality_frame", "frame_level", "block")


@dataclass(frozen=True)
class ProtocolEvidence:
    path: Path
    manifest_sha256: str
    fixed_mask_cache_sha256: str
    fixed_mask_cache_checksum: str
    domains: dict[str, dict[str, str]]
    conditions: dict[tuple[Any, ...], dict[str, Any]]


@dataclass
class EvalUnit:
    method: str
    seed: int
    path: Path
    rows: list[dict[str, str]]
    provenance: dict[str, str]
    fidelity: dict[str, str]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize complete MMW TWC post-selection confirmation evidence."
    )
    parser.add_argument("--eval-dir", required=True, help="Root containing <method>/seed<seed>/metrics.csv.")
    parser.add_argument(
        "--protocol-manifest",
        default="outputs/cache/mmw_twc_outer_v1/protocol_manifest.json",
        help="Immutable mmw_twc_outer_v1 protocol manifest.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    methods = _csv_values(args.methods)
    seeds = _csv_ints(args.seeds)
    if not methods or len(set(methods)) != len(methods) or set(methods) != set(METHODS):
        parser.error(f"--methods must contain exactly {METHODS}")
    if not seeds or len(set(seeds)) != len(seeds) or tuple(sorted(seeds)) != SEEDS:
        parser.error(f"--seeds must contain exactly {SEEDS}")
    if args.bootstrap_iterations <= 0 or not 0.0 < args.confidence < 1.0:
        parser.error("bootstrap iterations must be positive and confidence must be in (0, 1)")

    summarize(
        Path(args.eval_dir),
        Path(args.protocol_manifest),
        Path(args.output_dir),
        methods=tuple(methods),
        seeds=tuple(seeds),
        bootstrap_iterations=int(args.bootstrap_iterations),
        bootstrap_seed=int(args.bootstrap_seed),
        confidence=float(args.confidence),
    )
    return 0


def summarize(
    eval_dir: Path,
    protocol_manifest: Path,
    output_dir: Path,
    *,
    methods: tuple[str, ...] = METHODS,
    seeds: tuple[int, ...] = SEEDS,
    bootstrap_iterations: int = 10000,
    bootstrap_seed: int = 20260719,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Validate and summarize one complete frozen confirmation matrix.

    The function intentionally raises instead of returning a partial summary.
    A caller therefore cannot accidentally turn incomplete overnight output into
    a paper table.
    """
    if set(methods) != set(METHODS) or len(methods) != len(METHODS):
        raise ValueError(f"TWC summary requires exactly the six registered method cells {METHODS}.")
    if tuple(sorted(seeds)) != SEEDS or len(seeds) != len(SEEDS):
        raise ValueError(f"TWC summary requires the fixed seed set {SEEDS}.")
    if bootstrap_iterations <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap configuration.")

    protocol = _load_protocol_evidence(protocol_manifest)
    _validate_required_protocol_cells(protocol)
    units = _load_complete_units(eval_dir, protocol, methods, seeds)
    _validate_cross_unit_identity(units, protocol)

    domain_cells = _domain_main_cells(units, protocol)
    domain_macro = _domain_macro_cells(domain_cells)
    temporal_curve = _temporal_curve_rows(units, protocol)
    weather_macro = _group_macro_rows(domain_cells, "condition")
    scene_macro = _group_macro_rows(domain_cells, "scene")
    worst_domains = _worst_domain_rows(domain_cells)
    paper_table = _seed_aggregate_rows(
        [row for row in domain_macro if row["method"] in MAIN_COMPARISON_METHODS],
        group_fields=("method", "cell"),
    )
    temporal_table = _seed_aggregate_rows(
        [row for row in temporal_curve if row["method"] in MAIN_COMPARISON_METHODS],
        group_fields=("method", "curve_kind", "missing_rate"),
    )
    weather_table = _seed_aggregate_rows(
        [row for row in weather_macro if row["method"] in MAIN_COMPARISON_METHODS],
        group_fields=("method", "cell", "condition"),
    )
    scene_table = _seed_aggregate_rows(
        [row for row in scene_macro if row["method"] in MAIN_COMPARISON_METHODS],
        group_fields=("method", "cell", "scene"),
    )
    worst_table = _seed_aggregate_rows(
        [row for row in worst_domains if row["method"] in MAIN_COMPARISON_METHODS],
        group_fields=("method", "cell"),
    )
    paired_domains, paired_ci = _paired_statistics(
        domain_cells,
        methods=MAIN_COMPARISON_METHODS,
        seeds=seeds,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    coverage = _coverage_rows(units, protocol)
    fidelity = _fidelity_rows(units)
    provenance = _summary_provenance(
        protocol,
        eval_dir=eval_dir,
        methods=methods,
        seeds=seeds,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    plot_manifest = _plot_manifest(provenance)

    _prepare_output_dir(output_dir, provenance)
    _write_csv(output_dir / "coverage.csv", coverage)
    _write_csv(output_dir / "baseline_fidelity.csv", fidelity)
    _write_csv(output_dir / "domain_main_cells.csv", domain_cells)
    _write_csv(output_dir / "domain_macro_per_seed.csv", domain_macro)
    _write_csv(output_dir / "weather_macro_per_seed.csv", weather_macro)
    _write_csv(output_dir / "scene_macro_per_seed.csv", scene_macro)
    _write_csv(output_dir / "worst_domain_per_seed.csv", worst_domains)
    _write_csv(output_dir / "temporal_curve_per_seed.csv", temporal_curve)
    _write_csv(output_dir / "paper_main_table.csv", paper_table)
    _write_csv(output_dir / "paper_temporal_curve.csv", temporal_table)
    _write_csv(output_dir / "paper_weather_table.csv", weather_table)
    _write_csv(output_dir / "paper_scene_table.csv", scene_table)
    _write_csv(output_dir / "paper_worst_domain_table.csv", worst_table)
    _write_csv(output_dir / "paired_domain_seed_deltas.csv", paired_domains)
    _write_csv(output_dir / "paired_bootstrap_ci.csv", paired_ci)
    _write_json(output_dir / "provenance.json", provenance)
    _write_json(output_dir / "plot_manifest.json", plot_manifest)
    _write_text(
        output_dir / "summary.md",
        _markdown(
            provenance,
            paper_table,
            paired_ci,
            fidelity,
        ),
    )
    return {
        "protocol": protocol,
        "units": units,
        "coverage": coverage,
        "domain_cells": domain_cells,
        "domain_macro": domain_macro,
        "temporal_curve": temporal_curve,
        "paper_table": paper_table,
        "paired_ci": paired_ci,
        "provenance": provenance,
    }


def _load_protocol_evidence(path: Path) -> ProtocolEvidence:
    path = path.resolve()
    payload = _read_json(path)
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"TWC summary requires protocol_id={PROTOCOL_ID!r}.")
    if payload.get("protocol_kind") != PROTOCOL_KIND:
        raise ValueError("TWC summary requires the post-selection confirmation protocol kind.")
    recorded_manifest_sha = _required_text(payload, "manifest_sha256", "protocol manifest")
    without_manifest_sha = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if recorded_manifest_sha != _sha256_payload(without_manifest_sha):
        raise ValueError("TWC protocol manifest checksum mismatch.")

    cache_record = payload.get("fixed_mask_cache")
    if not isinstance(cache_record, Mapping):
        raise ValueError("TWC protocol has no fixed-mask cache record.")
    cache_path = _resolve_artifact_path(path.parent, _required_text(cache_record, "path", "fixed-mask cache"))
    if not cache_path.is_file():
        raise FileNotFoundError(f"TWC fixed-mask cache is missing: {cache_path}")
    cache_sha256 = _required_text(cache_record, "sha256", "fixed-mask cache")
    if _sha256_file(cache_path) != cache_sha256:
        raise ValueError("TWC fixed-mask cache file SHA256 mismatch.")
    cache = _read_json(cache_path)
    cache_checksum = _required_text(cache_record, "cache_checksum", "fixed-mask cache")
    if cache.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("TWC fixed-mask cache protocol identity mismatch.")
    if str(cache.get("checksum", "")) != cache_checksum:
        raise ValueError("TWC fixed-mask cache checksum mismatch.")
    without_cache_checksum = {key: value for key, value in cache.items() if key != "checksum"}
    if cache_checksum != _sha256_payload(without_cache_checksum):
        raise ValueError("TWC fixed-mask cache canonical checksum mismatch.")

    domains_payload = payload.get("domains")
    if not isinstance(domains_payload, list) or len(domains_payload) != 15:
        raise ValueError("TWC protocol must contain exactly 15 outer-evidence domains.")
    domains: dict[str, dict[str, str]] = {}
    for raw in domains_payload:
        if not isinstance(raw, Mapping):
            raise ValueError("TWC protocol domain record must be a mapping.")
        domain_id = _required_text(raw, "id", "protocol domain")
        split = raw.get("split")
        if not isinstance(split, Mapping) or not isinstance(split.get(OUTER_ROLE), Mapping):
            raise ValueError(f"TWC protocol domain {domain_id} is missing its outer-evidence split.")
        outer = split[OUTER_ROLE]
        outer_path = _resolve_artifact_path(path.parent, _required_text(outer, "csv", f"outer split {domain_id}"))
        outer_sha256 = _required_text(outer, "sha256", f"outer split {domain_id}")
        if not outer_path.is_file() or _sha256_file(outer_path) != outer_sha256:
            raise ValueError(f"TWC outer-evidence split changed for {domain_id}.")
        if domain_id in domains:
            raise ValueError(f"TWC protocol repeats outer-evidence domain {domain_id}.")
        domains[domain_id] = {
            "condition": _required_text(raw, "condition", f"protocol domain {domain_id}"),
            "scene": _required_text(raw, "scene", f"protocol domain {domain_id}"),
            "sample_csv_sha256": outer_sha256,
            "expected_sample_count": str(_coerce_int(outer.get("row_count"), f"outer row_count {domain_id}")),
        }

    conditions_payload = cache.get("conditions")
    if not isinstance(conditions_payload, list) or not conditions_payload:
        raise ValueError("TWC fixed-mask cache has no conditions.")
    conditions: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in conditions_payload:
        if not isinstance(raw, Mapping):
            raise ValueError("TWC fixed-mask condition must be a mapping.")
        condition = dict(raw)
        key = _condition_key(condition, source="fixed-mask cache")
        digest = str(condition["mask_digest"])
        if not _is_sha256(digest):
            raise ValueError(f"TWC fixed-mask digest is not a full SHA256: {digest!r}.")
        if key in conditions:
            raise ValueError(f"TWC fixed-mask cache repeats condition identity {key}.")
        conditions[key] = condition
    recorded_condition_count = int(cache_record.get("condition_count", -1))
    if recorded_condition_count != len(conditions):
        raise ValueError("TWC fixed-mask cache condition count mismatch.")
    return ProtocolEvidence(
        path=path,
        manifest_sha256=recorded_manifest_sha,
        fixed_mask_cache_sha256=cache_sha256,
        fixed_mask_cache_checksum=cache_checksum,
        domains=domains,
        conditions=conditions,
    )


def _validate_required_protocol_cells(protocol: ProtocolEvidence) -> None:
    conditions = list(protocol.conditions.values())
    required = {
        "Clean": lambda item: _is_whole(item, drop_count=0),
        "Drop1": lambda item: _is_whole(item, drop_count=1),
        "Drop2": lambda item: _is_whole(item, drop_count=2),
        "Drop3": lambda item: _is_whole(item, drop_count=3),
        "Block80": lambda item: _is_temporal(item, rate=0.8, mask_type="block"),
    }
    for label, selector in required.items():
        if not any(selector(item) for item in conditions):
            raise ValueError(f"TWC fixed-mask cache is missing required main cell {label}.")
    for rate in TEMPORAL_RATES:
        for mask_type in TEMPORAL_MASK_TYPES:
            if not any(_is_temporal(item, rate=rate, mask_type=mask_type) for item in conditions):
                raise ValueError(
                    f"TWC fixed-mask cache is missing temporal condition rate={rate:g}, type={mask_type}."
                )


def _load_complete_units(
    eval_dir: Path,
    protocol: ProtocolEvidence,
    methods: Iterable[str],
    seeds: Iterable[int],
) -> dict[tuple[str, int], EvalUnit]:
    units: dict[tuple[str, int], EvalUnit] = {}
    for method in methods:
        for seed in seeds:
            metrics_path = _metrics_path(eval_dir, method, seed)
            if not metrics_path.is_file():
                raise FileNotFoundError(f"Missing strict evidence metrics CSV: {metrics_path}")
            rows = _read_csv(metrics_path)
            if not rows:
                raise ValueError(f"Strict evidence metrics CSV is empty: {metrics_path}")
            provenance, fidelity = _validate_unit(rows, metrics_path, method, seed, protocol)
            units[(method, seed)] = EvalUnit(method, seed, metrics_path, rows, provenance, fidelity)
    return units


def _metrics_path(eval_dir: Path, method: str, seed: int) -> Path:
    return eval_dir / method / f"seed{seed}" / "metrics.csv"


def _validate_unit(
    rows: list[dict[str, str]],
    path: Path,
    method: str,
    seed: int,
    protocol: ProtocolEvidence,
) -> tuple[dict[str, str], dict[str, str]]:
    expected_count = len(protocol.domains) * len(protocol.conditions)
    if len(rows) != expected_count:
        raise ValueError(
            f"Strict evidence row coverage mismatch for {method}/seed{seed}: "
            f"observed={len(rows)}, expected={expected_count}."
        )
    seen: set[tuple[str, tuple[Any, ...]]] = set()
    condition_domains: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    provenance_values: dict[str, set[str]] = defaultdict(set)
    fidelity_values: dict[str, set[str]] = defaultdict(set)
    domain_counts: dict[str, tuple[int, int]] = {}
    for index, row in enumerate(rows, start=2):
        _validate_common_row_fields(row, path, index, method, seed)
        _validate_protocol_row(row, path, index, protocol)
        domain_id = str(row["domain_id"]).strip()
        expected_domain = protocol.domains.get(domain_id)
        if expected_domain is None:
            raise ValueError(f"Unexpected domain {domain_id!r} in {path}:{index}.")
        for field in ("condition", "scene", "sample_csv_sha256"):
            if str(row.get(field, "")).strip() != expected_domain[field]:
                raise ValueError(f"Domain identity mismatch for {domain_id}/{field} in {path}:{index}.")
        condition_key = _condition_key(row, source=f"{path}:{index}")
        if condition_key not in protocol.conditions:
            raise ValueError(f"Condition identity is not in the frozen fixed-mask cache: {path}:{index}.")
        cache_condition = protocol.conditions[condition_key]
        if not _close(_observed_missing_rate(row), _observed_missing_rate(cache_condition)):
            raise ValueError(f"Observed missing rate mismatch for fixed mask at {path}:{index}.")
        identity = (domain_id, condition_key)
        if identity in seen:
            raise ValueError(f"Duplicate domain/mask evidence row in {path}:{index}.")
        seen.add(identity)
        condition_domains[condition_key].add(domain_id)
        observed_count = _int_field(row, "sample_count", path, index)
        expected_sample_count = _int_field(row, "expected_sample_count", path, index)
        if observed_count <= 0 or expected_sample_count <= 0 or observed_count != expected_sample_count:
            raise ValueError(f"Incomplete sample coverage at {path}:{index}.")
        previous_counts = domain_counts.setdefault(domain_id, (observed_count, expected_sample_count))
        if previous_counts != (observed_count, expected_sample_count):
            raise ValueError(f"Sample-count provenance changes within domain {domain_id} in {path}.")
        if expected_sample_count != int(expected_domain["expected_sample_count"]):
            raise ValueError(f"Expected sample count differs from frozen outer split for {domain_id} at {path}:{index}.")
        for metric in METRICS:
            value = _finite_float(row.get(metric))
            if value is None:
                raise ValueError(f"Missing or non-finite metric {metric} at {path}:{index}.")
        for field in _UNIT_PROVENANCE_FIELDS:
            provenance_values[field].add(_row_alias_value(row, field, path, index))
        for field in _FIDELITY_FIELDS:
            value = _fidelity_value(row, field)
            if value:
                fidelity_values[field].add(value)

    if len(seen) != expected_count:
        raise ValueError(f"Incomplete unique row coverage for {method}/seed{seed}.")
    expected_domains = set(protocol.domains)
    for condition_key, domains in condition_domains.items():
        if domains != expected_domains:
            raise ValueError(
                f"Incomplete 15-domain coverage for fixed condition {condition_key} in {method}/seed{seed}."
            )
    if set(condition_domains) != set(protocol.conditions):
        raise ValueError(f"Missing fixed-mask conditions in {method}/seed{seed}.")
    provenance = _singleton_values(provenance_values, f"unit provenance {method}/seed{seed}")
    provenance["condition_identity_sha256"] = _condition_identity_from_rows(rows)
    fidelity = _singleton_values(fidelity_values, f"fidelity metadata {method}/seed{seed}", allow_empty=True)
    _validate_unit_provenance(provenance, method, seed, protocol)
    _validate_fidelity(fidelity, method)
    return provenance, fidelity


_UNIT_PROVENANCE_FIELDS = (
    "checkpoint_sha256",
    "checkpoint_role",
    "checkpoint_policy",
    "metric_profile",
    "protocol_id",
    "protocol_kind",
    "protocol_manifest_sha256",
    "confirmation_split_manifest_sha256",
    "split_role",
    "training_role",
    "training_mask_seed",
    "training_mask_seed_algorithm",
    "smoke_preflight",
    "training_batch_size",
    "training_epochs",
    "evaluation_mask_cache_sha256",
    "evaluation_mask_cache_checksum",
    "topology_id",
    "topology_descriptor_sha256",
    "topology_mapping_sha256",
    "evaluation_topology_id",
    "evaluation_topology_descriptor_sha256",
    "recipe_fingerprint",
)
_FIDELITY_FIELDS = (
    "reproduction_scope",
    "paper_equivalent",
    "temporal_result_scope",
    "baseline_adaptation_scope",
    "missing_paper_components",
)
_RECIPE_ALIASES = (
    "recipe_fingerprint",
    "config_recipe_sha256",
    "config_sha256",
    "design_config_sha256",
)
_FIDELITY_ALIASES = {
    "baseline_adaptation_scope": ("baseline_adaptation_scope", "architecture_scope", "baseline_scope"),
    "missing_paper_components": ("missing_paper_components",),
}


def _validate_common_row_fields(
    row: Mapping[str, str], path: Path, line: int, method: str, seed: int
) -> None:
    required = (
        "method",
        "seed",
        "domain_id",
        "condition",
        "scene",
        "sample_csv_sha256",
        "sample_count",
        "expected_sample_count",
        "coverage_status",
        "checkpoint_sha256",
        "checkpoint_role",
        "checkpoint_policy",
        "metric_profile",
        "eval_family",
        "pattern",
        "mask_type",
        "available_modalities",
        "drop_count",
        "mask_digest",
        "observed_missing_rate",
        "protocol_id",
        "protocol_kind",
        "protocol_manifest_sha256",
        "confirmation_split_manifest_sha256",
        "split_role",
        "training_role",
        "training_mask_seed",
        "training_mask_seed_algorithm",
        "smoke_preflight",
        "training_batch_size",
        "training_epochs",
        "evaluation_mask_cache_sha256",
        "evaluation_mask_cache_checksum",
        "topology_id",
        "topology_descriptor_sha256",
        "topology_mapping_sha256",
        "evaluation_topology_id",
        "evaluation_topology_descriptor_sha256",
        "reproduction_scope",
        "paper_equivalent",
        "temporal_result_scope",
        *METRICS,
    )
    missing = [field for field in required if not str(row.get(field, "")).strip()]
    if missing:
        raise ValueError(f"Missing strict evidence fields at {path}:{line}: {missing}.")
    if not any(str(row.get(field, "")).strip() for field in _RECIPE_ALIASES):
        raise ValueError(f"Missing recipe fingerprint at {path}:{line}.")
    if str(row.get("method", "")).strip() != method or _int_field(row, "seed", path, line) != seed:
        raise ValueError(f"Method/seed provenance mismatch at {path}:{line}.")
    if str(row.get("coverage_status", "")).strip() != "complete" or _truthy(row.get("partial_request")):
        raise ValueError(f"Partial strict evidence is not admissible: {path}:{line}.")
    digest = str(row.get("mask_digest", "")).strip()
    if not _is_sha256(digest):
        raise ValueError(f"Strict evidence requires a full 64-hex mask digest: {path}:{line}.")
    checkpoint_role = str(row.get("checkpoint_role", "")).strip()
    checkpoint_policy = str(row.get("checkpoint_policy", "")).strip()
    if checkpoint_role not in {"last", "fixed_epoch_last_pth"} or checkpoint_policy != "fixed_epoch_last_pth":
        raise ValueError(f"Strict evidence must use the fixed-epoch last checkpoint: {path}:{line}.")
    checkpoint = str(row.get("checkpoint", "")).strip()
    if checkpoint and Path(checkpoint).name != "last.pth":
        raise ValueError(f"Strict evidence checkpoint is not last.pth: {path}:{line}.")


def _validate_protocol_row(row: Mapping[str, str], path: Path, line: int, protocol: ProtocolEvidence) -> None:
    expected = {
        "protocol_id": PROTOCOL_ID,
        "protocol_kind": PROTOCOL_KIND,
        "protocol_manifest_sha256": protocol.manifest_sha256,
        "split_role": OUTER_ROLE,
        "training_role": TRAINING_ROLE,
        "evaluation_mask_cache_sha256": protocol.fixed_mask_cache_sha256,
        "evaluation_mask_cache_checksum": protocol.fixed_mask_cache_checksum,
    }
    for field, value in expected.items():
        if str(row.get(field, "")).strip() != value:
            raise ValueError(f"Strict protocol identity mismatch for {field} at {path}:{line}.")
    if _int_field(row, "training_mask_seed", path, line) != _int_field(row, "seed", path, line):
        raise ValueError(f"Training mask seed must equal the fixed experiment seed at {path}:{line}.")
    if _truthy(row.get("smoke_preflight")):
        raise ValueError(f"Smoke-preflight evidence is not admissible at {path}:{line}.")
    if _int_field(row, "training_batch_size", path, line) != 64 or _int_field(row, "training_epochs", path, line) != 40:
        raise ValueError(f"Strict evidence requires the frozen 40-epoch, batch-64 training recipe at {path}:{line}.")


def _validate_unit_provenance(
    provenance: Mapping[str, str], method: str, seed: int, protocol: ProtocolEvidence
) -> None:
    if provenance["protocol_id"] != PROTOCOL_ID or provenance["protocol_kind"] != PROTOCOL_KIND:
        raise ValueError(f"Incorrect confirmation protocol provenance for {method}/seed{seed}.")
    if provenance["protocol_manifest_sha256"] != protocol.manifest_sha256:
        raise ValueError(f"Protocol manifest SHA mismatch for {method}/seed{seed}.")
    if provenance["split_role"] != OUTER_ROLE or provenance["training_role"] != TRAINING_ROLE:
        raise ValueError(f"Split role provenance mismatch for {method}/seed{seed}.")
    if provenance["evaluation_mask_cache_sha256"] != protocol.fixed_mask_cache_sha256:
        raise ValueError(f"Mask-cache file SHA mismatch for {method}/seed{seed}.")
    if provenance["evaluation_mask_cache_checksum"] != protocol.fixed_mask_cache_checksum:
        raise ValueError(f"Mask-cache checksum mismatch for {method}/seed{seed}.")
    if provenance["checkpoint_role"] not in {"last", "fixed_epoch_last_pth"}:
        raise ValueError(f"Checkpoint role mismatch for {method}/seed{seed}.")
    if provenance["checkpoint_policy"] != "fixed_epoch_last_pth":
        raise ValueError(f"Checkpoint policy mismatch for {method}/seed{seed}.")
    if int(provenance["training_mask_seed"]) != seed:
        raise ValueError(f"Training mask seed mismatch for {method}/seed{seed}.")
    if _truthy(provenance["smoke_preflight"]):
        raise ValueError(f"Smoke-preflight evidence is not admissible for {method}/seed{seed}.")
    if int(provenance["training_batch_size"]) != 64 or int(provenance["training_epochs"]) != 40:
        raise ValueError(f"Training recipe budget mismatch for {method}/seed{seed}.")


def _validate_fidelity(fidelity: Mapping[str, str], method: str) -> None:
    required = ("reproduction_scope", "paper_equivalent", "temporal_result_scope")
    missing = [field for field in required if not fidelity.get(field, "")]
    if missing:
        raise ValueError(f"Missing baseline-fidelity metadata for {method}: {missing}.")
    if method not in {"masktrain_cls", "amber_full", "rmbp_mm", "amr_net_4m"}:
        return
    if _truthy(fidelity["paper_equivalent"]):
        raise ValueError(f"{method} is a local adaptation and must not be marked paper-equivalent.")
    if not fidelity.get("baseline_adaptation_scope", ""):
        raise ValueError(f"Missing baseline adaptation scope for {method}.")
    if not fidelity.get("missing_paper_components", ""):
        raise ValueError(f"Missing omitted paper components metadata for {method}.")


def _validate_cross_unit_identity(units: Mapping[tuple[str, int], EvalUnit], protocol: ProtocolEvidence) -> None:
    expected_units = {(method, seed) for method in METHODS for seed in SEEDS}
    if set(units) != expected_units:
        raise ValueError("Strict evidence matrix is missing one or more required method/seed units.")
    shared_fields = (
        "protocol_id",
        "protocol_kind",
        "protocol_manifest_sha256",
        "confirmation_split_manifest_sha256",
        "training_mask_seed_algorithm",
        "smoke_preflight",
        "training_batch_size",
        "training_epochs",
        "evaluation_mask_cache_sha256",
        "evaluation_mask_cache_checksum",
        "evaluation_topology_id",
        "evaluation_topology_descriptor_sha256",
        "metric_profile",
        "condition_identity_sha256",
    )
    for field in shared_fields:
        values = {unit.provenance[field] for unit in units.values()}
        if len(values) != 1:
            raise ValueError(f"Strict evidence has inconsistent cross-unit provenance for {field}: {sorted(values)}.")
    if next(iter(units.values())).provenance["protocol_manifest_sha256"] != protocol.manifest_sha256:
        raise ValueError("Cross-unit protocol manifest provenance does not match the frozen manifest.")
    expected_condition_identity = _protocol_condition_identity(protocol)
    if next(iter(units.values())).provenance["condition_identity_sha256"] != expected_condition_identity:
        raise ValueError("Cross-unit condition identity does not match the frozen 15-domain fixed-mask protocol.")

    expected_sample_counts: dict[str, tuple[int, int]] = {}
    for unit in units.values():
        for row in unit.rows:
            domain = str(row["domain_id"])
            count = (int(float(row["sample_count"])), int(float(row["expected_sample_count"])))
            previous = expected_sample_counts.setdefault(domain, count)
            if previous != count:
                raise ValueError(f"Cross-unit sample-count identity mismatch for domain {domain}.")


def _domain_main_cells(
    units: Mapping[tuple[str, int], EvalUnit], protocol: ProtocolEvidence
) -> list[dict[str, Any]]:
    selectors = _main_cell_selectors()
    result: list[dict[str, Any]] = []
    for (method, seed), unit in sorted(units.items()):
        by_domain: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in unit.rows:
            by_domain[str(row["domain_id"])].append(row)
        for domain_id, rows in sorted(by_domain.items()):
            metadata = protocol.domains[domain_id]
            for cell, selector in selectors.items():
                selected = [row for row in rows if selector(row)]
                if not selected:
                    raise ValueError(f"No rows for main cell {cell} in {method}/seed{seed}/{domain_id}.")
                result.append(
                    _aggregate_metric_rows(
                        selected,
                        {
                            "method": method,
                            "seed": seed,
                            "domain_id": domain_id,
                            "condition": metadata["condition"],
                            "scene": metadata["scene"],
                            "cell": cell,
                            "aggregation": "equal_weight_fixed_masks_within_domain",
                        },
                    )
                )
            auc_values = _temporal_auc_for_domain(rows)
            result.append(
                {
                    "method": method,
                    "seed": seed,
                    "domain_id": domain_id,
                    "condition": metadata["condition"],
                    "scene": metadata["scene"],
                    "cell": "TemporalAUC0_80",
                    "aggregation": "clean_plus_type_equal_temporal_rate_trapezoid",
                    "mask_count": "",
                    **auc_values,
                }
            )
    return result


def _main_cell_selectors():
    return {
        "Clean": lambda row: _is_whole(row, drop_count=0),
        "Drop1": lambda row: _is_whole(row, drop_count=1),
        "Drop2": lambda row: _is_whole(row, drop_count=2),
        "Drop3": lambda row: _is_whole(row, drop_count=3),
        "Block80": lambda row: _is_temporal(row, rate=0.8, mask_type="block"),
    }


def _temporal_auc_for_domain(rows: list[dict[str, str]]) -> dict[str, float]:
    rate_values: dict[float, dict[str, float]] = {}
    clean = [row for row in rows if _is_whole(row, drop_count=0)]
    if len(clean) != 1:
        raise ValueError("Temporal AUC requires exactly one clean whole-modality condition per domain.")
    rate_values[0.0] = {metric: float(clean[0][metric]) for metric in METRICS}
    for rate in TEMPORAL_RATES:
        by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if _is_temporal(row, rate=rate):
                by_type[str(row["mask_type"])].append(row)
        if set(by_type) != set(TEMPORAL_MASK_TYPES) or any(not values for values in by_type.values()):
            raise ValueError(f"Temporal AUC requires all three temporal mask types at rate={rate:g}.")
        rate_values[rate] = {
            metric: _mean([_mean([float(row[metric]) for row in by_type[mask_type]]) for mask_type in TEMPORAL_MASK_TYPES])
            for metric in METRICS
        }
    return {
        metric: _normalized_auc({rate: values[metric] for rate, values in rate_values.items()})
        for metric in METRICS
    }


def _temporal_curve_rows(
    units: Mapping[tuple[str, int], EvalUnit], protocol: ProtocolEvidence
) -> list[dict[str, Any]]:
    by_unit_domain: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for (method, seed), unit in units.items():
        for row in unit.rows:
            by_unit_domain[(method, seed, str(row["domain_id"]))].append(row)
    per_domain: list[dict[str, Any]] = []
    for (method, seed, domain_id), rows in sorted(by_unit_domain.items()):
        metadata = protocol.domains[domain_id]
        clean = [row for row in rows if _is_whole(row, drop_count=0)]
        per_domain.append(
            _aggregate_metric_rows(
                clean,
                {
                    "method": method,
                    "seed": seed,
                    "domain_id": domain_id,
                    "condition": metadata["condition"],
                    "scene": metadata["scene"],
                    "curve_kind": "type_equal_temporal",
                    "missing_rate": 0.0,
                    "aggregation": "clean_whole_modality",
                },
            )
        )
        for rate in TEMPORAL_RATES:
            by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                if _is_temporal(row, rate=rate):
                    by_type[str(row["mask_type"])].append(row)
            selected_type_means = []
            for mask_type in TEMPORAL_MASK_TYPES:
                items = by_type.get(mask_type, [])
                if not items:
                    raise ValueError(f"Missing temporal curve type {mask_type} rate={rate:g}.")
                selected_type_means.append(_aggregate_metric_rows(items, {}))
            entry: dict[str, Any] = {
                "method": method,
                "seed": seed,
                "domain_id": domain_id,
                "condition": metadata["condition"],
                "scene": metadata["scene"],
                "curve_kind": "type_equal_temporal",
                "missing_rate": rate,
                "aggregation": "equal_weight_mask_type_after_within_type_masks",
                "mask_count": sum(int(item["mask_count"]) for item in selected_type_means),
            }
            for metric in METRICS:
                entry[metric] = _mean([float(item[metric]) for item in selected_type_means])
            per_domain.append(entry)
    return _macro_from_domain_rows(per_domain, group_fields=("method", "seed", "curve_kind", "missing_rate"))


def _domain_macro_cells(domain_cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _macro_from_domain_rows(domain_cells, group_fields=("method", "seed", "cell"))


def _group_macro_rows(domain_cells: list[dict[str, Any]], group_field: str) -> list[dict[str, Any]]:
    return _macro_from_domain_rows(domain_cells, group_fields=("method", "seed", "cell", group_field))


def _macro_from_domain_rows(
    rows: list[dict[str, Any]], *, group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    result: list[dict[str, Any]] = []
    for key, selected in sorted(grouped.items()):
        base = dict(zip(group_fields, key))
        result.append(
            {
                **base,
                "domain_count": len(selected),
                "aggregation": "domain_macro_equal_weight",
                "mask_count": _mean([_numeric_or_zero(row.get("mask_count")) for row in selected]),
                **{metric: _mean([float(row[metric]) for row in selected]) for metric in METRICS},
            }
        )
    return result


def _worst_domain_rows(domain_cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in domain_cells:
        grouped[(str(row["method"]), int(row["seed"]), str(row["cell"]))].append(row)
    result = []
    for (method, seed, cell), selected in sorted(grouped.items()):
        worst = min(selected, key=lambda row: (float(row[PRIMARY_METRIC]), str(row["domain_id"])))
        result.append(
            {
                "method": method,
                "seed": seed,
                "cell": cell,
                "domain_count": len(selected),
                "worst_domain_id": worst["domain_id"],
                "worst_condition": worst["condition"],
                "worst_scene": worst["scene"],
                "aggregation": "minimum_adba_domain_within_seed_cell",
                **{metric: float(worst[metric]) for metric in METRICS},
            }
        )
    return result


def _seed_aggregate_rows(
    rows: list[dict[str, Any]], *, group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    result: list[dict[str, Any]] = []
    for key, selected in sorted(grouped.items()):
        seeds = {int(row["seed"]) for row in selected}
        if seeds != set(SEEDS):
            raise ValueError(f"Cannot aggregate incomplete fixed seed set for {key}: {sorted(seeds)}.")
        if len(selected) != len(SEEDS):
            raise ValueError(f"Expected one row per seed for aggregate {key}, found {len(selected)}.")
        result.append(
            {
                **dict(zip(group_fields, key)),
                "requested_seed_count": len(SEEDS),
                "available_seed_count": len(selected),
                "aggregation_status": "complete",
                "aggregation": "equal_weight_fixed_seed_mean",
                **{
                    f"{metric}_mean": _mean([float(row[metric]) for row in selected])
                    for metric in METRICS
                },
                **{
                    f"{metric}_std": _sample_std([float(row[metric]) for row in selected])
                    for metric in METRICS
                },
            }
        )
    return result


def _paired_statistics(
    domain_cells: list[dict[str, Any]],
    *,
    methods: Iterable[str],
    seeds: Iterable[int],
    bootstrap_iterations: int,
    bootstrap_seed: int,
    confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values = {
        (str(row["method"]), int(row["seed"]), str(row["domain_id"]), str(row["cell"])): row
        for row in domain_cells
    }
    comparators = tuple(method for method in methods if method != "T2")
    expected_domains = {str(row["domain_id"]) for row in domain_cells if row["method"] == "T2" and row["seed"] == 1}
    paired_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    cells = sorted({str(row["cell"]) for row in domain_cells})
    for comparator in comparators:
        for cell in cells:
            values_by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for seed in seeds:
                for domain_id in sorted(expected_domains):
                    t2 = values.get(("T2", int(seed), domain_id, cell))
                    other = values.get((comparator, int(seed), domain_id, cell))
                    if t2 is None or other is None:
                        raise ValueError(
                            f"Missing paired domain cell T2/{comparator}/seed{seed}/{domain_id}/{cell}."
                        )
                    for metric in METRICS:
                        delta = float(t2[metric]) - float(other[metric])
                        entry = {
                            "treatment": "T2",
                            "comparator": comparator,
                            "seed": int(seed),
                            "domain_id": domain_id,
                            "cell": cell,
                            "metric": metric,
                            "preferred_direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                            "t2_value": float(t2[metric]),
                            "comparator_value": float(other[metric]),
                            "delta_t2_minus_comparator": delta,
                            "comparison_scope": _comparison_scope(comparator),
                        }
                        paired_rows.append(entry)
                        values_by_metric[metric].append(entry)
            for metric, entries in sorted(values_by_metric.items()):
                ci_low, ci_high = _paired_domain_seed_bootstrap(
                    entries,
                    seeds=tuple(seeds),
                    expected_domains=expected_domains,
                    iterations=bootstrap_iterations,
                    seed=_stable_seed(bootstrap_seed, comparator, cell, metric),
                    confidence=confidence,
                )
                seed_means = [
                    _mean(
                        [
                            float(item["delta_t2_minus_comparator"])
                            for item in entries
                            if int(item["seed"]) == int(seed)
                        ]
                    )
                    for seed in seeds
                ]
                ci_rows.append(
                    {
                        "treatment": "T2",
                        "comparator": comparator,
                        "cell": cell,
                        "metric": metric,
                        "preferred_direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                        "paired_seed_count": len(tuple(seeds)),
                        "paired_domain_count_per_seed": len(expected_domains),
                        "paired_domain_seed_count": len(entries),
                        "mean_delta_t2_minus_comparator": _mean(
                            [float(item["delta_t2_minus_comparator"]) for item in entries]
                        ),
                        "seed_delta_std": _sample_std(seed_means),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "confidence": confidence,
                        "bootstrap_iterations": bootstrap_iterations,
                        "bootstrap_unit": "paired_domains_within_fixed_seed_then_equal_seed_weight",
                        "comparison_scope": _comparison_scope(comparator),
                    }
                )
    return paired_rows, ci_rows


def _comparison_scope(comparator: str) -> str:
    del comparator
    return "main_method_comparison"


def _paired_domain_seed_bootstrap(
    rows: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    expected_domains: set[str],
    iterations: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        numeric_seed = int(row["seed"])
        domain = str(row["domain_id"])
        if domain in by_seed[numeric_seed]:
            raise ValueError("Duplicate paired domain/seed bootstrap unit.")
        by_seed[numeric_seed][domain] = float(row["delta_t2_minus_comparator"])
    if set(by_seed) != set(seeds) or any(set(by_seed[item]) != expected_domains for item in seeds):
        raise ValueError("Paired bootstrap requires complete identical domain coverage for every fixed seed.")
    rng = random.Random(seed)
    draws: list[float] = []
    ordered_domains = tuple(sorted(expected_domains))
    for _ in range(iterations):
        seed_means = []
        for fixed_seed in seeds:
            values = by_seed[fixed_seed]
            seed_means.append(_mean([values[rng.choice(ordered_domains)] for _ in ordered_domains]))
        draws.append(_mean(seed_means))
    alpha = (1.0 - confidence) / 2.0
    return _quantile(draws, alpha), _quantile(draws, 1.0 - alpha)


def _coverage_rows(units: Mapping[tuple[str, int], EvalUnit], protocol: ProtocolEvidence) -> list[dict[str, Any]]:
    expected_rows = len(protocol.domains) * len(protocol.conditions)
    result = []
    for (method, seed), unit in sorted(units.items()):
        result.append(
            {
                "method": method,
                "seed": seed,
                "status": "complete",
                "domain_count": len(protocol.domains),
                "condition_count": len(protocol.conditions),
                "row_count": len(unit.rows),
                "expected_row_count": expected_rows,
                "metrics_path": str(unit.path.resolve()),
                **unit.provenance,
            }
        )
    return result


def _fidelity_rows(units: Mapping[tuple[str, int], EvalUnit]) -> list[dict[str, Any]]:
    grouped: dict[str, list[EvalUnit]] = defaultdict(list)
    for unit in units.values():
        grouped[unit.method].append(unit)
    result = []
    for method, selected in sorted(grouped.items()):
        values: dict[str, set[str]] = defaultdict(set)
        for unit in selected:
            for key, value in unit.fidelity.items():
                values[key].add(value)
        metadata = _singleton_values(values, f"cross-seed fidelity metadata {method}", allow_empty=True)
        result.append(
            {
                "method": method,
                "seed_count": len(selected),
                "baseline_fidelity_status": "complete",
                **metadata,
            }
        )
    return result


def _summary_provenance(
    protocol: ProtocolEvidence,
    *,
    eval_dir: Path,
    methods: Iterable[str],
    seeds: Iterable[int],
    bootstrap_iterations: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any]:
    request = {
        "protocol_id": PROTOCOL_ID,
        "protocol_kind": PROTOCOL_KIND,
        "protocol_manifest": str(protocol.path.resolve()),
        "protocol_manifest_sha256": protocol.manifest_sha256,
        "evaluation_mask_cache_sha256": protocol.fixed_mask_cache_sha256,
        "evaluation_mask_cache_checksum": protocol.fixed_mask_cache_checksum,
        "eval_dir": str(eval_dir.resolve()),
        "methods": list(methods),
        "main_comparison_methods": list(MAIN_COMPARISON_METHODS),
        "seeds": list(seeds),
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "confidence": float(confidence),
        "primary_metric": PRIMARY_METRIC,
        "secondary_metric": SECONDARY_METRIC,
        "adba_definition": "progressive_top3_minimum_circular_beam_distance",
        "adba_delta": ADBA_DELTA,
        "adba_distance_mode": ADBA_DISTANCE_MODE,
    }
    return {
        "schema_version": 2,
        "status": "complete",
        "claim_boundary": "post-selection confirmation evidence; historical h5p1_strict_v2 test was not reused",
        "request": request,
        "request_sha256": _sha256_payload(request),
        "statistics": {
            "domain_aggregation": "equal_weight_15_domain_macro_within_each_seed",
            "seed_aggregation": "equal_weight_fixed_seed_mean",
            "paired_ci": "paired_domains_within_fixed_seed_then_equal_seed_weight",
        },
    }


def _plot_manifest(provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "data_ready",
        "protocol_id": provenance["request"]["protocol_id"],
        "protocol_kind": provenance["request"]["protocol_kind"],
        "provenance_request_sha256": provenance["request_sha256"],
        "figures": [
            {
                "id": "main_cells_adba",
                "source_csv": "paper_main_table.csv",
                "kind": "grouped_table_or_bar",
                "x": "cell",
                "y": "adba_mean",
                "error": "adba_std",
                "series": "method",
            },
            {
                "id": "temporal_robustness_curve",
                "source_csv": "paper_temporal_curve.csv",
                "kind": "line_with_seed_std",
                "x": "missing_rate",
                "y": "adba_mean",
                "error": "adba_std",
                "series": "method",
            },
            {
                "id": "weather_main_cells",
                "source_csv": "paper_weather_table.csv",
                "kind": "weather_faceted_table_or_bar",
                "x": "cell",
                "y": "adba_mean",
                "series": "method",
                "facet": "condition",
            },
            {
                "id": "paired_t2_contrasts",
                "source_csv": "paired_bootstrap_ci.csv",
                "kind": "forest",
                "x": "mean_delta_t2_minus_comparator",
                "ci_low": "ci_low",
                "ci_high": "ci_high",
                "series": "comparator",
                "filter": {"metric": "adba"},
            },
            {
                "id": "worst_domain_adba",
                "source_csv": "paper_worst_domain_table.csv",
                "kind": "table",
                "x": "cell",
                "y": "adba_mean",
                "series": "method",
            },
            {
                "id": "main_cells_top1_secondary",
                "source_csv": "paper_main_table.csv",
                "kind": "grouped_table_or_bar",
                "x": "cell",
                "y": "top1_mean",
                "error": "top1_std",
                "series": "method",
            },
        ],
    }


def _markdown(
    provenance: Mapping[str, Any],
    paper_table: list[dict[str, Any]],
    paired_ci: list[dict[str, Any]],
    fidelity: list[dict[str, Any]],
) -> str:
    lines = [
        "# MMW TWC Post-selection Confirmation Summary",
        "",
        "This report is derived only from the frozen post-selection confirmation fold. "
        "The historical h5p1_strict_v2 test was not reused.",
        "",
        "## Protocol",
        "",
        f"- Protocol: `{provenance['request']['protocol_id']}`",
        f"- Kind: `{provenance['request']['protocol_kind']}`",
        f"- Fixed seeds: `{','.join(str(seed) for seed in provenance['request']['seeds'])}`",
        f"- Fixed mask checksum: `{provenance['request']['evaluation_mask_cache_checksum']}`",
        "- Primary metric: circular progressive Top-3 ADBA (delta=5); Top-1 is secondary.",
        "- Domain macro: equal weight across 15 weather x scene domains within each seed, then equal weight across seeds.",
        "- Paired CI: resample domains within each fixed seed; seeds retain equal weight.",
        "",
        "## Main ADBA",
        "",
        "| Method | Clean | Drop1 | Drop2 | Drop3 | Block80 | Temporal AUC 0-80 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    indexed = {(str(row["method"]), str(row["cell"])): row for row in paper_table}
    cells = ("Clean", "Drop1", "Drop2", "Drop3", "Block80", "TemporalAUC0_80")
    for method in MAIN_COMPARISON_METHODS:
        values = [
            _mean_std(indexed.get((method, cell), {}).get("adba_mean"), indexed.get((method, cell), {}).get("adba_std"))
            for cell in cells
        ]
        lines.append(f"| {_display_name(method)} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Paired ADBA Contrasts",
            "",
            "| Contrast | Cell | Delta (T2 - comparator) | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for row in paired_ci:
        if row["metric"] != "adba":
            continue
        lines.append(
            f"| T2 - {row['comparator']} | {row['cell']} | "
            f"{_fmt(row['mean_delta_t2_minus_comparator'])} | "
            f"[{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}] |"
        )
    lines.extend(
        [
            "",
            "## Secondary Top-1",
            "",
            "| Method | Clean | Drop1 | Drop2 | Drop3 | Block80 | Temporal AUC 0-80 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in MAIN_COMPARISON_METHODS:
        values = [
            _mean_std(indexed.get((method, cell), {}).get("top1_mean"), indexed.get((method, cell), {}).get("top1_std"))
            for cell in cells
        ]
        lines.append(f"| {_display_name(method)} | " + " | ".join(values) + " |")
    lines.extend(["", "## Baseline Fidelity", "", "| Method | Paper equivalent | Scope | Missing paper components |", "|---|---|---|---|"])
    for row in fidelity:
        lines.append(
            f"| {_display_name(str(row['method']))} | {row.get('paper_equivalent', '')} | {row.get('reproduction_scope', '')} | "
            f"{row.get('missing_paper_components', '')} |"
        )
    lines.extend(
        [
            "",
            "All detailed domain, weather, scene, worst-domain, fixed-mask, fidelity, and paired-statistics data are emitted as CSV files beside this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def _display_name(method: str) -> str:
    return method


def _aggregate_metric_rows(rows: list[Mapping[str, Any]], base: Mapping[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate an empty fixed-mask cell.")
    return {
        **base,
        "mask_count": len(rows),
        **{metric: _mean([float(row[metric]) for row in rows]) for metric in METRICS},
    }


def _condition_key(item: Mapping[str, Any], *, source: str) -> tuple[Any, ...]:
    family = _required_text(item, "eval_family" if "eval_family" in item else "family", source)
    if "family" in item and "eval_family" in item and str(item["family"]).strip() != family:
        raise ValueError(f"family and eval_family disagree in {source}.")
    pattern = _required_text(item, "pattern", source)
    mask_type = _required_text(item, "mask_type", source)
    available = _modalities(item.get("available_modalities"), source)
    drop_count = _coerce_int(item.get("drop_count"), f"drop_count in {source}")
    digest = _required_text(item, "mask_digest", source)
    rate = _requested_rate(item, source)
    return family, pattern, mask_type, rate, available, drop_count, digest


def _condition_identity_from_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    entries = [
        {
            "domain_id": str(row["domain_id"]).strip(),
            "sample_csv_sha256": str(row["sample_csv_sha256"]).strip(),
            "expected_sample_count": _coerce_int(row["expected_sample_count"], "row expected_sample_count"),
            "condition": _condition_key(row, source="evidence condition identity"),
        }
        for row in rows
    ]
    entries.sort(key=lambda item: (item["domain_id"], json.dumps(item["condition"], separators=(",", ":"))))
    return _sha256_payload({"entries": entries})


def _protocol_condition_identity(protocol: ProtocolEvidence) -> str:
    entries = [
        {
            "domain_id": domain_id,
            "sample_csv_sha256": metadata["sample_csv_sha256"],
            "expected_sample_count": int(metadata["expected_sample_count"]),
            "condition": condition_key,
        }
        for domain_id, metadata in protocol.domains.items()
        for condition_key in protocol.conditions
    ]
    entries.sort(key=lambda item: (item["domain_id"], json.dumps(item["condition"], separators=(",", ":"))))
    return _sha256_payload({"entries": entries})


def _requested_rate(item: Mapping[str, Any], source: str) -> float:
    requested_raw = item.get("requested_missing_rate")
    missing_raw = item.get("missing_rate")
    if requested_raw in {None, ""} and missing_raw in {None, ""}:
        raise ValueError(f"Missing requested/missing rate in {source}.")
    requested = _coerce_rate(requested_raw if requested_raw not in {None, ""} else missing_raw, source)
    if requested_raw not in {None, ""} and missing_raw not in {None, ""}:
        missing = _coerce_rate(missing_raw, source)
        if not _close(requested, missing):
            raise ValueError(f"requested_missing_rate and missing_rate disagree in {source}.")
    return requested


def _is_whole(item: Mapping[str, Any], *, drop_count: int) -> bool:
    try:
        family = str(item.get("eval_family", item.get("family", ""))).strip()
        return family == "whole_modality" and _coerce_int(item.get("drop_count"), "drop_count") == drop_count
    except ValueError:
        return False


def _is_temporal(item: Mapping[str, Any], *, rate: float, mask_type: str | None = None) -> bool:
    try:
        family = str(item.get("eval_family", item.get("family", ""))).strip()
        return (
            family == "temporal_missing"
            and _close(_requested_rate(item, "temporal selector"), rate)
            and (mask_type is None or str(item.get("mask_type", "")).strip() == mask_type)
        )
    except ValueError:
        return False


def _observed_missing_rate(item: Mapping[str, Any]) -> float:
    return _coerce_rate(item.get("observed_missing_rate"), "observed_missing_rate")


def _row_alias_value(row: Mapping[str, str], field: str, path: Path, line: int) -> str:
    if field == "recipe_fingerprint":
        for alias in _RECIPE_ALIASES:
            value = str(row.get(alias, "")).strip()
            if value:
                return value
        raise ValueError(f"Missing recipe fingerprint at {path}:{line}.")
    value = str(row.get(field, "")).strip()
    if not value:
        raise ValueError(f"Missing provenance field {field} at {path}:{line}.")
    return value


def _fidelity_value(row: Mapping[str, str], field: str) -> str:
    aliases = _FIDELITY_ALIASES.get(field, (field,))
    if field == "missing_paper_components":
        direct = str(row.get("missing_paper_components", "")).strip()
        if direct:
            return direct
        parts = [
            str(row.get("omitted_paper_inputs", "")).strip(),
            str(row.get("omitted_paper_training_stages", "")).strip(),
            str(row.get("omitted_paper_inputs_json", "")).strip(),
            str(row.get("omitted_paper_training_stages_json", "")).strip(),
        ]
        return "; ".join(part for part in parts if part)
    for alias in aliases:
        value = str(row.get(alias, "")).strip()
        if value:
            return value
    return ""


def _singleton_values(
    values: Mapping[str, set[str]], label: str, *, allow_empty: bool = False
) -> dict[str, str]:
    result: dict[str, str] = {}
    for field, field_values in values.items():
        nonempty = {value for value in field_values if value}
        if len(nonempty) > 1:
            raise ValueError(f"Inconsistent {label} for {field}: {sorted(nonempty)}.")
        if not nonempty:
            if not allow_empty:
                raise ValueError(f"Missing {label} for {field}.")
            result[field] = ""
        else:
            result[field] = next(iter(nonempty))
    return result


def _prepare_output_dir(output_dir: Path, provenance: Mapping[str, Any]) -> None:
    if output_dir.exists():
        existing = output_dir / "provenance.json"
        if existing.is_file():
            previous = _read_json(existing)
            if previous.get("request_sha256") != provenance.get("request_sha256"):
                raise ValueError(
                    f"Refusing to overwrite a summary produced from another immutable request: {output_dir}."
                )
    output_dir.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be a mapping: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Metrics CSV has no header: {path}")
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    temporary.replace(path)


def _resolve_artifact_path(parent: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (parent / path).resolve()


def _required_text(item: Mapping[str, Any], field: str, label: str) -> str:
    value = str(item.get(field, "")).strip()
    if not value:
        raise ValueError(f"Missing {field} in {label}.")
    return value


def _modalities(value: Any, source: str) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        items = tuple(str(item).strip() for item in value if str(item).strip())
    else:
        items = tuple(item.strip() for item in str(value or "").split(",") if item.strip())
    if not items or len(set(items)) != len(items):
        raise ValueError(f"Invalid available_modalities in {source}.")
    return items


def _coerce_rate(value: Any, source: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid missing rate in {source}: {value!r}.") from exc
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"Invalid missing rate in {source}: {value!r}.")
    return round(number, 6)


def _coerce_int(value: Any, source: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer {source}: {value!r}.") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"Invalid integer {source}: {value!r}.")
    return int(number)


def _int_field(row: Mapping[str, str], field: str, path: Path, line: int) -> int:
    return _coerce_int(row.get(field), f"{field} at {path}:{line}")


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_seed(base: int, *parts: str) -> int:
    text = ":".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("Cannot compute a mean of no values.")
    return sum(values) / len(values)


def _sample_std(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _normalized_auc(rate_values: Mapping[float, float]) -> float:
    rates = sorted(rate_values)
    if rates != [0.0, *TEMPORAL_RATES]:
        raise ValueError(f"Temporal AUC rate inventory mismatch: {rates}.")
    area = sum(
        (right - left) * (rate_values[left] + rate_values[right]) / 2.0
        for left, right in zip(rates, rates[1:])
    )
    return area / (rates[-1] - rates[0])


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _numeric_or_zero(value: Any) -> float:
    numeric = _finite_float(value)
    return numeric if numeric is not None else 0.0


def _close(first: float, second: float, tolerance: float = 1.0e-9) -> bool:
    return abs(float(first) - float(second)) <= tolerance


def _csv_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _csv_values(value))


def _fmt(value: Any) -> str:
    numeric = _finite_float(value)
    return f"{100.0 * numeric:.2f}%" if numeric is not None else "-"


def _mean_std(mean: Any, std: Any) -> str:
    numeric_mean = _finite_float(mean)
    numeric_std = _finite_float(std)
    if numeric_mean is None or numeric_std is None:
        return "-"
    return f"{100.0 * numeric_mean:.2f}% +/- {100.0 * numeric_std:.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
