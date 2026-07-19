#!/usr/bin/env python3
"""Strict summary for balanced 20-cell MMW temporal token-stress evidence."""

from __future__ import annotations

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

from kd_sensing.data.mmw.twc_temporal_token_stress import (
    STRESS_PROTOCOL_ID,
    STRESS_RATES,
    load_temporal_token_stress_protocol,
)


METHODS = ("T2", "S1", "amber_full", "rmbp_mm")
MAIN_COMPARISON_METHODS = METHODS
SEEDS = (1, 2, 3, 4, 5)
METRICS = ("top1", "top3", "top5", "within_1", "within_3", "adba", "mae")
HIGHER_IS_BETTER = frozenset(("top1", "top3", "top5", "within_1", "within_3"))
MAIN_RATES = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9)
EXTREME_RATE = 0.95


@dataclass
class EvalUnit:
    method: str
    seed: int
    path: Path
    rows: list[dict[str, str]]
    provenance: dict[str, str]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument(
        "--evaluation-extension-manifest",
        default="outputs/cache/mmw_twc_temporal_token_stress_v3/protocol_manifest.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    try:
        summarize(
            Path(args.eval_dir),
            Path(args.evaluation_extension_manifest),
            Path(args.output_dir),
            bootstrap_iterations=int(args.bootstrap_iterations),
            bootstrap_seed=int(args.bootstrap_seed),
            confidence=float(args.confidence),
        )
    except Exception as exc:  # noqa: BLE001
        parser.error(f"{type(exc).__name__}: {exc}")
    return 0


def summarize(
    eval_dir: Path,
    extension_manifest: Path,
    output_dir: Path,
    *,
    bootstrap_iterations: int = 10000,
    bootstrap_seed: int = 20260720,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if bootstrap_iterations <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Bootstrap iterations must be positive and confidence must be in (0, 1).")
    extension = load_temporal_token_stress_protocol(extension_manifest)
    cache = _load_cache(extension)
    conditions = _conditions(cache)
    units = _load_units(Path(eval_dir), extension, cache, conditions)
    _validate_cross_unit_identity(units, extension, cache)
    domain_rows = _domain_rate_rows(units, conditions)
    domain_macro = _domain_macro(domain_rows)
    paper_table = _seed_aggregate(
        [
            row
            for row in domain_macro
            if row["method"] in MAIN_COMPARISON_METHODS and row["cell"] != "SingleCell95"
        ],
        group_fields=("method", "cell", "missing_rate"),
    )
    extreme_table = _seed_aggregate(
        [
            row
            for row in domain_macro
            if row["method"] in MAIN_COMPARISON_METHODS and row["cell"] == "SingleCell95"
        ],
        group_fields=("method", "cell", "missing_rate"),
    )
    curve_table = _seed_aggregate(
        [
            row
            for row in domain_macro
            if row["method"] in MAIN_COMPARISON_METHODS and row["cell"] != "TokenAUC0_90"
        ],
        group_fields=("method", "missing_rate", "cell"),
    )
    paired_rows, paired_ci = _paired_statistics(
        domain_rows,
        comparators=MAIN_COMPARISON_METHODS[1:],
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    coverage = _coverage_rows(units, extension, cache)
    condition_catalog = _condition_catalog(cache)
    balance_rows = _balance_rows(cache)
    provenance = _provenance(
        extension,
        cache=cache,
        eval_dir=Path(eval_dir),
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    _prepare_output(output_dir, provenance)
    _write_csv(output_dir / "coverage.csv", coverage)
    _write_csv(output_dir / "condition_catalog.csv", condition_catalog)
    _write_csv(output_dir / "mask_balance_audit.csv", balance_rows)
    _write_csv(output_dir / "domain_token_stress_per_seed.csv", domain_rows)
    _write_csv(output_dir / "domain_macro_per_seed.csv", domain_macro)
    _write_csv(output_dir / "paper_temporal_token_stress_table.csv", paper_table)
    _write_csv(output_dir / "paper_single_cell_95_table.csv", extreme_table)
    _write_csv(output_dir / "paper_temporal_token_stress_curve.csv", curve_table)
    _write_csv(output_dir / "paired_domain_seed_deltas.csv", paired_rows)
    _write_csv(output_dir / "paired_bootstrap_ci.csv", paired_ci)
    _write_json(output_dir / "provenance.json", provenance)
    _write_json(output_dir / "plot_manifest.json", _plot_manifest(provenance))
    _write_text(
        output_dir / "summary.md",
        _markdown(
            provenance,
            paper_table,
            extreme_table,
            paired_ci,
        ),
    )
    _write_text(output_dir / "reproducibility.md", _reproducibility_markdown(provenance, extension, cache))
    return {
        "extension": extension,
        "units": units,
        "coverage": coverage,
        "paper_table": paper_table,
        "extreme_table": extreme_table,
        "paired_ci": paired_ci,
        "provenance": provenance,
    }


def _load_cache(extension: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(extension["fixed_mask_cache"]["resolved_path"])).resolve()
    cache = _read_json(path)
    if _sha256_file(path) != str(extension["fixed_mask_cache"]["sha256"]):
        raise ValueError("Temporal-token stress cache file SHA256 mismatch.")
    if str(cache.get("checksum", "")) != str(extension["fixed_mask_cache"]["cache_checksum"]):
        raise ValueError("Temporal-token stress cache checksum mismatch.")
    return cache


def _conditions(cache: Mapping[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result = {}
    for condition in cache["conditions"]:
        key = _condition_key(condition)
        if key in result:
            raise ValueError(f"Duplicate temporal-token stress condition: {key}.")
        result[key] = dict(condition)
    return result


def _load_units(
    eval_dir: Path,
    extension: Mapping[str, Any],
    cache: Mapping[str, Any],
    conditions: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[tuple[str, int], EvalUnit]:
    units = {}
    for method in METHODS:
        for seed in SEEDS:
            path = eval_dir / method / f"seed{seed}" / "metrics.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing temporal-token stress metrics CSV: {path}")
            rows = _read_csv(path)
            provenance = _validate_unit(rows, path, method, seed, extension, cache, conditions)
            units[(method, seed)] = EvalUnit(method=method, seed=seed, path=path, rows=rows, provenance=provenance)
    return units


def _validate_unit(
    rows: list[dict[str, str]],
    path: Path,
    method: str,
    seed: int,
    extension: Mapping[str, Any],
    cache: Mapping[str, Any],
    conditions: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, str]:
    domains = {str(item["id"]): item for item in extension["domains"]}
    expected_count = len(domains) * len(conditions)
    if len(rows) != expected_count:
        raise ValueError(f"Temporal-token stress row count mismatch for {method}/seed{seed}: {len(rows)} != {expected_count}.")
    seen = set()
    shared: dict[str, set[str]] = defaultdict(set)
    for line, row in enumerate(rows, start=2):
        _validate_common_row(row, path, line, method, seed)
        domain_id = str(row["domain_id"])
        domain = domains.get(domain_id)
        if domain is None:
            raise ValueError(f"Unexpected domain {domain_id} at {path}:{line}.")
        outer = domain["split"]["outer_evidence"]
        if str(row["sample_csv_sha256"]) != str(outer["sha256"]):
            raise ValueError(f"Outer split SHA256 mismatch at {path}:{line}.")
        if _as_int(row["expected_sample_count"], path, line) != int(outer["row_count"]):
            raise ValueError(f"Outer split sample count mismatch at {path}:{line}.")
        key = _condition_key(row)
        expected = conditions.get(key)
        if expected is None:
            raise ValueError(f"Unexpected temporal-token stress condition at {path}:{line}: {key}.")
        identity = (domain_id, key)
        if identity in seen:
            raise ValueError(f"Duplicate temporal-token stress domain/condition at {path}:{line}.")
        seen.add(identity)
        _validate_condition_row(row, expected, path, line)
        for field in _UNIT_PROVENANCE_FIELDS:
            shared[field].add(str(row[field]))
    if len(seen) != expected_count:
        raise ValueError(f"Incomplete temporal-token stress condition coverage for {method}/seed{seed}.")
    for domain_id in domains:
        observed = {key for observed_domain, key in seen if observed_domain == domain_id}
        if observed != set(conditions):
            raise ValueError(f"Incomplete temporal-token stress cache coverage for {method}/seed{seed}/{domain_id}.")
    provenance = _singleton_values(shared, f"temporal-token stress provenance {method}/seed{seed}")
    _validate_provenance(provenance, extension, cache, seed)
    provenance["condition_identity_sha256"] = _condition_identity(rows)
    return provenance


_SHARED_PROVENANCE_FIELDS = (
    "protocol_id",
    "protocol_manifest_sha256",
    "evaluation_extension_id",
    "evaluation_extension_kind",
    "evaluation_extension_manifest_sha256",
    "evaluation_extension_parent_protocol_id",
    "evaluation_extension_parent_protocol_manifest_sha256",
    "evaluation_mask_cache_sha256",
    "evaluation_mask_cache_checksum",
    "evaluation_extension_token_count",
    "evaluation_extension_mask_type",
    "evaluation_extension_rates_json",
    "evaluation_extension_masks_per_rate",
    "evaluation_extension_single_cell_mask_count",
    "evaluation_extension_per_rate_mask_counts_json",
    "evaluation_extension_rate_balance_audit_json",
    "evaluation_extension_balance_policy",
    "training_batch_size",
    "training_epochs",
    "checkpoint_role",
    "checkpoint_policy",
    "metric_profile",
)
_UNIT_PROVENANCE_FIELDS = (
    *_SHARED_PROVENANCE_FIELDS,
    "training_mask_seed",
    "training_mask_seed_algorithm",
    "checkpoint_sha256",
    "config_recipe_sha256",
)


def _validate_common_row(row: Mapping[str, str], path: Path, line: int, method: str, seed: int) -> None:
    required = (
        "method",
        "seed",
        "domain_id",
        "sample_csv_sha256",
        "expected_sample_count",
        "sample_count",
        "coverage_status",
        "eval_family",
        "pattern",
        "mask_type",
        "mask_digest",
        "requested_missing_rate",
        "observed_missing_rate",
        "mask_matrix_json",
        "token_count",
        "retained_token_count",
        "dropped_token_count",
        "per_modality_retained_counts_json",
        "per_modality_dropped_counts_json",
        "per_frame_retained_counts_json",
        "per_frame_dropped_counts_json",
        "mask_set_index",
        "mask_set_size",
        "mask_balance_policy",
        *_UNIT_PROVENANCE_FIELDS,
        *METRICS,
    )
    missing = [field for field in required if not str(row.get(field, "")).strip()]
    if missing:
        raise ValueError(f"Missing temporal-token stress fields at {path}:{line}: {missing}.")
    if str(row["method"]) != method or _as_int(row["seed"], path, line) != seed:
        raise ValueError(f"Method/seed mismatch at {path}:{line}.")
    if str(row["coverage_status"]) != "complete" or _truthy(row.get("partial_request")):
        raise ValueError(f"Partial temporal-token stress evidence is inadmissible at {path}:{line}.")
    if _as_int(row["sample_count"], path, line) != _as_int(row["expected_sample_count"], path, line):
        raise ValueError(f"Incomplete temporal-token stress sample coverage at {path}:{line}.")
    for metric in METRICS:
        if _finite_float(row[metric]) is None:
            raise ValueError(f"Missing or non-finite metric {metric} at {path}:{line}.")


def _validate_condition_row(row: Mapping[str, str], expected: Mapping[str, Any], path: Path, line: int) -> None:
    comparisons = {
        "eval_family": "family",
        "pattern": "pattern",
        "mask_type": "mask_type",
        "mask_digest": "mask_digest",
        "mask_balance_policy": "mask_balance_policy",
    }
    for row_field, condition_field in comparisons.items():
        if str(row[row_field]) != str(expected[condition_field]):
            raise ValueError(f"Temporal-token stress {row_field} mismatch at {path}:{line}.")
    for field in ("token_count", "retained_token_count", "dropped_token_count", "mask_set_index", "mask_set_size"):
        if _as_int(row[field], path, line) != int(expected[field]):
            raise ValueError(f"Temporal-token stress {field} mismatch at {path}:{line}.")
    for field in ("requested_missing_rate", "observed_missing_rate"):
        if not _close(float(row[field]), float(expected[field])):
            raise ValueError(f"Temporal-token stress {field} mismatch at {path}:{line}.")
    matrix = json.loads(str(row["mask_matrix_json"]))
    if matrix != expected["modality_temporal_mask"]:
        raise ValueError(f"Temporal-token stress mask matrix mismatch at {path}:{line}.")
    json_fields = {
        "per_modality_retained_counts_json": "per_modality_retained_counts",
        "per_modality_dropped_counts_json": "per_modality_dropped_counts",
        "per_frame_retained_counts_json": "per_frame_retained_counts",
        "per_frame_dropped_counts_json": "per_frame_dropped_counts",
    }
    for row_field, condition_field in json_fields.items():
        if json.loads(str(row[row_field])) != expected[condition_field]:
            raise ValueError(f"Temporal-token stress {row_field} mismatch at {path}:{line}.")


def _validate_provenance(provenance: Mapping[str, str], extension: Mapping[str, Any], cache: Mapping[str, Any], seed: int) -> None:
    parent = extension["parent_training_protocol"]
    expected = {
        "protocol_id": str(parent["protocol_id"]),
        "protocol_manifest_sha256": str(parent["protocol_manifest_sha256"]),
        "evaluation_extension_id": STRESS_PROTOCOL_ID,
        "evaluation_extension_kind": str(extension["protocol_kind"]),
        "evaluation_extension_manifest_sha256": str(extension["manifest_sha256"]),
        "evaluation_extension_parent_protocol_id": str(parent["protocol_id"]),
        "evaluation_extension_parent_protocol_manifest_sha256": str(parent["protocol_manifest_sha256"]),
        "evaluation_mask_cache_sha256": str(extension["fixed_mask_cache"]["sha256"]),
        "evaluation_mask_cache_checksum": str(extension["fixed_mask_cache"]["cache_checksum"]),
        "evaluation_extension_token_count": str(cache["token_count"]),
        "evaluation_extension_mask_type": "modality_frame",
        "evaluation_extension_rates_json": json.dumps(cache["rates"], separators=(",", ":")),
        "evaluation_extension_masks_per_rate": str(cache["masks_per_rate"]),
        "evaluation_extension_single_cell_mask_count": str(cache["single_cell_mask_count"]),
        "evaluation_extension_per_rate_mask_counts_json": json.dumps(
            cache["per_rate_mask_counts"], separators=(",", ":"), sort_keys=True
        ),
        "evaluation_extension_rate_balance_audit_json": json.dumps(
            cache["rate_balance_audit"], separators=(",", ":")
        ),
        "evaluation_extension_balance_policy": str(cache["balance_policy"]),
        "training_batch_size": "64",
        "training_epochs": "40",
        "checkpoint_role": "last",
        "checkpoint_policy": "fixed_epoch_last_pth",
    }
    for field, value in expected.items():
        if str(provenance[field]) != value:
            raise ValueError(f"Temporal-token stress provenance mismatch for {field}: {provenance[field]!r} != {value!r}.")
    if _as_int(provenance.get("training_mask_seed", seed), Path("<provenance>"), 0) != seed:
        raise ValueError("Temporal-token stress training seed provenance mismatch.")


def _validate_cross_unit_identity(
    units: Mapping[tuple[str, int], EvalUnit], extension: Mapping[str, Any], cache: Mapping[str, Any]
) -> None:
    if set(units) != {(method, seed) for method in METHODS for seed in SEEDS}:
        raise ValueError("Temporal-token stress matrix is incomplete.")
    for field in (*_SHARED_PROVENANCE_FIELDS, "condition_identity_sha256"):
        values = {unit.provenance[field] for unit in units.values()}
        if len(values) != 1:
            raise ValueError(f"Temporal-token stress has inconsistent cross-unit {field}: {sorted(values)}.")
    if next(iter(units.values())).provenance["evaluation_extension_manifest_sha256"] != str(extension["manifest_sha256"]):
        raise ValueError("Temporal-token stress extension identity mismatch.")
    if next(iter(units.values())).provenance["evaluation_mask_cache_checksum"] != str(cache["checksum"]):
        raise ValueError("Temporal-token stress cache identity mismatch.")


def _domain_rate_rows(
    units: Mapping[tuple[str, int], EvalUnit], conditions: Mapping[tuple[str, str, str], Mapping[str, Any]]
) -> list[dict[str, Any]]:
    expected_counts: dict[float, int] = defaultdict(int)
    for condition in conditions.values():
        expected_counts[float(condition["requested_missing_rate"])] += 1
    result = []
    for (method, seed), unit in sorted(units.items()):
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in unit.rows:
            grouped[str(row["domain_id"])].append(row)
        for domain_id, rows in grouped.items():
            by_rate: dict[float, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                by_rate[float(row["requested_missing_rate"])].append(row)
            rate_metrics: dict[float, dict[str, float]] = {}
            for rate in (0.0, *STRESS_RATES):
                selected = by_rate.get(rate, [])
                expected_count = expected_counts[rate]
                if len(selected) != expected_count:
                    raise ValueError(f"Temporal-token stress requires {expected_count} masks at rate={rate:g}.")
                values = {metric: _mean(float(row[metric]) for row in selected) for metric in METRICS}
                rate_metrics[rate] = values
                result.append(
                    {
                        "method": method,
                        "seed": seed,
                        "domain_id": domain_id,
                        "cell": _cell_name(rate),
                        "missing_rate": rate,
                        "mask_count": len(selected),
                        "aggregation": "equal_weight_fixed_balanced_modality_frame_masks",
                        **values,
                    }
                )
            auc_values = {
                metric: _normalized_auc({rate: rate_metrics[rate][metric] for rate in MAIN_RATES}) for metric in METRICS
            }
            result.append(
                {
                    "method": method,
                    "seed": seed,
                    "domain_id": domain_id,
                    "cell": "TokenAUC0_90",
                    "missing_rate": 0.9,
                    "mask_count": "",
                    "aggregation": "clean_plus_balanced_modality_frame_rate_trapezoid_excludes_single_cell_95",
                    **auc_values,
                }
            )
    return result


def _domain_macro(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["seed"]), str(row["cell"]), float(row["missing_rate"]))].append(row)
    result = []
    for (method, seed, cell, rate), selected in sorted(grouped.items()):
        if len(selected) != 15:
            raise ValueError(f"Temporal-token stress requires 15 domains for {method}/seed{seed}/{cell}.")
        result.append(
            {
                "method": method,
                "seed": seed,
                "cell": cell,
                "missing_rate": rate,
                "domain_count": len(selected),
                "mask_count": _mean(_numeric_or_zero(row["mask_count"]) for row in selected),
                "aggregation": "equal_weight_15_domain_macro",
                **{metric: _mean(float(row[metric]) for row in selected) for metric in METRICS},
            }
        )
    return result


def _seed_aggregate(rows: list[dict[str, Any]], *, group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    result = []
    for key, selected in sorted(grouped.items()):
        if {int(row["seed"]) for row in selected} != set(SEEDS) or len(selected) != len(SEEDS):
            raise ValueError(f"Temporal-token stress has incomplete fixed seeds for {key}.")
        result.append(
            {
                **dict(zip(group_fields, key)),
                "requested_seed_count": len(SEEDS),
                "available_seed_count": len(selected),
                "aggregation_status": "complete",
                "aggregation": "equal_weight_fixed_seed_mean",
                **{f"{metric}_mean": _mean(float(row[metric]) for row in selected) for metric in METRICS},
                **{f"{metric}_std": _sample_std(float(row[metric]) for row in selected) for metric in METRICS},
            }
        )
    return result


def _paired_statistics(
    domain_rows: list[dict[str, Any]],
    *,
    comparators: Iterable[str],
    bootstrap_iterations: int,
    bootstrap_seed: int,
    confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {(row["method"], row["seed"], row["domain_id"], row["cell"]): row for row in domain_rows}
    domains = sorted({str(row["domain_id"]) for row in domain_rows if row["method"] == "T2" and row["seed"] == 1})
    cells = sorted({str(row["cell"]) for row in domain_rows})
    deltas = []
    intervals = []
    for comparator in comparators:
        for cell in cells:
            by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for seed in SEEDS:
                for domain_id in domains:
                    t2 = indexed.get(("T2", seed, domain_id, cell))
                    other = indexed.get((comparator, seed, domain_id, cell))
                    if t2 is None or other is None:
                        raise ValueError(f"Missing paired temporal-token stress row for T2/{comparator}/seed{seed}/{domain_id}/{cell}.")
                    for metric in METRICS:
                        item = {
                            "treatment": "T2",
                            "comparator": comparator,
                            "seed": seed,
                            "domain_id": domain_id,
                            "cell": cell,
                            "metric": metric,
                            "preferred_direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                            "t2_value": float(t2[metric]),
                            "comparator_value": float(other[metric]),
                            "delta_t2_minus_comparator": float(t2[metric]) - float(other[metric]),
                            "comparison_scope": _comparison_scope(comparator),
                        }
                        deltas.append(item)
                        by_metric[metric].append(item)
            for metric, items in by_metric.items():
                low, high = _paired_bootstrap(
                    items,
                    domains=domains,
                    seed=_stable_seed(bootstrap_seed, comparator, cell, metric),
                    iterations=bootstrap_iterations,
                    confidence=confidence,
                )
                seed_means = [
                    _mean(item["delta_t2_minus_comparator"] for item in items if item["seed"] == seed) for seed in SEEDS
                ]
                intervals.append(
                    {
                        "treatment": "T2",
                        "comparator": comparator,
                        "cell": cell,
                        "metric": metric,
                        "preferred_direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                        "paired_seed_count": len(SEEDS),
                        "paired_domain_count_per_seed": len(domains),
                        "paired_domain_seed_count": len(items),
                        "mean_delta_t2_minus_comparator": _mean(item["delta_t2_minus_comparator"] for item in items),
                        "seed_delta_std": _sample_std(seed_means),
                        "ci_low": low,
                        "ci_high": high,
                        "confidence": confidence,
                        "bootstrap_iterations": bootstrap_iterations,
                        "bootstrap_unit": "paired_domains_within_fixed_seed_then_equal_seed_weight",
                        "comparison_scope": _comparison_scope(comparator),
                    }
                )
    return deltas, intervals


def _comparison_scope(comparator: str) -> str:
    del comparator
    return "main_method_comparison"


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    *,
    domains: list[str],
    seed: int,
    iterations: int,
    confidence: float,
) -> tuple[float, float]:
    values: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        values[int(row["seed"])][str(row["domain_id"])] = float(row["delta_t2_minus_comparator"])
    if set(values) != set(SEEDS) or any(set(values[seed]) != set(domains) for seed in SEEDS):
        raise ValueError("Temporal-token stress paired bootstrap needs complete fixed seed/domain coverage.")
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        draws.append(
            _mean(_mean(values[seed][rng.choice(domains)] for _ in domains) for seed in SEEDS)
        )
    alpha = (1.0 - confidence) / 2.0
    return _quantile(draws, alpha), _quantile(draws, 1.0 - alpha)


def _coverage_rows(units: Mapping[tuple[str, int], EvalUnit], extension: Mapping[str, Any], cache: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected = len(extension["domains"]) * len(cache["conditions"])
    return [
        {
            "method": unit.method,
            "seed": unit.seed,
            "status": "complete",
            "domain_count": len(extension["domains"]),
            "condition_count": len(cache["conditions"]),
            "row_count": len(unit.rows),
            "expected_row_count": expected,
            "metrics_path": str(unit.path.resolve()),
            **unit.provenance,
        }
        for unit in units.values()
    ]


def _condition_catalog(cache: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = (
        "family",
        "pattern",
        "mask_type",
        "requested_missing_rate",
        "observed_missing_rate",
        "available_modalities",
        "drop_count",
        "mask_digest",
        "token_count",
        "retained_token_count",
        "dropped_token_count",
        "per_modality_retained_counts",
        "per_modality_dropped_counts",
        "per_frame_retained_counts",
        "per_frame_dropped_counts",
        "mask_set_index",
        "mask_set_size",
        "mask_balance_policy",
        "modality_temporal_mask",
    )
    return [
        {
            key: json.dumps(condition[key], separators=(",", ":")) if isinstance(condition[key], (list, dict)) else condition[key]
            for key in fields
        }
        for condition in cache["conditions"]
    ]


def _balance_rows(cache: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
            for key, value in audit.items()
        }
        for audit in cache["rate_balance_audit"]
    ]


def _provenance(
    extension: Mapping[str, Any],
    *,
    cache: Mapping[str, Any],
    eval_dir: Path,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any]:
    parent = extension["parent_training_protocol"]
    request = {
        "evaluation_extension_id": STRESS_PROTOCOL_ID,
        "evaluation_extension_manifest": str(extension["path"]),
        "evaluation_extension_manifest_sha256": str(extension["manifest_sha256"]),
        "parent_training_protocol_id": str(parent["protocol_id"]),
        "parent_training_protocol_manifest_sha256": str(parent["protocol_manifest_sha256"]),
        "evaluation_mask_cache_sha256": str(extension["fixed_mask_cache"]["sha256"]),
        "evaluation_mask_cache_checksum": str(extension["fixed_mask_cache"]["cache_checksum"]),
        "per_rate_mask_counts": dict(cache["per_rate_mask_counts"]),
        "single_cell_mask_count": int(cache["single_cell_mask_count"]),
        "eval_dir": str(eval_dir.resolve()),
        "methods": list(METHODS),
        "main_comparison_methods": list(MAIN_COMPARISON_METHODS),
        "seeds": list(SEEDS),
        "main_rates": list(MAIN_RATES),
        "extreme_rate": EXTREME_RATE,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "confidence": confidence,
    }
    return {
        "schema_version": 2,
        "status": "complete",
        "claim_boundary": "post-selection confirmation evaluation extension; 95% is single-cell extreme stress and excluded from main AUC",
        "request": request,
        "request_sha256": _sha256_payload(request),
        "statistics": {
            "domain_aggregation": "equal_weight_15_domain_macro_within_each_seed",
            "seed_aggregation": "equal_weight_fixed_seed_mean",
            "paired_ci": "paired_domains_within_fixed_seed_then_equal_seed_weight",
            "main_auc": "trapezoid_clean_to_90_percent_excludes_single_cell_95",
        },
    }


def _plot_manifest(provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "data_ready",
        "evaluation_extension_id": STRESS_PROTOCOL_ID,
        "provenance_request_sha256": provenance["request_sha256"],
        "figures": [
            {
                "id": "balanced_temporal_token_stress_curve",
                "source_csv": "paper_temporal_token_stress_curve.csv",
                "kind": "line_with_seed_std",
                "x": "missing_rate",
                "y": "top1_mean",
                "error": "top1_std",
                "series": "method",
                "filter": {"cell": ["Clean", "TokenDrop20", "TokenDrop40", "TokenDrop60", "TokenDrop80", "TokenDrop90"]},
            },
            {
                "id": "single_cell_95_extreme_stress",
                "source_csv": "paper_single_cell_95_table.csv",
                "kind": "grouped_table_or_bar",
                "x": "method",
                "y": "top1_mean",
                "error": "top1_std",
            },
            {
                "id": "balanced_temporal_token_t2_contrasts",
                "source_csv": "paired_bootstrap_ci.csv",
                "kind": "forest",
                "x": "mean_delta_t2_minus_comparator",
                "ci_low": "ci_low",
                "ci_high": "ci_high",
                "series": "comparator",
                "filter": {"metric": "top1"},
            },
        ],
    }


def _markdown(
    provenance: Mapping[str, Any],
    table: list[dict[str, Any]],
    extreme: list[dict[str, Any]],
    paired_ci: list[dict[str, Any]],
) -> str:
    indexed = {(row["method"], row["cell"]): row for row in table}
    cells = ("Clean", "TokenDrop20", "TokenDrop40", "TokenDrop60", "TokenDrop80", "TokenDrop90", "TokenAUC0_90")
    lines = [
        "# MMW TWC Balanced Temporal Token-Stress Summary",
        "",
        "This is an immutable post-selection evaluation extension of the frozen v1 training protocol.",
        "Each rate uses exact 5x4 modality-frame cardinality and exact cross-mask modality/frame balance.",
        "",
        "## Main Top1",
        "",
        "| Method | Clean | Drop20 | Drop40 | Drop60 | Drop80 | Drop90 | AUC 0-90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in MAIN_COMPARISON_METHODS:
        values = [_mean_std(indexed[(method, cell)]["top1_mean"], indexed[(method, cell)]["top1_std"]) for cell in cells]
        lines.append(f"| {_display_name(method)} | " + " | ".join(values) + " |")
    lines.extend(["", "## Single-Cell 95%", "", "| Method | Top1 |", "|---|---:|"])
    for row in sorted(extreme, key=lambda item: MAIN_COMPARISON_METHODS.index(str(item["method"]))):
        lines.append(f"| {_display_name(str(row['method']))} | {_mean_std(row['top1_mean'], row['top1_std'])} |")
    lines.extend(["", "## Paired Top1 Contrasts", "", "| Contrast | Cell | Delta | 95% CI |", "|---|---|---:|---:|"])
    for row in paired_ci:
        if row["metric"] == "top1" and row["cell"] in {"TokenDrop90", "SingleCell95", "TokenAUC0_90"}:
            lines.append(
                f"| T2 - {row['comparator']} | {row['cell']} | {_pct(row['mean_delta_t2_minus_comparator'])} | "
                f"[{_pct(row['ci_low'])}, {_pct(row['ci_high'])}] |"
            )
    lines.extend(
        [
            "",
            "`SingleCell95` retains exactly one of 20 modality-frame cells per mask. It is an extreme fallback endpoint and is excluded from AUC 0-90.",
            "",
        ]
    )
    return "\n".join(lines)


def _display_name(method: str) -> str:
    return method


def _reproducibility_markdown(provenance: Mapping[str, Any], extension: Mapping[str, Any], cache: Mapping[str, Any]) -> str:
    request = provenance["request"]
    lines = [
        "# Reproducibility Record",
        "",
        f"- Evaluation extension: `{request['evaluation_extension_id']}`",
        f"- Extension manifest SHA256: `{request['evaluation_extension_manifest_sha256']}`",
        f"- Parent training protocol SHA256: `{request['parent_training_protocol_manifest_sha256']}`",
        f"- Cache SHA256: `{request['evaluation_mask_cache_sha256']}`",
        f"- Cache checksum: `{request['evaluation_mask_cache_checksum']}`",
        f"- Mask generator: `{cache['generator']}`",
        f"- Mask seed: `{cache['seed']}`",
        f"- Tensor contract: `{cache['history_window']} frames x {len(cache['modalities'])} modalities = {cache['token_count']} cells`",
        f"- Rates: `{','.join(str(int(rate * 100)) for rate in cache['rates'])}%`",
        f"- Masks per 20--90% rate: `{cache['masks_per_rate']}`",
        f"- Single-cell 95% masks: `{cache['single_cell_mask_count']}`",
        f"- Per-rate mask counts: `{json.dumps(cache['per_rate_mask_counts'], sort_keys=True)}`",
        f"- Balance policy: `{cache['balance_policy']}`",
        f"- Training checkpoint policy: `last.pth`, 40 epochs, batch 64, seeds 1..5",
        f"- Bootstrap: `{request['bootstrap_iterations']}` iterations, seed `{request['bootstrap_seed']}`, confidence `{request['confidence']}`",
        "",
        "The cache manifest, condition catalog, per-rate balance audit, per-domain rows, paired deltas, and every evaluator provenance file are co-located with this report.",
        "",
    ]
    return "\n".join(lines)


def _prepare_output(path: Path, provenance: Mapping[str, Any]) -> None:
    existing = path / "provenance.json"
    if existing.exists():
        current = _read_json(existing)
        if current.get("request_sha256") != provenance["request_sha256"]:
            raise ValueError(f"Refusing to overwrite summary from another immutable request: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _condition_key(item: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(item.get("family", item.get("eval_family", ""))), str(item["pattern"]), str(item["mask_digest"])


def _condition_identity(rows: Iterable[Mapping[str, str]]) -> str:
    entries = [
        {
            "domain_id": str(row["domain_id"]),
            "sample_csv_sha256": str(row["sample_csv_sha256"]),
            "condition": _condition_key(row),
        }
        for row in rows
    ]
    entries.sort(key=lambda item: (item["domain_id"], item["condition"]))
    return _sha256_payload(entries)


def _cell_name(rate: float) -> str:
    if rate == 0.0:
        return "Clean"
    if _close(rate, EXTREME_RATE):
        return "SingleCell95"
    return f"TokenDrop{int(round(rate * 100))}"


def _normalized_auc(values: Mapping[float, float]) -> float:
    rates = sorted(values)
    if rates != list(MAIN_RATES):
        raise ValueError(f"Temporal-token stress AUC rate mismatch: {rates}.")
    area = sum((right - left) * (values[left] + values[right]) / 2.0 for left, right in zip(rates, rates[1:]))
    return area / (rates[-1] - rates[0])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be a mapping: {path}")
    return payload


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"CSV rows have inconsistent schema: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    temporary.replace(path)


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _singleton_values(values: Mapping[str, set[str]], label: str) -> dict[str, str]:
    result = {}
    for field, candidates in values.items():
        if len(candidates) != 1:
            raise ValueError(f"Expected one {label} value for {field}, got {sorted(candidates)}.")
        result[field] = next(iter(candidates))
    return result


def _as_int(value: Any, path: Path, line: int) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer at {path}:{line}: {value!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"Invalid integer at {path}:{line}: {value!r}")
    return int(numeric)


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _truthy(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("Cannot average no values.")
    return sum(values) / len(values)


def _sample_std(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _numeric_or_zero(value: Any) -> float:
    numeric = _finite_float(value)
    return numeric if numeric is not None else 0.0


def _stable_seed(base: int, *parts: str) -> int:
    payload = ":".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _close(first: float, second: float, tolerance: float = 1.0e-9) -> bool:
    return abs(float(first) - float(second)) <= tolerance


def _pct(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def _mean_std(mean: Any, std: Any) -> str:
    return f"{_pct(mean)} +/- {_pct(std)}"


if __name__ == "__main__":
    raise SystemExit(main())
