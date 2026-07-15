#!/usr/bin/env python3
"""Summarize seed-stratified MMW baseline evaluations without pooling seeds early."""

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("T2", "amber_full", "rmbp_mm")
METHOD_LABELS = {"T2": "T2", "amber_full": "AMBER-Full", "rmbp_mm": "RMBP-MM"}
SEEDS = (1, 2, 3)
MAIN_RATES = (0.0, 0.2, 0.4, 0.6, 0.8)
EXTREME_RATES = (0.85, 0.9, 0.95)
EXTREME_CURVE_RATES = (0.8, *EXTREME_RATES)
MAIN_MASK_TYPES = {"frame_level", "block", "modality_frame"}
MAIN_MASK_COUNTS = {
    0.0: {"clean": 1},
    0.2: {"modality_frame": 16, "frame_level": 5, "block": 5},
    0.4: {"modality_frame": 16, "frame_level": 10, "block": 4},
    0.6: {"modality_frame": 16, "frame_level": 10, "block": 3},
    0.8: {"modality_frame": 16, "frame_level": 5, "block": 2},
}
SCOPES = {
    "T2": ("project_mainline", "False", "mainline_local_validation"),
    "amber_full": ("amber_full_local_adaptation", "False", "local_adaptation_diagnostic"),
    "rmbp_mm": ("rmbp_mm_channel_attention_local", "False", "out_of_paper_scope_diagnostic"),
}
IDENTITY_FIELDS = (
    "domain_id",
    "condition",
    "scene",
    "sample_count",
    "sample_csv_sha256",
    "eval_family",
    "pattern",
    "available_modalities",
    "missing_rate",
    "drop_count",
    "mask_index",
    "mask_type",
    "mask_digest",
    "mask_cache_checksum",
    "mask_cache_seed",
    "observed_missing_rate",
    "last_frame_available",
    "last_frame_available_modalities",
    "trailing_fully_missing_frames",
)


@dataclass
class EvalUnit:
    method: str
    seed: int
    path: Path
    rows: list[dict[str, str]]
    status: str
    reason: str
    scope: tuple[str, str, str]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize MMW T2/baseline seeds without pre-pooling seeds.")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--extreme-eval-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--expected-domains", type=int, default=15)
    parser.add_argument("--bootstrap-iterations", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260714)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    unknown = sorted(set(methods) - set(METHODS))
    if not methods or unknown or not seeds or len(set(seeds)) != len(seeds):
        parser.error(f"invalid methods/seeds: methods={methods}, seeds={seeds}, unknown={unknown}")
    if args.expected_domains <= 0 or args.bootstrap_iterations <= 0 or not 0.0 < args.confidence < 1.0:
        parser.error("expected-domains/bootstrap-iterations must be positive and confidence must be in (0,1)")
    summarize(
        Path(args.eval_dir),
        Path(args.output_dir),
        extreme_eval_dir=Path(args.extreme_eval_dir) if args.extreme_eval_dir else None,
        methods=methods,
        seeds=seeds,
        expected_domains=int(args.expected_domains),
        bootstrap_iterations=int(args.bootstrap_iterations),
        bootstrap_seed=int(args.bootstrap_seed),
        confidence=float(args.confidence),
    )
    return 0


def summarize(
    eval_dir: Path,
    output_dir: Path,
    *,
    extreme_eval_dir: Path | None = None,
    methods: tuple[str, ...] = METHODS,
    seeds: tuple[int, ...] = SEEDS,
    expected_domains: int = 15,
    bootstrap_iterations: int = 4000,
    bootstrap_seed: int = 20260714,
    confidence: float = 0.95,
) -> dict[str, Any]:
    main_units = _load_units(eval_dir, methods, seeds, expected_domains, "main")
    _enforce_shared_identity(main_units)
    extreme_units = (
        _load_units(extreme_eval_dir, methods, seeds, expected_domains, "extreme")
        if extreme_eval_dir is not None
        else {
            (method, seed): EvalUnit(
                method, seed, Path(""), [], "unavailable", "extreme_eval_dir_not_provided", SCOPES[method]
            )
            for method in methods
            for seed in seeds
        }
    )
    if extreme_eval_dir is not None:
        _enforce_shared_identity(extreme_units)

    availability = _availability_rows(main_units, extreme_units, methods, seeds)
    per_seed, main_curves, main_domain_curves, clean_domains = _main_summaries(main_units, methods, seeds)
    domain_summary, weather_summary, scene_summary, worst_domains = _grouped_main_summaries(
        main_units, methods, seeds
    )
    bridge_curves = _modality_frame_bridge(main_units, methods, seeds)
    extreme_curves, _ = _curve_summaries(extreme_units, methods, seeds, EXTREME_RATES, "extreme_modality_frame")
    extreme_curves = [*bridge_curves, *extreme_curves]
    main_curve_rows = _with_seed_aggregates(main_curves, methods, seeds, MAIN_RATES)
    extreme_curve_rows = _with_seed_aggregates(extreme_curves, methods, seeds, EXTREME_CURVE_RATES)
    multiseed = _multiseed_rows(per_seed, methods, seeds)
    paired_seed, paired_domains = _paired_units(
        per_seed,
        main_domain_curves,
        clean_domains,
        seeds,
    )
    comparisons = _comparisons(
        paired_seed,
        paired_domains,
        seeds,
        expected_domains=expected_domains,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    decision = {
        "status": _overall_status(comparisons),
        "requested_methods": list(methods),
        "requested_seeds": list(seeds),
        "bootstrap": {
            "unit": "paired_seed_domain",
            "grouping": "fixed_seed_equal_weight_domain_resampling",
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
            "confidence": confidence,
        },
        "comparisons": comparisons,
        "claim_scope": "MMW fixed-split local validation against locally adapted baselines",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "availability.csv", availability)
    _write_csv(output_dir / "per_seed_summary.csv", per_seed)
    _write_csv(output_dir / "per_seed_domain_summary.csv", domain_summary)
    _write_csv(output_dir / "per_seed_weather_summary.csv", weather_summary)
    _write_csv(output_dir / "per_seed_scene_summary.csv", scene_summary)
    _write_csv(output_dir / "per_seed_worst_domain_summary.csv", worst_domains)
    _write_csv(output_dir / "main_type_equal_curve.csv", main_curve_rows)
    _write_csv(output_dir / "extreme_modality_frame_curve.csv", extreme_curve_rows, _curve_columns())
    _write_csv(output_dir / "multiseed_summary.csv", multiseed)
    _write_csv(output_dir / "paired_seed_deltas.csv", paired_seed, _paired_seed_columns())
    _write_csv(output_dir / "paired_seed_domain_units.csv", paired_domains, _paired_domain_columns())
    _write_csv(output_dir / "comparisons.csv", comparisons, _comparison_columns())
    _plot_top1_curves(main_curve_rows, extreme_curve_rows, output_dir / "top1_robustness_curves.png")
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(
        _markdown(availability, multiseed, extreme_curve_rows, comparisons, seeds),
        encoding="utf-8",
    )
    return {
        "availability": availability,
        "per_seed": per_seed,
        "domain_summary": domain_summary,
        "weather_summary": weather_summary,
        "scene_summary": scene_summary,
        "worst_domains": worst_domains,
        "main_curves": main_curve_rows,
        "extreme_curves": extreme_curve_rows,
        "multiseed": multiseed,
        "paired_seed": paired_seed,
        "paired_domains": paired_domains,
        "comparisons": comparisons,
        "decision": decision,
    }


def _load_units(
    root: Path,
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    expected_domains: int,
    kind: str,
) -> dict[tuple[str, int], EvalUnit]:
    units = {}
    for method in methods:
        for seed in seeds:
            path = root / method / f"seed{seed}" / "metrics.csv"
            rows = _read_csv(path)
            status, reason, scope = _validate_unit(rows, path, method, seed, expected_domains, kind)
            units[(method, seed)] = EvalUnit(method, seed, path, rows, status, reason, scope)
    return units


def _validate_unit(
    rows: list[dict[str, str]],
    path: Path,
    method: str,
    seed: int,
    expected_domains: int,
    kind: str,
) -> tuple[str, str, tuple[str, str, str]]:
    default_scope = SCOPES[method]
    if not path.exists():
        return "unavailable", "metrics_missing", default_scope
    if not rows:
        return "unavailable", "metrics_empty", default_scope
    scopes = {
        (row.get("reproduction_scope", ""), row.get("paper_equivalent", ""), row.get("temporal_result_scope", ""))
        for row in rows
    }
    scope = next(iter(scopes)) if len(scopes) == 1 else default_scope
    if len(scopes) != 1 or not all(scope):
        return "unavailable", "baseline_scope_missing_or_inconsistent", scope
    if scope != SCOPES[method]:
        return "unavailable", "baseline_scope_mismatch", scope
    if any(row.get("method") != method for row in rows):
        return "unavailable", "method_provenance_mismatch", scope
    if any(_int(row.get("seed")) != seed for row in rows):
        return "unavailable", "seed_provenance_mismatch", scope
    selected = [row for row in rows if row.get("eval_family") in {"whole_modality", "temporal_missing"}]
    if kind == "extreme":
        selected = [row for row in selected if row.get("eval_family") == "temporal_missing"]
    domains = {row.get("domain_id", "") for row in selected}
    if "" in domains or len(domains) != expected_domains:
        return "unavailable", f"domain_count_{len(domains)}_expected_{expected_domains}", scope
    required = ("sample_csv_sha256", "mask_digest", "mask_cache_checksum", "checkpoint", "checkpoint_policy", "top1")
    if any(any(not str(row.get(field, "")).strip() for field in required) for row in selected):
        return "unavailable", "required_provenance_missing", scope
    if any(_float(row.get("top1")) is None for row in selected):
        return "unavailable", "top1_missing", scope
    checkpoints = {row.get("checkpoint", "") for row in selected}
    checkpoint = Path(next(iter(checkpoints), ""))
    if len(checkpoints) != 1 or not _artifact_exists(str(checkpoint)):
        return "unavailable", "checkpoint_missing_or_inconsistent", scope
    if (
        checkpoint.name != "last.pth"
        or checkpoint.parent.name != "checkpoints"
        or checkpoint.parent.parent.name != f"seed{seed}"
        or checkpoint.parent.parent.parent.name != method
    ):
        return "unavailable", "checkpoint_not_method_seed_last_pth", scope
    if {row.get("checkpoint_policy", "") for row in selected} != {"fixed_epoch_last_pth"}:
        return "unavailable", "checkpoint_policy_mismatch", scope
    identities = [_identity(row) for row in selected]
    if len(identities) != len(set(identities)):
        return "unavailable", "duplicate_mask_identity", scope

    temporal = [row for row in selected if row.get("eval_family") == "temporal_missing"]
    rates = {_rate(row.get("missing_rate")) for row in temporal}
    expected_rates = set(EXTREME_RATES if kind == "extreme" else MAIN_RATES)
    if rates != expected_rates:
        return "unavailable", f"rate_set_mismatch_{sorted(rates)}", scope
    for rate in rates:
        rate_rows = [row for row in temporal if _rate(row.get("missing_rate")) == rate]
        mask_types = {row.get("mask_type", "") for row in rate_rows}
        expected_types = {"modality_frame"} if kind == "extreme" else ({"clean"} if rate == 0.0 else MAIN_MASK_TYPES)
        if mask_types != expected_types:
            return "unavailable", f"mask_types_mismatch_rate_{rate:g}", scope
        for mask_type in expected_types:
            digests = {row.get("mask_digest", "") for row in rate_rows if row.get("mask_type") == mask_type}
            expected_count = 16 if kind == "extreme" else MAIN_MASK_COUNTS[rate][mask_type]
            if len(digests) != expected_count:
                return "unavailable", f"mask_count_mismatch_{rate:g}_{mask_type}_{len(digests)}", scope
            selected_rows = [row for row in rate_rows if row.get("mask_type") == mask_type]
            if len(selected_rows) != expected_count * expected_domains:
                return "unavailable", f"mask_row_count_mismatch_{rate:g}_{mask_type}_{len(selected_rows)}", scope
            digest_domains = Counter((row.get("mask_digest", ""), row.get("domain_id", "")) for row in selected_rows)
            if set(digest_domains.values()) != {1}:
                return "unavailable", f"duplicate_digest_domain_{rate:g}_{mask_type}", scope
            for digest in digests:
                mask_domains = {
                    row.get("domain_id", "")
                    for row in rate_rows
                    if row.get("mask_type") == mask_type and row.get("mask_digest") == digest
                }
                if len(mask_domains) != expected_domains:
                    return "unavailable", f"incomplete_mask_{rate:g}_{mask_type}_{digest}", scope
    if kind == "main":
        clean = [row for row in selected if row.get("eval_family") == "whole_modality" and row.get("pattern") == "full"]
        if len(clean) != expected_domains or len({row.get("domain_id", "") for row in clean}) != expected_domains:
            return "unavailable", "clean_full_domain_incomplete", scope
    return "available", "", scope


def _enforce_shared_identity(units: dict[tuple[str, int], EvalUnit]) -> None:
    available = [unit for unit in units.values() if unit.status == "available"]
    if len(available) < 2:
        return
    signatures = Counter(frozenset(_identity(row) for row in _selected_rows(unit)) for unit in available)
    expected, _ = signatures.most_common(1)[0]
    for unit in available:
        if frozenset(_identity(row) for row in _selected_rows(unit)) != expected:
            unit.status = "unavailable"
            unit.reason = "cross_method_seed_mask_identity_mismatch"


def _selected_rows(unit: EvalUnit) -> list[dict[str, str]]:
    rows = [row for row in unit.rows if row.get("eval_family") in {"whole_modality", "temporal_missing"}]
    return rows if any(row.get("eval_family") == "whole_modality" for row in rows) else [
        row for row in rows if row.get("eval_family") == "temporal_missing"
    ]


def _availability_rows(main_units, extreme_units, methods, seeds) -> list[dict[str, Any]]:
    result = []
    for method in methods:
        for seed in seeds:
            main = main_units[(method, seed)]
            extreme = extreme_units[(method, seed)]
            result.append(
                {
                    "method": method,
                    "seed": seed,
                    "main_status": main.status,
                    "main_reason": main.reason,
                    "main_metrics_path": str(main.path),
                    "extreme_status": extreme.status,
                    "extreme_reason": extreme.reason,
                    "extreme_metrics_path": str(extreme.path) if str(extreme.path) != "." else "",
                    "reproduction_scope": main.scope[0],
                    "paper_equivalent": main.scope[1],
                    "temporal_result_scope": main.scope[2],
                }
            )
    return result


def _main_summaries(main_units, methods, seeds):
    per_seed = []
    curves = []
    domain_curves = {}
    clean_domains = {}
    for method in methods:
        for seed in seeds:
            unit = main_units[(method, seed)]
            if unit.status != "available":
                continue
            clean = _clean_by_domain(unit.rows)
            curve = _curve_by_domain(unit.rows)
            clean_domains[(method, seed)] = clean
            domain_curves[(method, seed)] = curve
            rate_values = {rate: _mean([curve[(domain, rate)] for domain in clean]) for rate in MAIN_RATES}
            per_seed.append(
                {
                    "method": method,
                    "seed": seed,
                    "domain_count": len(clean),
                    "clean_top1": _mean(list(clean.values())),
                    "auc_top1_0_80": _normalized_auc(rate_values),
                    "drop80_top1": rate_values[0.8],
                    "reproduction_scope": unit.scope[0],
                    "paper_equivalent": unit.scope[1],
                    "temporal_result_scope": unit.scope[2],
                }
            )
            curves.extend(_per_seed_curve_rows(method, seed, rate_values, unit.scope, "main_type_equal"))
    return per_seed, curves, domain_curves, clean_domains


def _grouped_main_summaries(main_units, methods, seeds):
    domain_rows = []
    for method in methods:
        for seed in seeds:
            unit = main_units[(method, seed)]
            if unit.status != "available":
                continue
            metadata = {
                row["domain_id"]: row
                for row in unit.rows
                if row.get("eval_family") == "whole_modality" and row.get("pattern") == "full"
            }
            curves = _curve_by_domain(unit.rows)
            for domain_id, row in sorted(metadata.items()):
                rate_values = {rate: curves[(domain_id, rate)] for rate in MAIN_RATES}
                domain_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "domain_id": domain_id,
                        "condition": row["condition"],
                        "scene": row["scene"],
                        "clean_top1": float(row["top1"]),
                        "auc_top1_0_80": _normalized_auc(rate_values),
                        "drop80_top1": rate_values[0.8],
                        "reproduction_scope": unit.scope[0],
                        "paper_equivalent": unit.scope[1],
                        "temporal_result_scope": unit.scope[2],
                    }
                )

    def grouped(field):
        groups = defaultdict(list)
        for row in domain_rows:
            groups[(row["method"], row["seed"], row[field])].append(row)
        result = []
        for (method, seed, value), selected in sorted(groups.items()):
            result.append(
                {
                    "method": method,
                    "seed": seed,
                    field: value,
                    "domain_count": len(selected),
                    **{
                        metric: _mean([float(row[metric]) for row in selected])
                        for metric in ("clean_top1", "auc_top1_0_80", "drop80_top1")
                    },
                    "reproduction_scope": selected[0]["reproduction_scope"],
                    "paper_equivalent": selected[0]["paper_equivalent"],
                    "temporal_result_scope": selected[0]["temporal_result_scope"],
                }
            )
        return result

    worst_rows = []
    for method in methods:
        for seed in seeds:
            selected = [row for row in domain_rows if row["method"] == method and row["seed"] == seed]
            if not selected:
                continue
            item = {"method": method, "seed": seed, "domain_count": len(selected)}
            for metric in ("clean_top1", "auc_top1_0_80", "drop80_top1"):
                worst = min(selected, key=lambda row: (float(row[metric]), row["domain_id"]))
                item[f"worst_{metric}_domain"] = worst["domain_id"]
                item[f"worst_{metric}"] = worst[metric]
            item.update(
                {
                    "reproduction_scope": selected[0]["reproduction_scope"],
                    "paper_equivalent": selected[0]["paper_equivalent"],
                    "temporal_result_scope": selected[0]["temporal_result_scope"],
                }
            )
            worst_rows.append(item)
    return domain_rows, grouped("condition"), grouped("scene"), worst_rows


def _curve_summaries(units, methods, seeds, rates, curve_kind):
    rows = []
    domain_curves = {}
    for method in methods:
        for seed in seeds:
            unit = units[(method, seed)]
            if unit.status != "available":
                continue
            curve = _curve_by_domain(unit.rows)
            domain_curves[(method, seed)] = curve
            domains = sorted({domain for domain, _ in curve})
            values = {rate: _mean([curve[(domain, rate)] for domain in domains]) for rate in rates}
            rows.extend(_per_seed_curve_rows(method, seed, values, unit.scope, curve_kind))
    return rows, domain_curves


def _modality_frame_bridge(main_units, methods, seeds):
    rows = []
    for method in methods:
        for seed in seeds:
            unit = main_units[(method, seed)]
            if unit.status != "available":
                continue
            selected = [
                row
                for row in unit.rows
                if row.get("eval_family") == "temporal_missing"
                and _rate(row.get("missing_rate")) == 0.8
                and row.get("mask_type") == "modality_frame"
            ]
            domains = sorted({row["domain_id"] for row in selected})
            values = [
                _mean([float(row["top1"]) for row in selected if row["domain_id"] == domain])
                for domain in domains
            ]
            rows.extend(
                _per_seed_curve_rows(
                    method,
                    seed,
                    {0.8: _mean(values)},
                    unit.scope,
                    "extreme_modality_frame",
                )
            )
    return rows


def _clean_by_domain(rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        row["domain_id"]: float(row["top1"])
        for row in rows
        if row.get("eval_family") == "whole_modality" and row.get("pattern") == "full"
    }


def _curve_by_domain(rows: list[dict[str, str]]) -> dict[tuple[str, float], float]:
    per_type = defaultdict(list)
    for row in rows:
        if row.get("eval_family") != "temporal_missing":
            continue
        per_type[(row["domain_id"], _rate(row["missing_rate"]), row["mask_type"])].append(float(row["top1"]))
    per_rate = defaultdict(list)
    for (domain, rate, _), values in per_type.items():
        per_rate[(domain, rate)].append(_mean(values))
    return {key: _mean(values) for key, values in per_rate.items()}


def _normalized_auc(rate_values: dict[float, float]) -> float:
    rates = sorted(rate_values)
    area = sum(
        (right - left) * (rate_values[left] + rate_values[right]) / 2.0
        for left, right in zip(rates, rates[1:])
    )
    return area / (rates[-1] - rates[0])


def _per_seed_curve_rows(method, seed, rate_values, scope, curve_kind):
    return [
        {
            "method": method,
            "seed": seed,
            "aggregation": "per_seed",
            "curve_kind": curve_kind,
            "missing_rate": rate,
            "top1": value,
            "top1_mean": "",
            "top1_std": "",
            "available_top1_mean": value,
            "available_seed_count": 1,
            "aggregation_status": "per_seed",
            "reproduction_scope": scope[0],
            "paper_equivalent": scope[1],
            "temporal_result_scope": scope[2],
        }
        for rate, value in sorted(rate_values.items())
    ]


def _with_seed_aggregates(rows, methods, seeds, rates):
    result = list(rows)
    for method in methods:
        scope = SCOPES[method]
        for rate in rates:
            values = [
                float(row["top1"])
                for row in rows
                if row["method"] == method and row["missing_rate"] == rate
            ]
            complete = len(values) == len(seeds)
            result.append(
                {
                    "method": method,
                    "seed": "all",
                    "aggregation": "requested_seed_mean",
                    "curve_kind": rows[0]["curve_kind"] if rows else ("main_type_equal" if rates == MAIN_RATES else "extreme_modality_frame"),
                    "missing_rate": rate,
                    "top1": "",
                    "top1_mean": _mean(values) if complete else "",
                    "top1_std": _sample_std(values) if complete else "",
                    "available_top1_mean": _mean(values) if values else "",
                    "available_seed_count": len(values),
                    "aggregation_status": "complete" if complete else "partial" if values else "unavailable",
                    "reproduction_scope": scope[0],
                    "paper_equivalent": scope[1],
                    "temporal_result_scope": scope[2],
                }
            )
    return result


def _multiseed_rows(per_seed, methods, seeds):
    result = []
    for method in methods:
        selected = [row for row in per_seed if row["method"] == method]
        complete = len(selected) == len(seeds)
        row = {
            "method": method,
            "requested_seed_count": len(seeds),
            "available_seed_count": len(selected),
            "available_seeds": ",".join(str(item["seed"]) for item in selected),
            "missing_seeds": ",".join(str(seed) for seed in seeds if seed not in {item["seed"] for item in selected}),
            "aggregation_status": "complete" if complete else "partial" if selected else "unavailable",
            "reproduction_scope": SCOPES[method][0],
            "paper_equivalent": SCOPES[method][1],
            "temporal_result_scope": SCOPES[method][2],
        }
        for metric in ("clean_top1", "auc_top1_0_80", "drop80_top1"):
            values = [float(item[metric]) for item in selected]
            row[f"{metric}_mean"] = _mean(values) if complete else ""
            row[f"{metric}_std"] = _sample_std(values) if complete else ""
        result.append(row)
    return result


def _paired_units(per_seed, domain_curves, clean_domains, seeds):
    summaries = {(row["method"], row["seed"]): row for row in per_seed}
    paired_seed = []
    paired_domains = []
    for baseline in ("amber_full", "rmbp_mm"):
        for seed in seeds:
            t2 = summaries.get(("T2", seed))
            other = summaries.get((baseline, seed))
            if t2 is None or other is None:
                continue
            paired_seed.append(
                {
                    "baseline": baseline,
                    "seed": seed,
                    "clean_top1_delta": t2["clean_top1"] - other["clean_top1"],
                    "auc_top1_0_80_delta": t2["auc_top1_0_80"] - other["auc_top1_0_80"],
                    "drop80_top1_delta": t2["drop80_top1"] - other["drop80_top1"],
                    "baseline_reproduction_scope": SCOPES[baseline][0],
                    "baseline_paper_equivalent": SCOPES[baseline][1],
                    "baseline_temporal_result_scope": SCOPES[baseline][2],
                }
            )
            domains = sorted(set(clean_domains[("T2", seed)]) & set(clean_domains[(baseline, seed)]))
            for domain in domains:
                t2_curve = {rate: domain_curves[("T2", seed)][(domain, rate)] for rate in MAIN_RATES}
                baseline_curve = {rate: domain_curves[(baseline, seed)][(domain, rate)] for rate in MAIN_RATES}
                paired_domains.append(
                    {
                        "baseline": baseline,
                        "seed": seed,
                        "domain_id": domain,
                        "clean_top1_delta": clean_domains[("T2", seed)][domain] - clean_domains[(baseline, seed)][domain],
                        "auc_top1_0_80_delta": _normalized_auc(t2_curve) - _normalized_auc(baseline_curve),
                        "drop80_top1_delta": t2_curve[0.8] - baseline_curve[0.8],
                    }
                )
    return paired_seed, paired_domains


def _comparisons(
    paired_seed,
    paired_domains,
    seeds,
    *,
    expected_domains,
    bootstrap_iterations,
    bootstrap_seed,
    confidence,
):
    result = []
    for baseline in ("amber_full", "rmbp_mm"):
        seed_rows = [row for row in paired_seed if row["baseline"] == baseline]
        available = sorted(int(row["seed"]) for row in seed_rows)
        complete = available == sorted(seeds)
        metric_values = {
            metric: [float(row[metric]) for row in seed_rows]
            for metric in ("clean_top1_delta", "auc_top1_0_80_delta", "drop80_top1_delta")
        }
        means = {metric: (_mean(values) if complete else None) for metric, values in metric_values.items()}
        auc_positive = sum(value > 0.0 for value in metric_values["auc_top1_0_80_delta"])
        gates = {
            "complete_seed_set": complete,
            "auc_positive_at_least_2_of_3": complete and auc_positive >= 2,
            "mean_clean_nonnegative": complete and means["clean_top1_delta"] >= 0.0,
            "mean_auc_nonnegative": complete and means["auc_top1_0_80_delta"] >= 0.0,
            "mean_drop80_nonnegative": complete and means["drop80_top1_delta"] >= 0.0,
        }
        if all(gates.values()):
            status = "supported"
        elif complete and all(means[metric] < 0.0 for metric in means):
            status = "unsupported"
        else:
            status = "partial"
        row = {
            "baseline": baseline,
            "status": status,
            "requested_seed_count": len(seeds),
            "paired_seed_count": len(seed_rows),
            "paired_seeds": ",".join(str(seed) for seed in available),
            "auc_positive_seed_count": auc_positive,
            **{f"gate_{key}": value for key, value in gates.items()},
            "failed_gates": ",".join(key for key, value in gates.items() if not value),
            "baseline_reproduction_scope": SCOPES[baseline][0],
            "baseline_paper_equivalent": SCOPES[baseline][1],
            "baseline_temporal_result_scope": SCOPES[baseline][2],
        }
        domain_rows = [item for item in paired_domains if item["baseline"] == baseline]
        for metric, values in metric_values.items():
            row[f"{metric}_mean"] = means[metric] if complete else ""
            row[f"{metric}_std"] = _sample_std(values) if complete else ""
            ci = _fixed_seed_grouped_bootstrap(
                domain_rows,
                metric,
                seeds,
                expected_domains=expected_domains,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
                confidence=confidence,
            ) if complete else None
            row[f"{metric}_ci_low"] = ci[0] if ci else ""
            row[f"{metric}_ci_high"] = ci[1] if ci else ""
        result.append(row)
    return result


def _fixed_seed_grouped_bootstrap(rows, metric, seeds, *, expected_domains, iterations, seed, confidence):
    by_seed = {requested: [float(row[metric]) for row in rows if int(row["seed"]) == requested] for requested in seeds}
    if any(len(values) != expected_domains for values in by_seed.values()):
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        seed_means = [_mean([rng.choice(by_seed[item]) for _ in range(expected_domains)]) for item in seeds]
        draws.append(_mean(seed_means))
    alpha = (1.0 - confidence) / 2.0
    return _quantile(draws, alpha), _quantile(draws, 1.0 - alpha)


def _overall_status(comparisons):
    statuses = [row["status"] for row in comparisons]
    if statuses and all(status == "supported" for status in statuses):
        return "supported"
    if statuses and all(status == "unsupported" for status in statuses):
        return "unsupported"
    return "partial"


def _plot_top1_curves(main_rows, extreme_rows, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for axis, rows, title in (
        (axes[0], main_rows, "0-80% type-equal temporal missing"),
        (axes[1], extreme_rows, "85-95% independent modality-frame masks"),
    ):
        for method in METHODS:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["method"] == method
                    and row["seed"] == "all"
                    and row["aggregation_status"] == "complete"
                    and (axis is axes[0] or float(row["missing_rate"]) >= 0.85)
                ),
                key=lambda row: float(row["missing_rate"]),
            )
            if not selected:
                continue
            x = [100.0 * float(row["missing_rate"]) for row in selected]
            y = [100.0 * float(row["top1_mean"]) for row in selected]
            yerr = [100.0 * float(row["top1_std"]) for row in selected]
            axis.errorbar(x, y, yerr=yerr, marker="o", capsize=2.5, label=METHOD_LABELS[method])
        axis.set_title(title)
        axis.set_xlabel("Missing modality-time cells (%)")
        axis.set_ylabel("Top1 (%)")
        axis.grid(alpha=0.22)
    axes[0].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _markdown(availability, multiseed, extreme_curves, comparisons, seeds):
    lines = [
        "# MMW T2 vs local baselines: multi-seed summary",
        "",
        "## Availability",
        "",
        "| Method | Seed | Main | Extreme | Scope |",
        "|---|---:|---|---|---|",
    ]
    for row in availability:
        lines.append(
            f"| {row['method']} | {row['seed']} | {row['main_status']} | {row['extreme_status']} | {row['temporal_result_scope']} |"
        )
    lines.extend(["", "## Three-seed Top1", "", "| Method | Clean mean/std | AUC mean/std | Drop80 mean/std |", "|---|---:|---:|---:|"])
    for row in multiseed:
        lines.append(
            f"| {row['method']} | {_mean_std(row, 'clean_top1')} | {_mean_std(row, 'auc_top1_0_80')} | {_mean_std(row, 'drop80_top1')} |"
        )
    lines.extend(["", "## Gates", "", "| Comparison | Status | Clean delta | AUC delta | Drop80 delta | Failed gates |", "|---|---|---:|---:|---:|---|"])
    for row in comparisons:
        lines.append(
            f"| T2 - {row['baseline']} | {row['status']} | {_fmt(row.get('clean_top1_delta_mean'))} | "
            f"{_fmt(row.get('auc_top1_0_80_delta_mean'))} | {_fmt(row.get('drop80_top1_delta_mean'))} | {row['failed_gates'] or 'none'} |"
        )
    aggregate_extreme = [row for row in extreme_curves if row["seed"] == "all"]
    if aggregate_extreme:
        lines.extend(["", "## Extreme modality-frame Top1", "", "| Method | 80% | 85% | 90% | 95% |", "|---|---:|---:|---:|---:|"])
        for method in METHODS:
            selected = {row["missing_rate"]: row for row in aggregate_extreme if row["method"] == method}
            values = " | ".join(_fmt(selected.get(rate, {}).get("top1_mean")) for rate in EXTREME_CURVE_RATES)
            lines.append(f"| {method} | {values} |")
    lines.extend(
        [
            "",
            f"Requested seeds: {','.join(str(seed) for seed in seeds)}. AUC is trapezoidal Top1 over 0-80%, normalized by the 0.8 rate span.",
            "",
            "Bootstrap resamples domains within each fixed seed and then gives each seed equal weight; masks and samples are not treated as independent units.",
            "",
            "The table includes the main evaluation's 80% modality-frame result as a reference. The plotted extreme curve starts at 85% because the main and extreme evaluations use separate, non-nested fixed mask inventories; adjacent rates need not be monotonic.",
            "",
            "AMBER-Full is a local adaptation without historical beam input. RMBP-MM is an out-of-paper-scope local diagnostic without the paper's partial-beam and pretraining stages.",
        ]
    )
    return "\n".join(lines) + "\n"


def _identity(row):
    return tuple(str(row.get(field, "")).strip() for field in IDENTITY_FIELDS)


def _artifact_exists(raw: str) -> bool:
    path = Path(raw)
    return bool(raw) and path.exists()


def _rate(value: Any) -> float:
    return round(float(value), 6)


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: Any) -> int | None:
    number = _float(value)
    return int(number) if number is not None and number.is_integer() else None


def _mean(values):
    return sum(values) / len(values)


def _sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _quantile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if columns is None:
        columns = []
        for row in rows:
            columns.extend(key for key in row if key not in columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if columns:
            writer.writeheader()
            writer.writerows(rows)


def _curve_columns():
    return [
        "method", "seed", "aggregation", "curve_kind", "missing_rate", "top1", "top1_mean", "top1_std",
        "available_top1_mean", "available_seed_count", "aggregation_status", "reproduction_scope",
        "paper_equivalent", "temporal_result_scope",
    ]


def _paired_seed_columns():
    return [
        "baseline", "seed", "clean_top1_delta", "auc_top1_0_80_delta", "drop80_top1_delta",
        "baseline_reproduction_scope", "baseline_paper_equivalent", "baseline_temporal_result_scope",
    ]


def _paired_domain_columns():
    return ["baseline", "seed", "domain_id", "clean_top1_delta", "auc_top1_0_80_delta", "drop80_top1_delta"]


def _comparison_columns():
    base = [
        "baseline", "status", "requested_seed_count", "paired_seed_count", "paired_seeds", "auc_positive_seed_count",
        "gate_complete_seed_set", "gate_auc_positive_at_least_2_of_3", "gate_mean_clean_nonnegative",
        "gate_mean_auc_nonnegative", "gate_mean_drop80_nonnegative", "failed_gates", "baseline_reproduction_scope",
        "baseline_paper_equivalent", "baseline_temporal_result_scope",
    ]
    for metric in ("clean_top1_delta", "auc_top1_0_80_delta", "drop80_top1_delta"):
        base.extend((f"{metric}_mean", f"{metric}_std", f"{metric}_ci_low", f"{metric}_ci_high"))
    return base


def _mean_std(row, prefix):
    mean = _float(row.get(f"{prefix}_mean"))
    std = _float(row.get(f"{prefix}_std"))
    return "n/a" if mean is None or std is None else f"{mean:.4f} / {std:.4f}"


def _fmt(value):
    number = _float(value)
    return "n/a" if number is None else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
