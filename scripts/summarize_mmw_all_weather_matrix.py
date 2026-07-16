#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = (
    "top1",
    "top3",
    "top5",
    "within_3",
    "adba",
    "mae",
    "gate_entropy",
    "mean_gate_gps",
)
_REQUIRED_EVIDENCE_FIELDS = (
    "method",
    "seed",
    "domain_id",
    "sample_csv_sha256",
    "checkpoint_sha256",
    "checkpoint_role",
    "mask_digest",
    "metric_profile",
    "coverage_status",
    "sample_count",
    "expected_sample_count",
    "expected_domain_count",
    "training_profile_id",
    "training_profile_sha256",
    "design_candidate_id",
    "design_config_sha256",
)
_NONEMPTY_EVIDENCE_FIELDS = (
    "method",
    "seed",
    "domain_id",
    "sample_csv_sha256",
    "checkpoint_sha256",
    "checkpoint_role",
    "mask_digest",
    "metric_profile",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize MMW all-weather missing-modality evaluation.")
    parser.add_argument("--eval-dir", default="outputs/mmw_all_weather_h5p1_seed1_v2/eval_matrix_v2")
    parser.add_argument("--append-eval-dir", action="append", default=[])
    parser.add_argument("--output-dir", default="outputs/mmw_all_weather_h5p1_seed1_v2/final_summary_v2")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    eval_dir = Path(args.eval_dir)
    methods = ("S1", "T2", "amber_full", "rmbp_mm")
    eval_dirs = [eval_dir, *(Path(item) for item in args.append_eval_dir)]
    rows = [row for source in eval_dirs for method in methods for row in _read_method_rows(source, method)]
    if not rows:
        raise FileNotFoundError(f"No MMW evaluation rows under {eval_dir}")
    _validate_complete_evidence(rows)
    cells = _domain_cells(rows)
    rollups = _rollups(cells)
    temporal = _temporal_rate_summary(rows)
    paired = _paired_deltas(rows, "T2", "S1")
    decision = _weather_gate(cells)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "domain_cells.csv", cells)
    _write_csv(output_dir / "rollups.csv", rollups)
    _write_csv(output_dir / "temporal_rate_summary.csv", temporal)
    _write_csv(output_dir / "t2_vs_s1_paired_deltas.csv", paired)
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(_markdown(rollups, temporal, decision), encoding="utf-8")
    return 0


def _validate_complete_evidence(rows: list[dict[str, str]]) -> None:
    duplicate_identities: set[tuple[str, ...]] = set()
    seen_identities: set[tuple[str, ...]] = set()
    coverage_groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    comparison_profiles: dict[tuple[str, ...], set[str]] = defaultdict(set)
    method_provenance: dict[tuple[str, str], set[tuple[str, str, str, str, str]]] = defaultdict(set)
    for row in rows:
        missing = [field for field in _REQUIRED_EVIDENCE_FIELDS if field not in row]
        empty = [field for field in _NONEMPTY_EVIDENCE_FIELDS if not str(row.get(field, "")).strip()]
        if missing or empty:
            raise ValueError(
                "MMW summary requires complete evidence identity fields: "
                f"missing={missing}, empty={empty}, method={row.get('method', '')}, domain={row.get('domain_id', '')}."
            )
        if row.get("coverage_status") != "complete" or _truthy(row.get("partial_request")):
            raise ValueError(
                "MMW summary refuses partial evaluation evidence: "
                f"method={row['method']}, domain={row['domain_id']}, coverage_status={row.get('coverage_status')}."
            )
        try:
            sample_count = int(float(row["sample_count"]))
            expected_sample_count = int(float(row["expected_sample_count"]))
            expected_domain_count = int(float(row["expected_domain_count"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"MMW summary found an invalid coverage count in {row['method']}/{row['domain_id']}.") from exc
        if sample_count <= 0 or expected_sample_count <= 0 or sample_count != expected_sample_count or expected_domain_count <= 0:
            raise ValueError(
                "MMW summary refuses incomplete sample coverage: "
                f"method={row['method']}, domain={row['domain_id']}, observed={sample_count}, expected={expected_sample_count}."
            )
        identity = tuple(
            str(row.get(field, ""))
            for field in (
                "method",
                "seed",
                "domain_id",
                "eval_family",
                "pattern",
                "missing_rate",
                "available_modalities",
                "mask_digest",
                "sample_csv_sha256",
                "mask_cache_checksum",
                "checkpoint_sha256",
                "metric_profile",
            )
        )
        if identity in seen_identities:
            duplicate_identities.add(identity)
        seen_identities.add(identity)
        coverage_key = tuple(
            str(row.get(field, ""))
            for field in (
                "method",
                "seed",
                "training_profile_id",
                "training_profile_sha256",
                "design_candidate_id",
                "design_config_sha256",
                "checkpoint_sha256",
                "checkpoint_role",
                "metric_profile",
                "eval_family",
                "pattern",
                "missing_rate",
                "available_modalities",
                "mask_digest",
                "mask_cache_checksum",
            )
        )
        coverage_groups[coverage_key].append(row)
        comparison_key = tuple(
            str(row.get(field, ""))
            for field in ("seed", "eval_family", "pattern", "missing_rate", "available_modalities", "mask_digest", "mask_cache_checksum")
        )
        comparison_profiles[comparison_key].add(str(row["metric_profile"]))
        method_provenance[(str(row["method"]), str(row["seed"]))].add(
            tuple(
                str(row.get(field, ""))
                for field in (
                    "training_profile_id",
                    "training_profile_sha256",
                    "design_candidate_id",
                    "design_config_sha256",
                    "checkpoint_sha256",
                )
            )
        )
    if duplicate_identities:
        raise ValueError(f"MMW summary found duplicate evidence rows: {sorted(duplicate_identities)[0]}.")
    for key, items in coverage_groups.items():
        expected_values = {int(float(item["expected_domain_count"])) for item in items}
        domain_ids = {str(item["domain_id"]) for item in items}
        if len(expected_values) != 1 or len(domain_ids) != next(iter(expected_values)):
            raise ValueError(
                "MMW summary refuses incomplete domain coverage for evidence cell: "
                f"identity={key}, observed_domains={len(domain_ids)}, expected={sorted(expected_values)}."
            )
    for key, profiles in comparison_profiles.items():
        if len(profiles) != 1:
            raise ValueError(f"MMW summary cannot compare conflicting metric profiles {sorted(profiles)} for {key}.")
    for key, identities in method_provenance.items():
        if len(identities) != 1:
            raise ValueError(f"MMW summary found mixed profile/candidate/checkpoint provenance for {key}.")


def _domain_cells(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    keys = (
        "method",
        "seed",
        "training_profile_id",
        "training_profile_sha256",
        "design_candidate_id",
        "design_config_sha256",
        "checkpoint_sha256",
        "checkpoint_role",
        "metric_profile",
        "domain_id",
        "condition",
        "scene",
        "eval_family",
        "pattern",
        "missing_rate",
        "available_modalities",
    )
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    result = []
    for key, items in grouped.items():
        out = dict(zip(keys, key))
        out["sample_count"] = int(float(items[0].get("sample_count", 0) or 0))
        out["mask_count"] = len(items)
        out["sample_csv_sha256"] = items[0].get("sample_csv_sha256", "")
        out["mask_cache_checksums"] = ",".join(sorted({item.get("mask_cache_checksum", "") for item in items}))
        for metric in METRICS:
            values = [_float(item.get(metric)) for item in items]
            values = [value for value in values if value is not None]
            out[metric] = sum(values) / len(values) if values else ""
        result.append(out)
    return result


def _rollups(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    group_keys = (
        "method",
        "seed",
        "training_profile_id",
        "training_profile_sha256",
        "design_candidate_id",
        "design_config_sha256",
        "checkpoint_sha256",
        "checkpoint_role",
        "metric_profile",
        "eval_family",
        "pattern",
        "missing_rate",
        "available_modalities",
    )
    grouped = defaultdict(list)
    for row in cells:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)
    for key, items in grouped.items():
        base = dict(zip(group_keys, key))
        result.extend(_aggregate(base, "domain_macro", "all", items))
        for weather in sorted({item["condition"] for item in items}):
            result.extend(_aggregate(base, "weather_macro", weather, [item for item in items if item["condition"] == weather]))
        for scene in sorted({item["scene"] for item in items}):
            result.extend(_aggregate(base, "scene_macro", scene, [item for item in items if item["scene"] == scene]))
    return result


def _temporal_rate_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    mask_groups = defaultdict(list)
    for row in rows:
        if row.get("eval_family") != "temporal_missing":
            continue
        key = (
            row.get("method", ""),
            row.get("seed", ""),
            row.get("metric_profile", ""),
            row.get("missing_rate", ""),
            row.get("mask_type", ""),
            row.get("mask_digest", ""),
        )
        mask_groups[key].append(row)
    mask_rows = []
    for key, items in mask_groups.items():
        method, seed, metric_profile, rate, mask_type, digest = key
        out = {
            "method": method,
            "seed": seed,
            "metric_profile": metric_profile,
            "missing_rate": rate,
            "mask_type": mask_type,
            "mask_digest": digest,
            "domain_count": len({item.get("domain_id", "") for item in items}),
            "sample_count": sum(int(float(item.get("sample_count", 0) or 0)) for item in items),
            "observed_missing_rate": _mean_values(items, "observed_missing_rate"),
            "last_frame_available": _truthy(items[0].get("last_frame_available")),
            "last_frame_available_modalities": int(float(items[0].get("last_frame_available_modalities", 0) or 0)),
            "trailing_fully_missing_frames": int(float(items[0].get("trailing_fully_missing_frames", 0) or 0)),
            "reproduction_scope": items[0].get("reproduction_scope", ""),
            "paper_equivalent": _truthy(items[0].get("paper_equivalent")),
            "temporal_result_scope": items[0].get("temporal_result_scope", ""),
        }
        for metric in METRICS:
            out[metric] = _mean_values(items, metric)
        mask_rows.append(out)

    type_groups = defaultdict(list)
    for row in mask_rows:
        type_groups[(row["method"], row["seed"], row["metric_profile"], row["missing_rate"], row["mask_type"])].append(row)
    type_rows = []
    for (method, seed, metric_profile, rate, mask_type), items in type_groups.items():
        type_rows.append(_summarize_mask_rows(method, seed, metric_profile, rate, mask_type, items))

    result = list(type_rows)
    terminal_groups = defaultdict(list)
    for row in mask_rows:
        terminal = "terminal_available" if row["last_frame_available"] else "terminal_missing"
        terminal_groups[(row["method"], row["seed"], row["metric_profile"], row["missing_rate"], terminal)].append(row)
    for (method, seed, metric_profile, rate, terminal), items in terminal_groups.items():
        terminal_row = _summarize_mask_rows(method, seed, metric_profile, rate, terminal, items)
        terminal_row["aggregation"] = "terminal_state_stratified_after_domain_macro"
        result.append(terminal_row)
    rate_groups = defaultdict(list)
    for row in type_rows:
        rate_groups[(row["method"], row["seed"], row["metric_profile"], row["missing_rate"])].append(row)
    for (method, seed, metric_profile, rate), items in rate_groups.items():
        out = {
            "method": method,
            "seed": seed,
            "metric_profile": metric_profile,
            "missing_rate": rate,
            "mask_type": "type_equal_all",
            "mask_count": sum(int(item["mask_count"]) for item in items),
            "domain_count_min": min(int(item["domain_count_min"]) for item in items),
            "domain_count_max": max(int(item["domain_count_max"]) for item in items),
            "observed_missing_rate": sum(float(item["observed_missing_rate"]) for item in items) / len(items),
            "last_frame_unavailable_masks": sum(int(item["last_frame_unavailable_masks"]) for item in items),
            "reproduction_scope": items[0]["reproduction_scope"],
            "paper_equivalent": items[0]["paper_equivalent"],
            "temporal_result_scope": items[0]["temporal_result_scope"],
            "claim_status": _claim_status(items[0]["temporal_result_scope"]),
            "aggregation": "equal_weight_per_mask_type",
        }
        for metric in METRICS:
            values = [_float(item.get(metric)) for item in items]
            values = [value for value in values if value is not None]
            out[metric] = sum(values) / len(values) if values else ""
            out[f"{metric}_std_across_masks"] = ""
            out[f"{metric}_worst_mask"] = ""
        result.append(out)
    return sorted(result, key=lambda row: (row["method"], int(row["seed"]), float(row["missing_rate"]), row["mask_type"]))


def _summarize_mask_rows(
    method: str,
    seed: str,
    metric_profile: str,
    rate: str,
    mask_type: str,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    out = {
        "method": method,
        "seed": seed,
        "metric_profile": metric_profile,
        "missing_rate": rate,
        "mask_type": mask_type,
        "mask_count": len(items),
        "domain_count_min": min(int(item["domain_count"]) for item in items),
        "domain_count_max": max(int(item["domain_count"]) for item in items),
        "observed_missing_rate": sum(float(item["observed_missing_rate"]) for item in items) / len(items),
        "last_frame_unavailable_masks": sum(not bool(item["last_frame_available"]) for item in items),
        "reproduction_scope": items[0]["reproduction_scope"],
        "paper_equivalent": items[0]["paper_equivalent"],
        "temporal_result_scope": items[0]["temporal_result_scope"],
        "claim_status": _claim_status(items[0]["temporal_result_scope"]),
        "aggregation": "mean_over_unique_masks_after_domain_macro",
    }
    for metric in METRICS:
        values = [_float(item.get(metric)) for item in items]
        values = [value for value in values if value is not None]
        out[metric] = sum(values) / len(values) if values else ""
        out[f"{metric}_std_across_masks"] = _std(values)
        out[f"{metric}_worst_mask"] = (max(values) if metric == "mae" else min(values)) if values else ""
    return out


def _mean_values(rows: list[dict[str, Any]], key: str) -> float:
    values = [_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else float("nan")


def _claim_status(scope: str) -> str:
    if scope == "out_of_paper_scope_diagnostic":
        return "diagnostic_only_not_paper_equivalent"
    if scope == "local_adaptation_diagnostic":
        return "local_adaptation_only"
    return "local_validation_only"


def _std(values: list[float]) -> float | str:
    if not values:
        return ""
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _aggregate(base: dict[str, Any], level: str, group: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = {**base, "level": level, "group": group, "domain_count": len(items), "sample_count": sum(item["sample_count"] for item in items)}
    for metric in METRICS:
        values = [(item.get(metric), item["sample_count"]) for item in items if _float(item.get(metric)) is not None]
        out[metric] = sum(float(value) for value, _ in values) / len(values) if values else ""
        out[f"{metric}_micro"] = (
            sum(float(value) * weight for value, weight in values) / sum(weight for _, weight in values) if values else ""
        )
        out[f"{metric}_worst_domain"] = min(float(value) for value, _ in values) if values else ""
    return [out]


def _paired_deltas(rows: list[dict[str, str]], left: str, right: str) -> list[dict[str, Any]]:
    keys = (
        "seed",
        "metric_profile",
        "domain_id",
        "eval_family",
        "pattern",
        "missing_rate",
        "mask_digest",
        "sample_csv_sha256",
        "mask_cache_checksum",
    )
    index = {(row.get("method"), *(row.get(key, "") for key in keys)): row for row in rows}
    result = []
    for identity, left_row in index.items():
        if identity[0] != left:
            continue
        right_row = index.get((right, *identity[1:]))
        if right_row is None:
            continue
        out = {"method": left, "baseline": right, **dict(zip(keys, identity[1:]))}
        for metric in METRICS:
            left_value = _float(left_row.get(metric))
            right_value = _float(right_row.get(metric))
            out[f"delta_{metric}"] = "" if left_value is None or right_value is None else left_value - right_value
        result.append(out)
    return result


def _weather_gate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in cells if row["eval_family"] == "temporal_missing"]
    by_method_weather = defaultdict(list)
    for row in selected:
        value = _float(row.get("top1"))
        if value is not None:
            by_method_weather[(row["method"], row["condition"])].append(value)
    comparisons = {}
    passed = True
    for weather in ("rainy", "foggy"):
        t2 = by_method_weather.get(("T2", weather), [])
        s1 = by_method_weather.get(("S1", weather), [])
        delta = (sum(t2) / len(t2) - sum(s1) / len(s1)) if t2 and s1 else None
        comparisons[weather] = {"t2_minus_s1_temporal_top1": delta, "tolerance": -0.005}
        passed = passed and delta is not None and delta >= -0.005
    return {
        "status": "keep_t2_no_weather_module" if passed else "diagnose_reliability_failure_before_new_module",
        "comparisons": comparisons,
        "claim_eligibility": "local_validation_only",
        "rule": "Do not add a weather-specific module unless repeated paired domain diagnostics show the same failure mode.",
    }


def _markdown(rollups: list[dict[str, Any]], temporal: list[dict[str, Any]], decision: dict[str, Any]) -> str:
    lines = ["# MMW all-weather validation summary", "", f"Decision: `{decision['status']}`", "", "## Clean domain macro", "", "| Method | Top1 | Worst domain |", "|---|---:|---:|"]
    for method in ("S1", "T2", "amber_full", "rmbp_mm"):
        row = next(
            (
                item for item in rollups
                if item["method"] == method and item["level"] == "domain_macro" and item["eval_family"] == "whole_modality" and item["pattern"] == "full"
            ),
            {},
        )
        lines.append(f"| {method} | {_fmt(row.get('top1'))} | {_fmt(row.get('top1_worst_domain'))} |")
    rates = sorted(
        {
            float(item["missing_rate"])
            for item in temporal
            if item["mask_type"] == "type_equal_all"
        }
    )
    rate_headers = " | ".join(f"{rate:.0%}" for rate in rates)
    lines.extend(
        [
            "",
            "## Temporal missing Top1",
            "",
            f"| Method | {rate_headers} | Scope |",
            "|---|" + "---:|" * len(rates) + "---|",
        ]
    )
    for method in ("S1", "T2", "amber_full", "rmbp_mm"):
        selected = {
            float(item["missing_rate"]): item
            for item in temporal
            if item["method"] == method and item["mask_type"] == "type_equal_all"
        }
        scope = next(iter(selected.values()), {}).get("claim_status", "n/a")
        values = " | ".join(_fmt(selected.get(rate, {}).get("top1")) for rate in rates)
        lines.append(f"| {method} | {values} | {scope} |")
    lines.extend(
        [
            "",
            "Temporal values use equal weight per mask type after per-mask 15-domain macro aggregation. See `temporal_rate_summary.csv` for mask counts, standard deviation, worst mask, and terminal-frame coverage.",
            "",
            "RMBP-MM temporal rows are out-of-paper-scope diagnostics because the original model is single-time and has no temporal aggregator.",
            "",
            "Results are local validation evidence from fixed-epoch `last.pth`; they are not a formal multi-seed test claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_method_rows(eval_dir: Path, method: str) -> list[dict[str, str]]:
    method_dir = eval_dir / method
    paths = [method_dir / "metrics.csv", *sorted(method_dir.glob("seed*/metrics.csv"))]
    return [row for path in paths for row in _read_csv(path)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _fmt(value: Any) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
