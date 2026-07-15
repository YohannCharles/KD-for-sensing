#!/usr/bin/env python3
import argparse
import csv
import json
import math
import time
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize MMW all-weather missing-modality evaluation.")
    parser.add_argument("--eval-dir", default="outputs/mmw_all_weather_h5p1_seed1_v2/eval_matrix_v2")
    parser.add_argument("--append-eval-dir", action="append", default=[])
    parser.add_argument("--output-dir", default="outputs/mmw_all_weather_h5p1_seed1_v2/final_summary_v2")
    parser.add_argument("--coordinate-world-eval-dir", default=None)
    parser.add_argument("--coordinate-local-eval-dir", default=None)
    parser.add_argument("--wait-for-coordinate-pair", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if bool(args.coordinate_world_eval_dir) != bool(args.coordinate_local_eval_dir):
        parser.error("coordinate pair summary requires both world and local eval directories")
    if args.coordinate_world_eval_dir:
        if args.wait_for_coordinate_pair:
            _wait_for_coordinate_pair(
                Path(args.coordinate_world_eval_dir),
                Path(args.coordinate_local_eval_dir),
                poll_seconds=args.poll_seconds,
            )
        return _write_coordinate_pair_summary(
            Path(args.coordinate_world_eval_dir),
            Path(args.coordinate_local_eval_dir),
            output_dir,
        )
    eval_dir = Path(args.eval_dir)
    methods = ("S1", "T2", "amber_full", "rmbp_mm")
    eval_dirs = [eval_dir, *(Path(item) for item in args.append_eval_dir)]
    rows = [row for source in eval_dirs for method in methods for row in _read_csv(source / method / "metrics.csv")]
    if not rows:
        raise FileNotFoundError(f"No MMW evaluation rows under {eval_dir}")
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


def _wait_for_coordinate_pair(world_dir: Path, local_dir: Path, *, poll_seconds: int) -> None:
    targets = (world_dir / "T2" / "metrics.csv", local_dir / "T2" / "metrics.csv")
    statuses = (world_dir.parent / "eval_orchestrator_status.json", local_dir.parent / "eval_orchestrator_status.json")
    while not all(path.exists() for path in targets):
        for path in statuses:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            state = str(payload.get("state", ""))
            if state.startswith("blocked_"):
                raise RuntimeError(f"Coordinate evaluation failed before summary: {path}: {payload}")
        time.sleep(max(5, int(poll_seconds)))


def _write_coordinate_pair_summary(world_dir: Path, local_dir: Path, output_dir: Path) -> int:
    world_rows = _read_csv(world_dir / "T2" / "metrics.csv")
    local_rows = _read_csv(local_dir / "T2" / "metrics.csv")
    if not world_rows or not local_rows:
        raise FileNotFoundError("Coordinate pair summary requires T2/metrics.csv under both eval directories.")
    world = [{**row, "method": "T2_world"} for row in world_rows]
    local = [{**row, "method": "T2_local"} for row in local_rows]
    _validate_coordinate_pair_rows(world, local)
    rows = [*world, *local]
    cells = _domain_cells(rows)
    rollups = _rollups(cells)
    temporal = _temporal_rate_summary(rows)
    paired = _paired_deltas(rows, "T2_local", "T2_world")
    temporal_deltas = _paired_temporal_deltas(temporal, "T2_local", "T2_world")
    gps_branches = _gps_branch_summary(rollups)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "coordinate_paired_deltas.csv", paired)
    _write_csv(output_dir / "coordinate_rollups.csv", rollups)
    _write_csv(output_dir / "coordinate_temporal_summary.csv", temporal)
    _write_csv(output_dir / "coordinate_temporal_deltas.csv", temporal_deltas)
    _write_csv(output_dir / "gps_branch_summary.csv", gps_branches)
    summary = {
        "status": "complete",
        "world_row_count": len(world),
        "local_row_count": len(local),
        "paired_row_count": len(paired),
        "domain_count": len({row.get("domain_id", "") for row in rows}),
        "world_eval_dir": str(world_dir),
        "local_eval_dir": str(local_dir),
        "checkpoint_policy": "fixed_epoch_last_pth",
        "claim_eligibility": "seed1_local_validation_only",
    }
    (output_dir / "coordinate_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


def _validate_coordinate_pair_rows(world: list[dict[str, str]], local: list[dict[str, str]]) -> None:
    keys = (
        "domain_id",
        "eval_family",
        "pattern",
        "missing_rate",
        "mask_digest",
        "sample_csv_sha256",
        "mask_cache_checksum",
    )
    world_ids = [tuple(row.get(key, "") for key in keys) for row in world]
    local_ids = [tuple(row.get(key, "") for key in keys) for row in local]
    if len(world_ids) != len(set(world_ids)) or len(local_ids) != len(set(local_ids)):
        raise ValueError("Coordinate pair evaluation contains duplicate sample/mask identities.")
    if set(world_ids) != set(local_ids):
        missing_local = len(set(world_ids) - set(local_ids))
        missing_world = len(set(local_ids) - set(world_ids))
        raise ValueError(
            "Coordinate pair evaluation is not one-to-one paired; "
            f"missing_local={missing_local}, missing_world={missing_world}."
        )


def _paired_temporal_deltas(
    rows: list[dict[str, Any]],
    left: str,
    right: str,
) -> list[dict[str, Any]]:
    keys = ("missing_rate", "mask_type")
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
    return sorted(result, key=lambda row: (float(row["missing_rate"]), row["mask_type"]))


def _gps_branch_summary(rollups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    branch_patterns = {
        "full": "full",
        "gps_only": "available_gps",
        "missing_gps": "available_image_radar_lidar",
    }
    index = {
        (row.get("method"), row.get("pattern")): row
        for row in rollups
        if row.get("level") == "domain_macro" and row.get("eval_family") == "whole_modality"
    }
    result = []
    for branch, pattern in branch_patterns.items():
        world = index.get(("T2_world", pattern), {})
        local = index.get(("T2_local", pattern), {})
        out = {
            "branch": branch,
            "pattern": pattern,
            "domain_count": local.get("domain_count", world.get("domain_count", 0)),
        }
        for metric in METRICS:
            world_value = _float(world.get(metric))
            local_value = _float(local.get(metric))
            out[f"world_{metric}"] = "" if world_value is None else world_value
            out[f"local_{metric}"] = "" if local_value is None else local_value
            out[f"delta_{metric}"] = (
                "" if world_value is None or local_value is None else local_value - world_value
            )
        result.append(out)
    return result


def _domain_cells(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    keys = ("method", "domain_id", "condition", "scene", "eval_family", "pattern", "missing_rate", "available_modalities")
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
    group_keys = ("method", "eval_family", "pattern", "missing_rate", "available_modalities")
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
        key = (row.get("method", ""), row.get("missing_rate", ""), row.get("mask_type", ""), row.get("mask_digest", ""))
        mask_groups[key].append(row)
    mask_rows = []
    for key, items in mask_groups.items():
        method, rate, mask_type, digest = key
        out = {
            "method": method,
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
        type_groups[(row["method"], row["missing_rate"], row["mask_type"])].append(row)
    type_rows = []
    for (method, rate, mask_type), items in type_groups.items():
        type_rows.append(_summarize_mask_rows(method, rate, mask_type, items))

    result = list(type_rows)
    terminal_groups = defaultdict(list)
    for row in mask_rows:
        terminal = "terminal_available" if row["last_frame_available"] else "terminal_missing"
        terminal_groups[(row["method"], row["missing_rate"], terminal)].append(row)
    for (method, rate, terminal), items in terminal_groups.items():
        terminal_row = _summarize_mask_rows(method, rate, terminal, items)
        terminal_row["aggregation"] = "terminal_state_stratified_after_domain_macro"
        result.append(terminal_row)
    rate_groups = defaultdict(list)
    for row in type_rows:
        rate_groups[(row["method"], row["missing_rate"])].append(row)
    for (method, rate), items in rate_groups.items():
        out = {
            "method": method,
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
    return sorted(result, key=lambda row: (row["method"], float(row["missing_rate"]), row["mask_type"]))


def _summarize_mask_rows(method: str, rate: str, mask_type: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "method": method,
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
    keys = ("domain_id", "eval_family", "pattern", "missing_rate", "mask_digest", "sample_csv_sha256", "mask_cache_checksum")
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
