#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from typing import Any


DEFAULT_METHODS = ("ours_c2_main", "ours_b4_nonrouter_soft_jepa", "ours_e5_low_lr_pcpg", "amber_full", "rmbp_mm")
S1_LIGHTWEIGHT_METHODS = ("S1", "T2", "T1", "A1", "A2", "A3", "T1+T2", "J1")
PROFILE_METHODS = {"default": DEFAULT_METHODS, "s1_lightweight": S1_LIGHTWEIGHT_METHODS}
PROFILE_ROOTS = {
    "default": Path("outputs/h5_p1_temporal_models_v1"),
    "s1_lightweight": Path("outputs/h5_p1_temporal_models_v1/s1_lightweight"),
}
MATRIX_FILES = {"top1": "top1_matrix.csv", "within3": "within3_matrix.csv", "mae": "mae_matrix.csv"}
S1_MATRIX_FILES = {**MATRIX_FILES, "top3": "top3_matrix.csv", "adba": "adba_matrix.csv"}
MATRIX_COLUMNS = ["missing_rate", "full", "drop1", "drop2", "drop3"]
METRIC_TITLES = {"top1": "Top1", "top3": "Top3", "within3": "Within@3", "adba": "ADBA", "mae": "MAE"}
PATTERN_METRIC_KEYS = {"top1": "top1", "top3": "top3", "within3": "within_3", "adba": "adba", "mae": "mae"}
DIAGNOSTIC_ID_COLUMNS = {"method", "seed", "missing_rate", "drop_count", "pattern"}
MASK_ID_COLUMNS = {
    "mask_index",
    "mask_type",
    "mask_digest",
    "mask_cache_checksum",
    "mask_cache_seed",
}
DIAGNOSTIC_ID_COLUMNS |= MASK_ID_COLUMNS
REQUIRED_RATES = (0.0, 0.2, 0.4, 0.6, 0.8)
DEFAULT_PAIRED_BASELINES = "T2:S1,T2-LG:S1-LG,T2-CLS:S1-CLS"
PAIR_METRICS = ("top1", "top3", "within_3", "adba", "mae")
DISTANCE_METRICS = {"within_3", "adba", "mae"}
MATCHED_PROVENANCE_FIELDS = (
    "training_beam_geometry",
    "prototype_target_geometry",
    "router_oracle_geometry",
    "head_type",
    "prototype_enabled",
    "metric_profile",
    "dba_distance_mode",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize H5/P1 temporal matrix v1 evaluation outputs.")
    parser.add_argument("--profile", choices=tuple(PROFILE_METHODS), default="default")
    parser.add_argument("--eval_dir", "--eval-dir", default=None)
    parser.add_argument("--output_dir", "--output-dir", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--guardrail_baseline", "--guardrail-baseline", default="S1")
    parser.add_argument("--drop0_guardrail", "--drop0-guardrail", type=float, default=0.005)
    parser.add_argument("--paired_baselines", "--paired-baselines", default=DEFAULT_PAIRED_BASELINES)
    parser.add_argument("--current_t2_method", "--current-t2-method", default="T2")
    args = parser.parse_args(argv)
    root = PROFILE_ROOTS[args.profile]
    eval_dir = Path(args.eval_dir) if args.eval_dir else root / "eval_matrix"
    out_dir = Path(args.output_dir) if args.output_dir else root / "final_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [item.strip() for item in (args.methods or ",".join(PROFILE_METHODS[args.profile])).split(",") if item.strip()]
    matrix_files = S1_MATRIX_FILES if args.profile == "s1_lightweight" else MATRIX_FILES
    summary_rows = []
    all_markdown = {metric: [] for metric in matrix_files}
    pattern_rows = []
    diagnostic_rows = []
    for method in methods:
        matrices = {}
        for metric, filename in matrix_files.items():
            rows = _aggregate_method(
                eval_dir / method,
                filename,
                distance_sensitive=metric in {"within3", "adba", "mae"},
            )
            matrices[metric] = rows
            _write_csv(out_dir / f"{method}_{metric}_matrix.csv", rows, MATRIX_COLUMNS)
            all_markdown[metric].append(_matrix_markdown(method, rows))
        diagnostic_row = _aggregate_diagnostics(eval_dir / method, method)
        diagnostic_rows.append(diagnostic_row)
        summary_rows.append({**_summary_row(method, matrices), **{key: value for key, value in diagnostic_row.items() if key != "method"}})
        for seed_dir in sorted((eval_dir / method).glob("seed*")):
            pattern_path = seed_dir / "pattern_metrics.csv"
            for row in _read_csv(pattern_path):
                row["method"] = method
                row["seed"] = seed_dir.name.removeprefix("seed")
                pattern_rows.append(row)
    include_drop0_guardrail = args.profile == "s1_lightweight"
    seed_summary_rows: list[dict[str, Any]] = []
    seed_delta_rows: list[dict[str, Any]] = []
    paired_mask_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    if include_drop0_guardrail:
        _apply_drop0_guardrail(summary_rows, str(args.guardrail_baseline), float(args.drop0_guardrail))
        seed_summary_rows = _seed_summary_rows(eval_dir, methods, matrix_files)
        pair_specs = _parse_pair_specs(str(args.paired_baselines), methods)
        paired_mask_rows, pair_status = _paired_mask_rows(pattern_rows, pair_specs)
        seed_delta_rows = _seed_delta_rows(seed_summary_rows, paired_mask_rows, pair_status, pair_specs)
        gate_rows = _gate_decision_rows(
            seed_summary_rows,
            seed_delta_rows,
            pair_specs,
            current_t2_method=str(args.current_t2_method),
            max_drop=float(args.drop0_guardrail),
        )
    _write_csv(out_dir / "summary.csv", summary_rows, _columns(summary_rows))
    _write_csv(out_dir / "pattern_metrics.csv", pattern_rows, _columns(pattern_rows))
    _write_csv(out_dir / "diagnostics.csv", diagnostic_rows, _columns(diagnostic_rows))
    if include_drop0_guardrail:
        _write_csv(out_dir / "seed_summary.csv", seed_summary_rows, _columns(seed_summary_rows))
        _write_csv(out_dir / "seed_deltas.csv", seed_delta_rows, _columns(seed_delta_rows))
        _write_csv(out_dir / "paired_mask_deltas.csv", paired_mask_rows, _columns(paired_mask_rows))
        _write_csv(out_dir / "gate_decisions.csv", gate_rows, _columns(gate_rows))
    for metric, chunks in all_markdown.items():
        (out_dir / f"all_methods_{metric}_matrices.md").write_text("\n".join(chunks) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(
        _summary_markdown(
            summary_rows,
            all_markdown,
            diagnostic_rows,
            include_drop0_guardrail=include_drop0_guardrail,
            seed_summary_rows=seed_summary_rows,
            gate_rows=gate_rows,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'summary.csv'}")
    print(f"wrote {out_dir / 'summary.md'}")
    return 0


def _aggregate_method(
    method_dir: Path,
    filename: str,
    *,
    distance_sensitive: bool = False,
) -> list[dict[str, Any]]:
    if distance_sensitive:
        modes = {
            str(row.get("dba_distance_mode", ""))
            for path in sorted(method_dir.glob("seed*/training_diagnostics.csv"))
            for row in _read_csv(path)
            if row.get("dba_distance_mode") not in {None, ""}
        }
        expected_seed_count = len(list(method_dir.glob(f"seed*/{filename}")))
        mode_seed_count = len(list(method_dir.glob("seed*/training_diagnostics.csv")))
        if len(modes) != 1 or mode_seed_count != expected_seed_count:
            return []
    by_rate: dict[str, dict[str, list[float]]] = {}
    for path in sorted(method_dir.glob(f"seed*/{filename}")):
        for row in _read_csv(path):
            rate = str(row.get("missing_rate", ""))
            by_rate.setdefault(rate, {column: [] for column in MATRIX_COLUMNS if column != "missing_rate"})
            for column in MATRIX_COLUMNS:
                if column == "missing_rate":
                    continue
                value = _float(row.get(column))
                if value is not None:
                    by_rate[rate][column].append(value)
    rows = []
    for rate in sorted(by_rate, key=lambda item: float(item) if item else -1.0):
        out = {"missing_rate": rate}
        for column, values in by_rate[rate].items():
            out[column] = "" if not values else f"{sum(values) / len(values):.6g}"
            out[f"{column}_std"] = "" if len(values) < 2 else f"{_std(values):.6g}"
        rows.append(out)
    return rows


def _seed_summary_rows(
    eval_dir: Path,
    methods: list[str],
    matrix_files: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        for seed_dir in sorted((eval_dir / method).glob("seed*"), key=lambda path: _seed_from_name(path.name)):
            seed = _seed_from_name(seed_dir.name)
            matrices = {metric: _read_csv(seed_dir / filename) for metric, filename in matrix_files.items()}
            top1 = matrices.get("top1", [])
            status = "complete"
            reasons: list[str] = []
            for metric, metric_rows in matrices.items():
                matrix_protocol_error = _matrix_protocol_error(metric_rows, metric=metric)
                if matrix_protocol_error:
                    status = "unavailable"
                    reasons.append(matrix_protocol_error)
                if any(_cell_value(metric_rows, rate, "full") is None for rate in REQUIRED_RATES):
                    status = "unavailable"
                    reasons.append(f"missing or non-finite {metric} full cell")
            diagnostics = _read_csv(seed_dir / "training_diagnostics.csv")
            provenance = diagnostics[0] if diagnostics else {}
            mask_stats = _read_csv(seed_dir / "mask_stats.csv")
            pattern_metrics = _read_csv(seed_dir / "pattern_metrics.csv")
            identity_index, identity_error = _entry_identity_index(pattern_metrics)
            protocol_error = _frozen_protocol_error(identity_index)
            if identity_error or protocol_error:
                status = "unavailable"
                reasons.extend(reason for reason in (identity_error, protocol_error) if reason)
            for metric, pattern_key in PATTERN_METRIC_KEYS.items():
                metric_rows = matrices.get(metric, [])
                for rate in REQUIRED_RATES:
                    raw_values = [
                        _float(item.get(pattern_key))
                        for item in pattern_metrics
                        if _float(item.get("missing_rate")) == rate and _int_value(item.get("drop_count")) == 0
                    ]
                    raw_mean = _mean_optional(raw_values)
                    matrix_value = _cell_value(metric_rows, rate, "full")
                    if raw_mean is None or matrix_value is None or abs(raw_mean - matrix_value) > 1e-12:
                        status = "unavailable"
                        reasons.append(
                            f"{metric} matrix/raw 4-entry mismatch at rate={rate}: "
                            f"matrix={matrix_value} raw={raw_mean}"
                        )
            provenance_keys = MATCHED_PROVENANCE_FIELDS
            if not diagnostics:
                status = "unavailable"
                reasons.append("training diagnostics provenance missing")
            for key in provenance_keys:
                values = {
                    str(item.get(key, ""))
                    for item in pattern_metrics
                    if item.get(key) not in {None, ""}
                }
                expected = str(provenance.get(key, ""))
                if len(values) != 1 or not expected or values != {expected}:
                    status = "unavailable"
                    reasons.append(f"mixed, missing, or inconsistent {key}: rows={sorted(values)} summary={expected!r}")
            row: dict[str, Any] = {
                "method": method,
                "seed": seed,
                "status": status,
                "reason": "; ".join(reasons),
                "mean_top1_five_rates": _required_rate_mean_value(top1),
                "mean_top1_drop0_60": _required_rate_mean_value(top1, max_rate=0.6),
                "top1_drop0": _cell_value(top1, 0.0, "full"),
                "top1_drop80": _cell_value(top1, 0.8, "full"),
                "mean_top3_five_rates": _required_rate_mean_value(matrices.get("top3", [])),
                "within3_mean_five_rates": _required_rate_mean_value(matrices.get("within3", [])),
                "mean_adba_five_rates": _required_rate_mean_value(matrices.get("adba", [])),
                "mae_mean_five_rates": _required_rate_mean_value(matrices.get("mae", [])),
                "mask_cache_entry_count": sum(int(_float(item.get("num_masks")) or 0) for item in mask_stats),
                "mask_cache_unique_count": sum(int(_float(item.get("num_unique_masks")) or 0) for item in mask_stats),
                "mask_cache_checksums": ",".join(
                    sorted(
                        {
                            str(item.get("mask_cache_checksum"))
                            for item in pattern_metrics
                            if item.get("mask_cache_checksum")
                        }
                    )
                ),
                "mask_cache_seeds": ",".join(
                    sorted(
                        {
                            str(item.get("mask_cache_seed"))
                            for item in pattern_metrics
                            if item.get("mask_cache_seed") not in {None, ""}
                        }
                    )
                ),
            }
            for key in provenance_keys:
                row[key] = provenance.get(key, "")
            rows.append(row)
    return rows


def _parse_pair_specs(value: str, methods: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for item in str(value or "").split(","):
        text = item.strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(f"paired baseline must be candidate:baseline, got {text!r}.")
        candidate, baseline = (part.strip() for part in text.split(":", 1))
        if not candidate or not baseline:
            raise ValueError(f"paired baseline must be candidate:baseline, got {text!r}.")
        if candidate in methods:
            specs.append((candidate, baseline))
    return specs


def _paired_mask_rows(
    pattern_rows: list[dict[str, Any]],
    pair_specs: list[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    by_method_seed: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in pattern_rows:
        method = str(row.get("method", ""))
        seed = _int_value(row.get("seed"))
        if method and seed is not None:
            by_method_seed.setdefault((method, seed), []).append(row)
    paired_rows: list[dict[str, Any]] = []
    statuses: dict[tuple[str, str, int], dict[str, Any]] = {}
    for candidate, baseline in pair_specs:
        seeds = sorted(
            {seed for method, seed in by_method_seed if method in {candidate, baseline}}
        )
        for seed in seeds:
            status_key = (candidate, baseline, seed)
            candidate_index, candidate_error = _entry_identity_index(by_method_seed.get((candidate, seed), []))
            baseline_index, baseline_error = _entry_identity_index(by_method_seed.get((baseline, seed), []))
            reasons = [reason for reason in (candidate_error, baseline_error) if reason]
            if not candidate_index or not baseline_index:
                reasons.append("candidate or baseline pattern rows missing")
            if set(candidate_index) != set(baseline_index):
                missing_candidate = len(set(baseline_index) - set(candidate_index))
                missing_baseline = len(set(candidate_index) - set(baseline_index))
                reasons.append(
                    f"entry identity mismatch: missing_candidate={missing_candidate} missing_baseline={missing_baseline}"
                )
            candidate_modes = {
                str(row.get("dba_distance_mode", ""))
                for row in candidate_index.values()
                if row.get("dba_distance_mode") not in {None, ""}
            }
            baseline_modes = {
                str(row.get("dba_distance_mode", ""))
                for row in baseline_index.values()
                if row.get("dba_distance_mode") not in {None, ""}
            }
            if len(candidate_modes) == 1 and len(baseline_modes) == 1 and candidate_modes != baseline_modes:
                reasons.append(
                    "candidate and baseline dba_distance_mode mismatch: "
                    f"{sorted(candidate_modes)} != {sorted(baseline_modes)}"
                )
            for field in MATCHED_PROVENANCE_FIELDS:
                candidate_values = {
                    str(row.get(field, ""))
                    for row in candidate_index.values()
                    if row.get(field) not in {None, ""}
                }
                baseline_values = {
                    str(row.get(field, ""))
                    for row in baseline_index.values()
                    if row.get(field) not in {None, ""}
                }
                if len(candidate_values) != 1 or len(baseline_values) != 1 or candidate_values != baseline_values:
                    reasons.append(
                        f"matched provenance mismatch for {field}: "
                        f"candidate={sorted(candidate_values)} baseline={sorted(baseline_values)}"
                    )
            for label, index in (("candidate", candidate_index), ("baseline", baseline_index)):
                protocol_error = _frozen_protocol_error(index)
                if protocol_error:
                    reasons.append(f"{label} {protocol_error}")
                modes = {
                    str(row.get("dba_distance_mode", ""))
                    for row in index.values()
                    if row.get("dba_distance_mode") not in {None, ""}
                }
                if len(modes) != 1:
                    reasons.append(f"{label} mixed or missing dba_distance_mode: {sorted(modes)}")
                for metric in PAIR_METRICS:
                    if any(_float(row.get(metric)) is None for row in index.values()):
                        reasons.append(f"{label} missing or non-finite {metric} metric")
            if reasons:
                statuses[status_key] = {"status": "unavailable", "reason": "; ".join(dict.fromkeys(reasons))}
                continue
            grouped: dict[tuple[float, int, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
            for identity in sorted(candidate_index, key=_identity_sort_key):
                candidate_row = candidate_index[identity]
                baseline_row = baseline_index[identity]
                rate = float(identity[0])
                drop_count = int(identity[1])
                digest = str(identity[4])
                grouped.setdefault((rate, drop_count, digest), []).append((candidate_row, baseline_row))
            for (rate, drop_count, digest), entries in sorted(grouped.items()):
                candidate_modes = {str(item[0].get("dba_distance_mode", "")) for item in entries}
                baseline_modes = {str(item[1].get("dba_distance_mode", "")) for item in entries}
                distance_compatible = (
                    len(candidate_modes) == 1
                    and len(baseline_modes) == 1
                    and candidate_modes == baseline_modes
                )
                out: dict[str, Any] = {
                    "pair": f"{candidate}:{baseline}",
                    "candidate_method": candidate,
                    "baseline_method": baseline,
                    "seed": seed,
                    "missing_rate": rate,
                    "drop_count": drop_count,
                    "mask_digest": digest,
                    "source_mask_indices": ",".join(
                        str(item[0].get("mask_index", "")) for item in entries
                    ),
                    "source_mask_types": ",".join(
                        str(item[0].get("mask_type", "")) for item in entries
                    ),
                    "duplicate_entry_count": len(entries),
                    "mask_cache_checksum": str(entries[0][0].get("mask_cache_checksum", "")),
                    "mask_cache_seed": str(entries[0][0].get("mask_cache_seed", "")),
                    "candidate_dba_distance_mode": ",".join(sorted(candidate_modes)),
                    "baseline_dba_distance_mode": ",".join(sorted(baseline_modes)),
                    "distance_metrics_status": "compatible" if distance_compatible else "incompatible",
                }
                for metric in PAIR_METRICS:
                    candidate_value = _mean_present(item[0].get(metric) for item in entries)
                    baseline_value = _mean_present(item[1].get(metric) for item in entries)
                    if metric in DISTANCE_METRICS and not distance_compatible:
                        candidate_value = None
                        baseline_value = None
                    out[f"{metric}_candidate"] = candidate_value
                    out[f"{metric}_baseline"] = baseline_value
                    out[f"{metric}_delta"] = (
                        None if candidate_value is None or baseline_value is None else candidate_value - baseline_value
                    )
                paired_rows.append(out)
            statuses[status_key] = {
                "status": "complete",
                "reason": "",
                "entry_count": len(candidate_index),
                "unique_mask_count": len(grouped),
            }
    return paired_rows, statuses


def _entry_identity_index(
    rows: list[dict[str, Any]],
) -> tuple[dict[tuple[Any, ...], dict[str, Any]], str]:
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        missing = [key for key in MASK_ID_COLUMNS if row.get(key) in {None, ""}]
        rate = _float(row.get("missing_rate"))
        drop_count = _int_value(row.get("drop_count"))
        if missing or rate is None or drop_count is None:
            return {}, f"missing mask identity fields: {','.join(sorted(missing)) or 'rate/drop_count'}"
        identity = (
            rate,
            drop_count,
            str(row.get("mask_index")),
            str(row.get("mask_type")),
            str(row.get("mask_digest")),
            str(row.get("mask_cache_checksum")),
            str(row.get("mask_cache_seed")),
        )
        if identity in index:
            return {}, f"duplicate full entry identity: {identity}"
        index[identity] = row
    return index, ""


def _frozen_protocol_error(index: dict[tuple[Any, ...], dict[str, Any]]) -> str:
    if not index:
        return "frozen protocol entries missing"
    counts: dict[tuple[float, int], int] = {}
    for identity in index:
        key = (float(identity[0]), int(identity[1]))
        counts[key] = counts.get(key, 0) + 1
    expected = {(rate, 0): 4 for rate in REQUIRED_RATES}
    if counts != expected:
        return f"frozen protocol mismatch: expected={expected} observed={counts}"
    return ""


def _matrix_protocol_error(rows: list[dict[str, Any]], *, metric: str) -> str:
    counts: dict[float, int] = {}
    invalid_rate_count = 0
    for row in rows:
        rate = _float(row.get("missing_rate"))
        if rate is None:
            invalid_rate_count += 1
            continue
        counts[rate] = counts.get(rate, 0) + 1
    expected = {rate: 1 for rate in REQUIRED_RATES}
    if counts == expected and invalid_rate_count == 0 and len(rows) == len(REQUIRED_RATES):
        return ""
    return (
        f"{metric} matrix rate rows mismatch: expected={expected} observed={counts} "
        f"invalid_rate_rows={invalid_rate_count} total_rows={len(rows)}"
    )


def _identity_sort_key(identity: tuple[Any, ...]) -> tuple[Any, ...]:
    return (float(identity[0]), int(identity[1]), int(identity[2]), str(identity[3]), str(identity[4]))


def _seed_delta_rows(
    seed_summary_rows: list[dict[str, Any]],
    paired_mask_rows: list[dict[str, Any]],
    pair_status: dict[tuple[str, str, int], dict[str, Any]],
    pair_specs: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    summaries = {
        (str(row.get("method")), int(row.get("seed"))): row
        for row in seed_summary_rows
        if _int_value(row.get("seed")) is not None
    }
    paired_by_pair_seed: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in paired_mask_rows:
        key = (
            str(row.get("candidate_method")),
            str(row.get("baseline_method")),
            int(row.get("seed")),
        )
        paired_by_pair_seed.setdefault(key, []).append(row)
    rows: list[dict[str, Any]] = []
    delta_fields = (
        "mean_top1_five_rates",
        "mean_top1_drop0_60",
        "top1_drop0",
        "top1_drop80",
        "mean_top3_five_rates",
        "within3_mean_five_rates",
        "mean_adba_five_rates",
        "mae_mean_five_rates",
    )
    distance_fields = {
        "within3_mean_five_rates",
        "mean_adba_five_rates",
        "mae_mean_five_rates",
    }
    for candidate, baseline in pair_specs:
        seeds = sorted(
            {seed for method, seed in summaries if method in {candidate, baseline}}
            | {seed for cand, base, seed in pair_status if cand == candidate and base == baseline}
        )
        for seed in seeds:
            candidate_summary = summaries.get((candidate, seed))
            baseline_summary = summaries.get((baseline, seed))
            strict = pair_status.get((candidate, baseline, seed), {})
            reasons: list[str] = []
            if candidate_summary is None or baseline_summary is None:
                reasons.append("candidate or baseline seed summary missing")
            elif candidate_summary.get("status") != "complete" or baseline_summary.get("status") != "complete":
                reasons.append("candidate or baseline seed summary incomplete")
            if strict.get("status") != "complete":
                reasons.append(str(strict.get("reason") or "paired mask evidence unavailable"))
            candidate_mode = str((candidate_summary or {}).get("dba_distance_mode", ""))
            baseline_mode = str((baseline_summary or {}).get("dba_distance_mode", ""))
            distance_compatible = bool(candidate_mode and candidate_mode == baseline_mode)
            candidate_cache = _cache_signature(candidate_summary)
            baseline_cache = _cache_signature(baseline_summary)
            if candidate_cache is None or baseline_cache is None:
                reasons.append("candidate or baseline cache provenance missing")
            elif candidate_cache != baseline_cache:
                reasons.append("candidate and baseline cache provenance mismatch")
            for field in MATCHED_PROVENANCE_FIELDS:
                candidate_value = str((candidate_summary or {}).get(field, ""))
                baseline_value = str((baseline_summary or {}).get(field, ""))
                if not candidate_value or not baseline_value or candidate_value != baseline_value:
                    reasons.append(
                        f"matched seed provenance mismatch for {field}: "
                        f"candidate={candidate_value!r} baseline={baseline_value!r}"
                    )
            row: dict[str, Any] = {
                "pair": f"{candidate}:{baseline}",
                "candidate_method": candidate,
                "baseline_method": baseline,
                "seed": seed,
                "status": "unavailable" if reasons else "complete",
                "reason": "; ".join(dict.fromkeys(reasons)),
                "candidate_dba_distance_mode": candidate_mode,
                "baseline_dba_distance_mode": baseline_mode,
                "distance_metrics_status": "compatible" if distance_compatible else "incompatible",
                "paired_entry_count": strict.get("entry_count", ""),
                "paired_unique_mask_count": strict.get("unique_mask_count", ""),
                "candidate_cache_signature": candidate_cache or "",
                "baseline_cache_signature": baseline_cache or "",
            }
            for field in MATCHED_PROVENANCE_FIELDS:
                row[f"candidate_{field}"] = (candidate_summary or {}).get(field, "")
                row[f"baseline_{field}"] = (baseline_summary or {}).get(field, "")
            for field in delta_fields:
                candidate_value = _float((candidate_summary or {}).get(field))
                baseline_value = _float((baseline_summary or {}).get(field))
                if field in distance_fields and not distance_compatible:
                    candidate_value = None
                    baseline_value = None
                row[f"{field}_candidate"] = candidate_value
                row[f"{field}_baseline"] = baseline_value
                row[f"{field}_delta"] = (
                    None if candidate_value is None or baseline_value is None else candidate_value - baseline_value
                )
            paired = paired_by_pair_seed.get((candidate, baseline, seed), [])
            row["paired_mean_top1_five_rates_delta"] = _paired_rate_equal_mean(
                paired,
                "top1_delta",
            )
            if row["paired_mean_top1_five_rates_delta"] is None:
                row["status"] = "unavailable"
                extra_reason = "paired evidence missing one or more required rates"
                row["reason"] = "; ".join(filter(None, (str(row["reason"]), extra_reason)))
            rows.append(row)
    return rows


def _gate_decision_rows(
    seed_summary_rows: list[dict[str, Any]],
    seed_delta_rows: list[dict[str, Any]],
    pair_specs: list[tuple[str, str]],
    *,
    current_t2_method: str,
    max_drop: float,
) -> list[dict[str, Any]]:
    summaries = {
        (str(row.get("method")), int(row.get("seed"))): row
        for row in seed_summary_rows
        if _int_value(row.get("seed")) is not None
    }
    deltas: dict[tuple[str, str, int], dict[str, Any]] = {
        (
            str(row.get("candidate_method")),
            str(row.get("baseline_method")),
            int(row.get("seed")),
        ): row
        for row in seed_delta_rows
    }
    rows: list[dict[str, Any]] = []
    current_seed1 = summaries.get((current_t2_method, 1))
    for candidate, baseline in pair_specs:
        if candidate != current_t2_method:
            seed1_delta = deltas.get((candidate, baseline, 1))
            candidate_seed1 = summaries.get((candidate, 1))
            reasons: list[str] = []
            if seed1_delta is None or seed1_delta.get("status") != "complete":
                reasons.append("seed1 paired delta unavailable")
            if candidate_seed1 is None or candidate_seed1.get("status") != "complete":
                reasons.append("candidate seed1 summary unavailable")
            if current_seed1 is None or current_seed1.get("status") != "complete":
                reasons.append("current T2 seed1 summary unavailable")
            mean5_delta = _float((seed1_delta or {}).get("mean_top1_five_rates_delta"))
            drop80_delta = _float((seed1_delta or {}).get("top1_drop80_delta"))
            drop0_delta = _float((seed1_delta or {}).get("top1_drop0_delta"))
            variant_mean5 = _float((candidate_seed1 or {}).get("mean_top1_five_rates"))
            current_mean5 = _float((current_seed1 or {}).get("mean_top1_five_rates"))
            vs_current = None if variant_mean5 is None or current_mean5 is None else variant_mean5 - current_mean5
            candidate_cache = _cache_signature(candidate_seed1)
            current_cache = _cache_signature(current_seed1)
            same_cache = candidate_cache is not None and candidate_cache == current_cache
            if candidate_cache is None or current_cache is None:
                reasons.append("candidate or current T2 cache provenance missing")
            elif not same_cache:
                reasons.append("candidate and current T2 cache provenance mismatch")
            checks = {
                "mean5_positive": mean5_delta is not None and mean5_delta > 0.0,
                "drop80_positive": drop80_delta is not None and drop80_delta > 0.0,
                "drop0_guardrail": drop0_delta is not None and drop0_delta >= -max_drop - 1e-12,
                "vs_current_t2_guardrail": vs_current is not None and vs_current >= -max_drop - 1e-12,
                "same_fixed_mask_cache": same_cache,
            }
            if reasons:
                status = "unavailable"
            else:
                status = "pass" if all(checks.values()) else "fail"
                reasons.extend(name for name, passed in checks.items() if not passed)
            rows.append(
                {
                    "pair": f"{candidate}:{baseline}",
                    "stage": "candidate_screen",
                    "status": status,
                    "reason": "; ".join(reasons),
                    "mean_top1_five_rates_delta": mean5_delta,
                    "top1_drop80_delta": drop80_delta,
                    "top1_drop0_delta": drop0_delta,
                    "variant_vs_current_t2_mean5_delta": vs_current,
                    "drop0_guardrail_limit": max_drop,
                    **{f"criterion_{name}": _check_status(passed) for name, passed in checks.items()},
                }
            )
        required_seeds = {1, 2, 3}
        available = {
            seed: deltas.get((candidate, baseline, seed))
            for seed in required_seeds
        }
        observed_seeds = {
            seed
            for cand, base, seed in deltas
            if cand == candidate and base == baseline
        }
        reasons = []
        if observed_seeds != required_seeds:
            reasons.append(
                f"required seeds mismatch: expected={sorted(required_seeds)} observed={sorted(observed_seeds)}"
            )
        if any(row is None or row.get("status") != "complete" for row in available.values()):
            reasons.append("one or more seed deltas unavailable")
        final_summaries = [
            summaries.get((method, seed))
            for method in (candidate, baseline)
            for seed in required_seeds
        ]
        cache_signatures = {_cache_signature(row) for row in final_summaries}
        same_cache = len(cache_signatures) == 1 and None not in cache_signatures
        if not same_cache:
            reasons.append("final seeds do not share one fixed mask cache")
        provenance_signatures = {_matched_provenance_signature(row) for row in final_summaries}
        same_provenance = len(provenance_signatures) == 1 and None not in provenance_signatures
        if not same_provenance:
            reasons.append("final seeds do not share one matched geometry/head/metric provenance")
        mean5_values = [_float((row or {}).get("mean_top1_five_rates_delta")) for row in available.values()]
        drop0_60_values = [_float((row or {}).get("mean_top1_drop0_60_delta")) for row in available.values()]
        drop80_values = [_float((row or {}).get("top1_drop80_delta")) for row in available.values()]
        drop0_values = [_float((row or {}).get("top1_drop0_delta")) for row in available.values()]
        mean5 = _mean_optional(mean5_values)
        drop0_60 = _mean_optional(drop0_60_values)
        drop80 = _mean_optional(drop80_values)
        drop0 = _mean_optional(drop0_values)
        positive_seed_count = sum(value is not None and value > 0.0 for value in mean5_values)
        checks = {
            "mean5_positive": mean5 is not None and mean5 > 0.0,
            "drop0_60_positive": drop0_60 is not None and drop0_60 > 0.0,
            "drop80_positive": drop80 is not None and drop80 > 0.0,
            "two_of_three_seeds_positive": positive_seed_count >= 2,
            "drop0_guardrail": drop0 is not None and drop0 >= -max_drop - 1e-12,
            "same_fixed_mask_cache": same_cache,
            "same_matched_provenance": same_provenance,
        }
        if reasons:
            status = "unavailable"
        else:
            status = "pass" if all(checks.values()) else "fail"
            reasons.extend(name for name, passed in checks.items() if not passed)
        rows.append(
            {
                "pair": f"{candidate}:{baseline}",
                "stage": "final_multiseed",
                "status": status,
                "reason": "; ".join(reasons),
                "mean_top1_five_rates_delta": mean5,
                "mean_top1_drop0_60_delta": drop0_60,
                "top1_drop80_delta": drop80,
                "top1_drop0_delta": drop0,
                "positive_seed_count": positive_seed_count,
                "drop0_guardrail_limit": max_drop,
                **{f"criterion_{name}": _check_status(passed) for name, passed in checks.items()},
            }
        )
    return rows


def _paired_rate_equal_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    by_rate: dict[float, list[float]] = {}
    for row in rows:
        rate = _float(row.get("missing_rate"))
        value = _float(row.get(key))
        if rate is not None and value is not None:
            by_rate.setdefault(rate, []).append(value)
    if set(by_rate) != set(REQUIRED_RATES):
        return None
    return sum(sum(values) / len(values) for values in by_rate.values()) / len(REQUIRED_RATES)


def _check_status(passed: bool) -> str:
    return "pass" if passed else "fail"


def _cache_signature(row: dict[str, Any] | None) -> str | None:
    if not row:
        return None
    checksums = str(row.get("mask_cache_checksums", "")).strip()
    seeds = str(row.get("mask_cache_seeds", "")).strip()
    entry_count = _int_value(row.get("mask_cache_entry_count"))
    if not checksums or not seeds or entry_count != 20:
        return None
    return f"checksums={checksums};seeds={seeds};entries={entry_count}"


def _matched_provenance_signature(row: dict[str, Any] | None) -> str | None:
    if not row:
        return None
    values = [str(row.get(field, "")).strip() for field in MATCHED_PROVENANCE_FIELDS]
    if any(not value for value in values):
        return None
    return ";".join(f"{field}={value}" for field, value in zip(MATCHED_PROVENANCE_FIELDS, values))


def _summary_row(method: str, matrices: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    top1 = matrices.get("top1", [])
    top3 = matrices.get("top3", [])
    within3 = matrices.get("within3", [])
    adba = matrices.get("adba", [])
    mae = matrices.get("mae", [])
    return {
        "method": method,
        "mean_top1_five_rates": _matrix_mean(top1, columns=("full",)),
        "mean_top1_drop0_60": _matrix_mean(top1, max_rate=0.6, columns=("full",)),
        "top1_drop0": _cell(top1, 0.0, "full"),
        "top1_drop80": _cell(top1, 0.8, "full"),
        "mean_top1_all_cells": _matrix_mean(top1),
        "mean_top1_rate20_80": _matrix_mean(top1, min_rate=0.2),
        "mean_top1_drop1_3": _matrix_mean(top1, columns=("drop1", "drop2", "drop3")),
        "mean_top1_severe_cells": _matrix_mean(top1, min_rate=0.6, columns=("drop2", "drop3")),
        "top1_full_0": _cell(top1, 0.0, "full"),
        "top1_drop3_80": _cell(top1, 0.8, "drop3"),
        "mean_top3_five_rates": _matrix_mean(top3, columns=("full",)),
        "within3_mean": _matrix_mean(within3),
        "mean_adba_five_rates": _matrix_mean(adba, columns=("full",)),
        "mae_mean": _matrix_mean(mae),
    }


def _summary_markdown(
    summary_rows: list[dict[str, Any]],
    all_markdown: dict[str, list[str]],
    diagnostic_rows: list[dict[str, Any]],
    *,
    include_drop0_guardrail: bool = False,
    seed_summary_rows: list[dict[str, Any]] | None = None,
    gate_rows: list[dict[str, Any]] | None = None,
) -> str:
    lines = ["# H5/P1 Temporal Matrix v1 Summary", ""]
    for metric, chunks in all_markdown.items():
        lines.append(f"## {METRIC_TITLES[metric]} Matrices")
        lines.extend(chunks)
    lines.append("## Method Comparison")
    lines.append(_table(summary_rows, _columns(summary_rows)))
    if include_drop0_guardrail:
        lines.append("## Drop0 Guardrail")
        lines.append(
            _table(
                summary_rows,
                ["method", "top1_drop0", "drop0_delta_vs_s1", "drop0_guardrail_limit", "drop0_guardrail_status"],
            )
        )
        if seed_summary_rows:
            lines.append("## Per-Seed Summary")
            lines.append(_table(seed_summary_rows, _columns(seed_summary_rows)))
        if gate_rows:
            lines.append("## Gate Decisions")
            lines.append(_table(gate_rows, _columns(gate_rows)))
    if any(len(row) > 1 for row in diagnostic_rows):
        lines.append("## Diagnostics")
        lines.append(_table(diagnostic_rows, _columns(diagnostic_rows)))
    lines.append("## 自动分析")
    lines.extend(_analysis(summary_rows))
    return "\n".join(lines) + "\n"


def _analysis(rows: list[dict[str, Any]]) -> list[str]:
    best_full = _best(rows, "top1_full_0")
    best_high = _best(rows, "mean_top1_rate20_80")
    best_drop3 = _best(rows, "top1_drop3_80")
    c2 = next((row for row in rows if row.get("method") == "ours_c2_main"), None)
    amber = next((row for row in rows if row.get("method") == "amber_full"), None)
    rmbp = next((row for row in rows if row.get("method") == "rmbp_mm"), None)
    return [
        f"- full 0% 最好: {best_full or '暂无数据'}。",
        f"- high temporal missing 下最好: {best_high or '暂无数据'}。",
        f"- drop3 + 80% 最好: {best_drop3 or '暂无数据'}。",
        f"- AMBER Full vs ours_c2_main gap: {_gap(c2, amber, 'mean_top1_all_cells')}。",
        f"- RMBP-MM vs ours_c2_main gap: {_gap(c2, rmbp, 'mean_top1_all_cells')}。",
        "- 时序缺失是否放大模态缺失: 查看 `mean_top1_rate20_80` 与 `mean_top1_drop1_3`，严重 cell 单独看 `mean_top1_severe_cells`。",
        "- C2 是否保持鲁棒: 优先看 `ours_c2_main` 在 severe cells 和 `top1_drop3_80` 的排名。",
        "- 是否需要 time-aware router: 若 masked_mean 下 drop3/high-rate 明显劣化，应把 time-aware router 作为下一轮 change。",
    ]


def _matrix_markdown(method: str, rows: list[dict[str, Any]]) -> str:
    return f"### {method}\n\n" + _table(rows, MATRIX_COLUMNS)


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_暂无数据_\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _matrix_mean(
    rows: list[dict[str, Any]],
    min_rate: float | None = None,
    max_rate: float | None = None,
    columns: tuple[str, ...] = ("full", "drop1", "drop2", "drop3"),
) -> str:
    values = []
    for row in rows:
        rate = _float(row.get("missing_rate"))
        if min_rate is not None and (rate is None or rate < min_rate):
            continue
        if max_rate is not None and (rate is None or rate > max_rate):
            continue
        for column in columns:
            value = _float(row.get(column))
            if value is not None:
                values.append(value)
    return "" if not values else f"{sum(values) / len(values):.6g}"


def _matrix_mean_value(
    rows: list[dict[str, Any]],
    min_rate: float | None = None,
    max_rate: float | None = None,
    columns: tuple[str, ...] = ("full", "drop1", "drop2", "drop3"),
) -> float | None:
    values: list[float] = []
    for row in rows:
        rate = _float(row.get("missing_rate"))
        if min_rate is not None and (rate is None or rate < min_rate):
            continue
        if max_rate is not None and (rate is None or rate > max_rate):
            continue
        for column in columns:
            value = _float(row.get(column))
            if value is not None:
                values.append(value)
    return None if not values else sum(values) / len(values)


def _required_rate_mean_value(
    rows: list[dict[str, Any]],
    *,
    max_rate: float | None = None,
) -> float | None:
    rates = [rate for rate in REQUIRED_RATES if max_rate is None or rate <= max_rate]
    values = [_cell_value(rows, rate, "full") for rate in rates]
    return _mean_optional(values)


def _cell(rows: list[dict[str, Any]], rate: float, column: str) -> str:
    for row in rows:
        value = _float(row.get("missing_rate"))
        if value is not None and abs(value - rate) < 1e-9:
            return str(row.get(column, ""))
    return ""


def _cell_value(rows: list[dict[str, Any]], rate: float, column: str) -> float | None:
    for row in rows:
        value = _float(row.get("missing_rate"))
        if value is not None and abs(value - rate) < 1e-9:
            return _float(row.get(column))
    return None


def _best(rows: list[dict[str, Any]], key: str) -> str:
    scored = [(row.get("method", ""), _float(row.get(key))) for row in rows]
    scored = [(method, value) for method, value in scored if value is not None]
    if not scored:
        return ""
    method, value = max(scored, key=lambda item: item[1])
    return f"{method} ({value:.6g})"


def _gap(c2: dict[str, Any] | None, other: dict[str, Any] | None, key: str) -> str:
    c2_value = _float((c2 or {}).get(key))
    other_value = _float((other or {}).get(key))
    if c2_value is None or other_value is None:
        return "暂无数据"
    return f"{other_value - c2_value:+.6g}"


def _aggregate_diagnostics(method_dir: Path, method: str) -> dict[str, Any]:
    rows = []
    for filename in ("training_diagnostics.csv", "diagnostics.csv", "router_diagnostics.csv"):
        for path in sorted(method_dir.glob(f"seed*/{filename}")):
            rows.extend(_read_csv(path))
    out: dict[str, Any] = {"method": method}
    for key in sorted({key for row in rows for key in row} - DIAGNOSTIC_ID_COLUMNS):
        values = [row.get(key) for row in rows if row.get(key) not in {None, ""}]
        numbers = [_float(value) for value in values]
        finite = [value for value in numbers if value is not None]
        if finite:
            out[key] = f"{sum(finite) / len(finite):.6g}"
        elif values and len({str(value) for value in values}) == 1:
            out[key] = str(values[0])
    return out


def _apply_drop0_guardrail(rows: list[dict[str, Any]], baseline_method: str, max_drop: float) -> None:
    baseline = next((row for row in rows if row.get("method") == baseline_method), None)
    baseline_value = _float((baseline or {}).get("top1_drop0"))
    for row in rows:
        value = _float(row.get("top1_drop0"))
        row["drop0_guardrail_limit"] = f"{max_drop:.6g}"
        if baseline_value is None or value is None:
            row["drop0_delta_vs_s1"] = ""
            row["drop0_guardrail_status"] = "unavailable"
        elif row.get("method") == baseline_method:
            row["drop0_delta_vs_s1"] = "0"
            row["drop0_guardrail_status"] = "baseline"
        else:
            delta = value - baseline_value
            row["drop0_delta_vs_s1"] = f"{delta:+.6g}"
            row["drop0_guardrail_status"] = "pass" if delta >= -max_drop else "fail"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_value(value: Any) -> int | None:
    number = _float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _seed_from_name(value: str) -> int:
    seed = _int_value(str(value).removeprefix("seed"))
    if seed is None:
        raise ValueError(f"Invalid seed directory name {value!r}.")
    return seed


def _mean_present(values: Any) -> float | None:
    return _mean_optional([_float(value) for value in values])


def _mean_optional(values: list[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    finite = [float(value) for value in values if value is not None]
    return sum(finite) / len(finite)


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


if __name__ == "__main__":
    raise SystemExit(main())
