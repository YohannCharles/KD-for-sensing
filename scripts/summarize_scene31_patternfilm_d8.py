#!/usr/bin/env python3

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_scene31_bc_next


D8_METHOD = "proto_sampler_uniform_pattern_film_d8_es40"
UNIFORM_METHOD = "proto_sampler_uniform_es40"
UNIFORM_REFERENCE = {
    "full_top1": 0.4216,
    "avg_missing_top1": 0.2856,
    "overall_mean_top1": 0.2784,
    "balanced": 0.3560,
}
FULL_FLOOR = 0.4078


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = _roots(args)

    forwarded = ["--manifest", args.manifest, "--out", str(out_dir), "--name-prefix", ""]
    for root in roots:
        forwarded.extend(["--root", str(root)])
    for metrics in args.metrics:
        forwarded.extend(["--metrics", metrics])
    summarize_scene31_bc_next.main(forwarded)

    _copy_standard_outputs(out_dir)
    per_run_rows = _read_csv(out_dir / "per_run.csv")
    method_rows = _read_csv(out_dir / "method_mean_std.csv")
    status_summary = _fresh_eval_status_summary(roots)
    decision = _decision(method_rows, per_run_rows, status_summary)
    _annotate_main_read(method_rows, decision["decision"])
    _write_csv(out_dir / "patternfilm_method_mean_std.csv", method_rows, _fields(method_rows))
    _write_json(out_dir / "patternfilm_fresh_eval_status.json", status_summary)
    _write_sanity(out_dir / "patternfilm_sanity_check.md", method_rows, status_summary, decision)
    conclusion = _conclusion_lines(method_rows, status_summary, decision)
    (out_dir / "patternfilm_conclusion.txt").write_text("\n".join(conclusion) + "\n", encoding="utf-8")
    print("\n".join(conclusion))
    print(f"Wrote Scene31 PatternFiLM d8 summary to {out_dir}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Scene31 PatternFiLM d8 fresh eval metrics.")
    parser.add_argument("--root", action="append", default=None, help="Primary output root. Can repeat.")
    parser.add_argument("--extra-root", action="append", default=[], help="Additional eval/run root. Can repeat.")
    parser.add_argument("--metrics", action="append", default=[], help="Explicit metrics CSV. Can repeat.")
    parser.add_argument("--manifest", default="configs/scene31/funnel/experiment_manifest.csv")
    parser.add_argument("--out", default="outputs/scene31_funnel_lmdb/patternfilm_d8_summary")
    return parser


def _roots(args: argparse.Namespace) -> list[Path]:
    values = [*(args.root or ["outputs/scene31_funnel_lmdb"]), *args.extra_root]
    roots: list[Path] = []
    for value in values:
        root = Path(value)
        if root not in roots:
            roots.append(root)
        for child in ("patternfilm_d8_fresh_eval", "fresh_eval", "p0_fresh_eval"):
            candidate = root / child
            if candidate.exists() and candidate not in roots:
                roots.append(candidate)
    return roots


def _copy_standard_outputs(out_dir: Path) -> None:
    for src_name, dst_name in (
        ("per_run.csv", "patternfilm_per_run.csv"),
        ("method_mean_std.csv", "patternfilm_method_mean_std.csv"),
        ("delta_vs_uniform.csv", "patternfilm_delta_vs_uniform.csv"),
        ("missing_bucket_mapping.json", "patternfilm_missing_bucket_mapping.json"),
    ):
        src = out_dir / src_name
        if src.exists():
            shutil.copyfile(src, out_dir / dst_name)


def _fresh_eval_status_summary(roots: list[Path]) -> dict[str, Any]:
    seen: set[Path] = set()
    status = Counter()
    checkpoint_used = Counter()
    max_batches = Counter()
    strict_retry_runs: list[str] = []
    warning_runs: dict[str, list[str]] = {}
    for root in roots:
        for path in root.rglob("checkpoint_manifest.json"):
            if not (path.parent / "apples_to_apples_metrics.csv").exists():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                status["invalid_manifest"] += 1
                continue
            top_max_batches = data.get("max_batches", "")
            runs = data.get("runs") if isinstance(data.get("runs"), dict) else {}
            if not runs:
                run_name = path.parent.name
                runs = {run_name: data}
            for run_name, item in runs.items():
                if not isinstance(item, dict):
                    continue
                run_status = str(item.get("status") or data.get("status") or "unknown")
                status[run_status] += 1
                checkpoint_used[str(item.get("checkpoint_used") or "unknown")] += 1
                value = item.get("max_batches", top_max_batches)
                if value not in (None, ""):
                    max_batches[str(value)] += 1
                warnings = [str(value) for value in item.get("warnings", [])]
                warning_runs[str(run_name)] = warnings
                if any("strict checkpoint load failed" in value for value in warnings):
                    strict_retry_runs.append(str(run_name))
    return {
        "manifest_count": len(seen),
        "status_counts": dict(sorted(status.items())),
        "checkpoint_used_counts": dict(sorted(checkpoint_used.items())),
        "max_batches_counts": dict(sorted(max_batches.items())),
        "strict_retry_runs": sorted(dict.fromkeys(strict_retry_runs)),
        "warning_runs": {key: value for key, value in sorted(warning_runs.items()) if value},
    }


def _decision(
    method_rows: list[dict[str, str]],
    per_run_rows: list[dict[str, str]],
    status_summary: dict[str, Any],
) -> dict[str, Any]:
    by_method = {row.get("method", ""): row for row in method_rows}
    candidate_row = by_method.get(D8_METHOD)
    uniform, uniform_source = _uniform_reference(by_method.get(UNIFORM_METHOD))
    if candidate_row is None:
        return {"decision": "do_not_promote", "reason": "PatternFiLM d8 metrics unavailable", "uniform_source": uniform_source}
    candidate, candidate_source, excluded_runs = _candidate_decision_values(
        candidate_row,
        per_run_rows,
        set(status_summary.get("strict_retry_runs") or []),
    )
    n = int(_float(candidate.get("n")) if _isnum(candidate.get("n")) else 0)
    avg = _metric(candidate, "avg_missing_top1")
    overall = _metric(candidate, "overall_mean_top1")
    full = _metric(candidate, "full_top1")
    avg_delta = _delta(avg, uniform.get("avg_missing_top1"))
    overall_delta = _delta(overall, uniform.get("overall_mean_top1"))
    full_ok = _isnum(full) and full >= FULL_FLOOR
    proximity = _beam_proximity_improves(candidate, uniform)
    if n < 3:
        decision = "quick_screen_spike_not_promoted"
        reason = "fewer than 3 usable PatternFiLM d8 seeds"
    elif _isnum(avg_delta) and avg_delta > 0 and _isnum(overall_delta) and overall_delta > 0 and full_ok:
        decision = "promote_to_main_candidate"
        reason = "avg_missing and overall mean beat uniform while full top1 stays above floor"
    elif proximity:
        decision = "auxiliary_candidate_only"
        reason = "beam-proximity metric improves, but exact Top1 promotion gate is not met"
    else:
        decision = "do_not_promote"
        reason = "does not beat uniform on the required exact Top1 gates"
    return {
        "decision": decision,
        "reason": reason,
        "n": n,
        "avg_missing_top1": avg,
        "overall_mean_top1": overall,
        "full_top1": full,
        "avg_missing_delta_vs_uniform": avg_delta,
        "overall_mean_delta_vs_uniform": overall_delta,
        "candidate_source": candidate_source,
        "excluded_runs": excluded_runs,
        "uniform_source": uniform_source,
        "uniform_reference": uniform,
    }


def _candidate_decision_values(
    method_row: dict[str, str],
    per_run_rows: list[dict[str, str]],
    excluded_runs: set[str],
) -> tuple[dict[str, Any], str, list[str]]:
    rows = [row for row in per_run_rows if row.get("method") == D8_METHOD]
    formal_rows = [row for row in rows if row.get("run_name") not in excluded_runs]
    excluded = sorted(row.get("run_name", "") for row in rows if row.get("run_name") in excluded_runs)
    if rows and formal_rows and excluded:
        metrics = {
            "n": len(formal_rows),
            "full_top1": _mean(_metric(row, "full_top1") for row in formal_rows),
            "avg_missing_top1": _mean(_metric(row, "avg_missing_top1") for row in formal_rows),
            "overall_mean_top1": _mean(_metric(row, "overall_mean_top1") for row in formal_rows),
            "balanced": _mean(_metric(row, "balanced") for row in formal_rows),
            "avg_missing_within_3": _mean(_metric(row, "avg_missing_within_3") for row in formal_rows),
            "avg_missing_mae": _mean(_metric(row, "avg_missing_mae") for row in formal_rows),
        }
        return metrics, "fresh PatternFiLM-trained seeds only", excluded
    return method_row, "all evaluated PatternFiLM d8 rows", excluded


def _uniform_reference(row: dict[str, str] | None) -> tuple[dict[str, float], str]:
    if row is not None:
        n = int(_float(row.get("n")) if _isnum(row.get("n")) else 0)
        if n >= 5:
            return (
                {
                    "full_top1": _metric(row, "full_top1"),
                    "avg_missing_top1": _metric(row, "avg_missing_top1"),
                    "overall_mean_top1": _metric(row, "overall_mean_top1"),
                    "balanced": _metric(row, "balanced"),
                    "avg_missing_within_3": _metric(row, "avg_missing_within_3"),
                    "avg_missing_mae": _metric(row, "avg_missing_mae"),
                },
                f"actual {UNIFORM_METHOD} n={n}",
            )
    return dict(UNIFORM_REFERENCE), "fixed reference"


def _annotate_main_read(rows: list[dict[str, str]], decision: str) -> None:
    for row in rows:
        method = row.get("method", "")
        if method == D8_METHOD:
            row["main_read"] = decision
        elif method == UNIFORM_METHOD:
            row["main_read"] = "uniform_reference"
        elif method == "proto_sampler_uniform_pattern_film_d16_es40":
            row["main_read"] = "quick_screen_context_only"
        else:
            row.setdefault("main_read", "")


def _write_sanity(
    path: Path,
    method_rows: list[dict[str, str]],
    status_summary: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    lines = ["# PatternFiLM d8 Sanity Check", ""]
    warnings: list[str] = []
    by_method = {row.get("method", ""): row for row in method_rows}
    d8 = by_method.get(D8_METHOD)
    if d8 is None:
        warnings.append("PatternFiLM d8 method row is missing")
    elif int(_float(d8.get("n")) if _isnum(d8.get("n")) else 0) < 5:
        warnings.append(f"PatternFiLM d8 has n={d8.get('n')} usable seeds")
    if status_summary.get("max_batches_counts"):
        warnings.append(f"fresh eval recorded max_batches values: {status_summary['max_batches_counts']}")
    if status_summary.get("strict_retry_runs"):
        warnings.append("strict checkpoint fallback occurred for: " + ", ".join(status_summary["strict_retry_runs"]))
    if decision.get("excluded_runs"):
        warnings.append("formal d8 decision excludes compatibility eval runs: " + ", ".join(decision["excluded_runs"]))
    if decision.get("decision") != "promote_to_main_candidate":
        warnings.append(f"promotion gate not met: {decision.get('reason', '')}")
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- ok")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _conclusion_lines(
    method_rows: list[dict[str, str]],
    status_summary: dict[str, Any],
    decision: dict[str, Any],
) -> list[str]:
    by_method = {row.get("method", ""): row for row in method_rows}
    candidate = by_method.get(D8_METHOD, {})
    lines = [
        f"PatternFiLM d8 decision: {decision.get('decision', 'do_not_promote')}",
        f"Reason: {decision.get('reason', '')}",
        "",
        "PatternFiLM d8 formal metrics:",
        f"- source: {decision.get('candidate_source', '')}",
        f"- n: {decision.get('n', '')}",
        f"- full_top1_mean: {_fmt(decision.get('full_top1'))}",
        f"- avg_missing_top1_mean: {_fmt(decision.get('avg_missing_top1'))}",
        f"- overall_mean_top1_mean: {_fmt(decision.get('overall_mean_top1'))}",
        f"- delta_avg_missing_vs_uniform: {_fmt(decision.get('avg_missing_delta_vs_uniform'))}",
        f"- delta_overall_mean_vs_uniform: {_fmt(decision.get('overall_mean_delta_vs_uniform'))}",
        "",
        "PatternFiLM d8 all evaluated rows:",
        f"- n: {candidate.get('n', '')}",
        f"- full_top1_mean: {_fmt(_metric(candidate, 'full_top1'))}",
        f"- avg_missing_top1_mean: {_fmt(_metric(candidate, 'avg_missing_top1'))}",
        f"- overall_mean_top1_mean: {_fmt(_metric(candidate, 'overall_mean_top1'))}",
        "",
        f"Uniform reference: {decision.get('uniform_source', 'fixed reference')}",
        f"Fresh eval status counts: {status_summary.get('status_counts', {})}",
        f"Checkpoint policy counts: {status_summary.get('checkpoint_used_counts', {})}",
    ]
    strict_retry = status_summary.get("strict_retry_runs") or []
    if strict_retry:
        lines.extend(
            [
                "",
                "Compatibility note:",
                "Seed(s) with strict-load fallback are treated as compatibility eval evidence, not as final PatternFiLM training evidence: "
                + ", ".join(strict_retry),
            ]
        )
    return lines


def _beam_proximity_improves(candidate: dict[str, str], uniform: dict[str, float]) -> bool:
    within = _metric(candidate, "avg_missing_within_3")
    base_within = _float(uniform.get("avg_missing_within_3"))
    mae = _metric(candidate, "avg_missing_mae")
    base_mae = _float(uniform.get("avg_missing_mae"))
    return (_isnum(within) and _isnum(base_within) and within > base_within) or (
        _isnum(mae) and _isnum(base_mae) and mae < base_mae
    )


def _metric(row: dict[str, Any], name: str) -> float:
    return _float(row.get(f"{name}_mean", row.get(name)))


def _mean(values: Any) -> float:
    nums = [_float(value) for value in values if _isnum(value)]
    return sum(nums) / len(nums) if nums else float("nan")


def _delta(value: Any, base: Any) -> float:
    value_f = _float(value)
    base_f = _float(base)
    return value_f - base_f if _isnum(value_f) and _isnum(base_f) else float("nan")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _isnum(value: Any) -> bool:
    return math.isfinite(_float(value))


def _fmt(value: Any) -> str:
    value_f = _float(value)
    return f"{value_f:.8g}" if math.isfinite(value_f) else ""


if __name__ == "__main__":
    raise SystemExit(main())
