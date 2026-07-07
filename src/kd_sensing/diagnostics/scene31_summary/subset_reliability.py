#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from kd_sensing.diagnostics.scene31_summary import bc_next as bc
from kd_sensing.eval.missing_buckets import missing_bucket_mapping_from_rows, write_missing_bucket_mapping


SUBSET_METHOD = "proto_randomdrop_subset_es40"
UNIFORM_METHOD = "proto_sampler_uniform_es40"
FIXED_SUBSET_REFERENCE = {
    "method": SUBSET_METHOD,
    "n": 0,
    "full_top1_mean": 0.4220,
    "avg_missing_top1_mean": 0.3055,
}
DELTA_METRICS = (
    "full_top1",
    "avg_missing_top1",
    "miss1_top1",
    "miss2_top1",
    "miss3_top1",
    "overall_mean_top1",
    "avg_missing_within_3",
    "avg_missing_mae",
    "balanced",
)
PRIMARY_SORT = ("avg_missing_top1", "miss2_top1", "miss3_top1", "full_top1")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summarize(
        roots=[Path(args.baseline_root), Path(args.new_root)],
        out_dir=Path(args.out),
        output_prefix="combined",
        conclusion_name="combined_conclusion.txt",
        include_combined_sections=True,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Scene31 subset reliability and PatternFiLM fresh eval metrics.")
    parser.add_argument("--baseline-root", default="outputs/scene31_baseline_pack_lmdb")
    parser.add_argument("--new-root", default="outputs/scene31_subset_reliability_lmdb")
    parser.add_argument("--out", default="outputs/scene31_subset_reliability_lmdb/summary")
    return parser


def summarize(
    *,
    roots: list[Path],
    out_dir: Path,
    output_prefix: str,
    conclusion_name: str,
    include_combined_sections: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = _load_metric_rows(roots)
    bucket_mapping, bucket_warnings = missing_bucket_mapping_from_rows(metric_rows)
    write_missing_bucket_mapping(out_dir / "missing_bucket_mapping.json", bucket_mapping)
    per_run = bc._per_run_rows(metric_rows, {}, bucket_mapping)
    _attach_mask_suspect(per_run, metric_rows)
    trusted_per_run = [
        row for row in per_run
        if not _truthy(row.get("mask_suspect")) and not _truthy(row.get("excluded_from_official_ranking"))
    ]
    method_rows = bc._method_rows(trusted_per_run)
    _attach_method_read_and_suspect_counts(method_rows, per_run)
    reference, reference_warning = _subset_reference(method_rows)
    delta_rows = _delta_rows(method_rows, reference)
    _write_outputs(
        out_dir,
        output_prefix=output_prefix,
        per_run=per_run,
        method_rows=method_rows,
        delta_rows=delta_rows,
        reference=reference,
        warnings=[*bucket_warnings, *([reference_warning] if reference_warning else [])],
    )
    conclusion = _conclusion_lines(
        method_rows,
        per_run,
        reference,
        reference_warning=reference_warning,
        include_combined_sections=include_combined_sections,
    )
    (out_dir / conclusion_name).write_text("\n".join(conclusion) + "\n", encoding="utf-8")
    _print_mask_status(per_run)
    print(f"Wrote Scene31 subset summary to {out_dir}.")
    return {
        "per_run": per_run,
        "method_rows": method_rows,
        "delta_rows": delta_rows,
        "reference": reference,
        "conclusion": conclusion,
    }


def _load_metric_rows(roots: list[Path]) -> list[dict[str, Any]]:
    args = argparse.Namespace(root=[str(root) for root in roots], metrics=[], manifest="", name_prefix="")
    rows = bc._load_metric_rows(args)
    for row in rows:
        row.setdefault("source_root", _source_root(row.get("metrics_path", ""), roots))
    return _prefer_maskfix_rows(rows)


def _prefer_maskfix_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        run_name = str(row.get("run_name") or "")
        if _is_modular_lite_run(run_name):
            grouped[run_name].append(row)
        else:
            passthrough.append(row)
    for run_name, run_rows in grouped.items():
        maskfix_rows = [row for row in run_rows if _is_maskfix_metrics_path(row.get("metrics_path"))]
        selected = maskfix_rows or run_rows
        if maskfix_rows:
            for row in selected:
                row["maskfix_eval"] = row.get("maskfix_eval", "true")
        passthrough.extend(selected)
    return passthrough


def _source_root(metrics_path: Any, roots: list[Path]) -> str:
    path = Path(str(metrics_path or ""))
    for root in roots:
        try:
            path.relative_to(root)
            return str(root)
        except Exception:
            continue
    return ""


def _is_modular_lite_run(run_name: str) -> bool:
    return _is_modular_lite_method(_method_from_run(run_name))


def _is_modular_lite_method(method: str) -> bool:
    return method.startswith(("amr_lite", "amber_lite"))


def _method_from_run(run_name: str) -> str:
    return re.sub(r"_seed\d+$", "", str(run_name or ""))


def _is_maskfix_metrics_path(path: Any) -> bool:
    return "fresh_eval_maskfix" in Path(str(path or "")).parts


def _mask_suspect_sidecar(metrics_path: Path) -> Path:
    return metrics_path.parent / "mask_suspect.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _attach_mask_suspect(per_run: list[dict[str, Any]], metric_rows: list[dict[str, Any]]) -> None:
    info_by_run: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in metric_rows:
        run_name = str(row.get("run_name") or "")
        info = info_by_run[run_name]
        metrics_path = Path(str(row.get("metrics_path") or ""))
        if _truthy(row.get("maskfix_eval")) or _is_maskfix_metrics_path(metrics_path):
            info["maskfix_eval"] = True
        if _truthy(row.get("mask_suspect")):
            info["mask_suspect"] = True
        sidecar = _mask_suspect_sidecar(metrics_path)
        if sidecar.exists():
            payload = _read_json(sidecar)
            if payload:
                info["maskfix_eval"] = bool(payload.get("maskfix_eval", info.get("maskfix_eval", False)))
                info["mask_suspect"] = bool(payload.get("mask_suspect", info.get("mask_suspect", False)))
                info["mask_suspect_reason"] = payload.get("reason", info.get("mask_suspect_reason", ""))
    for row in per_run:
        run_name = str(row.get("run_name") or "")
        method = str(row.get("method") or "")
        info = info_by_run.get(run_name, {})
        maskfix_eval = bool(info.get("maskfix_eval", False))
        suspect = bool(info.get("mask_suspect", False))
        reason = str(info.get("mask_suspect_reason") or "")
        excluded = False
        if _is_modular_lite_run(run_name) or _is_modular_lite_method(method):
            if not maskfix_eval:
                suspect = True
                excluded = True
                reason = reason or "no_fresh_eval_maskfix"
            elif suspect:
                excluded = True
                reason = reason or "mask_suspect"
        row["mask_suspect"] = str(suspect).lower()
        row["maskfix_eval"] = str(maskfix_eval).lower()
        row["excluded_from_official_ranking"] = str(excluded).lower()
        row["mask_suspect_reason"] = reason
        row["main_read"] = "excluded_from_official_ranking" if excluded else "mask_suspect_excluded" if suspect else _main_read(row)


def _attach_method_read_and_suspect_counts(method_rows: list[dict[str, Any]], per_run: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = defaultdict(int)
    excluded_counts: dict[str, int] = defaultdict(int)
    for row in per_run:
        if _truthy(row.get("mask_suspect")):
            counts[str(row.get("method") or "")] += 1
        if _truthy(row.get("excluded_from_official_ranking")):
            excluded_counts[str(row.get("method") or "")] += 1
    for row in method_rows:
        row["mask_suspect_count"] = counts.get(str(row.get("method") or ""), 0)
        row["excluded_from_official_ranking_count"] = excluded_counts.get(str(row.get("method") or ""), 0)
        row["main_read"] = _main_read(row)


def _subset_reference(method_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    subset = next((row for row in method_rows if row.get("method") == SUBSET_METHOD), None)
    if subset is not None and int(float(subset.get("n", 0) or 0)) >= 3:
        return subset, ""
    ref = dict(FIXED_SUBSET_REFERENCE)
    if subset is not None:
        ref.update({key: value for key, value in subset.items() if key.endswith("_mean") and _isnum(value)})
        ref["n"] = subset.get("n", 0)
    return ref, "proto_randomdrop_subset_es40 actual n<3; using fixed reference avg_missing_top1=0.3055 full_top1≈0.4220"


def _delta_rows(method_rows: list[dict[str, Any]], reference: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in method_rows:
        item = {"method": row.get("method", ""), "n": row.get("n", "")}
        for metric in DELTA_METRICS:
            value = _method_value(row, metric)
            base = _method_value(reference, metric)
            label = metric.replace("within_3", "within@3").replace("mae", "MAE")
            item[metric] = value
            item[f"delta_{label}_vs_subset"] = _delta(value, base)
        out.append(item)
    return out


def _write_outputs(
    out_dir: Path,
    *,
    output_prefix: str,
    per_run: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    reference: dict[str, Any],
    warnings: list[str],
) -> None:
    _write_csv(out_dir / f"{output_prefix}_per_run.csv", per_run, _fields(per_run, _per_run_first_fields()))
    _write_csv(out_dir / f"{output_prefix}_method_mean_std.csv", method_rows, _fields(method_rows, _method_first_fields()))
    _write_csv(out_dir / "delta_vs_randomdrop_subset.csv", delta_rows, _fields(delta_rows, ["method", "n"]))
    _write_rank(out_dir / "rank_by_avg_missing_top1.md", method_rows, "avg_missing_top1", reference, warnings)
    _write_rank(out_dir / "rank_by_miss1_top1.md", method_rows, "miss1_top1", reference, warnings)
    _write_rank(out_dir / "rank_by_miss2_top1.md", method_rows, "miss2_top1", reference, warnings)
    _write_rank(out_dir / "rank_by_miss3_top1.md", method_rows, "miss3_top1", reference, warnings)
    _write_beam_rank(out_dir / "rank_by_beam_proximity.md", method_rows, warnings)
    _write_suspect(out_dir / "suspect_modular_results.md", per_run)


def _write_rank(path: Path, rows: list[dict[str, Any]], metric: str, reference: dict[str, Any], warnings: list[str]) -> None:
    lines = [f"# Rank By {metric}", "", f"Reference: `{SUBSET_METHOD}`", ""]
    columns = ["method", "n", "full_top1", "miss1_top1", "miss2_top1", "miss3_top1", "avg_missing_top1", "overall_mean_top1", "delta_vs_subset"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    ordered = sorted(rows, key=lambda row: (_zero(row.get(f"{metric}_mean")), *_rank_key(row)), reverse=True)
    for row in ordered:
        delta = _delta(_method_value(row, metric), _method_value(reference, metric))
        lines.append(
            "| {method} | {n} | {full} | {miss1} | {miss2} | {miss3} | {avg} | {overall} | {delta} |".format(
                method=row.get("method", ""),
                n=row.get("n", ""),
                full=_mean_std(row, "full_top1"),
                miss1=_mean_std(row, "miss1_top1"),
                miss2=_mean_std(row, "miss2_top1"),
                miss3=_mean_std(row, "miss3_top1"),
                avg=_mean_std(row, "avg_missing_top1"),
                overall=_mean_std(row, "overall_mean_top1"),
                delta=_fmt(delta),
            )
        )
    _append_warnings(lines, warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_beam_rank(path: Path, rows: list[dict[str, Any]], warnings: list[str]) -> None:
    lines = ["# Rank By Beam Proximity", ""]
    columns = ["method", "n", "avg_missing_within@3", "avg_missing_MAE", "avg_missing_top1"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    ordered = sorted(rows, key=lambda row: (-_zero(row.get("avg_missing_within_3_mean")), _zero(row.get("avg_missing_mae_mean"))))
    for row in ordered:
        lines.append(
            "| {method} | {n} | {within} | {mae} | {top1} |".format(
                method=row.get("method", ""),
                n=row.get("n", ""),
                within=_mean_std(row, "avg_missing_within_3"),
                mae=_mean_std(row, "avg_missing_mae"),
                top1=_mean_std(row, "avg_missing_top1"),
            )
        )
    _append_warnings(lines, warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_suspect(path: Path, per_run: list[dict[str, Any]]) -> None:
    rows = [row for row in per_run if _truthy(row.get("mask_suspect")) or str(row.get("method", "")).startswith(("amr_lite", "amber_lite"))]
    lines = ["# Suspect Modular Results", ""]
    columns = ["run_name", "method", "mask_suspect", "full_top1", "avg_missing_top1", "metrics_path"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _conclusion_lines(
    method_rows: list[dict[str, Any]],
    per_run: list[dict[str, Any]],
    reference: dict[str, Any],
    *,
    reference_warning: str,
    include_combined_sections: bool,
) -> list[str]:
    promoted = [row for row in method_rows if _promotion_label(row, reference) in {"promote_to_main_candidate", "candidate_continue_to_seed5"}]
    auxiliary = [row for row in method_rows if _promotion_label(row, reference) == "auxiliary_candidate_only"]
    suspect = [row for row in per_run if _truthy(row.get("mask_suspect"))]
    excluded = [row for row in per_run if _truthy(row.get("excluded_from_official_ranking"))]
    modular = [row for row in per_run if _is_modular_lite_method(str(row.get("method", "")))]
    reliability = next((row for row in method_rows if "reliability_fusion" in str(row.get("method", ""))), None)
    pattern_film = next((row for row in method_rows if "pattern_film_d8" in str(row.get("method", ""))), None)
    reliability_status = _promotion_label(reliability, reference) if reliability else "not_run"
    reliability_next = (
        "1. Reliability fusion seed1-3 is positive; run seed4/5."
        if reliability_status == "candidate_continue_to_seed5"
        else "1. Reliability fusion seed1-3 is not positive; do not run seed4/5 now."
    )
    lines = [
        "Current trusted Scene31 reference:",
        SUBSET_METHOD,
        "",
    ]
    if reference_warning:
        lines.extend(["Warnings:", f"- {reference_warning}", ""])
    lines.extend(
        [
            "AMR/AMBER-lite mask status:",
            f"- fresh_eval_maskfix exists: {len([row for row in modular if _truthy(row.get('maskfix_eval'))])}/{len(modular)}",
            f"- mask_suspect: {len([row for row in modular if _truthy(row.get('mask_suspect'))])}",
            f"- excluded_from_official_ranking: {len(excluded)}",
            "- excluded runs: " + ", ".join(row.get("run_name", "") for row in excluded) if excluded else "- excluded runs: none",
            "",
        ]
    )
    if include_combined_sections:
        lines.extend(_status_block("proto_randomdrop_subset_reliability_fusion_es40", reliability, reference))
        lines.extend(_status_block("proto_randomdrop_subset_pattern_film_d8_es40", pattern_film, reference))
    lines.extend(
        [
            "Methods exceeding proto_randomdrop_subset:",
            *[f"- {row.get('method')}: {_fmt(_delta(_method_value(row, 'avg_missing_top1'), _method_value(reference, 'avg_missing_top1')))}" for row in method_rows if _delta(_method_value(row, "avg_missing_top1"), _method_value(reference, "avg_missing_top1")) > 0],
            "",
            "Methods promoted:",
            *([f"- {row.get('method')}" for row in promoted] or ["- none"]),
            "",
            "Methods not promoted:",
            *[f"- {row.get('method')}: {_promotion_label(row, reference)}" for row in method_rows if row not in promoted],
            "",
            "Next step recommendation:",
            reliability_next,
            "2. If AMR/AMBER maskfix ok, include them as external baselines.",
            "3. Run Scene31-34 quick_seed1 for natural/uniform/subset/subset+reliability.",
            "4. Do not continue PatternFiLM/JTT/MVFR/MPDRO unless new evidence appears.",
        ]
    )
    if auxiliary:
        lines.extend(["", "Auxiliary candidates:", *[f"- {row.get('method')}" for row in auxiliary]])
    return lines


def _status_block(label: str, row: dict[str, Any] | None, reference: dict[str, Any]) -> list[str]:
    status = _promotion_label(row, reference) if row else "not_run"
    lines = [
        f"{label}:",
        f"- n = {row.get('n', 0) if row else 0}",
        f"- status = {status}",
        f"- avg_missing_delta_vs_subset = {_fmt(_delta(_method_value(row, 'avg_missing_top1'), _method_value(reference, 'avg_missing_top1'))) if row else 'unavailable'}",
        f"- overall_delta_vs_subset = {_fmt(_delta(_method_value(row, 'overall_mean_top1'), _method_value(reference, 'overall_mean_top1'))) if row else 'unavailable'}",
    ]
    if "pattern_film" in label:
        lines.append("- reason = avg_missing/full/miss3 lower than randomdrop_subset reference")
    if row and "reliability_fusion" in label and _delta(_method_value(row, "miss3_top1"), _method_value(reference, "miss3_top1")) < 0:
        lines.append("- caveat = miss3 remains below randomdrop_subset reference")
    lines.append("")
    return lines


def _promotion_label(row: dict[str, Any] | None, reference: dict[str, Any]) -> str:
    if row is None:
        return "not_run"
    if int(float(row.get("excluded_from_official_ranking_count", 0) or 0)) > 0:
        return "excluded_from_official_ranking"
    if int(float(row.get("mask_suspect_count", 0) or 0)) > 0:
        return "do_not_promote_mask_suspect"
    if int(float(row.get("n", 0) or 0)) < 3:
        return "auxiliary_candidate_only"
    avg = _method_value(row, "avg_missing_top1")
    ref_avg = _method_value(reference, "avg_missing_top1")
    overall = _method_value(row, "overall_mean_top1")
    ref_overall = _method_value(reference, "overall_mean_top1")
    full = _method_value(row, "full_top1")
    ref_full = _method_value(reference, "full_top1")
    mae = _method_value(row, "avg_missing_mae")
    ref_mae = _method_value(reference, "avg_missing_mae")
    passes_continue = avg > ref_avg and overall >= ref_overall and full >= ref_full - 0.005 and (not _isnum(ref_mae) or mae <= ref_mae)
    if passes_continue and "reliability_fusion" in str(row.get("method", "")):
        return "candidate_continue_to_seed5"
    if passes_continue:
        return "promote_to_main_candidate"
    if "reliability_fusion" in str(row.get("method", "")) and int(float(row.get("n", 0) or 0)) >= 3:
        return "do_not_expand_now"
    if avg > ref_avg or _method_value(row, "miss2_top1") > _method_value(reference, "miss2_top1") or _method_value(row, "miss3_top1") > _method_value(reference, "miss3_top1"):
        return "auxiliary_candidate_only"
    return "do_not_promote"


def _method_value(row: dict[str, Any] | None, metric: str) -> float:
    if row is None:
        return float("nan")
    return _float(row.get(f"{metric}_mean", row.get(metric)))


def _main_read(row: dict[str, Any]) -> str:
    n = int(float(row.get("n", 1) or 1)) if _isnum(row.get("n", 1)) else 1
    method = str(row.get("method", ""))
    if method == SUBSET_METHOD:
        return "current_trusted_proto_reference"
    if method == UNIFORM_METHOD:
        return "ablation"
    return "standard_compare" if n >= 3 else "quick_screen"


def _rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(_zero(row.get(f"{metric}_mean")) for metric in PRIMARY_SORT)


def _mean_std(row: dict[str, Any], metric: str) -> str:
    value = _method_value(row, metric)
    std = _float(row.get(f"{metric}_std"))
    if not _isnum(value):
        return ""
    return f"{_fmt(value)}±{_fmt(std)}" if _isnum(std) else _fmt(value)


def _delta(value: Any, base: Any) -> float:
    left = _float(value)
    right = _float(base)
    return left - right if _isnum(left) and _isnum(right) else float("nan")


def _fields(rows: list[dict[str, Any]], first: list[str]) -> list[str]:
    fields = list(first)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _per_run_first_fields() -> list[str]:
    return [
        "run_name",
        "method",
        "seed",
        "maskfix_eval",
        "mask_suspect",
        "excluded_from_official_ranking",
        "mask_suspect_reason",
        "main_read",
        "full_top1",
        "miss1_top1",
        "miss2_top1",
        "miss3_top1",
        "avg_missing_top1",
        "overall_mean_top1",
        "avg_missing_within@3",
        "avg_missing_mae",
        "balanced",
        "metrics_path",
    ]


def _method_first_fields() -> list[str]:
    return [
        "method",
        "n",
        "mask_suspect_count",
        "excluded_from_official_ranking_count",
        "main_read",
        "full_top1_mean",
        "miss1_top1_mean",
        "miss2_top1_mean",
        "miss3_top1_mean",
        "avg_missing_top1_mean",
        "overall_mean_top1_mean",
        "avg_missing_within_3_mean",
        "avg_missing_mae_mean",
        "balanced_mean",
    ]


def _print_mask_status(per_run: list[dict[str, Any]]) -> None:
    modular = [row for row in per_run if _is_modular_lite_method(str(row.get("method", "")))]
    print("AMR/AMBER-lite mask status:")
    print(f"- fresh_eval_maskfix exists: {len([row for row in modular if _truthy(row.get('maskfix_eval'))])}/{len(modular)}")
    print(f"- mask_suspect: {len([row for row in modular if _truthy(row.get('mask_suspect'))])}")
    print(f"- excluded_from_official_ranking: {len([row for row in modular if _truthy(row.get('excluded_from_official_ranking'))])}")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(value) if isinstance(value, float) else value for key, value in row.items()})


def _append_warnings(lines: list[str], warnings: list[str]) -> None:
    warnings = [warning for warning in warnings if warning]
    if warnings:
        lines.extend(["", "## Warnings", *[f"- {warning}" for warning in warnings]])


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _isnum(value: Any) -> bool:
    return math.isfinite(_float(value))


def _zero(value: Any) -> float:
    value = _float(value)
    return value if math.isfinite(value) else -math.inf


def _fmt(value: Any) -> str:
    value = _float(value)
    return f"{value:.5f}" if math.isfinite(value) else ""


if __name__ == "__main__":
    raise SystemExit(main())
