#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path
from typing import Any


REFERENCE_METHOD = "scenes31_34_proto_randomdrop_subset_es40"
BERNOULLI_METHOD = "scenes31_34_proto_randomdrop_bernoulli_k075_es40"
CLASSIFIER_SUBSET = "scenes31_34_classifier_randomdrop_subset_es40"
CLASSIFIER_NATURAL = "scenes31_34_classifier_natural_es40"
PROTO_NATURAL = "scenes31_34_proto_natural_es40"
METHOD_LABELS = {
    "scenes31_34_proto_natural_es40": "Proto natural",
    "scenes31_34_proto_sampler_uniform_es40": "Proto uniform pattern exposure",
    "scenes31_34_proto_randomdrop_bernoulli_k075_es40": "Proto Bernoulli randomdrop",
    "scenes31_34_proto_randomdrop_subset_es40": "Proto random subset exposure",
    "scenes31_34_classifier_natural_es40": "Classifier natural",
    "scenes31_34_classifier_randomdrop_subset_es40": "Classifier random subset",
    "scenes31_34_amr_lite_natural_es40": "AMR-lite natural",
    "scenes31_34_amber_lite_natural_es40": "AMBER-lite natural",
    "scenes31_34_amr_lite_uniform_es40": "AMR-lite uniform",
    "scenes31_34_amber_lite_uniform_es40": "AMBER-lite uniform",
}
CORE_METHODS = tuple(method for method in METHOD_LABELS if "proto_" in method)
EXTERNAL_METHODS = tuple(method for method in METHOD_LABELS if "amr_lite" in method or "amber_lite" in method)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary_root = Path(args.summary_root)
    paper_root = Path(args.paper_table_root)
    figure_root = Path(args.figure_root)
    profile_root = Path(args.profile_root)
    statistics_root = Path(args.statistics_root)
    pattern_root = Path(args.pattern_root)
    cdf_root = Path(args.cdf_root)
    sampling_root = Path(args.sampling_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(summary_root / "final_method_mean_std.csv") or _read_csv(summary_root / "method_mean_std.csv")
    checklist = _read_csv(summary_root / "final_evidence_checklist.csv")
    cost_rows = _read_csv(profile_root / "method_profile_summary.csv")
    by_method = {row.get("method"): row for row in rows}
    winner = _winner(rows)
    subset = by_method.get(REFERENCE_METHOD, {})
    bernoulli = by_method.get(BERNOULLI_METHOD, {})
    classifier_subset = by_method.get(CLASSIFIER_SUBSET, {})
    classifier_natural = by_method.get(CLASSIFIER_NATURAL, {})
    proto_natural = by_method.get(PROTO_NATURAL, {})
    external = [row for row in rows if row.get("method") in EXTERNAL_METHODS]
    best_external = _winner(external)
    external_gap = _delta(subset.get("avg_missing_top1_mean"), by_method.get(best_external, {}).get("avg_missing_top1_mean"))
    subset_final = winner == REFERENCE_METHOD and _int(subset.get("n")) >= 5
    significance = _read_csv(statistics_root / "significance_summary.csv")
    wins = _read_csv(pattern_root / "pattern_win_count_summary.csv")
    cdf_rows = _read_csv(cdf_root / "abs_error_cdf_data.csv")
    sampling_summary = sampling_root / "sampling_distribution_summary.md"

    lines = [
        "Scene31-34 final main conclusion",
        "",
        "1. Main winner",
        f"- Final trusted method: {'prototype + random non-empty subset exposure' if subset_final else METHOD_LABELS.get(winner, winner or 'unavailable')}.",
        f"- Current official Avg-Missing winner: {METHOD_LABELS.get(winner, winner or 'unavailable')}.",
        f"- Proto random subset n={_int(subset.get('n'))}, Avg-Missing={_pct(subset.get('avg_missing_top1_mean'))}, Miss3={_pct(subset.get('miss3_top1_mean'))}, MAE={_raw(subset.get('avg_missing_MAE_mean'))}.",
        "- Interpretation is conservative: proto natural and classifier natural are close, so the claim is the prototype + subset combination, not prototype alone.",
        "",
        "2. Statistical evidence",
        *_significance_lines(significance),
        "",
        "3. Pattern-level evidence",
        *_pattern_lines(wins),
        "",
        "4. Error CDF evidence",
        *_cdf_lines(cdf_rows),
        "",
        "5. Sampling distribution explanation",
        *_sampling_lines(sampling_summary),
        "",
        "6. Compute cost evidence",
        *_cost_lines(cost_rows),
        "",
        "7. External baseline result",
        *_external_lines(external, best_external, external_gap),
        "",
        "Supporting classifier/prototype comparisons",
        f"- Proto subset vs classifier subset: {_comparison(subset, classifier_subset, higher_metric='avg_missing_top1_mean')}",
        f"- Proto natural vs classifier natural: {_comparison(proto_natural, classifier_natural, higher_metric='avg_missing_top1_mean')}",
        f"- Classifier subset vs classifier natural: {_comparison(classifier_subset, classifier_natural, higher_metric='avg_missing_top1_mean')}",
        "",
        "Exposure comparison snapshot",
        f"- Random subset exposure vs Bernoulli Avg-Missing: {_signed_pp(_delta(subset.get('avg_missing_top1_mean'), bernoulli.get('avg_missing_top1_mean')))}.",
        f"- Random subset exposure vs Bernoulli Miss3: {_signed_pp(_delta(subset.get('miss3_top1_mean'), bernoulli.get('miss3_top1_mean')))}.",
        f"- Random subset exposure vs Bernoulli MAE improvement: {_signed_raw(_delta(bernoulli.get('avg_missing_MAE_mean'), subset.get('avg_missing_MAE_mean')))}.",
        f"- Random subset exposure vs Bernoulli Top1 drop reduction: {_signed_pp(_delta(bernoulli.get('top1_drop_0_to_75_mean'), subset.get('top1_drop_0_to_75_mean')))}.",
        "",
        "8. Evidence checklist caveat",
        *_evidence_lines(checklist, rows),
        "",
        "9. Whether any further experiments are needed",
        f"- Random subset remains final main method: {'yes' if subset_final else 'pending_or_no'}.",
        f"- Need AMR/AMBER seed2/3: {_need_external_seeds(external_gap, best_external)}.",
        "- Extra experiments still needed: no new module search is needed for the paper claim; only missing analysis artifacts should be regenerated if their files are absent.",
        "- Do not continue reliability fusion, PatternFiLM, JTT, MVFR, MPDRO, beamsoft, condBTAPA, weakKD, or AMR/AMBER seed2/3 for this final main claim.",
        "",
        f"Summary root: {summary_root}",
        f"Paper table root: {paper_root}",
        f"Figure root: {figure_root}",
        f"Profile root: {profile_root}",
        f"Statistics root: {statistics_root}",
        f"Pattern root: {pattern_root}",
        f"Error CDF root: {cdf_root}",
        f"Sampling root: {sampling_root}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote final Scene31-34 main conclusion to {out}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write Scene31-34 main final conclusion.")
    parser.add_argument("--summary-root", default="outputs/scenes31_34_main_lmdb/summary")
    parser.add_argument("--paper-table-root", default="outputs/paper_tables/scenes31_34_main")
    parser.add_argument("--figure-root", default="outputs/scenes31_34_main_lmdb/figures")
    parser.add_argument("--profile-root", default="outputs/scenes31_34_main_lmdb/profile")
    parser.add_argument("--statistics-root", default="outputs/scenes31_34_main_lmdb/statistics")
    parser.add_argument("--pattern-root", default="outputs/scenes31_34_main_lmdb/pattern_analysis")
    parser.add_argument("--cdf-root", default="outputs/scenes31_34_main_lmdb/error_cdf")
    parser.add_argument("--sampling-root", default="outputs/scenes31_34_main_lmdb/sampling_analysis")
    parser.add_argument("--out", default="outputs/scenes31_34_main_lmdb/summary/final_main_conclusion.txt")
    return parser


def _comparison(left: dict[str, str], right: dict[str, str], *, higher_metric: str) -> str:
    if _int(left.get("n")) <= 0 or _int(right.get("n")) <= 0:
        return "not available until both baselines have completed runs"
    delta = _delta(left.get(higher_metric), right.get(higher_metric))
    label = "better" if delta > 0 else ("worse" if delta < 0 else "tied")
    return f"{label} by {_signed_pp(delta)} Avg-Missing Top1"


def _external_lines(rows: list[dict[str, str]], best_external: str, gap: float) -> list[str]:
    if not rows or all(_int(row.get("n")) <= 0 for row in rows):
        return ["- AMR/AMBER-lite multi-scene baselines were prepared but not run; they remain optional external baselines."]
    status = [
        f"{METHOD_LABELS.get(row.get('method', ''), row.get('method', ''))}: n={_int(row.get('n'))}, mask_suspect_count={_int(row.get('mask_suspect_count'))}, official={row.get('official_ranking_included')}"
        for row in rows
    ]
    return [
        "- AMR/AMBER-lite maskfix status:",
        *[f"  - {item}" for item in status],
        f"- Best external baseline: {METHOD_LABELS.get(best_external, best_external or 'unavailable')}.",
        f"- Gap to proto random subset: {_signed_pp(gap)} Avg-Missing Top1.",
    ]


def _cost_lines(rows: list[dict[str, str]]) -> list[str]:
    subset = next((row for row in rows if row.get("method") == REFERENCE_METHOD), {})
    line = subset.get("extra_inference_cost") or "none at inference; training-only exposure strategy"
    return [
        "- Random subset exposure introduces no extra inference-time parameters or latency relative to the same proto model; it is a training exposure strategy.",
        f"- Profile status: {line}.",
    ]


def _evidence_lines(checklist: list[dict[str, str]], rows: list[dict[str, str]]) -> list[str]:
    pending = [
        f"{row.get('item')}: {row.get('status')} ({row.get('caveat') or row.get('next_action')})"
        for row in checklist
        if str(row.get("status", "")).lower() not in {"", "complete"}
    ]
    if pending:
        return ["- Final evidence is not yet complete:", *[f"  - {item}" for item in pending]]
    method_caveats = [
        f"{METHOD_LABELS.get(str(row.get('method') or ''), str(row.get('method') or 'method'))}: {row.get('claim_status')} ({row.get('caveat')})"
        for row in rows
        if str(row.get("claim_status", "")).strip() not in {"", "complete"}
    ]
    if method_caveats:
        return ["- Method-level caveats remain:", *[f"  - {item}" for item in method_caveats]]
    return ["- Final evidence checklist is complete in the available summary artifacts."]


def _significance_lines(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["- Significance analysis artifact not found; rerun python -m kd_sensing.diagnostics.scene31_34_final_analysis --artifact significance for final paper numbers."]
    wanted = {
        ("Proto random subset vs Proto Bernoulli randomdrop", "avg_missing_top1"),
        ("Proto random subset vs Proto Bernoulli randomdrop", "miss3_top1"),
        ("Proto random subset vs Proto Bernoulli randomdrop", "avg_missing_MAE"),
        ("Proto random subset vs Classifier random subset", "avg_missing_top1"),
        ("Proto random subset vs AMBER-lite best external", "avg_missing_top1"),
    }
    out = []
    for row in rows:
        key = (row.get("comparison", ""), row.get("metric", ""))
        if key in wanted:
            out.append(
                f"- {row.get('comparison')} / {row.get('metric')}: seed delta={_sig_delta(row, 'seed')}, "
                f"bootstrap mean={_sig_delta(row, 'bootstrap')}, "
                f"bootstrap CI={_sig_ci(row)}, "
                f"P(delta>0)={_raw(row.get('prob_delta_positive'))}, conclusion={row.get('conclusion')}."
            )
    return out or ["- Significance CSV was present but did not include the expected final comparisons."]


def _pattern_lines(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["- Pattern heatmap/win-count artifact not found; rerun python -m kd_sensing.diagnostics.scene31_34_final_analysis --artifact heatmap."]
    subset_rows = [row for row in rows if row.get("method") == "Proto random subset exposure"]
    if not subset_rows:
        return ["- Pattern win-count table was present, but proto random subset row was unavailable."]
    return [
        f"- {row.get('metric')}: best on {row.get('num_patterns_best')}/{row.get('num_patterns_total')} patterns, "
        f"beats Bernoulli on {row.get('num_patterns_beats_bernoulli')} and classifier subset on {row.get('num_patterns_beats_classifier_subset')} patterns."
        for row in subset_rows
    ]


def _cdf_lines(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["- Error CDF artifact not found; rerun python -m kd_sensing.diagnostics.scene31_34_final_analysis --artifact error-cdf."]
    subset = [row for row in rows if row.get("method") == REFERENCE_METHOD and row.get("abs_error_threshold") == "3"]
    if not subset:
        return ["- Error CDF data was present, but Within@3 threshold rows were unavailable."]
    by_condition = {row.get("condition"): row for row in subset}
    return [
        f"- {condition}: proto random subset CDF at |error|<=3 is {_pct(row.get('cdf'))} over n={row.get('num_samples')} prediction rows."
        for condition, row in sorted(by_condition.items())
    ]


def _sampling_lines(path: Path) -> list[str]:
    if not path.exists():
        return ["- Sampling distribution artifact not found; rerun python -m kd_sensing.diagnostics.scene31_34_final_analysis --artifact sampling."]
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("- ")]
    return lines[:6] if lines else ["- Sampling summary file was present but empty."]


def _need_external_seeds(gap: float, best_external: str) -> str:
    if not best_external:
        return "no, first run seed1"
    if not math.isfinite(gap):
        return "unknown"
    return "yes" if gap <= 0.03 else "no"


def _winner(rows: list[dict[str, str]]) -> str:
    valid = [
        row for row in rows
        if _truthy(row.get("official_ranking_included")) and math.isfinite(_float(row.get("avg_missing_top1_mean")))
    ]
    if not valid:
        return ""
    return max(valid, key=lambda row: _float(row.get("avg_missing_top1_mean"))).get("method", "")


def _delta(left: Any, right: Any) -> float:
    a = _float(left)
    b = _float(right)
    return a - b if math.isfinite(a) and math.isfinite(b) else math.nan


def _pct(value: Any) -> str:
    number = _float(value)
    if not math.isfinite(number):
        return "unavailable"
    number = number * 100.0 if abs(number) <= 1.5 else number
    return f"{number:.2f}%"


def _signed_pp(value: float) -> str:
    if not math.isfinite(value):
        return "unavailable"
    value = value * 100.0 if abs(value) <= 1.5 else value
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f} pp"


def _raw(value: Any) -> str:
    number = _float(value)
    return f"{number:.3f}" if math.isfinite(number) else "unavailable"


def _signed_raw(value: float) -> str:
    if not math.isfinite(value):
        return "unavailable"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.3f}"


def _sig_delta(row: dict[str, str], kind: str) -> str:
    metric = row.get("metric", "")
    if metric in {"avg_missing_MAE", "mae_at_75"}:
        key = "seed_mean_delta" if kind == "seed" else "bootstrap_mean_delta"
        return _signed_raw(_float(row.get(key)))
    key = "seed_mean_delta_pp" if kind == "seed" else "bootstrap_mean_delta_pp"
    return _signed_pp(_float(row.get(key)))


def _sig_ci(row: dict[str, str]) -> str:
    metric = row.get("metric", "")
    if metric in {"avg_missing_MAE", "mae_at_75"}:
        return f"[{_raw(row.get('bootstrap_ci_low'))}, {_raw(row.get('bootstrap_ci_high'))}]"
    return f"[{_signed_pp(_float(row.get('bootstrap_ci_low_pp')))}, {_signed_pp(_float(row.get('bootstrap_ci_high_pp')))}]"


def _int(value: Any) -> int:
    number = _float(value)
    return int(number) if math.isfinite(number) else 0


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
