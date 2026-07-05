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
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(summary_root / "final_method_mean_std.csv") or _read_csv(summary_root / "method_mean_std.csv")
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

    lines = [
        "Scene31-34 final main conclusion",
        "",
        "Main method",
        f"- Final trusted method: {'prototype + random non-empty subset exposure' if subset_final else METHOD_LABELS.get(winner, winner or 'unavailable')}.",
        f"- Current official Avg-Missing winner: {METHOD_LABELS.get(winner, winner or 'unavailable')}.",
        f"- Proto random subset n={_int(subset.get('n'))}, Avg-Missing={_pct(subset.get('avg_missing_top1_mean'))}, Miss3={_pct(subset.get('miss3_top1_mean'))}, MAE={_raw(subset.get('avg_missing_MAE_mean'))}.",
        "",
        "Prototype baseline conclusion",
        f"- Proto subset vs classifier subset: {_comparison(subset, classifier_subset, higher_metric='avg_missing_top1_mean')}",
        f"- Proto natural vs classifier natural: {_comparison(proto_natural, classifier_natural, higher_metric='avg_missing_top1_mean')}",
        f"- Classifier subset vs classifier natural: {_comparison(classifier_subset, classifier_natural, higher_metric='avg_missing_top1_mean')}",
        "",
        "Exposure conclusion",
        f"- Random subset exposure vs Bernoulli Avg-Missing: {_signed_pp(_delta(subset.get('avg_missing_top1_mean'), bernoulli.get('avg_missing_top1_mean')))}.",
        f"- Random subset exposure vs Bernoulli Miss3: {_signed_pp(_delta(subset.get('miss3_top1_mean'), bernoulli.get('miss3_top1_mean')))}.",
        f"- Random subset exposure vs Bernoulli MAE: {_signed_raw(_delta(subset.get('avg_missing_MAE_mean'), bernoulli.get('avg_missing_MAE_mean')))}.",
        f"- Random subset exposure vs Bernoulli Top1 drop 0%->75%: {_signed_pp(_delta(subset.get('top1_drop_0_to_75_mean'), bernoulli.get('top1_drop_0_to_75_mean')))}.",
        "",
        "External baseline conclusion",
        *_external_lines(external, best_external, external_gap),
        "",
        "Compute cost conclusion",
        *_cost_lines(cost_rows),
        "",
        "Decision",
        f"- Random subset remains final main method: {'yes' if subset_final else 'pending_or_no'}.",
        f"- Need AMR/AMBER seed2/3: {_need_external_seeds(external_gap, best_external)}.",
        "- Extra experiments still needed: classifier/external rows with n=0 or not run should be launched before paper freeze; core proto does not need retraining.",
        "",
        f"Summary root: {summary_root}",
        f"Paper table root: {paper_root}",
        f"Figure root: {figure_root}",
        f"Profile root: {profile_root}",
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
