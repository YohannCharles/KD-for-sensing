#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

try:
    from scripts.scene31_34_final_analysis_common import (
        BERNOULLI_METHOD,
        CLASSIFIER_SUBSET_METHOD,
        PROTO_NATURAL_METHOD,
        PROTO_UNIFORM_METHOD,
        REFERENCE_METHOD,
        best_external_method,
        float_or_nan,
        load_pattern_metrics,
        load_predictions,
        method_label,
        method_rows_from_summary,
        roots_from_args,
        seed_from_run,
        write_csv,
        write_md_table,
    )
except ModuleNotFoundError:
    from scene31_34_final_analysis_common import (
        BERNOULLI_METHOD,
        CLASSIFIER_SUBSET_METHOD,
        PROTO_NATURAL_METHOD,
        PROTO_UNIFORM_METHOD,
        REFERENCE_METHOD,
        best_external_method,
        float_or_nan,
        load_pattern_metrics,
        load_predictions,
        method_label,
        method_rows_from_summary,
        roots_from_args,
        seed_from_run,
        write_csv,
        write_md_table,
    )


METRICS = (
    "full_top1",
    "avg_missing_top1",
    "miss1_top1",
    "miss2_top1",
    "miss3_top1",
    "avg_missing_within@3",
    "avg_missing_MAE",
    "top1_drop_0_to_75",
    "mae_at_75",
)
LOWER_BETTER = {"avg_missing_MAE", "top1_drop_0_to_75", "mae_at_75"}
SUMMARY_FIELDS = [
    "comparison",
    "metric",
    "method",
    "baseline",
    "method_n",
    "baseline_n",
    "mean_method_fraction",
    "mean_baseline_fraction",
    "seed_mean_delta_fraction",
    "seed_mean_delta_pp",
    "bootstrap_mean_delta_fraction",
    "bootstrap_mean_delta_pp",
    "bootstrap_ci_low_fraction",
    "bootstrap_ci_high_fraction",
    "bootstrap_ci_low_pp",
    "bootstrap_ci_high_pp",
    "mean_method",
    "mean_baseline",
    "seed_mean_delta",
    "bootstrap_mean_delta",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "metric_unit",
    "seed_paired_t_p",
    "seed_wilcoxon_p",
    "prob_delta_positive",
    "num_scenes_positive",
    "conclusion",
    "notes",
]
BOOT_FIELDS = ["comparison", "metric", "iteration", "bootstrap_delta", "bootstrap_delta_pp"]
SCENE_FIELDS = ["comparison", "metric", "scene", "method_value", "baseline_value", "delta"]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    roots = roots_from_args([args.root], args.old_root, args.classifier_root, args.external_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paper_root = Path(args.paper_table_root)
    paper_root.mkdir(parents=True, exist_ok=True)

    summary_root = Path(args.root) / "summary"
    method_rows = method_rows_from_summary(summary_root)
    external = best_external_method(method_rows, "amber_lite")
    comparisons = [
        ("Proto random subset vs Proto Bernoulli randomdrop", REFERENCE_METHOD, BERNOULLI_METHOD),
        ("Proto random subset vs Classifier random subset", REFERENCE_METHOD, CLASSIFIER_SUBSET_METHOD),
        ("Proto random subset vs Proto natural", REFERENCE_METHOD, PROTO_NATURAL_METHOD),
        ("Proto random subset vs Proto uniform", REFERENCE_METHOD, PROTO_UNIFORM_METHOD),
    ]
    if external:
        comparisons.append(("Proto random subset vs AMBER-lite best external", REFERENCE_METHOD, external))

    needed = {item for _, method, baseline in comparisons for item in (method, baseline)}
    pattern_df = load_pattern_metrics(roots, needed)
    pred_df = load_predictions(roots, needed)
    warnings: list[str] = []
    if pred_df.empty:
        warnings.append("WARNING: per-sample predictions unavailable; bootstrap and scene statistics are weak or skipped.")
    if pattern_df.empty:
        warnings.append("WARNING: pattern metrics unavailable; falling back to summary/method_mean_std.csv only.")

    run_metrics = _run_metrics(pattern_df, method_rows)
    rng = np.random.default_rng(int(args.seed))
    summary: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []

    for comparison, method, baseline in comparisons:
        for metric in METRICS:
            seed_stats = _seed_stats(run_metrics, method, baseline, metric)
            boot = _bootstrap(pred_df, method, baseline, metric, int(args.bootstrap), rng)
            for idx, delta in enumerate(boot.get("samples", [])):
                boot_rows.append(
                    {
                        "comparison": comparison,
                        "metric": metric,
                        "iteration": idx,
                        "bootstrap_delta": delta,
                        "bootstrap_delta_pp": delta * 100.0 if _is_pp_metric(metric) and math.isfinite(float_or_nan(delta)) else math.nan,
                    }
                )
            scene = _scene_deltas(pred_df, method, baseline, metric)
            scene_rows.extend(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "scene": row["scene"],
                    "method_value": row["method_value"],
                    "baseline_value": row["baseline_value"],
                    "delta": row["delta"],
                }
                for row in scene
            )
            notes = []
            if min(seed_stats["method_n"], seed_stats["baseline_n"]) < 3:
                notes.append("seed_paired_test_unavailable_common_seed_count_lt_3")
                warnings.append(f"WARNING: skipped seed-level paired test for {comparison}/{metric}; common seed count < 3.")
            notes.extend(_sanity_notes(metric, boot, _conclusion(boot.get("mean_delta", math.nan), boot.get("ci_low", math.nan), boot.get("prob_positive", math.nan))))
            conclusion = _conclusion(boot.get("mean_delta", math.nan), boot.get("ci_low", math.nan), boot.get("prob_positive", math.nan))
            if not math.isfinite(boot.get("mean_delta", math.nan)):
                conclusion = _conclusion(seed_stats["seed_mean_delta"], math.nan, math.nan)
            summary.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "method": method,
                    "baseline": baseline,
                    "method_n": seed_stats["method_n"],
                    "baseline_n": seed_stats["baseline_n"],
                    **_unit_fields(metric, seed_stats, boot),
                    "seed_paired_t_p": seed_stats["paired_t_p"],
                    "seed_wilcoxon_p": seed_stats["wilcoxon_p"],
                    "prob_delta_positive": boot.get("prob_positive", math.nan),
                    "num_scenes_positive": sum(1 for row in scene if row["delta"] > 0),
                    "conclusion": conclusion,
                    "notes": "; ".join(notes),
                }
            )
            for note in notes:
                if note.startswith("WARNING:"):
                    warnings.append(note)

    write_csv(out / "significance_summary.csv", summary, SUMMARY_FIELDS)
    write_csv(out / "bootstrap_deltas.csv", boot_rows, BOOT_FIELDS)
    write_csv(out / "per_scene_deltas.csv", scene_rows, SCENE_FIELDS)
    _write_summary_md(out / "significance_summary.md", summary, warnings)
    _write_summary_md(paper_root / "table_significance_tests.md", summary, warnings)
    if warnings:
        (out / "warnings.txt").write_text("\n".join(sorted(set(warnings))) + "\n", encoding="utf-8")
    print(f"Wrote Scene31-34 significance tests to {out}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Scene31-34 final significance and bootstrap analyses.")
    parser.add_argument("--root", default="outputs/scenes31_34_main_lmdb")
    parser.add_argument("--classifier-root", action="append", default=[])
    parser.add_argument("--external-root", action="append", default=[])
    parser.add_argument("--old-root", action="append", default=[])
    parser.add_argument("--out", default="outputs/scenes31_34_main_lmdb/statistics")
    parser.add_argument("--paper-table-root", default="outputs/paper_tables/scenes31_34_main")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260705)
    return parser


def _run_metrics(pattern_df: pd.DataFrame, summary_rows: list[dict[str, str]]) -> pd.DataFrame:
    if pattern_df.empty:
        rows = []
        for row in summary_rows:
            item = {"method": row.get("method", ""), "seed": 0}
            for metric in METRICS:
                if metric == "top1_drop_0_to_75":
                    item[metric] = float_or_nan(row.get("top1_drop_0_to_75_mean"))
                elif metric == "mae_at_75":
                    item[metric] = float_or_nan(row.get("mae_at_75_mean"))
                else:
                    item[metric] = float_or_nan(row.get(f"{metric}_mean"))
            rows.append(item)
        return pd.DataFrame(rows)

    rows = []
    for (run_name, method, seed), group in pattern_df.groupby(["run_name", "method", "seed"], dropna=False):
        seed = int(seed) if math.isfinite(float_or_nan(seed)) else seed_from_run(str(run_name))
        item = {"run_name": run_name, "method": method, "seed": seed}
        for metric in METRICS:
            item[metric] = _metric_from_pattern_group(group, metric)
        rows.append(item)
    return pd.DataFrame(rows)


def _metric_from_pattern_group(group: pd.DataFrame, metric: str) -> float:
    first = {
        "full_top1": "full_top1",
        "avg_missing_top1": "avg_missing_top1",
        "miss1_top1": "miss1_top1",
        "miss2_top1": "miss2_top1",
        "miss3_top1": "miss3_top1",
        "avg_missing_within@3": "avg_missing_within@3",
        "avg_missing_MAE": "avg_missing_MAE",
    }.get(metric)
    if first and first in group.columns:
        value = pd.to_numeric(group[first], errors="coerce").dropna()
        if not value.empty:
            return float(value.iloc[0])
    if metric == "top1_drop_0_to_75":
        return _metric_from_pattern_group(group, "full_top1") - _metric_from_pattern_group(group, "miss3_top1")
    if metric == "mae_at_75":
        part = group[pd.to_numeric(group.get("missing_count"), errors="coerce") == 3]
        return _weighted_mean(part, "mae")
    if metric.startswith("miss") and metric.endswith("_top1"):
        count = int(metric[4])
        return _weighted_mean(group[pd.to_numeric(group.get("missing_count"), errors="coerce") == count], "top1")
    if metric == "full_top1":
        return _weighted_mean(group[pd.to_numeric(group.get("missing_count"), errors="coerce") == 0], "top1")
    if metric == "avg_missing_top1":
        return _weighted_mean(group[pd.to_numeric(group.get("missing_count"), errors="coerce") > 0], "top1")
    if metric == "avg_missing_within@3":
        col = "within3" if "within3" in group.columns else ("within_3" if "within_3" in group.columns else "within@3")
        return _weighted_mean(group[pd.to_numeric(group.get("missing_count"), errors="coerce") > 0], col)
    if metric == "avg_missing_MAE":
        return _weighted_mean(group[pd.to_numeric(group.get("missing_count"), errors="coerce") > 0], "mae")
    return math.nan


def _weighted_mean(group: pd.DataFrame, column: str) -> float:
    if group.empty or column not in group.columns:
        return math.nan
    values = pd.to_numeric(group[column], errors="coerce")
    weights = pd.to_numeric(group["num_samples"], errors="coerce") if "num_samples" in group.columns else pd.Series(1.0, index=group.index)
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return math.nan
    return float(np.average(values[mask], weights=weights[mask]))


def _seed_stats(run_metrics: pd.DataFrame, method: str, baseline: str, metric: str) -> dict[str, float]:
    empty = {
        "method_n": 0,
        "baseline_n": 0,
        "mean_method": math.nan,
        "mean_baseline": math.nan,
        "seed_mean_delta": math.nan,
        "std_delta": math.nan,
        "ci_low": math.nan,
        "ci_high": math.nan,
        "paired_t_p": math.nan,
        "wilcoxon_p": math.nan,
    }
    if run_metrics.empty or metric not in run_metrics.columns:
        return empty
    left = run_metrics[run_metrics["method"].astype(str) == method][["seed", metric]].dropna()
    right = run_metrics[run_metrics["method"].astype(str) == baseline][["seed", metric]].dropna()
    out = dict(empty)
    out["method_n"] = int(len(left))
    out["baseline_n"] = int(len(right))
    out["mean_method"] = float(left[metric].mean()) if not left.empty else math.nan
    out["mean_baseline"] = float(right[metric].mean()) if not right.empty else math.nan
    out["seed_mean_delta"] = _signed_delta(out["mean_method"], out["mean_baseline"], metric)
    paired = left.merge(right, on="seed", suffixes=("_method", "_baseline"))
    if len(paired) < 3:
        return out
    diffs = np.array([_signed_delta(a, b, metric) for a, b in zip(paired[f"{metric}_method"], paired[f"{metric}_baseline"])], dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 3:
        return out
    out["std_delta"] = float(diffs.std(ddof=1))
    sem = stats.sem(diffs)
    ci = stats.t.interval(0.95, len(diffs) - 1, loc=diffs.mean(), scale=sem) if math.isfinite(sem) and sem > 0 else (diffs.mean(), diffs.mean())
    out["ci_low"] = float(ci[0])
    out["ci_high"] = float(ci[1])
    out["paired_t_p"] = float(stats.ttest_rel(paired[f"{metric}_method"], paired[f"{metric}_baseline"], nan_policy="omit").pvalue)
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(diffs).pvalue)
    except ValueError:
        out["wilcoxon_p"] = math.nan
    return out


def _bootstrap(pred_df: pd.DataFrame, method: str, baseline: str, metric: str, n: int, rng: np.random.Generator) -> dict[str, Any]:
    diffs = _prediction_diffs(pred_df, method, baseline, metric)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return {"samples": [], "mean_delta": math.nan, "ci_low": math.nan, "ci_high": math.nan, "prob_positive": math.nan}
    samples = np.empty(int(n), dtype=float)
    for idx in range(int(n)):
        samples[idx] = float(rng.choice(diffs, size=diffs.size, replace=True).mean())
    return {
        "samples": samples.tolist(),
        "mean_delta": float(diffs.mean()),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "prob_positive": float((samples > 0).mean()),
    }


def _prediction_diffs(pred_df: pd.DataFrame, method: str, baseline: str, metric: str) -> np.ndarray:
    if pred_df.empty:
        return np.array([], dtype=float)
    data = pred_df[pred_df["method"].astype(str).isin({method, baseline})].copy()
    if data.empty:
        return np.array([], dtype=float)
    if metric == "top1_drop_0_to_75":
        return _drop_diffs(data, method, baseline)
    if metric == "mae_at_75":
        condition = data["missing_count"] == 3
        column = "abs_error"
    elif metric == "avg_missing_MAE":
        condition = data["missing_count"] > 0
        column = "abs_error"
    elif metric == "avg_missing_within@3":
        condition = data["missing_count"] > 0
        column = "within3_correct"
    else:
        column = "top1_correct"
        if metric == "full_top1":
            condition = data["missing_count"] == 0
        elif metric.startswith("miss") and metric.endswith("_top1"):
            condition = data["missing_count"] == int(metric[4])
        else:
            condition = data["missing_count"] > 0
    keys = ["scene", "sample_id", "pattern"]
    values = data.loc[condition, [*keys, "method", column]].dropna()
    if values.empty:
        return np.array([], dtype=float)
    pivot = values.groupby([*keys, "method"], dropna=False)[column].mean().unstack("method")
    if method not in pivot or baseline not in pivot:
        return np.array([], dtype=float)
    if metric in LOWER_BETTER:
        diffs = pivot[baseline] - pivot[method]
    else:
        diffs = pivot[method] - pivot[baseline]
    return diffs.to_numpy(dtype=float)


def _drop_diffs(data: pd.DataFrame, method: str, baseline: str) -> np.ndarray:
    keys = ["scene", "sample_id"]
    full = data[data["missing_count"] == 0].groupby([*keys, "method"], dropna=False)["top1_correct"].mean().unstack("method")
    miss3 = data[data["missing_count"] == 3].groupby([*keys, "method"], dropna=False)["top1_correct"].mean().unstack("method")
    if method not in full or baseline not in full or method not in miss3 or baseline not in miss3:
        return np.array([], dtype=float)
    method_drop = full[method] - miss3[method]
    baseline_drop = full[baseline] - miss3[baseline]
    return (baseline_drop - method_drop).dropna().to_numpy(dtype=float)


def _scene_deltas(pred_df: pd.DataFrame, method: str, baseline: str, metric: str) -> list[dict[str, Any]]:
    if pred_df.empty:
        return []
    rows = []
    for scene, group in pred_df.groupby("scene"):
        method_value = _metric_from_predictions(group[group["method"].astype(str) == method], metric)
        baseline_value = _metric_from_predictions(group[group["method"].astype(str) == baseline], metric)
        delta = _signed_delta(method_value, baseline_value, metric)
        if math.isfinite(delta):
            rows.append({"scene": scene, "method_value": method_value, "baseline_value": baseline_value, "delta": delta})
    return rows


def _metric_from_predictions(group: pd.DataFrame, metric: str) -> float:
    if group.empty:
        return math.nan
    if metric == "top1_drop_0_to_75":
        return _metric_from_predictions(group, "full_top1") - _metric_from_predictions(group, "miss3_top1")
    if metric == "mae_at_75":
        return float(group.loc[group["missing_count"] == 3, "abs_error"].mean())
    if metric == "avg_missing_MAE":
        return float(group.loc[group["missing_count"] > 0, "abs_error"].mean())
    if metric == "avg_missing_within@3":
        return float(group.loc[group["missing_count"] > 0, "within3_correct"].mean())
    if metric == "full_top1":
        return float(group.loc[group["missing_count"] == 0, "top1_correct"].mean())
    if metric.startswith("miss") and metric.endswith("_top1"):
        return float(group.loc[group["missing_count"] == int(metric[4]), "top1_correct"].mean())
    return float(group.loc[group["missing_count"] > 0, "top1_correct"].mean())


def _signed_delta(method_value: Any, baseline_value: Any, metric: str) -> float:
    left = float_or_nan(method_value)
    right = float_or_nan(baseline_value)
    if not (math.isfinite(left) and math.isfinite(right)):
        return math.nan
    return right - left if metric in LOWER_BETTER else left - right


def _conclusion(delta: float, ci_low: float, prob: float) -> str:
    if not math.isfinite(delta) or delta <= 0:
        return "not_positive"
    if math.isfinite(ci_low) and ci_low > 0 and math.isfinite(prob) and prob >= 0.95:
        return "robust_positive"
    return "positive_but_uncertain"


def _is_mae_metric(metric: str) -> bool:
    return metric in {"avg_missing_MAE", "mae_at_75"}


def _is_pp_metric(metric: str) -> bool:
    return not _is_mae_metric(metric)


def _unit_fields(metric: str, seed_stats: dict[str, float], boot: dict[str, Any]) -> dict[str, Any]:
    mean_method = seed_stats["mean_method"]
    mean_baseline = seed_stats["mean_baseline"]
    seed_delta = seed_stats["seed_mean_delta"]
    boot_delta = boot.get("mean_delta", math.nan)
    ci_low = boot.get("ci_low", math.nan)
    ci_high = boot.get("ci_high", math.nan)
    if _is_mae_metric(metric):
        return {
            "mean_method_fraction": math.nan,
            "mean_baseline_fraction": math.nan,
            "seed_mean_delta_fraction": math.nan,
            "seed_mean_delta_pp": math.nan,
            "bootstrap_mean_delta_fraction": math.nan,
            "bootstrap_mean_delta_pp": math.nan,
            "bootstrap_ci_low_fraction": math.nan,
            "bootstrap_ci_high_fraction": math.nan,
            "bootstrap_ci_low_pp": math.nan,
            "bootstrap_ci_high_pp": math.nan,
            "mean_method": mean_method,
            "mean_baseline": mean_baseline,
            "seed_mean_delta": seed_delta,
            "bootstrap_mean_delta": boot_delta,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "metric_unit": "beam_index",
        }
    return {
        "mean_method_fraction": mean_method,
        "mean_baseline_fraction": mean_baseline,
        "seed_mean_delta_fraction": seed_delta,
        "seed_mean_delta_pp": seed_delta * 100.0 if math.isfinite(seed_delta) else math.nan,
        "bootstrap_mean_delta_fraction": boot_delta,
        "bootstrap_mean_delta_pp": boot_delta * 100.0 if math.isfinite(boot_delta) else math.nan,
        "bootstrap_ci_low_fraction": ci_low,
        "bootstrap_ci_high_fraction": ci_high,
        "bootstrap_ci_low_pp": ci_low * 100.0 if math.isfinite(ci_low) else math.nan,
        "bootstrap_ci_high_pp": ci_high * 100.0 if math.isfinite(ci_high) else math.nan,
        "mean_method": math.nan,
        "mean_baseline": math.nan,
        "seed_mean_delta": math.nan,
        "bootstrap_mean_delta": math.nan,
        "bootstrap_ci_low": math.nan,
        "bootstrap_ci_high": math.nan,
        "metric_unit": "fraction",
    }


def _sanity_notes(metric: str, boot: dict[str, Any], conclusion: str) -> list[str]:
    notes: list[str] = []
    mean_delta = float_or_nan(boot.get("mean_delta"))
    ci_low = float_or_nan(boot.get("ci_low"))
    ci_high = float_or_nan(boot.get("ci_high"))
    prob = float_or_nan(boot.get("prob_positive"))
    if _is_pp_metric(metric) and all(math.isfinite(value) for value in (mean_delta, ci_low, ci_high)):
        mean_pp = mean_delta * 100.0
        low_pp = ci_low * 100.0
        high_pp = ci_high * 100.0
        if not (low_pp <= mean_pp <= high_pp):
            notes.append(
                "WARNING: bootstrap_mean_delta_pp_outside_ci; "
                f"bootstrap_mean_delta_pp={mean_pp:.6g}, ci=[{low_pp:.6g}, {high_pp:.6g}]"
            )
    if conclusion == "robust_positive" and not (math.isfinite(ci_low) and ci_low > 0 and math.isfinite(prob) and prob >= 0.95):
        notes.append("WARNING: robust_positive_failed_bootstrap_guard")
    return notes


def _write_summary_md(path: Path, rows: list[dict[str, Any]], warnings: list[str]) -> None:
    visible = [
        {
            "Comparison": row["comparison"],
            "Metric": row["metric"],
            "Seed delta": _display_delta(row, prefix="seed_mean_delta"),
            "Bootstrap mean delta": _display_delta(row, prefix="bootstrap_mean_delta"),
            "Bootstrap 95% CI": _display_ci(row),
            "P(delta>0)": _fmt(row["prob_delta_positive"]),
            "Scenes +": row["num_scenes_positive"],
            "Conclusion": row["conclusion"],
            "Notes": row.get("notes", ""),
        }
        for row in rows
        if row["metric"] in {"avg_missing_top1", "miss3_top1", "avg_missing_MAE", "top1_drop_0_to_75"}
    ]
    write_md_table(path, visible, ["Comparison", "Metric", "Seed delta", "Bootstrap mean delta", "Bootstrap 95% CI", "P(delta>0)", "Scenes +", "Conclusion", "Notes"])
    if warnings:
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\nWarnings\n\n")
            for warning in sorted(set(warnings)):
                handle.write(f"- {warning}\n")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\nMethod labels\n\n")
        for _, method, baseline in {
            (row["comparison"], row["method"], row["baseline"]) for row in rows
        }:
            handle.write(f"- {method_label(method)} vs {method_label(baseline)}\n")


def _fmt(value: Any) -> str:
    number = float_or_nan(value)
    return f"{number:.6g}" if math.isfinite(number) else ""


def _display_delta(row: dict[str, Any], *, prefix: str) -> str:
    metric = str(row.get("metric", ""))
    if _is_mae_metric(metric):
        return _fmt(row.get(prefix))
    return f"{_fmt(row.get(prefix + '_pp'))} pp"


def _display_ci(row: dict[str, Any]) -> str:
    metric = str(row.get("metric", ""))
    if _is_mae_metric(metric):
        return f"[{_fmt(row.get('bootstrap_ci_low'))}, {_fmt(row.get('bootstrap_ci_high'))}]"
    return f"[{_fmt(row.get('bootstrap_ci_low_pp'))}, {_fmt(row.get('bootstrap_ci_high_pp'))}] pp"


if __name__ == "__main__":
    raise SystemExit(main())
