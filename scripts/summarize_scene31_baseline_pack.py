#!/usr/bin/env python3

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from statistics import mean
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_scene31_bc_next


UNIFORM_REFERENCE = {
    "method": "proto_sampler_uniform_es40",
    "full_top1": 0.42509,
    "avg_missing_top1": 0.28255,
    "overall_mean_top1": 0.27576,
    "miss2_top1": 0.27639,
    "avg_missing_within_3": 0.68875,
    "avg_missing_mae": 4.64303,
}
PRIMARY_SORT = ("avg_missing_top1_mean", "miss2_top1_mean", "miss3_top1_mean", "full_top1_mean")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    forwarded = ["--root", args.root, "--out", str(out_dir), "--name-prefix", ""]
    if args.uniform_root:
        for path in _uniform_metric_paths(Path(args.uniform_root)):
            forwarded.extend(["--metrics", str(path)])
    if args.proto_root:
        forwarded.extend(["--root", args.proto_root])
    for metrics in args.metrics:
        forwarded.extend(["--metrics", metrics])
    summarize_scene31_bc_next.main(forwarded)

    per_run = _read_csv(out_dir / "per_run.csv")
    _annotate_rows(per_run)
    params = _collect_params([Path(args.root), *([Path(args.uniform_root)] if args.uniform_root else [])], per_run)
    _merge_params(per_run, params)
    method_rows = _method_rows(per_run)
    delta_rows = _delta_rows(method_rows)
    comparison_rows = _backbone_comparison(method_rows)

    _write_csv(out_dir / "baseline_per_run.csv", per_run, _fields(per_run, _per_run_first_fields()))
    _write_csv(out_dir / "baseline_method_mean_std.csv", method_rows, _fields(method_rows, _method_first_fields()))
    _write_csv(out_dir / "baseline_delta_vs_uniform.csv", delta_rows, _fields(delta_rows, ["method", "n"]))
    _write_csv(out_dir / "backbone_training_comparison.csv", comparison_rows, _fields(comparison_rows, _comparison_fields()))
    _write_csv(out_dir / "params_comparison.csv", _params_rows(method_rows), _fields(_params_rows(method_rows), ["method", "model_family"]))
    _copy_if_exists(out_dir / "rank_by_avg_missing_top1.md", out_dir / "rank_by_avg_missing_top1.md")
    _copy_if_exists(out_dir / "rank_by_miss1_top1.md", out_dir / "rank_by_miss1_top1.md")
    _copy_if_exists(out_dir / "rank_by_miss2_top1.md", out_dir / "rank_by_miss2_top1.md")
    _copy_if_exists(out_dir / "rank_by_miss3_top1.md", out_dir / "rank_by_miss3_top1.md")
    _copy_if_exists(out_dir / "rank_by_beam_proximity.md", out_dir / "rank_by_beam_proximity.md")
    _write_conclusion(out_dir / "baseline_conclusion.txt", method_rows, Path(args.root))

    _copy_if_exists(out_dir / "per_run.csv", out_dir / "baseline_per_run_raw.csv")
    _copy_if_exists(out_dir / "method_mean_std.csv", out_dir / "baseline_method_mean_std_raw.csv")
    print(f"Wrote Scene31 baseline pack summary to {out_dir}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Scene31 baseline pack fresh eval metrics.")
    parser.add_argument("--root", default="outputs/scene31_baseline_pack_lmdb")
    parser.add_argument("--uniform-root", default="outputs/scene31_funnel_lmdb")
    parser.add_argument("--proto-root", default="")
    parser.add_argument("--out", default="outputs/scene31_baseline_pack_lmdb/summary")
    parser.add_argument("--metrics", action="append", default=[])
    return parser


def _uniform_metric_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in sorted(root.rglob("apples_to_apples_metrics.csv"))
        if path.parent.name.startswith("proto_sampler_uniform_es40_seed")
    ]


def _annotate_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        method = str(row.get("method") or _method_name(str(row.get("run_name") or "")))
        row["method"] = method
        if row.get("avg_missing_mae") not in (None, ""):
            row["avg_missing_MAE"] = row.get("avg_missing_mae", "")
        row["model_family"] = _model_family(method)
        row["training_strategy"] = _training_strategy(method)
        row["main_read"] = _main_read(row)


def _collect_params(roots: list[Path], rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_run: dict[str, dict[str, float]] = {}
    for row in rows:
        run_name = str(row.get("run_name") or "")
        if not run_name or run_name in by_run:
            continue
        for root in roots:
            summary = _startup_summary_path(root, run_name)
            if summary is None:
                continue
            params = _params_from_startup(summary)
            if params:
                by_run[run_name] = params
                break
    return by_run


def _merge_params(rows: list[dict[str, Any]], params: dict[str, dict[str, float]]) -> None:
    proto_values = [
        item["total_params"]
        for row in rows
        for item in [params.get(str(row.get("run_name") or ""), {})]
        if row.get("model_family") == "proto" and _isnum(item.get("total_params"))
    ]
    proto_total = mean(proto_values) if proto_values else float("nan")
    for row in rows:
        values = params.get(str(row.get("run_name") or ""), {})
        row["total_params"] = values.get("total_params", "")
        row["trainable_params"] = values.get("trainable_params", "")
        row["extra_params_vs_proto"] = (
            values["total_params"] - proto_total
            if _isnum(values.get("total_params")) and _isnum(proto_total)
            else ""
        )


def _method_rows(per_run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in per_run:
        grouped.setdefault(str(row["method"]), []).append(row)
    metric_fields = [
        field
        for row in per_run
        for field, value in row.items()
        if field not in {"seed"} and _isnum(value)
    ]
    ordered_metrics = list(dict.fromkeys(metric_fields))
    out: list[dict[str, Any]] = []
    for method, rows in grouped.items():
        item: dict[str, Any] = {
            "method": method,
            "model_family": _model_family(method),
            "training_strategy": _training_strategy(method),
            "n": len(rows),
        }
        for metric in ordered_metrics:
            values = [_float(row.get(metric)) for row in rows if _isnum(row.get(metric))]
            item[f"{metric}_mean"] = mean(values) if values else ""
            item[f"{metric}_std"] = _std(values) if len(values) > 1 else 0.0 if values else ""
        if not _isnum(item.get("total_params_mean")):
            item["total_params_mean"] = ""
        if not _isnum(item.get("trainable_params_mean")):
            item["trainable_params_mean"] = ""
        if not _isnum(item.get("extra_params_vs_proto_mean")):
            item["extra_params_vs_proto"] = ""
        else:
            item["extra_params_vs_proto"] = item["extra_params_vs_proto_mean"]
        item["main_read"] = _method_read(item)
        out.append(item)
    return sorted(out, key=_rank_key, reverse=True)


def _delta_rows(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    uniform = _find(method_rows, "proto_sampler_uniform_es40")
    out = []
    for row in method_rows:
        item = {"method": row["method"], "n": row.get("n", "")}
        for metric in ("full_top1", "miss1_top1", "miss2_top1", "miss3_top1", "avg_missing_top1", "overall_mean_top1", "avg_missing_within_3", "avg_missing_mae", "balanced"):
            value = _method_value(row, metric)
            base = _method_value(uniform, metric) if uniform is not None else UNIFORM_REFERENCE.get(metric)
            item[metric] = value
            item[f"delta_vs_uniform_{metric}"] = _delta(value, base)
        out.append(item)
    return out


def _backbone_comparison(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_family = {}
    for row in method_rows:
        by_family.setdefault(row.get("model_family"), []).append(row)
    for family in ("proto", "amr_lite", "amber_lite", "featuremod_lite"):
        candidates = by_family.get(family, [])
        uniform = next((row for row in candidates if row.get("training_strategy") == "pattern_balanced_uniform"), None)
        if uniform is None and family == "proto":
            uniform = _find(method_rows, "proto_sampler_uniform_es40")
        for baseline in candidates:
            if baseline is uniform or baseline.get("training_strategy") == "pattern_balanced_uniform":
                continue
            item = {
                "model_family": family,
                "baseline_training": baseline.get("training_strategy", ""),
                "pattern_balanced_training": "pattern_balanced_uniform" if uniform else "unavailable",
            }
            for metric, out_key in (
                ("full_top1", "delta_full"),
                ("miss1_top1", "delta_miss1"),
                ("miss2_top1", "delta_miss2"),
                ("miss3_top1", "delta_miss3"),
                ("avg_missing_top1", "delta_avg_missing"),
                ("overall_mean_top1", "delta_overall"),
                ("avg_missing_within_3", "delta_within@3"),
                ("avg_missing_mae", "delta_MAE"),
            ):
                item[out_key] = _delta(_method_value(uniform, metric), _method_value(baseline, metric))
            rows.append(item)
    return rows


def _write_conclusion(path: Path, rows: list[dict[str, Any]], root: Path) -> None:
    best = rows[0]["method"] if rows else "unavailable"
    lightweight = next((row["method"] for row in rows if row.get("model_family") in {"proto", "amr_lite", "featuremod_lite"}), "unavailable")
    subset = _find(rows, "proto_randomdrop_subset_es40")
    bernoulli = _find(rows, "proto_randomdrop_bernoulli_k075_es40")
    proto_uniform = _find(rows, "proto_sampler_uniform_es40")
    amr_uniform = _find(rows, "amr_lite_uniform_es40")
    amber_uniform = _find(rows, "amber_lite_uniform_es40")
    lines = [
        "Fresh eval status:",
        *_status_lines(root),
        "",
        f"Best method by avg_missing_top1: {best}",
        f"Best lightweight method: {lightweight}",
        "",
        "Does random dropout match uniform?",
        f"- randomdrop_subset vs uniform: {_compare_text(subset, proto_uniform, 'avg_missing_top1')}",
        f"- randomdrop_bernoulli vs uniform: {_compare_text(bernoulli, proto_uniform, 'avg_missing_top1')}",
        "",
        "Does pattern-balanced exposure generalize across backbones?",
        f"- proto: {_family_delta(rows, 'proto')}",
        f"- amr_lite: {_family_delta(rows, 'amr_lite')}",
        f"- amber_lite: {_family_delta(rows, 'amber_lite')}",
        "",
        "Does a complex baseline outperform proto+uniform?",
        f"- amr_lite_uniform vs proto_uniform: {_compare_text(amr_uniform, proto_uniform, 'avg_missing_top1')}",
        f"- amber_lite_uniform vs proto_uniform: {_compare_text(amber_uniform, proto_uniform, 'avg_missing_top1')}",
        "",
        "Parameter efficiency:",
        *_param_lines(rows),
        "",
        "Recommendation:",
        *_recommendation_lines(rows),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _startup_summary_path(root: Path, run_name: str) -> Path | None:
    for candidate in (root / run_name / "startup_summary.json", root / "scene31" / run_name / "startup_summary.json"):
        if candidate.exists():
            return candidate
    return None


def _params_from_startup(path: Path) -> dict[str, float]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    params = data.get("parameters", {}) if isinstance(data, dict) else {}
    return {
        "total_params": _float(params.get("total_params")),
        "trainable_params": _float(params.get("trainable_params")),
    }


def _model_family(method: str) -> str:
    if method.startswith("amr_lite"):
        return "amr_lite"
    if method.startswith("amber_lite"):
        return "amber_lite"
    if method.startswith("featuremod_lite"):
        return "featuremod_lite"
    return "proto"


def _training_strategy(method: str) -> str:
    if "randomdrop_bernoulli" in method:
        return "randomdrop_bernoulli"
    if "randomdrop_subset" in method:
        return "randomdrop_subset"
    if "uniform" in method or "sampler_uniform" in method:
        return "pattern_balanced_uniform"
    if "natural" in method:
        return "natural"
    return "unknown"


def _main_read(row: dict[str, Any]) -> str:
    n = int(_float(row.get("n", 1)) if _isnum(row.get("n", 1)) else 1)
    return "quick_screen" if n == 1 else "standard_compare"


def _method_read(row: dict[str, Any]) -> str:
    if int(_float(row.get("n", 0)) if _isnum(row.get("n", 0)) else 0) < 3:
        return "quick_screen"
    return "standard_compare"


def _status_lines(root: Path) -> list[str]:
    names = [
        ("ok", "baseline_pack_completed_runs.txt"),
        ("skipped", "baseline_pack_skipped_runs.txt"),
        ("failed", "baseline_pack_failed_runs.txt"),
        ("missing_config", "baseline_pack_missing_config_runs.txt"),
        ("missing_checkpoint", "baseline_pack_missing_checkpoint_runs.txt"),
    ]
    return [f"- {label}: {len(_list_file(root / filename))}" for label, filename in names]


def _param_lines(rows: list[dict[str, Any]]) -> list[str]:
    out = []
    for row in rows:
        total = _method_value(row, "total_params")
        extra = row.get("extra_params_vs_proto", "")
        out.append(f"- {row.get('method')}: total={_fmt(total)} extra_vs_proto={_fmt(extra)}")
    return out or ["- unavailable"]


def _recommendation_lines(rows: list[dict[str, Any]]) -> list[str]:
    out = []
    proto = _find(rows, "proto_sampler_uniform_es40")
    for row in rows:
        delta = _delta(_method_value(row, "avg_missing_top1"), _method_value(proto, "avg_missing_top1"))
        verdict = "promote" if _isnum(delta) and delta > 0 else "do_not_promote"
        if row.get("main_read") == "quick_screen":
            verdict = "quick_screen_only"
        out.append(f"- {row.get('method')}: {verdict}")
    return out or ["- unavailable"]


def _family_delta(rows: list[dict[str, Any]], family: str) -> str:
    candidates = [row for row in rows if row.get("model_family") == family]
    uniform = next((row for row in candidates if row.get("training_strategy") == "pattern_balanced_uniform"), None)
    baseline = next((row for row in candidates if row.get("training_strategy") in {"natural", "randomdrop_subset"}), None)
    if uniform is None or baseline is None:
        return "unavailable"
    return f"delta_avg_missing={_fmt(_delta(_method_value(uniform, 'avg_missing_top1'), _method_value(baseline, 'avg_missing_top1')))}"


def _compare_text(candidate: dict[str, Any] | None, baseline: dict[str, Any] | None, metric: str) -> str:
    delta = _delta(_method_value(candidate, metric), _method_value(baseline, metric))
    if not _isnum(delta):
        return "unavailable"
    return f"delta={_fmt(delta)}"


def _params_rows(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "method": row.get("method", ""),
            "model_family": row.get("model_family", ""),
            "total_params_mean": row.get("total_params_mean", ""),
            "trainable_params_mean": row.get("trainable_params_mean", ""),
            "extra_params_vs_proto": row.get("extra_params_vs_proto", ""),
        }
        for row in method_rows
    ]


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists() and source != target:
        shutil.copyfile(source, target)


def _find(rows: list[dict[str, Any]], method: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("method") == method), None)


def _method_value(row: dict[str, Any] | None, metric: str) -> float:
    if row is None:
        return float("nan")
    return _float(row.get(f"{metric}_mean", row.get(metric)))


def _rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(_zero_nan(row.get(metric)) for metric in PRIMARY_SORT)


def _delta(value: Any, base: Any) -> float:
    value_f = _float(value)
    base_f = _float(base)
    return value_f - base_f if _isnum(value_f) and _isnum(base_f) else float("nan")


def _std(values: list[float]) -> float:
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def _method_name(run_name: str) -> str:
    import re

    return re.sub(r"_seed\d+$", "", run_name)


def _list_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(value) if isinstance(value, float) else value for key, value in row.items()})


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
        "model_family",
        "training_strategy",
        "n",
        "full_top1",
        "miss1_top1",
        "miss2_top1",
        "miss3_top1",
        "avg_missing_top1",
        "overall_mean_top1",
        "avg_missing_within@3",
        "avg_missing_mae",
        "balanced",
        "total_params",
        "trainable_params",
        "extra_params_vs_proto",
        "main_read",
    ]


def _method_first_fields() -> list[str]:
    return [
        "method",
        "model_family",
        "training_strategy",
        "n",
        "full_top1_mean",
        "full_top1_std",
        "miss1_top1_mean",
        "miss1_top1_std",
        "miss2_top1_mean",
        "miss2_top1_std",
        "miss3_top1_mean",
        "miss3_top1_std",
        "avg_missing_top1_mean",
        "avg_missing_top1_std",
        "overall_mean_top1_mean",
        "overall_mean_top1_std",
        "avg_missing_within_3_mean",
        "avg_missing_within_3_std",
        "avg_missing_mae_mean",
        "avg_missing_mae_std",
        "balanced_mean",
        "balanced_std",
        "total_params_mean",
        "trainable_params_mean",
        "extra_params_vs_proto",
        "main_read",
    ]


def _comparison_fields() -> list[str]:
    return [
        "model_family",
        "baseline_training",
        "pattern_balanced_training",
        "delta_full",
        "delta_miss1",
        "delta_miss2",
        "delta_miss3",
        "delta_avg_missing",
        "delta_overall",
        "delta_within@3",
        "delta_MAE",
    ]


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _isnum(value: Any) -> bool:
    return math.isfinite(_float(value))


def _zero_nan(value: Any) -> float:
    value_f = _float(value)
    return value_f if math.isfinite(value_f) else 0.0


def _fmt(value: Any) -> str:
    value_f = _float(value)
    return f"{value_f:.8g}" if math.isfinite(value_f) else ""


if __name__ == "__main__":
    raise SystemExit(main())
