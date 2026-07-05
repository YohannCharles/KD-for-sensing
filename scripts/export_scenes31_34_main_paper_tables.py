#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path
from typing import Any


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
CORE_METHODS = (
    "scenes31_34_proto_natural_es40",
    "scenes31_34_proto_sampler_uniform_es40",
    "scenes31_34_proto_randomdrop_bernoulli_k075_es40",
    "scenes31_34_proto_randomdrop_subset_es40",
)
CLASSIFIER_TABLE = (
    "scenes31_34_classifier_natural_es40",
    "scenes31_34_proto_natural_es40",
    "scenes31_34_classifier_randomdrop_subset_es40",
    "scenes31_34_proto_randomdrop_subset_es40",
)
EXTERNAL_TABLE = (
    "scenes31_34_amr_lite_natural_es40",
    "scenes31_34_amber_lite_natural_es40",
    "scenes31_34_amr_lite_uniform_es40",
    "scenes31_34_amber_lite_uniform_es40",
    "scenes31_34_proto_randomdrop_subset_es40",
)
REFERENCE_METHOD = "scenes31_34_proto_randomdrop_subset_es40"
MAIN_FIELDS = ["Method", "Family", "n", "Full", "Miss-1", "Miss-2", "Miss-3", "Avg-Missing", "Within@3", "MAE", "Drop 0%->75%", "Main read"]
CURVE_FIELDS = [
    "Method",
    "Top1@0%",
    "Top1@25%",
    "Top1@50%",
    "Top1@75%",
    "Drop 0%->75%",
    "Within3@0%",
    "Within3@25%",
    "Within3@50%",
    "Within3@75%",
    "MAE@0%",
    "MAE@25%",
    "MAE@50%",
    "MAE@75%",
]
COST_FIELDS = ["Method", "Params", "Model size", "Train time / epoch", "Inference latency / sample", "Samples / second", "Extra inference cost"]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary_root = Path(args.summary_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    methods = _load_methods(summary_root)
    curve = _read_csv(summary_root / "final_missing_count_curve.csv") or _read_csv(summary_root / "missing_count_curve.csv")
    scene = _read_csv(summary_root / "mean_over_scenes.csv")

    main_rows = _main_rows(methods)
    ablation_rows = _ablation_rows(methods)
    classifier_rows = [_table_row(_method_row(methods, method)) for method in CLASSIFIER_TABLE]
    external_rows = [_external_row(_method_row(methods, method)) for method in EXTERNAL_TABLE]
    curve_rows = _curve_rows(curve)
    scene_rows = _scene_rows(scene)
    cost_rows = _cost_rows(Path(args.profile_root))

    _write_csv(out / "table_scenes31_34_main.csv", main_rows, MAIN_FIELDS)
    _write_md(out / "table_scenes31_34_main.md", main_rows, MAIN_FIELDS)
    _write_csv(out / "table_scenes31_34_ablation.csv", ablation_rows, MAIN_FIELDS)
    _write_md(out / "table_scenes31_34_ablation.md", ablation_rows, MAIN_FIELDS)
    _write_csv(out / "table_scenes31_34_classifier_baseline.csv", classifier_rows, MAIN_FIELDS)
    _write_md(out / "table_scenes31_34_classifier_baseline.md", classifier_rows, MAIN_FIELDS)
    _write_csv(out / "table_scenes31_34_external_baselines.csv", external_rows, MAIN_FIELDS)
    _write_md(out / "table_scenes31_34_external_baselines.md", external_rows, MAIN_FIELDS)
    _write_csv(out / "table_scenes31_34_missing_count_curve.csv", curve_rows, CURVE_FIELDS)
    _write_csv(out / "table_scenes31_34_scene_stability.csv", scene_rows, list(scene_rows[0]) if scene_rows else ["Method"])
    _write_csv(out / "table_compute_cost.csv", cost_rows, COST_FIELDS)
    _write_md(out / "table_compute_cost.md", cost_rows, COST_FIELDS)
    _write_notes(out / "scenes31_34_main_paper_notes.txt", Path(args.fig_root), methods, cost_rows)
    print(f"Wrote Scene31-34 paper tables to {out}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Scene31-34 main paper tables.")
    parser.add_argument("--summary-root", default="outputs/scenes31_34_main_lmdb/summary")
    parser.add_argument("--fig-root", default="outputs/scenes31_34_main_lmdb/figures")
    parser.add_argument("--profile-root", default="outputs/scenes31_34_main_lmdb/profile")
    parser.add_argument("--out", default="outputs/paper_tables/scenes31_34_main")
    return parser


def _load_methods(summary_root: Path) -> list[dict[str, str]]:
    return _read_csv(summary_root / "final_method_mean_std.csv") or _read_csv(summary_root / "method_mean_std.csv")


def _main_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = [_table_row(_method_row(rows, method)) for method in CORE_METHODS]
    out.extend(_table_row(_method_row(rows, method)) for method in ("scenes31_34_classifier_natural_es40", "scenes31_34_classifier_randomdrop_subset_es40"))
    out.append(_best_external_row(rows, "amr_lite", "AMR-lite best available"))
    out.append(_best_external_row(rows, "amber_lite", "AMBER-lite best available"))
    return out


def _ablation_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [_table_row(_method_row(rows, method)) for method in CORE_METHODS]


def _best_external_row(rows: list[dict[str, str]], token: str, label: str) -> dict[str, str]:
    candidates = [row for row in rows if token in str(row.get("method") or "") and _truthy(row.get("official_ranking_included"))]
    if not candidates:
        visible = [row for row in rows if token in str(row.get("method") or "") and _int(row.get("mask_suspect_count")) > 0]
        return _status_row(label, "external_lite", "incomplete: mask_suspect excluded" if visible else "pending: not run")
    best = max(candidates, key=lambda row: _float(row.get("avg_missing_top1_mean")))
    item = _table_row(best)
    item["Method"] = label
    return item


def _table_row(row: dict[str, str]) -> dict[str, str]:
    method = str(row.get("method") or "")
    n = _int(row.get("n"))
    if not method:
        return _status_row("not run", "", "not run")
    status = _row_status(row)
    if status != "ok":
        caveat = row.get("caveat", "")
        return _status_row(METHOD_LABELS.get(method, method), row.get("family", ""), f"{status}; {caveat}" if caveat else status)
    return {
        "Method": METHOD_LABELS.get(method, method),
        "Family": row.get("family", ""),
        "n": str(n),
        "Full": _mean_std_pct(row, "full_top1"),
        "Miss-1": _mean_std_pct(row, "miss1_top1"),
        "Miss-2": _mean_std_pct(row, "miss2_top1"),
        "Miss-3": _mean_std_pct(row, "miss3_top1"),
        "Avg-Missing": _mean_std_pct(row, "avg_missing_top1"),
        "Within@3": _mean_std_pct(row, "avg_missing_within@3"),
        "MAE": _mean_std_raw(row, "avg_missing_MAE"),
        "Drop 0%->75%": _mean_std_pct(row, "top1_drop_0_to_75"),
        "Main read": row.get("main_read", ""),
    }


def _external_row(row: dict[str, str]) -> dict[str, str]:
    return _table_row(row)


def _status_row(label: str, family: str, status: str) -> dict[str, str]:
    row = {field: "" for field in MAIN_FIELDS}
    row["Method"] = label
    row["Family"] = family
    row["Main read"] = status
    return row


def _curve_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_method_count = {(row.get("method"), _int(row.get("missing_count"))): row for row in rows}
    out = []
    for method in CORE_METHODS:
        item = {"Method": METHOD_LABELS[method]}
        for ratio, count in (("0%", 0), ("25%", 1), ("50%", 2), ("75%", 3)):
            row = by_method_count.get((method, count), {})
            item[f"Top1@{ratio}"] = _pct(row.get("top1_mean"))
            item[f"Within3@{ratio}"] = _pct(row.get("within3_mean"))
            item[f"MAE@{ratio}"] = _raw(row.get("mae_mean"), digits=3)
        item["Drop 0%->75%"] = _delta_pct(by_method_count.get((method, 0), {}).get("top1_mean"), by_method_count.get((method, 3), {}).get("top1_mean"))
        out.append(item)
    return out


def _scene_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in sorted(rows, key=lambda item: _method_rank(str(item.get("method") or ""))):
        out.append(
            {
                "Method": METHOD_LABELS.get(str(row.get("method") or ""), str(row.get("method") or "")),
                "Scene-mean Avg-Missing": _pct(row.get("avg_missing_top1_mean_over_scenes")),
                "Scene std Avg-Missing": _pct(row.get("avg_missing_top1_std_over_scenes")),
                "Scene-mean MAE": _raw(row.get("avg_missing_MAE_mean_over_scenes"), digits=3),
                "Balanced scene mean": _pct(row.get("balanced_mean_over_scenes")),
            }
        )
    return out


def _cost_rows(profile_root: Path) -> list[dict[str, str]]:
    path = profile_root / "method_profile_summary.csv"
    rows = _read_csv(path)
    if not rows:
        return [{field: ("not run" if field == "Extra inference cost" else "NaN") for field in COST_FIELDS}]
    out = []
    for row in rows:
        out.append(
            {
                "Method": METHOD_LABELS.get(str(row.get("method") or ""), str(row.get("method") or "")),
                "Params": _compact_int(row.get("num_params")),
                "Model size": _mb(row.get("model_size_mb")),
                "Train time / epoch": _sec(row.get("train_time_per_epoch_sec")),
                "Inference latency / sample": _ms(row.get("eval_latency_per_sample_ms")),
                "Samples / second": _raw(row.get("eval_samples_per_second"), digits=1) or "NaN",
                "Extra inference cost": row.get("extra_inference_cost", ""),
            }
        )
    return out


def _write_notes(path: Path, fig_root: Path, rows: list[dict[str, str]], cost_rows: list[dict[str, str]]) -> None:
    winner = _winner(rows)
    subset = _method_row(rows, REFERENCE_METHOD)
    trusted = (
        "Final trusted method: prototype + random non-empty subset exposure."
        if winner == REFERENCE_METHOD and _int(subset.get("n")) >= 5
        else f"Current Avg-Missing winner: {METHOD_LABELS.get(winner, winner)}; final text must follow the actual summary."
    )
    lines = [
        "Scene31-34 is the main missing-modality evaluation setting.",
        trusted,
        "Uniform is an ablation, not the final reference.",
        "Reliability fusion and PatternFiLM are not promoted; JTT/MVFR/MPDRO/beamsoft/condBTAPA/weakKD are excluded from this main table.",
        "Classifier baselines test whether beam-centered prototype prediction adds value over an ordinary CE classifier.",
        "AMR/AMBER-lite rows enter official ranking only when mask_suspect=false; missing rows are reported as pending and mask_suspect rows as incomplete/excluded.",
        "Random subset exposure introduces no extra inference-time parameters or latency relative to the same proto model; it is a training exposure strategy.",
        "The missing-count degradation curve reports missing_count=0/1/2/3, corresponding to 0%/25%/50%/75% missing ratio.",
        f"Figure root: {fig_root}",
        f"Compute rows visible: {len(cost_rows)}",
    ]
    caveats = _method_caveats(rows)
    if caveats:
        lines.extend(["", "Pending / incomplete caveats:", *[f"- {item}" for item in caveats]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _method_row(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    return next((row for row in rows if row.get("method") == method), {"method": method, "family": _family(method), "n": "0", "main_read": "not run"})


def _row_status(row: dict[str, str]) -> str:
    if _int(row.get("n")) <= 0:
        if _int(row.get("mask_suspect_count")) > 0 or row.get("main_read") == "excluded":
            return "incomplete"
        return "pending"
    if not _truthy(row.get("official_ranking_included")):
        return "incomplete"
    return "ok"


def _method_caveats(rows: list[dict[str, str]]) -> list[str]:
    caveats = []
    for row in rows:
        status = str(row.get("claim_status") or "").strip()
        caveat = str(row.get("caveat") or "").strip()
        if status and status != "complete":
            label = METHOD_LABELS.get(str(row.get("method") or ""), str(row.get("method") or "method"))
            caveats.append(f"{label}: {status}" + (f" ({caveat})" if caveat else ""))
    return caveats


def _winner(rows: list[dict[str, str]]) -> str:
    valid = [row for row in rows if _truthy(row.get("official_ranking_included")) and math.isfinite(_float(row.get("avg_missing_top1_mean")))]
    if not valid:
        return ""
    return max(valid, key=lambda row: _float(row.get("avg_missing_top1_mean"))).get("method", "")


def _family(method: str) -> str:
    if method.startswith("scenes31_34_proto"):
        return "proto"
    if "classifier" in method:
        return "classifier"
    if "amr_lite" in method or "amber_lite" in method:
        return "external_lite"
    return "auxiliary"


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    lines = ["| " + " | ".join(fieldnames) + " |", "| " + " | ".join("---" for _ in fieldnames) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_std_pct(row: dict[str, str], metric: str) -> str:
    value = _pct(row.get(f"{metric}_mean"))
    std = _pct(row.get(f"{metric}_std"))
    return value if not std else f"{value}+-{std}"


def _mean_std_raw(row: dict[str, str], metric: str) -> str:
    value = _raw(row.get(f"{metric}_mean"), digits=3)
    std = _raw(row.get(f"{metric}_std"), digits=3)
    return value if not std else f"{value}+-{std}"


def _pct(value: Any) -> str:
    number = _float(value)
    if not math.isfinite(number):
        return ""
    number = number * 100.0 if abs(number) <= 1.5 else number
    return f"{number:.2f}"


def _raw(value: Any, *, digits: int) -> str:
    number = _float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else ""


def _delta_pct(left: Any, right: Any) -> str:
    a = _float(left)
    b = _float(right)
    if not (math.isfinite(a) and math.isfinite(b)):
        return ""
    delta = a - b
    delta = delta * 100.0 if abs(delta) <= 1.5 else delta
    return f"{delta:.2f}"


def _compact_int(value: Any) -> str:
    number = _float(value)
    return f"{int(number):,}" if math.isfinite(number) else "NaN"


def _mb(value: Any) -> str:
    number = _float(value)
    return f"{number:.2f} MB" if math.isfinite(number) else "NaN"


def _sec(value: Any) -> str:
    number = _float(value)
    return f"{number:.2f} s" if math.isfinite(number) else "NaN"


def _ms(value: Any) -> str:
    number = _float(value)
    return f"{number:.3f} ms" if math.isfinite(number) else "NaN"


def _method_rank(method: str) -> tuple[int, str]:
    order = (*CORE_METHODS, *CLASSIFIER_TABLE, *EXTERNAL_TABLE)
    try:
        return (order.index(method), method)
    except ValueError:
        return (len(order), method)


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
