#!/usr/bin/env python3

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_overnight_branch_router_v2 as overnight
import launch_final_c2_ablation_v1 as launcher


DEFAULT_ROOT = "outputs/final_c2_ablation_v1"
DEFAULT_BASELINE_ROOTS = (
    "outputs/overnight_branch_router_v2,"
    "outputs/pcpg_radar_balance_v1,"
    "outputs/bprr_reliability_router_v1_retry_gpus0_6_20260706_193654"
)
METRICS = (
    "full",
    "avg_missing",
    "missing_image",
    "drop2",
    "drop3",
    "drop1",
    "drop1_3_mean",
    "single_modality_mean",
    "radar_only",
    "within3",
    "MAE",
)
DELTA_METRICS = ("avg_missing", "missing_image", "drop2", "radar_only", "full")
CONFIG_FIELDS = (
    "fusion_type",
    "router_supervision",
    "router_use_pattern_features",
    "router_use_reliability_features",
    "router_use_prototype_margin",
    "use_beam_prototype_alignment",
    "use_modality_prototype_loss",
    "use_circular_soft_targets",
    "head_type",
    "prototype_margin_enabled",
    "hard_subset_weighting",
    "use_jepa",
    "branch_aux_loss",
)
ROUTER_FIELDS = overnight.ROUTER_FIELDS + (
    "router_use_pattern_features",
    "router_use_reliability_features",
    "router_use_prototype_margin",
)


TABLES = {
    "main_results": (
        "e5_pcpg_low_encoder_lr",
        "a1_e5_low_encoder_lr_anchor",
        "e6_pcpg_hard_subset_jepa",
        "a2_e6_hard_subset_jepa_anchor",
        "a1_b4_nonrouter_soft_jepa",
        "b4_hard_soft_jepa",
        "a0_c2_full_main",
        "c2_supervised_router_hard_soft",
    ),
    "ablation_router": (
        "a0_c2_full_main",
        "b0_no_router_supervision",
        "b1_no_pattern_features",
        "b2_no_prototype_margin_feature",
        "b3_no_reliability_features_pattern_only",
        "b4_no_router_focus_all_patterns",
        "d2_raw_confidence_gate",
        "d3_bprr_unsupervised_router",
    ),
    "ablation_prototype": (
        "a0_c2_full_main",
        "c0_no_beam_prototype_alignment_loss",
        "c1_no_modality_prototype_loss",
        "c2_no_circular_soft_targets",
        "c3_classifier_head_no_prototype",
    ),
    "ablation_fusion": (
        "a0_c2_full_main",
        "d0_weighted_sum_fusion",
        "d1_average_fusion",
        "d2_raw_confidence_gate",
        "d3_bprr_unsupervised_router",
    ),
    "ablation_pattern_weighting": (
        "a0_c2_full_main",
        "e0_no_soft_hard_subset",
        "e1_static_hard_subset",
        "e2_soft_hard_without_router",
    ),
    "ablation_negative": (
        "a0_c2_full_main",
        "a1_b4_nonrouter_soft_jepa",
        "a2_c2_plus_jepa_negative",
        "a3_c2_plus_branch_aux_negative",
        "a4_c2_plus_branch_aux_jepa_negative",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize final c2 ablation v1 outputs.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--baseline_roots", "--baseline-roots", default=DEFAULT_BASELINE_ROOTS)
    parser.add_argument(
        "--include_gate_csv",
        "--include-gate-csv",
        action="store_true",
        help="Also read large per-epoch gate CSV files. Pattern eval rows are enough for the default paper tables.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    baseline_roots = [Path(item) for item in overnight.split_csv(args.baseline_roots)]
    all_roots = [root, *baseline_roots]

    pattern_rows = overnight.collect_pattern_rows(all_roots)
    gate_rows = overnight.collect_gate_rows(all_roots) if args.include_gate_csv else []
    run_rows = enrich_run_rows(summarize_runs_fast(pattern_rows, gate_rows))
    official_run_rows = [row for row in run_rows if not overnight.truthy(row.get("oracle_gate"))]
    summary_rows = aggregate_experiments(official_run_rows)
    router_rows = final_router_diagnostics(run_rows, gate_rows)

    write_csv(root / "pattern_metrics.csv", pattern_rows)
    write_csv(root / "router_diagnostics.csv", router_rows)
    write_csv(root / "run_summary.csv", run_rows)
    write_csv(root / "summary.csv", summary_rows)
    for name, experiments in TABLES.items():
        write_csv(root / f"{name}.csv", select_rows(summary_rows, experiments))
    (root / "summary.md").write_text(render_markdown(summary_rows, baseline_roots), encoding="utf-8")
    print(f"Wrote final c2 ablation v1 summary to {root}")
    return 0


def enrich_run_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        updated["single_modality_mean"] = overnight.mean_num(
            [
                overnight.float_or_nan(row.get("image_only")),
                overnight.float_or_nan(row.get("lidar_only")),
                overnight.float_or_nan(row.get("radar_only")),
                overnight.float_or_nan(row.get("gps_only")),
            ]
        )
        updated.update(inferred_config_values(str(row.get("experiment", ""))))
        out.append(updated)
    return out


def summarize_runs_fast(pattern_rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pattern_rows:
        by_run[(str(row.get("source_root")), str(row.get("experiment")), str(row.get("seed")))].append(row)
    gate_summary = overnight.summarize_gate_rows(gate_rows) if gate_rows else {}
    rows: list[dict[str, Any]] = []
    for (source_root, experiment, seed), items in sorted(by_run.items()):
        values = {str(row.get("pattern") or row.get("pattern_name")): row for row in items}
        drop1 = overnight.mean_patterns(values, missing_count=1)
        drop2 = overnight.mean_patterns(values, missing_count=2)
        drop3 = overnight.mean_patterns(values, missing_count=3)
        avg_missing = overnight.top1(values.get("avg_missing"))
        if not overnight.isnum(avg_missing):
            avg_missing = overnight.mean_num([drop1, drop2, drop3])
        row = {
            "source_root": source_root,
            "experiment": experiment,
            "seed": seed,
            "oracle_gate": str(any(overnight.truthy(item.get("oracle_gate")) for item in items)).lower(),
            "full": overnight.top1(values.get("full")),
            "drop1": drop1,
            "drop2": drop2,
            "drop3": drop3,
            "drop1_3_mean": overnight.mean_num([drop1, drop2, drop3]),
            "avg_missing": avg_missing,
            "image_only": overnight.top1(values.get("image_only")),
            "lidar_only": overnight.top1(values.get("lidar_only")),
            "radar_only": overnight.top1(values.get("radar_only")),
            "gps_only": overnight.top1(values.get("gps_only")),
            "missing_image": overnight.top1(values.get("missing_image")),
            "missing_lidar": overnight.top1(values.get("missing_lidar")),
            "missing_radar": overnight.top1(values.get("missing_radar")),
            "missing_gps": overnight.top1(values.get("missing_gps")),
            "within3": overnight.metric(values.get("avg_missing"), "within_3", "within3"),
            "MAE": overnight.metric(values.get("avg_missing"), "mae", "MAE"),
            **selection_info_fast(Path(source_root), experiment, seed),
            **gate_summary.get((source_root, experiment, seed), {}),
            **overnight.router_from_pattern_rows(items),
        }
        rows.append(row)
    return rows


def selection_info_fast(root: Path, experiment: str, seed: str) -> dict[str, Any]:
    for run_dir in candidate_run_dirs_fast(root, experiment, seed):
        for path in (
            run_dir / "checkpoints" / "best_avg_missing_top1.pth.json",
            run_dir / "checkpoints" / "best_top1.pth.json",
            run_dir / "checkpoints" / "best.pth.json",
        ):
            if path.exists():
                try:
                    import json

                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                return {
                    "selection_metric": payload.get("selection_metric", ""),
                    "best_epoch": payload.get("selected_epoch", payload.get("epoch", "")),
                    "clean_val_acc": (payload.get("task_metrics") or {}).get("val_acc", ""),
                }
        metrics = run_dir / "metrics.json"
        if metrics.exists():
            try:
                import json

                payload = json.loads(metrics.read_text(encoding="utf-8"))
            except Exception:
                continue
            latest = payload.get("latest", {}) if isinstance(payload, dict) else {}
            return {
                "selection_metric": latest.get("checkpoint_selection_metric", ""),
                "best_epoch": latest.get("best_early_stopping_epoch", ""),
                "clean_val_acc": latest.get("val_acc", ""),
            }
    return {}


def candidate_run_dirs_fast(root: Path, experiment: str, seed: str) -> list[Path]:
    candidates = overnight.candidate_run_dirs(root, experiment, seed)
    parent = root / experiment
    if parent.exists():
        candidates.extend(sorted(path for path in parent.glob(f"seed{seed}*") if path.is_dir()))
    return candidates


def inferred_config_values(experiment: str) -> dict[str, Any]:
    try:
        specs = launcher.experiment_specs()
    except Exception:
        specs = {}
    if experiment not in specs:
        return {}
    flat = flatten(specs[experiment].get("overrides", {}))
    head_type = str(flat.get("model.primary.head_type", "legacy") or "legacy")
    proto_margin = flat.get("model.primary.router_use_prototype_margin", True)
    return {
        "fusion_type": flat.get("model.primary.fusion_type", ""),
        "router_supervision": flat.get("loss.router_supervision", flat.get("model.primary.router_supervision", "")),
        "router_use_pattern_features": flat.get("model.primary.router_use_pattern_features", ""),
        "router_use_reliability_features": flat.get("model.primary.router_use_reliability_features", ""),
        "router_use_prototype_margin": proto_margin,
        "use_beam_prototype_alignment": flat.get(
            "training.use_beam_prototype_alignment", flat.get("model.primary.use_beam_prototype_alignment", "")
        ),
        "use_modality_prototype_loss": flat.get("training.use_modality_prototype_loss", ""),
        "use_circular_soft_targets": flat.get("training.use_circular_soft_targets", ""),
        "head_type": head_type,
        "prototype_margin_enabled": head_type != "classifier" and truthyish(proto_margin),
        "hard_subset_weighting": flat.get("loss.hard_subset_weighting.mode", ""),
        "use_jepa": flat.get("loss.use_jepa", flat.get("model.primary.use_jepa_loss", "")),
        "branch_aux_loss": flat.get("loss.branch_aux_loss", ""),
        "selection_metric": "avg_missing_top1",
    }


def flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(value, name))
        else:
            out[name] = value
    return out


def values_from_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    primary = payload.get("model", {}).get("primary", {}) if isinstance(payload.get("model"), dict) else {}
    loss = payload.get("loss", {}) if isinstance(payload.get("loss"), dict) else {}
    training = payload.get("training", {}) if isinstance(payload.get("training"), dict) else {}
    hard = loss.get("hard_subset_weighting", {})
    hard_mode = hard.get("mode", "") if isinstance(hard, dict) else str(hard or "")
    head_type = str(primary.get("head_type", training.get("head_type", "legacy")) or "legacy")
    use_proto_margin = primary.get("router_use_prototype_margin", True)
    return {
        "config_path": str(path),
        "fusion_type": primary.get("fusion_type", ""),
        "router_supervision": loss.get("router_supervision", primary.get("router_supervision", "")),
        "router_use_pattern_features": primary.get("router_use_pattern_features", ""),
        "router_use_reliability_features": primary.get("router_use_reliability_features", ""),
        "router_use_prototype_margin": use_proto_margin,
        "use_beam_prototype_alignment": training.get(
            "use_beam_prototype_alignment", primary.get("use_beam_prototype_alignment", "")
        ),
        "use_modality_prototype_loss": training.get(
            "use_modality_prototype_loss", primary.get("use_modality_prototype_loss", "")
        ),
        "use_circular_soft_targets": training.get("use_circular_soft_targets", primary.get("use_circular_soft_targets", "")),
        "head_type": head_type,
        "prototype_margin_enabled": str(head_type).lower() != "classifier" and truthyish(use_proto_margin),
        "hard_subset_weighting": hard_mode,
        "use_jepa": loss.get("use_jepa", primary.get("use_jepa_loss", "")),
        "branch_aux_loss": loss.get("branch_aux_loss", ""),
    }


def aggregate_experiments(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row.get("experiment"))].append(row)
    rows: list[dict[str, Any]] = []
    for experiment, items in sorted(grouped.items()):
        row: dict[str, Any] = {
            "experiment": experiment,
            "n": len(items),
            "seeds": ",".join(str(item.get("seed")) for item in items),
        }
        for metric_name in (*METRICS, *ROUTER_FIELDS):
            values = [overnight.float_or_nan(item.get(metric_name)) for item in items]
            valid = [value for value in values if overnight.isnum(value)]
            if valid:
                row[f"{metric_name}_mean"] = mean(valid)
                row[f"{metric_name}_std"] = pstdev(valid) if len(valid) > 1 else 0.0
        for field in CONFIG_FIELDS:
            row[field] = first_nonempty(item.get(field) for item in items)
        row["selection_metric"] = first_nonempty(item.get("selection_metric") for item in items)
        rows.append(row)
    return attach_c2_deltas(rows)


def attach_c2_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchor = find_first(rows, ("a0_c2_full_main", "c2_supervised_router_hard_soft"))
    out = []
    for row in rows:
        updated = dict(row)
        for metric_name in DELTA_METRICS:
            updated[f"delta_{metric_name}"] = overnight.float_or_nan(row.get(f"{metric_name}_mean")) - overnight.float_or_nan(
                (anchor or {}).get(f"{metric_name}_mean")
            )
        out.append(updated)
    return out


def final_router_diagnostics(run_rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fields = ("source_root", "experiment", "seed", "oracle_gate", *ROUTER_FIELDS)
    for row in run_rows:
        payload = {key: row.get(key, "") for key in fields}
        if any(str(payload.get(field, "")) not in {"", "nan"} for field in ROUTER_FIELDS):
            rows.append(payload)
    rows.extend(gate_rows)
    return rows


def select_rows(rows: list[dict[str, Any]], experiments: tuple[str, ...]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for name in experiments:
        match = find_first(rows, (name,))
        if match is not None and match not in selected:
            selected.append(match)
    return selected


def render_markdown(rows: list[dict[str, Any]], baseline_roots: list[Path]) -> str:
    lines = ["# Final C2 Ablation V1 Summary", ""]
    lines.extend(render_metric_table("7.1 主结果表", select_rows(rows, TABLES["main_results"])))
    lines.extend(render_delta_table("7.2 Router 消融表", select_rows(rows, TABLES["ablation_router"])))
    lines.extend(render_delta_table("7.3 Prototype 消融表", select_rows(rows, TABLES["ablation_prototype"])))
    lines.extend(render_delta_table("7.4 Fusion baseline 表", select_rows(rows, TABLES["ablation_fusion"])))
    lines.extend(render_delta_table("7.5 Pattern weighting 表", select_rows(rows, TABLES["ablation_pattern_weighting"])))
    lines.extend(render_delta_table("7.6 Negative / trade-off 表", select_rows(rows, TABLES["ablation_negative"])))
    lines.extend(render_recommendations(rows, baseline_roots))
    return "\n".join(lines) + "\n"


def render_metric_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    columns = ("full", "avg_missing", "missing_image", "drop2", "drop3", "single_modality_mean", "radar_only", "within3", "MAE")
    lines = [f"## {title}", "", "| experiment | n | " + " | ".join(columns) + " |", "| --- | ---: | " + " | ".join("---:" for _ in columns) + " |"]
    for row in rows:
        lines.append(f"| {row.get('experiment')} | {row.get('n')} | " + " | ".join(pm(row, col) for col in columns) + " |")
    return lines + [""]


def render_delta_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| experiment | n | avg_missing | missing_image | drop2 | radar_only | full | delta_avg_missing | delta_missing_image | delta_drop2 | delta_radar_only | delta_full |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('experiment')} | {row.get('n')} | {pm(row, 'avg_missing')} | {pm(row, 'missing_image')} | "
            f"{pm(row, 'drop2')} | {pm(row, 'radar_only')} | {pm(row, 'full')} | "
            f"{fmt(row.get('delta_avg_missing'))} | {fmt(row.get('delta_missing_image'))} | {fmt(row.get('delta_drop2'))} | "
            f"{fmt(row.get('delta_radar_only'))} | {fmt(row.get('delta_full'))} |"
        )
    lines.append("")
    lines.extend(auto_notes(title, rows))
    return lines + [""]


def auto_notes(title: str, rows: list[dict[str, Any]]) -> list[str]:
    c2 = find_first(rows, ("a0_c2_full_main",))
    if c2 is None:
        return ["- 自动结论：等待 final c2 run 完成后生成。"]
    notes = []
    for row in rows:
        exp = str(row.get("experiment"))
        if exp == "a0_c2_full_main":
            continue
        notes.append(f"- {exp}: avg_missing 相对 c2 为 {fmt(row.get('delta_avg_missing'))}，missing_image 为 {fmt(row.get('delta_missing_image'))}。")
    if "Fusion" in title:
        notes.append("- supervised router 若在 avg_missing/missing_image/drop2 上保持正优势，weighted_sum/average/raw confidence/BPRR 保留为对照而非主方法。")
    elif "Router" in title:
        notes.append("- 重点看 b0/b1/b2 的 delta：分别对应 oracle distillation、pattern features、prototype margin 的必要性。")
    elif "Prototype" in title:
        notes.append("- 重点看 c0/c1/c2/c3：分别对应 alignment、modality proto、soft target 与 prototype head。")
    elif "Pattern" in title:
        notes.append("- 重点看 e0/e1/e2：区分 soft_static 本身、static trade-off 与 router 协同。")
    elif "Negative" in title:
        notes.append("- a2/a3/a4 若提升 radar_only 但损害 avg_missing/full，应作为负交互或 trade-off 证据。")
    return notes


def render_recommendations(rows: list[dict[str, Any]], baseline_roots: list[Path]) -> list[str]:
    c2 = find_first(rows, ("a0_c2_full_main",))
    b4 = find_first(rows, ("a1_b4_nonrouter_soft_jepa", "b4_hard_soft_jepa"))
    best = best_experiment(rows, "avg_missing_mean")
    return [
        "## 7.7 最终推荐结论",
        "",
        f"- 最终主方法: {'a0_c2_full_main' if c2 else '等待 final c2 完成'}；当前 avg_missing 最优为 {best}。",
        f"- 最强非-router baseline: {b4.get('experiment') if b4 else '等待 a1/b4 完成'}。",
        "- 必须保留模块由 b0/b1/b2/c0/e0 的负 delta 判定；负交互由 a2/a3/a4 判定。",
        "- 主文优先放主表、router/prototype/fusion/pattern 表；appendix 放 per-pattern、BPRR 和 branch/JEPA trade-off 明细。",
        f"- Baseline roots 已合并: {', '.join(str(path) for path in baseline_roots) or '无'}。",
        "- seed/significance: 多 seed 完成后再根据 std 与 delta 大小决定是否补 significance test。",
    ]


def find_first(rows: list[dict[str, Any]], prefixes: tuple[str, ...]) -> dict[str, Any] | None:
    for row in rows:
        experiment = str(row.get("experiment", ""))
        if any(experiment == prefix or experiment.startswith(prefix) for prefix in prefixes):
            return row
    return None


def best_experiment(rows: list[dict[str, Any]], field: str) -> str:
    valid = [row for row in rows if math.isfinite(overnight.float_or_nan(row.get(field)))]
    return "unavailable" if not valid else str(max(valid, key=lambda row: overnight.float_or_nan(row.get(field))).get("experiment"))


def first_nonempty(values) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return ""


def truthyish(value: Any) -> bool:
    return str(value).strip().lower() not in {"", "0", "false", "no", "off", "none"}


def fmt(value: Any) -> str:
    number = overnight.float_or_nan(value)
    return "" if not overnight.isnum(number) else f"{number:.4f}"


def pm(row: dict[str, Any], metric_name: str) -> str:
    avg = overnight.float_or_nan(row.get(f"{metric_name}_mean"))
    std = overnight.float_or_nan(row.get(f"{metric_name}_std"))
    return "" if not overnight.isnum(avg) else f"{avg:.4f}+/-{std:.4f}" if overnight.isnum(std) else f"{avg:.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
