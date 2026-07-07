#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import yaml


DEFAULT_ROOT = "outputs/overnight_branch_router_v2"
METRICS = ("full", "drop1", "drop2", "drop3", "drop1_3_mean", "avg_missing", "radar_only", "missing_image", "within3", "MAE")
DELTA_METRICS = ("full", "avg_missing", "drop3", "radar_only", "missing_image")
PATTERN_FILES = {"eval_matrix.csv", "oracle_eval_matrix.csv", "pattern_metrics.csv"}
GATE_FILES = {"reliability_weights_epoch.csv", "pcpg_gate_diagnostics.csv", "gate_diagnostics.csv"}
ROUTER_FIELDS = (
    "mean_gate_image",
    "mean_gate_lidar",
    "mean_gate_radar",
    "mean_gate_gps",
    "gate_entropy",
    "router_oracle_acc",
    "router_oracle_acc_missing_image",
    "router_oracle_acc_drop2",
    "oracle_target_image_rate",
    "oracle_target_lidar_rate",
    "oracle_target_radar_rate",
    "oracle_target_gps_rate",
    "radar_gate_missing_image",
    "radar_gate_drop2",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize overnight branch/router v2 outputs.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--baseline_roots", "--baseline-roots", default="")
    args = parser.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    baseline_roots = [Path(item) for item in split_csv(args.baseline_roots)]
    all_roots = [root, *baseline_roots]

    pattern_rows = collect_pattern_rows(all_roots)
    gate_rows = collect_gate_rows(all_roots)
    run_rows = summarize_runs(pattern_rows, gate_rows, all_roots)
    official_run_rows = [row for row in run_rows if not truthy(row.get("oracle_gate"))]
    summary_rows = aggregate_experiments(official_run_rows)
    drop_rows = drop_count_summary(pattern_rows)
    router_rows = router_diagnostics(run_rows, gate_rows)

    write_csv(root / "pattern_metrics.csv", pattern_rows)
    write_csv(root / "drop_count_summary.csv", drop_rows)
    write_csv(root / "router_diagnostics.csv", router_rows)
    write_csv(root / "run_summary.csv", run_rows)
    write_csv(root / "summary.csv", summary_rows)
    (root / "summary.md").write_text(render_markdown(summary_rows, official_run_rows, baseline_roots), encoding="utf-8")
    print(f"Wrote overnight branch/router v2 summary to {root}")
    return 0


def collect_pattern_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            if path.name not in PATTERN_FILES and not path.name.endswith("_missing_patterns.csv"):
                continue
            for row in read_csv(path):
                run_name = str(row.get("run_name") or run_name_from_path(path, root))
                experiment, seed = experiment_seed(run_name, path, root, row)
                rows.append(
                    {
                        **row,
                        "run_name": run_name,
                        "experiment": experiment,
                        "seed": seed,
                        "source_root": str(root),
                        "source_path": str(path),
                        "oracle_gate": row.get("oracle_gate", "true" if "oracle" in path.name else "false"),
                    }
                )
    return rows


def collect_gate_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            if path.name not in GATE_FILES:
                continue
            run_name = run_name_from_path(path, root)
            experiment, seed = experiment_seed(run_name, path, root, {})
            for row in read_csv(path):
                rows.append({**row, "run_name": run_name, "experiment": experiment, "seed": seed, "source_root": str(root), "source_path": str(path)})
    return rows


def summarize_runs(pattern_rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]], roots: list[Path]) -> list[dict[str, Any]]:
    by_run: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pattern_rows:
        by_run[(str(row.get("source_root")), str(row.get("experiment")), str(row.get("seed")))].append(row)
    gate_summary = summarize_gate_rows(gate_rows)
    rows: list[dict[str, Any]] = []
    for (source_root, experiment, seed), items in sorted(by_run.items()):
        values = {str(row.get("pattern") or row.get("pattern_name")): row for row in items}
        drop1 = mean_patterns(values, missing_count=1)
        drop2 = mean_patterns(values, missing_count=2)
        drop3 = mean_patterns(values, missing_count=3)
        avg_missing = top1(values.get("avg_missing"))
        if not isnum(avg_missing):
            avg_missing = mean_num([drop1, drop2, drop3])
        info = selection_info(Path(source_root), experiment, seed)
        run_router = router_from_pattern_rows(items)
        row = {
            "source_root": source_root,
            "experiment": experiment,
            "seed": seed,
            "oracle_gate": str(any(truthy(item.get("oracle_gate")) for item in items)).lower(),
            "full": top1(values.get("full")),
            "drop1": drop1,
            "drop2": drop2,
            "drop3": drop3,
            "drop1_3_mean": mean_num([drop1, drop2, drop3]),
            "avg_missing": avg_missing,
            "image_only": top1(values.get("image_only")),
            "lidar_only": top1(values.get("lidar_only")),
            "radar_only": top1(values.get("radar_only")),
            "gps_only": top1(values.get("gps_only")),
            "missing_image": top1(values.get("missing_image")),
            "missing_lidar": top1(values.get("missing_lidar")),
            "missing_radar": top1(values.get("missing_radar")),
            "missing_gps": top1(values.get("missing_gps")),
            "within3": metric(values.get("avg_missing"), "within_3", "within3"),
            "MAE": metric(values.get("avg_missing"), "mae", "MAE"),
            **info,
            **gate_summary.get((source_root, experiment, seed), {}),
            **run_router,
        }
        rows.append(row)
    return rows


def summarize_gate_rows(gate_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in gate_rows:
        grouped[(str(row.get("source_root")), str(row.get("experiment")), str(row.get("seed")))].append(row)
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        result: dict[str, Any] = {}
        for modality in ("image", "lidar", "radar", "gps"):
            result[f"mean_gate_{modality}"] = mean_num([float_or_nan(row.get("mean_weight")) for row in rows if str(row.get("modality")) == modality])
        result["gate_entropy"] = mean_num([float_or_nan(row.get("gate_entropy")) for row in rows])
        radar_rows = [row for row in rows if str(row.get("modality")) == "radar"]
        result["radar_gate_missing_image"] = mean_num([float_or_nan(row.get("mean_weight")) for row in radar_rows if str(row.get("pattern")) == "missing_image"])
        result["radar_gate_drop2"] = mean_num([float_or_nan(row.get("mean_weight")) for row in radar_rows if missing_count(str(row.get("pattern")), row.get("mask"), modalities(row)) == 2])
        out[key] = result
    return out


def router_from_pattern_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in ROUTER_FIELDS:
        out[field] = mean_num([float_or_nan(row.get(field)) for row in rows])
    return {key: value for key, value in out.items() if isnum(value)}


def aggregate_experiments(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row.get("experiment"))].append(row)
    rows: list[dict[str, Any]] = []
    for experiment, items in sorted(grouped.items()):
        row: dict[str, Any] = {"experiment": experiment, "n": len(items), "seeds": ",".join(str(item.get("seed")) for item in items)}
        for metric_name in [*METRICS, "drop2", *ROUTER_FIELDS]:
            values = [float_or_nan(item.get(metric_name)) for item in items]
            valid = [value for value in values if isnum(value)]
            if valid:
                row[f"{metric_name}_mean"] = mean(valid)
                row[f"{metric_name}_std"] = pstdev(valid) if len(valid) > 1 else 0.0
        row["hard_subset_weighting"] = first_config_value(items, "loss.hard_subset_weighting.mode")
        row["selection_metric"] = first_nonempty(item.get("selection_metric") for item in items)
        rows.append(row)
    return attach_deltas(rows)


def attach_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors = {
        "e5": find_anchor(rows, ("a1_e5", "e5_pcpg_low_encoder_lr")),
        "e6": find_anchor(rows, ("a2_e6", "e6_pcpg_hard_subset_jepa")),
        "b3": find_anchor(rows, ("b3_hard_soft_no_jepa",)),
        "b4": find_anchor(rows, ("b4_hard_soft_jepa",)),
    }
    out = []
    for row in rows:
        updated = dict(row)
        for anchor_name, anchor in anchors.items():
            for metric_name in DELTA_METRICS:
                updated[f"delta_{metric_name}_vs_{anchor_name}"] = float_or_nan(row.get(f"{metric_name}_mean")) - float_or_nan((anchor or {}).get(f"{metric_name}_mean"))
        out.append(updated)
    return out


def drop_count_summary(pattern_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pattern_rows:
        count = missing_count(str(row.get("pattern") or row.get("pattern_name")), row.get("mask"), modalities(row))
        if count is None:
            continue
        grouped[(str(row.get("source_root")), str(row.get("experiment")), str(row.get("seed")), int(count))].append(row)
    out = []
    for (source_root, experiment, seed, count), rows in sorted(grouped.items()):
        out.append(
            {
                "source_root": source_root,
                "experiment": experiment,
                "seed": seed,
                "missing_count": count,
                "top1": mean_num([top1(row) for row in rows]),
                "within3": mean_num([metric(row, "within_3", "within3") for row in rows]),
                "MAE": mean_num([metric(row, "mae", "MAE") for row in rows]),
                "num_patterns": len(rows),
                "num_samples": sum(int(float(row.get("num_samples", row.get("sample_count", 0)) or 0)) for row in rows),
            }
        )
    return out


def router_diagnostics(run_rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in run_rows:
        payload = {key: row.get(key, "") for key in ("source_root", "experiment", "seed", "oracle_gate", *ROUTER_FIELDS)}
        if any(str(payload.get(field, "")) not in {"", "nan"} for field in ROUTER_FIELDS):
            rows.append(payload)
    for row in gate_rows:
        rows.append(row)
    return rows


def render_markdown(summary_rows: list[dict[str, Any]], run_rows: list[dict[str, Any]], baseline_roots: list[Path]) -> str:
    lines = ["# Overnight Branch Router V2 Summary", ""]
    lines.extend(render_main_table(summary_rows))
    lines.extend(render_delta_table(summary_rows))
    lines.extend(render_e6_breakdown(summary_rows))
    lines.extend(render_router_table(summary_rows))
    lines.extend(render_recommendations(summary_rows, baseline_roots))
    return "\n".join(lines) + "\n"


def render_main_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## 7.1 Mean/Std 主表", "", "| experiment | n | full | drop1 | drop2 | drop3 | avg_missing | radar_only | missing_image | within3 | MAE |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            f"| {row.get('experiment')} | {row.get('n')} | {pm(row, 'full')} | {pm(row, 'drop1')} | {pm(row, 'drop2')} | {pm(row, 'drop3')} | "
            f"{pm(row, 'avg_missing')} | {pm(row, 'radar_only')} | {pm(row, 'missing_image')} | {pm(row, 'within3')} | {pm(row, 'MAE')} |"
        )
    return lines + [""]


def render_delta_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## 7.2 Delta 表", "", "| experiment | vs | delta_full | delta_avg_missing | delta_drop3 | delta_radar_only | delta_missing_image |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        for anchor in ("e5", "e6", "b3", "b4"):
            lines.append(
                f"| {row.get('experiment')} | {anchor} | {fmt(row.get(f'delta_full_vs_{anchor}'))} | "
                f"{fmt(row.get(f'delta_avg_missing_vs_{anchor}'))} | {fmt(row.get(f'delta_drop3_vs_{anchor}'))} | "
                f"{fmt(row.get(f'delta_radar_only_vs_{anchor}'))} | {fmt(row.get(f'delta_missing_image_vs_{anchor}'))} |"
            )
    return lines + [""]


def render_e6_breakdown(rows: list[dict[str, Any]]) -> list[str]:
    by_exp = {str(row.get("experiment")): row for row in rows}
    e6 = find_anchor(rows, ("a2_e6", "e6_pcpg_hard_subset_jepa"))
    b1 = by_exp.get("b1_hard_static_no_jepa", {})
    b2 = by_exp.get("b2_jepa_no_hard", {})
    b4 = by_exp.get("b4_hard_soft_jepa", {})
    b5 = by_exp.get("b5_branch_aux_hard_soft_no_jepa", {})
    b6 = by_exp.get("b6_branch_aux_hard_soft_jepa", {})
    lines = ["## 7.3 e6 来源拆解表", ""]
    lines.append(f"- hard_static_no_jepa vs e6 avg_missing delta: {delta_text(b1, e6, 'avg_missing')}；drop3 delta: {delta_text(b1, e6, 'drop3')}。")
    lines.append(f"- jepa_no_hard vs e6 avg_missing delta: {delta_text(b2, e6, 'avg_missing')}；full delta: {delta_text(b2, e6, 'full')}。")
    lines.append(f"- hard_soft_jepa vs e6 full delta: {delta_text(b4, e6, 'full')}；drop3 delta: {delta_text(b4, e6, 'drop3')}。")
    lines.append(f"- branch_aux_hard_soft_no_jepa radar_only/drop3: {fmt_metric(b5, 'radar_only')} / {fmt_metric(b5, 'drop3')}。")
    lines.append(f"- branch_aux_hard_soft_jepa radar_only/drop3: {fmt_metric(b6, 'radar_only')} / {fmt_metric(b6, 'drop3')}。")
    lines.append(f"- 当前 full/robustness trade-off 最高 avg_missing 候选: {best_experiment(rows, 'avg_missing_mean')}。")
    return lines + [""]


def render_router_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## 7.4 Supervised Router 诊断表", "", "| experiment | missing_image | drop2 | full | router_acc | radar_gate_missing_image | radar_gate_drop2 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        if "supervised_router" not in str(row.get("experiment")):
            continue
        lines.append(
            f"| {row.get('experiment')} | {pm(row, 'missing_image')} | {pm(row, 'drop2')} | {pm(row, 'full')} | "
            f"{pm(row, 'router_oracle_acc')} | {pm(row, 'radar_gate_missing_image')} | {pm(row, 'radar_gate_drop2')} |"
        )
    lines.append("")
    lines.append("如果 router 只提升 missing_image/drop2 而不提升 drop3，这是预期现象：drop3/radar-only 主要是单模态 branch weakness，不是 routing failure。")
    return lines + [""]


def render_recommendations(rows: list[dict[str, Any]], baseline_roots: list[Path]) -> list[str]:
    return [
        "## 7.5 推荐结论",
        "",
        f"- 综合主方法候选: {best_experiment(rows, 'avg_missing_mean')}。",
        f"- robustness-first 候选: {best_experiment(rows, 'drop3_mean')}。",
        f"- 值得扩大多 seed 的候选: {best_experiment(rows, 'drop1_3_mean_mean')}。",
        f"- supervised router 去留: {'等待完整 run' if not any('supervised_router' in str(row.get('experiment')) for row in rows) else '根据上表 missing_image/drop2 与 full 损失决定'}。",
        f"- BPRR baseline roots: {', '.join(str(path) for path in baseline_roots) or '未提供'}；若仍弱于 e5/e6/soft-hard，应作为 negative result/ablation 保留。",
    ]


def selection_info(root: Path, experiment: str, seed: str) -> dict[str, Any]:
    for run_dir in candidate_run_dirs(root, experiment, seed):
        cfg = run_dir / "final_config.yaml"
        out: dict[str, Any] = {"config_path": str(cfg) if cfg.exists() else ""}
        if cfg.exists():
            payload = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            hard = payload.get("loss", {}).get("hard_subset_weighting", {})
            out["hard_subset_weighting"] = hard.get("mode", "") if isinstance(hard, dict) else str(hard)
        for path in (
            run_dir / "checkpoints" / "best_avg_missing_top1.pth.json",
            run_dir / "checkpoints" / "best_top1.pth.json",
            run_dir / "checkpoints" / "best.pth.json",
        ):
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                out.update(
                    {
                        "selection_metric": payload.get("selection_metric", ""),
                        "best_epoch": payload.get("selected_epoch", payload.get("epoch", "")),
                        "clean_val_acc": (payload.get("task_metrics") or {}).get("val_acc", ""),
                    }
                )
                return out
        metrics = run_dir / "metrics.json"
        if metrics.exists():
            payload = json.loads(metrics.read_text(encoding="utf-8"))
            latest = payload.get("latest", {}) if isinstance(payload, dict) else {}
            out.update(
                {
                    "selection_metric": latest.get("checkpoint_selection_metric", ""),
                    "best_epoch": latest.get("best_early_stopping_epoch", ""),
                    "clean_val_acc": latest.get("val_acc", ""),
                }
            )
            return out
    return {}


def candidate_run_dirs(root: Path, experiment: str, seed: str) -> list[Path]:
    return [
        root / experiment / f"seed{seed}",
        root / f"{experiment}_seed{seed}",
        root / experiment / str(seed),
    ]


def mean_patterns(values: dict[str, dict[str, Any]], *, missing_count: int) -> float:
    return mean_num([top1(row) for name, row in values.items() if missing_count_fn(name, row.get("mask"), modalities(row)) == missing_count])


def missing_count(pattern: str, mask: Any, names: list[str]) -> int | None:
    return missing_count_fn(pattern, mask, names)


def missing_count_fn(pattern: str, mask: Any, names: list[str]) -> int | None:
    if isinstance(mask, str) and mask and mask not in {"aggregate"} and not mask.startswith("random_"):
        values = [item.strip() for item in mask.split(",") if item.strip()]
        if values and all(item in {"0", "1"} for item in values):
            return values.count("0")
    name = str(pattern)
    if name == "full":
        return 0
    if name in {"avg_missing"} or name.startswith("random_"):
        return None
    if name in {"miss1", "drop1"}:
        return 1
    if name in {"miss2", "drop2"}:
        return 2
    if name in {"miss3", "drop3"}:
        return 3
    if name.endswith("_only"):
        return max(len(names), 4) - 1
    if name == "non_gps_only":
        return 1
    if name.startswith("missing_"):
        return len([item for item in name.removeprefix("missing_").split("_") if item])
    return None


def modalities(row: dict[str, Any]) -> list[str]:
    raw = row.get("modalities")
    if isinstance(raw, str) and raw:
        return [item for item in raw.replace(",", "|").split("|") if item]
    return ["image", "radar", "lidar", "gps"]


def top1(row: dict[str, Any] | None) -> float:
    return metric(row, "top1", "full_top1", "avg_missing_top1")


def metric(row: dict[str, Any] | None, *keys: str) -> float:
    if row is None:
        return math.nan
    for key in keys:
        value = float_or_nan(row.get(key))
        if isnum(value):
            return value
    return math.nan


def run_name_from_path(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    if "eval" in parts and path.name.endswith("_missing_patterns.csv"):
        return path.stem.removesuffix("_missing_patterns")
    if len(parts) >= 2 and parts[1].startswith("seed"):
        return f"{parts[0]}/{parts[1]}"
    return parts[0] if parts else path.parent.name


def experiment_seed(run_name: str, path: Path, root: Path, row: dict[str, Any]) -> tuple[str, str]:
    if row.get("experiment") and row.get("seed"):
        return str(row["experiment"]), str(row["seed"])
    name = str(run_name)
    if "/" in name:
        exp, seed_part = name.split("/", 1)
        return exp, seed_part.removeprefix("seed")
    if "_seed" in name:
        exp, seed = name.rsplit("_seed", 1)
        return exp, "".join(ch for ch in seed if ch.isdigit()) or str(row.get("seed", ""))
    parts = path.relative_to(root).parts
    if len(parts) >= 2 and parts[1].startswith("seed"):
        return parts[0], parts[1].removeprefix("seed")
    return name, str(row.get("seed", ""))


def find_anchor(rows: list[dict[str, Any]], prefixes: tuple[str, ...]) -> dict[str, Any] | None:
    for row in rows:
        experiment = str(row.get("experiment"))
        if any(experiment.startswith(prefix) for prefix in prefixes):
            return row
    return None


def best_experiment(rows: list[dict[str, Any]], field: str) -> str:
    valid = [row for row in rows if isnum(float_or_nan(row.get(field)))]
    if not valid:
        return "unavailable"
    return str(max(valid, key=lambda row: float_or_nan(row.get(field))).get("experiment"))


def first_config_value(items: list[dict[str, Any]], dotted: str) -> str:
    for item in items:
        path = item.get("config_path")
        if not path:
            continue
        try:
            payload = yaml.safe_load(Path(str(path)).read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        value: Any = payload
        for key in dotted.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if value not in (None, ""):
            return str(value)
    return ""


def first_nonempty(values) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return ""


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def float_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def isnum(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean_num(values: list[float]) -> float:
    valid = [float(value) for value in values if isnum(value)]
    return mean(valid) if valid else math.nan


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def fmt(value: Any) -> str:
    number = float_or_nan(value)
    return "" if not isnum(number) else f"{number:.4f}"


def pm(row: dict[str, Any], metric_name: str) -> str:
    avg = float_or_nan(row.get(f"{metric_name}_mean"))
    std = float_or_nan(row.get(f"{metric_name}_std"))
    return "" if not isnum(avg) else f"{avg:.4f}±{std:.4f}" if isnum(std) else f"{avg:.4f}"


def fmt_metric(row: dict[str, Any] | None, metric_name: str) -> str:
    return "" if not row else pm(row, metric_name)


def delta_text(row: dict[str, Any] | None, anchor: dict[str, Any] | None, metric_name: str) -> str:
    if not row or not anchor:
        return "unavailable"
    return fmt(float_or_nan(row.get(f"{metric_name}_mean")) - float_or_nan(anchor.get(f"{metric_name}_mean")))


if __name__ == "__main__":
    raise SystemExit(main())
