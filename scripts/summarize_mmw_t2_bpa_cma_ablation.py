#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_mmw_task_output_robustness as task_output


METHODS = ("T2", "T2-NoBPA", "T2-BPA2CMA", "T2-Linear", "T2-CLS", "T2-CLS-CMA")
SEEDS = (1, 2, 3)
METRICS = ("top1", "within1", "within3", "circular_mae")
SLICES = ("all", "exact_endpoint", "near_endpoint", "interior")
SLICE_LABELS = {
    "all": "All beams",
    "exact_endpoint": "Endpoints {0,63}",
    "near_endpoint": "Near endpoints {62,63,0,1}",
    "interior": "Interior {2,...,61}",
}
METHOD_LABELS = {
    "T2": "T2",
    "T2-NoBPA": "T2 w/o BPA",
    "T2-BPA2CMA": "T2: BPA to CMA",
    "T2-Linear": "T2: linear BPA",
    "T2-CLS": "T2: classifier",
    "T2-CLS-CMA": "T2: classifier + CMA",
}
COMPARISONS = (
    ("bpa_auxiliary_removal", "T2-NoBPA", "T2"),
    ("bpa_to_cma", "T2-BPA2CMA", "T2-NoBPA"),
    ("circular_vs_linear", "T2", "T2-Linear"),
    ("cma_without_prototypes", "T2-CLS-CMA", "T2-CLS"),
    ("prototype_package", "T2", "T2-CLS"),
)
COLORS = {
    "T2": "#222222",
    "T2-NoBPA": "#D55E00",
    "T2-BPA2CMA": "#0072B2",
    "T2-Linear": "#CC79A7",
    "T2-CLS": "#6B7280",
    "T2-CLS-CMA": "#009E73",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize the paired MMW T2 BPA/CMA ablation.")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--expected-domains", type=int, default=15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summarize_ablation(
        Path(args.raw_root),
        Path(args.output_dir),
        seeds=tuple(int(item) for item in _csv(args.seeds)),
        expected_domains=int(args.expected_domains),
    )
    return 0


def summarize_ablation(
    raw_root: Path,
    output_dir: Path,
    *,
    seeds: tuple[int, ...] = SEEDS,
    expected_domains: int = 15,
) -> dict[str, list[dict[str, Any]]]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    bundles = task_output.load_bundles(
        raw_root,
        methods=METHODS,
        seeds=seeds,
        expected_domains=expected_domains,
    )
    seed_rates = build_seed_rate_slice_metrics(bundles)
    multiseed_rates = aggregate_multiseed(seed_rates, ("method", "slice", "rate"), METRICS, len(seeds))
    seed_auc = build_missing_auc(seed_rates)
    multiseed_auc = aggregate_multiseed(seed_auc, ("method", "slice"), METRICS, len(seeds), suffix="_auc")
    seed_rate_deltas = build_paired_deltas(seed_rates, value_suffix="")
    multiseed_rate_deltas = aggregate_multiseed(
        seed_rate_deltas,
        ("comparison", "left_method", "right_method", "slice", "rate"),
        tuple(f"{metric}_delta" for metric in METRICS),
        len(seeds),
    )
    seed_auc_deltas = build_paired_deltas(seed_auc, value_suffix="_auc")
    multiseed_auc_deltas = aggregate_multiseed(
        seed_auc_deltas,
        ("comparison", "left_method", "right_method", "slice"),
        tuple(f"{metric}_auc_delta" for metric in METRICS),
        len(seeds),
    )
    outputs = {
        "per_seed_rate_slice_metrics.csv": seed_rates,
        "multiseed_rate_slice_metrics.csv": multiseed_rates,
        "per_seed_missing_auc.csv": seed_auc,
        "multiseed_missing_auc.csv": multiseed_auc,
        "per_seed_paired_rate_deltas.csv": seed_rate_deltas,
        "multiseed_paired_rate_deltas.csv": multiseed_rate_deltas,
        "per_seed_paired_auc_deltas.csv": seed_auc_deltas,
        "multiseed_paired_auc_deltas.csv": multiseed_auc_deltas,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        _write_csv(output_dir / name, rows)
    plot_objective_curves(multiseed_rates, output_dir / "objective_ablation_curves.png")
    plot_topology_endpoint_deltas(multiseed_rate_deltas, output_dir / "topology_endpoint_deltas.png")
    plot_classifier_cma_curves(multiseed_rates, output_dir / "classifier_cma_curves.png")
    payload = {
        "protocol": {
            "methods": list(METHODS),
            "seeds": list(seeds),
            "expected_domains": expected_domains,
            "slices": {
                "all": "all labels",
                "exact_endpoint": [0, 63],
                "near_endpoint": [62, 63, 0, 1],
                "interior": "labels 2 through 61",
            },
            "missing_auc": "normalized trapezoidal integral over non-zero evaluated missing rates",
            "paired_delta": "left_method minus right_method; lower circular_mae is better",
            "cma_scope": "AMBER-style objective analogue on T2 pooled features, not full AMBER Class-Former",
        },
        **{name.removesuffix(".csv"): rows for name, rows in outputs.items()},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        render_markdown(multiseed_rates, multiseed_auc, multiseed_rate_deltas, multiseed_auc_deltas, seeds),
        encoding="utf-8",
    )
    return {
        "seed_rate": seed_rates,
        "multiseed_rate": multiseed_rates,
        "seed_auc": seed_auc,
        "multiseed_auc": multiseed_auc,
        "seed_rate_deltas": seed_rate_deltas,
        "multiseed_rate_deltas": multiseed_rate_deltas,
        "seed_auc_deltas": seed_auc_deltas,
        "multiseed_auc_deltas": multiseed_auc_deltas,
    }


def build_seed_rate_slice_metrics(
    bundles: dict[int, dict[str, dict[str, dict[str, Any]]]],
) -> list[dict[str, Any]]:
    domain_rows: list[dict[str, Any]] = []
    for seed, method_bundles in sorted(bundles.items()):
        for method in METHODS:
            domains = method_bundles[method]
            for domain_id, domain in sorted(domains.items()):
                labels = np.asarray(domain["labels"], dtype=np.int64)
                predictions = np.asarray(domain["predictions"], dtype=np.int64)
                rates = np.asarray(domain["rates"], dtype=np.float64)
                for rate in sorted(set(rates.tolist())):
                    condition_indices = np.flatnonzero(np.isclose(rates, rate, atol=1e-6))
                    for slice_name in SLICES:
                        subset = beam_slice(labels, slice_name)
                        metrics = [_metrics(predictions[index][subset], labels[subset]) for index in condition_indices]
                        domain_rows.append(
                            {
                                "seed": seed,
                                "method": method,
                                "slice": slice_name,
                                "rate": float(rate),
                                "domain_id": domain_id,
                                "mask_count": int(condition_indices.size),
                                "sample_count": int(subset.sum()),
                                "status": "available" if subset.any() else "empty_slice",
                                **{
                                    metric: _finite_mean(item[metric] for item in metrics)
                                    for metric in METRICS
                                },
                            }
                        )
    rows: list[dict[str, Any]] = []
    keys = sorted({(row["seed"], row["method"], row["slice"], row["rate"]) for row in domain_rows})
    for seed, method, slice_name, rate in keys:
        selected = [
            row
            for row in domain_rows
            if (row["seed"], row["method"], row["slice"], row["rate"]) == (seed, method, slice_name, rate)
        ]
        eligible = [row for row in selected if row["status"] == "available"]
        mask_counts = {int(row["mask_count"]) for row in selected}
        if len(mask_counts) != 1:
            raise ValueError(f"mask count differs across domains for seed={seed} method={method} rate={rate}")
        rows.append(
            {
                "seed": seed,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "slice": slice_name,
                "slice_label": SLICE_LABELS[slice_name],
                "rate": float(rate),
                "mask_count_per_domain": next(iter(mask_counts)),
                "domain_count": len(selected),
                "eligible_domain_count": len(eligible),
                "sample_count": sum(int(row["sample_count"]) for row in eligible),
                "status": "available" if eligible else "unavailable_empty_slice",
                **{metric: _finite_mean(row[metric] for row in eligible) for metric in METRICS},
            }
        )
    return rows


def beam_slice(labels: np.ndarray, slice_name: str) -> np.ndarray:
    target = np.asarray(labels, dtype=np.int64)
    if slice_name == "all":
        return np.ones(target.shape, dtype=bool)
    if slice_name == "exact_endpoint":
        return np.isin(target, (0, 63))
    if slice_name == "near_endpoint":
        return np.isin(target, (62, 63, 0, 1))
    if slice_name == "interior":
        return ~np.isin(target, (62, 63, 0, 1))
    raise ValueError(f"Unknown beam slice: {slice_name}")


def build_missing_auc(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    keys = sorted({(int(row["seed"]), str(row["method"]), str(row["slice"])) for row in seed_rows})
    for seed, method, slice_name in keys:
        selected = sorted(
            (
                row
                for row in seed_rows
                if int(row["seed"]) == seed
                and row["method"] == method
                and row["slice"] == slice_name
                and float(row["rate"]) > 0
            ),
            key=lambda row: float(row["rate"]),
        )
        rates = [float(row["rate"]) for row in selected]
        available = len(rates) >= 2 and all(row["status"] == "available" for row in selected)
        rows.append(
            {
                "seed": seed,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "slice": slice_name,
                "slice_label": SLICE_LABELS[slice_name],
                "rate_min": min(rates) if rates else math.nan,
                "rate_max": max(rates) if rates else math.nan,
                "rate_count": len(rates),
                "status": "available" if available else "unavailable_incomplete_rates",
                **{
                    f"{metric}_auc": _normalized_auc(rates, [float(row[metric]) for row in selected])
                    if available
                    else math.nan
                    for metric in METRICS
                },
            }
        )
    return rows


def build_paired_deltas(rows: list[dict[str, Any]], *, value_suffix: str) -> list[dict[str, Any]]:
    value_fields = tuple(f"{metric}{value_suffix}" for metric in METRICS)
    identity_fields = ("seed", "slice", *(("rate",) if not value_suffix else ()))
    lookup = {tuple(row[field] for field in identity_fields) + (row["method"],): row for row in rows}
    result = []
    identities = sorted({tuple(row[field] for field in identity_fields) for row in rows})
    for comparison, left_method, right_method in COMPARISONS:
        for identity in identities:
            left = lookup.get(identity + (left_method,))
            right = lookup.get(identity + (right_method,))
            if left is None or right is None:
                raise ValueError(f"Missing paired row for {comparison} identity={identity}")
            available = left["status"] == "available" and right["status"] == "available"
            item = {
                **dict(zip(identity_fields, identity)),
                "comparison": comparison,
                "left_method": left_method,
                "left_label": METHOD_LABELS[left_method],
                "right_method": right_method,
                "right_label": METHOD_LABELS[right_method],
                "delta_definition": "left_minus_right",
                "status": "available" if available else "unavailable",
            }
            item.update(
                {
                    f"{field}_delta": float(left[field]) - float(right[field]) if available else math.nan
                    for field in value_fields
                }
            )
            result.append(item)
    return result


def aggregate_multiseed(
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
    value_fields: tuple[str, ...],
    requested_seed_count: int,
    *,
    suffix: str = "",
) -> list[dict[str, Any]]:
    result = []
    keys = sorted({tuple(row[field] for field in key_fields) for row in rows})
    for key in keys:
        selected = [row for row in rows if tuple(row[field] for field in key_fields) == key]
        seeds = {int(row["seed"]) for row in selected}
        complete = len(seeds) == requested_seed_count and all(row["status"] == "available" for row in selected)
        item = {
            **dict(zip(key_fields, key)),
            "seed_count": len(seeds),
            "requested_seed_count": requested_seed_count,
            "status": "complete" if complete else "partial",
        }
        first = selected[0]
        for field in ("method_label", "slice_label", "left_label", "right_label", "delta_definition"):
            if field in first:
                item[field] = first[field]
        for field in (
            "mask_count_per_domain",
            "domain_count",
            "eligible_domain_count",
            "sample_count",
            "rate_min",
            "rate_max",
            "rate_count",
        ):
            if field in first:
                values = {row[field] for row in selected}
                if len(values) != 1:
                    raise ValueError(f"support field {field} differs across paired seeds for key={key}")
                item[field] = first[field]
        for field in value_fields:
            source = f"{field}{suffix}" if suffix and not field.endswith(suffix) else field
            values = np.asarray([float(row[source]) for row in selected if complete], dtype=np.float64)
            item[f"{source}_mean"] = float(values.mean()) if values.size else math.nan
            item[f"{source}_std"] = float(values.std(ddof=1)) if values.size > 1 else math.nan
        result.append(item)
    return result


def plot_objective_curves(rows: list[dict[str, Any]], path: Path) -> None:
    _plot_method_curves(
        rows,
        path,
        methods=("T2", "T2-NoBPA", "T2-BPA2CMA"),
        title="BPA auxiliary objective ablation",
    )


def plot_classifier_cma_curves(rows: list[dict[str, Any]], path: Path) -> None:
    _plot_method_curves(
        rows,
        path,
        methods=("T2-CLS", "T2-CLS-CMA"),
        title="CMA effect without the prototype package",
    )


def _plot_method_curves(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    methods: tuple[str, ...],
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharex=True, layout="constrained")
    for axis, metric, label in zip(axes, ("top1", "within3"), ("Top-1 (%)", "Within-3 (%)")):
        for method in methods:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["method"] == method and row["slice"] == "all" and row["status"] == "complete"
                ),
                key=lambda row: float(row["rate"]),
            )
            x = np.asarray([100.0 * float(row["rate"]) for row in selected])
            y = np.asarray([100.0 * float(row[f"{metric}_mean"]) for row in selected])
            std = np.nan_to_num([100.0 * float(row[f"{metric}_std"]) for row in selected], nan=0.0)
            axis.plot(x, y, marker="o", linewidth=2, color=COLORS[method], label=METHOD_LABELS[method])
            axis.fill_between(x, y - std, y + std, color=COLORS[method], alpha=0.14)
        axis.set_xlabel("Missing rate (%)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    _save_figure(fig, path)


def plot_topology_endpoint_deltas(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharex=True, layout="constrained")
    selected_rows = [row for row in rows if row["comparison"] == "circular_vs_linear" and row["status"] == "complete"]
    styles = ("-", "--", "-.", ":")
    for slice_name, style in zip(SLICES, styles):
        selected = sorted(
            (row for row in selected_rows if row["slice"] == slice_name),
            key=lambda row: float(row["rate"]),
        )
        x = np.asarray([100.0 * float(row["rate"]) for row in selected])
        top1 = np.asarray([100.0 * float(row["top1_delta_mean"]) for row in selected])
        mae = np.asarray([float(row["circular_mae_delta_mean"]) for row in selected])
        top1_std = np.nan_to_num([100.0 * float(row["top1_delta_std"]) for row in selected], nan=0.0)
        mae_std = np.nan_to_num([float(row["circular_mae_delta_std"]) for row in selected], nan=0.0)
        top1_line = axes[0].plot(
            x,
            top1,
            marker="o",
            linestyle=style,
            linewidth=1.8,
            label=SLICE_LABELS[slice_name],
        )[0]
        mae_line = axes[1].plot(
            x,
            mae,
            marker="o",
            linestyle=style,
            linewidth=1.8,
            label=SLICE_LABELS[slice_name],
        )[0]
        axes[0].fill_between(x, top1 - top1_std, top1 + top1_std, color=top1_line.get_color(), alpha=0.10)
        axes[1].fill_between(x, mae - mae_std, mae + mae_std, color=mae_line.get_color(), alpha=0.10)
    axes[0].set_ylabel("Top-1 delta: T2 - Linear (pp)")
    axes[1].set_ylabel("Circular MAE delta: T2 - Linear")
    for axis in axes:
        axis.axhline(0.0, color="#555555", linewidth=1)
        axis.set_xlabel("Missing rate (%)")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Circular BPA target: endpoint-localized paired effect")
    _save_figure(fig, path)


def render_markdown(
    rate_rows: list[dict[str, Any]],
    auc_rows: list[dict[str, Any]],
    rate_deltas: list[dict[str, Any]],
    auc_deltas: list[dict[str, Any]],
    seeds: tuple[int, ...],
) -> str:
    lines = [
        "# MMW T2 BPA/CMA 配对消融",
        "",
        "> CMA 行是 T2 池化特征上的 AMBER-style objective analogue，不是完整 AMBER Class-Former 复现。",
        "",
        "## 协议",
        "",
        f"- 方法固定为 `{', '.join(METHODS)}`；seed 固定为 `{', '.join(str(seed) for seed in seeds)}`。",
        "- 切片预先固定：全部 beam、精确端点 `{0,63}`、近端点 `{62,63,0,1}`、内部 `{2,...,61}`。",
        "- 指标按 sample、同缺失率固定 mask 等权、domain 等权、seed 均值/样本标准差汇总；不筛选样本、domain 或 seed。",
        "- Missing AUC 是所有非零已评估缺失率上的归一化梯形积分。paired delta 均为左方法减右方法；Top-1/Within 越大越好，circular MAE 越小越好。",
        "",
        "## 全样本缺失率曲线",
        "",
        "| 方法 | 缺失率 | Top-1 | Within-1 | Within-3 | Circular MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rate_rows:
        if row["slice"] != "all":
            continue
        lines.append(
            f"| {row['method_label']} | {100 * float(row['rate']):.0f}% | "
            f"{_mean_std(row, 'top1', percent=True)} | {_mean_std(row, 'within1', percent=True)} | "
            f"{_mean_std(row, 'within3', percent=True)} | {_mean_std(row, 'circular_mae')} |"
        )
    lines.extend(
        [
            "",
            "## Missing AUC",
            "",
            "| 方法 | Beam 切片 | Top-1 AUC | Within-1 AUC | Within-3 AUC | Circular MAE AUC |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in auc_rows:
        lines.append(
            f"| {row['method_label']} | {row['slice_label']} | "
            f"{_mean_std(row, 'top1_auc')} | {_mean_std(row, 'within1_auc')} | "
            f"{_mean_std(row, 'within3_auc')} | {_mean_std(row, 'circular_mae_auc')} |"
        )
    lines.extend(
        [
            "",
            "## 预注册配对差值",
            "",
            "| 比较（左-右） | 切片 | 缺失率 | Top-1 delta (pp) | Within-3 delta (pp) | Circular MAE delta |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rate_deltas:
        if not math.isclose(float(row["rate"]), 0.8, abs_tol=1e-6):
            continue
        lines.append(
            f"| {row['left_label']} - {row['right_label']} | {SLICE_LABELS[row['slice']]} | 80% | "
            f"{_mean_std(row, 'top1_delta', scale=100)} | {_mean_std(row, 'within3_delta', scale=100)} | "
            f"{_mean_std(row, 'circular_mae_delta')} |"
        )
    lines.extend(
        [
            "",
            "### Missing AUC 配对差值",
            "",
            "| 比较（左-右） | 切片 | Top-1 AUC delta | Within-3 AUC delta | Circular MAE AUC delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in auc_deltas:
        lines.append(
            f"| {row['left_label']} - {row['right_label']} | {SLICE_LABELS[row['slice']]} | "
            f"{_mean_std(row, 'top1_auc_delta')} | {_mean_std(row, 'within3_auc_delta')} | "
            f"{_mean_std(row, 'circular_mae_auc_delta')} |"
        )
    lines.extend(
        [
            "",
            "## 图文件",
            "",
            "- `objective_ablation_curves.{png,pdf}`：T2、去 BPA、BPA 换 CMA 的纯辅助目标对照。",
            "- `topology_endpoint_deltas.{png,pdf}`：T2 减 Linear 在端点/内部切片上的配对差值。",
            "- `classifier_cma_curves.{png,pdf}`：无 prototype package 时 CMA 的净效应。",
            "",
            "若 paired delta 的 seed 波动跨过零或收益只存在于端点切片，论文必须按局部/不稳定结果表述。",
            "",
        ]
    )
    return "\n".join(lines)


def _metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if labels.size == 0:
        return {metric: math.nan for metric in METRICS}
    distance = task_output.circular_beam_distance(predictions, labels)
    return {
        "top1": float(np.mean(predictions == labels)),
        "within1": float(np.mean(distance <= 1)),
        "within3": float(np.mean(distance <= 3)),
        "circular_mae": float(np.mean(distance)),
    }


def _normalized_auc(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(x) != len(y) or not all(math.isfinite(value) for value in (*x, *y)):
        return math.nan
    width = x[-1] - x[0]
    if width <= 0:
        return math.nan
    area = sum((x[index + 1] - x[index]) * (y[index + 1] + y[index]) / 2 for index in range(len(x) - 1))
    return float(area / width)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _finite_mean(values: Any) -> float:
    selected = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(selected)) if selected else math.nan


def _mean_std(row: dict[str, Any], metric: str, *, percent: bool = False, scale: float = 1.0) -> str:
    factor = 100.0 if percent else scale
    mean = factor * float(row[f"{metric}_mean"])
    std = factor * float(row[f"{metric}_std"])
    if not math.isfinite(mean):
        return "NA"
    return f"{mean:.2f} +/- {std:.2f}" if math.isfinite(std) else f"{mean:.2f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    return value


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
