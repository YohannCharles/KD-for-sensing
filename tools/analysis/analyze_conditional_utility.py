#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.diagnostics.conditional_utility import read_table  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Conditional Utility Audit figures.")
    parser.add_argument("--input", required=True, help="conditional_utility output directory.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = build_parser().parse_args(argv)
    output = analyze_conditional_utility(Path(args.input))
    print(json.dumps(output, indent=2))
    return output


def analyze_conditional_utility(input_dir: str | Path) -> dict[str, str]:
    audit_dir = Path(input_dir)
    figures_dir = audit_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_json(audit_dir / "conditional_utility_summary.json")
    subset = read_table(audit_dir, "subset_predictions")
    delta = read_table(audit_dir, "conditional_utility_per_sample_delta")
    bucket = _read_csv_if_exists(audit_dir / "conditional_utility_by_bucket.csv")
    teacher = _read_optional_table(audit_dir, "teacher_predictions")

    outputs = {
        "subset_metrics_bar": str(_plot_subset_metrics(summary, figures_dir / "subset_metrics_bar.png")),
        "marginal_delta_by_horizon": str(
            _plot_marginal_delta_by_horizon(delta, figures_dir / "marginal_delta_by_horizon.png")
        ),
        "oracle_choice_distribution": str(
            _plot_oracle_choice_distribution(summary, figures_dir / "oracle_choice_distribution.png")
        ),
        "teacher_rescue_rate": str(
            _plot_teacher_rescue_rate(summary, teacher, figures_dir / "teacher_rescue_rate.png")
        ),
        "bucket_heatmap_delta_dba": str(
            _plot_bucket_heatmap(bucket, figures_dir / "bucket_heatmap_delta_dba.png")
        ),
    }
    for modality in sorted(delta["weak_modality"].dropna().unique().tolist()) if not delta.empty else []:
        path = figures_dir / f"delta_ce_histogram_{modality}.png"
        outputs[f"delta_ce_histogram_{modality}"] = str(_plot_delta_ce_histogram(delta, modality, path))
    return outputs


def _plot_subset_metrics(summary: dict[str, Any], path: Path) -> Path:
    metrics = summary.get("aggregate_metrics") or {}
    rows = []
    for subset_name, values in metrics.items():
        topk = values.get("topk") or {}
        rows.append(
            {
                "subset": subset_name,
                "top1": _mean(topk.get("1")),
                "top3": _mean(topk.get("3")),
                "top5": _mean(topk.get("5")),
                "dba": _mean(values.get("dba")),
            }
        )
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if not frame.empty:
        x = np.arange(len(frame))
        width = 0.2
        for offset, metric in enumerate(("top1", "top3", "top5", "dba")):
            ax.bar(x + (offset - 1.5) * width, frame[metric], width=width, label=metric)
        ax.set_xticks(x)
        ax.set_xticklabels(frame["subset"], rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, ncols=4)
    ax.set_ylabel("score")
    ax.set_title("Subset metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_marginal_delta_by_horizon(delta: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if not delta.empty:
        pivot = delta.pivot_table(
            index="horizon_name",
            columns="weak_modality",
            values="delta_dba",
            aggfunc="mean",
        ).sort_index()
        colors = ["#2f7d32" if value >= 0 else "#b23b3b" for value in pivot.to_numpy().reshape(-1)]
        flat = pivot.stack()
        labels = [f"{h}\n{m}" for h, m in flat.index]
        ax.bar(np.arange(len(flat)), flat.values, color=colors)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(np.arange(len(flat)))
        ax.set_xticklabels(labels)
    ax.set_ylabel("delta DBA")
    ax.set_title("Marginal delta by horizon")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_oracle_choice_distribution(summary: dict[str, Any], path: Path) -> Path:
    choices = summary.get("oracle_subset", {}).get("oracle_choice_distribution", {}) or {}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if choices:
        names = list(choices)
        values = [choices[name] for name in names]
        x = np.arange(len(names))
        ax.bar(x, values, color="#4c78a8")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylim(0, max(1.0, max(values) * 1.1))
    ax.set_ylabel("choice rate")
    ax.set_title("Oracle subset choice")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_teacher_rescue_rate(summary: dict[str, Any], teacher: pd.DataFrame, path: Path) -> Path:
    complementarity = summary.get("teacher_complementarity") or {}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if complementarity:
        names = list(complementarity)
        values = [
            float(complementarity[name].get("rescue_rate_given_strong_top1_wrong", 0.0) or 0.0)
            for name in names
        ]
        ax.bar(names, values, color="#7a5195")
        ax.set_ylim(0, max(1.0, max(values) * 1.1 if values else 1.0))
    elif not teacher.empty and "teacher_modality" in teacher.columns:
        counts = teacher["teacher_modality"].value_counts()
        ax.bar(counts.index.tolist(), counts.values, color="#7a5195")
    ax.set_ylabel("rescue rate")
    ax.set_title("Teacher rescue")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_delta_ce_histogram(delta: pd.DataFrame, modality: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    values = delta.loc[delta["weak_modality"] == modality, "delta_ce"].dropna()
    if not values.empty:
        ax.hist(values, bins=40, color="#4c78a8", alpha=0.85)
        ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("delta CE")
    ax.set_ylabel("samples")
    ax.set_title(f"Delta CE: {modality}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_bucket_heatmap(bucket: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    if not bucket.empty:
        bucket = bucket.copy()
        bucket["row"] = bucket["bucket_feature"].astype(str) + ":" + bucket["bucket_name"].astype(str)
        pivot = bucket.pivot_table(
            index="row",
            columns="weak_modality",
            values="delta_dba",
            aggfunc="mean",
        ).fillna(0.0)
        data = pivot.to_numpy()
        vmax = max(float(np.abs(data).max()), 1e-6)
        image = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        fig.colorbar(image, ax=ax, label="delta DBA")
    ax.set_title("Bucket delta DBA")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_optional_table(input_dir: Path, stem: str) -> pd.DataFrame:
    try:
        return read_table(input_dir, stem)
    except FileNotFoundError:
        return pd.DataFrame()


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _mean(values: Any) -> float:
    if values is None:
        return 0.0
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.nanmean(array))


if __name__ == "__main__":
    main()
