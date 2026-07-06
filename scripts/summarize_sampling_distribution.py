#!/usr/bin/env python3

import argparse
import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from scripts.scene31_34_final_analysis_common import MODALITIES, write_csv
except ModuleNotFoundError:
    from scene31_34_final_analysis_common import MODALITIES, write_csv


BY_COUNT_FIELDS = ["method", "distribution_source", "missing_count", "missing_ratio", "probability"]
BY_SUBSET_FIELDS = ["method", "distribution_source", "available_modalities", "missing_modalities", "missing_count", "probability"]
METHODS = {
    "Natural": "scenes31_34_proto_natural_es40",
    "Uniform pattern exposure": "scenes31_34_proto_sampler_uniform_es40",
    "Bernoulli randomdrop k075": "scenes31_34_proto_randomdrop_bernoulli_k075_es40",
    "Random non-empty subset exposure": "scenes31_34_proto_randomdrop_subset_es40",
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    roots = [Path(args.root), *[Path(item) for item in args.old_root]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    subset_rows: list[dict[str, Any]] = []
    subset_rows.extend(_natural_rows())
    subset_rows.extend(_uniform_rows())
    subset_rows.extend(_empirical_or_bernoulli(roots, "Bernoulli randomdrop k075", METHODS["Bernoulli randomdrop k075"]))
    subset_rows.extend(_empirical_or_subset(roots, "Random non-empty subset exposure", METHODS["Random non-empty subset exposure"]))
    count_rows = _count_rows(subset_rows)
    write_csv(out / "sampling_distribution_by_subset.csv", subset_rows, BY_SUBSET_FIELDS)
    write_csv(out / "sampling_distribution_by_missing_count.csv", count_rows, BY_COUNT_FIELDS)
    _write_summary(out / "sampling_distribution_summary.md", subset_rows)
    _plot_count(count_rows, out / "fig_sampling_distribution_missing_count")
    _plot_subset(subset_rows, out / "fig_sampling_distribution_subset")
    print(f"Wrote Scene31-34 sampling analysis to {out}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Scene31-34 training sampling distributions.")
    parser.add_argument("--root", default="outputs/scenes31_34_main_lmdb")
    parser.add_argument("--old-root", action="append", default=[])
    parser.add_argument("--out", default="outputs/scenes31_34_main_lmdb/sampling_analysis")
    return parser


def _natural_rows() -> list[dict[str, Any]]:
    return [_row("Natural", "theoretical_from_config", MODALITIES, 1.0)]


def _uniform_rows() -> list[dict[str, Any]]:
    rows = [_row("Uniform pattern exposure", "theoretical_from_config", MODALITIES, 1 / 5)]
    for missing in MODALITIES:
        available = tuple(item for item in MODALITIES if item != missing)
        rows.append(_row("Uniform pattern exposure", "theoretical_from_config", available, 1 / 5))
    return rows


def _empirical_or_bernoulli(roots: list[Path], display: str, method: str) -> list[dict[str, Any]]:
    empirical = _empirical_rows(roots, display, method)
    if empirical:
        return empirical
    keep = 0.75
    denom = 1.0 - (1.0 - keep) ** len(MODALITIES)
    rows = []
    for available in _nonempty_subsets(MODALITIES):
        prob = (keep ** len(available)) * ((1.0 - keep) ** (len(MODALITIES) - len(available))) / denom
        rows.append(_row(display, "theoretical_from_config", available, prob))
    return rows


def _empirical_or_subset(roots: list[Path], display: str, method: str) -> list[dict[str, Any]]:
    empirical = _empirical_rows(roots, display, method)
    if empirical:
        return empirical
    subsets = list(_nonempty_subsets(MODALITIES))
    prob = 1.0 / len(subsets)
    return [_row(display, "theoretical_from_config", available, prob) for available in subsets]


def _empirical_rows(roots: list[Path], display: str, method: str) -> list[dict[str, Any]]:
    counts: dict[tuple[str, ...], float] = defaultdict(float)
    for root in roots:
        for path in root.glob(f"**/{method}_seed*/random_dropout_pattern_stats.csv"):
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            if "pattern_or_available_set" not in frame or "num_samples" not in frame:
                continue
            for _, row in frame.iterrows():
                token = str(row.get("pattern_or_available_set") or "")
                if not token.startswith("available:"):
                    continue
                available = _canonical(token.split(":", 1)[1].split("+"))
                counts[available] += float(row.get("num_samples") or 0)
    total = sum(counts.values())
    if total <= 0:
        return []
    return [_row(display, "empirical_from_train_log", available, count / total) for available, count in sorted(counts.items(), key=lambda item: (len(item[0]), item[0]))]


def _count_rows(subset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], float] = defaultdict(float)
    for row in subset_rows:
        key = (str(row["method"]), str(row["distribution_source"]), int(row["missing_count"]))
        grouped[key] += float(row["probability"])
    return [
        {
            "method": method,
            "distribution_source": source,
            "missing_count": count,
            "missing_ratio": count / len(MODALITIES),
            "probability": prob,
        }
        for (method, source, count), prob in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2]))
    ]


def _row(method: str, source: str, available: tuple[str, ...], probability: float) -> dict[str, Any]:
    available = _canonical(available)
    missing = tuple(item for item in MODALITIES if item not in available)
    return {
        "method": method,
        "distribution_source": source,
        "available_modalities": ",".join(available),
        "missing_modalities": ",".join(missing),
        "missing_count": len(missing),
        "probability": float(probability),
    }


def _nonempty_subsets(items: tuple[str, ...]):
    for size in range(1, len(items) + 1):
        yield from itertools.combinations(items, size)


def _canonical(items) -> tuple[str, ...]:
    present = {str(item).strip() for item in items if str(item).strip()}
    return tuple(item for item in MODALITIES if item in present)


def _write_summary(path: Path, subset_rows: list[dict[str, Any]]) -> None:
    sources = {str(row["method"]): str(row["distribution_source"]) for row in subset_rows}
    lines = [
        "# Sampling Distribution Summary",
        "",
        "- Natural: full only.",
        "- Uniform pattern exposure: fixed pattern set distribution over full plus one-missing-modality patterns.",
        "- Bernoulli randomdrop: induced by independent modality keep probability; this run uses k075 with keep_prob=0.75 when empirical logs are unavailable.",
        "- Random subset exposure: covers non-empty modality subsets.",
        "- If random subset is uniform over all non-empty subsets, P(S)=1/(2^M-1), S != empty.",
        "",
        "Distribution sources:",
        *[f"- {method}: {source}" for method, source in sorted(sources.items())],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_count(rows: list[dict[str, Any]], stem: Path) -> None:
    frame = pd.DataFrame(rows)
    pivot = frame.pivot_table(index="method", columns="missing_count", values="probability", fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Sampling probability by missing count")
    ax.set_xlabel("Method")
    ax.set_ylabel("Probability")
    ax.legend(title="Missing count")
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=220)
    plt.close(fig)


def _plot_subset(rows: list[dict[str, Any]], stem: Path) -> None:
    frame = pd.DataFrame(rows)
    frame["subset"] = frame["available_modalities"].replace("", "none")
    pivot = frame.pivot_table(index="method", columns="subset", values="probability", fill_value=0)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * len(pivot.columns)), 4.2))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Sampling probability by available subset")
    ax.set_xlabel("Method")
    ax.set_ylabel("Probability")
    ax.legend(title="Available subset", fontsize=7, ncols=2)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
