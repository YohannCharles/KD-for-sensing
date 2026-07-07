#!/usr/bin/env python3

import argparse
import csv
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


RELATED_WORK_ROWS = [
    {
        "Related work": "ModDrop / modality dropout",
        "Overlap risk": "Training-time random modality/channel dropping for missing-signal robustness.",
        "Our positioning": "Non-empty subset exposure is evaluated as a beam-prediction deployment strategy and compared directly against Bernoulli randomdrop.",
        "Avoid saying": "Do not claim random modality dropping is new.",
        "Citation cue": "arXiv:1501.00102",
    },
    {
        "Related work": "Missing-modality multimodal learning",
        "Overlap risk": "The broad problem is already mature.",
        "Our positioning": "Scope is DeepSense6G Scene31-34 beam prediction with missing-ratio curves, pattern evidence, CDF, and multi-seed statistics.",
        "Avoid saying": "Do not claim first missing-modality learning method.",
        "Citation cue": "arXiv:2409.07825",
    },
    {
        "Related work": "Robust multimodal beam prediction",
        "Overlap risk": "Same application area: beam prediction under missing modalities.",
        "Our positioning": "We emphasize training-only subset coverage and no missing-modality imputation or extra inference module.",
        "Avoid saying": "Do not imply imputation/channel-attention baselines are absent from the literature.",
        "Citation cue": "IWCL 2025 / IEEE 11089951",
    },
    {
        "Related work": "AMBER",
        "Overlap risk": "Closest arbitrary missing-modality beam-prediction work.",
        "Our positioning": "Lightweight alternative to adaptive mask transformers; local AMBER-lite is a caveated baseline, not official AMBER reproduction.",
        "Avoid saying": "Do not claim full official AMBER is beaten.",
        "Citation cue": "arXiv:2512.11331",
    },
    {
        "Related work": "Prototypical Networks",
        "Overlap risk": "Prototype-based prediction is not new.",
        "Our positioning": "Beam-centered prototype head adds gain under subset exposure; subset exposure remains the primary driver.",
        "Avoid saying": "Do not claim prototype alone explains the result.",
        "Citation cue": "NeurIPS 2017 / arXiv:1703.05175",
    },
]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out = Path(args.out)
    paper = Path(args.paper_table_root)
    out.mkdir(parents=True, exist_ok=True)
    paper.mkdir(parents=True, exist_ok=True)

    deltas = _load_key_deltas(Path(args.statistics_root) / "significance_summary.csv")
    _plot_method_overview(out / "fig_method_overview_presentation", deltas)
    _write_related_work_tables(out, paper)
    print(f"Wrote Scene31-34 presentation artifacts to {out}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export slide-friendly Scene31-34 presentation artifacts.")
    parser.add_argument("--statistics-root", default="outputs/scenes31_34_main_lmdb/statistics")
    parser.add_argument("--paper-table-root", default="outputs/paper_tables/scenes31_34_main")
    parser.add_argument("--out", default="outputs/scenes31_34_main_lmdb/presentation")
    return parser


def _plot_method_overview(base: Path, deltas: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.5,
        0.94,
        "Lightweight Robust Beam Prediction Under Arbitrary Missing Modalities",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    boxes = [
        (0.04, 0.54, 0.16, 0.20, "Multimodal input\nimage | radar\nGPS | LiDAR", "#E8F1FA"),
        (0.27, 0.54, 0.19, 0.20, "Train-time sampler\nrandom non-empty subset\nS subset M, S != empty", "#FFF2CC"),
        (0.53, 0.54, 0.17, 0.20, "Shared encoder + fusion\nuses available modalities\nand missing mask", "#E2F0D9"),
        (0.76, 0.54, 0.18, 0.20, "Beam-centered\nprototype head\nbeam-index aware scores", "#FCE4D6"),
        (0.76, 0.18, 0.18, 0.16, "Beam prediction\nTop-1 | Within@3 | MAE", "#EADCF8"),
    ]
    centers: dict[str, tuple[float, float]] = {}
    for idx, (x, y, w, h, text, color) in enumerate(boxes):
        _box(ax, x, y, w, h, text, color=color)
        centers[str(idx)] = (x + w / 2, y + h / 2)
    _arrow(ax, (0.20, 0.755), (0.27, 0.755))
    _arrow(ax, (0.46, 0.755), (0.53, 0.755))
    _arrow(ax, (0.70, 0.755), (0.76, 0.755))
    _arrow(ax, (0.85, 0.54), (0.85, 0.34))

    _box(
        ax,
        0.27,
        0.22,
        0.19,
        0.15,
        "Bernoulli randomdrop baseline\nindependent keep/drop\nnot enough severe-missing coverage",
        color="#F7F7F7",
        linestyle="--",
    )
    _arrow(ax, (0.365, 0.54), (0.365, 0.37), linestyle="--", color="#777777")

    avg_delta = _fmt_pp(deltas.get("bernoulli_avg_missing_top1"))
    cls_delta = _fmt_pp(deltas.get("classifier_avg_missing_top1"))
    ax.text(
        0.5,
        0.07,
        f"Evidence: +{avg_delta} Avg-Missing Top-1 vs Bernoulli; +{cls_delta} vs classifier subset\n"
        "Deployment: training-only exposure, no extra inference-time parameters",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#333333",
    )
    _save(fig, base)


def _write_related_work_tables(out: Path, paper: Path) -> None:
    csv_path = paper / "table_related_work_positioning.csv"
    md_path = paper / "table_related_work_positioning.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RELATED_WORK_ROWS[0]))
        writer.writeheader()
        writer.writerows(RELATED_WORK_ROWS)
    md_path.write_text(_markdown_table(RELATED_WORK_ROWS), encoding="utf-8")
    _plot_related_work_table(out / "fig_related_work_positioning_presentation")


def _plot_related_work_table(base: Path) -> None:
    columns = ["Related line", "Overlap risk", "Our positioning", "Avoid"]
    fig, ax = plt.subplots(figsize=(13.8, 7.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.95, "Related-Work Positioning for the Talk", ha="center", va="center", fontsize=18, fontweight="bold")

    xs = [0.02, 0.20, 0.42, 0.78, 0.98]
    widths = [xs[i + 1] - xs[i] for i in range(4)]
    header_y = 0.82
    header_h = 0.075
    row_h = 0.125
    for idx, label in enumerate(columns):
        _cell(ax, xs[idx], header_y, widths[idx], header_h, label, face="#1F4E79", color="white", weight="bold", ha="center")
    wrap_widths = [18, 28, 42, 28]
    for row_idx, row in enumerate(RELATED_WORK_ROWS):
        y = header_y - (row_idx + 1) * row_h
        face = "#F7FBFF" if row_idx % 2 == 0 else "#FFFFFF"
        values = [row["Related work"], row["Overlap risk"], row["Our positioning"], row["Avoid saying"]]
        for col_idx, value in enumerate(values):
            _cell(
                ax,
                xs[col_idx],
                y,
                widths[col_idx],
                row_h,
                _wrap(value, wrap_widths[col_idx]),
                face=face,
                fontsize=8.7 if col_idx != 2 else 8.2,
            )
    fig.text(
        0.5,
        0.04,
        "Talk stance: lightweight, training-only robust beam prediction; not a first-ever modality-dropout or full AMBER reproduction claim.",
        ha="center",
        fontsize=11,
        color="#333333",
    )
    _save(fig, base)


def _cell(
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    face: str,
    color: str = "#111111",
    weight: str = "normal",
    ha: str = "left",
    fontsize: float = 9.0,
) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", facecolor=face, edgecolor="#D0D0D0", linewidth=0.8))
    tx = x + w / 2 if ha == "center" else x + 0.012
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fontsize, color=color, fontweight=weight, linespacing=1.15)


def _load_key_deltas(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("metric") != "avg_missing_top1":
                continue
            comparison = str(row.get("comparison") or "")
            value = _float(row.get("bootstrap_mean_delta_pp") or row.get("seed_mean_delta_pp"))
            if "Bernoulli" in comparison:
                out["bernoulli_avg_missing_top1"] = value
            if "Classifier" in comparison:
                out["classifier_avg_missing_top1"] = value
    return out


def _box(
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    color: str,
    linestyle: str = "-",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.5,
        edgecolor="#444444",
        facecolor=color,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11.5, linespacing=1.25)


def _arrow(ax: Any, start: tuple[float, float], end: tuple[float, float], *, linestyle: str = "-", color: str = "#333333") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.6,
            linestyle=linestyle,
            color=color,
        )
    )


def _markdown_table(rows: list[dict[str, str]]) -> str:
    fields = list(rows[0])
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_escape_md(row[field]) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def _escape_md(value: str) -> str:
    return str(value).replace("|", "\\|")


def _wrap(value: str, width: int) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _fmt_pp(value: float) -> str:
    return f"{value:.2f} pp" if math.isfinite(value) else "reported"


def _save(fig: Any, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
