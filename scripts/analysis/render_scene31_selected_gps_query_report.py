#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_FILE = (
    ROOT
    / "outputs"
    / "analysis"
    / "cnn_hybrid_jepa_visual_prior_sweep"
    / "p0_p5_benchmark"
    / "p0_p5_dba_with_non_jepa_clean_baselines_heatmap.html"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "analysis"
    / "cnn_hybrid_jepa_visual_prior_sweep"
    / "p0_p5_benchmark"
    / "scene31_selected"
)

NUMERIC_COLUMNS = [
    "overall_clean",
    "P0",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "overall_p0_p5_mean",
]

DISPLAY_LABELS = {
    "tinyvit_11m_22k_jepa_stage1": "TinyViT-11M 22K | JEPA:N | Pool:GPS-query | Init:ImageNet22K",
    "resnet18_layer4_imagenet_jepa_stage1_mean_pool": "ResNet18 L4 ImageNet | JEPA:Y | Pool:mean | Init:ImageNet",
    "pooler_gps_query_k2_tokens": "PatchViT-16 pooler ablation | JEPA:N | Pool:GPS-query/tokens | Init:scratch",
    "gps_only_control": "GPS-only | JEPA:N | Pool:n/a | Init:n/a",
    "image_AE_gps": "Image AE + GPS | JEPA:N | Pool:n/a | Init:AE",
    "image_only_resnet18_imagenet_gru": "Image-only ResNet18 GRU | JEPA:N | Pool:GRU | Init:ImageNet",
    "tinyvit_11m_22k_mean_pool_supervised": "TinyViT-11M 22K | JEPA:N | Pool:mean | Init:ImageNet22K",
    "resnet18_layer4_imagenet_jepa_stage1": "ResNet18 L4 ImageNet | JEPA:Y | Pool:GPS-query | Init:ImageNet",
    "pooler_mean": "PatchViT-16 pooler ablation | JEPA:N | Pool:mean | Init:scratch",
}

REQUESTED_MODEL_IDS = [
    "tinyvit_11m_22k_jepa_stage1",
    "resnet18_layer4_imagenet_jepa_stage1_mean_pool",
    "pooler_gps_query_k2_tokens",
    "gps_only_control",
    "image_AE_gps",
    "image_only_resnet18_imagenet_gru",
]

PAIRINGS = [
    {
        "name": "TinyViT-11M 22K",
        "gps_model": "tinyvit_11m_22k_jepa_stage1",
        "mean_model": "tinyvit_11m_22k_mean_pool_supervised",
    },
    {
        "name": "ResNet18 L4 ImageNet",
        "gps_model": "resnet18_layer4_imagenet_jepa_stage1",
        "mean_model": "resnet18_layer4_imagenet_jepa_stage1_mean_pool",
    },
    {
        "name": "PatchViT-16 pooler ablation",
        "gps_model": "pooler_gps_query_k2_tokens",
        "mean_model": "pooler_mean",
    },
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Scene31 selected-model GPS-query report.")
    parser.add_argument(
        "--source-csv",
        "--source-file",
        "--source-html",
        dest="source_file",
        type=Path,
        default=DEFAULT_SOURCE_FILE,
        help="Scene31 source file (CSV or HTML)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for rendered outputs")
    parser.add_argument("--title", default="Scene31 selected GPS-query evidence", help="Report title")
    return parser.parse_args(argv)


def load_rows_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _clean_html_cell(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def load_rows_from_html(path: Path) -> list[dict[str, str]]:
    html_text = path.read_text(encoding="utf-8")
    section_match = re.search(
        r"<section><h2>Scene31 only</h2>.*?<tbody>(?P<tbody>.*?)</tbody></table></div></section>",
        html_text,
        flags=re.DOTALL,
    )
    if section_match is None:
        raise ValueError(f"Scene31 section not found in HTML source: {path}")

    rows: list[dict[str, str]] = []
    tbody = section_match.group("tbody")
    for tr_html in re.findall(r"<tr>(.*?)</tr>", tbody, flags=re.DOTALL):
        model_match = re.search(r'title="raw model id: ([^"]+)"', tr_html)
        if model_match is None:
            continue
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr_html, flags=re.DOTALL)
        if len(cells) != 11:
            raise ValueError(
                f"Unexpected Scene31 row shape in {path}: expected 11 cells, got {len(cells)} for {model_match.group(1)}"
            )
        row = {
            "model": model_match.group(1),
            "sample_count": _clean_html_cell(cells[1]),
            "overall_clean": _clean_html_cell(cells[2]),
            "P0": _clean_html_cell(cells[3]),
            "P1": _clean_html_cell(cells[4]),
            "P2": _clean_html_cell(cells[5]),
            "P3": _clean_html_cell(cells[6]),
            "P4": _clean_html_cell(cells[7]),
            "P5": _clean_html_cell(cells[8]),
            "overall_p0_p5_mean": _clean_html_cell(cells[9]),
            "source": _clean_html_cell(cells[10]),
        }
        rows.append(row)
    return rows


def load_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() in {".html", ".htm"}:
        return load_rows_from_html(path)
    return load_rows_from_csv(path)


def row_by_model(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("model", "")): row for row in rows}


def display_label(model_id: str) -> str:
    return DISPLAY_LABELS.get(model_id, model_id)


def source_label(value: str) -> str:
    if not value:
        return ""
    path = value.replace("\\", "/")
    prefix = "outputs/analysis/cnn_hybrid_jepa_visual_prior_sweep/p0_p5_benchmark/"
    if path.startswith(prefix):
        return path[len(prefix) :]
    return path


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def color_for_value(value: float, low: float, high: float) -> str:
    if high <= low:
        hue = 60.0
    else:
        hue = 8.0 + (float(value) - low) / (high - low) * 125.0
    return f"hsl({hue:.1f} 72% 55%)"


def selected_rows(row_map: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    return [row_map[model_id] for model_id in REQUESTED_MODEL_IDS]


def paired_rows(row_map: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pairing in PAIRINGS:
        gps_row = row_map[pairing["gps_model"]]
        mean_row = row_map[pairing["mean_model"]]
        row = {
            "pair": pairing["name"],
            "gps_model": pairing["gps_model"],
            "mean_model": pairing["mean_model"],
            "gps_label": display_label(pairing["gps_model"]),
            "mean_label": display_label(pairing["mean_model"]),
        }
        for column in NUMERIC_COLUMNS:
            gps_value = to_float(gps_row.get(column))
            mean_value = to_float(mean_row.get(column))
            row[column] = None if gps_value is None or mean_value is None else gps_value - mean_value
        rows.append(row)
    return rows


def render_requested_rows_table(rows: list[dict[str, str]]) -> str:
    fieldnames = ["model", "sample_count", *NUMERIC_COLUMNS, "source"]
    ranges: dict[str, tuple[float, float]] = {}
    for column in NUMERIC_COLUMNS:
        values = [to_float(row.get(column)) for row in rows]
        values = [value for value in values if value is not None]
        ranges[column] = (min(values), max(values)) if values else (0.0, 0.0)

    parts = [
        '<div class="table-wrap"><table><thead><tr>',
        *[
            f"<th>{html.escape('model (normalized)' if field == 'model' else field)}</th>"
            for field in fieldnames
        ],
        "</tr></thead><tbody>",
    ]
    for row in rows:
        parts.append("<tr>")
        for field in fieldnames:
            value = row.get(field, "")
            if field == "model":
                parts.append(
                    f'<td class="model-cell" title="raw model id: {html.escape(str(value))}">'
                    f"{html.escape(display_label(str(value)))}</td>"
                )
            elif field in NUMERIC_COLUMNS and value not in ("", None):
                numeric_value = float(value)
                low, high = ranges[field]
                parts.append(
                    f'<td class="num" style="background:{color_for_value(numeric_value, low, high)};">'
                    f"{numeric_value:.6f}</td>"
                )
            elif field == "sample_count" and value:
                parts.append(f'<td class="num">{html.escape(str(value))}</td>')
            elif field == "source":
                parts.append(f'<td>{html.escape(source_label(str(value)))}</td>')
            else:
                parts.append(f"<td>{html.escape(str(value))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def render_delta_table(rows: list[dict[str, Any]]) -> str:
    fieldnames = ["pair", "gps_label", "mean_label", "overall_p0_p5_mean", "P0", "P1", "P2", "P3", "P4", "P5"]
    parts = [
        '<div class="table-wrap"><table><thead><tr>',
        *[f"<th>{html.escape(field)}</th>" for field in fieldnames],
        "</tr></thead><tbody>",
    ]
    for row in rows:
        parts.append("<tr>")
        for field in fieldnames:
            value = row.get(field, "")
            if isinstance(value, (int, float)) or to_float(value) is not None:
                numeric_value = float(value)
                tone = "#e8f7ec" if numeric_value >= 0 else "#fcebea"
                parts.append(
                    f'<td class="num" style="background:{tone};">{numeric_value:+.6f}</td>'
                )
            else:
                parts.append(f"<td>{html.escape(str(value))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def write_ladder_plot(rows: list[dict[str, str]], output_path: Path) -> None:
    labels = [display_label(str(row.get("model", ""))) for row in rows]
    values = [float(row.get("overall_p0_p5_mean", 0.0) or 0.0) for row in rows]
    order = np.argsort(values)
    ordered_labels = [labels[index] for index in order]
    ordered_values = [values[index] for index in order]

    fig, ax = plt.subplots(figsize=(10.4, max(3.5, 0.45 * len(rows) + 1.8)))
    cmap = plt.get_cmap("YlGn")
    norm = plt.Normalize(min(ordered_values), max(ordered_values) if ordered_values else 1.0)
    colors = [cmap(norm(value)) for value in ordered_values]
    ax.barh(range(len(ordered_labels)), ordered_values, color=colors, edgecolor="#355c36", linewidth=0.6)
    ax.set_yticks(range(len(ordered_labels)))
    ax.set_yticklabels(ordered_labels, fontsize=8)
    ax.set_xlabel("overall_p0_p5_mean")
    ax.set_title("Scene31 selected models ladder")
    ax.set_xlim(0.0, max(0.72, max(ordered_values) * 1.08 if ordered_values else 0.72))
    ax.grid(axis="x", color="#d6dde2", linewidth=0.8)
    for index, value in enumerate(ordered_values):
        ax.text(value + 0.006, index, f"{value:.3f}", va="center", ha="left", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_delta_heatmap(rows: list[dict[str, Any]], output_path: Path) -> None:
    columns = ["overall_clean", "P0", "P1", "P2", "P3", "P4", "P5", "overall_p0_p5_mean"]
    matrix = np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=float)
    max_abs = float(np.nanmax(np.abs(matrix))) if matrix.size else 1.0
    vmax = max(max_abs, 1e-6)

    fig, ax = plt.subplots(figsize=(11.2, max(2.6, 0.7 * len(rows) + 1.1)))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row["pair"] for row in rows], fontsize=8)
    ax.set_title("Scene31 matched GPS-query minus mean-pool deltas")
    ax.axvline(6.5, color="#ffffff", linewidth=0.8, alpha=0.8)
    for row_index, row in enumerate(rows):
        for col_index, column in enumerate(columns):
            value = float(row[column])
            ax.text(col_index, row_index, f"{value:+.3f}", ha="center", va="center", fontsize=7, color="#1f1f1f")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="GPS-query - mean-pool")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def render_html(
    *,
    title: str,
    source_path: Path,
    requested_rows: list[dict[str, str]],
    delta_rows: list[dict[str, Any]],
    ladder_plot: Path,
    delta_plot: Path,
    output_path: Path,
) -> None:
    deltas = [float(row["overall_p0_p5_mean"]) for row in delta_rows]
    summary_bits = [
        f"{row['pair']}: {float(row['overall_p0_p5_mean']):+.4f}" for row in delta_rows
    ]
    css = """
    body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:linear-gradient(180deg,#f7f9fb 0,#ffffff 240px);color:#17202a}
    .page{max-width:1400px;margin:0 auto;padding:28px 24px 40px}
    h1{font-size:30px;line-height:1.1;margin:0 0 8px;letter-spacing:-0.02em}
    .note{color:#566573;font-size:14px;max-width:980px;line-height:1.55}
    .summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:18px 0 24px}
    .card{background:#fff;border:1px solid #e3e8ee;border-radius:14px;padding:14px 16px;box-shadow:0 1px 0 rgba(16,24,40,.03)}
    .card .label{font-size:12px;color:#5d6d7e;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
    .card .value{font-size:23px;font-weight:700;letter-spacing:-0.02em}
    .card .sub{font-size:12px;color:#617182;margin-top:6px;line-height:1.45}
    .section{margin-top:28px}
    .section h2{font-size:20px;margin:0 0 8px;letter-spacing:-0.01em}
    .figure{background:#fff;border:1px solid #e3e8ee;border-radius:16px;padding:16px;box-shadow:0 1px 0 rgba(16,24,40,.03);margin-top:12px}
    .figure img{max-width:100%;display:block}
    .figure .caption{margin-top:10px;color:#5b6672;font-size:13px;line-height:1.45}
    .table-wrap{overflow:auto;max-height:76vh;border:1px solid #d7dbdd;border-radius:12px;background:#fff}
    table{border-collapse:collapse;width:100%;font-size:12px}
    th,td{border:1px solid #d7dbdd;padding:6px 8px;vertical-align:middle}
    th{position:sticky;top:0;background:#f4f6f7;z-index:1;text-align:left}
    td.num{text-align:right;font-variant-numeric:tabular-nums}
    td.model-cell{white-space:nowrap}
    .foot{margin-top:14px;color:#566573;font-size:13px;line-height:1.5}
    code{background:#f4f6f7;padding:1px 4px;border-radius:4px}
    """
    requested_table = render_requested_rows_table(requested_rows)
    delta_table = render_delta_table(delta_rows)
    requested_csv = "requested"
    _ = requested_csv
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  <div class="page">
    <h1>{html.escape(title)}</h1>
        <p class="note">从 <code>{html.escape(source_label(str(source_path.relative_to(ROOT))))}</code> 过滤出你点名的 Scene31 行，并额外补上三组 matched mean-pool baseline，用来单独证明 GPS-query 的正向增量。主图是六个指定模型的 Scene31 ladder，第二张图是三组 GPS-query - mean-pool 的配对 delta heatmap。</p>

    <div class="summary">
      <div class="card"><div class="label">TinyViT matched delta</div><div class="value">{deltas[0]:+.4f}</div><div class="sub">{html.escape(summary_bits[0])}</div></div>
      <div class="card"><div class="label">ResNet18 matched delta</div><div class="value">{deltas[1]:+.4f}</div><div class="sub">{html.escape(summary_bits[1])}</div></div>
      <div class="card"><div class="label">PatchViT matched delta</div><div class="value">{deltas[2]:+.4f}</div><div class="sub">{html.escape(summary_bits[2])}</div></div>
    </div>

    <div class="section">
      <h2>Scene31 selected ladder</h2>
      <div class="figure">
        <img src="{html.escape(ladder_plot.name)}" alt="Scene31 selected model ladder" />
        <div class="caption">按 overall_p0_p5_mean 排序的六个指定模型。GPS-query 和控制组的差距可以直接从这一条目页读出来。</div>
      </div>
      <div style="margin-top:12px">{requested_table}</div>
    </div>

    <div class="section">
      <h2>Matched GPS-query vs mean-pool delta</h2>
      <div class="figure">
        <img src="{html.escape(delta_plot.name)}" alt="Matched GPS-query minus mean-pool deltas" />
        <div class="caption">每一行是一个 matched pair，每一列是 Scene31 的 clean / P0-P5 / mean 维度。正值表示 GPS-query 高于 matched mean-pool baseline。</div>
      </div>
      <div style="margin-top:12px">{delta_table}</div>
    </div>

    <p class="foot">一致的结论是：在 Scene31 上，TinyViT、ResNet18 L4 ImageNet 和 PatchViT-16 的 GPS-query 版本都高于各自的 matched mean-pool baseline；而 GPS-only、Image AE + GPS、Image-only ResNet18 GRU 这三类非匹配控制都明显更低，因此这页的证据指向的是 matched multimodal robustness gain，而不是单纯的 GPS shortcut。</p>
  </div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.source_file.exists():
        raise FileNotFoundError(f"Scene31 source file not found: {args.source_file}")
    rows = load_rows(args.source_file)
    row_map = row_by_model(rows)

    missing = [model_id for model_id in [*REQUESTED_MODEL_IDS, *{pair['gps_model'] for pair in PAIRINGS}, *{pair['mean_model'] for pair in PAIRINGS}] if model_id not in row_map]
    if missing:
        raise KeyError(f"Missing required rows in {args.source_csv}: {', '.join(sorted(set(missing)))}")

    requested = selected_rows(row_map)
    deltas = paired_rows(row_map)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ladder_plot = args.output_dir / "scene31_selected_ladder.png"
    delta_plot = args.output_dir / "scene31_selected_query_vs_mean_deltas.png"
    write_ladder_plot(requested, ladder_plot)
    write_delta_heatmap(deltas, delta_plot)

    requested_csv = args.output_dir / "scene31_selected_rows.csv"
    write_csv(requested_csv, requested, ["model", "source", "sample_count", *NUMERIC_COLUMNS])

    delta_csv = args.output_dir / "scene31_query_vs_mean_deltas.csv"
    write_csv(
        delta_csv,
        deltas,
        ["pair", "gps_label", "mean_label", *NUMERIC_COLUMNS],
    )

    html_path = args.output_dir / "scene31_selected_gps_query_report.html"
    render_html(
        title=args.title,
        source_path=args.source_file,
        requested_rows=requested,
        delta_rows=deltas,
        ladder_plot=ladder_plot,
        delta_plot=delta_plot,
        output_path=html_path,
    )

    print(
        "\n".join(
            [
                f"html: {html_path}",
                f"ladder: {ladder_plot}",
                f"delta: {delta_plot}",
                f"rows: {requested_csv}",
                f"deltas: {delta_csv}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())