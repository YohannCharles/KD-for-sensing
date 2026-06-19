from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kd_sensing.diagnostics.jepa_benchmark_artifacts import _read_csv, _write_csv
from kd_sensing.diagnostics.jepa_benchmark_common import _json_ready
from kd_sensing.utils.paths import resolve_path


DEFAULT_OUTPUT_DIR = "outputs/analysis/predictive_gps_query_plus_plus/diagnostics"


def run_predictive_gps_query_visualizations(
    *,
    manifest_path: str | Path,
    output_dir: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    manifest_file = _resolve_existing(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    root = _resolve_output_root(manifest, manifest_file)
    out = _resolve_output_dir(output_dir)
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"Diagnostics output directory is not empty. Use --force to write into it: {out}")
    tables_dir = out / "tables"
    figures_dir = out / "figures"
    manifest_dir = out / "manifest"
    for directory in (tables_dir, figures_dir, manifest_dir):
        directory.mkdir(parents=True, exist_ok=True)

    advantage_rows = _load_manifest_csv(
        manifest,
        root=root,
        output_key="predictive_gps_query_advantage_metrics",
        fallback="results/predictive_gps_query_advantage_metrics.csv",
    )
    condition_rows = _load_manifest_csv(
        manifest,
        root=root,
        output_key="predictive_condition_metrics",
        fallback="results/predictive_condition_metrics.csv",
    )
    all_rows = advantage_rows or condition_rows
    branch_rows = _branch_weight_rows(all_rows)
    latent_rows = _latent_consistency_rows(all_rows)
    rank_rows = _target_rank_cdf_rows(all_rows)
    attention_rows = _attention_summary_rows(all_rows)

    branch_path = tables_dir / "branch_weight_by_condition.csv"
    latent_path = tables_dir / "latent_consistency_by_condition.csv"
    rank_path = tables_dir / "target_rank_cdf.csv"
    attention_path = tables_dir / "attention_summary_by_condition.csv"
    _write_csv(branch_path, branch_rows)
    _write_csv(latent_path, latent_rows)
    _write_csv(rank_path, rank_rows)
    _write_csv(attention_path, attention_rows)

    figure_paths = [
        _write_bar_figure(figures_dir / "branch_weight_by_condition.png", branch_rows, "condition", "gps_residual_weight"),
        _write_bar_figure(figures_dir / "latent_consistency_by_condition.png", latent_rows, "condition", "current_temporal_l2"),
        _write_rank_cdf_figure(figures_dir / "target_rank_cdf.png", rank_rows),
        _write_bar_figure(figures_dir / "attention_entropy_by_condition.png", attention_rows, "condition", "gps_attention_entropy"),
    ]
    bundle = {
        "version": "predictive_gps_query_visual_diagnostics_v1",
        "input_manifest": str(manifest_file),
        "benchmark_output_dir": str(root),
        "output_dir": str(out),
        "evidence_scope": "explanatory_diagnostics_not_primary_claim",
        "claim_statement": "Numeric claims must use strict benchmark metrics and provenance, not these explanatory figures alone.",
        "tables": {
            "branch_weight_by_condition": _relative(branch_path, out),
            "latent_consistency_by_condition": _relative(latent_path, out),
            "target_rank_cdf": _relative(rank_path, out),
            "attention_summary_by_condition": _relative(attention_path, out),
        },
        "figures": [_relative(path, out) for path in figure_paths if path is not None],
        "statuses": {
            "branch_weights": _status(branch_rows),
            "latent_consistency": _status(latent_rows),
            "target_rank_cdf": _status(rank_rows),
            "attention": _status(attention_rows),
        },
    }
    bundle_path = manifest_dir / "predictive_gps_query_visual_diagnostics_manifest.json"
    bundle_path.write_text(json.dumps(_json_ready(bundle), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": str(bundle_path), "output_dir": str(out), **bundle["tables"]}


def _branch_weight_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        weights = _json_field(row.get("gate_weight_mean", row.get("gate_weights", "")))
        output.append(
            {
                "condition": row.get("advantage_condition", row.get("predictive_condition", row.get("condition", ""))),
                "model": row.get("model", ""),
                "current_content_weight": _number(weights.get("current_content", weights.get("current", ""))),
                "temporal_predicted_weight": _number(weights.get("temporal_predicted", "")),
                "gps_residual_weight": _number(weights.get("gps_residual", weights.get("gps", ""))),
                "status": "generated" if weights else "unavailable",
            }
        )
    return output or [{"condition": "", "model": "", "status": "unavailable"}]


def _latent_consistency_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        consistency = _json_field(row.get("latent_consistency", ""))
        output.append(
            {
                "condition": row.get("advantage_condition", row.get("predictive_condition", row.get("condition", ""))),
                "model": row.get("model", ""),
                "current_temporal_l2": _number(
                    consistency.get("current_temporal_l2_mean", row.get("current_temporal_l2", ""))
                ),
                "current_gps_residual_l2": _number(
                    consistency.get("current_gps_residual_l2_mean", row.get("current_gps_residual_l2", ""))
                ),
                "status": "generated" if consistency else "unavailable",
            }
        )
    return output or [{"condition": "", "model": "", "status": "unavailable"}]


def _target_rank_cdf_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        condition = row.get("advantage_condition", row.get("predictive_condition", row.get("condition", "")))
        for rank, column in ((1, "top1"), (3, "top3"), (5, "top5")):
            output.append(
                {
                    "condition": condition,
                    "model": row.get("model", ""),
                    "rank_leq": rank,
                    "cdf": _number(row.get(column, "")),
                    "status": "generated" if row.get(column, "") not in ("", None) else "unavailable",
                }
            )
    return output or [{"condition": "", "model": "", "rank_leq": "", "cdf": "", "status": "unavailable"}]


def _attention_summary_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        summary = _json_field(row.get("gps_query_attention_summary", row.get("attention_summary", "")))
        output.append(
            {
                "condition": row.get("advantage_condition", row.get("predictive_condition", row.get("condition", ""))),
                "model": row.get("model", ""),
                "gps_attention_entropy": _number(summary.get("mean_entropy", row.get("gps_attention_entropy", ""))),
                "status": "generated" if summary else "unavailable",
            }
        )
    return output or [{"condition": "", "model": "", "status": "unavailable"}]


def _write_bar_figure(path: Path, rows: list[Mapping[str, Any]], x_key: str, y_key: str) -> Path | None:
    values = [(str(row.get(x_key, "")), _float_or_none(row.get(y_key))) for row in rows]
    values = [(x, y) for x, y in values if x and y is not None]
    if not values:
        return _write_placeholder_figure(path, "Diagnostics unavailable")
    import matplotlib.pyplot as plt

    labels = [item[0] for item in values]
    y = [float(item[1]) for item in values]
    fig, ax = plt.subplots(figsize=(max(6, min(14, len(labels) * 0.8)), 4))
    ax.bar(range(len(labels)), y)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(y_key)
    ax.set_title("Explanatory diagnostic, not claim evidence")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _write_rank_cdf_figure(path: Path, rows: list[Mapping[str, Any]]) -> Path | None:
    values = [(row.get("rank_leq"), _float_or_none(row.get("cdf"))) for row in rows]
    values = [(int(rank), cdf) for rank, cdf in values if str(rank) and cdf is not None]
    if not values:
        return _write_placeholder_figure(path, "Rank CDF unavailable")
    import matplotlib.pyplot as plt

    by_rank: dict[int, list[float]] = {}
    for rank, value in values:
        by_rank.setdefault(rank, []).append(float(value))
    ranks = sorted(by_rank)
    y = [sum(by_rank[rank]) / len(by_rank[rank]) for rank in ranks]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ranks, y, marker="o")
    ax.set_xlabel("rank <= k")
    ax.set_ylabel("mean CDF")
    ax.set_ylim(0, 1)
    ax.set_title("Explanatory diagnostic, not claim evidence")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _write_placeholder_figure(path: Path, text: str) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _load_manifest_csv(manifest: Mapping[str, Any], *, root: Path, output_key: str, fallback: str) -> list[dict[str, Any]]:
    output_files = manifest.get("output_files", {}) if isinstance(manifest.get("output_files"), Mapping) else {}
    path = root / str(output_files.get(output_key, fallback))
    return _read_csv(path) if path.exists() else []


def _resolve_existing(path: str | Path) -> Path:
    resolved = resolve_path(path)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"Diagnostics manifest not found: {path}")
    return resolved


def _resolve_output_root(manifest: Mapping[str, Any], manifest_file: Path) -> Path:
    raw = manifest.get("output_dir") or manifest.get("outputs", {}).get("output_dir")
    if raw:
        resolved = resolve_path(str(raw))
        if resolved is not None:
            return resolved
    return manifest_file.parent


def _resolve_output_dir(path: str | Path | None) -> Path:
    resolved = resolve_path(path or DEFAULT_OUTPUT_DIR)
    if resolved is None:
        raise FileNotFoundError(f"Output directory could not be resolved: {path}")
    return resolved


def _json_field(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _number(value: Any) -> float | str:
    parsed = _float_or_none(value)
    return "" if parsed is None else parsed


def _float_or_none(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _status(rows: list[Mapping[str, Any]]) -> str:
    statuses = {str(row.get("status", "")) for row in rows if row.get("status", "")}
    return "generated" if "generated" in statuses else "unavailable"


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


__all__ = ["run_predictive_gps_query_visualizations"]
