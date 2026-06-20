#!/usr/bin/env python
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from kd_sensing.evaluation.metrics import dba_from_circular_distances


RUNS = {
    "5%": Path("outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep/r05/mapping_disabled"),
    "10%": Path("outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep/r10/mapping_disabled"),
    "15%": Path("outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep/r15/mapping_disabled"),
    "20%": Path("outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep/r20/mapping_disabled"),
}
SWEEP_OUT = Path("outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep")
DBA_DELTA = 5.0
NUM_BEAMS = 64


def main() -> int:
    ratio_payloads: dict[str, dict[str, Any]] = {}
    for ratio, run_dir in RUNS.items():
        summary = _read_csv(run_dir / "summary_by_scene.csv")
        predictions = _read_csv(run_dir / "predictions.csv")
        target_best = _best_rows(summary, protocol="target_adapt_beambench")
        within_best = _best_rows(summary, protocol="within_scene_train")
        ratio_payloads[ratio] = {
            "run_dir": run_dir,
            "summary": summary,
            "predictions": predictions,
            "target_best": target_best,
            "within_best": within_best,
        }
        target_rows = _selected_prediction_rows(predictions, target_best, protocol="target_adapt_beambench")
        within_rows = _selected_prediction_rows(predictions, within_best, protocol="within_scene_train")
        _plot_trajectory(
            target_rows,
            run_dir / "gps_prediction_trajectory" / "target_adapt_beambench_best",
            title=f"DeepSense6G GPS v2 target adapt best ({ratio} support)",
            prefix="deepsense6g_gps_v2_target_adapt_best",
        )
        if ratio == "20%":
            _plot_trajectory(
                within_rows,
                run_dir / "gps_prediction_trajectory" / "within_scene_train_best",
                title="DeepSense6G GPS v2 within-scene train best",
                prefix="deepsense6g_gps_v2_within_scene_best",
            )
        _plot_residual_probability(
            target_rows,
            run_dir / "residual_probability" / "target_adapt_beambench_best",
            prefix="target_adapt_best",
            title=f"DeepSense6G target adapt best ({ratio} support)",
        )

    combined_dir = SWEEP_OUT / "residual_probability" / "mapping_disabled" / "target_adapt_beambench_best"
    combined_dir.mkdir(parents=True, exist_ok=True)
    _plot_combined_probability(ratio_payloads, combined_dir)
    report = _write_report(ratio_payloads)
    print(json.dumps({"report": str(report), "runs": {k: str(v) for k, v in RUNS.items()}}, indent=2))
    return 0


def _write_report(ratio_payloads: dict[str, dict[str, Any]]) -> Path:
    rows: list[dict[str, Any]] = []
    within_rows: list[dict[str, Any]] = []
    for ratio, payload in ratio_payloads.items():
        target_rows = _selected_prediction_rows(
            payload["predictions"],
            payload["target_best"],
            protocol="target_adapt_beambench",
        )
        for scene, scene_rows in sorted(_group_by(target_rows, "scene").items()):
            summary = _metric_summary(scene_rows)
            rows.append({"ratio": ratio, "protocol": "target_adapt_beambench", "scene": scene, **summary})
        rows.append({"ratio": ratio, "protocol": "target_adapt_beambench", "scene": "overall", **_metric_summary(target_rows)})
        if ratio == "20%":
            within_selected = _selected_prediction_rows(
                payload["predictions"],
                payload["within_best"],
                protocol="within_scene_train",
            )
            for scene, scene_rows in sorted(_group_by(within_selected, "scene").items()):
                within_rows.append({"protocol": "within_scene_train", "scene": scene, **_metric_summary(scene_rows)})
            within_rows.append({"protocol": "within_scene_train", "scene": "overall", **_metric_summary(within_selected)})

    out_dir = SWEEP_OUT / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "target_adapt_best_by_ratio.csv", rows)
    _write_csv(out_dir / "within_scene_best_error_lt4_upper.csv", within_rows)
    payload = {"target_adapt_best_by_ratio": rows, "within_scene_train_best": within_rows}
    report_path = out_dir / "deepsense6g_gps_v2_support_sweep_summary.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def _metric_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    errors = np.asarray([_float(row.get("circular_error")) for row in rows], dtype=np.float64)
    if errors.size == 0:
        return {
            "sample_count": 0,
            "DBA": 0.0,
            "DBA_error_lt4_to_zero": 0.0,
            "DBA_gain_error_lt4_to_zero": 0.0,
            "p_error_lt4": 0.0,
            "p_error_le4": 0.0,
            "mean_circular_error": 0.0,
            "mean_error_after_lt4_to_zero": 0.0,
            "best_ablation": "",
        }
    transformed = errors.copy()
    transformed[transformed < 4.0] = 0.0
    dba = dba_from_circular_distances(errors, delta=DBA_DELTA)
    dba_upper = dba_from_circular_distances(transformed, delta=DBA_DELTA)
    ablations = Counter(str(row.get("ablation", "")) for row in rows)
    return {
        "sample_count": int(errors.size),
        "DBA": float(dba),
        "DBA_error_lt4_to_zero": float(dba_upper),
        "DBA_gain_error_lt4_to_zero": float(dba_upper - dba),
        "p_error_lt4": float(np.mean(errors < 4.0)),
        "p_error_le4": float(np.mean(errors <= 4.0)),
        "mean_circular_error": float(np.mean(errors)),
        "mean_error_after_lt4_to_zero": float(np.mean(transformed)),
        "exact_acc": float(np.mean(errors == 0.0)),
        "pm1_acc": float(np.mean(errors <= 1.0)),
        "pm2_acc": float(np.mean(errors <= 2.0)),
        "pm4_acc": float(np.mean(errors <= 4.0)),
        "best_ablation": ablations.most_common(1)[0][0] if ablations else "",
    }


def _plot_trajectory(rows: list[dict[str, str]], output_dir: Path, *, title: str, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_groups = sorted(_group_by(rows, "scene").items())
    if not scene_groups:
        return
    fig, axes = plt.subplots(len(scene_groups), 3, figsize=(12.0, max(3.0, 2.8 * len(scene_groups))), constrained_layout=True)
    axes = np.asarray(axes).reshape(len(scene_groups), 3)
    for row_idx, (scene, scene_rows) in enumerate(scene_groups):
        xs = np.asarray([_float(row.get("E")) for row in scene_rows], dtype=float)
        ys = np.asarray([_float(row.get("N")) for row in scene_rows], dtype=float)
        truth = np.asarray([_float(row.get("true_beam")) for row in scene_rows], dtype=float)
        pred = np.asarray([_float(row.get("pred_beam")) for row in scene_rows], dtype=float)
        error = np.asarray([_float(row.get("circular_error")) for row in scene_rows], dtype=float)
        dba_values = np.maximum(1.0 - error / DBA_DELTA, 0.0)
        configs = (("True Beam", truth, 0.0, 63.0), ("Pred Beam", pred, 0.0, 63.0), ("DBA", dba_values, 0.0, 1.0))
        x_min, x_max, y_min, y_max = _xy_limits(xs, ys)
        order = np.argsort([_float(row.get("sample_id").split(":")[-2] if ":" in row.get("sample_id", "") else idx) for idx, row in enumerate(scene_rows)])
        for col_idx, (label, values, vmin, vmax) in enumerate(configs):
            ax = axes[row_idx, col_idx]
            ax.plot(xs[order], ys[order], color="#888888", linewidth=0.7, alpha=0.35)
            sc = ax.scatter(xs, ys, c=values, cmap="viridis", vmin=vmin, vmax=vmax, s=11, linewidths=0.0, alpha=0.9)
            ax.scatter([0.0], [0.0], marker="x", color="red", s=40, linewidths=2.0)
            ax.set_title(f"{scene} {label}", fontsize=10)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, color="#d0d0d0", linewidth=0.6)
            ax.set_xlabel("relative x [m]")
            ax.set_ylabel("relative y [m]")
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{title} Spatial Trajectory", fontsize=13)
    fig.savefig(output_dir / f"{prefix}_spatial_trajectory.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(scene_groups), 2, figsize=(12.0, max(3.0, 2.6 * len(scene_groups))), constrained_layout=True)
    axes = np.asarray(axes).reshape(len(scene_groups), 2)
    summary = []
    for row_idx, (scene, scene_rows) in enumerate(scene_groups):
        sorted_rows = sorted(scene_rows, key=lambda row: row.get("sample_id", ""))
        truth = np.asarray([_float(row.get("true_beam")) for row in sorted_rows], dtype=float)
        pred = np.asarray([_float(row.get("pred_beam")) for row in sorted_rows], dtype=float)
        error = np.asarray([_float(row.get("circular_error")) for row in sorted_rows], dtype=float)
        x = np.arange(len(sorted_rows), dtype=float)
        axes[row_idx, 0].plot(x, truth, color="#1f1f1f", linewidth=1.0, alpha=0.82, label="true")
        axes[row_idx, 0].plot(x, pred, color="#d62728", linewidth=1.0, alpha=0.78, label="pred")
        axes[row_idx, 0].set_title(f"{scene} true vs pred")
        axes[row_idx, 0].set_ylabel("beam")
        axes[row_idx, 0].set_ylim(-2, 65)
        axes[row_idx, 0].grid(True, color="#d0d0d0", linewidth=0.6)
        axes[row_idx, 0].legend(fontsize=8)
        axes[row_idx, 1].plot(x, error, color="#2b6cb0", linewidth=0.8, alpha=0.82)
        axes[row_idx, 1].set_title(f"{scene} circular error")
        axes[row_idx, 1].set_ylabel("error")
        axes[row_idx, 1].set_ylim(-1, 33)
        axes[row_idx, 1].grid(True, color="#d0d0d0", linewidth=0.6)
        summary.append({"scene": scene, **_metric_summary(scene_rows)})
    fig.suptitle(f"{title} Beam Sequence", fontsize=13)
    fig.savefig(output_dir / f"{prefix}_beam_sequence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    (output_dir / f"{prefix}_trajectory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _plot_residual_probability(rows: list[dict[str, str]], output_dir: Path, *, prefix: str, title: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    signed = [_int(row.get("signed_residual")) for row in rows]
    error = [_int(row.get("circular_error")) for row in rows]
    _bar_probability(signed, output_dir / f"{prefix}_signed_residual_probability.png", title=f"{title}: signed residual", xlabel="pred - true circular residual")
    _bar_probability(error, output_dir / f"{prefix}_circular_error_probability.png", title=f"{title}: circular error", xlabel="circular error")


def _plot_combined_probability(payloads: dict[str, dict[str, Any]], output_dir: Path) -> None:
    ratio_rows = {
        ratio: _selected_prediction_rows(payload["predictions"], payload["target_best"], protocol="target_adapt_beambench")
        for ratio, payload in payloads.items()
    }
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=160)
    for ratio, rows in ratio_rows.items():
        values = [_int(row.get("circular_error")) for row in rows]
        xs, ys = _probability(values)
        ax.plot(xs, ys, marker="o", linewidth=1.4, markersize=3.5, label=ratio)
    ax.set_xlabel("circular error")
    ax.set_ylabel("probability")
    ax.set_title("DeepSense6G target adapt best circular-error probability")
    ax.grid(True, color="#d0d0d0", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "target_adapt_best_circular_error_probability_by_ratio.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=160)
    for ratio, rows in ratio_rows.items():
        values = [_int(row.get("signed_residual")) for row in rows]
        xs, ys = _probability(values)
        ax.plot(xs, ys, marker="o", linewidth=1.4, markersize=3.5, label=ratio)
    ax.set_xlabel("signed residual")
    ax.set_ylabel("probability")
    ax.set_title("DeepSense6G target adapt best signed-residual probability")
    ax.grid(True, color="#d0d0d0", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "target_adapt_best_signed_residual_probability_by_ratio.png")
    plt.close(fig)

    scenes = sorted({row.get("scene", "") for rows in ratio_rows.values() for row in rows})
    fig, axes = plt.subplots(len(scenes), 1, figsize=(8.0, max(3.0, 2.2 * len(scenes))), dpi=160, constrained_layout=True)
    axes = np.asarray(axes).reshape(len(scenes))
    for idx, scene in enumerate(scenes):
        ax = axes[idx]
        for ratio, rows in ratio_rows.items():
            scene_rows = [row for row in rows if row.get("scene") == scene]
            xs, ys = _probability([_int(row.get("signed_residual")) for row in scene_rows])
            ax.plot(xs, ys, marker="o", linewidth=1.1, markersize=3.0, label=ratio)
        ax.set_title(scene, fontsize=10)
        ax.set_xlabel("signed residual")
        ax.set_ylabel("prob.")
        ax.grid(True, color="#d0d0d0", linewidth=0.6)
        ax.legend(fontsize=8)
    fig.savefig(output_dir / "target_adapt_best_signed_residual_probability_by_scene_and_ratio.png", bbox_inches="tight")
    plt.close(fig)

    summary = {ratio: _metric_summary(rows) for ratio, rows in ratio_rows.items()}
    (output_dir / "target_adapt_best_residual_probability_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def _bar_probability(values: list[int], path: Path, *, title: str, xlabel: str) -> None:
    xs, ys = _probability(values)
    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=160)
    ax.bar(xs, ys, width=0.85, color="#4c78a8", alpha=0.88)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability")
    ax.set_title(title)
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _probability(values: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    counter = Counter(int(item) for item in values)
    xs = np.asarray(sorted(counter), dtype=int)
    ys = np.asarray([counter[int(x)] / len(values) for x in xs], dtype=float)
    return xs, ys


def _best_rows(rows: list[dict[str, str]], *, protocol: str) -> dict[str, str]:
    grouped = _group_by([row for row in rows if row.get("protocol") == protocol], "scene")
    best: dict[str, str] = {}
    for scene, scene_rows in grouped.items():
        chosen = max(
            scene_rows,
            key=lambda row: (_float(row.get("DBA")), -_float(row.get("mean_circular_error")), row.get("ablation", "")),
        )
        best[scene] = str(chosen.get("ablation", ""))
    return best


def _selected_prediction_rows(rows: list[dict[str, str]], best: dict[str, str], *, protocol: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("protocol") == protocol and str(row.get("ablation", "")) == str(best.get(str(row.get("scene", "")), ""))
    ]


def _group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return dict(grouped)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _xy_limits(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    x_values = np.concatenate([xs, np.asarray([0.0])])
    y_values = np.concatenate([ys, np.asarray([0.0])])
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    x_pad = max((x_max - x_min) * 0.08, 1.0)
    y_pad = max((y_max - y_min) * 0.08, 1.0)
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def _float(value: Any) -> float:
    try:
        if value in {None, ""}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    return int(round(_float(value)))


if __name__ == "__main__":
    raise SystemExit(main())
