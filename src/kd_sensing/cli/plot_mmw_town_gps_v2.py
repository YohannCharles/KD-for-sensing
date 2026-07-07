import csv
from pathlib import Path
from typing import Any


def plot_results(results_dir: Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    rows = _read_csv(results_dir / "predictions.csv")
    out_dir = Path(output_dir or results_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        (out_dir / "plot_unavailable.txt").write_text("predictions.csv is empty or missing.\n", encoding="utf-8")
        return {"figures_dir": str(out_dir), "figure_count": 0}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenes = sorted({row.get("scene", "") for row in rows if row.get("scene")})
    figure_count = 0
    for scene in scenes:
        scene_rows = [row for row in rows if row.get("scene") == scene]
        figure_count += _scatter(
            scene_rows,
            out_dir / f"{scene}_enu_true_label_scatter.png",
            x_key="E",
            y_key="N",
            color_key="true_beam",
            title=f"{scene} true label",
            plt=plt,
        )
        figure_count += _scatter(
            scene_rows,
            out_dir / f"{scene}_enu_prediction_scatter.png",
            x_key="E",
            y_key="N",
            color_key="pred_beam",
            title=f"{scene} prediction",
            plt=plt,
        )
        figure_count += _scatter(
            scene_rows,
            out_dir / f"{scene}_circular_error_heatmap.png",
            x_key="E",
            y_key="N",
            color_key="circular_error",
            title=f"{scene} circular error",
            plt=plt,
        )
        figure_count += _theta_residual(scene_rows, out_dir / f"{scene}_signed_residual_vs_theta.png", plt=plt)
        figure_count += _hist(scene_rows, out_dir / f"{scene}_residual_histogram.png", key="signed_residual", plt=plt)
        figure_count += _label_distribution(scene_rows, out_dir / f"{scene}_label_distribution_compare.png", plt=plt)
        if any(str(row.get("branch_id", "")) != "" for row in scene_rows):
            figure_count += _scatter(
                scene_rows,
                out_dir / f"{scene}_branch_visualization.png",
                x_key="E",
                y_key="N",
                color_key="branch_id",
                title=f"{scene} branch id",
                plt=plt,
            )
        elif "crossroad" in scene or "Hroad" in scene:
            (out_dir / f"{scene}_branch_visualization_unavailable.txt").write_text(
                "branch_id is unavailable in predictions.csv.\n",
                encoding="utf-8",
            )
    return {"figures_dir": str(out_dir), "figure_count": figure_count}


def _scatter(rows: list[dict[str, str]], path: Path, *, x_key: str, y_key: str, color_key: str, title: str, plt: Any) -> int:
    xs = [_float(row.get(x_key)) for row in rows]
    ys = [_float(row.get(y_key)) for row in rows]
    cs = [_float(row.get(color_key)) for row in rows]
    if not xs:
        return 0
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=140)
    scatter = ax.scatter(xs, ys, c=cs, s=18, cmap="viridis", edgecolors="none")
    fig.colorbar(scatter, ax=ax)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return 1


def _theta_residual(rows: list[dict[str, str]], path: Path, *, plt: Any) -> int:
    xs = [_float(row.get("theta_degrees")) for row in rows]
    ys = [_float(row.get("signed_residual")) for row in rows]
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=140)
    ax.scatter(xs, ys, s=14, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("theta_degrees")
    ax.set_ylabel("signed_residual")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return 1


def _hist(rows: list[dict[str, str]], path: Path, *, key: str, plt: Any) -> int:
    values = [_float(row.get(key)) for row in rows]
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=140)
    ax.hist(values, bins=31, color="#4c78a8", alpha=0.85)
    ax.set_xlabel(key)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return 1


def _label_distribution(rows: list[dict[str, str]], path: Path, *, plt: Any) -> int:
    truth: dict[int, int] = {}
    pred: dict[int, int] = {}
    for row in rows:
        truth[int(_float(row.get("true_beam")))] = truth.get(int(_float(row.get("true_beam"))), 0) + 1
        pred[int(_float(row.get("pred_beam")))] = pred.get(int(_float(row.get("pred_beam"))), 0) + 1
    labels = sorted(set(truth) | set(pred))
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
    ax.plot(labels, [truth.get(label, 0) for label in labels], label="true", linewidth=1.4)
    ax.plot(labels, [pred.get(label, 0) for label in labels], label="pred", linewidth=1.4)
    ax.set_xlabel("beam")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return 1


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(value: Any) -> float:
    try:
        if value in {None, ""}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
