import csv
import json
from pathlib import Path
from typing import Any


def compare_results(previous_dir: Path, new_dir: Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    out_dir = Path(output_dir or new_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    new_rows = _read_csv(new_dir / "summary_by_scene.csv")
    previous = _previous_metrics(previous_dir)
    comparison_rows = []
    for row in new_rows:
        scene = str(row.get("scene", ""))
        prior = _match_previous(previous, scene)
        new_dba = _float(row.get("DBA"))
        old_dba = prior.get("DBA")
        comparison_rows.append(
            {
                "scene": scene,
                "protocol": row.get("protocol", ""),
                "ablation": row.get("ablation", ""),
                "label_space": row.get("label_space", ""),
                "previous_DBA": "" if old_dba is None else old_dba,
                "new_DBA": new_dba,
                "delta_DBA": "" if old_dba is None else new_dba - float(old_dba),
                "previous_source": prior.get("source", "unavailable"),
                "new_mean_circular_error": row.get("mean_circular_error", ""),
                "new_DBA_zero_ratio": row.get("DBA_zero_ratio", ""),
            }
        )
    _write_csv(out_dir / "comparison_with_previous.csv", comparison_rows)
    report = _comparison_report(comparison_rows)
    (out_dir / "comparison_report.md").write_text(report, encoding="utf-8")
    return {
        "comparison_csv": str(out_dir / "comparison_with_previous.csv"),
        "comparison_report": str(out_dir / "comparison_report.md"),
        "rows": len(comparison_rows),
    }


def _previous_metrics(previous_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(previous_dir.rglob("metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metrics = payload.get("metrics", payload)
        scene = payload.get("target_scene") or payload.get("scene") or path.parent.name
        value = metrics.get("DBA", metrics.get("dba_avg", metrics.get("mean_DBA")))
        error = metrics.get("mean_circular_error", metrics.get("circular_beam_error_mean"))
        rows.append({"scene": str(scene), "DBA": _optional_float(value), "mean_circular_error": _optional_float(error), "source": str(path)})
    for path in sorted(previous_dir.rglob("*summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        scene_results = payload.get("scene_results") if isinstance(payload, dict) else None
        if isinstance(scene_results, list):
            for item in scene_results:
                metrics = item.get("metrics", item)
                scene = item.get("target_scene") or item.get("scene") or item.get("scenario") or path.parent.name
                rows.append(
                    {
                        "scene": str(scene),
                        "DBA": _optional_float(metrics.get("DBA", metrics.get("dba_avg"))),
                        "mean_circular_error": _optional_float(metrics.get("mean_circular_error", metrics.get("circular_beam_error_mean"))),
                        "source": str(path),
                    }
                )
    return [row for row in rows if row.get("DBA") is not None or row.get("mean_circular_error") is not None]


def _match_previous(rows: list[dict[str, Any]], scene: str) -> dict[str, Any]:
    scene_lower = scene.lower()
    for row in rows:
        if scene_lower in str(row.get("scene", "")).lower() or str(row.get("scene", "")).lower() in scene_lower:
            return row
    return {}


def _comparison_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# MMW Town GPS-only v2 Comparison Report",
        "",
        "This report compares v2 circular scene-adapter summaries with available previous diagnostics.",
        "",
        "| Scene | Protocol | Ablation | Previous DBA | New DBA | Delta |",
        "|---|---|---|---:|---:|---:|",
    ]
    focus = {"crossroad", "hroad", "curvyroad", "skybridge"}
    for row in rows:
        scene = str(row.get("scene", ""))
        if not any(token in scene.lower() for token in focus):
            continue
        lines.append(
            "| {scene} | {protocol} | {ablation} | {old} | {new:.4f} | {delta} |".format(
                scene=scene,
                protocol=row.get("protocol", ""),
                ablation=row.get("ablation", ""),
                old=row.get("previous_DBA", "unavailable") or "unavailable",
                new=float(row.get("new_DBA", 0.0)),
                delta=_format_delta(row.get("delta_DBA")),
            )
        )
    if len(lines) == 6:
        lines.append("| unavailable | unavailable | unavailable | unavailable | 0.0000 | unavailable |")
    lines.extend(
        [
            "",
            "Notes:",
            "- crossroad and Hroad should be read primarily through residual_by_theta_bin.csv and residual_by_branch.csv.",
            "- within_scene_train rows are sanity upper bounds, not cross-scene generalization claims.",
            "- Missing previous DBA means the previous diagnostics did not expose a compatible scalar in the scanned JSON files.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _optional_float(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float:
    result = _optional_float(value)
    return 0.0 if result is None else float(result)


def _format_delta(value: Any) -> str:
    try:
        if value in {None, ""}:
            return "unavailable"
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "unavailable"
