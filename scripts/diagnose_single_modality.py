#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


MODALITIES = ("gps", "image", "radar", "lidar")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize single-modality diagnostic runs.")
    parser.add_argument("--root", default="outputs/scene31")
    parser.add_argument("--fusion_eval", default="outputs/scene31/eval/strong_weighted_sum_reliability_beam_proto_missing_patterns.csv")
    parser.add_argument("--out_dir", default="outputs/scene31/analysis")
    args = parser.parse_args(argv)

    root = Path(args.root)
    fusion = _fusion_rows(Path(args.fusion_eval))
    rows = []
    for modality in MODALITIES:
        run = _find_run(root, modality)
        metrics = _metrics(run) if run else {}
        fusion_row = fusion.get(f"{modality}_only", {})
        rows.append(
            {
                "modality": modality,
                "run_name": run.name if run else "",
                "best_top1": metrics.get("best_top1", ""),
                "best_top3": metrics.get("best_top3", ""),
                "best_top5": metrics.get("best_top5", ""),
                "best_adba": metrics.get("best_adba", ""),
                "best_epoch": metrics.get("best_epoch", ""),
                "final_loss": metrics.get("final_loss", ""),
                "fusion_only_top1": fusion_row.get("top1", ""),
                "fusion_only_top3": fusion_row.get("top3", ""),
                "fusion_only_top5": fusion_row.get("top5", ""),
                "fusion_only_adba": fusion_row.get("adba", ""),
                "fusion_gap_top1": _delta(metrics.get("best_top1"), fusion_row.get("top1")),
                "interpretation": _interpret(metrics.get("best_top1"), fusion_row.get("top1")),
            }
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "single_modality_diagnostics.csv", rows)
    _write_md(out_dir / "single_modality_diagnostics.md", rows)
    return 0


def _find_run(root: Path, modality: str) -> Path | None:
    names = [f"diagnostic_{modality}_only_strong", f"m2beam_scene31_{modality}_only"]
    for name in names:
        path = root / name
        if (path / "metrics.csv").exists() or (path / "metrics.json").exists():
            return path
    matches = sorted(root.glob(f"*{modality}_only*"))
    return next((path for path in matches if (path / "metrics.csv").exists() or (path / "metrics.json").exists()), None)


def _metrics(run: Path) -> dict[str, str]:
    rows = _read_csv(run / "metrics.csv") if (run / "metrics.csv").exists() else _json_epoch_logs(run / "metrics.json")
    if not rows:
        return {}
    return {
        "best_top1": _best(rows, "val_acc"),
        "best_top3": _best(rows, "val_atop3", "val_beam_top3"),
        "best_top5": _best(rows, "val_atop5", "val_beam_top5"),
        "best_adba": _best(rows, "val_adba"),
        "best_epoch": str(max(rows, key=lambda row: _float(row.get("val_acc"))).get("epoch", "")),
        "final_loss": str(rows[-1].get("val_loss") or rows[-1].get("train_loss") or ""),
    }


def _fusion_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = _read_csv(path)
    out = {}
    for row in rows:
        pattern = str(row.get("pattern", ""))
        if pattern.startswith("only_"):
            pattern = f"{pattern.removeprefix('only_')}_only"
        out[pattern] = row
    return out


def _best(rows: list[dict], *keys: str) -> str:
    values = []
    for row in rows:
        for key in keys:
            value = _float(row.get(key))
            if value == value:
                values.append(value)
                break
    return "" if not values else f"{max(values):.8g}"


def _interpret(single_top1, fusion_top1) -> str:
    delta = _float(single_top1) - _float(fusion_top1)
    if delta != delta:
        return "missing_data"
    return "fusion_suppression" if delta > 0.03 else "encoder_or_data_bottleneck"


def _delta(a, b) -> str:
    left = _float(a)
    right = _float(b)
    return "" if left != left or right != right else f"{left - right:.8g}"


def _json_epoch_logs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("epoch_logs", []) if isinstance(data, dict) else []


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict]) -> None:
    lines = ["# Single Modality Diagnostics", "", "| modality | run | best_top1 | best_top3 | best_top5 | best_adba | fusion_only_top1 | gap | interpretation |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    lines.extend(
        f"| {row['modality']} | {row['run_name']} | {row['best_top1']} | {row['best_top3']} | {row['best_top5']} | {row['best_adba']} | {row['fusion_only_top1']} | {row['fusion_gap_top1']} | {row['interpretation']} |"
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
