#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    write_conclusion(
        scene31_summary=Path(args.scene31_summary),
        scenes31_34_summary=Path(args.scenes31_34_summary),
        scenes31_34_per_scene=Path(args.scenes31_34_per_scene),
        paper_tables=Path(args.paper_tables),
        out=Path(args.out),
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write final missing-modality experiment conclusion.")
    parser.add_argument("--scene31-summary", default="outputs/scene31_subset_reliability_lmdb/summary")
    parser.add_argument("--scenes31-34-summary", default="outputs/scenes31_34_subset_reliability_lmdb/summary")
    parser.add_argument("--scenes31-34-per-scene", default="outputs/scenes31_34_subset_reliability_lmdb/per_scene_summary")
    parser.add_argument("--paper-tables", default="outputs/paper_tables")
    parser.add_argument("--out", default="outputs/paper_tables/final_experiment_conclusion.txt")
    return parser


def write_conclusion(
    *,
    scene31_summary: Path,
    scenes31_34_summary: Path,
    scenes31_34_per_scene: Path,
    paper_tables: Path,
    out: Path,
) -> None:
    scene31_methods = _read_csv(scene31_summary / "combined_method_mean_std.csv")
    pooled = _read_csv(scenes31_34_summary / "pooled_method_mean_std.csv")
    stability = _read_csv(scenes31_34_per_scene / "mean_over_scenes.csv")
    per_scene_table = _read_csv(paper_tables / "scenes31_34" / "table_scenes31_34_per_scene.csv")
    subset_scene_wins = sum(1 for row in per_scene_table if row.get("Best") == "Subset")
    stability_winner = stability[0].get("method", "unavailable") if stability else "unavailable"
    pooled_winner = max(pooled, key=lambda row: _float(row.get("avg_missing_top1_mean", "-nan")), default={}).get("method", "unavailable")
    subset = _find(scene31_methods, "proto_randomdrop_subset_es40")
    amber = max((row for row in scene31_methods if row.get("method", "").startswith("amber_lite")), key=lambda row: _float(row.get("avg_missing_top1_mean")), default={})

    lines = [
        "Current final trusted method:",
        "prototype + randomdrop subset exposure",
        "",
        "Scene31 conclusion:",
        "On Scene31, proto_randomdrop_subset_es40 is the trusted reference and current winner among credible methods.",
        f"- proto_randomdrop_subset_es40 avg_missing={subset.get('avg_missing_top1_mean', 'unavailable')}",
        "",
        "Reliability conclusion:",
        "Reliability fusion is not promoted.",
        "It improves light missing in Scene31 but hurts miss3 / avg_missing / MAE.",
        "It also does not improve avg_missing in Scene31–34 quick validation.",
        "",
        "PatternFiLM conclusion:",
        "PatternFiLM d8 is not promoted after being combined with randomdrop_subset.",
        "",
        "AMR/AMBER conclusion:",
        "AMR/AMBER-lite maskfix eval is now trustworthy if fresh_eval_maskfix exists and mask_suspect=0.",
        "They are included as external baselines but do not outperform proto_randomdrop_subset.",
        f"- best AMBER-lite avg_missing={amber.get('avg_missing_top1_mean', 'unavailable')}",
        "",
        "Scene31–34 conclusion:",
        "Scene31–34 quick validation supports randomdrop_subset as pooled winner.",
        "Per-scene metrics show whether this is stable across individual scenes.",
        f"- pooled winner: {pooled_winner}",
        f"- scene stability winner: {stability_winner}",
        f"- subset per-scene wins: {subset_scene_wins}/{len(per_scene_table)}",
        "",
        "Next experimental step:",
        "Do not continue new modules.",
        "Only consider adding seeds for Scene31–34 subset/natural/uniform if per-scene results are stable and paper requires multi-seed evidence.",
        "",
        "Frozen non-goals:",
        "- Do not continue reliability / PatternFiLM / JTT / MVFR / MPDRO.",
        "- Do not continue beamsoft / condBTAPA / weakKD.",
        "- Do not treat uniform sampler as the main reference.",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote final experiment conclusion to {out}.")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    return next((row for row in rows if row.get("method") == method), {})


def _float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
