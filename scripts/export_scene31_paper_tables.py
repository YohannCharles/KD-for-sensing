#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path
from typing import Any


MAIN_METHODS = [
    "proto_natural_es40",
    "proto_sampler_uniform_es40",
    "proto_randomdrop_bernoulli_k075_es40",
    "proto_randomdrop_subset_es40",
    "amr_lite_best_available",
    "amber_lite_best_available",
    "proto_randomdrop_subset_reliability_fusion_es40",
    "proto_randomdrop_subset_pattern_film_d8_es40",
]
ABLATION_METHODS = [
    "proto_natural_es40",
    "proto_sampler_uniform_es40",
    "proto_randomdrop_bernoulli_k075_es40",
    "proto_randomdrop_subset_es40",
    "proto_randomdrop_subset_reliability_fusion_es40",
    "proto_randomdrop_subset_pattern_film_d8_es40",
]
COLUMNS = ["Method", "Full", "Miss-1", "Miss-2", "Miss-3", "Avg-Missing", "Within@3", "MAE", "Overall", "Main read"]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    export(Path(args.summary_root), Path(args.out))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Scene31 paper-ready result tables.")
    parser.add_argument("--summary-root", default="outputs/scene31_subset_reliability_lmdb/summary")
    parser.add_argument("--out", default="outputs/paper_tables/scene31")
    return parser


def export(summary_root: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(summary_root)
    by_method = {row.get("method", ""): row for row in rows}
    notes: list[str] = []

    main_sources = [_resolve_method(name, rows, notes) for name in MAIN_METHODS]
    main_sources = [item for item in main_sources if item is not None]
    ablation_sources = [_resolve_method(name, rows, notes) for name in ABLATION_METHODS]
    ablation_sources = [item for item in ablation_sources if item is not None]
    external_sources = [
        _resolve_external("amr_lite_best_available", rows, notes),
        _resolve_external("amber_lite_best_available", rows, notes),
        by_method.get("proto_randomdrop_subset_es40"),
    ]
    external_sources = [item for item in external_sources if item is not None]

    main_table = _table_rows(main_sources)
    ablation_table = _table_rows(ablation_sources)
    external_table = _table_rows(external_sources)

    _write_csv(out_dir / "table_scene31_main.csv", main_table)
    _write_md(out_dir / "table_scene31_main.md", main_table)
    _write_csv(out_dir / "table_scene31_ablation.csv", ablation_table)
    _write_md(out_dir / "table_scene31_ablation.md", ablation_table)
    _write_csv(out_dir / "table_scene31_external_baselines.csv", external_table)
    _write_md(out_dir / "table_scene31_external_baselines.md", external_table)
    _write_notes(out_dir / "scene31_paper_table_notes.txt", notes)
    print(f"Wrote Scene31 paper tables to {out_dir}.")
    return {"main": main_table, "ablation": ablation_table, "external": external_table, "notes": notes}


def _load_rows(summary_root: Path) -> list[dict[str, str]]:
    rows = _read_csv(summary_root / "combined_method_mean_std.csv")
    present = {row.get("method", "") for row in rows}
    fallback_paths = [
        Path("outputs/scene31_baseline_pack_lmdb/summary/baseline_method_mean_std.csv"),
        Path("outputs/scene31_baseline_pack_lmdb/summary/method_mean_std.csv"),
    ]
    for path in fallback_paths:
        if not path.exists():
            continue
        for row in _read_csv(path):
            method = row.get("method", "")
            if method and method not in present:
                item = dict(row)
                item["fallback_source"] = str(path)
                rows.append(item)
                present.add(method)
    return rows


def _resolve_method(name: str, rows: list[dict[str, str]], notes: list[str]) -> dict[str, str] | None:
    by_method = {row.get("method", ""): row for row in rows}
    if name == "amr_lite_best_available":
        return _resolve_external(name, rows, notes)
    if name == "amber_lite_best_available":
        return _resolve_external(name, rows, notes)
    if name in by_method:
        row = by_method[name]
        if row.get("fallback_source"):
            notes.append(f"{name} read from fallback summary {row.get('fallback_source')}.")
        return row
    token = name.replace("_es40", "")
    match = next((row for row in rows if token in row.get("method", "")), None)
    if match:
        notes.append(f"{name} matched to existing method {match.get('method')}.")
    else:
        notes.append(f"{name} unavailable in Scene31 summary.")
    return match


def _resolve_external(name: str, rows: list[dict[str, str]], notes: list[str]) -> dict[str, str] | None:
    prefix = "amr_lite" if name.startswith("amr") else "amber_lite"
    candidates = [
        row
        for row in rows
        if str(row.get("method", "")).startswith(prefix)
        and _int(row.get("mask_suspect_count")) == 0
        and _int(row.get("excluded_from_official_ranking_count")) == 0
    ]
    if not candidates:
        notes.append(f"{name} excluded: no trusted fresh_eval_maskfix row with mask_suspect=0.")
        return None
    best = max(candidates, key=lambda row: _float(_metric(row, "avg_missing_top1")))
    item = dict(best)
    item["display_method"] = "AMR-lite best available" if prefix == "amr_lite" else "AMBER-lite best available"
    notes.append(f"{name} matched to {best.get('method')} with trusted maskfix.")
    return item


def _table_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best_avg = max((_float(_metric(row, "avg_missing_top1")) for row in rows), default=math.nan)
    out: list[dict[str, str]] = []
    for row in rows:
        avg = _float(_metric(row, "avg_missing_top1"))
        best = _isnum(avg) and _isnum(best_avg) and abs(avg - best_avg) < 1e-12
        out.append(
            {
                "Method": row.get("display_method") or row.get("method", ""),
                "Full": _pct(_metric(row, "full_top1")),
                "Miss-1": _pct(_metric(row, "miss1_top1")),
                "Miss-2": _pct(_metric(row, "miss2_top1")),
                "Miss-3": _pct(_metric(row, "miss3_top1")),
                "Avg-Missing": _pct(avg),
                "Within@3": _pct(_metric(row, "avg_missing_within@3", "avg_missing_within_3")),
                "MAE": _num(_metric(row, "avg_missing_MAE", "avg_missing_mae")),
                "Overall": _pct(_metric(row, "overall_mean_top1")),
                "Main read": _main_read(row, best=best),
            }
        )
    return out


def _main_read(row: dict[str, str], *, best: bool) -> str:
    method = str(row.get("method", ""))
    parts: list[str] = []
    if best:
        parts.append("best Avg-Missing")
    if method == "proto_randomdrop_subset_es40":
        parts.append("trusted reference")
    elif method == "proto_sampler_uniform_es40":
        parts.append("ablation, not final reference")
    elif "reliability_fusion" in method:
        parts.append("not promoted: avg_missing/miss3/MAE worse")
    elif "pattern_film" in method:
        parts.append("not promoted: avg_missing/miss3/MAE worse")
    elif method.startswith(("amr_lite", "amber_lite")):
        parts.append("external baseline, maskfix ok")
    elif "bernoulli" in method or "natural" in method:
        parts.append("training exposure ablation")
    return "; ".join(parts) or str(row.get("main_read") or "comparison")


def _write_notes(path: Path, notes: list[str]) -> None:
    required = [
        "trusted reference = proto_randomdrop_subset_es40",
        "uniform sampler is an ablation, not final reference",
        "reliability fusion is not promoted because avg_missing/miss3/MAE are worse",
        "PatternFiLM d8 is not promoted for same reason",
        "AMR/AMBER-lite are included only if fresh_eval_maskfix exists and mask_suspect=0",
    ]
    unique_notes = list(dict.fromkeys(notes))
    path.write_text("\n".join([*required, "", "Method matching notes:", *[f"- {note}" for note in unique_notes]]) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, str]]) -> None:
    lines = ["| " + " | ".join(COLUMNS) + " |", "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    for row in rows:
        item = dict(row)
        if "best Avg-Missing" in item.get("Main read", ""):
            item["Avg-Missing"] = f"**{item['Avg-Missing']}**"
        lines.append("| " + " | ".join(item.get(column, "") for column in COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric(row: dict[str, str], *names: str) -> float:
    for name in names:
        for key in (f"{name}_mean", name):
            if key in row and row.get(key) not in (None, ""):
                return _float(row.get(key))
    return math.nan


def _pct(value: Any) -> str:
    value = _float(value)
    return f"{value * 100:.2f}%" if _isnum(value) else ""


def _num(value: Any) -> str:
    value = _float(value)
    return f"{value:.2f}" if _isnum(value) else ""


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _isnum(value: Any) -> bool:
    return math.isfinite(_float(value))


def _int(value: Any) -> int:
    value = _float(value)
    return int(value) if _isnum(value) else 0


if __name__ == "__main__":
    raise SystemExit(main())
