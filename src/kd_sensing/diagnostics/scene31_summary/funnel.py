#!/usr/bin/env python3

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

from kd_sensing.diagnostics.scene31_summary import bc_next

UNIFORM_REFERENCE = {
    "avg_missing_top1": 0.2856,
    "overall_mean_top1": 0.2784,
    "full_top1": 0.4216,
    "balanced": 0.3560,
    "avg_missing_within_3": 0.6916,
    "avg_missing_mae": 4.6359,
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    forwarded = ["--root", args.root, "--manifest", args.manifest, "--out", str(out_dir), "--name-prefix", ""]
    for metrics in args.metrics:
        forwarded.extend(["--metrics", metrics])
    result = bc_next.main(forwarded)

    methods = _read_csv(out_dir / "method_mean_std.csv")
    selection = _collect_selection(Path(args.root))
    _write_csv(out_dir / "checkpoint_selection_summary.csv", selection, _selection_fields(selection))
    _annotate_methods(methods)
    _write_csv(out_dir / "funnel_method_mean_std.csv", methods, _method_fields(methods))
    _copy_if_exists(out_dir / "per_run.csv", out_dir / "funnel_per_run.csv")
    _copy_if_exists(out_dir / "delta_vs_uniform.csv", out_dir / "funnel_delta_vs_uniform.csv")
    _write_conclusion(out_dir / "funnel_conclusion.txt", methods, selection)
    print(f"Wrote Scene31 funnel summary to {out_dir}.")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Scene31 funnel fresh eval metrics.")
    parser.add_argument("--root", default="outputs/scene31_funnel_lmdb")
    parser.add_argument("--out", default="outputs/scene31_funnel_lmdb/summary")
    parser.add_argument("--manifest", default="configs/scene31/funnel/experiment_manifest.csv")
    parser.add_argument("--metrics", action="append", default=[])
    return parser


def _annotate_methods(rows: list[dict[str, Any]]) -> None:
    by_method = {row.get("method", ""): row for row in rows}
    uniform = by_method.get("proto_sampler_uniform_es40")
    for row in rows:
        method = str(row.get("method") or "")
        labels = []
        if _float(row.get("avg_missing_top1_mean")) > UNIFORM_REFERENCE["avg_missing_top1"]:
            labels.append("candidate_second_innovation")
        elif _improves_bucket(row, uniform) or _improves_beam(row, uniform):
            labels.append("auxiliary_analysis_candidate")
        if _is_quick(method):
            labels.append(_promotion_label(row, uniform))
        row["main_read"] = ",".join(dict.fromkeys(labels)) if labels else "standard_compare"


def _promotion_label(row: dict[str, Any], uniform: dict[str, Any] | None) -> str:
    if _float(row.get("avg_missing_top1_mean")) >= UNIFORM_REFERENCE["avg_missing_top1"]:
        return "promote_to_full_seeds"
    if _float(row.get("overall_mean_top1_mean")) >= UNIFORM_REFERENCE["overall_mean_top1"]:
        return "promote_to_full_seeds"
    if _improves_bucket(row, uniform) or _improves_beam(row, uniform):
        return "promote_to_full_seeds"
    return "do_not_promote"


def _improves_bucket(row: dict[str, Any], uniform: dict[str, Any] | None) -> bool:
    if uniform is None:
        return False
    return any(
        _float(row.get(f"{metric}_mean")) > _float(uniform.get(f"{metric}_mean"))
        for metric in ("miss2_top1", "miss3_top1")
    )


def _improves_beam(row: dict[str, Any], uniform: dict[str, Any] | None) -> bool:
    base_within = _float(uniform.get("avg_missing_within_3_mean")) if uniform else UNIFORM_REFERENCE["avg_missing_within_3"]
    base_mae = _float(uniform.get("avg_missing_mae_mean")) if uniform else UNIFORM_REFERENCE["avg_missing_mae"]
    return _float(row.get("avg_missing_within_3_mean")) > base_within and _float(row.get("avg_missing_mae_mean")) < base_mae


def _is_quick(method: str) -> bool:
    return any(
        token in method
        for token in (
            "pattern_logit_bias",
            "modbias_entropy",
            "pattern_film",
            "tta_entropy_bn",
            "pbpr_fixed",
        )
    )


def _collect_selection(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "checkpoint_selection").rglob("checkpoint_selection_summary.csv")):
        for row in _read_csv(path):
            row["selection_path"] = str(path)
            rows.append(row)
    return rows


def _write_conclusion(path: Path, methods: list[dict[str, Any]], selection: list[dict[str, Any]]) -> None:
    exact = _best(methods, "avg_missing_top1_mean", reverse=True)
    exceeding_avg = _methods_over(methods, "avg_missing_top1_mean", UNIFORM_REFERENCE["avg_missing_top1"])
    exceeding_overall = _methods_over(methods, "overall_mean_top1_mean", UNIFORM_REFERENCE["overall_mean_top1"])
    promoted = [row["method"] for row in methods if "promote_to_full_seeds" in str(row.get("main_read", ""))]
    not_promoted = [row["method"] for row in methods if "do_not_promote" in str(row.get("main_read", ""))]
    lines = [
        "Current exact Top1 winner:",
        exact or "unavailable",
        "",
        "Methods exceeding uniform reference avg_missing_top1=0.2856:",
        *(_dash(exceeding_avg) or ["- none"]),
        "",
        "Methods exceeding uniform reference overall_mean_top1=0.2784:",
        *(_dash(exceeding_overall) or ["- none"]),
        "",
        "Methods improving miss2/miss3:",
        *(_dash(_bucket_improvers(methods)) or ["- none"]),
        "",
        "Methods improving beam proximity:",
        *(_dash(_beam_improvers(methods)) or ["- none"]),
        "",
        "Missing-aware selection:",
        *(_selection_lines(selection) or ["- unavailable"]),
        "",
        "JTT stability:",
        "- inspect JTT n=5 mean after all seeds finish",
        "",
        "MVFR vs JTT:",
        "- compare avg_missing_top1 and miss2/miss3 after MVFR seeds finish",
        "",
        "Mild MP-DRO vs tau1 MP-DRO:",
        "- compare full_top1 drop and miss bucket gains against tau1 baseline",
        "",
        "Quick screens promoted to full seeds:",
        *(_dash(promoted) or ["- none"]),
        "",
        "Quick screens not promoted:",
        *(_dash(not_promoted) or ["- none"]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selection_lines(rows: list[dict[str, Any]]) -> list[str]:
    out = []
    for row in rows[:20]:
        out.append(f"- {row.get('run')} {row.get('rule')}: epoch {row.get('selected_epoch')} score {row.get('score')}")
    return out


def _bucket_improvers(rows: list[dict[str, Any]]) -> list[str]:
    uniform = next((row for row in rows if row.get("method") == "proto_sampler_uniform_es40"), None)
    if uniform is None:
        return []
    return [
        str(row.get("method"))
        for row in rows
        if row is not uniform and (_improves_bucket(row, uniform))
    ]


def _beam_improvers(rows: list[dict[str, Any]]) -> list[str]:
    uniform = next((row for row in rows if row.get("method") == "proto_sampler_uniform_es40"), None)
    return [str(row.get("method")) for row in rows if _improves_beam(row, uniform)]


def _methods_over(rows: list[dict[str, Any]], metric: str, threshold: float) -> list[str]:
    return [str(row.get("method")) for row in rows if _float(row.get(metric)) > threshold]


def _best(rows: list[dict[str, Any]], metric: str, *, reverse: bool) -> str:
    valid = [row for row in rows if _float(row.get(metric)) == _float(row.get(metric))]
    if not valid:
        return ""
    return str(sorted(valid, key=lambda row: _float(row.get(metric)), reverse=reverse)[0].get("method", ""))


def _dash(values: list[str]) -> list[str]:
    return [f"- {value}" for value in values]


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        shutil.copyfile(source, target)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _method_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _selection_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields = ["run", "rule", "selected_epoch", "checkpoint", "full_top1", "miss1_top1", "miss2_top1", "miss3_top1", "avg_missing_top1", "score", "warning", "selection_path"]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
