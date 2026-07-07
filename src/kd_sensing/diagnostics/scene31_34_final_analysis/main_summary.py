#!/usr/bin/env python3

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REFERENCE_METHOD = "scenes31_34_proto_randomdrop_subset_es40"
MODALITIES = ("image", "radar", "gps", "lidar")
SCENES = ("Scene31", "Scene32", "Scene33", "Scene34")
CORE_PATTERNS = ("full", "missing_gps", "missing_radar", "radar_only", "lidar_only")
METHOD_ORDER = (
    "scenes31_34_proto_natural_es40",
    "scenes31_34_proto_sampler_uniform_es40",
    "scenes31_34_proto_randomdrop_bernoulli_k075_es40",
    "scenes31_34_proto_randomdrop_subset_es40",
    "scenes31_34_classifier_natural_es40",
    "scenes31_34_classifier_randomdrop_subset_es40",
    "scenes31_34_amr_lite_natural_es40",
    "scenes31_34_amber_lite_natural_es40",
    "scenes31_34_amr_lite_uniform_es40",
    "scenes31_34_amber_lite_uniform_es40",
)
CORE_METHODS = METHOD_ORDER[:4]
CLASSIFIER_METHODS = METHOD_ORDER[4:6]
EXTERNAL_METHODS = METHOD_ORDER[6:]
RUN_FIELDS = [
    "source_root",
    "eval_dir",
    "run_name",
    "method",
    "seed",
    "status",
    "checkpoint_used",
    "max_batches",
    "full_top1",
    "miss1_top1",
    "miss2_top1",
    "miss3_top1",
    "avg_missing_top1",
    "overall_mean_top1",
    "avg_missing_within@3",
    "avg_missing_MAE",
    "balanced",
    "num_patterns",
    "num_samples",
    "family",
    "maskfix_eval",
    "mask_suspect",
    "excluded_from_official_ranking",
    "official_ranking_included",
    "main_read",
]
METHOD_FIELDS = [
    "method",
    "n",
    "full_top1_mean",
    "full_top1_std",
    "miss1_top1_mean",
    "miss1_top1_std",
    "miss2_top1_mean",
    "miss2_top1_std",
    "miss3_top1_mean",
    "miss3_top1_std",
    "avg_missing_top1_mean",
    "avg_missing_top1_std",
    "overall_mean_top1_mean",
    "overall_mean_top1_std",
    "avg_missing_within@3_mean",
    "avg_missing_within@3_std",
    "avg_missing_MAE_mean",
    "avg_missing_MAE_std",
    "balanced_mean",
    "balanced_std",
]
FINAL_METHOD_FIELDS = [
    "method",
    "family",
    "n",
    "full_top1_mean",
    "full_top1_std",
    "miss1_top1_mean",
    "miss1_top1_std",
    "miss2_top1_mean",
    "miss2_top1_std",
    "miss3_top1_mean",
    "miss3_top1_std",
    "avg_missing_top1_mean",
    "avg_missing_top1_std",
    "overall_mean_top1_mean",
    "overall_mean_top1_std",
    "avg_missing_within@3_mean",
    "avg_missing_within@3_std",
    "avg_missing_MAE_mean",
    "avg_missing_MAE_std",
    "top1_drop_0_to_75_mean",
    "top1_drop_0_to_75_std",
    "mae_at_75_mean",
    "mae_at_75_std",
    "mask_suspect_count",
    "official_ranking_included",
    "claim_status",
    "caveat",
    "main_read",
]
EVIDENCE_CHECKLIST_FIELDS = ["item", "status", "required", "observed", "caveat", "next_action"]
PER_SCENE_RUN_FIELDS = [
    "scene",
    "run_name",
    "method",
    "seed",
    "full_top1",
    "miss1_top1",
    "miss2_top1",
    "miss3_top1",
    "avg_missing_top1",
    "overall_mean_top1",
    "avg_missing_within@3",
    "avg_missing_MAE",
    "balanced",
    "num_patterns",
    "num_samples",
]
PER_SCENE_METHOD_FIELDS = ["scene", *METHOD_FIELDS]
MEAN_OVER_SCENES_FIELDS = [
    "method",
    "n",
    "avg_missing_top1_mean_over_scenes",
    "avg_missing_top1_std_over_scenes",
    "full_top1_mean_over_scenes",
    "full_top1_std_over_scenes",
    "miss1_top1_mean_over_scenes",
    "miss1_top1_std_over_scenes",
    "miss2_top1_mean_over_scenes",
    "miss2_top1_std_over_scenes",
    "miss3_top1_mean_over_scenes",
    "miss3_top1_std_over_scenes",
    "avg_missing_within@3_mean_over_scenes",
    "avg_missing_MAE_mean_over_scenes",
    "balanced_mean_over_scenes",
]
CURVE_FIELDS = [
    "method",
    "n",
    "missing_count",
    "missing_ratio",
    "top1_mean",
    "top1_std",
    "within3_mean",
    "within3_std",
    "mae_mean",
    "mae_std",
    "num_patterns",
    "num_samples",
]
CURVE_BY_SCENE_FIELDS = [
    "scene",
    "method",
    "seed",
    "missing_count",
    "missing_ratio",
    "top1",
    "within3",
    "mae",
    "num_patterns",
    "num_samples",
]
DELTA_FIELDS = [
    "method",
    "n",
    "reference_method",
    "delta_full_top1",
    "delta_miss1_top1",
    "delta_miss2_top1",
    "delta_miss3_top1",
    "delta_avg_missing_top1",
    "delta_overall_mean_top1",
    "delta_avg_missing_within@3",
    "delta_avg_missing_MAE",
    "delta_balanced",
]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summarize(
        Path(args.root),
        Path(args.out),
        [Path(item) for item in args.old_root],
        [Path(item) for item in args.classifier_root],
        [Path(item) for item in args.external_root],
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Scene31-34 main missing-modality experiments.")
    parser.add_argument("--root", default="outputs/scenes31_34_main_lmdb")
    parser.add_argument("--old-root", action="append", default=[])
    parser.add_argument("--classifier-root", action="append", default=[])
    parser.add_argument("--external-root", action="append", default=[])
    parser.add_argument("--out", default="outputs/scenes31_34_main_lmdb/summary")
    return parser


def summarize(
    root: Path,
    out_dir: Path,
    old_roots: list[Path],
    classifier_roots: list[Path] | None = None,
    external_roots: list[Path] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    classifier_roots = classifier_roots or []
    external_roots = external_roots or []
    roots = [*old_roots, root, *classifier_roots, *external_roots]
    eval_dirs = _find_eval_dirs(roots)
    pattern_rows, prediction_rows, warnings = _load_eval_rows(eval_dirs)
    pattern_rows, prediction_rows = _filter_supported_methods(pattern_rows, prediction_rows, warnings)
    per_run = _per_run_rows(pattern_rows)
    per_scene_run = _per_scene_rows(prediction_rows)
    if not per_scene_run:
        per_scene_run = _fallback_per_scene_rows(roots)
        if per_scene_run:
            warnings.append("per-scene rows loaded from existing per_scene_summary fallback")
    method_rows = _method_rows(per_run, official_only=True)
    per_scene_method = _per_scene_method_rows(per_scene_run)
    mean_over_scenes = _mean_over_scenes(per_scene_method)
    curve_by_run = _missing_count_curve_by_run(pattern_rows)
    curve = _missing_count_curve(curve_by_run, warnings)
    curve_by_scene = _missing_count_curve_by_scene(prediction_rows, warnings)
    final_method_rows = _final_method_rows(per_run, curve_by_run=curve_by_run)
    evidence_checklist = _final_evidence_checklist(
        final_method_rows,
        per_run=per_run,
        per_scene_run=per_scene_run,
        curve=curve,
        out_dir=out_dir,
    )
    delta_rows = _delta_rows(method_rows)
    final_delta_rows = _delta_rows(final_method_rows)
    classifier_rows = _baseline_rows(final_method_rows, [
        "scenes31_34_classifier_natural_es40",
        "scenes31_34_proto_natural_es40",
        "scenes31_34_classifier_randomdrop_subset_es40",
        "scenes31_34_proto_randomdrop_subset_es40",
    ])
    external_rows = _baseline_rows(
        final_method_rows,
        [
            "scenes31_34_amr_lite_natural_es40",
            "scenes31_34_amber_lite_natural_es40",
            "scenes31_34_amr_lite_uniform_es40",
            "scenes31_34_amber_lite_uniform_es40",
            "scenes31_34_proto_randomdrop_subset_es40",
        ],
    )
    conclusion = _conclusion_lines(per_run, method_rows, mean_over_scenes, curve, warnings)

    _write_csv(out_dir / "per_run.csv", per_run, RUN_FIELDS)
    _write_csv(out_dir / "method_mean_std.csv", method_rows, METHOD_FIELDS)
    _write_csv(out_dir / "per_scene_per_run.csv", per_scene_run, PER_SCENE_RUN_FIELDS)
    _write_csv(out_dir / "per_scene_method_mean_std.csv", per_scene_method, PER_SCENE_METHOD_FIELDS)
    _write_csv(out_dir / "mean_over_scenes.csv", mean_over_scenes, MEAN_OVER_SCENES_FIELDS)
    _write_csv(out_dir / "missing_count_curve.csv", curve, CURVE_FIELDS)
    _write_csv(out_dir / "missing_count_curve_by_scene.csv", curve_by_scene, CURVE_BY_SCENE_FIELDS)
    _write_csv(out_dir / "delta_vs_randomdrop_subset.csv", delta_rows, DELTA_FIELDS)
    _write_csv(out_dir / "final_method_mean_std.csv", final_method_rows, FINAL_METHOD_FIELDS)
    _write_csv(out_dir / "final_missing_count_curve.csv", curve, CURVE_FIELDS)
    _write_csv(out_dir / "final_external_baselines.csv", external_rows, FINAL_METHOD_FIELDS)
    _write_csv(out_dir / "final_classifier_baselines.csv", classifier_rows, FINAL_METHOD_FIELDS)
    _write_csv(out_dir / "final_delta_vs_proto_subset.csv", final_delta_rows, DELTA_FIELDS)
    _write_csv(out_dir / "final_evidence_checklist.csv", evidence_checklist, EVIDENCE_CHECKLIST_FIELDS)
    _write_checklist_md(out_dir / "final_evidence_checklist.md", evidence_checklist)
    _write_rank(out_dir / "rank_by_avg_missing_top1.md", method_rows)
    _write_stability_rank(out_dir / "rank_by_scene_stability.md", mean_over_scenes)
    (out_dir / "scenes31_34_main_conclusion.txt").write_text("\n".join(conclusion) + "\n", encoding="utf-8")
    if warnings:
        (out_dir / "summary_warnings.txt").write_text("\n".join(f"- {item}" for item in sorted(set(warnings))) + "\n", encoding="utf-8")
    print(f"Wrote Scene31-34 main summary to {out_dir}.")
    return {
        "per_run": per_run,
        "method_rows": method_rows,
        "final_method_rows": final_method_rows,
        "per_scene_per_run": per_scene_run,
        "per_scene_method": per_scene_method,
        "mean_over_scenes": mean_over_scenes,
        "missing_count_curve": curve,
        "missing_count_curve_by_scene": curve_by_scene,
        "final_evidence_checklist": evidence_checklist,
        "delta_rows": delta_rows,
        "warnings": warnings,
    }


def _filter_supported_methods(
    pattern_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    official = set(METHOD_ORDER)
    methods = {
        str(row.get("method") or "")
        for row in [*pattern_rows, *prediction_rows]
        if row.get("method")
    }
    excluded = sorted(method for method in methods if method not in official)
    if excluded:
        warnings.append("excluded auxiliary methods from official Scene31-34 main summary: " + ",".join(excluded))
    return (
        [row for row in pattern_rows if str(row.get("method") or "") in official],
        [row for row in prediction_rows if str(row.get("method") or "") in official],
    )


def _find_eval_dirs(roots: list[Path]) -> list[Path]:
    by_run: dict[str, Path] = {}
    scores: dict[str, tuple[int, int]] = {}
    for root in roots:
        candidates = [
            *(root / "fresh_eval_maskfix_with_scene").glob("*"),
            *(root / "fresh_eval_maskfix").glob("*"),
            *(root / "fresh_eval_with_scene").glob("*"),
            *(root / "fresh_eval").glob("*"),
        ]
        for path in candidates:
            if path.is_dir() and ((path / "pattern_metrics.csv").exists() or (path / "apples_to_apples_metrics.csv").exists()):
                score = (
                    int((path / "predictions_by_pattern.csv").exists()),
                    int(path.parent.name in {"fresh_eval_with_scene", "fresh_eval_maskfix_with_scene"}),
                )
                if path.name not in by_run or score >= scores[path.name]:
                    by_run[path.name] = path
                    scores[path.name] = score
    return [by_run[name] for name in sorted(by_run)]


def _load_eval_rows(eval_dirs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    pattern_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for eval_dir in eval_dirs:
        metrics_path = eval_dir / "pattern_metrics.csv"
        if not metrics_path.exists():
            metrics_path = eval_dir / "apples_to_apples_metrics.csv"
        run_name = eval_dir.name
        mask_meta = _mask_meta(eval_dir)
        if metrics_path.exists():
            rows = _read_csv(metrics_path)
            for row in rows:
                pattern = _pattern(row.get("pattern"))
                if pattern == "avg_missing":
                    continue
                enriched = _enrich_pattern_row(row, run_name, eval_dir, metrics_path, mask_meta)
                pattern_rows.append(enriched)
        else:
            warnings.append(f"missing metrics for {eval_dir}")
        predictions_path = eval_dir / "predictions_by_pattern.csv"
        if predictions_path.exists():
            for row in _read_csv(predictions_path):
                prediction_rows.append(_enrich_prediction_row(row, run_name, eval_dir, predictions_path, mask_meta))
    return pattern_rows, prediction_rows, warnings


def _enrich_pattern_row(
    row: dict[str, Any],
    run_name: str,
    eval_dir: Path,
    metrics_path: Path,
    mask_meta: dict[str, Any],
) -> dict[str, Any]:
    run_name = str(row.get("run_name") or run_name)
    method = str(row.get("method") or _method_name(run_name))
    seed = row.get("seed") if row.get("seed") not in (None, "") else _seed(run_name)
    pattern = _pattern(row.get("pattern"))
    missing_count, missing_ratio, available, missing = _pattern_metadata(pattern, row)
    maskfix_eval = _truthy(row.get("maskfix_eval")) or bool(mask_meta.get("maskfix_eval"))
    mask_suspect = _truthy(row.get("mask_suspect")) or bool(mask_meta.get("mask_suspect"))
    return {
        **row,
        "source_root": str(_source_root(eval_dir)),
        "eval_dir": str(eval_dir),
        "metrics_path": str(metrics_path),
        "run_name": run_name,
        "method": method,
        "seed": seed,
        "pattern": pattern,
        "missing_count": missing_count,
        "missing_ratio": missing_ratio,
        "available_modalities": row.get("available_modalities") or ",".join(available),
        "missing_modalities": row.get("missing_modalities") or ",".join(missing),
        "top1": _metric_value(row, "top1", "top1_correct"),
        "top3": _metric_value(row, "top3"),
        "top5": _metric_value(row, "top5"),
        "within3": _metric_value(row, "within3", "within_3", "within@3", "within3_correct"),
        "mae": _metric_value(row, "mae", "abs_error"),
        "num_samples": _metric_value(row, "num_samples", "count", "sample_count"),
        "maskfix_eval": str(maskfix_eval).lower(),
        "mask_suspect": str(mask_suspect).lower(),
        "excluded_from_official_ranking": str(mask_suspect).lower(),
    }


def _enrich_prediction_row(
    row: dict[str, Any],
    run_name: str,
    eval_dir: Path,
    path: Path,
    mask_meta: dict[str, Any],
) -> dict[str, Any]:
    run_name = str(row.get("run_name") or run_name)
    pattern = _pattern(row.get("pattern"))
    missing_count, missing_ratio, available, missing = _pattern_metadata(pattern, row)
    maskfix_eval = _truthy(row.get("maskfix_eval")) or bool(mask_meta.get("maskfix_eval"))
    mask_suspect = _truthy(row.get("mask_suspect")) or bool(mask_meta.get("mask_suspect"))
    return {
        **row,
        "source_root": str(_source_root(eval_dir)),
        "eval_dir": str(eval_dir),
        "predictions_path": str(path),
        "run_name": run_name,
        "method": row.get("method") or _method_name(run_name),
        "seed": row.get("seed") if row.get("seed") not in (None, "") else _seed(run_name),
        "scene": _scene_label(row.get("scene") or row.get("sample_id")),
        "pattern": pattern,
        "missing_count": missing_count,
        "missing_ratio": missing_ratio,
        "available_modalities": row.get("available_modalities") or ",".join(available),
        "missing_modalities": row.get("missing_modalities") or ",".join(missing),
        "top1": _metric_value(row, "top1_correct", "top1"),
        "within3": _metric_value(row, "within3_correct", "within3", "within_3", "within@3"),
        "mae": _metric_value(row, "abs_error", "mae"),
        "maskfix_eval": str(maskfix_eval).lower(),
        "mask_suspect": str(mask_suspect).lower(),
        "excluded_from_official_ranking": str(mask_suspect).lower(),
    }


def _per_run_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("run_name") or "")].append(row)
    out: list[dict[str, Any]] = []
    for run_name, items in grouped.items():
        by_pattern = {str(row.get("pattern")): row for row in items}
        row = _summary_row(items, scene="")
        row.update(
            {
                "source_root": items[0].get("source_root", ""),
                "eval_dir": items[0].get("eval_dir", ""),
                "run_name": run_name,
                "method": items[0].get("method") or _method_name(run_name),
                "seed": items[0].get("seed") or _seed(run_name),
                "status": _status(items),
                "checkpoint_used": items[0].get("checkpoint_used", ""),
                "max_batches": items[0].get("max_batches", ""),
                "full_top1": _float_or_blank(by_pattern.get("full", {}).get("top1")),
            }
        )
        method = str(row.get("method") or "")
        mask_suspect = any(_truthy(item.get("mask_suspect")) for item in items)
        row["family"] = _family(method)
        row["maskfix_eval"] = str(any(_truthy(item.get("maskfix_eval")) for item in items)).lower()
        row["mask_suspect"] = str(mask_suspect).lower()
        row["excluded_from_official_ranking"] = str(mask_suspect).lower()
        row["official_ranking_included"] = str(_official_included(method, mask_suspect, row.get("status"))).lower()
        row["main_read"] = _main_read(method, bool(_truthy(row["official_ranking_included"])))
        if not _truthy(row["official_ranking_included"]):
            row["status"] = "excluded" if row.get("status") in {"", "ok"} else row.get("status")
        out.append(row)
    return sorted(out, key=lambda row: (_method_rank(str(row.get("method"))), _int(row.get("seed")), str(row.get("run_name"))))


def _per_scene_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene = _scene_label(row.get("scene") or row.get("sample_id"))
        run_name = str(row.get("run_name") or "")
        if scene and run_name:
            grouped[(scene, run_name)].append(row)
    out: list[dict[str, Any]] = []
    for (scene, run_name), items in grouped.items():
        row = _summary_row(items, scene=scene)
        row.update(
            {
                "scene": scene,
                "run_name": run_name,
                "method": items[0].get("method") or _method_name(run_name),
                "seed": items[0].get("seed") or _seed(run_name),
            }
        )
        out.append(row)
    return sorted(out, key=lambda row: (_scene_rank(str(row.get("scene"))), _method_rank(str(row.get("method"))), _int(row.get("seed"))))


def _summary_row(items: list[dict[str, Any]], *, scene: str) -> dict[str, Any]:
    by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in items:
        by_pattern[str(row.get("pattern") or "")].append(row)
    pattern_metrics = {
        pattern: {
            "top1": _mean(_values(rows, "top1")),
            "within3": _mean(_values(rows, "within3")),
            "mae": _mean(_values(rows, "mae")),
            "missing_count": _missing_count(rows[0]),
            "num_samples": _sample_count(rows),
        }
        for pattern, rows in by_pattern.items()
        if pattern
    }
    top1 = {pattern: values["top1"] for pattern, values in pattern_metrics.items()}
    within3 = {pattern: values["within3"] for pattern, values in pattern_metrics.items()}
    mae = {pattern: values["mae"] for pattern, values in pattern_metrics.items()}
    out = {
        "scene": scene,
        "full_top1": top1.get("full", math.nan),
        "miss1_top1": _bucket_mean(top1, pattern_metrics, 1),
        "miss2_top1": _bucket_mean(top1, pattern_metrics, 2),
        "miss3_top1": _bucket_mean(top1, pattern_metrics, 3),
        "avg_missing_top1": _avg_missing(top1),
        "overall_mean_top1": _overall_mean(top1),
        "avg_missing_within@3": _avg_missing(within3),
        "avg_missing_MAE": _avg_missing(mae),
        "num_patterns": len(pattern_metrics),
        "num_samples": sum(int(values.get("num_samples") or 0) for values in pattern_metrics.values()),
    }
    out["balanced"] = _balanced(out, top1)
    return out


def _method_rows(rows: list[dict[str, Any]], *, official_only: bool = True) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        status_ok = str(row.get("status") or "ok") in {"", "ok"}
        official_ok = _truthy(row.get("official_ranking_included"))
        if status_ok and (official_ok or not official_only):
            grouped[str(row.get("method") or "")].append(row)
    out = [_aggregate_rows(method, items) for method, items in grouped.items()]
    return sorted(out, key=lambda row: (_zero(row.get("avg_missing_top1_mean")), _zero(row.get("overall_mean_top1_mean"))), reverse=True)


def _per_scene_method_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("scene") or ""), str(row.get("method") or ""))].append(row)
    out = []
    for (scene, method), items in grouped.items():
        item = _aggregate_rows(method, items)
        item["scene"] = scene
        out.append(item)
    return sorted(out, key=lambda row: (_scene_rank(str(row.get("scene"))), -_zero(row.get("avg_missing_top1_mean"))))


def _aggregate_rows(method: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "full_top1",
        "miss1_top1",
        "miss2_top1",
        "miss3_top1",
        "avg_missing_top1",
        "overall_mean_top1",
        "avg_missing_within@3",
        "avg_missing_MAE",
        "balanced",
    ]
    out: dict[str, Any] = {"method": method, "n": len({str(row.get("seed") or row.get("run_name")) for row in items})}
    for metric in metrics:
        values = [_float(row.get(metric)) for row in items if _isnum(row.get(metric))]
        out[f"{metric}_mean"] = mean(values) if values else math.nan
        out[f"{metric}_std"] = stdev(values) if len(values) > 1 else math.nan
    return out


def _final_method_rows(per_run: list[dict[str, Any]], curve_by_run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_run:
        if str(row.get("status") or "ok") in {"", "ok", "excluded"}:
            grouped[str(row.get("method") or "")].append(row)
    curve_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in curve_by_run:
        curve_grouped[str(row.get("method") or "")].append(row)
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        items = grouped.get(method, [])
        official_items = [row for row in items if _truthy(row.get("official_ranking_included")) and str(row.get("status") or "ok") in {"", "ok"}]
        aggregate = _aggregate_rows(method, official_items) if official_items else _empty_method_row(method)
        aggregate["family"] = _family(method)
        aggregate["mask_suspect_count"] = len({row.get("run_name") for row in items if _truthy(row.get("mask_suspect"))})
        aggregate["official_ranking_included"] = str(bool(official_items)).lower()
        claim_status, caveat = _final_claim_status(method, items, official_items)
        aggregate["claim_status"] = claim_status
        aggregate["caveat"] = caveat
        aggregate["main_read"] = _main_read(method, bool(official_items), visible=bool(items))
        if caveat:
            aggregate["main_read"] = f"{aggregate['main_read']}; {caveat}"
        aggregate.update(_drop_summary(curve_grouped.get(method, [])))
        rows.append(aggregate)
    return sorted(rows, key=lambda row: _method_rank(str(row.get("method") or "")))


def _empty_method_row(method: str) -> dict[str, Any]:
    row: dict[str, Any] = {"method": method, "n": 0}
    for field in METHOD_FIELDS:
        if field not in row and field != "method":
            row[field] = math.nan if field.endswith("_mean") or field.endswith("_std") else ""
    return row


def _drop_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed_count = {(str(row.get("seed") or row.get("run_name")), _int(row.get("missing_count"))): row for row in rows}
    seeds = sorted({seed for seed, _ in by_seed_count})
    top1_drops: list[float] = []
    mae75: list[float] = []
    for seed in seeds:
        full = _float(by_seed_count.get((seed, 0), {}).get("top1"))
        miss3 = _float(by_seed_count.get((seed, 3), {}).get("top1"))
        if math.isfinite(full) and math.isfinite(miss3):
            top1_drops.append(full - miss3)
        value = _float(by_seed_count.get((seed, 3), {}).get("mae"))
        if math.isfinite(value):
            mae75.append(value)
    return {
        "top1_drop_0_to_75_mean": _mean(top1_drops),
        "top1_drop_0_to_75_std": _std(top1_drops),
        "mae_at_75_mean": _mean(mae75),
        "mae_at_75_std": _std(mae75),
    }


def _baseline_rows(rows: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    by_method = {str(row.get("method") or ""): row for row in rows}
    return [by_method.get(method, _empty_final_row(method)) for method in methods]


def _empty_final_row(method: str) -> dict[str, Any]:
    row = _empty_method_row(method)
    row.update(
        {
            "family": _family(method),
            "top1_drop_0_to_75_mean": math.nan,
            "top1_drop_0_to_75_std": math.nan,
            "mae_at_75_mean": math.nan,
            "mae_at_75_std": math.nan,
            "mask_suspect_count": 0,
            "official_ranking_included": "false",
            "claim_status": "pending",
            "caveat": "not run",
            "main_read": "not run",
        }
    )
    return row


def _final_claim_status(
    method: str,
    items: list[dict[str, Any]],
    official_items: list[dict[str, Any]],
) -> tuple[str, str]:
    required = _required_seed_count(method)
    visible = bool(items)
    official_n = len({str(row.get("seed") or row.get("run_name")) for row in official_items})
    if not visible:
        return "pending", f"not run; needs n>={required}"
    if not official_items:
        if any(_truthy(row.get("mask_suspect")) for row in items):
            return "incomplete", "mask_suspect external row excluded from official ranking"
        return "pending", "no official eligible row"
    if official_n < required:
        return "pending", f"needs n>={required}; current n={official_n}"
    return "complete", ""


def _required_seed_count(method: str) -> int:
    if method in CORE_METHODS:
        return 5
    if method in CLASSIFIER_METHODS:
        return 3
    if method in EXTERNAL_METHODS:
        return 1
    return 1


def _final_evidence_checklist(
    final_method_rows: list[dict[str, Any]],
    *,
    per_run: list[dict[str, Any]],
    per_scene_run: list[dict[str, Any]],
    curve: list[dict[str, Any]],
    out_dir: Path,
) -> list[dict[str, str]]:
    by_method = {str(row.get("method") or ""): row for row in final_method_rows}
    rows: list[dict[str, str]] = []
    rows.append(
        _checklist_row(
            "core proto n=5",
            all(_int(by_method.get(method, {}).get("n")) >= 5 and _truthy(by_method.get(method, {}).get("official_ranking_included")) for method in CORE_METHODS),
            required="all core methods n>=5",
            observed=", ".join(f"{method}:n={_int(by_method.get(method, {}).get('n'))}" for method in CORE_METHODS),
            caveat="core proto claim remains pending until every core method reaches n=5",
            next_action="run core_seed23/core_seed45/core_all_missing then summarize_final_all",
        )
    )
    rows.append(
        _checklist_row(
            "ordinary classifier baseline",
            all(_int(by_method.get(method, {}).get("n")) >= 3 and _truthy(by_method.get(method, {}).get("official_ranking_included")) for method in CLASSIFIER_METHODS),
            required="classifier natural/subset n>=3",
            observed=", ".join(f"{method}:n={_int(by_method.get(method, {}).get('n'))}" for method in CLASSIFIER_METHODS),
            caveat="prototype-vs-classifier conclusion is incomplete until classifier rows exist",
            next_action="run classifier_seed123 and eval_all_baselines",
        )
    )
    external_complete = all(
        _int(by_method.get(method, {}).get("n")) >= 1 and _truthy(by_method.get(method, {}).get("official_ranking_included"))
        for method in EXTERNAL_METHODS
    )
    external_suspect = any(_int(by_method.get(method, {}).get("mask_suspect_count")) > 0 for method in EXTERNAL_METHODS)
    rows.append(
        _checklist_row(
            "AMR/AMBER-lite external maskfix",
            external_complete,
            required="all AMR/AMBER-lite rows n>=1 and mask_suspect=false",
            observed=", ".join(
                f"{method}:n={_int(by_method.get(method, {}).get('n'))},suspect={_int(by_method.get(method, {}).get('mask_suspect_count'))}"
                for method in EXTERNAL_METHODS
            ),
            caveat="mask_suspect rows are excluded" if external_suspect else "external-lite rows remain optional but incomplete when absent",
            next_action="run external_lite_seed1 or external_lite_seed123 with maskfix eval",
            forced_status="incomplete" if external_suspect and not external_complete else None,
        )
    )
    rows.append(
        _checklist_row(
            "fresh eval",
            bool(per_run),
            required="fresh eval rows with best checkpoint and no max_batches",
            observed=f"{len(per_run)} visible per-run rows",
            caveat="summary is pending if fresh eval rows are absent",
            next_action="run eval_core_all/eval_all_baselines",
        )
    )
    curve_methods = {str(row.get("method")) for row in curve}
    curve_counts = {
        method: sorted({_int(row.get("missing_count")) for row in curve if row.get("method") == method and _isnum(row.get("top1_mean"))})
        for method in curve_methods
    }
    rows.append(
        _checklist_row(
            "missing-count degradation curve",
            all(curve_counts.get(method) == [0, 1, 2, 3] for method in CORE_METHODS),
            required="missing_count 0/1/2/3 for each core method",
            observed=", ".join(f"{method}:{curve_counts.get(method, [])}" for method in CORE_METHODS),
            caveat="paper curve is incomplete if any bucket is missing",
            next_action="rerun fresh eval with predictions_by_pattern/pattern_metrics",
        )
    )
    scenes = sorted({str(row.get("scene") or "") for row in per_scene_run if row.get("scene")}, key=_scene_rank)
    rows.append(
        _checklist_row(
            "per-scene stability",
            set(SCENES) <= set(scenes),
            required="Scene31, Scene32, Scene33 and Scene34 visible",
            observed=",".join(scenes) if scenes else "none",
            caveat="scene stability ranking is incomplete if any scene is absent",
            next_action="run fresh eval with scene metadata",
        )
    )
    profile_path = out_dir.parent / "profile" / "method_profile_summary.csv"
    rows.append(
        _checklist_row(
            "compute profile",
            profile_path.exists(),
            required=str(profile_path),
            observed="present" if profile_path.exists() else "missing",
            caveat="paper table must mark compute cost pending until profile exists",
            next_action="run python -m kd_sensing.diagnostics.scene31_34_final_analysis --artifact profile",
        )
    )
    paper_table_path = Path("outputs/paper_tables/scenes31_34_main/table_scenes31_34_main.md")
    rows.append(
        _checklist_row(
            "paper tables",
            paper_table_path.exists(),
            required=str(paper_table_path),
            observed="present" if paper_table_path.exists() else "missing",
            caveat="paper export is pending until table export has been run",
            next_action="run python -m kd_sensing.diagnostics.scene31_34_final_analysis --artifact paper-tables",
        )
    )
    rows.append(
        _checklist_row(
            "final conclusion artifact",
            True,
            required=str(out_dir / "scenes31_34_main_conclusion.txt"),
            observed="written by this summary run",
            caveat="final_main_conclusion.txt should be regenerated after paper tables/profile update",
            next_action="run python -m kd_sensing.diagnostics.scene31_34_final_analysis --artifact conclusion",
        )
    )
    return rows


def _checklist_row(
    item: str,
    complete: bool,
    *,
    required: str,
    observed: str,
    caveat: str,
    next_action: str,
    forced_status: str | None = None,
) -> dict[str, str]:
    return {
        "item": item,
        "status": forced_status or ("complete" if complete else "pending"),
        "required": required,
        "observed": observed,
        "caveat": "" if complete and forced_status is None else caveat,
        "next_action": "" if complete and forced_status is None else next_action,
    }


def _mean_over_scenes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("method") or "")].append(row)
    out: list[dict[str, Any]] = []
    for method, items in grouped.items():
        item: dict[str, Any] = {"method": method, "n": len(items)}
        for metric in ("avg_missing_top1", "full_top1", "miss1_top1", "miss2_top1", "miss3_top1"):
            values = [_float(row.get(f"{metric}_mean")) for row in items if _isnum(row.get(f"{metric}_mean"))]
            item[f"{metric}_mean_over_scenes"] = mean(values) if values else math.nan
            item[f"{metric}_std_over_scenes"] = stdev(values) if len(values) > 1 else math.nan
        for metric in ("avg_missing_within@3", "avg_missing_MAE", "balanced"):
            values = [_float(row.get(f"{metric}_mean")) for row in items if _isnum(row.get(f"{metric}_mean"))]
            item[f"{metric}_mean_over_scenes"] = mean(values) if values else math.nan
        out.append(item)
    return sorted(out, key=lambda row: (_zero(row.get("avg_missing_top1_mean_over_scenes")), -_large(row.get("avg_missing_top1_std_over_scenes"))), reverse=True)


def _missing_count_curve_by_run(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        count = _missing_count(row)
        if count is not None:
            grouped[(str(row.get("run_name") or ""), count)].append(row)
    out: list[dict[str, Any]] = []
    for (run_name, count), items in grouped.items():
        out.append(
            {
                "run_name": run_name,
                "method": items[0].get("method") or _method_name(run_name),
                "seed": items[0].get("seed") or _seed(run_name),
                "missing_count": count,
                "missing_ratio": count / len(MODALITIES),
                "top1": _mean(_values(items, "top1")),
                "within3": _mean(_values(items, "within3")),
                "mae": _mean(_values(items, "mae")),
                "num_patterns": len({str(row.get("pattern") or "") for row in items}),
                "num_samples": sum(_int(row.get("num_samples")) for row in items),
            }
        )
    return out


def _missing_count_curve(rows: list[dict[str, Any]], warnings: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    methods = sorted({str(row.get("method") or "") for row in rows if row.get("method")}, key=_method_rank)
    methods = [method for method in methods if method]
    for row in rows:
        grouped[(str(row.get("method") or ""), int(row.get("missing_count")))].append(row)
    out: list[dict[str, Any]] = []
    for method in methods:
        for count in range(4):
            items = grouped.get((method, count), [])
            if not items:
                warnings.append(f"missing_count={count} has no patterns for {method}")
            out.append(
                {
                    "method": method,
                    "n": len({str(row.get("seed") or row.get("run_name")) for row in items}),
                    "missing_count": count,
                    "missing_ratio": count / len(MODALITIES),
                    "top1_mean": _mean(_values(items, "top1")),
                    "top1_std": _std(_values(items, "top1")),
                    "within3_mean": _mean(_values(items, "within3")),
                    "within3_std": _std(_values(items, "within3")),
                    "mae_mean": _mean(_values(items, "mae")),
                    "mae_std": _std(_values(items, "mae")),
                    "num_patterns": sum(_int(row.get("num_patterns")) for row in items),
                    "num_samples": sum(_int(row.get("num_samples")) for row in items),
                }
            )
    return out


def _missing_count_curve_by_scene(rows: list[dict[str, Any]], warnings: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene = _scene_label(row.get("scene") or row.get("sample_id"))
        count = _missing_count(row)
        if scene and count is not None:
            grouped[(scene, str(row.get("method") or ""), str(row.get("seed") or ""), count)].append(row)
    out: list[dict[str, Any]] = []
    for (scene, method, seed, count), items in grouped.items():
        out.append(
            {
                "scene": scene,
                "method": method,
                "seed": seed,
                "missing_count": count,
                "missing_ratio": count / len(MODALITIES),
                "top1": _mean(_values(items, "top1")),
                "within3": _mean(_values(items, "within3")),
                "mae": _mean(_values(items, "mae")),
                "num_patterns": len({str(row.get("pattern") or "") for row in items}),
                "num_samples": len(items),
            }
        )
    if not out and rows:
        warnings.append("prediction rows were present but no scene-level missing_count rows could be built")
    return sorted(out, key=lambda row: (_scene_rank(str(row.get("scene"))), _method_rank(str(row.get("method"))), _int(row.get("seed")), _int(row.get("missing_count"))))


def _delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ref = next((row for row in rows if row.get("method") == REFERENCE_METHOD), None)
    out = []
    for row in rows:
        item = {"method": row.get("method", ""), "n": row.get("n", ""), "reference_method": REFERENCE_METHOD}
        for metric in (
            "full_top1",
            "miss1_top1",
            "miss2_top1",
            "miss3_top1",
            "avg_missing_top1",
            "overall_mean_top1",
            "avg_missing_within@3",
            "avg_missing_MAE",
            "balanced",
        ):
            item[f"delta_{metric}"] = _delta(row.get(f"{metric}_mean"), ref.get(f"{metric}_mean") if ref else math.nan)
        out.append(item)
    return out


def _fallback_per_scene_rows(roots: list[Path]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for root in roots:
        path = root / "per_scene_summary" / "per_scene_per_run.csv"
        if not path.exists():
            continue
        for row in _read_csv(path):
            run_name = str(row.get("run_name") or "")
            scene = _scene_label(row.get("scene"))
            if run_name and scene:
                row["scene"] = scene
                row["method"] = row.get("method") or _method_name(run_name)
                row["seed"] = row.get("seed") or _seed(run_name)
                by_key[(scene, run_name)] = row
    return sorted(by_key.values(), key=lambda row: (_scene_rank(str(row.get("scene"))), _method_rank(str(row.get("method"))), _int(row.get("seed"))))


def _conclusion_lines(
    per_run: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    mean_over_scenes: list[dict[str, Any]],
    curve: list[dict[str, Any]],
    warnings: list[str],
) -> list[str]:
    winner = method_rows[0].get("method", "unavailable") if method_rows else "unavailable"
    subset = next((row for row in method_rows if row.get("method") == REFERENCE_METHOD), None)
    subset_wins = winner == REFERENCE_METHOD
    subset_n = _int(subset.get("n") if subset else "")
    subset_status = "yes_after_seed123" if subset_wins and subset_n >= 3 else f"pending_seed123_current_visible_winner_{'yes' if subset_wins else 'no'}"
    drops = _drop_lines(curve)
    lines = [
        "Scene31-34 main missing-modality conclusion:",
        "- Scene31-34 is now treated as the main evaluation setting.",
        f"- visible ok runs: {len([row for row in per_run if row.get('status') in {'', 'ok'}])}",
        f"- current Avg-Missing winner: {winner}",
        f"- proto_randomdrop_subset_es40 final trusted method: {subset_status}",
        f"- proto_randomdrop_subset_es40 Avg-Missing Top1: {_fmt(subset.get('avg_missing_top1_mean') if subset else math.nan)}",
        "- Uniform pattern exposure is an ablation, not the final reference.",
        "- Reliability fusion and PatternFiLM are not promoted.",
        "- Do not continue module search.",
        "- Optional next step: add AMR/AMBER multi-scene maskfix baselines only if the paper needs external baselines.",
    ]
    if mean_over_scenes:
        lines.append(f"- scene-stability winner: {mean_over_scenes[0].get('method', 'unavailable')}")
    if drops:
        lines.extend(["", "Top1 drop 0%->75% missing:", *drops])
    if warnings:
        lines.extend(["", "Warnings:", *[f"- {warning}" for warning in sorted(set(warnings))]])
    return lines


def _drop_lines(curve: list[dict[str, Any]]) -> list[str]:
    by_method_count = {(str(row.get("method")), int(row.get("missing_count"))): row for row in curve}
    lines = []
    for method in sorted({key[0] for key in by_method_count}, key=_method_rank):
        v0 = _float(by_method_count.get((method, 0), {}).get("top1_mean"))
        v3 = _float(by_method_count.get((method, 3), {}).get("top1_mean"))
        if _isnum(v0) and _isnum(v3):
            lines.append(f"- {method}: {_fmt(v0 - v3)}")
    return lines


def _write_rank(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["method", "n", "full_top1", "miss1_top1", "miss2_top1", "miss3_top1", "avg_missing_top1", "overall_mean_top1", "avg_missing_MAE"]
    lines = ["# Rank By Avg Missing Top1", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append(
            "| {method} | {n} | {full} | {miss1} | {miss2} | {miss3} | {avg} | {overall} | {mae} |".format(
                method=row.get("method", ""),
                n=row.get("n", ""),
                full=_mean_std(row, "full_top1"),
                miss1=_mean_std(row, "miss1_top1"),
                miss2=_mean_std(row, "miss2_top1"),
                miss3=_mean_std(row, "miss3_top1"),
                avg=_mean_std(row, "avg_missing_top1"),
                overall=_mean_std(row, "overall_mean_top1"),
                mae=_mean_std(row, "avg_missing_MAE"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stability_rank(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["method", "n", "avg_missing_top1_mean_over_scenes", "avg_missing_top1_std_over_scenes", "avg_missing_MAE_mean_over_scenes"]
    lines = ["# Rank By Scene Stability", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) if column in {"method", "n"} else _fmt(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _write_checklist_md(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = EVIDENCE_CHECKLIST_FIELDS
    lines = ["# Scene31-34 Final Evidence Checklist", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pattern(value: Any) -> str:
    return str(value or "").strip()


def _pattern_metadata(pattern: str, row: dict[str, Any]) -> tuple[int | None, float, list[str], list[str]]:
    raw_count = _float(row.get("missing_count"))
    available = _split_modalities(row.get("available_modalities"))
    missing = _split_modalities(row.get("missing_modalities"))
    if math.isfinite(raw_count):
        count = int(raw_count)
        return count, count / len(MODALITIES), available, missing
    if pattern == "full":
        return 0, 0.0, list(MODALITIES), []
    if pattern == "avg_missing" or not pattern:
        return None, math.nan, available, missing
    if pattern.startswith("missing_"):
        missing = [item for item in pattern.removeprefix("missing_").split("_") if item]
        available = [item for item in MODALITIES if item not in missing]
        count = len(missing)
        return count, count / len(MODALITIES), available, missing
    if pattern == "non_gps_only":
        available = [item for item in MODALITIES if item != "gps"]
        return 1, 1 / len(MODALITIES), available, ["gps"]
    if pattern.endswith("_only"):
        available = [item for item in pattern.removesuffix("_only").split("_") if item]
        missing = [item for item in MODALITIES if item not in available]
        count = len(missing)
        return count, count / len(MODALITIES), available, missing
    return None, math.nan, available, missing


def _split_modalities(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _metric_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def _source_root(eval_dir: Path) -> Path:
    for parent in eval_dir.parents:
        if parent.name in {"fresh_eval_with_scene", "fresh_eval", "fresh_eval_maskfix_with_scene", "fresh_eval_maskfix"}:
            return parent.parent
    return eval_dir.parent


def _mask_meta(eval_dir: Path) -> dict[str, Any]:
    path = eval_dir / "mask_suspect.json"
    if not path.exists():
        return {"maskfix_eval": "maskfix" in eval_dir.parent.name, "mask_suspect": False}
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"maskfix_eval": "maskfix" in eval_dir.parent.name, "mask_suspect": False}
    return {
        "maskfix_eval": bool(payload.get("maskfix_eval")),
        "mask_suspect": bool(payload.get("mask_suspect")),
        "excluded_from_official_ranking": bool(payload.get("excluded_from_official_ranking", payload.get("mask_suspect"))),
    }


def _family(method: str) -> str:
    if method in CORE_METHODS:
        return "proto"
    if method in CLASSIFIER_METHODS:
        return "classifier"
    if method in EXTERNAL_METHODS:
        return "external_lite"
    return "auxiliary"


def _official_included(method: str, mask_suspect: bool, status: Any) -> bool:
    if method not in METHOD_ORDER:
        return False
    if mask_suspect:
        return False
    return str(status or "ok") in {"", "ok"}


def _main_read(method: str, official: bool, *, visible: bool = True) -> str:
    if not visible:
        return "not run"
    if not official:
        return "excluded"
    if method == REFERENCE_METHOD:
        return "final trusted method candidate"
    if method in CORE_METHODS:
        return "core proto baseline"
    if method in CLASSIFIER_METHODS:
        return "classifier baseline"
    if method in EXTERNAL_METHODS:
        return "external lite baseline"
    return "auxiliary"


def _scene_label(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    match = re.search(r"(?:Scene|scene|scenario)(\d+)", text)
    if match:
        return f"Scene{int(match.group(1))}"
    if _isnum(text):
        return f"Scene{int(float(text))}"
    return text


def _method_name(run_name: str) -> str:
    return re.sub(r"_seed\d+$", "", str(run_name))


def _seed(run_name: str) -> int | str:
    match = re.search(r"_seed(\d+)$", str(run_name))
    return int(match.group(1)) if match else ""


def _status(rows: list[dict[str, Any]]) -> str:
    values = {str(row.get("status") or "ok") for row in rows}
    values.discard("")
    return "ok" if not values or values == {"ok"} else ";".join(sorted(values))


def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [_float(row.get(key)) for row in rows if _isnum(row.get(key))]


def _missing_count(row: dict[str, Any]) -> int | None:
    value = _float(row.get("missing_count"))
    return int(value) if math.isfinite(value) else None


def _sample_count(rows: list[dict[str, Any]]) -> int:
    values = [_int(row.get("num_samples")) for row in rows if _int(row.get("num_samples")) > 0]
    return sum(values) if values else len(rows)


def _bucket_mean(values: dict[str, float], pattern_metrics: dict[str, dict[str, Any]], count: int) -> float:
    nums = [
        value
        for pattern, value in values.items()
        if pattern not in {"full", "avg_missing"} and int(pattern_metrics.get(pattern, {}).get("missing_count", -1)) == count and math.isfinite(value)
    ]
    return mean(nums) if nums else math.nan


def _avg_missing(values: dict[str, float]) -> float:
    nums = [value for pattern, value in values.items() if pattern not in {"full", "avg_missing"} and math.isfinite(value)]
    return mean(nums) if nums else math.nan


def _overall_mean(values: dict[str, float]) -> float:
    nums = [_float(values.get(pattern)) for pattern in CORE_PATTERNS]
    return mean(nums) if all(math.isfinite(value) for value in nums) else math.nan


def _balanced(row: dict[str, Any], top1: dict[str, float]) -> float:
    avg = _float(row.get("avg_missing_top1"))
    if not math.isfinite(avg):
        return math.nan
    radar = _float(top1.get("radar_only"))
    lidar = _float(top1.get("lidar_only"))
    return avg + 0.25 * (radar if math.isfinite(radar) else 0.0) + 0.25 * (lidar if math.isfinite(lidar) else 0.0)


def _delta(value: Any, base: Any) -> float:
    left = _float(value)
    right = _float(base)
    return left - right if math.isfinite(left) and math.isfinite(right) else math.nan


def _mean(values: list[float]) -> float:
    nums = [value for value in values if math.isfinite(value)]
    return mean(nums) if nums else math.nan


def _std(values: list[float]) -> float:
    nums = [value for value in values if math.isfinite(value)]
    return stdev(nums) if len(nums) > 1 else math.nan


def _float_or_blank(value: Any) -> float | str:
    number = _float(value)
    return number if math.isfinite(number) else ""


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _isnum(value: Any) -> bool:
    return math.isfinite(_float(value))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _zero(value: Any) -> float:
    value = _float(value)
    return value if math.isfinite(value) else -math.inf


def _large(value: Any) -> float:
    value = _float(value)
    return value if math.isfinite(value) else math.inf


def _int(value: Any) -> int:
    number = _float(value)
    return int(number) if math.isfinite(number) else 0


def _fmt(value: Any) -> str:
    number = _float(value)
    return f"{number:.5f}" if math.isfinite(number) else ""


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return _fmt(value)
    return value


def _mean_std(row: dict[str, Any], metric: str) -> str:
    value = _float(row.get(f"{metric}_mean"))
    std = _float(row.get(f"{metric}_std"))
    if not math.isfinite(value):
        return ""
    return f"{_fmt(value)}+-{_fmt(std)}" if math.isfinite(std) else _fmt(value)


def _method_rank(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)


def _scene_rank(scene: str) -> int:
    try:
        return SCENES.index(scene)
    except ValueError:
        return len(SCENES)


if __name__ == "__main__":
    raise SystemExit(main())
