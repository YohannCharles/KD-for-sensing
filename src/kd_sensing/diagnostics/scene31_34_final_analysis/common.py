#!/usr/bin/env python3

import csv
import math
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REFERENCE_METHOD = "scenes31_34_proto_randomdrop_subset_es40"
BERNOULLI_METHOD = "scenes31_34_proto_randomdrop_bernoulli_k075_es40"
CLASSIFIER_SUBSET_METHOD = "scenes31_34_classifier_randomdrop_subset_es40"
PROTO_NATURAL_METHOD = "scenes31_34_proto_natural_es40"
PROTO_UNIFORM_METHOD = "scenes31_34_proto_sampler_uniform_es40"

METHOD_LABELS = {
    PROTO_NATURAL_METHOD: "Proto natural",
    PROTO_UNIFORM_METHOD: "Proto uniform pattern exposure",
    BERNOULLI_METHOD: "Proto Bernoulli randomdrop",
    REFERENCE_METHOD: "Proto random subset exposure",
    "scenes31_34_classifier_natural_es40": "Classifier natural",
    CLASSIFIER_SUBSET_METHOD: "Classifier random subset",
    "scenes31_34_amr_lite_natural_es40": "AMR-lite natural",
    "scenes31_34_amber_lite_natural_es40": "AMBER-lite natural",
    "scenes31_34_amr_lite_uniform_es40": "AMR-lite uniform",
    "scenes31_34_amber_lite_uniform_es40": "AMBER-lite uniform",
}
METHOD_ORDER = tuple(METHOD_LABELS)
CORE_METHODS = (PROTO_NATURAL_METHOD, PROTO_UNIFORM_METHOD, BERNOULLI_METHOD, REFERENCE_METHOD)
DEFAULT_ANALYSIS_METHODS = (
    PROTO_NATURAL_METHOD,
    PROTO_UNIFORM_METHOD,
    BERNOULLI_METHOD,
    REFERENCE_METHOD,
    CLASSIFIER_SUBSET_METHOD,
)
MODALITIES = ("image", "radar", "gps", "lidar")


def roots_from_args(*groups: Iterable[str | Path]) -> list[Path]:
    seen: set[str] = set()
    roots: list[Path] = []
    for group in groups:
        for item in group:
            path = Path(item)
            key = str(path)
            if key and key not in seen:
                roots.append(path)
                seen.add(key)
    return roots


def method_label(method: str) -> str:
    return METHOD_LABELS.get(str(method), str(method))


def family(method: str) -> str:
    method = str(method)
    if "classifier" in method:
        return "classifier"
    if "amr_lite" in method or "amber_lite" in method:
        return "external_lite"
    if method.startswith("scenes31_34_proto"):
        return "proto"
    return "auxiliary"


def method_rank(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)


def method_from_run(run_name: str) -> str:
    return re.sub(r"_seed\d+$", "", str(run_name))


def seed_from_run(run_name: str) -> int:
    match = re.search(r"_seed(\d+)$", str(run_name))
    return int(match.group(1)) if match else 0


def find_eval_dirs(roots: Iterable[Path]) -> list[Path]:
    by_run: dict[str, Path] = {}
    scores: dict[str, tuple[int, int, int]] = {}
    parents = (
        "fresh_eval_maskfix_with_scene",
        "fresh_eval_with_scene",
        "fresh_eval_maskfix",
        "fresh_eval",
    )
    for root in roots:
        if not root.exists():
            continue
        for parent_rank, parent in enumerate(parents):
            base = root / parent
            if not base.exists():
                continue
            for path in base.iterdir():
                if not path.is_dir():
                    continue
                if not ((path / "pattern_metrics.csv").exists() or (path / "apples_to_apples_metrics.csv").exists()):
                    continue
                score = (
                    int((path / "predictions_by_pattern.csv").exists()),
                    int("with_scene" in parent),
                    -parent_rank,
                )
                if path.name not in by_run or score > scores[path.name]:
                    by_run[path.name] = path
                    scores[path.name] = score
    return [by_run[name] for name in sorted(by_run)]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fieldnames})


def write_md_table(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(fieldnames) + " |", "| " + " | ".join("---" for _ in fieldnames) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.8g}"
    return value


def load_pattern_metrics(roots: Iterable[Path], methods: set[str] | None = None) -> pd.DataFrame:
    frames = []
    for eval_dir in find_eval_dirs(roots):
        path = eval_dir / "pattern_metrics.csv"
        if not path.exists():
            path = eval_dir / "apples_to_apples_metrics.csv"
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty:
            continue
        run_name = eval_dir.name
        if "run_name" not in frame:
            frame["run_name"] = run_name
        if "method" not in frame:
            frame["method"] = method_from_run(run_name)
        if "seed" not in frame:
            frame["seed"] = seed_from_run(run_name)
        frame["eval_dir"] = str(eval_dir)
        frame["maskfix_eval"] = eval_dir.parent.name.startswith("fresh_eval_maskfix")
        if "mask_suspect" not in frame:
            frame["mask_suspect"] = False
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    if methods is not None:
        data = data[data["method"].astype(str).isin(methods)]
    for column in ("top1", "within3", "within_3", "within@3", "mae", "missing_count", "missing_ratio", "seed"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def load_predictions(roots: Iterable[Path], methods: set[str] | None = None) -> pd.DataFrame:
    frames = []
    for eval_dir in find_eval_dirs(roots):
        path = eval_dir / "predictions_by_pattern.csv"
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty:
            continue
        run_name = eval_dir.name
        if "run_name" not in frame:
            frame["run_name"] = run_name
        if "method" not in frame:
            frame["method"] = method_from_run(run_name)
        if "seed" not in frame:
            frame["seed"] = seed_from_run(run_name)
        frame["eval_dir"] = str(eval_dir)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    if methods is not None:
        data = data[data["method"].astype(str).isin(methods)]
    for column in ("top1_correct", "top3_correct", "top5_correct", "within3_correct", "abs_error", "missing_count", "missing_ratio", "seed"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def metric_column(frame: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def best_external_method(method_rows: list[dict[str, str]], token: str = "amber_lite") -> str:
    candidates = [
        row
        for row in method_rows
        if token in str(row.get("method") or "")
        and int_or_zero(row.get("n")) > 0
        and truthy(row.get("official_ranking_included", "true"))
        and int_or_zero(row.get("mask_suspect_count")) == 0
        and math.isfinite(float_or_nan(row.get("avg_missing_top1_mean")))
    ]
    if not candidates:
        return ""
    return max(candidates, key=lambda row: float_or_nan(row.get("avg_missing_top1_mean"))).get("method", "")


def method_rows_from_summary(summary_root: Path) -> list[dict[str, str]]:
    return read_csv(summary_root / "final_method_mean_std.csv") or read_csv(summary_root / "method_mean_std.csv")


def float_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def int_or_zero(value: Any) -> int:
    number = float_or_nan(value)
    return int(number) if math.isfinite(number) else 0


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def pct(value: Any, digits: int = 2) -> str:
    number = float_or_nan(value)
    if not math.isfinite(number):
        return ""
    number = number * 100.0 if abs(number) <= 1.5 else number
    return f"{number:.{digits}f}%"


def raw(value: Any, digits: int = 3) -> str:
    number = float_or_nan(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else ""
