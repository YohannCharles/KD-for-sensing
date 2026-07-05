#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METHOD_LABELS = {
    "scenes31_34_proto_natural_es40": "Proto natural",
    "scenes31_34_proto_sampler_uniform_es40": "Proto uniform pattern exposure",
    "scenes31_34_proto_randomdrop_bernoulli_k075_es40": "Proto Bernoulli randomdrop",
    "scenes31_34_proto_randomdrop_subset_es40": "Proto random subset exposure",
    "scenes31_34_classifier_natural_es40": "Classifier natural",
    "scenes31_34_classifier_randomdrop_subset_es40": "Classifier random subset",
    "scenes31_34_amr_lite_natural_es40": "AMR-lite natural",
    "scenes31_34_amber_lite_natural_es40": "AMBER-lite natural",
    "scenes31_34_amr_lite_uniform_es40": "AMR-lite uniform",
    "scenes31_34_amber_lite_uniform_es40": "AMBER-lite uniform",
}
METHOD_ORDER = tuple(METHOD_LABELS)
FIELDS = [
    "method",
    "run_name",
    "seed",
    "num_params",
    "trainable_params",
    "model_size_mb",
    "train_time_per_epoch_sec",
    "total_train_time_sec",
    "eval_latency_per_batch_ms",
    "eval_latency_per_sample_ms",
    "eval_samples_per_second",
    "gpu_memory_peak_mb",
    "extra_inference_cost",
    "notes",
]
SUMMARY_FIELDS = ["method", "n", *[field for field in FIELDS if field not in {"method", "run_name", "seed", "notes"}], "notes"]
COST_FIELDS = [
    "Method",
    "Params",
    "Model size",
    "Train time / epoch",
    "Inference latency / sample",
    "Samples / second",
    "Extra inference cost",
]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    roots = [Path(args.root), *[Path(item) for item in args.old_root], *[Path(item) for item in args.classifier_root], *[Path(item) for item in args.external_root]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _profile_runs(roots)
    summary = _summary_rows(rows)
    paper_rows = _paper_rows(summary)
    paper_root = Path("outputs/paper_tables/scenes31_34_main")
    paper_root.mkdir(parents=True, exist_ok=True)

    _write_csv(out / "method_profile_per_run.csv", rows, FIELDS)
    _write_csv(out / "method_profile_summary.csv", summary, SUMMARY_FIELDS)
    _write_csv(paper_root / "table_compute_cost.csv", paper_rows, COST_FIELDS)
    _write_md(paper_root / "table_compute_cost.md", paper_rows, COST_FIELDS)
    print(f"Wrote Scene31-34 method profile to {out}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Scene31-34 methods from existing run artifacts.")
    parser.add_argument("--root", default="outputs/scenes31_34_main_lmdb")
    parser.add_argument("--old-root", action="append", default=[])
    parser.add_argument("--classifier-root", action="append", default=[])
    parser.add_argument("--external-root", action="append", default=[])
    parser.add_argument("--out", default="outputs/scenes31_34_main_lmdb/profile")
    return parser


def _profile_runs(roots: list[Path]) -> list[dict[str, Any]]:
    by_run: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("**/run_status.json"):
            run_dir = path.parent
            run_name = run_dir.name
            method = _method(run_name)
            if method in METHOD_LABELS and run_name not in by_run:
                by_run[run_name] = run_dir
    rows = []
    for run_name, run_dir in sorted(by_run.items(), key=lambda item: (_method_rank(_method(item[0])), _seed(item[0]))):
        method = _method(run_name)
        checkpoint = _checkpoint_path(run_dir)
        param_stats = _param_stats(checkpoint)
        train_stats = _train_stats(run_dir / "train_log.json")
        notes = []
        if not checkpoint:
            notes.append("checkpoint unavailable")
        if not math.isfinite(train_stats["train_time_per_epoch_sec"]):
            notes.append("train timing unavailable in logs")
        notes.append("eval latency not sampled from artifacts")
        row = {
            "method": method,
            "run_name": run_name,
            "seed": _seed(run_name),
            "num_params": param_stats["num_params"],
            "trainable_params": train_stats["trainable_params"] if math.isfinite(train_stats["trainable_params"]) else param_stats["num_params"],
            "model_size_mb": param_stats["model_size_mb"],
            "train_time_per_epoch_sec": train_stats["train_time_per_epoch_sec"],
            "total_train_time_sec": train_stats["total_train_time_sec"],
            "eval_latency_per_batch_ms": math.nan,
            "eval_latency_per_sample_ms": math.nan,
            "eval_samples_per_second": math.nan,
            "gpu_memory_peak_mb": train_stats["gpu_memory_peak_mb"],
            "extra_inference_cost": _extra_cost(method),
            "notes": "; ".join(notes),
        }
        rows.append(row)
    return rows


def _checkpoint_path(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "checkpoints" / "best_top1.pth",
        run_dir / "checkpoints" / "best.pth",
        run_dir / "checkpoints" / "last.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _param_stats(path: Path | None) -> dict[str, float]:
    if path is None:
        return {"num_params": math.nan, "model_size_mb": math.nan}
    size_mb = path.stat().st_size / (1024 * 1024)
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
        state = payload.get("state_dict") if isinstance(payload, dict) else payload
        if isinstance(state, dict):
            num_params = sum(int(value.numel()) for value in state.values() if hasattr(value, "numel"))
        else:
            num_params = math.nan
    except Exception:
        num_params = math.nan
    return {"num_params": num_params, "model_size_mb": size_mb}


def _train_stats(path: Path) -> dict[str, float]:
    out = {
        "trainable_params": math.nan,
        "train_time_per_epoch_sec": math.nan,
        "total_train_time_sec": math.nan,
        "gpu_memory_peak_mb": math.nan,
    }
    if not path.exists():
        return out
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out
    epoch_logs = payload.get("epoch_logs") if isinstance(payload, dict) else []
    if isinstance(epoch_logs, list):
        durations = [
            _float(item.get(key))
            for item in epoch_logs
            if isinstance(item, dict)
            for key in ("epoch_time_sec", "epoch_seconds", "duration_sec", "wall_time_sec")
            if math.isfinite(_float(item.get(key)))
        ]
        if durations:
            out["train_time_per_epoch_sec"] = mean(durations)
            out["total_train_time_sec"] = sum(durations)
    if isinstance(payload, dict):
        last_epoch = epoch_logs[-1] if isinstance(epoch_logs, list) and epoch_logs and isinstance(epoch_logs[-1], dict) else {}
        out["trainable_params"] = _first_number(
            payload,
            last_epoch,
            keys=("optimizer/params/main", "trainable_params", "optimizer_params"),
        )
        out["gpu_memory_peak_mb"] = _recursive_find_number(payload, ("gpu_memory_peak_mb", "max_memory_allocated_mb", "memory_peak_mb"))
    return out


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("method") or "")].append(row)
    out = []
    for method in METHOD_ORDER:
        items = grouped.get(method, [])
        row: dict[str, Any] = {"method": method, "n": len(items)}
        for field in FIELDS:
            if field in {"method", "run_name", "seed", "notes", "extra_inference_cost"}:
                continue
            values = [_float(item.get(field)) for item in items if math.isfinite(_float(item.get(field)))]
            row[field] = mean(values) if values else math.nan
        row["extra_inference_cost"] = _extra_cost(method)
        row["notes"] = "not run" if not items else "; ".join(sorted({str(item.get("notes") or "") for item in items if item.get("notes")}))
        out.append(row)
    return out


def _paper_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        out.append(
            {
                "Method": METHOD_LABELS.get(str(row.get("method") or ""), str(row.get("method") or "")),
                "Params": _compact_int(row.get("num_params")),
                "Model size": _mb(row.get("model_size_mb")),
                "Train time / epoch": _sec(row.get("train_time_per_epoch_sec")),
                "Inference latency / sample": _ms(row.get("eval_latency_per_sample_ms")),
                "Samples / second": _raw(row.get("eval_samples_per_second"), digits=1),
                "Extra inference cost": str(row.get("extra_inference_cost") or ""),
            }
        )
    return out


def _extra_cost(method: str) -> str:
    if method == "scenes31_34_proto_randomdrop_subset_es40":
        return "none at inference; training-only exposure strategy"
    if "randomdrop_subset" in method:
        return "none at inference; training-only exposure strategy"
    return "none beyond the configured model"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _write_md(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    lines = ["| " + " | ".join(fieldnames) + " |", "| " + " | ".join("---" for _ in fieldnames) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _first_number(*dicts: dict[str, Any], keys: tuple[str, ...]) -> float:
    for data in dicts:
        for key in keys:
            value = _float(data.get(key))
            if math.isfinite(value):
                return value
    return math.nan


def _recursive_find_number(value: Any, keys: tuple[str, ...]) -> float:
    if isinstance(value, dict):
        for key in keys:
            number = _float(value.get(key))
            if math.isfinite(number):
                return number
        for item in value.values():
            found = _recursive_find_number(item, keys)
            if math.isfinite(found):
                return found
    if isinstance(value, list):
        for item in value:
            found = _recursive_find_number(item, keys)
            if math.isfinite(found):
                return found
    return math.nan


def _method(run_name: str) -> str:
    return re.sub(r"_seed\d+$", "", str(run_name))


def _seed(run_name: str) -> int:
    match = re.search(r"_seed(\d+)$", str(run_name))
    return int(match.group(1)) if match else 0


def _method_rank(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.6g}"
    return value


def _compact_int(value: Any) -> str:
    number = _float(value)
    return f"{int(number):,}" if math.isfinite(number) else "NaN"


def _mb(value: Any) -> str:
    number = _float(value)
    return f"{number:.2f} MB" if math.isfinite(number) else "NaN"


def _sec(value: Any) -> str:
    number = _float(value)
    return f"{number:.2f} s" if math.isfinite(number) else "NaN"


def _ms(value: Any) -> str:
    number = _float(value)
    return f"{number:.3f} ms" if math.isfinite(number) else "NaN"


def _raw(value: Any, *, digits: int) -> str:
    number = _float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "NaN"


if __name__ == "__main__":
    raise SystemExit(main())
