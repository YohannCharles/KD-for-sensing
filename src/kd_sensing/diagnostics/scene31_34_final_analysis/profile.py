#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import math
import os
import re
import subprocess
import sys
import time
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
    "family",
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
    "benchmark_device",
    "notes",
]
SUMMARY_FIELDS = ["method", "family", "n", *[field for field in FIELDS if field not in {"method", "family", "run_name", "seed", "notes"}], "notes"]
COST_FIELDS = [
    "Method",
    "Family",
    "Params",
    "Model size",
    "Latency / sample",
    "Samples / second",
    "GPU memory",
    "Extra inference cost",
]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    roots = [Path(args.root), *[Path(item) for item in args.old_root], *[Path(item) for item in args.classifier_root], *[Path(item) for item in args.external_root]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _profile_runs(roots)
    if args.benchmark_latency:
        _apply_latency_benchmarks(rows, args)
    summary = _summary_rows(rows)
    paper_rows = _paper_rows(summary)
    paper_root = Path(args.paper_table_root)
    paper_root.mkdir(parents=True, exist_ok=True)

    _write_csv(out / "method_profile_per_run.csv", rows, FIELDS)
    _write_csv(out / "method_profile_summary.csv", summary, SUMMARY_FIELDS)
    _write_csv(out / "table_compute_cost.csv", paper_rows, COST_FIELDS)
    _write_md(out / "table_compute_cost.md", paper_rows, COST_FIELDS)
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
    parser.add_argument("--paper-table-root", default="outputs/paper_tables/scenes31_34_main")
    parser.add_argument("--benchmark-latency", action="store_true")
    parser.add_argument("--gpus", default="")
    parser.add_argument(
        "--latency-method",
        action="append",
        default=[],
        help="Benchmark only this method id. May be repeated; mainly used by parallel workers.",
    )
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--benchmark-batches", type=int, default=50)
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
        param_stats = _param_stats(run_dir, checkpoint)
        train_stats = _train_stats(run_dir / "train_log.json")
        timing_stats = _timing_stats(run_dir)
        notes = []
        if not checkpoint:
            notes.append("checkpoint unavailable")
        if not math.isfinite(train_stats["train_time_per_epoch_sec"]):
            notes.append("train timing unavailable in logs")
        notes.append("eval latency unavailable; artifact-only profile did not run a forward benchmark")
        row = {
            "method": method,
            "family": _family(method),
            "run_name": run_name,
            "seed": _seed(run_name),
            "num_params": param_stats["num_params"],
            "trainable_params": param_stats["trainable_params"] if math.isfinite(param_stats["trainable_params"]) else train_stats["trainable_params"],
            "model_size_mb": param_stats["model_size_mb"],
            "train_time_per_epoch_sec": train_stats["train_time_per_epoch_sec"],
            "total_train_time_sec": train_stats["total_train_time_sec"],
            "eval_latency_per_batch_ms": math.nan,
            "eval_latency_per_sample_ms": math.nan,
            "eval_samples_per_second": math.nan,
            "gpu_memory_peak_mb": train_stats["gpu_memory_peak_mb"] if math.isfinite(train_stats["gpu_memory_peak_mb"]) else timing_stats["gpu_memory_peak_mb"],
            "extra_inference_cost": _extra_cost(method),
            "benchmark_device": "",
            "notes": "; ".join(notes),
            "_run_dir": run_dir,
            "_checkpoint": checkpoint,
        }
        rows.append(row)
    return rows


def _apply_latency_benchmarks(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    targets = set(args.latency_method) if args.latency_method else {
        "scenes31_34_proto_natural_es40",
        "scenes31_34_proto_sampler_uniform_es40",
        "scenes31_34_proto_randomdrop_bernoulli_k075_es40",
        "scenes31_34_proto_randomdrop_subset_es40",
        "scenes31_34_classifier_randomdrop_subset_es40",
        "scenes31_34_amber_lite_uniform_es40",
    }
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        method = str(row.get("method") or "")
        if method not in targets:
            continue
        row["gpu_memory_peak_mb"] = math.nan
        current = selected.get(method)
        if current is None or _seed(str(row.get("run_name") or "")) < _seed(str(current.get("run_name") or "")):
            selected[method] = row
    gpu_ids = _gpu_ids(str(args.gpus or ""))
    if len(gpu_ids) > 1 and not args.latency_method:
        _apply_parallel_latency_benchmarks(selected, args, gpu_ids)
        return
    for method in sorted(targets, key=_method_rank):
        row = selected.get(method)
        if row is None:
            continue
        stats = _benchmark_run(row.get("_run_dir"), row.get("_checkpoint"), args)
        row["eval_latency_per_batch_ms"] = stats["eval_latency_per_batch_ms"]
        row["eval_latency_per_sample_ms"] = stats["eval_latency_per_sample_ms"]
        row["eval_samples_per_second"] = stats["eval_samples_per_second"]
        row["benchmark_device"] = stats["benchmark_device"]
        if math.isfinite(stats["gpu_memory_peak_mb"]):
            row["gpu_memory_peak_mb"] = stats["gpu_memory_peak_mb"]
        base_notes = str(row.get("notes") or "")
        if math.isfinite(stats["eval_latency_per_sample_ms"]):
            base_notes = base_notes.replace("eval latency unavailable; artifact-only profile did not run a forward benchmark", "")
        row["notes"] = _merge_notes(base_notes, stats["notes"])


def _apply_parallel_latency_benchmarks(
    selected: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    gpu_ids: list[str],
) -> None:
    worker_root = Path(args.out) / "latency_workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    commands = []
    for index, method in enumerate(sorted(selected, key=_method_rank)):
        gpu_id = gpu_ids[index % len(gpu_ids)]
        work_dir = worker_root / _safe_name(method)
        work_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--root",
            str(args.root),
            "--out",
            str(work_dir / "profile"),
            "--paper-table-root",
            str(work_dir / "paper"),
            "--benchmark-latency",
            "--gpus",
            gpu_id,
            "--warmup-batches",
            str(args.warmup_batches),
            "--benchmark-batches",
            str(args.benchmark_batches),
            "--latency-method",
            method,
        ]
        for value in args.old_root:
            cmd.extend(["--old-root", str(value)])
        for value in args.classifier_root:
            cmd.extend(["--classifier-root", str(value)])
        for value in args.external_root:
            cmd.extend(["--external-root", str(value)])
        commands.append((method, gpu_id, work_dir, cmd))

    processes = []
    for method, gpu_id, work_dir, cmd in commands:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        stdout_path = work_dir / "stdout.log"
        stderr_path = work_dir / "stderr.log"
        print(f"[latency] launch {method} on GPU {gpu_id}", flush=True)
        stdout = stdout_path.open("w", encoding="utf-8")
        stderr = stderr_path.open("w", encoding="utf-8")
        process = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[1], env=env, stdout=stdout, stderr=stderr)
        processes.append((method, work_dir, process, stdout, stderr))

    for method, work_dir, process, stdout, stderr in processes:
        return_code = process.wait()
        stdout.close()
        stderr.close()
        row = selected[method]
        if return_code != 0:
            row["notes"] = _merge_notes(
                row.get("notes"),
                f"latency benchmark unavailable: worker failed with exit code {return_code}; see {work_dir}",
            )
            continue
        stats = _read_worker_latency_stats(work_dir / "profile" / "method_profile_per_run.csv", row)
        row["eval_latency_per_batch_ms"] = stats["eval_latency_per_batch_ms"]
        row["eval_latency_per_sample_ms"] = stats["eval_latency_per_sample_ms"]
        row["eval_samples_per_second"] = stats["eval_samples_per_second"]
        row["benchmark_device"] = stats["benchmark_device"]
        if math.isfinite(stats["gpu_memory_peak_mb"]):
            row["gpu_memory_peak_mb"] = stats["gpu_memory_peak_mb"]
        base_notes = str(row.get("notes") or "")
        if math.isfinite(stats["eval_latency_per_sample_ms"]):
            base_notes = base_notes.replace("eval latency unavailable; artifact-only profile did not run a forward benchmark", "")
        row["notes"] = _merge_notes(base_notes, stats["notes"])


def _read_worker_latency_stats(path: Path, parent_row: dict[str, Any]) -> dict[str, Any]:
    fallback = {
        "eval_latency_per_batch_ms": math.nan,
        "eval_latency_per_sample_ms": math.nan,
        "eval_samples_per_second": math.nan,
        "gpu_memory_peak_mb": math.nan,
        "benchmark_device": "",
        "notes": f"latency benchmark unavailable: worker output missing at {path}",
    }
    if not path.exists():
        return fallback
    rows = _read_csv(path)
    run_name = str(parent_row.get("run_name") or "")
    method = str(parent_row.get("method") or "")
    candidates = [row for row in rows if str(row.get("run_name") or "") == run_name]
    if not candidates:
        candidates = [row for row in rows if str(row.get("method") or "") == method]
    if not candidates:
        return fallback
    row = candidates[0]
    return {
        "eval_latency_per_batch_ms": _float(row.get("eval_latency_per_batch_ms")),
        "eval_latency_per_sample_ms": _float(row.get("eval_latency_per_sample_ms")),
        "eval_samples_per_second": _float(row.get("eval_samples_per_second")),
        "gpu_memory_peak_mb": _float(row.get("gpu_memory_peak_mb")),
        "benchmark_device": str(row.get("benchmark_device") or ""),
        "notes": str(row.get("notes") or ""),
    }


def _benchmark_run(run_dir: Any, checkpoint: Any, args: argparse.Namespace) -> dict[str, Any]:
    out = {
        "eval_latency_per_batch_ms": math.nan,
        "eval_latency_per_sample_ms": math.nan,
        "eval_samples_per_second": math.nan,
        "gpu_memory_peak_mb": math.nan,
        "benchmark_device": "",
        "notes": "",
    }
    run_path = Path(run_dir) if run_dir else None
    checkpoint_path = Path(checkpoint) if checkpoint else None
    if run_path is None or checkpoint_path is None or not checkpoint_path.exists():
        out["notes"] = "latency benchmark unavailable: checkpoint unavailable"
        return out
    config_path = run_path / "resolved_config.yaml"
    if not config_path.exists():
        config_path = run_path / "final_config.yaml"
    if not config_path.exists():
        out["notes"] = "latency benchmark unavailable: config unavailable"
        return out
    try:
        import torch
        import yaml

        from kd_sensing.engine.data_factory import build_dataloader, build_split_dataset
        from kd_sensing.engine.evaluation_pass_runtime import evaluation_split_name, prepare_evaluation_batch
        from kd_sensing.engine.optim import build_model
        from kd_sensing.engine.runtime import (
            autocast_context,
            configure_cuda_performance_settings,
            configure_torch_runtime_threads,
            forward_task_model,
            resolve_amp_settings,
            transfer_non_blocking,
        )
        from kd_sensing.utils.checkpoint import load_model_state

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg.setdefault("experiment", {})["device"] = str(device)
        loader_cfg = dict(cfg.get("data", {}).get("dataloader", {}))
        configure_torch_runtime_threads(cfg)
        configure_cuda_performance_settings(cfg, device)
        dataset = build_split_dataset(cfg, "test")
        dataloader = build_dataloader(
            dataset,
            loader_cfg,
            split="test",
            experiment_seed=cfg.get("experiment", {}).get("seed", 0),
        )
        model = build_model(cfg["model"]["primary"]).to(device)
        load_model_state(
            checkpoint_path,
            model,
            role="latency_benchmark",
            map_location=device,
            strict=bool(cfg.get("checkpoint", {}).get("strict_load", True)),
        )
        model.eval()
        non_blocking = transfer_non_blocking(cfg)
        amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
        model_cfg = cfg["model"]["primary"]
        seq_length = int(cfg.get("model", {}).get("seq_length", cfg.get("data", {}).get("dataset", {}).get("seq_len", 1)))
        num_pred = int(cfg.get("model", {}).get("num_pred", cfg.get("data", {}).get("dataset", {}).get("num_pred", 1)))
        task = str(cfg.get("experiment", {}).get("task", "fusion"))
        split_name = evaluation_split_name(dataloader, cfg)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        elapsed: list[float] = []
        samples = 0
        warmup = max(0, int(args.warmup_batches))
        limit = max(1, int(args.benchmark_batches))
        with torch.no_grad():
            for step_index, raw_batch in enumerate(dataloader):
                batch = prepare_evaluation_batch(
                    raw_batch,
                    cfg=cfg,
                    split_name=split_name,
                    difficulty_seed=int(cfg.get("experiment", {}).get("seed", 0)),
                    step_index=step_index,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                with autocast_context(amp_enabled, device, amp_dtype):
                    forward_task_model(
                        model,
                        task,
                        batch,
                        model_cfg=model_cfg,
                        seq_length=seq_length,
                        num_pred=num_pred,
                        device=device,
                        non_blocking=non_blocking,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if step_index >= warmup:
                    elapsed.append(time.perf_counter() - start)
                    samples += _batch_size(batch)
                    if len(elapsed) >= limit:
                        break
        if not elapsed or samples <= 0:
            out["notes"] = "latency benchmark unavailable: eval dataloader produced no benchmark batches"
            return out
        total = sum(elapsed)
        out["eval_latency_per_batch_ms"] = (total / len(elapsed)) * 1000.0
        out["eval_latency_per_sample_ms"] = (total / samples) * 1000.0
        out["eval_samples_per_second"] = samples / total
        out["benchmark_device"] = _device_label(device)
        if device.type == "cuda":
            out["gpu_memory_peak_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        out["notes"] = f"latency benchmark: warmup={warmup}, batches={len(elapsed)}"
        return out
    except Exception as exc:
        out["notes"] = f"latency benchmark unavailable: {type(exc).__name__}: {exc}"
        return out
    finally:
        try:
            del model  # type: ignore[name-defined]
            del dataloader  # type: ignore[name-defined]
            del dataset  # type: ignore[name-defined]
        except Exception:
            pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


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


def _param_stats(run_dir: Path, path: Path | None) -> dict[str, float]:
    startup = _startup_param_stats(run_dir / "startup_summary.json")
    if path is None:
        return {"num_params": startup["num_params"], "trainable_params": startup["trainable_params"], "model_size_mb": math.nan}
    size_mb = path.stat().st_size / (1024 * 1024)
    try:
        from kd_sensing.utils.checkpoint import load_torch_payload

        payload = load_torch_payload(path, map_location="cpu")
        state = payload.get("state_dict") if isinstance(payload, dict) else payload
        if isinstance(state, dict):
            num_params = sum(int(value.numel()) for value in state.values() if hasattr(value, "numel"))
        else:
            num_params = math.nan
    except Exception:
        num_params = math.nan
    if not math.isfinite(num_params):
        num_params = startup["num_params"]
    return {"num_params": num_params, "trainable_params": startup["trainable_params"], "model_size_mb": size_mb}


def _startup_param_stats(path: Path) -> dict[str, float]:
    out = {"num_params": math.nan, "trainable_params": math.nan}
    if not path.exists():
        return out
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out
    candidates = [
        payload.get("architecture_summary", {}).get("model", {}) if isinstance(payload, dict) else {},
        payload.get("architecture_summary", {}) if isinstance(payload, dict) else {},
    ]
    for item in candidates:
        if not isinstance(item, dict):
            continue
        total = _float(item.get("total_params"))
        trainable = _float(item.get("trainable_params"))
        if math.isfinite(total):
            out["num_params"] = total
        if math.isfinite(trainable):
            out["trainable_params"] = trainable
    return out


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


def _timing_stats(run_dir: Path) -> dict[str, float]:
    out = {"gpu_memory_peak_mb": math.nan}
    timing_dir = run_dir.parent / "logs"
    candidates = list(timing_dir.glob(f"{run_dir.name}_timing.csv"))
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            continue
        values = []
        for row in rows:
            for key in ("gpu_mem_reserved_mb", "gpu_mem_alloc_mb"):
                value = _float(row.get(key))
                if math.isfinite(value):
                    values.append(value)
        if values:
            out["gpu_memory_peak_mb"] = max(values)
            break
    return out


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("method") or "")].append(row)
    out = []
    for method in METHOD_ORDER:
        items = grouped.get(method, [])
        row: dict[str, Any] = {"method": method, "family": _family(method), "n": len(items)}
        for field in FIELDS:
            if field in {"method", "family", "run_name", "seed", "notes", "extra_inference_cost", "benchmark_device"}:
                continue
            values = [_float(item.get(field)) for item in items if math.isfinite(_float(item.get(field)))]
            row[field] = mean(values) if values else math.nan
        row["extra_inference_cost"] = _extra_cost(method)
        row["benchmark_device"] = "; ".join(sorted({str(item.get("benchmark_device") or "") for item in items if item.get("benchmark_device")}))
        row["notes"] = "not run" if not items else "; ".join(sorted({str(item.get("notes") or "") for item in items if item.get("notes")}))
        out.append(row)
    return out


def _paper_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        out.append(
            {
                "Method": METHOD_LABELS.get(str(row.get("method") or ""), str(row.get("method") or "")),
                "Family": str(row.get("family") or ""),
                "Params": _compact_int(row.get("num_params")),
                "Model size": _mb(row.get("model_size_mb")),
                "Latency / sample": _ms(row.get("eval_latency_per_sample_ms")),
                "Samples / second": _raw(row.get("eval_samples_per_second"), digits=1),
                "GPU memory": _mb(row.get("gpu_memory_peak_mb")),
                "Extra inference cost": str(row.get("extra_inference_cost") or ""),
            }
        )
    return out


def _extra_cost(method: str) -> str:
    if method == "scenes31_34_proto_randomdrop_subset_es40":
        return "None; training-only exposure strategy"
    if method == "scenes31_34_proto_randomdrop_bernoulli_k075_es40":
        return "None; training-only dropout strategy"
    if method == "scenes31_34_proto_sampler_uniform_es40":
        return "None; training-only exposure strategy"
    if "randomdrop_subset" in method or "sampler_uniform" in method:
        return "None; training-only exposure strategy"
    return "None beyond the configured model"


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _family(method: str) -> str:
    if "classifier" in method:
        return "classifier"
    if "amr_lite" in method or "amber_lite" in method:
        return "external_lite"
    if method.startswith("scenes31_34_proto"):
        return "proto"
    return "auxiliary"


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


def _batch_size(batch: Any) -> int:
    if isinstance(batch, dict):
        for value in batch.values():
            size = _batch_size(value)
            if size > 0:
                return size
        return 0
    if isinstance(batch, (list, tuple)):
        for value in batch:
            size = _batch_size(value)
            if size > 0:
                return size
        return 0
    shape = getattr(batch, "shape", None)
    return int(shape[0]) if shape is not None and len(shape) > 0 else 0


def _device_label(device: Any) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    suffix = f" (CUDA_VISIBLE_DEVICES={visible})" if visible else ""
    return f"{device}{suffix}"


def _gpu_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "worker"


def _merge_notes(*items: Any) -> str:
    notes: list[str] = []
    for item in items:
        for part in str(item or "").split(";"):
            part = part.strip()
            if part and part not in notes:
                notes.append(part)
    return "; ".join(notes)


if __name__ == "__main__":
    raise SystemExit(main())
