#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from eval_mmw_all_weather_matrix import _load_or_create_temporal_cache

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("S1", "T2", "amber_full", "rmbp_mm")


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for MMW training, then run two domain shards per method on GPU0-7.")
    parser.add_argument("--root", default="outputs/mmw_all_weather_h5p1_seed1_v2")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--domain-shards", type=int, default=2)
    parser.add_argument("--gpus", default=",".join(str(index) for index in range(8)))
    parser.add_argument(
        "--wait-root",
        action="append",
        default=[],
        help="Additional training root to wait for before launching evaluation; repeat as needed.",
    )
    parser.add_argument("--skip-summary", action="store_true")
    args = parser.parse_args()
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = sorted(set(methods) - set(METHODS))
    if not methods or unknown:
        parser.error(f"methods must be a non-empty subset of {METHODS}; got {methods}")
    if args.domain_shards <= 0:
        parser.error("--domain-shards must be positive")
    gpus = tuple(int(item.strip()) for item in args.gpus.split(",") if item.strip())
    if len(gpus) != len(methods) * int(args.domain_shards):
        parser.error("--gpus must provide exactly one GPU per method/domain shard job")
    root = ROOT / args.root
    wait_roots = tuple(dict.fromkeys([root, *(ROOT / item for item in args.wait_root)]))
    required_runs = tuple((wait_root, method) for wait_root in wait_roots for method in methods)
    status_path = root / "eval_orchestrator_status.json"
    _write_status(status_path, "waiting_for_training", wait_roots=[str(item) for item in wait_roots])
    while True:
        states = {
            f"{wait_root}:{method}": str(
                (_read_json(wait_root / method / "seed1" / "run_status.json") or {}).get("state", "pending")
            )
            for wait_root, method in required_runs
        }
        if any(state in {"failed", "error"} for state in states.values()):
            _write_status(status_path, "blocked_training_failed", states=states)
            return 2
        if all(
            states[f"{wait_root}:{method}"] == "complete"
            and (wait_root / method / "seed1" / "checkpoints" / "last.pth").exists()
            for wait_root, method in required_runs
        ):
            break
        time.sleep(max(5, int(args.poll_seconds)))
    missing = [
        f"{wait_root}:{method}"
        for wait_root, method in required_runs
        if not (wait_root / method / "seed1" / "checkpoints" / "last.pth").exists()
    ]
    if missing:
        _write_status(status_path, "blocked_last_checkpoint_missing", missing=missing)
        return 2
    mask_cache = ROOT / "outputs/mmw_all_weather_h5p1_eval_masks_v2"
    _load_or_create_temporal_cache(mask_cache, modality_frame_masks=16)
    _write_status(status_path, "evaluating", gpu_count=len(gpus), domain_shards=args.domain_shards)
    processes = []
    log_dir = root / "eval_logs"
    log_dir.mkdir(exist_ok=True)
    shard_root = root / "eval_matrix_v2_shards"
    for shard_index in range(int(args.domain_shards)):
        for method_index, method in enumerate(methods):
            gpu = gpus[method_index + shard_index * len(methods)]
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": "2",
                }
            )
            handle = (log_dir / f"{method}_shard{shard_index}.log").open("w", encoding="utf-8")
            command = [
                "conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "python",
                "scripts/eval_mmw_all_weather_matrix.py",
                "--root", args.root,
                "--methods", method,
                "--output-dir", str(shard_root / f"shard{shard_index}"),
                "--mask-cache", str(mask_cache),
                "--domain-shard-index", str(shard_index),
                "--domain-shard-count", str(args.domain_shards),
            ]
            label = f"{method}/shard{shard_index}"
            processes.append((label, subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT), handle))
    failures = {}
    for method, process, handle in processes:
        code = process.wait()
        handle.close()
        if code:
            failures[method] = code
    if failures:
        _write_status(status_path, "blocked_evaluation_failed", failures=failures)
        return 3
    eval_dir = root / "eval_matrix_v2"
    try:
        _merge_eval_shards(shard_root, eval_dir, methods=methods, shard_count=args.domain_shards)
    except Exception as exc:  # noqa: BLE001 - preserve merge failure in orchestrator status.
        _write_status(status_path, "blocked_shard_merge_failed", error=f"{type(exc).__name__}: {exc}")
        return 4
    if args.skip_summary:
        _write_status(status_path, "done", summary_skipped=True)
        return 0
    summary_log = log_dir / "summary.log"
    with summary_log.open("w", encoding="utf-8") as handle:
        code = subprocess.call(
            [
                "conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "python",
                "scripts/summarize_mmw_all_weather_matrix.py",
                "--eval-dir", str(eval_dir),
                "--output-dir", str(root / "final_summary_v2"),
            ],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    _write_status(status_path, "done" if code == 0 else "blocked_summary_failed", return_code=code)
    return int(code)


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_status(path: Path, state: str, **extra) -> None:
    payload = {"state": state, "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _merge_eval_shards(
    shard_root: Path,
    output_dir: Path,
    *,
    methods: tuple[str, ...] = METHODS,
    shard_count: int = 2,
) -> None:
    for method in methods:
        rows = []
        provenances = []
        for shard_index in range(int(shard_count)):
            method_dir = shard_root / f"shard{shard_index}" / method
            with (method_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows.extend(dict(row) for row in csv.DictReader(handle))
            provenances.append(_read_json(method_dir / "provenance.json"))
        domains = {row.get("domain_id", "") for row in rows}
        if len(domains) != 15:
            raise ValueError(f"{method}: merged evaluation must contain 15 domains, got {len(domains)}")
        identity_fields = ("domain_id", "eval_family", "pattern", "missing_rate", "mask_digest")
        identities = [tuple(row.get(field, "") for field in identity_fields) for row in rows]
        if len(identities) != len(set(identities)):
            raise ValueError(f"{method}: duplicate rows found while merging domain shards")
        target = output_dir / method
        target.mkdir(parents=True, exist_ok=True)
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        with (target / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        (target / "provenance.json").write_text(
            json.dumps(
                {
                    "method": method,
                    "domain_count": 15,
                    "row_count": len(rows),
                    "merge_status": f"complete_{int(shard_count)}_domain_shards",
                    "source_provenance": provenances,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    raise SystemExit(main())
