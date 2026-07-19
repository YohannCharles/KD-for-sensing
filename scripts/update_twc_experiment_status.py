#!/usr/bin/env python3
"""Regenerate the local TWC experiment ledger from immutable manifests."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Update the local TWC experiment status Markdown ledger.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output", default="outputs/twc_experiment_status.md")
    parser.add_argument("--watch-seconds", type=float, default=None)
    args = parser.parse_args()
    output = _repo_path(args.output)
    if args.watch_seconds is not None and args.watch_seconds < 10:
        parser.error("--watch-seconds must be at least 10")
    while True:
        manifests = discover_manifests(_repo_path(args.outputs_root))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_ledger(manifests), encoding="utf-8")
        print(json.dumps({"output": str(output), "manifest_count": len(manifests)}, indent=2), flush=True)
        if args.watch_seconds is None:
            return 0
        time.sleep(args.watch_seconds)


def discover_manifests(outputs_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    result = []
    paths = {*outputs_root.rglob("training_manifest*.json"), *outputs_root.rglob("posthoc_manifest*.json")}
    for path in sorted(paths):
        relative_parts = set(path.relative_to(outputs_root).parts)
        if {"archive", "archived", "invalid", "invalidated"} & relative_parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
            result.append((path, payload))
    return result


def render_ledger(manifests: list[tuple[Path, dict[str, Any]]]) -> str:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# TWC 实验运行台账",
        "",
        f"更新时间：`{now}`",
        "",
        "> 本文件由 `scripts/update_twc_experiment_status.py` 从本地 manifest 自动生成；它是运行台账，不是正式论文 claim 来源。",
        "",
        "## 固定实验约束",
        "",
        "- MMW 主比较：6 方法 x 5 seeds，固定 confirmation train / outer evidence split，固定 600-entry 平衡训练 mask 调度。",
        "- MMW 外部评估：固定 mask cache；分类、物理 codebook 距离和完整 64-beam power 通信指标共同报告。",
        "- DeepSense6G：Scene31-34 合并为一个完整数据集，5 方法 x 3 seeds = 15 个 checkpoint；不与 MMW 合并汇总。",
        "- checkpoint：固定 epoch 的 `last.pth`，禁止按 outer evidence 选择 checkpoint。",
        "- 可靠性压力测试（GPS/image/Radar/LiDAR corruption）默认关闭；只有显式 `--run-reliability-stress` 才生成/运行，不进入默认主队列。",
        "",
        "## GPU 快照",
        "",
        "| GPU | 显存已用 MiB | 显存空闲 MiB | 利用率 % |",
        "|---:|---:|---:|---:|",
    ]
    gpu_rows = _gpu_snapshot()
    lines.extend(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |" for row in gpu_rows)
    if not gpu_rows:
        lines.append("| - | 无法读取 | 无法读取 | 无法读取 |")

    all_jobs = [job for _, payload in manifests for job in payload.get("jobs", []) if isinstance(job, dict)]
    counts = Counter(str(job.get("status", "unknown")) for job in all_jobs)
    evaluation_counts = Counter(
        str(job.get("evaluation_status", "not_recorded")) for job in all_jobs if job.get("kind") is None
    )
    lines.extend(
        [
            "",
            "## 总览",
            "",
            f"发现 `{len(manifests)}` 个 manifest、`{len(all_jobs)}` 个底层任务。",
            "",
            f"任务状态：`{_format_counts(counts)}`  ",
            f"训练任务评估状态：`{_format_counts(evaluation_counts)}`",
        ]
    )

    for path, payload in manifests:
        jobs = [job for job in payload.get("jobs", []) if isinstance(job, dict)]
        relative = _display_path(path)
        lines.extend(
            [
                "",
                f"## `{relative}`",
                "",
                f"协议：`{payload.get('protocol', '未记录')}`  ",
                f"plan SHA256：`{payload.get('plan_sha256', '未记录')}`",
                "",
                "| 方法/variant | 范围 | Seed | 状态 | GPU | PID | Checkpoint | 评估 | Claim | 关键路径 |",
                "|---|---|---:|---|---:|---:|---|---|---|---|",
            ]
        )
        if jobs and all(job.get("kind") in {"corruption", "complexity"} for job in jobs):
            reliability = bool(payload.get("reliability_stress_enabled", payload.get("request", {}).get("reliability_stress_enabled", False)))
            lines.append(
                "可靠性压力测试：`已显式启用`" if reliability else "可靠性压力测试：`默认关闭（显式 --run-reliability-stress 才可运行）`"
            )
            lines.append("")
            lines.extend(_posthoc_rows(jobs))
            continue
        for job in jobs:
            run_dir = Path(str(job.get("run_dir", "")))
            checkpoint = run_dir / "checkpoints" / "last.pth"
            pid = _integer(job.get("pid"))
            live = bool(pid and Path(f"/proc/{pid}").exists())
            status = str(job.get("status", "unknown"))
            if status == "running" and not live:
                status = "running(manifest)/进程未发现"
            evaluation = str(job.get("evaluation_status", "未记录"))
            claim = _claim_status(status, evaluation, checkpoint.is_file())
            scope = str(job.get("scope") or (f"Scene{job['scene']}" if job.get("scene") is not None else "MMW-15域"))
            config_path = _display_path(Path(str(job["config_path"]))) if job.get("config_path") else "-"
            run_path = _display_path(run_dir) if str(run_dir) not in {"", "."} else "-"
            lines.append(
                "| {method} | {scope} | {seed} | {status} | {gpu} | {pid} | {checkpoint} | {evaluation} | {claim} | `{config}` / `{run}` |".format(
                    method=_job_label(job),
                    scope=scope,
                    seed=job.get("seed", "-"),
                    status=status,
                    gpu=job.get("gpu") if job.get("gpu") is not None else "-",
                    pid=pid or "-",
                    checkpoint="存在" if checkpoint.is_file() else "缺失",
                    evaluation=evaluation,
                    claim=claim,
                    config=config_path,
                    run=run_path,
                )
            )
    lines.append("")
    return "\n".join(lines)


def _posthoc_rows(jobs: list[dict[str, Any]]) -> list[str]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for job in jobs:
        groups.setdefault((str(job["method"]), str(job["kind"])), []).append(job)
    rows = []
    for (method, kind), selected in sorted(groups.items()):
        counts = Counter(str(job.get("status", "unknown")) for job in selected)
        active = [job for job in selected if job.get("status") == "running"]
        task = f"{method} / {kind} ({len(selected)} shards)"
        seeds = sorted({int(job["seed"]) for job in selected if job.get("seed") is not None})
        seed_text = "-" if not seeds else str(seeds[0]) if len(seeds) == 1 else f"{seeds[0]}-{seeds[-1]}"
        gpu = ",".join(str(job.get("gpu")) for job in active) or "-"
        pid = ",".join(str(job.get("pid")) for job in active) or "-"
        claim = "完成后作为后处理证据" if counts.get("done") != len(selected) else "后处理完整"
        log_root = Path(str(selected[0].get("log_path", ""))).parent
        rows.append(
            f"| {task} | MMW-15域 | {seed_text} | {_format_counts(counts)} | {gpu} | {pid} | 复用主模型 | - | {claim} | `-` / `{_display_path(log_root)}` |"
        )
    return rows


def _job_label(job: dict[str, Any]) -> str:
    method = str(job.get("variant", job.get("method", "unknown")))
    kind = job.get("kind")
    if kind == "corruption":
        return f"{method} / {job.get('corruption')}:S{job.get('severity')}"
    return f"{method} / {kind}" if kind else method


def _gpu_snapshot() -> list[tuple[str, str, str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [tuple(part.strip() for part in line.split(",")) for line in output.splitlines() if line.strip()]


def _format_counts(counts: Counter[str]) -> str:
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "无任务"


def _claim_status(training: str, evaluation: str, checkpoint_exists: bool) -> str:
    if training != "done" or not checkpoint_exists:
        return "不可：训练未完成"
    if evaluation != "done":
        return "不可：固定评估未完成"
    return "可纳入（仍须完整矩阵审计）"


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
