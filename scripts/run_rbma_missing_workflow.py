#!/usr/bin/env python3
import argparse
import subprocess
import time
from pathlib import Path


DEFAULT_CONFIGS = [
    "configs/fusion/experiments/rbma_missing_workflow/amber_style_mask_baseline_fullrun.yaml",
    "configs/fusion/experiments/rbma_missing_workflow/weighted_sum_mask.yaml",
    "configs/fusion/experiments/rbma_missing_workflow/weighted_sum_reliability.yaml",
    "configs/fusion/experiments/rbma_missing_workflow/weighted_sum_reliability_beam_proto.yaml",
    "configs/fusion/experiments/rbma_missing_workflow/weighted_sum_reliability_beam_proto_kd.yaml",
    "configs/fusion/experiments/rbma_missing_workflow/no_jepa_rbma_proto_kd_fullrun.yaml",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run missing-modality workflow configs with bounded concurrency.")
    parser.add_argument("--config", action="append", dest="configs", help="Config to run. Repeatable.")
    parser.add_argument("--max-parallel", "--max_parallel", type=int, default=1)
    parser.add_argument("--auto-resume", "--auto_resume", action="store_true")
    parser.add_argument("--num-workers", "--num_workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args, unknown = parser.parse_known_args(argv)
    configs = args.configs or DEFAULT_CONFIGS
    max_parallel = max(1, int(args.max_parallel))
    running: list[tuple[subprocess.Popen, str]] = []
    failures = 0
    for config in configs:
        while len(running) >= max_parallel:
            failures += _poll_one(running)
        cmd = ["conda", "run", "-n", "kd_mm_beam", "kd-sensing-train", "--config", config]
        if args.auto_resume:
            cmd.append("--auto-resume")
        if args.num_workers is not None:
            cmd.extend(["--num-workers", str(int(args.num_workers))])
        if args.dry_run:
            cmd.append("--dry-run")
        cmd.extend(unknown)
        print(" ".join(cmd), flush=True)
        running.append((subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[1]), config))
    while running:
        failures += _poll_one(running)
    return 1 if failures else 0


def _poll_one(running: list[tuple[subprocess.Popen, str]]) -> int:
    while True:
        for index, (proc, config) in enumerate(list(running)):
            code = proc.poll()
            if code is None:
                continue
            running.pop(index)
            if code != 0:
                print(f"{config} failed with exit code {code}", flush=True)
                return 1
            return 0
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
