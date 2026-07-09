#!/usr/bin/env python
import argparse
import os
import subprocess
from pathlib import Path


EXPERIMENTS = {
    "tm0": ("tm0_no_temporal_missing", "none", 0.0, 1),
    "tm1": ("tm1_frame_missing_20", "frame_bernoulli", 0.2, 1),
    "tm2": ("tm2_modality_frame_missing_20", "modality_frame_bernoulli", 0.2, 1),
    "tm3": ("tm3_block_missing_len2", "block", 1.0, 2),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch temporal missing v1 experiments.")
    parser.add_argument("--config", default="configs/fusion/u_mask_beam_jepa_smoke.yaml")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--max_jobs", "--max-jobs", type=int, default=1)
    parser.add_argument("--per_gpu", "--per-gpu", type=int, default=1)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--experiments", default="tm0,tm1,tm2,tm3")
    parser.add_argument("--output_root", "--output-root", default="outputs/temporal_missing_v1")
    parser.add_argument("--max_epochs", "--max-epochs", type=int, default=None)
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--skip_completed", "--skip-completed", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = _commands(args)
    if args.dry_run:
        for _, command in commands:
            print(" ".join(command))
        return 0
    running: list[subprocess.Popen] = []
    gpus = _csv(args.gpus)
    for index, (run_dir, command) in enumerate(commands):
        if args.skip_completed and (run_dir / "run_status.json").exists() and not args.force:
            print(f"skip completed: {run_dir}")
            continue
        while len(running) >= max(int(args.max_jobs), 1):
            running = [proc for proc in running if proc.poll() is None]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus[index % max(len(gpus), 1)] if gpus else ""
        print("launch:", " ".join(command))
        running.append(subprocess.Popen(command, env=env))
    failures = 0
    for proc in running:
        failures += int(proc.wait() != 0)
    return 1 if failures else 0


def _commands(args: argparse.Namespace) -> list[tuple[Path, list[str]]]:
    seeds = [int(value) for value in _csv(args.seeds)]
    experiments = _csv(args.experiments)
    commands = []
    for exp in experiments:
        if exp not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment {exp!r}; choose from {sorted(EXPERIMENTS)}.")
        name, mode, prob, block_len = EXPERIMENTS[exp]
        for seed in seeds:
            run_dir = Path(args.output_root) / name / f"seed{seed}"
            command = [
                "conda",
                "run",
                "-n",
                "kd_mm_beam",
                "kd-sensing-train",
                "--config",
                str(args.config),
                "--history_window",
                "5",
                "--prediction_window",
                "1",
                "--temporal_missing_mode",
                mode,
                "--temporal_missing_prob",
                str(prob),
                "--temporal_missing_block_len",
                str(block_len),
                "--temporal_missing_apply",
                "train",
                "--temporal_missing_seed",
                str(seed),
                "--override",
                f"experiment.seed={seed}",
                "--override",
                f"output.dir={run_dir.parent}",
                "--override",
                f"output.run_name=seed{seed}",
            ]
            if args.max_epochs is not None:
                command.extend(["--override", f"training.epochs={int(args.max_epochs)}"])
            if args.force:
                command.extend(["--override", "output.overwrite=true"])
            commands.append((run_dir, command))
    return commands


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
