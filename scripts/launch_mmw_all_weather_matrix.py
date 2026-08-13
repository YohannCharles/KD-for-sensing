#!/usr/bin/env python3
"""Generate and launch retained MMW routes on one trajectory split."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kd_sensing.config import dump_config, load_config
from kd_sensing.config.io import deep_merge, load_config_source
from kd_sensing.data.mmw.trajectory_protocol import (
    TRAJECTORY_PROTOCOL_MODE,
    TRAJECTORY_SPLIT_SEED,
    bind_trajectory_config,
    trajectory_audit_path,
    trajectory_manifest_path,
    validate_trajectory_config_protocol,
)
from kd_sensing.data.temporal_missing import (
    WHOLE_ONLY_PATTERN_CONDITION_COUNTS,
    WHOLE_ONLY_PATTERN_PANEL_SIZE,
    WHOLE_ONLY_PATTERN_SCHEDULE_ID,
)


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("U0", "amber_full", "rmbp_mm")
SENSING_MODALITIES = ("image", "radar", "gps", "lidar")
METHOD_BASES = {
    "U0": "configs/mmw/u0.yaml",
    "amber_full": "configs/mmw/amber_full.yaml",
    "rmbp_mm": "configs/mmw/rmbp_mm.yaml",
}
DEFAULT_OUTPUT_ROOT = "outputs/mmw_trajectory_u0"
DEFAULT_PROTOCOL = trajectory_manifest_path(ROOT / "outputs", TRAJECTORY_SPLIT_SEED)


def _normalise_checkpoint_selection(value: str | None) -> str | None:
    """Validate the trainer's checkpoint policy while preserving omitted defaults."""

    if value is None:
        return None
    normalised = str(value).strip().lower()
    if normalised not in {"last", "best_validation_loss"}:
        raise ValueError("checkpoint_selection must be 'last' or 'best_validation_loss'.")
    return normalised


def _bind_optional_strict_caches(
    cfg: dict[str, Any],
    *,
    frame_cache_root: Path | None,
    gps_coordinate_cache_root: Path | None,
) -> None:
    """Bind a matched pair of strict image/LiDAR and GPS caches, when requested."""

    if (frame_cache_root is None) != (gps_coordinate_cache_root is None):
        raise ValueError("frame_cache_root and gps_coordinate_cache_root must be provided together")
    if frame_cache_root is None:
        return

    frame_root = Path(frame_cache_root).expanduser().resolve()
    gps_root = Path(gps_coordinate_cache_root).expanduser().resolve()
    if not frame_root.is_dir():
        raise FileNotFoundError(f"MMW frame cache root is missing: {frame_root}")
    if not gps_root.is_dir():
        raise FileNotFoundError(f"MMW GPS coordinate cache root is missing: {gps_root}")

    dataset = cfg.setdefault("data", {}).setdefault("dataset", {})
    dataset.update(
        {
            "frame_cache_root": str(frame_root),
            "frame_cache_strict": True,
            "gps_coordinate_cache_root": str(gps_root),
        }
    )


def _apply_temporal_schedule(cfg: dict[str, Any], *, whole_modality: bool) -> None:
    """Apply the optional pre-registered whole-modality schedule."""

    if not whole_modality:
        return
    temporal = cfg.setdefault("temporal_missing", {})
    temporal.update(
        {
            "enabled": True,
            "mode": "balanced_pattern_schedule",
            "schedule_id": WHOLE_ONLY_PATTERN_SCHEDULE_ID,
            "panel_size": int(WHOLE_ONLY_PATTERN_PANEL_SIZE),
            "condition_counts": copy.deepcopy(WHOLE_ONLY_PATTERN_CONDITION_COUNTS),
            "subset_balance": "exact_within_drop_count",
            "token_balance": "exact_aggregate_cell_marginals",
        }
    )


def build_config(
    method: str,
    output_root: Path,
    *,
    protocol_path: Path,
    split_seed: int,
    train_seed: int,
    epochs: int,
    batch_size: int,
    smoke: bool = False,
    whole_modality: bool = False,
    checkpoint_selection: str | None = None,
    frame_cache_root: Path | None = None,
    gps_coordinate_cache_root: Path | None = None,
) -> dict[str, Any]:
    """Build one train/validation config from a retained route and manifest.

    The canonical launcher path intentionally leaves the tracked recipe's temporal
    schedule and checkpoint policy untouched.  Ablation/baseline runs opt into the
    pre-registered whole-modality panel and validation-best selection explicitly.
    """
    if method not in METHOD_BASES:
        raise ValueError(f"Unsupported MMW method: {method}")

    selected_checkpoint = _normalise_checkpoint_selection(checkpoint_selection)

    cfg = _load_base_config(ROOT / METHOD_BASES[method])
    dataset = cfg.setdefault("data", {}).setdefault("dataset", {})
    dataset["type"] = "mmw"
    cfg.setdefault("runtime", {})["evaluate_test_requested"] = False
    cfg.setdefault("data", {}).update(split_protocol=TRAJECTORY_PROTOCOL_MODE, split_seed=int(split_seed))
    cfg.setdefault("experiment", {}).update(name=method, seed=int(train_seed), train_seed=int(train_seed))
    bind_trajectory_config(cfg, protocol_path)
    _apply_temporal_schedule(cfg, whole_modality=whole_modality)
    _bind_optional_strict_caches(
        cfg,
        frame_cache_root=frame_cache_root,
        gps_coordinate_cache_root=gps_coordinate_cache_root,
    )

    primary = cfg.setdefault("model", {}).setdefault("primary", {})
    if method in {"amber_full", "rmbp_mm"} and tuple(primary.get("modalities", ())) != SENSING_MODALITIES:
        raise ValueError(f"{method} sensing-only baseline must use modalities {list(SENSING_MODALITIES)}")

    loader = cfg.setdefault("data", {}).setdefault("dataloader", {})
    loader.update(
        {
            "train_batch_size": int(batch_size),
            "validation_batch_size": int(batch_size),
            "test_batch_size": int(batch_size),
        }
    )
    training = cfg.setdefault("training", {})
    training.update({"epochs": 1 if smoke else int(epochs), "max_epochs": 1 if smoke else int(epochs)})
    if selected_checkpoint is not None:
        training["checkpoint_selection"] = selected_checkpoint
    training["final_test"] = {"enabled": False}
    cfg["output"] = {
        "dir": str(output_root / method),
        "run_name": f"train_seed{train_seed}",
        "group_by_scene": False,
        "overwrite": False,
        "progress": {"enabled": False},
    }
    return cfg


def build_job_matrix(
    methods: tuple[str, ...], train_seeds: tuple[int, ...], gpus: tuple[int, ...] | None, output_root: Path
) -> list[dict[str, Any]]:
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError(f"methods must be unique members of: {', '.join(METHODS)}")
    if not train_seeds or len(set(train_seeds)) != len(train_seeds) or any(seed < 0 for seed in train_seeds):
        raise ValueError("train seeds must be unique non-negative integers")
    pairs = [(method, train_seed) for method in methods for train_seed in train_seeds]
    selected_gpus = tuple(range(len(pairs))) if gpus is None else gpus
    if len(selected_gpus) != len(pairs) or len(set(selected_gpus)) != len(selected_gpus) or any(gpu < 0 for gpu in selected_gpus):
        raise ValueError(f"provide {len(pairs)} unique non-negative GPU ids")
    return [
        {
            "method": method,
            "train_seed": train_seed,
            "gpu": gpu,
            "config_path": output_root / "generated_configs" / f"{method}_train_seed{train_seed}.yaml",
            "log_path": output_root / "logs" / f"{method}_train_seed{train_seed}.log",
            "run_dir": output_root / method / f"train_seed{train_seed}",
            "status": "planned",
        }
        for (method, train_seed), gpu in zip(pairs, selected_gpus)
    ]


def validate_job_targets(jobs: list[dict[str, Any]], manifest_path: Path) -> None:
    targets = [manifest_path]
    for job in jobs:
        targets.extend((Path(job["config_path"]), Path(job["log_path"]), Path(job["run_dir"])))
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing targets:\n" + "\n".join(str(path) for path in existing))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch MMW U0, AMBER, and RMBP on one trajectory split.")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL), help="MMW trajectory manifest JSON.")
    parser.add_argument("--split-seed", type=int, default=TRAJECTORY_SPLIT_SEED)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--train-seeds", default="0")
    parser.add_argument("--gpus")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--whole-modality",
        action="store_true",
        help="Use the pre-registered panel480 clean/drop1/drop2/drop3 schedule.",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=("last", "best_validation_loss"),
        default=None,
        help="Optional checkpoint policy; omitted leaves the canonical recipe unchanged.",
    )
    parser.add_argument("--frame-cache-root", default=None, help="Optional strict MMW image/LiDAR cache root.")
    parser.add_argument(
        "--gps-coordinate-cache-root",
        default=None,
        help="Optional strict MMW GPS coordinate cache root; must be paired with --frame-cache-root.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        methods = tuple(_csv(args.methods))
        train_seeds = tuple(int(item) for item in _csv(args.train_seeds))
        gpus = None if args.gpus is None else tuple(int(item) for item in _csv(args.gpus))
        checkpoint_selection = _normalise_checkpoint_selection(args.checkpoint_selection)
        frame_cache_root = None if args.frame_cache_root is None else Path(args.frame_cache_root)
        gps_coordinate_cache_root = (
            None if args.gps_coordinate_cache_root is None else Path(args.gps_coordinate_cache_root)
        )
        protocol_path = Path(args.protocol).resolve()
        if protocol_path != trajectory_manifest_path(protocol_path.parents[2], args.split_seed):
            raise ValueError("--protocol path and --split-seed do not identify the same trajectory manifest.")
        if not trajectory_audit_path(protocol_path).is_file():
            raise FileNotFoundError(f"MMW split audit is missing: {trajectory_audit_path(protocol_path)}")
        output_root = (ROOT / args.output_root).resolve()
        jobs = build_job_matrix(methods, train_seeds, gpus, output_root)
        manifest_path = output_root / "jobs.json"
        validate_job_targets(jobs, manifest_path)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    output_root.mkdir(parents=True, exist_ok=True)
    try:
        for job in jobs:
            config_path = Path(job["config_path"])
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = build_config(
                job["method"],
                output_root,
                protocol_path=protocol_path,
                split_seed=args.split_seed,
                train_seed=int(job["train_seed"]),
                epochs=args.epochs,
                batch_size=args.batch_size,
                smoke=args.smoke,
                whole_modality=args.whole_modality,
                checkpoint_selection=checkpoint_selection,
                frame_cache_root=frame_cache_root,
                gps_coordinate_cache_root=gps_coordinate_cache_root,
            )
            dump_config(config, config_path)
            resolved = load_config(config_path)
            validate_trajectory_config_protocol(resolved)
            dump_config(resolved, config_path)
            for key in ("config_path", "log_path", "run_dir"):
                job[key] = _manifest_path(Path(job[key]))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    manifest_path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return 0
    return _run_jobs(jobs, manifest_path)


def _load_base_config(path: Path) -> dict[str, Any]:
    source = load_config_source(path)
    payload = copy.deepcopy(source.data)
    bases = payload.pop("_base_", [])
    if isinstance(bases, (str, Path)):
        bases = [bases]
    merged: dict[str, Any] = {}
    for base in bases:
        base_path = Path(str(base))
        if not base_path.is_absolute():
            base_path = source.path.parent / base_path
        merged = deep_merge(merged, _load_base_config(base_path))
    return deep_merge(merged, payload)


def _run_jobs(jobs: list[dict[str, Any]], manifest_path: Path) -> int:
    running = []
    for job in jobs:
        log_path = ROOT / job["log_path"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = ["conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "kd-sensing-train", "--config", job["config_path"]]
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": str(job["gpu"]), "PYTHONUNBUFFERED": "1"})
        handle = log_path.open("w", encoding="utf-8")
        job.update({"status": "running", "command": command, "start_time": _now()})
        running.append((subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT), job, handle))
    manifest_path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")

    failed = False
    for process, job, handle in running:
        return_code = process.wait()
        handle.close()
        job.update({"status": "done" if return_code == 0 else "failed", "return_code": return_code, "end_time": _now()})
        failed = failed or return_code != 0
        manifest_path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    return int(failed)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _manifest_path(path: Path) -> str:
    """Keep job manifests portable for roots inside or outside the repository."""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
