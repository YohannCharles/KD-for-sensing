#!/usr/bin/env python3
"""Prepare or launch the frozen eight-GPU fused-decision Router screen."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

import launch_mmw_dynamic_router_screen as base
from kd_sensing.data.mmw.twc_router_joint_training import (
    PANEL_SEED,
    PROTOCOL_ID as PANEL_PROTOCOL_ID,
    prepare_router_joint_training_panel,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "mmw_dynamic_router_decision_alignment_v1"
DEFAULT_OUTPUT = ROOT / "outputs/mmw_dynamic_router_decision_alignment_v1"
DEFAULT_SOURCE_CONFIG = base.DEFAULT_SOURCE_CONFIG
DEFAULT_SOURCE_CHECKPOINT = base.DEFAULT_SOURCE_CHECKPOINT
DEFAULT_SOURCE_SHA256 = base.DEFAULT_SOURCE_SHA256
SEED = 1
CANDIDATES = (
    ("PATR-Expected", "patr", "expected_utility"),
    ("PATR-JointCE", "patr", "joint_hard_ce"),
    ("PATR-PowerSoft", "patr", "power_soft_ce"),
    ("PATR-PowerMargin", "patr", "power_top1_margin"),
    ("H2R-Expected", "h2r", "expected_utility"),
    ("H2R-JointCE", "h2r", "joint_hard_ce"),
    ("H2R-PowerSoft", "h2r", "power_soft_ce"),
    ("H2R-PowerMargin", "h2r", "power_top1_margin"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--source-config", default=str(DEFAULT_SOURCE_CONFIG))
    parser.add_argument("--source-checkpoint", default=str(DEFAULT_SOURCE_CHECKPOINT))
    parser.add_argument("--expected-source-sha256", default=DEFAULT_SOURCE_SHA256)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-free-mib", type=int, default=40000)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    gpus = tuple(int(value.strip()) for value in args.gpus.split(",") if value.strip())
    if gpus != tuple(range(8)):
        parser.error("The frozen decision screen requires exactly --gpus 0,1,2,3,4,5,6,7.")
    if args.batch_size <= 0 or args.batch_size % 16:
        parser.error("--batch-size must be a positive multiple of 16.")
    if args.epochs <= 0 or args.min_free_mib <= 0 or args.poll_seconds <= 0:
        parser.error("--epochs, --min-free-mib, and --poll-seconds must be positive.")
    manifest = prepare_plan(
        output_root=base._path(args.output_root),
        source_config=base._path(args.source_config),
        source_checkpoint=base._path(args.source_checkpoint),
        expected_source_sha256=str(args.expected_source_sha256).strip().lower(),
        gpus=gpus,
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
    )
    if not args.launch:
        print(json.dumps({"status": "planned", "manifest": str(manifest), "jobs": 8}, indent=2))
        return 0
    return base.run_plan(
        manifest,
        min_free_mib=int(args.min_free_mib),
        poll_seconds=float(args.poll_seconds),
    )


def prepare_plan(
    *,
    output_root: Path,
    source_config: Path,
    source_checkpoint: Path,
    expected_source_sha256: str,
    gpus: tuple[int, ...],
    batch_size: int,
    epochs: int,
) -> Path:
    if not source_config.is_file() or not source_checkpoint.is_file():
        raise FileNotFoundError("Dynamic Router source config/checkpoint is missing.")
    actual_source_sha256 = base._sha256(source_checkpoint)
    if actual_source_sha256 != expected_source_sha256:
        raise ValueError(
            f"Source checkpoint SHA256 mismatch: expected={expected_source_sha256}, "
            f"actual={actual_source_sha256}."
        )
    source = yaml.safe_load(source_config.read_text(encoding="utf-8")) or {}
    base._validate_source_config(source)
    panel_path = output_root / "cache/joint_training_panel_v1.json"
    panel = prepare_router_joint_training_panel(panel_path, seed=PANEL_SEED)
    request = {
        "protocol": PROTOCOL_ID,
        "panel_protocol": PANEL_PROTOCOL_ID,
        "panel_checksum": str(panel["checksum"]),
        "panel_seed": PANEL_SEED,
        "source_config": str(source_config),
        "source_config_sha256": base._sha256(source_config),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": actual_source_sha256,
        "router_reliability_source_sha256": base._sha256(base.ROUTER_RELIABILITY_SOURCE),
        "utility_numeric_policy": base.UTILITY_NUMERIC_POLICY,
        "seed": SEED,
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "gpus": list(gpus),
        "candidates": [list(item) for item in CANDIDATES],
        "selection_split": "frozen_inner_validation_only",
        "claim_eligible": False,
    }
    request_sha256 = base._payload_sha256(request)
    manifest_path = output_root / "training_manifest_seed1.json"
    if manifest_path.is_file():
        existing = base._read_json(manifest_path)
        if existing.get("request_sha256") != request_sha256:
            raise ValueError(f"Existing decision-alignment plan differs from the frozen request: {manifest_path}")
        return manifest_path

    config_dir = output_root / "generated_configs"
    log_dir = output_root / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for gpu, (name, variant, objective) in zip(gpus, CANDIDATES, strict=True):
        config = build_candidate_config(
            source,
            name=name,
            variant=variant,
            objective=objective,
            output_root=output_root,
            panel_path=panel_path,
            panel_checksum=str(panel["checksum"]),
            source_checkpoint=source_checkpoint,
            source_sha256=actual_source_sha256,
            batch_size=batch_size,
            epochs=epochs,
        )
        config_path = config_dir / f"{name}_seed1.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        jobs.append(
            {
                "candidate": name,
                "router_variant": variant,
                "fused_decision_objective": objective,
                "supervision": "beam_power",
                "seed": SEED,
                "gpu": int(gpu),
                "config_path": str(config_path.resolve()),
                "config_sha256": base._sha256(config_path),
                "run_dir": str((output_root / name / "seed1").resolve()),
                "log_path": str((log_dir / f"{name}_seed1.log").resolve()),
                "status": "planned",
                "claim_eligible": False,
            }
        )
    base._write_json(
        manifest_path,
        {
            "schema_version": 1,
            "protocol": PROTOCOL_ID,
            "request": request,
            "request_sha256": request_sha256,
            "panel_path": str(panel_path.resolve()),
            "jobs": jobs,
            "status": "planned",
            "created_at": base._now(),
        },
    )
    return manifest_path


def build_candidate_config(
    source: Mapping[str, Any],
    *,
    name: str,
    variant: str,
    objective: str,
    output_root: Path,
    panel_path: Path,
    panel_checksum: str,
    source_checkpoint: Path,
    source_sha256: str,
    batch_size: int,
    epochs: int,
) -> dict[str, Any]:
    if (name, variant, objective) not in CANDIDATES:
        raise ValueError(f"Unknown decision-alignment candidate {(name, variant, objective)!r}.")
    template_name = "PATR-Power" if variant == "patr" else "H2R-Power"
    config = base.build_candidate_config(
        deepcopy(dict(source)),
        name=template_name,
        variant=variant,
        supervision="beam_power",
        output_root=output_root,
        panel_path=panel_path,
        panel_checksum=panel_checksum,
        source_checkpoint=source_checkpoint,
        source_sha256=source_sha256,
        batch_size=batch_size,
        epochs=epochs,
    )
    config["experiment"].update({"name": name, "ablation_id": name})
    config["output"].update(
        {"dir": str((output_root / name).resolve()), "run_name": "seed1", "overwrite": False}
    )
    dynamic = config["loss"]["u_mask_beam_jepa"]["dynamic_router"]
    dynamic.update(
        {
            "fused_decision_objective": objective,
            "fused_decision_margin": 0.5,
            "quality_regression_weight": 0.2,
            "fused_utility_weight": 1.0,
            "frame_rank_weight": 0.2 if variant == "h2r" else 0.0,
            "residual_anchor_weight": 0.01,
        }
    )
    dynamic["paired_joint"]["monotonic_weight"] = 0.2
    config.pop("mmw_dynamic_router_screen", None)
    config["mmw_dynamic_router_decision_screen"] = {
        "protocol": PROTOCOL_ID,
        "candidate": name,
        "router_variant": variant,
        "supervision": "beam_power",
        "fused_decision_objective": objective,
        "fused_decision_margin": 0.5,
        "seed": SEED,
        "source_checkpoint_sha256": source_sha256,
        "joint_panel_checksum": panel_checksum,
        "router_reliability_source_sha256": base._sha256(base.ROUTER_RELIABILITY_SOURCE),
        "utility_numeric_policy": base.UTILITY_NUMERIC_POLICY,
        "selection_split": "frozen_inner_validation_only",
        "claim_eligible": False,
    }
    if isinstance(config.get("mmw_all_weather_protocol"), dict):
        config["mmw_all_weather_protocol"]["split_tag"] = PROTOCOL_ID
    return config


if __name__ == "__main__":
    raise SystemExit(main())
