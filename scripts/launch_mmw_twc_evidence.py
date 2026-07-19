#!/usr/bin/env python3
"""Plan and run the reproducible MMW post-selection confirmation matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping

import yaml

from kd_sensing.data.mmw.twc_evidence import build_confirmation_train_domains, load_protocol
from kd_sensing.utils.artifact_registry import canonical_mmw_twc_evidence_config_sha256, training_profile_checkpoint_provenance


ROOT = Path(__file__).resolve().parents[1]
MAIN_VARIANTS = ("T2", "S1", "masktrain_cls", "amber_full", "rmbp_mm", "amr_net_4m")
SEEDS = (1, 2, 3, 4, 5)
DEFAULT_ALLOWED_GPUS = frozenset((4, 5, 6, 7))
EXPANDED_ALLOWED_GPUS = frozenset(range(8))
PLAN_SCHEMA_VERSION = 3
COMPARISON_CONTRACT_VERSION = 2
DEFAULT_OUTPUT_ROOT = "outputs/mmw_twc_fair_pattern_v1"
DEFAULT_PROTOCOL_MANIFEST = "outputs/cache/mmw_twc_outer_v1/protocol_manifest.json"
DEFAULT_TOPOLOGY_GLOB = "outputs/cache/mmw_codebook_topology/v1/*/topology_manifest.json"
TRAINING_MASK_SEED_ALGORITHM = (
    "sha256(base_seed,balanced_pattern_schedule,epoch); sample=(step*train_batch_size+row)%600"
)
WHOLE_ONLY_TRAINING_MASK_SEED_ALGORITHM = (
    "sha256(base_seed,balanced_whole_pattern_schedule,epoch); sample=(step*train_batch_size+row)%480"
)

VARIANT_PROTOCOL: dict[str, dict[str, Any]] = {
    "T2": {"base_method": "T2", "matched_control": "T2", "role": "main"},
    "S1": {"base_method": "S1", "matched_control": "T2", "role": "main_temporal_control"},
    "masktrain_cls": {"base_method": "masktrain_cls", "matched_control": "T2", "role": "main_simple_baseline"},
    "amber_full": {"base_method": "amber_full", "matched_control": "T2", "role": "main_baseline"},
    "rmbp_mm": {"base_method": "rmbp_mm", "matched_control": "T2", "role": "main_baseline"},
    "amr_net_4m": {"base_method": "amr_net_4m", "matched_control": "T2", "role": "main_baseline"},
    "T2-NoBPA": {"base_method": "T2", "matched_control": "T2", "role": "bpa_off"},
    "T2-TopologyLinear": {"base_method": "T2", "matched_control": "T2", "role": "bpa_linear"},
    "T2-TopologyPermuted": {"base_method": "T2", "matched_control": "T2", "role": "bpa_permuted"},
    "T2-CLS": {"base_method": "T2", "matched_control": "T2", "role": "classifier_head"},
    "T2-NoRouterOracle": {"base_method": "T2", "matched_control": "T2", "role": "router_oracle_off"},
    "T2-ReliabilityOnly": {"base_method": "T2", "matched_control": "T2", "role": "reliability_only"},
    "T2-Uniform": {"base_method": "T2", "matched_control": "T2", "role": "uniform_fusion"},
    "T2-WholeOnly": {"base_method": "T2", "matched_control": "T2", "role": "whole_modality_training_only"},
    "T2-BPA2CMA": {"base_method": "T2", "matched_control": "T2-NoBPA", "role": "cma_replacement"},
}
ABLATION_VARIANTS = tuple(item for item in VARIANT_PROTOCOL if item not in MAIN_VARIANTS)

# Every non-runtime recipe change must be pre-registered against its matched
# control.  Prefixes intentionally stay narrow enough to protect the shared
# split, data, optimizer, scheduler, mask, and seed contract.
VARIANT_ALLOWED_CONFIG_DIFFS: dict[str, tuple[str, ...]] = {
    "T2": (),
    "S1": (
        "loss.u_mask_beam_jepa.superset_consistency.confidence_gated_kl",
        "loss.u_mask_beam_jepa.superset_consistency.enabled",
        "loss.u_mask_beam_jepa.superset_consistency.kl_weight",
        "temporal_missing.preserve_unmasked_for_superset",
    ),
    "amber_full": (
        "loss",
        "mmw_all_weather_protocol.baseline_fidelity",
        "mmw_all_weather_protocol.router_architecture_profile",
        "mmw_all_weather_protocol.training_profile",
        "model.primary",
        "scheduler",
        "temporal_missing.preserve_unmasked_for_superset",
        "training.grad_clip",
        "training.optimizer",
        "training.weight_decay",
    ),
    "masktrain_cls": (
        "loss",
        "mmw_all_weather_protocol.baseline_fidelity",
        "mmw_all_weather_protocol.router_architecture_profile",
        "mmw_all_weather_protocol.training_profile",
        "model.primary",
        "temporal_missing.preserve_unmasked_for_superset",
        "training.grad_clip",
    ),
    "rmbp_mm": (
        "comparability",
        "loss",
        "mmw_all_weather_protocol.baseline_fidelity",
        "mmw_all_weather_protocol.router_architecture_profile",
        "mmw_all_weather_protocol.training_profile",
        "model.primary",
        "scheduler",
        "temporal_missing.preserve_unmasked_for_superset",
        "training.grad_clip",
        "training.optimizer",
        "training.weight_decay",
    ),
    "amr_net_4m": (
        "loss",
        "mmw_all_weather_protocol.baseline_fidelity",
        "mmw_all_weather_protocol.router_architecture_profile",
        "mmw_all_weather_protocol.training_profile",
        "model.primary",
        "temporal_missing.preserve_unmasked_for_superset",
        "training.grad_clip",
    ),
    "T2-NoBPA": (
        "loss.u_mask_beam_jepa.lambda_modality_proto",
        "loss.u_mask_beam_jepa.lambda_proto",
        "loss.u_mask_beam_jepa.prototype_target_circular",
        "loss.u_mask_beam_jepa.prototype_topology",
        "loss.u_mask_beam_jepa.use_beam_prototype_alignment",
    ),
    "T2-TopologyLinear": (
        "loss.u_mask_beam_jepa.prototype_target_circular",
        "loss.u_mask_beam_jepa.prototype_topology",
    ),
    "T2-TopologyPermuted": ("loss.u_mask_beam_jepa.prototype_topology",),
    "T2-CLS": (
        "loss.u_mask_beam_jepa.lambda_modality_proto",
        "loss.u_mask_beam_jepa.lambda_proto",
        "loss.u_mask_beam_jepa.prototype_target_circular",
        "loss.u_mask_beam_jepa.prototype_topology",
        "loss.u_mask_beam_jepa.use_beam_prototype_alignment",
        "model.primary.head_type",
        "model.primary.router_use_prototype_margin",
    ),
    "T2-NoRouterOracle": ("loss.u_mask_beam_jepa.router_oracle_weight",),
    "T2-ReliabilityOnly": (
        "loss.u_mask_beam_jepa.router_oracle_weight",
        "model.primary.fusion_type",
    ),
    "T2-Uniform": (
        "loss.u_mask_beam_jepa.router_oracle_weight",
        "model.primary.fusion_type",
    ),
    "T2-WholeOnly": (
        "temporal_missing.condition_counts",
        "temporal_missing.panel_size",
        "temporal_missing.schedule_id",
    ),
    "T2-BPA2CMA": (
        "loss.u_mask_beam_jepa.amber_cma_temperature",
        "loss.u_mask_beam_jepa.lambda_amber_cma",
        "loss.u_mask_beam_jepa.use_amber_cma_analogue",
    ),
}

_DYNAMIC_CONFIG_DIFF_PREFIXES = (
    "output",
    "runtime",
    "mmw_twc_evidence",
    "experiment.name",
    "experiment.ablation_id",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan or run MMW TWC post-selection confirmation training.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--protocol-manifest", default=DEFAULT_PROTOCOL_MANIFEST)
    parser.add_argument("--topology-manifest", default=None)
    parser.add_argument("--phase", choices=("main", "ablation", "custom"), default="main")
    parser.add_argument("--variants", default=None)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument(
        "--allow-gpu0-3",
        action="store_true",
        help="Allow GPU0--3 only after explicit user authorization; the default boundary is GPU4--7.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--smoke", action="store_true", help="Run a 1-epoch, tiny-portion preflight that is ineligible for evidence.")
    parser.add_argument("--min-free-mib", type=int, default=40000)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Train only and leave outer/token evaluation to a separately versioned evaluator.",
    )
    parser.add_argument(
        "--retry-failed-training",
        action="store_true",
        help="Archive incomplete failed run directories and explicitly requeue their immutable jobs.",
    )
    parser.add_argument(
        "--retry-failed-evaluations",
        action="store_true",
        help="Explicitly requeue incomplete failed outer evaluations while preserving their retry history.",
    )
    args = parser.parse_args()
    variants = _resolve_variants(args.phase, args.variants)
    seeds = _csv_ints(args.seeds)
    gpus = _csv_ints(args.gpus)
    if args.batch_size <= 0 or args.batch_size % 16:
        parser.error("--batch-size must be a positive multiple of 16")
    if not args.smoke and args.batch_size != 64:
        parser.error("The strict confirmation protocol is fixed to --batch-size 64.")
    if not args.smoke and args.epochs != 40:
        parser.error("The confirmation protocol is fixed to 40 epochs.")
    if not seeds or any(seed <= 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain unique positive integers")
    try:
        _validate_gpu_ids(gpus, allow_gpu0_3=bool(args.allow_gpu0_3))
    except ValueError as exc:
        parser.error(str(exc))
    if args.min_free_mib <= 0 or args.poll_seconds <= 0:
        parser.error("memory and poll settings must be positive")
    if (args.retry_failed_training or args.retry_failed_evaluations) and not args.launch:
        parser.error("retry options require --launch")

    output_root = _repo_path(args.output_root)
    protocol_path = _repo_path(args.protocol_manifest)
    topology_path = _resolve_topology_path(args.topology_manifest)
    epochs = 1 if args.smoke else int(args.epochs)
    manifest_path = prepare_plan(
        output_root,
        protocol_path=protocol_path,
        topology_path=topology_path,
        variants=variants,
        seeds=seeds,
        batch_size=int(args.batch_size),
        epochs=epochs,
        smoke=bool(args.smoke),
        evaluation_enabled=not bool(args.skip_evaluation),
    )
    if not args.launch:
        print(json.dumps({"status": "planned", "manifest": str(manifest_path), "jobs": len(variants) * len(seeds)}, indent=2))
        return 0
    return run_queue(
        manifest_path,
        gpus=tuple(gpus),
        min_free_mib=int(args.min_free_mib),
        poll_seconds=float(args.poll_seconds),
        max_jobs=args.max_jobs,
        allow_gpu0_3=bool(args.allow_gpu0_3),
        retry_failed_training=bool(args.retry_failed_training),
        retry_failed_evaluations=bool(args.retry_failed_evaluations),
    )


def prepare_plan(
    output_root: Path,
    *,
    protocol_path: Path,
    topology_path: Path,
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    batch_size: int,
    epochs: int,
    smoke: bool = False,
    evaluation_enabled: bool = True,
) -> Path:
    if int(batch_size) <= 0 or int(batch_size) % 16:
        raise ValueError("batch_size must be a positive multiple of 16")
    if not smoke and int(batch_size) != 64:
        raise ValueError("The strict confirmation protocol is fixed to batch_size=64")
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    protocol = load_protocol(protocol_path)
    topology = _load_topology(topology_path)
    plan_path = output_root / _plan_name(variants, seeds)
    request = {
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "comparison_contract_version": COMPARISON_CONTRACT_VERSION,
        "protocol_manifest_sha256": protocol["manifest_sha256"],
        "topology_descriptor_sha256": topology["descriptor_sha256"],
        "variants": list(variants),
        "seeds": list(seeds),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "smoke": bool(smoke),
        "evaluation_enabled": bool(evaluation_enabled),
    }
    request_sha256 = _sha256_payload(request)
    if plan_path.exists():
        payload = _read_json(plan_path)
        if payload.get("request_sha256") != request_sha256:
            raise ValueError(f"Existing TWC plan differs from the requested immutable protocol: {plan_path}")
        _validate_plan(payload)
        return plan_path

    confirmation_domains, confirmation_splits = build_confirmation_train_domains(protocol, output_root)
    launcher = _all_weather_launcher()
    config_dir = output_root / "generated_configs"
    control_config_dir = output_root / "generated_control_configs"
    logs_dir = output_root / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    control_config_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    missing_allowlists = sorted(set(VARIANT_PROTOCOL) - set(VARIANT_ALLOWED_CONFIG_DIFFS))
    if missing_allowlists:
        raise ValueError(f"Missing TWC allowed_config_diff declaration for {missing_allowlists}.")
    config_variants = tuple(
        dict.fromkeys(
            (*variants, *(str(VARIANT_PROTOCOL[variant]["matched_control"]) for variant in variants))
        )
    )
    generated_configs: dict[tuple[str, int], dict[str, str]] = {}
    for seed in seeds:
        for variant in config_variants:
            config = build_confirmation_config(
                launcher,
                variant,
                output_root,
                seed=seed,
                batch_size=batch_size,
                epochs=epochs,
                smoke=smoke,
                domains=confirmation_domains,
                protocol=protocol,
                confirmation_splits=confirmation_splits,
                topology=topology,
            )
            target_dir = config_dir if variant in variants else control_config_dir
            config_path = target_dir / f"{variant}_seed{seed}.yaml"
            if config_path.exists():
                existing = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if existing != config:
                    raise FileExistsError(f"Existing generated TWC config differs from this immutable request: {config_path}")
            else:
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            generated_configs[(variant, int(seed))] = {
                "path": str(config_path.resolve()),
                "sha256": _sha256_file(config_path),
            }

    jobs = []
    for seed in seeds:
        for variant in variants:
            current = generated_configs[(variant, int(seed))]
            matched_control = str(VARIANT_PROTOCOL[variant]["matched_control"])
            control = generated_configs[(matched_control, int(seed))]
            run_dir = output_root / variant / f"seed{seed}"
            jobs.append(
                {
                    "variant": variant,
                    "method": variant,
                    "display_name": variant,
                    "seed": int(seed),
                    "matched_control": matched_control,
                    "variant_role": VARIANT_PROTOCOL[variant]["role"],
                    "allowed_config_diff": list(VARIANT_ALLOWED_CONFIG_DIFFS[variant]),
                    "config_path": current["path"],
                    "config_sha256": current["sha256"],
                    "matched_control_config_path": control["path"],
                    "matched_control_config_sha256": control["sha256"],
                    "run_dir": str(run_dir.resolve()),
                    "log_path": str((logs_dir / f"{variant}_seed{seed}.log").resolve()),
                    "status": "planned",
                    "return_code": None,
                    "gpu": None,
                    "evaluation_status": "planned" if evaluation_enabled and not smoke else "skipped",
                    "evaluation_return_code": None,
                    "evaluation_gpu": None,
                    "evaluation_log_path": str((logs_dir / f"{variant}_seed{seed}.eval.log").resolve()),
                    "evaluation_output_path": str(
                        (output_root / "eval_outer" / variant / f"seed{seed}" / "metrics.csv").resolve()
                    ),
                }
            )
    payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol": "mmw_twc_fair_pattern_training_v1",
        "request": request,
        "request_sha256": request_sha256,
        "protocol_manifest": str(protocol_path.resolve()),
        "topology_manifest": str(topology_path.resolve()),
        "confirmation_splits_manifest": str((output_root / "confirmation_train_splits_manifest.json").resolve()),
        "jobs": jobs,
    }
    payload["plan_sha256"] = _plan_sha256(payload)
    _write_json(plan_path, payload)
    _validate_plan(payload)
    return plan_path


def build_confirmation_config(
    launcher: ModuleType,
    variant: str,
    output_root: Path,
    *,
    seed: int,
    batch_size: int,
    epochs: int,
    smoke: bool = False,
    domains: list[dict[str, str]],
    protocol: Mapping[str, Any],
    confirmation_splits: Mapping[str, Any],
    topology: Mapping[str, Any],
) -> dict[str, Any]:
    if variant not in VARIANT_PROTOCOL:
        raise ValueError(f"Unknown MMW TWC variant {variant!r}.")
    if not smoke and int(batch_size) != 64:
        raise ValueError("The strict confirmation protocol is fixed to batch_size=64")
    base_method = str(VARIANT_PROTOCOL[variant]["base_method"])
    training_profile = launcher.default_umask_training_profile(base_method)
    router_profile = launcher.default_umask_router_architecture_profile(base_method)
    config = launcher.build_config(
        base_method,
        output_root,
        seed=int(seed),
        smoke=bool(smoke),
        epochs=int(epochs),
        batch_size=int(batch_size),
        umask_training_profile=training_profile,
        umask_router_architecture_profile=router_profile,
    )
    dataset = config["data"]["dataset"]
    dataset["domains"] = [
        {
            key: value
            for key, value in domain.items()
            if key in {"id", "condition", "scene", "data_root", "train_csv_name", "val_csv_name", "test_csv_name"}
        }
        for domain in domains
    ]
    for key in ("train_csv_name", "val_csv_name", "test_csv_name"):
        dataset.pop(key, None)
    training = config["training"]
    training.update(
        {
            "epochs": int(epochs),
            "max_epochs": int(epochs),
            "final_test": {"enabled": False, "reason": "mmw_twc_outer_evidence_is_explicit_eval_only"},
            "allow_tf32": False,
            "cudnn_benchmark": False,
        }
    )
    config["data"]["domain_balanced_sampling"]["seed"] = int(seed)
    config["temporal_missing"]["seed"] = int(seed)
    config["experiment"].update({"name": variant, "seed": int(seed), "ablation_id": variant if variant != base_method else ""})
    config["output"] = {
        "dir": str(output_root / variant),
        "run_name": f"seed{seed}",
        "group_by_scene": False,
        "overwrite": False,
        "progress": {"enabled": False},
        "tensorboard": {"enabled": False},
    }
    _apply_variant(config, variant, topology)
    bpa_topology = _effective_bpa_topology(config)
    evidence = {
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_manifest_sha256": str(protocol["manifest_sha256"]),
        "confirmation_split_manifest_sha256": str(confirmation_splits["manifest_sha256"]),
        "training_role": "confirmation_train",
        "smoke_preflight": bool(smoke),
        "training_mask_seed": int(seed),
        "training_mask_seed_algorithm": _training_mask_seed_algorithm(config),
        "domain_sampling_seed": int(seed),
        "evaluation_mask_cache_sha256": str(protocol["fixed_mask_cache"]["sha256"]),
        "evaluation_mask_cache_checksum": str(protocol["fixed_mask_cache"]["cache_checksum"]),
        "topology_id": str(bpa_topology["id"]),
        "topology_descriptor_sha256": str(bpa_topology["descriptor_sha256"]),
        "topology_mapping_sha256": str(bpa_topology["mapping_sha256"]),
        "evaluation_topology_id": str(topology["descriptor"]["topology_id"]),
        "evaluation_topology_descriptor_sha256": str(topology["descriptor_sha256"]),
    }
    config["mmw_twc_evidence"] = evidence
    config["mmw_all_weather_protocol"].update(
        {
            "split_tag": str(protocol["protocol_id"]),
            "screening_role": "post_selection_confirmation_train",
            "checkpoint_policy": "fixed_epoch_last_pth",
            "twc_evidence_protocol": str(protocol["protocol_id"]),
            "twc_evidence_protocol_kind": str(protocol["protocol_kind"]),
            "domain_macro_primary": True,
            "smoke_preflight": bool(smoke),
        }
    )
    evaluation = config.setdefault("evaluation", {})
    evaluation.update(
        {
            "beam_distance_circular": True,
            "dba_distance_mode": "circular",
            "metric_profile": "64_beam_ula_dft_phase_cycle_topk_progressive_top3_dba_v1",
        }
    )
    evidence["config_recipe_sha256"] = canonical_mmw_twc_evidence_config_sha256(config)
    # Fail before a long queue starts if profiles, split provenance, or topology
    # cannot be serialized into the checkpoint metadata.
    training_profile_checkpoint_provenance(config)
    return config


def run_queue(
    manifest_path: Path,
    *,
    gpus: tuple[int, ...],
    min_free_mib: int,
    poll_seconds: float,
    max_jobs: int | None,
    allow_gpu0_3: bool = False,
    retry_failed_training: bool = False,
    retry_failed_evaluations: bool = False,
) -> int:
    gpus = _validate_gpu_ids(gpus, allow_gpu0_3=allow_gpu0_3)
    manifest = _read_json(manifest_path)
    _validate_plan(manifest)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("TWC training manifest has no jobs list.")
    started_count = sum(1 for job in jobs if job.get("start_time"))
    running: dict[int, tuple[subprocess.Popen, dict[str, Any], Any, str]] = {}
    _recover_orphaned_jobs(jobs)
    if retry_failed_training:
        _retry_failed_training(jobs, manifest_path.parent)
    if retry_failed_evaluations:
        _retry_failed_evaluations(jobs)
    _write_json(manifest_path, manifest)
    while True:
        _refresh_completed_jobs(jobs)
        _recover_orphaned_jobs(jobs)
        for gpu, (process, job, handle, phase) in list(running.items()):
            result = process.poll()
            if result is None:
                continue
            handle.close()
            running.pop(gpu)
            if phase == "training":
                job["return_code"] = int(result)
                job["end_time"] = _now()
                job["status"] = "done" if result == 0 and _completed_run(job) else "failed"
            else:
                job["evaluation_return_code"] = int(result)
                job["evaluation_end_time"] = _now()
                job["evaluation_status"] = "done" if result == 0 and _completed_evaluation(job) else "failed"
            _write_json(manifest_path, manifest)
        failed = [
            job
            for job in jobs
            if job.get("status") == "failed" or job.get("evaluation_status") == "failed"
        ]
        if failed:
            _write_json(manifest_path, manifest)
            if _has_live_manifest_jobs(jobs):
                time.sleep(float(poll_seconds))
                continue
            return 1
        if all(
            job.get("status") == "done" and job.get("evaluation_status") in {"done", "skipped"}
            for job in jobs
        ):
            _write_json(manifest_path, manifest)
            return 0
        training_budget_exhausted = max_jobs is not None and started_count >= int(max_jobs)
        eligible_evaluations = [job for job in jobs if job.get("status") == "done" and job.get("evaluation_status") == "planned"]
        eligible_training = [] if training_budget_exhausted else [job for job in jobs if job.get("status") == "planned"]
        if not running and not eligible_evaluations and not eligible_training:
            _write_json(manifest_path, manifest)
            return 0
        free_memory = _gpu_free_memory()
        occupied_gpus = _occupied_manifest_gpus(jobs)
        for gpu in gpus:
            if gpu in running or gpu in occupied_gpus or int(free_memory.get(gpu, 0)) < int(min_free_mib):
                continue
            # Complete the main 5x5 training matrix first.  Once no train job
            # remains planned, these exact same workers drain outer evaluation.
            if eligible_training:
                job = eligible_training.pop(0)
                _validate_job_config(job)
                is_resume = bool(job.get("resume_from_orphaned_checkpoint", False))
                process, handle = _start_training_job(job, gpu)
                started_at = _now()
                job.update({"status": "running", "gpu": int(gpu), "pid": process.pid, "start_time": started_at})
                if is_resume:
                    job["resume_from_orphaned_checkpoint"] = False
                    job.setdefault("recovery_history", []).append(
                        {
                            "kind": "training_auto_resume",
                            "at": started_at,
                            "checkpoint": str(Path(str(job["run_dir"])) / "checkpoints" / "last.pth"),
                        }
                    )
                phase = "training"
                started_count += 1
            elif eligible_evaluations:
                job = eligible_evaluations.pop(0)
                process, handle = _start_evaluation_job(job, manifest, gpu)
                job.update(
                    {
                        "evaluation_status": "running",
                        "evaluation_gpu": int(gpu),
                        "evaluation_pid": process.pid,
                        "evaluation_start_time": _now(),
                    }
                )
                phase = "evaluation"
            else:
                break
            running[gpu] = (process, job, handle, phase)
            _write_json(manifest_path, manifest)
        time.sleep(float(poll_seconds))


def _retry_failed_training(jobs: list[dict[str, Any]], output_root: Path) -> None:
    archive_root = output_root / "failed_attempts"
    for job in jobs:
        if job.get("status") != "failed":
            continue
        run_dir = Path(str(job["run_dir"]))
        if _completed_run(job):
            job["status"] = "done"
            continue
        archived = None
        if run_dir.exists():
            target_dir = archive_root / f"{job['variant']}_seed{job['seed']}"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"attempt_{len(list(target_dir.glob('attempt_*'))) + 1}"
            shutil.move(str(run_dir), str(target))
            archived = str(target.resolve())
        job.setdefault("recovery_history", []).append(
            {"kind": "explicit_failed_training_retry", "at": _now(), "archived_run_dir": archived}
        )
        job.update(status="planned", return_code=None, gpu=None, pid=None, start_time=None, end_time=None)


def _apply_variant(config: dict[str, Any], variant: str, topology: Mapping[str, Any]) -> None:
    if variant in {"masktrain_cls", "amber_full", "rmbp_mm", "amr_net_4m"}:
        return
    primary = config["model"]["primary"]
    loss = config["loss"]["u_mask_beam_jepa"]
    _set_physical_bpa(loss, topology)
    if variant == "T2-NoBPA":
        _disable_bpa(loss)
    elif variant == "T2-TopologyLinear":
        loss["prototype_target_circular"] = False
        loss["prototype_topology"] = {"id": "linear_index_v1"}
    elif variant == "T2-TopologyPermuted":
        loss["prototype_target_circular"] = True
        loss["prototype_topology"] = {"id": "permuted_index_v1", "permutation": _fixed_permutation()}
    elif variant == "T2-CLS":
        primary["head_type"] = "classifier"
        primary["router_use_prototype_margin"] = False
        _disable_bpa(loss)
    elif variant == "T2-NoRouterOracle":
        loss["router_oracle_weight"] = 0.0
    elif variant == "T2-ReliabilityOnly":
        primary["fusion_type"] = "reliability_mean"
        loss["router_oracle_weight"] = 0.0
    elif variant == "T2-Uniform":
        primary["fusion_type"] = "uniform_mean"
        loss["router_oracle_weight"] = 0.0
    elif variant == "T2-WholeOnly":
        config["temporal_missing"].update(
            {
                "schedule_id": "mmw_fair_whole_modality_v1",
                "panel_size": 480,
                "condition_counts": {
                    "clean": 120,
                    "drop1": 120,
                    "drop2": 120,
                    "drop3": 120,
                    "token20": 0,
                    "token40": 0,
                    "token60": 0,
                    "token80": 0,
                    "token90": 0,
                },
            }
        )
    elif variant == "T2-BPA2CMA":
        _disable_bpa(loss)
        loss.update({"use_amber_cma_analogue": True, "lambda_amber_cma": 0.2, "amber_cma_temperature": 0.2})
    elif variant not in {"T2", "S1"}:
        raise ValueError(f"Unsupported MMW TWC variant {variant!r}.")

def _set_physical_bpa(loss: dict[str, Any], topology: Mapping[str, Any]) -> None:
    descriptor = topology["descriptor"]
    if descriptor.get("topology_id") != "ula_dft_phase_cycle_v1":
        raise ValueError("MMW TWC training requires a verified ULA-DFT phase topology audit.")
    loss["prototype_target_circular"] = True
    loss["prototype_topology"] = {
        "id": "ula_dft_phase_cycle_v1",
        "descriptor_sha256": str(topology["descriptor_sha256"]),
        "audit_path": str(topology["path"]),
    }


def _disable_bpa(loss: dict[str, Any]) -> None:
    loss.update(
        {
            "use_beam_prototype_alignment": False,
            "lambda_proto": 0.0,
            "lambda_modality_proto": 0.0,
            "prototype_target_circular": False,
        }
    )
    loss.pop("prototype_topology", None)


def _training_mask_seed_algorithm(config: Mapping[str, Any]) -> str:
    schedule_id = str(config.get("temporal_missing", {}).get("schedule_id", ""))
    if schedule_id == "mmw_fair_pattern_v1":
        return TRAINING_MASK_SEED_ALGORITHM
    if schedule_id == "mmw_fair_whole_modality_v1":
        return WHOLE_ONLY_TRAINING_MASK_SEED_ALGORITHM
    raise ValueError(f"Unsupported MMW TWC training mask schedule {schedule_id!r}.")


def _effective_bpa_topology(config: Mapping[str, Any]) -> dict[str, str]:
    loss = config.get("loss", {}).get("u_mask_beam_jepa", {})
    if not isinstance(loss, Mapping) or not bool(loss.get("use_beam_prototype_alignment", False)):
        return {"id": "not_applicable", "descriptor_sha256": "not_applicable", "mapping_sha256": "not_applicable"}
    topology = loss.get("prototype_topology")
    if not isinstance(topology, Mapping):
        raise ValueError("Enabled TWC BPA requires an explicit prototype_topology mapping.")
    return {
        "id": str(topology.get("id", "")),
        "descriptor_sha256": str(topology.get("descriptor_sha256", "not_applicable")),
        "mapping_sha256": _sha256_payload({"id": topology.get("id"), "permutation": topology.get("permutation", [])}),
    }


def _fixed_permutation() -> list[int]:
    values = list(range(64))
    random.Random(20260718).shuffle(values)
    return values


def _load_topology(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    descriptor = payload.get("descriptor")
    descriptor_sha256 = payload.get("descriptor_sha256")
    if not isinstance(descriptor, dict) or not isinstance(descriptor_sha256, str):
        raise ValueError("MMW topology audit manifest lacks descriptor provenance.")
    if _sha256_payload(descriptor) != descriptor_sha256:
        raise ValueError("MMW topology audit descriptor checksum mismatch.")
    return {"path": str(path.resolve()), "descriptor": descriptor, "descriptor_sha256": descriptor_sha256}


def _plan_sha256(payload: Mapping[str, Any]) -> str:
    immutable_jobs = []
    for job in payload.get("jobs", []):
        if not isinstance(job, Mapping):
            raise ValueError("TWC plan job must be a mapping.")
        immutable_jobs.append(
            {
                key: job.get(key)
                for key in (
                    "variant",
                    "method",
                    "display_name",
                    "seed",
                    "matched_control",
                    "variant_role",
                    "allowed_config_diff",
                    "config_path",
                    "config_sha256",
                    "matched_control_config_path",
                    "matched_control_config_sha256",
                    "run_dir",
                    "evaluation_output_path",
                )
            }
        )
    return _sha256_payload(
        {
            "schema_version": payload.get("schema_version"),
            "protocol": payload.get("protocol"),
            "request": payload.get("request"),
            "request_sha256": payload.get("request_sha256"),
            "protocol_manifest": payload.get("protocol_manifest"),
            "topology_manifest": payload.get("topology_manifest"),
            "confirmation_splits_manifest": payload.get("confirmation_splits_manifest"),
            "jobs": immutable_jobs,
        }
    )


def _validate_plan(payload: Mapping[str, Any]) -> None:
    if int(payload.get("schema_version", -1)) != PLAN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported TWC training plan schema; expected v{PLAN_SCHEMA_VERSION}.")
    recorded = str(payload.get("plan_sha256", ""))
    if not recorded or recorded != _plan_sha256(payload):
        raise ValueError("TWC training plan immutable identity checksum mismatch.")
    request = payload.get("request")
    if not isinstance(request, Mapping) or str(payload.get("request_sha256", "")) != _sha256_payload(request):
        raise ValueError("TWC training plan request identity checksum mismatch.")
    if int(request.get("plan_schema_version", -1)) != PLAN_SCHEMA_VERSION or int(
        request.get("comparison_contract_version", -1)
    ) != COMPARISON_CONTRACT_VERSION:
        raise ValueError("TWC training plan does not carry the current comparison contract.")
    protocol_path = Path(str(payload.get("protocol_manifest", "")))
    if not protocol_path.is_file():
        raise FileNotFoundError(f"TWC training plan protocol is missing: {protocol_path}")
    load_protocol(protocol_path)
    topology_path = Path(str(payload.get("topology_manifest", "")))
    if not topology_path.is_file():
        raise FileNotFoundError(f"TWC training plan topology audit is missing: {topology_path}")
    _load_topology(topology_path)
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("TWC training plan has no jobs list.")
    for job in jobs:
        if not isinstance(job, Mapping):
            raise ValueError("TWC training plan job must be a mapping.")
        _validate_job_config(job)
        _validate_matched_control_config(job)


def _validate_job_config(job: Mapping[str, Any]) -> None:
    path = Path(str(job.get("config_path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"TWC job generated config is missing: {path}")
    expected = str(job.get("config_sha256", ""))
    if not expected or _sha256_file(path) != expected:
        raise ValueError(f"TWC job generated config checksum mismatch: {path}")


def _validate_matched_control_config(job: Mapping[str, Any]) -> None:
    variant = str(job.get("variant", ""))
    method = str(job.get("method", ""))
    if variant not in VARIANT_PROTOCOL or method != variant:
        raise ValueError(f"TWC job must declare a known variant as method: {variant!r}/{method!r}")
    expected_control = str(VARIANT_PROTOCOL[variant]["matched_control"])
    if str(job.get("matched_control", "")) != expected_control:
        raise ValueError(f"TWC job {variant} has an unexpected matched control.")
    configured_allowlist = job.get("allowed_config_diff")
    expected_allowlist = list(VARIANT_ALLOWED_CONFIG_DIFFS[variant])
    if configured_allowlist != expected_allowlist:
        raise ValueError(f"TWC job {variant} allowed_config_diff does not match the registered comparison contract.")

    control_path = Path(str(job.get("matched_control_config_path", "")))
    if not control_path.is_file():
        raise FileNotFoundError(f"TWC matched-control config is missing: {control_path}")
    expected_control_sha256 = str(job.get("matched_control_config_sha256", ""))
    if not expected_control_sha256 or _sha256_file(control_path) != expected_control_sha256:
        raise ValueError(f"TWC matched-control config checksum mismatch: {control_path}")
    if variant == expected_control:
        if expected_allowlist:
            raise ValueError(f"TWC self-control {variant} must not declare allowed config differences.")
        return

    candidate = _read_yaml_mapping(Path(str(job["config_path"])))
    control = _read_yaml_mapping(control_path)
    differences = [
        path
        for path in _config_diff_paths(control, candidate)
        if not _matches_path_prefix(path, _DYNAMIC_CONFIG_DIFF_PREFIXES)
    ]
    unexpected = [path for path in differences if not _matches_path_prefix(path, expected_allowlist)]
    if unexpected:
        raise ValueError(
            f"TWC variant {variant} changes fields outside its allowed_config_diff against {expected_control}: {unexpected}"
        )


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"TWC generated config must be a YAML mapping: {path}")
    return payload


def _config_diff_paths(left: Any, right: Any, *, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(_config_diff_paths(left[key], right[key], prefix=path))
        return differences
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [prefix]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(_config_diff_paths(left_item, right_item, prefix=f"{prefix}[{index}]"))
        return differences
    return [] if left == right else [prefix]


def _matches_path_prefix(path: str, prefixes: Iterable[str]) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}.") or path.startswith(f"{prefix}[") for prefix in prefixes)


def _all_weather_launcher() -> ModuleType:
    path = ROOT / "scripts" / "launch_mmw_all_weather_matrix.py"
    spec = importlib.util.spec_from_file_location("_mmw_all_weather_launcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load tracked MMW launcher: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _resolve_topology_path(value: str | None) -> Path:
    if value:
        path = _repo_path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Topology manifest is missing: {path}")
        return path
    paths = sorted(ROOT.glob(DEFAULT_TOPOLOGY_GLOB))
    verified = [path for path in paths if _read_json(path).get("descriptor", {}).get("topology_id") == "ula_dft_phase_cycle_v1"]
    if len(verified) != 1:
        raise ValueError(f"Expected exactly one verified MMW topology audit, found {len(verified)}.")
    return verified[0]


def _resolve_variants(phase: str, raw: str | None) -> tuple[str, ...]:
    if raw is None:
        values = MAIN_VARIANTS if phase == "main" else ABLATION_VARIANTS if phase == "ablation" else ()
    else:
        values = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(set(values) - set(VARIANT_PROTOCOL))
    if not values or unknown or len(set(values)) != len(values):
        raise ValueError(f"variants must be unique non-empty members of {tuple(VARIANT_PROTOCOL)}; unknown={unknown}")
    return tuple(values)


def _plan_name(variants: Iterable[str], seeds: Iterable[int]) -> str:
    variant_digest = hashlib.sha256(",".join(variants).encode("utf-8")).hexdigest()[:12]
    seed_text = "_".join(str(seed) for seed in seeds)
    return f"training_manifest_v{PLAN_SCHEMA_VERSION}_{variant_digest}_seeds_{seed_text}.json"


def _refresh_completed_jobs(jobs: list[dict[str, Any]]) -> None:
    for job in jobs:
        if job.get("status") != "done" and _completed_run(job):
            job.update({"status": "done", "return_code": 0, "recovered_at": _now()})
        if job.get("evaluation_status") != "done" and _completed_evaluation(job):
            job.update({"evaluation_status": "done", "evaluation_return_code": 0, "evaluation_recovered_at": _now()})


def _recover_orphaned_jobs(jobs: list[dict[str, Any]]) -> None:
    """Make a restarted launcher resume only phases whose prior process died."""
    for job in jobs:
        if job.get("status") == "running" and not _completed_run(job) and not _pid_is_running(job.get("pid")):
            recovered_at = _now()
            previous_pid = job.get("pid")
            checkpoint = Path(str(job["run_dir"])) / "checkpoints" / "last.pth"
            if checkpoint.is_file():
                job.update(
                    {
                        "status": "planned",
                        "gpu": None,
                        "pid": None,
                        "resume_from_orphaned_checkpoint": True,
                        "recovered_orphaned_training_at": recovered_at,
                    }
                )
                job.setdefault("recovery_history", []).append(
                    {
                        "kind": "orphaned_training_detected",
                        "at": recovered_at,
                        "previous_pid": previous_pid,
                        "checkpoint": str(checkpoint),
                    }
                )
            else:
                job.update(
                    {
                        "status": "failed",
                        "return_code": None,
                        "recovery_failure": "orphaned training has no resumable checkpoints/last.pth",
                        "recovered_orphaned_training_at": recovered_at,
                    }
                )
        if (
            job.get("evaluation_status") == "running"
            and not _completed_evaluation(job)
            and not _pid_is_running(job.get("evaluation_pid"))
        ):
            job.update({"evaluation_status": "planned", "recovered_orphaned_evaluation_at": _now()})


def _retry_failed_evaluations(jobs: list[dict[str, Any]]) -> bool:
    """Explicitly requeue only incomplete failed evaluation attempts."""
    changed = False
    for job in jobs:
        if job.get("evaluation_status") != "failed" or _completed_evaluation(job):
            continue
        requested_at = _now()
        job.setdefault("evaluation_retry_history", []).append(
            {
                "at": requested_at,
                "previous_return_code": job.get("evaluation_return_code"),
                "previous_end_time": job.get("evaluation_end_time"),
            }
        )
        job.update(
            {
                "evaluation_status": "planned",
                "evaluation_return_code": None,
                "evaluation_gpu": None,
                "evaluation_pid": None,
                "evaluation_retry_requested_at": requested_at,
            }
        )
        changed = True
    return changed


def _has_live_manifest_jobs(jobs: Iterable[Mapping[str, Any]]) -> bool:
    return any(
        (job.get("status") == "running" and _pid_is_running(job.get("pid")))
        or (job.get("evaluation_status") == "running" and _pid_is_running(job.get("evaluation_pid")))
        for job in jobs
    )


def _occupied_manifest_gpus(jobs: Iterable[Mapping[str, Any]]) -> set[int]:
    occupied: set[int] = set()
    for job in jobs:
        if job.get("status") == "running" and _pid_is_running(job.get("pid")):
            gpu = job.get("gpu")
            if isinstance(gpu, int):
                occupied.add(gpu)
        if job.get("evaluation_status") == "running" and _pid_is_running(job.get("evaluation_pid")):
            gpu = job.get("evaluation_gpu")
            if isinstance(gpu, int):
                occupied.add(gpu)
    return occupied


def _completed_run(job: Mapping[str, Any]) -> bool:
    run_dir = Path(str(job["run_dir"]))
    status_path = run_dir / "run_status.json"
    checkpoint = run_dir / "checkpoints" / "last.pth"
    if not status_path.is_file() or not checkpoint.is_file():
        return False
    try:
        return _read_json(status_path).get("state") == "complete"
    except ValueError:
        return False


def _completed_evaluation(job: Mapping[str, Any]) -> bool:
    target = Path(str(job.get("evaluation_output_path", "")))
    provenance = target.parent / "provenance.json"
    if not target.is_file() or not provenance.is_file():
        return False
    try:
        payload = _read_json(provenance)
    except ValueError:
        return False
    return payload.get("status") == "complete" and str(payload.get("metrics_path", "")) == str(target)


def _pid_is_running(value: Any) -> bool:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return False
    if pid <= 0 or not Path(f"/proc/{pid}").exists():
        return False
    try:
        state = (Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]).upper()
    except (OSError, IndexError):
        return False
    return state != "Z"


def _start_training_job(job: Mapping[str, Any], gpu: int) -> tuple[subprocess.Popen, Any]:
    log_path = Path(str(job["log_path"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    command = ["conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "kd-sensing-train", "--config", str(job["config_path"])]
    if bool(job.get("resume_from_orphaned_checkpoint", False)):
        command.append("--auto-resume")
    handle.write(f"\n[{_now()}] GPU{gpu}: {' '.join(command)}\n")
    handle.flush()
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
        }
    )
    return subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    ), handle


def _start_evaluation_job(
    job: Mapping[str, Any],
    manifest: Mapping[str, Any],
    gpu: int,
) -> tuple[subprocess.Popen, Any]:
    log_path = Path(str(job["evaluation_log_path"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    root = Path(str(job["run_dir"])).parents[1]
    command = [
        "conda",
        "run",
        "-n",
        "kd_mm_beam",
        "--no-capture-output",
        "python",
        "scripts/eval_mmw_twc_evidence.py",
        "--root",
        str(root),
        "--method",
        str(job["method"]),
        "--seed",
        str(job["seed"]),
        "--protocol-manifest",
        str(manifest["protocol_manifest"]),
        "--mechanism-trace",
    ]
    handle.write(f"\n[{_now()}] GPU{gpu}: {' '.join(command)}\n")
    handle.flush()
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
        }
    )
    return subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    ), handle


def _gpu_free_memory() -> dict[int, int]:
    command = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    output = subprocess.check_output(command, text=True)
    result = {}
    for line in output.splitlines():
        index, memory = (item.strip() for item in line.split(",", maxsplit=1))
        result[int(index)] = int(memory)
    return result


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def _validate_gpu_ids(gpus: Iterable[int], *, allow_gpu0_3: bool = False) -> tuple[int, ...]:
    values = tuple(int(gpu) for gpu in gpus)
    allowed = EXPANDED_ALLOWED_GPUS if allow_gpu0_3 else DEFAULT_ALLOWED_GPUS
    boundary = "0,1,2,3,4,5,6,7 with --allow-gpu0-3" if allow_gpu0_3 else "4,5,6,7"
    if not values or len(set(values)) != len(values) or any(gpu not in allowed for gpu in values):
        raise ValueError(f"--gpus must be a unique non-empty subset of {boundary}")
    return values


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be a mapping: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
