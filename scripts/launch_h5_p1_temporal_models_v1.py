#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = "outputs/h5_p1_temporal_models_v1"
DEFAULT_METHODS = "ours_c2_main,ours_b4_nonrouter_soft_jepa,ours_e5_low_lr_pcpg,amber_full,rmbp_mm"
S1_LIGHTWEIGHT_OUTPUT_ROOT = f"{DEFAULT_OUTPUT_ROOT}/s1_lightweight"
S1_LIGHTWEIGHT_METHODS = "S1,T2,T1,A1,A2,A3,T1+T2,J1"
PROFILE_METHODS = {"default": DEFAULT_METHODS, "s1_lightweight": S1_LIGHTWEIGHT_METHODS}
DEFAULT_C2_CONFIG = "outputs/final_c2_ablation_v1/generated_configs/a0_c2_full_main_seed{seed}.yaml"
DEFAULT_B4_CONFIG = "outputs/final_c2_ablation_v1/generated_configs/a1_b4_nonrouter_soft_jepa_seed{seed}.yaml"
DEFAULT_E5_CONFIG = "outputs/pcpg_radar_balance_v1/generated_configs/e5_pcpg_low_encoder_lr_seed{seed}.yaml"
DEFAULT_AMBER_CONFIG = "configs/fusion/amber_full_architecture.yaml"
DEFAULT_RMBP_CONFIG = "outputs/analysis/local_baselines/rmbp_mm/scene31/rmbp_mm/final_config.yaml"
MODALITIES = ["image", "radar", "gps", "lidar"]
COMMON_SCENES = [31, 32, 33, 34]
COMMON_DATASET_SPLIT = {
    "scenes": COMMON_SCENES,
    "train_scenes": COMMON_SCENES,
    "validation_scenes": COMMON_SCENES,
    "test_scenes": COMMON_SCENES,
    "split_protocol": "stratified_80_10_10",
    "split_strategy": "stratified_by_target_beam_per_scene_sequence_group",
    "split_group_identity_policy": "scene_id:seq_index",
    "split_seed": 42,
    "split_source_splits": ["train", "test"],
    "split_fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
}
BASELINE_METHODS = {"amber_full", "rmbp_mm"}
THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def method_specs() -> dict[str, dict[str, Any]]:
    temporal_only = {
        "difficulty": {"enabled": False, "profiles": []},
        "random_modality_dropout": {"enabled": False},
        "training": {
            "mask_sampler": "default",
            "missing_pattern_sampler": "default",
            "random_modality_dropout": {"enabled": False},
        },
        "loss": {
            "u_mask_beam_jepa": {
                "missing_pattern_sampler": "default",
                "missing_pattern": {"available_modalities": MODALITIES},
            }
        },
        "model": {"primary": {"mask_sampler": "temporal_missing"}},
    }
    proto = {
        "model": {"primary": {"head_type": "prototype", "use_beam_prototype_alignment": True}},
        "training": {
            "use_beam_prototype_alignment": True,
            "use_modality_prototype_loss": True,
            "use_circular_soft_targets": True,
        },
        "checkpoint": {"selection_metric": "avg_missing_top1"},
    }
    soft_hard = {"loss": {"hard_subset_weighting": {"enabled": True, "mode": "soft_static"}}}
    pcpg = {"model": {"primary": {"fusion_type": "pcpg", "pcpg_fuse_level": "logits"}}}
    no_branch = {"loss": {"branch_aux_loss": False, "radar_protect_loss": False, "unimodal_aux_weight": 0.0}}
    c2 = {
        "model": {
            "primary": {
                "fusion_type": "supervised_router",
                "router_supervision": "oracle",
                "router_distill_weight": 0.1,
                "router_focus_patterns": "missing_image,miss2,drop2",
                "router_use_pattern_features": True,
                "router_use_reliability_features": True,
                "router_use_prototype_margin": True,
            }
        },
        "loss": {
            "pcpg_radar_balance": {"enabled": True},
            "router_supervision": "oracle",
            "router_distill_weight": 0.1,
            "router_focus_patterns": "missing_image,miss2,drop2",
        },
    }
    low_lr = {
        "training": {
            "optimizer": {
                "parameter_groups": [
                    {"name": "slow_image_lidar_encoders", "module_patterns": ["encoders.image.*", "encoders.lidar.*"], "lr": 1.0e-5}
                ]
            }
        }
    }
    temporal = {
        "temporal_missing": {
            "enabled": True,
            "history_window": 5,
            "prediction_window": 1,
            "temporal_aggregation": "masked_mean",
            "mode": "stratified_modality_temporal",
            "mask_sampler": "stratified_modality_temporal",
            "train_missing_drop_counts": "0,1,2,3",
            "train_temporal_missing_rates": "0.0,0.2,0.4,0.6,0.8",
            "train_temporal_missing_types": "modality_level,frame_level,modality_frame,block",
            "ensure_at_least_one_cell": True,
            "ensure_at_least_one_frame": True,
            "ensure_at_least_one_modality": True,
            "seed": 0,
        }
    }
    amber_end_to_end = {
        "model": {
            "primary": {
                "encoders": {
                    "image": {"freeze_backbone": False},
                    "radar": {"freeze_backbone": False},
                    "lidar": {"freeze_backbone": False},
                }
            }
        }
    }
    rmbp_end_to_end = {
        "model": {
            "primary": {
                "encoders": {
                    "image": {"freeze_backbone": False, "unfreeze_stages": []},
                }
            }
        }
    }
    no_jepa = {
        "model": {"primary": {"use_jepa_loss": False}},
        "loss": {"use_jepa": False, "jepa_weight": 0.0},
    }
    linear_geometry = {
        "training": {
            "beam_label_circular": False,
            "circular_beam_distance": False,
            "use_circular_soft_targets": False,
            "use_gaussian_beam_targets": True,
            "proto_target_type": "gaussian",
        },
        "evaluation": {
            "beam_distance_circular": False,
            "circular_beam_distance": False,
            "dba_distance_mode": "linear",
            "metric_profile": "64_beam_linear_topk",
        },
    }
    classifier_head = {
        "model": {
            "primary": {
                "head_type": "classifier",
                "use_beam_prototype_alignment": False,
                "router_use_prototype_margin": False,
            }
        },
        "training": {
            "use_beam_prototype_alignment": False,
            "beam_proto_align_weight": 0.0,
            "lambda_proto": 0.0,
            "use_modality_prototype_loss": False,
            "modality_proto_weight": 0.0,
            "lambda_modality_proto": 0.0,
            "beam_label_circular": False,
            "circular_beam_distance": False,
            "use_circular_soft_targets": False,
            "use_gaussian_beam_targets": False,
            "proto_target_type": "onehot",
        },
        "loss": {
            "router_use_prototype_margin": False,
            "u_mask_beam_jepa": {
                "use_beam_prototype_alignment": False,
                "lambda_proto": 0.0,
                "lambda_modality_proto": 0.0,
            },
        },
        "evaluation": {
            "beam_distance_circular": False,
            "circular_beam_distance": False,
            "dba_distance_mode": "linear",
            "metric_profile": "64_beam_linear_topk",
        },
    }

    def s1_overrides(
        *,
        pooling_type: str = "masked_mean",
        use_mask_statistics: bool = False,
        confidence_gated_kl: bool = False,
        beam_monotonic_rank: bool = False,
    ) -> dict[str, Any]:
        superset = {
            "enabled": confidence_gated_kl or beam_monotonic_rank,
            "confidence_gated_kl": confidence_gated_kl,
            "kl_weight": 0.2 if confidence_gated_kl else 0.0,
            "temperature": 2.0,
            "beam_monotonic_rank": beam_monotonic_rank,
            "rank_weight": 0.1 if beam_monotonic_rank else 0.0,
            "rank_tolerance": 0.0,
            "feature_l2_weight": 0.0,
        }
        return _merge(
            proto,
            c2,
            soft_hard,
            no_branch,
            no_jepa,
            temporal_only,
            temporal,
            {
                "model": {
                    "primary": {
                        "fusion_type": "supervised_router",
                        "consume_missing_modality_metadata": True,
                        "use_mask_statistics": use_mask_statistics,
                        "temporal_pooling": {
                            "enabled": True,
                            "type": pooling_type,
                            "recency_decay": 1.0,
                            "hidden_dim": 32,
                        },
                    }
                },
                "temporal_missing": {
                    "preserve_unmasked_for_superset": confidence_gated_kl or beam_monotonic_rank,
                },
                "training": {"superset_consistency": superset},
                "loss": {
                    "u_mask_beam_jepa": {
                        "enabled": True,
                        "router_supervision": "oracle",
                        "router_distill_weight": 0.1,
                        "superset_consistency": superset,
                    }
                },
            },
        )

    s1_specs = {
        "S1": (s1_overrides(), "S1 masked temporal mean + supervised modality router"),
        "T2": (s1_overrides(confidence_gated_kl=True), "S1 + confidence-gated temporal superset KL"),
        "T1": (s1_overrides(beam_monotonic_rank=True), "S1 + circular beam-risk monotonic ranking"),
        "A1": (s1_overrides(use_mask_statistics=True), "S1 + router mask statistics"),
        "A2": (s1_overrides(pooling_type="fixed_recency"), "S1 + fixed-recency temporal pooling"),
        "A3": (s1_overrides(pooling_type="gap_aware_residual"), "S1 + gap-aware residual temporal pooling"),
        "T1+T2": (
            s1_overrides(confidence_gated_kl=True, beam_monotonic_rank=True),
            "S1 + temporal superset KL + circular beam-risk ranking",
        ),
        "J1": (
            s1_overrides(pooling_type="gap_aware_residual", confidence_gated_kl=True, beam_monotonic_rank=True),
            "S1 gap-aware residual pooling + temporal superset KL + circular beam-risk ranking",
        ),
        "S1-LG": (
            _merge(s1_overrides(), linear_geometry),
            "S1 + linear-Gaussian beam targets",
        ),
        "T2-LG": (
            _merge(s1_overrides(confidence_gated_kl=True), linear_geometry),
            "T2 + linear-Gaussian beam targets",
        ),
        "S1-CLS": (
            _merge(s1_overrides(), classifier_head),
            "S1 + classifier head without prototype losses",
        ),
        "T2-CLS": (
            _merge(s1_overrides(confidence_gated_kl=True), classifier_head),
            "T2 + classifier head without prototype losses",
        ),
    }
    return {
        "ours_c2_main": {
            "base_config": DEFAULT_C2_CONFIG,
            "overrides": _merge(proto, c2, soft_hard, no_branch, {"model": {"primary": {"use_jepa_loss": False}}, "loss": {"use_jepa": False, "jepa_weight": 0.0}}, temporal_only, temporal),
            "mapping": "u_mask_beam_jepa supervised_router + prototype + soft_static hard subset",
        },
        "ours_b4_nonrouter_soft_jepa": {
            "base_config": DEFAULT_B4_CONFIG,
            "overrides": _merge(proto, pcpg, soft_hard, {"model": {"primary": {"use_jepa_loss": True}}, "loss": {"use_jepa": True, "jepa_weight": 0.1}}, temporal_only, temporal),
            "mapping": "u_mask_beam_jepa pcpg + prototype + soft_static + JEPA",
        },
        "ours_e5_low_lr_pcpg": {
            "base_config": DEFAULT_E5_CONFIG,
            "overrides": _merge(proto, pcpg, low_lr, no_branch, {"loss": {"hard_subset_weighting": {"enabled": False, "mode": "none"}, "use_jepa": False, "jepa_weight": 0.0}}, temporal_only, temporal),
            "mapping": "u_mask_beam_jepa pcpg + low image/lidar encoder LR",
        },
        "amber_full": {
            "base_config": DEFAULT_AMBER_CONFIG,
            "overrides": _merge(temporal_only, temporal, amber_end_to_end),
            "mapping": "configs/fusion/amber_full_architecture.yaml modular AMBER full local reproduction, end-to-end finetuned",
        },
        "rmbp_mm": {
            "base_config": DEFAULT_RMBP_CONFIG,
            "overrides": _merge(temporal_only, temporal, rmbp_end_to_end),
            "mapping": "outputs/analysis/local_baselines/rmbp_mm/scene31/rmbp_mm/final_config.yaml + rmbp_channel_attention_fusion, end-to-end finetuned",
        },
        **{
            method: {"base_config": DEFAULT_C2_CONFIG, "overrides": overrides, "mapping": mapping}
            for method, (overrides, mapping) in s1_specs.items()
        },
    }


def plan_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = method_specs()
    methods = _csv(args.methods)
    seeds = [int(item) for item in _csv(args.seeds)]
    gpus = _csv(args.gpus)
    overrides = _method_config_overrides(args.method_config_overrides)
    jobs = []
    for method in methods:
        if method not in specs:
            raise ValueError(f"Unknown method {method!r}; choose from {sorted(specs)}.")
        spec = dict(specs[method])
        if method in overrides:
            spec["base_config"] = overrides[method]
            spec["mapping"] = f"user override config {overrides[method]}"
        for seed in seeds:
            index = len(jobs)
            gpu = gpus[(index // max(int(args.per_gpu), 1)) % len(gpus)] if gpus else ""
            output_dir = Path(args.output_root) / method / f"seed{seed}"
            config_path = Path(args.output_root) / "generated_configs" / f"{method}_seed{seed}.yaml"
            log_path = Path(args.output_root) / "logs" / f"{method}_seed{seed}.log"
            command = ["conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "kd-sensing-train", "--config", str(config_path)]
            if bool(getattr(args, "auto_resume", False)):
                command.append("--auto-resume")
            jobs.append({
                "method": method,
                "profile": str(getattr(args, "profile", "default")),
                "seed": seed,
                "gpu": gpu,
                "max_jobs": int(args.max_jobs),
                "per_gpu": int(args.per_gpu),
                "mask_sampler": str(args.mask_sampler),
                "torch_num_threads": int(args.torch_num_threads),
                "persistent_workers": bool(args.persistent_workers),
                "cmd": " ".join(command),
                "command": command,
                "status": "planned",
                "start_time": "",
                "end_time": "",
                "return_code": "",
                "log_path": str(log_path),
                "output_dir": str(output_dir),
                "history_window": int(args.history_window),
                "prediction_window": int(args.prediction_window),
                "config_path": str(config_path),
                "base_config": spec.get("base_config", ""),
                "mapping": spec.get("mapping", ""),
            })
    return jobs


def write_generated_configs(jobs: list[dict[str, Any]], args: argparse.Namespace) -> None:
    specs = method_specs()
    for job in jobs:
        if not job.get("base_config"):
            continue
        base_path = ROOT / str(job["base_config"]).format(seed=int(job["seed"]))
        if not base_path.exists():
            if not args.dry_run:
                raise FileNotFoundError(base_path)
            payload = {"_base_": str(job["base_config"])}
        else:
            payload = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
        payload = _merge(payload, specs[str(job["method"])]["overrides"])
        batch_size = int(args.baseline_batch_size) if str(job["method"]) in BASELINE_METHODS else int(args.umask_batch_size)
        payload = _merge(
            payload,
            {
                "experiment": {"name": str(job["method"]), "seed": int(job["seed"])},
                "training": {
                    "cpu_threads": {
                        "enabled": True,
                        "intra_op": int(args.torch_num_threads),
                        "inter_op": int(args.torch_num_interop_threads),
                    },
                },
                "output": {
                    "dir": str(Path(args.output_root) / str(job["method"])),
                    "run_name": f"seed{int(job['seed'])}",
                    "group_by_scene": False,
                    "progress": {"enabled": False},
                },
                "data": {
                    "dataset": {
                        **COMMON_DATASET_SPLIT,
                        "seq_len": int(args.history_window),
                        "num_pred": int(args.prediction_window),
                    },
                    "dataloader": {
                        "train_batch_size": batch_size,
                        "test_batch_size": batch_size,
                        "num_workers": int(args.num_workers),
                        "train_num_workers": int(args.num_workers),
                        "test_num_workers": int(args.num_workers),
                        "prefetch_factor": int(args.prefetch_factor),
                        "train_prefetch_factor": int(args.prefetch_factor),
                        "test_prefetch_factor": int(args.prefetch_factor),
                        "persistent_workers": bool(args.persistent_workers),
                        "train_persistent_workers": bool(args.persistent_workers),
                        "test_persistent_workers": bool(args.persistent_workers),
                        "pin_memory": True,
                    },
                },
                "model": {
                    "seq_length": int(args.history_window),
                    "num_pred": int(args.prediction_window),
                    "primary": {
                        "seq_length": int(args.history_window),
                        "num_pred": int(args.prediction_window),
                        "history_window": int(args.history_window),
                        "prediction_window": int(args.prediction_window),
                    },
                },
                "temporal_missing": {
                    "history_window": int(args.history_window),
                    "prediction_window": int(args.prediction_window),
                    "mask_sampler": str(args.mask_sampler),
                    "seed": int(args.temporal_missing_seed),
                },
            },
        )
        if args.max_epochs is not None:
            payload = _merge(payload, {"training": {"epochs": int(args.max_epochs), "max_epochs": int(args.max_epochs)}})
        if args.force:
            payload = _merge(payload, {"output": {"overwrite": True}})
        path = Path(str(job["config_path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_manifest(jobs: list[dict[str, Any]], output_root: str) -> None:
    fields = ["method", "profile", "seed", "gpu", "max_jobs", "per_gpu", "mask_sampler", "torch_num_threads", "persistent_workers", "cmd", "status", "start_time", "end_time", "return_code", "log_path", "output_dir", "history_window", "prediction_window", "config_path", "base_config", "mapping"]
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "job_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            writer.writerow({field: job.get(field, "") for field in fields})
    (root / "job_manifest.json").write_text(json.dumps(jobs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_jobs(jobs: list[dict[str, Any]], args: argparse.Namespace) -> int:
    running: list[tuple[subprocess.Popen, dict[str, Any], Any]] = []
    failed: list[dict[str, Any]] = []
    for job in jobs:
        if job["status"] == "blocked":
            failed.append(job)
            continue
        if args.skip_completed and not args.force and (Path(str(job["output_dir"])) / "run_status.json").exists():
            job["status"] = "skipped"
            write_manifest(jobs, args.output_root)
            continue
        while len(running) >= int(args.max_jobs) or _gpu_count(running, str(job["gpu"])) >= int(args.per_gpu):
            failed.extend(_poll(running))
            write_manifest(jobs, args.output_root)
            time.sleep(1.0)
        log_path = Path(str(job["log_path"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w" if args.force else "a", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
        env["CUDA_DEVICE_ORDER"] = env.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        env["PYTHONUNBUFFERED"] = "1"
        thread_env = {**THREAD_ENV_DEFAULTS, "OMP_NUM_THREADS": str(args.torch_num_threads), "MKL_NUM_THREADS": str(args.torch_num_threads), "OPENBLAS_NUM_THREADS": str(args.torch_num_threads)}
        for key, value in thread_env.items():
            env[key] = value
        job["status"] = "running"
        job["start_time"] = _now()
        write_manifest(jobs, args.output_root)
        running.append((subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT), job, handle))
    while running:
        failed.extend(_poll(running, wait=True))
        write_manifest(jobs, args.output_root)
    if failed:
        fields = ["method", "seed", "gpu", "status", "return_code", "log_path", "output_dir", "mapping"]
        with (Path(args.output_root) / "failed_jobs.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for job in failed:
                writer.writerow({field: job.get(field, "") for field in fields})
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch H5/P1 temporal missing matrix v1 experiments.")
    parser.add_argument("--profile", choices=tuple(PROFILE_METHODS), default="default")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max_jobs", "--max-jobs", type=int, default=8)
    parser.add_argument("--per_gpu", "--per-gpu", type=int, default=1)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--history_window", "--history-window", type=int, default=5)
    parser.add_argument("--prediction_window", "--prediction-window", type=int, default=1)
    parser.add_argument("--mask_sampler", "--mask-sampler", default="stratified_modality_temporal")
    parser.add_argument("--output_root", "--output-root", default=None)
    parser.add_argument("--temporal_missing_seed", "--temporal-missing-seed", type=int, default=0)
    parser.add_argument("--umask_batch_size", "--umask-batch-size", type=int, default=64)
    parser.add_argument("--baseline_batch_size", "--baseline-batch-size", type=int, default=128)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", "--prefetch-factor", type=int, default=2)
    parser.add_argument("--torch_num_threads", "--torch-num-threads", type=int, default=None)
    parser.add_argument("--torch_num_interop_threads", "--torch-num-interop-threads", type=int, default=1)
    parser.add_argument("--persistent_workers", "--persistent-workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no_persistent_workers", "--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--max_epochs", "--max-epochs", type=int, default=None)
    parser.add_argument("--method_config_overrides", "--method-config-overrides", default="")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--skip_completed", "--skip-completed", action="store_true")
    parser.add_argument("--auto_resume", "--auto-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(persistent_workers=None)
    args = parser.parse_args(argv)
    args.methods = args.methods or PROFILE_METHODS[args.profile]
    args.seeds = args.seeds or ("1" if args.profile == "s1_lightweight" else "1,2,3")
    args.output_root = args.output_root or (
        S1_LIGHTWEIGHT_OUTPUT_ROOT if args.profile == "s1_lightweight" else DEFAULT_OUTPUT_ROOT
    )
    if args.torch_num_threads is None:
        args.torch_num_threads = 12 if args.profile == "s1_lightweight" else 1
    if args.persistent_workers is None:
        args.persistent_workers = args.profile == "s1_lightweight"
    jobs = plan_jobs(args)
    write_generated_configs(jobs, args)
    write_manifest(jobs, args.output_root)
    if args.dry_run:
        for job in jobs:
            print(f"{job['method']} seed{job['seed']} gpu{job['gpu']}: {job['cmd']} [{job['status']}]")
        return 0
    return run_jobs(jobs, args)


def _merge(*items: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        for key, value in item.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = _merge(out[key], value)
            else:
                out[key] = value
    return out


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _method_config_overrides(raw: str) -> dict[str, str]:
    if not raw:
        return {}
    if raw.strip().startswith("{"):
        return {str(k): str(v) for k, v in json.loads(raw).items()}
    result = {}
    for item in _csv(raw):
        method, path = item.split("=", 1)
        result[method.strip()] = path.strip()
    return result


def _gpu_count(running: list[tuple[subprocess.Popen, dict[str, Any], Any]], gpu: str) -> int:
    return sum(1 for proc, job, _ in running if proc.poll() is None and str(job.get("gpu")) == gpu)


def _poll(running: list[tuple[subprocess.Popen, dict[str, Any], Any]], *, wait: bool = False) -> list[dict[str, Any]]:
    failed = []
    for item in list(running):
        proc, job, handle = item
        code = proc.wait() if wait else proc.poll()
        if code is None:
            continue
        handle.close()
        job["return_code"] = int(code)
        job["end_time"] = _now()
        job["status"] = "failed" if code else "done"
        if code:
            failed.append(job)
        running.remove(item)
    return failed


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
