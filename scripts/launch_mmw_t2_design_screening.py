#!/usr/bin/env python3
"""Launch staged, development-only MMW T2 design screens from the H4 profile."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import yaml

from launch_mmw_all_weather_matrix import MODALITIES, ROOT, build_config, domains, preflight
from launch_mmw_t2_hyperparameter_screening import (
    _force_single_probe_gpu,
    _sha256_payload,
    build_inner_validation_domains,
    collect_probe_results,
    probe_training_step,
    validate_batch_size,
)


DEFAULT_OUTPUT_ROOT = "outputs/mmw_t2_design_screening_v1"
H4_PROFILE = "umask_h4_v1"
CONTROL = "H4-control"
SELECTION_RULE = {
    "score": "0.20*clean + 0.20*mean(drop1,drop2,drop3) + 0.25*temporal_auc + 0.35*temporal_drop80",
    "minimum_delta_pp": {
        "j": 0.5,
        "clean": -0.5,
        "modality_missing_mean": -0.5,
        "temporal_drop80": -0.5,
    },
    "maximum_promoted_candidates": 2,
    "development_only": True,
}


def _protocol(*, matched_control: str | None, allowed_fields: list[str], wave: str) -> dict[str, Any]:
    return {
        "matched_control": matched_control,
        "allowed_effective_fields": allowed_fields,
        "wave": wave,
    }


VARIANT_PROTOCOL: dict[str, dict[str, Any]] = {
    CONTROL: _protocol(matched_control=None, allowed_fields=[], wave="capacity"),
    "D48": _protocol(
        matched_control=CONTROL,
        allowed_fields=["model.primary.d_model", "model.primary.encoders.*.output_dim"],
        wave="capacity",
    ),
    "D96": _protocol(
        matched_control=CONTROL,
        allowed_fields=["model.primary.d_model", "model.primary.encoders.*.output_dim"],
        wave="capacity",
    ),
    "RouterH32": _protocol(
        matched_control=CONTROL,
        allowed_fields=["model.primary.router_hidden_dim"],
        wave="capacity",
    ),
    "RouterH128": _protocol(
        matched_control=CONTROL,
        allowed_fields=["model.primary.router_hidden_dim"],
        wave="capacity",
    ),
    "RouterNoPattern": _protocol(
        matched_control=CONTROL,
        allowed_fields=["model.primary.router_use_pattern_features"],
        wave="capacity",
    ),
    "GPSH32": _protocol(
        matched_control=CONTROL,
        allowed_fields=["model.primary.encoders.gps.hidden_size"],
        wave="capacity",
    ),
    "GPSH128": _protocol(
        matched_control=CONTROL,
        allowed_fields=["model.primary.encoders.gps.hidden_size"],
        wave="capacity",
    ),
    "EncoderImageResNet18": _protocol(
        matched_control=CONTROL,
        allowed_fields=["model.primary.encoders.image"],
        wave="structure",
    ),
    "EncoderLidarResNet18": _protocol(
        matched_control=CONTROL,
        allowed_fields=["model.primary.encoders.lidar"],
        wave="structure",
    ),
    "EncoderBothResNet18": _protocol(
        matched_control=CONTROL,
        allowed_fields=["model.primary.encoders.image", "model.primary.encoders.lidar"],
        wave="structure",
    ),
    "FusionReliabilityMean": _protocol(
        matched_control=CONTROL,
        allowed_fields=["model.primary.fusion_type", "loss.u_mask_beam_jepa.router_oracle_weight"],
        wave="structure",
    ),
    "TemporalAttention": _protocol(
        matched_control=CONTROL,
        allowed_fields=["model.primary.temporal_pooling"],
        wave="structure",
    ),
    "GPSJitter005": _protocol(
        matched_control=CONTROL,
        allowed_fields=["model.primary.encoders.gps.normalized_feature_jitter_std"],
        wave="structure",
    ),
    "GPSDropout020": _protocol(
        matched_control=CONTROL,
        allowed_fields=["model.primary.encoders.gps.dropout"],
        wave="structure",
    ),
    "BPA-temp-008": _protocol(
        matched_control=CONTROL,
        allowed_fields=["model.primary.beam_proto_temperature"],
        wave="bpa",
    ),
    "BPA-temp-015": _protocol(
        matched_control=CONTROL,
        allowed_fields=["model.primary.beam_proto_temperature"],
        wave="bpa",
    ),
    "BPA-sigma-15": _protocol(
        matched_control=CONTROL,
        allowed_fields=["loss.u_mask_beam_jepa.beam_label_sigma"],
        wave="bpa",
    ),
    "BPA-sigma-30": _protocol(
        matched_control=CONTROL,
        allowed_fields=["loss.u_mask_beam_jepa.beam_label_sigma"],
        wave="bpa",
    ),
    "BPA-proto-025": _protocol(
        matched_control=CONTROL,
        allowed_fields=["loss.u_mask_beam_jepa.lambda_proto"],
        wave="bpa",
    ),
    "BPA-modality-015": _protocol(
        matched_control=CONTROL,
        allowed_fields=["loss.u_mask_beam_jepa.lambda_modality_proto"],
        wave="bpa",
    ),
    "BPA-both-strong": _protocol(
        matched_control=CONTROL,
        allowed_fields=["loss.u_mask_beam_jepa.lambda_proto", "loss.u_mask_beam_jepa.lambda_modality_proto"],
        wave="bpa",
    ),
    "NoBPA-control": _protocol(
        matched_control=CONTROL,
        allowed_fields=[
            "loss.u_mask_beam_jepa.use_beam_prototype_alignment",
            "loss.u_mask_beam_jepa.lambda_proto",
            "loss.u_mask_beam_jepa.lambda_modality_proto",
        ],
        wave="objective",
    ),
    "CMA-w005-t020": _protocol(
        matched_control="NoBPA-control",
        allowed_fields=["loss.u_mask_beam_jepa.use_amber_cma_analogue", "loss.u_mask_beam_jepa.lambda_amber_cma"],
        wave="objective",
    ),
    "CMA-w010-t020": _protocol(
        matched_control="NoBPA-control",
        allowed_fields=["loss.u_mask_beam_jepa.use_amber_cma_analogue", "loss.u_mask_beam_jepa.lambda_amber_cma"],
        wave="objective",
    ),
    "CMA-w020-t020": _protocol(
        matched_control="NoBPA-control",
        allowed_fields=["loss.u_mask_beam_jepa.use_amber_cma_analogue", "loss.u_mask_beam_jepa.lambda_amber_cma"],
        wave="objective",
    ),
    "CMA-w010-t010": _protocol(
        matched_control="NoBPA-control",
        allowed_fields=[
            "loss.u_mask_beam_jepa.use_amber_cma_analogue",
            "loss.u_mask_beam_jepa.lambda_amber_cma",
            "loss.u_mask_beam_jepa.amber_cma_temperature",
        ],
        wave="objective",
    ),
    "CMA-w010-t040": _protocol(
        matched_control="NoBPA-control",
        allowed_fields=[
            "loss.u_mask_beam_jepa.use_amber_cma_analogue",
            "loss.u_mask_beam_jepa.lambda_amber_cma",
            "loss.u_mask_beam_jepa.amber_cma_temperature",
        ],
        wave="objective",
    ),
    "KL-w010": _protocol(
        matched_control=CONTROL,
        allowed_fields=["loss.u_mask_beam_jepa.superset_consistency.kl_weight"],
        wave="objective",
    ),
    "KL-w050": _protocol(
        matched_control=CONTROL,
        allowed_fields=["loss.u_mask_beam_jepa.superset_consistency.kl_weight"],
        wave="objective",
    ),
}

WAVES: dict[str, tuple[str, ...]] = {
    "capacity": (CONTROL, "D48", "D96", "RouterH32", "RouterH128", "RouterNoPattern", "GPSH32", "GPSH128"),
    "structure": (
        CONTROL,
        "EncoderImageResNet18",
        "EncoderLidarResNet18",
        "EncoderBothResNet18",
        "FusionReliabilityMean",
        "TemporalAttention",
        "GPSJitter005",
        "GPSDropout020",
    ),
    "bpa": (
        CONTROL,
        "BPA-temp-008",
        "BPA-temp-015",
        "BPA-sigma-15",
        "BPA-sigma-30",
        "BPA-proto-025",
        "BPA-modality-015",
        "BPA-both-strong",
    ),
    "objective": (
        CONTROL,
        "NoBPA-control",
        "CMA-w005-t020",
        "CMA-w010-t020",
        "CMA-w020-t020",
        "CMA-w010-t010",
        "CMA-w010-t040",
        "KL-w050",
    ),
}


def build_design_config(
    candidate: str,
    output_root: Path,
    *,
    seed: int,
    batch_size: int,
    epochs: int = 40,
    domain_inventory: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if candidate not in VARIANT_PROTOCOL:
        raise ValueError(f"Unknown T2 design candidate {candidate!r}.")
    validate_batch_size(batch_size)
    if int(epochs) != 40:
        raise ValueError("T2 design screening is fixed to 40 epochs.")
    payload = build_config(
        "T2",
        output_root,
        seed=int(seed),
        smoke=False,
        epochs=int(epochs),
        batch_size=int(batch_size),
        umask_training_profile=H4_PROFILE,
    )
    if domain_inventory is not None:
        payload.setdefault("data", {}).setdefault("dataset", {})["domains"] = deepcopy(domain_inventory)
    _apply_candidate(payload, candidate)
    payload.setdefault("experiment", {}).update({"name": candidate, "t2_design_candidate": candidate})
    training = payload.setdefault("training", {})
    training.update(
        {
            "epochs": int(epochs),
            "max_epochs": int(epochs),
            "validation": {"interval_epochs": 5},
            "model_selection": {"enabled": False},
            "use_early_stopping": False,
        }
    )
    payload["output"] = {
        "dir": str(output_root / candidate),
        "run_name": f"seed{int(seed)}",
        "group_by_scene": False,
        "overwrite": False,
        "progress": {"enabled": False},
        "tensorboard": {"enabled": False},
    }
    profile = payload.setdefault("mmw_all_weather_protocol", {}).get("training_profile")
    if not isinstance(profile, dict) or profile.get("id") != H4_PROFILE or not profile.get("sha256"):
        raise ValueError("H4 design candidate must retain complete umask_h4_v1 training profile provenance.")
    protocol = VARIANT_PROTOCOL[candidate]
    payload["mmw_t2_design_screening"] = {
        "protocol": "mmw_t2_design_screening_v1",
        "candidate_id": candidate,
        "wave": protocol["wave"],
        "matched_control": protocol["matched_control"],
        "allowed_effective_fields": list(protocol["allowed_effective_fields"]),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "checkpoint_policy": "fixed_epoch_last_pth",
        "selection_split": "group_safe_inner_validation_only",
        "development_only": True,
        "claim_eligible": False,
        "training_profile_id": str(profile["id"]),
        "training_profile_sha256": str(profile["sha256"]),
        "selection_rule": deepcopy(SELECTION_RULE),
    }
    payload["mmw_t2_design_screening"]["config_sha256"] = _sha256_payload(payload)
    return payload


def _apply_candidate(payload: dict[str, Any], candidate: str) -> None:
    primary = payload.setdefault("model", {}).setdefault("primary", {})
    encoders = primary.setdefault("encoders", {})
    loss = payload.setdefault("loss", {}).setdefault("u_mask_beam_jepa", {})
    if candidate == CONTROL:
        return
    if candidate in {"D48", "D96"}:
        width = 48 if candidate == "D48" else 96
        primary["d_model"] = width
        payload.setdefault("model", {})["d_model"] = width
        for config in encoders.values():
            if isinstance(config, dict):
                config["output_dim"] = width
        return
    if candidate in {"RouterH32", "RouterH128"}:
        primary["router_hidden_dim"] = 32 if candidate == "RouterH32" else 128
        return
    if candidate == "RouterNoPattern":
        primary["router_use_pattern_features"] = False
        return
    gps = encoders.setdefault("gps", {})
    if candidate in {"GPSH32", "GPSH128"}:
        gps["hidden_size"] = 32 if candidate == "GPSH32" else 128
        return
    if candidate == "EncoderImageResNet18":
        _set_resnet_encoder(encoders.setdefault("image", {}), output_dim=int(primary.get("d_model", 64)))
        return
    if candidate == "EncoderLidarResNet18":
        _set_resnet_encoder(encoders.setdefault("lidar", {}), output_dim=int(primary.get("d_model", 64)))
        return
    if candidate == "EncoderBothResNet18":
        output_dim = int(primary.get("d_model", 64))
        _set_resnet_encoder(encoders.setdefault("image", {}), output_dim=output_dim)
        _set_resnet_encoder(encoders.setdefault("lidar", {}), output_dim=output_dim)
        return
    if candidate == "FusionReliabilityMean":
        primary["fusion_type"] = "reliability_mean"
        loss["router_oracle_weight"] = 0.0
        return
    if candidate == "TemporalAttention":
        primary["temporal_pooling"] = {"enabled": True, "type": "masked_attention"}
        return
    if candidate == "GPSJitter005":
        gps["normalized_feature_jitter_std"] = 0.05
        return
    if candidate == "GPSDropout020":
        gps["dropout"] = 0.2
        return
    if candidate == "BPA-temp-008":
        primary["beam_proto_temperature"] = 0.08
        return
    if candidate == "BPA-temp-015":
        primary["beam_proto_temperature"] = 0.15
        return
    if candidate == "BPA-sigma-15":
        loss["beam_label_sigma"] = 1.5
        return
    if candidate == "BPA-sigma-30":
        loss["beam_label_sigma"] = 3.0
        return
    if candidate == "BPA-proto-025":
        loss["lambda_proto"] = 0.25
        return
    if candidate == "BPA-modality-015":
        loss["lambda_modality_proto"] = 0.15
        return
    if candidate == "BPA-both-strong":
        loss.update({"lambda_proto": 0.25, "lambda_modality_proto": 0.15})
        return
    if candidate == "NoBPA-control":
        _disable_bpa(loss)
        return
    if candidate.startswith("CMA-"):
        _disable_bpa(loss)
        loss["use_amber_cma_analogue"] = True
        weights = {
            "CMA-w005-t020": (0.05, 0.2),
            "CMA-w010-t020": (0.1, 0.2),
            "CMA-w020-t020": (0.2, 0.2),
            "CMA-w010-t010": (0.1, 0.1),
            "CMA-w010-t040": (0.1, 0.4),
        }
        loss["lambda_amber_cma"], loss["amber_cma_temperature"] = weights[candidate]
        return
    if candidate.startswith("KL-"):
        superset = deepcopy(loss.get("superset_consistency", {}))
        superset.update(
            {
                "enabled": True,
                "confidence_gated_kl": True,
                "kl_weight": 0.1 if candidate == "KL-w010" else 0.5,
                "temperature": 2.0,
            }
        )
        loss["superset_consistency"] = superset
        return
    raise ValueError(f"No override implementation registered for {candidate!r}.")


def _set_resnet_encoder(config: dict[str, Any], *, output_dim: int) -> None:
    config.update(
        {
            "type": "resnet18_imagenet_rgb",
            "output_dim": int(output_dim),
            "pretrained": False,
            "freeze_backbone": False,
        }
    )


def _disable_bpa(loss: dict[str, Any]) -> None:
    loss.update(
        {
            "use_beam_prototype_alignment": False,
            "lambda_proto": 0.0,
            "lambda_modality_proto": 0.0,
        }
    )


def build_jobs(variants: tuple[str, ...], gpus: tuple[int, ...], output_root: Path, *, seed: int) -> list[dict[str, Any]]:
    if not variants or len(variants) != len(set(variants)):
        raise ValueError("Candidates must be non-empty and unique.")
    if len(gpus) != len(variants) or len(gpus) != len(set(gpus)) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Each candidate requires one unique non-negative GPU.")
    return [
        {
            "candidate": candidate,
            "seed": int(seed),
            "gpu": int(gpu),
            "config_path": output_root / "generated_configs" / f"{candidate}_seed{seed}.yaml",
            "log_path": output_root / "logs" / f"{candidate}_seed{seed}.log",
            "run_dir": output_root / candidate / f"seed{seed}",
            "status": "planned",
        }
        for candidate, gpu in zip(variants, gpus)
    ]


def write_design_plan(
    output_root: Path,
    *,
    variants: tuple[str, ...],
    gpus: tuple[int, ...],
    seed: int,
    batch_size: int,
    batch_probe: dict[str, Any] | None,
) -> tuple[Path, dict[str, Any]]:
    validate_batch_size(batch_size)
    manifest_path = output_root / "design_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing design manifest: {manifest_path}")
    jobs = build_jobs(variants, gpus, output_root, seed=seed)
    conflicts = [manifest_path]
    for job in jobs:
        conflicts.extend((job["config_path"], job["log_path"], job["run_dir"]))
    existing = [path for path in conflicts if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing design artifacts:\n" + "\n".join(map(str, existing)))

    inner_domains, inner_split = build_inner_validation_domains(output_root, seed=seed)
    report = preflight(inner_domains, enabled_modalities=MODALITIES)
    if report.get("status") != "ready":
        raise RuntimeError(f"MMW preflight failed: {report.get('failures', [])}")
    generated = output_root / "generated_configs"
    logs = output_root / "logs"
    generated.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "preflight.json", report)
    _write_json(output_root / "inner_split_manifest.json", inner_split)

    candidate_fingerprints: dict[str, str] = {}
    profile_fingerprints: dict[str, str] = {}
    for job in jobs:
        config = build_design_config(
            job["candidate"],
            output_root,
            seed=seed,
            batch_size=batch_size,
            domain_inventory=inner_domains,
        )
        job["config_path"].write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        screen = config["mmw_t2_design_screening"]
        candidate_fingerprints[job["candidate"]] = str(screen["config_sha256"])
        profile_fingerprints[job["candidate"]] = str(screen["training_profile_sha256"])
        job["config_path"] = str(job["config_path"].relative_to(ROOT))
        job["log_path"] = str(job["log_path"].relative_to(ROOT))
        job["run_dir"] = str(job["run_dir"].relative_to(ROOT))

    manifest: dict[str, Any] = {
        "protocol": "mmw_t2_design_screening_v1",
        "created_at": _now(),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "epochs": 40,
        "checkpoint_policy": "fixed_epoch_last_pth",
        "development_only": True,
        "claim_eligible": False,
        "selection_split": "group_safe_inner_validation_only",
        "selection_rule": deepcopy(SELECTION_RULE),
        "inner_split_fingerprint": str(inner_split["fingerprint"]),
        "training_profile_id": H4_PROFILE,
        "profile_fingerprints": profile_fingerprints,
        "candidate_fingerprints": candidate_fingerprints,
        "batch_probe": deepcopy(batch_probe),
        "jobs": jobs,
    }
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


def launch_jobs(manifest_path: Path, manifest: dict[str, Any]) -> int:
    running: list[tuple[Any, dict[str, Any], Any]] = []
    for job in manifest["jobs"]:
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "4",
            }
        )
        command = [
            "conda",
            "run",
            "-n",
            "kd_mm_beam",
            "--no-capture-output",
            "kd-sensing-train",
            "--config",
            job["config_path"],
        ]
        handle = (ROOT / job["log_path"]).open("w", encoding="utf-8")
        job.update({"status": "running", "start_time": _now(), "command": command})
        running.append((subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT), job, handle))
    _write_json(manifest_path, manifest)
    failed = False
    for process, job, handle in running:
        code = process.wait()
        handle.close()
        job.update({"status": "done" if code == 0 else "failed", "return_code": code, "end_time": _now()})
        failed = failed or code != 0
        _write_json(manifest_path, manifest)
    return 1 if failed else 0


def qualifies_for_promotion(control: dict[str, float], candidate: dict[str, float]) -> bool:
    """Apply the pre-registered single-seed promotion guard in percentage points."""
    required = ("j", "clean", "modality_missing_mean", "temporal_drop80")
    if any(key not in control or key not in candidate for key in required):
        raise ValueError("control and candidate summaries must include j, clean, modality_missing_mean, temporal_drop80.")
    delta = {key: float(candidate[key]) - float(control[key]) for key in required}
    limits = SELECTION_RULE["minimum_delta_pp"]
    return bool(
        delta["j"] >= float(limits["j"])
        and delta["clean"] >= float(limits["clean"])
        and delta["modality_missing_mean"] >= float(limits["modality_missing_mean"])
        and delta["temporal_drop80"] >= float(limits["temporal_drop80"])
    )


def _probe(
    output_root: Path,
    *,
    batch_size: int,
    physical_gpu: int,
    candidate: str,
    seed: int,
    memory_fraction_limit: float,
) -> int:
    if candidate not in VARIANT_PROTOCOL:
        raise ValueError(f"Unknown probe candidate {candidate!r}.")
    _force_single_probe_gpu(physical_gpu)
    probe_root = output_root / "batch_probes" / candidate / f"gpu{physical_gpu}_batch{batch_size}"
    probe_root.mkdir(parents=True, exist_ok=True)
    report = preflight(domains(), enabled_modalities=MODALITIES)
    if report.get("status") != "ready":
        raise RuntimeError(f"MMW probe preflight failed: {report.get('failures', [])}")
    config = build_design_config(
        candidate,
        probe_root,
        seed=seed,
        batch_size=batch_size,
        domain_inventory=[domains()[0]],
    )
    config["mmw_t2_design_screening"].update(
        {
            "probe_scope": "single_representative_domain",
            "probe_scope_reason": "per-step peak memory follows input shape and candidate model, not 15-domain sampler inventory",
        }
    )
    config_path = probe_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = probe_training_step(
        config_path,
        probe_root / "result.json",
        physical_gpu=physical_gpu,
        memory_fraction_limit=memory_fraction_limit,
    )
    result.update(
        {
            "candidate": candidate,
            "training_profile_id": H4_PROFILE,
            "config_sha256": config["mmw_t2_design_screening"]["config_sha256"],
            "preflight_status": report["status"],
        }
    )
    _write_json(probe_root / "result.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "safe" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch a development-only MMW T2 H4 design-screen wave.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--wave", choices=tuple(WAVES), default="capacity")
    parser.add_argument("--variants", default=None, help="Comma-separated explicit candidates; overrides --wave.")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--launch-existing",
        action="store_true",
        help="Launch the verified design_manifest.json created by an earlier --dry-run without regenerating splits/configs.",
    )
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-candidate", default="D96")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--memory-fraction-limit", type=float, default=0.90)
    parser.add_argument("--probe-results-root", action="append", default=[])
    parser.add_argument(
        "--allow-objective-wave",
        action="store_true",
        help="Allow CMA/KL candidates only after the separate BPA/CMA formal change has been closed.",
    )
    args = parser.parse_args(argv)
    try:
        batch_size = validate_batch_size(args.batch_size)
    except ValueError as exc:
        parser.error(str(exc))
    if args.seed <= 0:
        parser.error("seed must be positive.")
    if not 0.0 < float(args.memory_fraction_limit) <= 1.0:
        parser.error("memory-fraction-limit must be in (0, 1].")
    output_root = ROOT / str(args.output_root)
    if args.probe_only:
        if args.physical_gpu is None or int(args.physical_gpu) < 0:
            parser.error("--probe-only requires a non-negative --physical-gpu.")
        try:
            return _probe(
                output_root,
                batch_size=batch_size,
                physical_gpu=int(args.physical_gpu),
                candidate=str(args.probe_candidate),
                seed=int(args.seed),
                memory_fraction_limit=float(args.memory_fraction_limit),
            )
        except (RuntimeError, ValueError, FileExistsError) as exc:
            parser.error(str(exc))

    variants = (
        tuple(item.strip() for item in str(args.variants).split(",") if item.strip())
        if args.variants
        else WAVES[str(args.wave)]
    )
    unknown = sorted(set(variants) - set(VARIANT_PROTOCOL))
    if unknown:
        parser.error(f"Unknown candidate(s): {', '.join(unknown)}")
    if any(VARIANT_PROTOCOL[candidate]["wave"] == "objective" for candidate in variants) and not args.allow_objective_wave:
        parser.error("The CMA/KL objective wave requires --allow-objective-wave after the BPA/CMA formal change is closed.")
    try:
        gpus = tuple(int(item.strip()) for item in str(args.gpus).split(",") if item.strip())
    except ValueError as exc:
        parser.error(str(exc))
    if len(gpus) != len(variants):
        parser.error(f"Expected {len(variants)} GPUs for {len(variants)} candidates, got {len(gpus)}.")
    try:
        batch_probe = None
        if args.probe_results_root:
            roots = [ROOT / item if not Path(item).is_absolute() else Path(item) for item in args.probe_results_root]
            batch_probe = collect_probe_results(roots, gpus=gpus, memory_fraction_limit=float(args.memory_fraction_limit))
            if int(batch_probe["selected_common_batch_size"]) != batch_size:
                raise ValueError(
                    "Requested batch size does not equal highest trusted common probe result: "
                    f"requested={batch_size}, selected={batch_probe['selected_common_batch_size']}."
                )
        elif not args.dry_run:
            raise ValueError("Launching training requires --probe-results-root evidence for all requested GPUs.")
        if args.launch_existing:
            if args.dry_run:
                raise ValueError("--launch-existing cannot be combined with --dry-run.")
            manifest_path = output_root / "design_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Verified design manifest is missing: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _validate_existing_manifest(
                manifest,
                variants=variants,
                gpus=gpus,
                seed=int(args.seed),
                batch_size=batch_size,
                batch_probe=batch_probe,
            )
            manifest["batch_probe"] = deepcopy(batch_probe)
            _write_json(manifest_path, manifest)
            print(json.dumps({"manifest": str(manifest_path), "jobs": manifest["jobs"]}, indent=2))
            return launch_jobs(manifest_path, manifest)
        manifest_path, manifest = write_design_plan(
            output_root,
            variants=variants,
            gpus=gpus,
            seed=int(args.seed),
            batch_size=batch_size,
            batch_probe=batch_probe,
        )
    except (RuntimeError, ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps({"manifest": str(manifest_path), "jobs": manifest["jobs"]}, indent=2))
    return 0 if args.dry_run else launch_jobs(manifest_path, manifest)


def _validate_existing_manifest(
    manifest: dict[str, Any],
    *,
    variants: tuple[str, ...],
    gpus: tuple[int, ...],
    seed: int,
    batch_size: int,
    batch_probe: dict[str, Any] | None,
) -> None:
    if manifest.get("protocol") != "mmw_t2_design_screening_v1":
        raise ValueError("Existing manifest does not belong to the MMW T2 design-screening protocol.")
    if int(manifest.get("seed", -1)) != int(seed) or int(manifest.get("batch_size", -1)) != int(batch_size):
        raise ValueError("Existing manifest seed or batch size does not match the requested launch.")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(variants):
        raise ValueError("Existing manifest does not contain the requested number of jobs.")
    pairs = [(str(job.get("candidate")), int(job.get("gpu", -1))) for job in jobs if isinstance(job, dict)]
    if pairs != list(zip(variants, gpus)):
        raise ValueError("Existing manifest candidate/GPU mapping does not match the requested launch.")
    if any(job.get("status") != "planned" for job in jobs):
        raise ValueError("Existing manifest contains a non-planned job and cannot be relaunched.")
    if batch_probe is None or int(batch_probe.get("selected_common_batch_size", -1)) != int(batch_size):
        raise ValueError("Existing manifest launch requires a matching trusted common batch probe.")
    for job in jobs:
        config_path = ROOT / str(job["config_path"])
        if not config_path.is_file():
            raise FileNotFoundError(f"Existing manifest config is missing: {config_path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
