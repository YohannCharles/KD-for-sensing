#!/usr/bin/env python3
"""Run the eight-candidate seed1 Router utility screen on GPU0--7."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

import launch_mmw_tie_aware_router_screen as tie
from kd_sensing.data.mmw.twc_evidence import load_protocol
from kd_sensing.modalities import MODALITY_ORDER


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "mmw_router_expected_utility_screen_v3"
DEFAULT_PROTOCOL = ROOT / "outputs/cache/mmw_twc_outer_v1/protocol_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/mmw_router_expected_utility_screen_v3"
DEFAULT_PREFLIGHT_TRACES = ROOT / "outputs/mmw_router_utility_screen_v1/oracle_gap/CurrentControl"
SEED = 1
BATCH_SIZE = 64
EPOCHS = 40
CANDIDATES: dict[str, dict[str, Any]] = {
    "CurrentControl": {"main_target": "soft_confidence_tie", "main_temperature": 1.0, "paired": 0.0, "monotonic": 0.0, "epsilon": 0.001},
    "ExpectedMainT01": {"main_target": "beam_power_expected_soft", "main_temperature": 0.1, "paired": 0.0, "monotonic": 0.0, "epsilon": 0.001},
    "PairUngated": {"main_target": "soft_confidence_tie", "main_temperature": 1.0, "paired": 0.1, "monotonic": 0.0, "epsilon": 0.001, "start_epoch_index": 10, "max_target_entropy": None},
    "PairEntropy120": {"main_target": "soft_confidence_tie", "main_temperature": 1.0, "paired": 0.1, "monotonic": 0.0, "epsilon": 0.001, "start_epoch_index": 0, "max_target_entropy": 1.2},
    "PairEntropy130": {"main_target": "soft_confidence_tie", "main_temperature": 1.0, "paired": 0.1, "monotonic": 0.0, "epsilon": 0.001, "start_epoch_index": 0, "max_target_entropy": 1.3},
    "MonoW002": {"main_target": "soft_confidence_tie", "main_temperature": 1.0, "paired": 0.1, "monotonic": 0.02, "epsilon": 0.001, "start_epoch_index": 0, "max_target_entropy": 1.3},
    "MonoW005": {"main_target": "soft_confidence_tie", "main_temperature": 1.0, "paired": 0.1, "monotonic": 0.05, "epsilon": 0.001, "start_epoch_index": 0, "max_target_entropy": 1.3},
    "MonoW010": {"main_target": "soft_confidence_tie", "main_temperature": 1.0, "paired": 0.1, "monotonic": 0.1, "epsilon": 0.001, "start_epoch_index": 0, "max_target_entropy": 1.3},
}
BEAM_TEMPERATURE = 0.5
PAIR_TARGET_TEMPERATURE = 0.1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--protocol-manifest", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--preflight-traces", default=str(DEFAULT_PREFLIGHT_TRACES))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--min-free-mib", type=int, default=40000)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    gpus = tuple(int(value.strip()) for value in args.gpus.split(",") if value.strip())
    if gpus != tuple(range(8)):
        parser.error("This frozen screen requires exactly --gpus 0,1,2,3,4,5,6,7.")
    output_root = Path(args.output_root).resolve()
    preflight = run_preflight(Path(args.preflight_traces).resolve(), output_root)
    manifest = prepare_plan(
        output_root,
        Path(args.protocol_manifest).resolve(),
        gpus,
        preflight_path=preflight,
    )
    if not args.launch:
        print(json.dumps({"status": "planned", "manifest": str(manifest), "jobs": 8}, indent=2))
        return 0
    code = tie.run_plan(manifest, min_free_mib=int(args.min_free_mib), poll_seconds=float(args.poll_seconds))
    if code != 0:
        return code
    code = run_oracle_gap(manifest, poll_seconds=float(args.poll_seconds))
    if code == 0:
        subprocess.run(
            ["conda", "run", "-n", "kd_mm_beam", "python", "scripts/summarize_mmw_router_utility_screen.py", "--root", str(manifest.parent)],
            cwd=ROOT,
            check=True,
        )
    return code


def prepare_plan(
    output_root: Path,
    protocol_path: Path,
    gpus: tuple[int, ...],
    *,
    preflight_path: Path,
) -> Path:
    protocol = load_protocol(protocol_path)
    domains, split_sha256 = tie.inner_domains(protocol)
    launcher = tie._script_module("launch_mmw_all_weather_matrix")
    strict = tie._script_module("launch_mmw_twc_evidence")
    evaluator = tie._script_module("eval_mmw_all_weather_matrix")
    topology = strict._load_topology(strict._resolve_topology_path(None))
    mask_cache = output_root / "cache/inner_fixed_masks"
    evaluator._load_or_create_temporal_cache(
        mask_cache,
        modality_frame_masks=16,
        rates=evaluator.RATES,
        mask_types=evaluator.MASK_TYPES,
    )
    request = {
        "protocol_id": PROTOCOL_ID,
        "source_protocol_manifest_sha256": str(protocol["manifest_sha256"]),
        "split_sha256": split_sha256,
        "candidates": CANDIDATES,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "gpus": list(gpus),
        "train_corruption_seed": 20260719,
        "evaluation_corruption_seed": 20260718,
        "selection_split": "frozen_inner_validation_only",
        "claim_eligible": False,
        "later_seeds_planned": False,
        "preflight_path": str(preflight_path),
        "preflight_sha256": _sha256(preflight_path),
    }
    request_sha256 = _payload_sha256(request)
    path = output_root / "training_manifest_router_utility_seed1.json"
    if path.is_file():
        existing = _read_json(path)
        if existing.get("request_sha256") != request_sha256:
            raise ValueError(f"Existing Router utility plan differs from the frozen request: {path}")
        return path
    config_dir = output_root / "generated_configs"
    log_dir = output_root / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for (candidate, settings), gpu in zip(CANDIDATES.items(), gpus):
        config = tie.build_candidate_config(
            launcher,
            strict,
            topology,
            "SoftConfidenceTie",
            tie.CANDIDATES["SoftConfidenceTie"],
            output_root,
            domains=domains,
            source_protocol_sha256=str(protocol["manifest_sha256"]),
            split_sha256=split_sha256,
        )
        _apply_candidate(config, candidate, settings, output_root, split_sha256, str(protocol["manifest_sha256"]))
        config_path = config_dir / f"{candidate}_seed1.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run_dir = output_root / candidate / "seed1"
        jobs.append(
            {
                "variant": candidate,
                "method": candidate,
                "scope": "MMW-15-domain-inner-validation",
                "seed": SEED,
                "gpu": gpu,
                "config_path": str(config_path.resolve()),
                "config_sha256": _sha256(config_path),
                "run_dir": str(run_dir.resolve()),
                "log_path": str((log_dir / f"{candidate}_seed1.log").resolve()),
                "status": "planned",
                "evaluation_status": "planned",
                "evaluation_log_path": str((log_dir / f"{candidate}_seed1.eval.log").resolve()),
                "evaluation_output_path": str((output_root / "eval_inner" / candidate / "metrics.csv").resolve()),
                "oracle_gap_status": "planned",
                "oracle_gap_log_path": str((log_dir / f"{candidate}_seed1.oracle_gap.log").resolve()),
                "oracle_gap_output": str((output_root / "oracle_gap" / candidate).resolve()),
            }
        )
    _write_json(
        path,
        {
            "schema_version": 1,
            "protocol": PROTOCOL_ID,
            "request": request,
            "request_sha256": request_sha256,
            "protocol_manifest": str(protocol_path),
            "mask_cache": str(mask_cache.resolve()),
            "jobs": jobs,
        },
    )
    return path


def _apply_candidate(
    config: dict[str, Any],
    candidate: str,
    settings: Mapping[str, Any],
    output_root: Path,
    split_sha256: str,
    source_protocol_sha256: str,
) -> None:
    loss = config["loss"]["u_mask_beam_jepa"]
    pairing_enabled = float(settings["paired"]) > 0.0 or float(settings["monotonic"]) > 0.0
    main_target = str(settings["main_target"])
    config["data"]["dataset"]["include_router_utility_targets"] = pairing_enabled or main_target.startswith("beam_power_")
    loss["router_oracle_target_mode"] = main_target
    loss["router_oracle_temperature"] = float(settings["main_temperature"])
    loss["router_oracle_beam_temperature"] = BEAM_TEMPERATURE
    loss["router_quality_pairing"] = {
        "enabled": pairing_enabled,
        "utility_mode": "expected",
        "target_temperature": PAIR_TARGET_TEMPERATURE,
        "beam_temperature": BEAM_TEMPERATURE,
        "utility_weight": float(settings["paired"]),
        "monotonic_weight": float(settings["monotonic"]),
        "monotonic_margin_scale": 0.25,
        "quality_drop_epsilon": float(settings["epsilon"]),
        "start_epoch_index": int(settings.get("start_epoch_index", 0)),
        "max_target_entropy": settings.get("max_target_entropy"),
        "corruption_seed": 20260719,
    }
    config["experiment"].update({"name": candidate, "ablation_id": candidate})
    config["output"].update(
        {"dir": str((output_root / candidate).resolve()), "run_name": "seed1", "overwrite": False}
    )
    config.pop("mmw_tie_aware_router_screen", None)
    recipe = {
        "candidate": candidate,
        "settings": dict(settings),
        "source_protocol_manifest_sha256": source_protocol_sha256,
        "inner_split_sha256": split_sha256,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    }
    config["mmw_router_utility_screen"] = {
        "protocol": PROTOCOL_ID,
        "candidate_id": candidate,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "selection_split": "frozen_inner_validation_only",
        "source_protocol_manifest_sha256": source_protocol_sha256,
        "inner_split_sha256": split_sha256,
        "candidate_recipe_sha256": _payload_sha256(recipe),
        "claim_eligible": False,
    }
    config["mmw_all_weather_protocol"]["split_tag"] = PROTOCOL_ID


def run_oracle_gap(path: Path, *, poll_seconds: float) -> int:
    manifest = _read_json(path)
    running = []
    for job in manifest["jobs"]:
        summary = Path(job["oracle_gap_output"]) / "summary.json"
        if summary.is_file():
            job["oracle_gap_status"] = "done"
            continue
        checkpoint = Path(job["run_dir"]) / "checkpoints/last.pth"
        command = [
            "conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "python",
            "scripts/run_mmw_router_oracle_gap_candidate.py",
            "--candidate", job["method"],
            "--config", job["config_path"],
            "--checkpoint", str(checkpoint),
            "--output-root", job["oracle_gap_output"],
            "--batch-size", str(BATCH_SIZE),
        ]
        handle = Path(job["oracle_gap_log_path"]).open("a", encoding="utf-8")
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(job["gpu"]), "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONUNBUFFERED": "1"})
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        job.update({"oracle_gap_status": "running", "oracle_gap_pid": process.pid, "oracle_gap_start_time": _now()})
        running.append((process, handle, job))
    _write_json(path, manifest)
    while running:
        for process, handle, job in list(running):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            job.update(
                {
                    "oracle_gap_status": "done" if code == 0 else "failed",
                    "oracle_gap_return_code": int(code),
                    "oracle_gap_end_time": _now(),
                }
            )
            running.remove((process, handle, job))
            _write_json(path, manifest)
        time.sleep(poll_seconds)
    return 0 if all(job["oracle_gap_status"] == "done" for job in manifest["jobs"]) else 1


def run_preflight(trace_root: Path, output_root: Path) -> Path:
    clean = _load_trace_condition(trace_root / "clean")
    if not clean:
        raise ValueError(f"Expected CurrentControl clean traces under {trace_root}.")
    cells = []
    for modality_index, modality in enumerate(MODALITY_ORDER):
        for severity in (1, 2, 3):
            condition = _corruption_condition(modality, severity)
            corrupted = _load_trace_condition(trace_root / condition)
            if set(corrupted) != set(clean):
                raise ValueError(f"Preflight sample identities differ for {condition}.")
            drops = []
            same_argmax_active = 0
            for sample_id, (clean_logits, powers) in clean.items():
                corrupt_logits, corrupt_powers = corrupted[sample_id]
                if not np.allclose(powers, corrupt_powers, atol=0.0, rtol=0.0):
                    raise ValueError(f"Preflight beam powers differ for {condition}/{sample_id}.")
                clean_utility = _expected_utility(clean_logits[modality_index], powers)
                corrupt_utility = _expected_utility(corrupt_logits[modality_index], powers)
                drop = clean_utility - corrupt_utility
                drops.append(drop)
                if drop > 0.001 and clean_logits[modality_index].argmax() == corrupt_logits[modality_index].argmax():
                    same_argmax_active += 1
            values = np.asarray(drops, dtype=np.float64)
            active = values > 0.001
            cells.append(
                {
                    "modality": modality,
                    "severity": severity,
                    "condition": condition,
                    "sample_count": int(values.size),
                    "utility_drop_mean": float(values.mean()),
                    "utility_drop_q10": float(np.quantile(values, 0.1)),
                    "utility_drop_q50": float(np.quantile(values, 0.5)),
                    "utility_drop_q90": float(np.quantile(values, 0.9)),
                    "positive_ratio": float(np.mean(values > 0.0)),
                    "active_ratio": float(active.mean()),
                    "same_argmax_active_ratio": float(same_argmax_active / values.size),
                }
            )
    failed_modalities = [
        modality
        for modality in MODALITY_ORDER
        if not any(row["active_ratio"] > 0.0 for row in cells if row["modality"] == modality)
    ]
    if failed_modalities:
        raise ValueError(f"Expected-utility preflight has no active signal for {failed_modalities}.")
    gradient_norm = _paired_gradient_smoke()
    if not np.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise ValueError("Expected-utility paired Router gradient smoke failed.")
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "status": "passed",
        "claim_eligible": False,
        "source_trace_root": str(trace_root),
        "source_trace_identity_sha256": _trace_identity_sha256(trace_root),
        "modalities": list(MODALITY_ORDER),
        "beam_temperature": BEAM_TEMPERATURE,
        "quality_drop_epsilon": 0.001,
        "paired_gradient_l1": gradient_norm,
        "mature_target_entropy_gate_ratios": _mature_entropy_gate_ratios(trace_root),
        "cells": cells,
    }
    path = output_root / "preflight_expected_utility.json"
    _write_json(path, payload)
    return path


def _load_trace_condition(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for trace in sorted((path / "traces").glob("*.npz")):
        with np.load(trace) as values:
            for sample_id, logits, powers in zip(
                values["sample_id"], values["unimodal_logits"], values["beam_powers"], strict=True
            ):
                key = str(sample_id)
                if key in result:
                    raise ValueError(f"Duplicate preflight sample identity {key!r}.")
                result[key] = (logits.astype(np.float64), powers.astype(np.float64))
    return result


def _expected_utility(logits: np.ndarray, powers: np.ndarray) -> float:
    shifted = logits / BEAM_TEMPERATURE
    probabilities = np.exp(shifted - shifted.max())
    probabilities /= probabilities.sum()
    normalized = powers / max(float(powers.max()), np.finfo(np.float64).tiny)
    return float(probabilities @ normalized)


def _corruption_condition(modality: str, severity: int) -> str:
    suffix = {
        "image": "occlusion",
        "radar": "noise",
        "gps": "noise",
        "lidar": "sparsify",
    }[modality]
    return f"{modality}_{suffix}_s{severity}"


def _paired_gradient_smoke() -> float:
    import torch
    from kd_sensing.losses.u_mask_beam_jepa import _paired_router_quality_loss

    corrupt_logits = torch.tensor([[1.0, -1.0]], requires_grad=True)
    pair = {
        "clean_unimodal_logits": torch.tensor([[[4.0, 3.0, 0.0], [0.0, 4.0, 3.0]]]),
        "corrupted_unimodal_logits": torch.tensor([[[4.0, 0.0, 3.0], [0.0, 4.0, 3.0]]]),
        "available": torch.ones(1, 2, dtype=torch.bool),
        "clean_router_logits": torch.zeros(1, 2),
        "corrupted_router_logits": corrupt_logits,
        "clean_router_weights": torch.tensor([[0.8, 0.2]]),
        "corrupted_router_weights": torch.softmax(corrupt_logits, dim=1),
        "affected_modality_index": 0,
        "corruption_name": "image_occlusion",
        "corruption_severity": 3,
    }
    loss, diagnostics = _paired_router_quality_loss(
        pair,
        torch.tensor([[0]]),
        torch.tensor([[4.0, 3.0, 0.0]]),
        temperature=PAIR_TARGET_TEMPERATURE,
        utility_mode="expected",
        beam_temperature=BEAM_TEMPERATURE,
        max_target_entropy=1.3,
        utility_weight=0.1,
        monotonic_weight=0.05,
        margin_scale=0.25,
        quality_drop_epsilon=0.001,
    )
    loss.backward()
    if diagnostics["router_pair_active_ratio"] <= 0.0 or corrupt_logits.grad is None:
        return 0.0
    return float(corrupt_logits.grad.abs().sum().item())


def _mature_entropy_gate_ratios(root: Path) -> dict[str, dict[str, float]]:
    result = {}
    for condition_path in sorted(path for path in root.iterdir() if path.is_dir()):
        entropies = []
        for logits, powers in _load_trace_condition(condition_path).values():
            utilities = np.asarray([_expected_utility(item, powers) for item in logits])
            shifted = utilities / PAIR_TARGET_TEMPERATURE
            targets = np.exp(shifted - shifted.max())
            targets /= targets.sum()
            entropies.append(float(-(targets * np.log(np.maximum(targets, np.finfo(np.float64).tiny))).sum()))
        values = np.asarray(entropies)
        result[condition_path.name] = {
            "sample_count": int(values.size),
            "entropy_mean": float(values.mean()),
            "active_ratio_le_1_2": float(np.mean(values <= 1.2)),
            "active_ratio_le_1_3": float(np.mean(values <= 1.3)),
        }
    return result


def _trace_identity_sha256(root: Path) -> str:
    identity = []
    for path in sorted(root.glob("*/provenance.json")):
        payload = _read_json(path)
        identity.append(
            {
                "condition": payload.get("condition"),
                "trace_files": payload.get("trace_files"),
                "corruption_seed": payload.get("corruption_seed"),
            }
        )
    return _payload_sha256(identity)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
