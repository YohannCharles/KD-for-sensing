#!/usr/bin/env python3
"""Prepare, preflight, and launch the eight MMW PCER direction candidates."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import torch
import torch.nn as nn
import yaml

from kd_sensing.config import load_config
from kd_sensing.data.temporal_missing import apply_training_temporal_missing
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.optim import build_model
from kd_sensing.engine.runtime import autocast_context, prepare_task_batch, prepare_task_labels, run_model_step
from kd_sensing.engine.trainer_runtime_helpers import shutdown_all_dataloaders
from kd_sensing.engine.training_extensions import BatchState, ExtensionContext
from kd_sensing.losses.u_mask_beam_jepa import UMaskBeamJEPATrainingExtension
from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config

import run_quick_pcer_validation as quick


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/pcer_direction_search"
PROTOCOL_ID = "mmw_pcer_direction_search_v1"
EPOCHS = 16
BATCH_SIZE = 32
NUM_WORKERS = 4
EXPERIMENTS = (
    ("qv_b0_old_router_consistency", "B0", 0, "evidence_only", "none", False),
    ("qv_b1_proto_router_beam_only", "B1", 1, "block_router", "none", False),
    ("qv_b2_standalone_quality_router", "B2", 2, "block_router", "standalone_quality", False),
    ("qv_b3_onpolicy_block_router", "B3", 3, "block_router", "onpolicy_block", False),
    ("qv_b4_onpolicy_modality_group", "B4", 4, "hierarchical_router", "onpolicy_modality", False),
    ("qv_b5_hierarchical_router", "B5", 5, "hierarchical_router", "none", False),
    ("qv_b6_mask_prior_dynamic_residual", "B6", 6, "mask_residual_router", "none", False),
    ("qv_b7_modality_balanced_evidence", "B7", 7, "evidence_only", "none", True),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--preflight-all", action="store_true")
    action.add_argument("--preflight-worker", choices=[item[0] for item in EXPERIMENTS])
    action.add_argument("--launch", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=30000)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    output = Path(args.output_root).expanduser().resolve()
    if args.prepare:
        prepare(output)
        return 0
    if args.preflight_worker:
        return preflight_worker(output, args.preflight_worker)
    if args.preflight_all:
        return preflight_all(output)
    return launch(output, min_free_mib=args.min_free_mib, poll_seconds=args.poll_seconds)


def prepare(output: Path) -> Path:
    protocol = quick._read_json(quick.PROTOCOL_MANIFEST)
    domains = quick._quick_domains(protocol)
    request = {
        "protocol": PROTOCOL_ID,
        "source_protocol": quick.PROTOCOL_ID,
        "source_protocol_manifest": str(quick.PROTOCOL_MANIFEST),
        "source_protocol_sha256": quick._sha256(quick.PROTOCOL_MANIFEST),
        "seed": quick.SEED,
        "eval_seed": quick.EVAL_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "effective_batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "prefetch_factor": 1,
        "initialization": "same_seed_scratch_no_common_checkpoint_available",
        "selection_split": "frozen_inner_validation",
        "test_split": "historical_h5p1_strict_v2_claim_ineligible",
        "experiments": [list(item) for item in EXPERIMENTS],
        "claim_eligible": False,
    }
    request_sha = _payload_sha(request)
    manifest_path = output / "training_manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("request_sha256") != request_sha:
            raise ValueError(f"Existing direction-search request differs: {manifest_path}")
        return manifest_path
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, label, gpu, mode, route_target, evidence_learning in EXPERIMENTS:
        config = build_config(
            output, domains, name=name, label=label, mode=mode,
            route_target=route_target, evidence_learning=evidence_learning,
        )
        run_dir = output / name
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "resolved_config.yaml"
        _write_yaml(config_path, config)
        jobs.append(
            {
                "experiment": name,
                "candidate": label,
                "gpu": gpu,
                "config_path": str(config_path),
                "config_sha256": _sha256(config_path),
                "run_dir": str(run_dir),
                "train_log": str(run_dir / "train.log"),
                "preflight_log": str(run_dir / "preflight.log"),
                "status": "planned",
                "preflight_status": "planned",
                "claim_eligible": False,
            }
        )
    _write_yaml(output / "common_resolved_config.yaml", request)
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "request": request,
            "request_sha256": request_sha,
            "jobs": jobs,
            "status": "planned",
            "created_at": _now(),
        },
    )
    (output / "implementation_notes.md").write_text(_implementation_notes(request), encoding="utf-8")
    _write_shell(output)
    print(json.dumps({"status": "prepared", "manifest": str(manifest_path)}, indent=2))
    return manifest_path


def build_config(
    output: Path,
    domains: list[dict[str, str]],
    *,
    name: str,
    label: str,
    mode: str,
    route_target: str,
    evidence_learning: bool,
) -> dict[str, Any]:
    fusion = "supervised_router" if mode == "evidence_only" else "uniform_mean"
    config = quick.build_experiment_config(
        output,
        deepcopy(domains),
        name=name,
        ablation=label,
        pcer_mode=mode,
        fusion_type=fusion,
        oracle_weight=0.0,
    )
    config["data"]["dataloader"].update(
        {"num_workers": NUM_WORKERS, "persistent_workers": True, "prefetch_factor": 1}
    )
    loss = config["loss"]["u_mask_beam_jepa"]
    loss["pcer"] = {
        "lambda_mask": 0.5,
        "lambda_route": 0.2 if route_target != "none" else 0.0,
        "route_target": route_target,
        "distill_temperature": 2.0,
        "contribution_temperature": 0.5,
        "quality_temperature": 0.5,
        "modality_contribution_temperature": 0.5,
        "contribution_clip": 5.0,
        "evidence_learning": {
            "enabled": evidence_learning,
            "lambda_lomo": 0.5 if evidence_learning else 0.0,
            "lambda_unimodal": 0.1 if evidence_learning else 0.0,
            "distill_temperature": 2.0,
        },
    }
    config["training"]["timing"] = {
        "enabled": True,
        "profile": "host",
        "log_interval": 10,
        "slow_batch_seconds": 20.0,
    }
    config["mmw_pcer_direction_search"] = {
        "protocol": PROTOCOL_ID,
        "candidate": label,
        "experiment": name,
        "router_mode": mode,
        "route_target": route_target,
        "evidence_learning": evidence_learning,
        "seed": quick.SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "claim_eligible": False,
    }
    u_mask_beam_jepa_config(config)
    return config


def preflight_all(output: Path) -> int:
    manifest_path = output / "training_manifest.json"
    manifest = _read_json(manifest_path)
    free = _gpu_free_memory()
    running = []
    for job in manifest["jobs"]:
        gpu = int(job["gpu"])
        if free.get(gpu, 0) < 30000:
            job["preflight_status"] = "blocked_gpu"
            continue
        handle = Path(job["preflight_log"]).open("w", encoding="utf-8")
        command = [
            "conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "python",
            str(Path(__file__).resolve()), "--output-root", str(output),
            "--preflight-worker", job["experiment"],
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "OMP_NUM_THREADS": "4"}
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        job.update(preflight_status="running", preflight_pid=process.pid)
        running.append((process, handle, job))
    _write_json(manifest_path, manifest)
    while running:
        for process, handle, job in list(running):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            result_path = Path(job["run_dir"]) / "preflight.json"
            job["preflight_status"] = "passed" if code == 0 and result_path.is_file() else "failed"
            job["preflight_return_code"] = int(code)
            running.remove((process, handle, job))
        _write_json(manifest_path, manifest)
        if running:
            time.sleep(2)
    for job in manifest["jobs"]:
        job["config_sha256"] = _sha256(Path(job["config_path"]))
    manifest["preflight_status"] = "passed" if all(job["preflight_status"] == "passed" for job in manifest["jobs"]) else "failed"
    _write_json(manifest_path, manifest)
    return 0 if manifest["preflight_status"] == "passed" else 1


def preflight_worker(output: Path, experiment: str) -> int:
    manifest = _read_json(output / "training_manifest.json")
    job = next(item for item in manifest["jobs"] if item["experiment"] == experiment)
    config_path = Path(job["config_path"])
    cfg = load_config(config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]["primary"]).to(device).train()
    dataloaders = build_dataloaders(cfg)
    extension = UMaskBeamJEPATrainingExtension()
    context = ExtensionContext(
        cfg=cfg,
        task="fusion",
        model_cfg=cfg["model"]["primary"],
        training_cfg=cfg["training"],
        primary_model=model,
        task_criterion=nn.CrossEntropyLoss(),
        run_dir=Path(job["run_dir"]),
        device=device,
        num_pred=1,
        num_classes=64,
        seq_length=5,
        non_blocking=False,
    )
    state = extension.setup(context)
    try:
        raw = next(iter(dataloaders["train"]))
        batch = apply_training_temporal_missing(prepare_task_batch(raw), cfg, epoch=0, step=0)
        labels = prepare_task_labels(batch, num_pred=1, device=device)
        with autocast_context(True, device, torch.bfloat16):
            controls = extension.before_forward(context, state, batch, labels, epoch=0, step=0)
            step = run_model_step(
                model, "fusion", batch, seq_length=5, num_pred=1, device=device,
                extra_model_kwargs=controls.model_kwargs,
            )
            batch_state = BatchState(
                epoch=0, step=0, batch=batch, labels=labels,
                primary_output=step.model_output, primary_logits=step.logits, controls=controls,
            )
            result = extension.compute_base_loss(context, state, batch_state)
            if result is None:
                raise RuntimeError("Direction preflight requires U-Mask base loss.")
        if not torch.isfinite(result.total_loss):
            raise ValueError("Preflight loss is not finite.")
        result.total_loss.backward()
        extension.after_backward(context, state, batch_state)
        diagnostics = dict(result.diagnostics)
        model_output = step.model_output.diagnostics
        block_weight = model_output["pcer_block_router_weights"]
        block_available = model_output["pcer_block_availability"]
        if not torch.equal(block_weight.masked_select(~block_available), torch.zeros_like(block_weight.masked_select(~block_available))):
            raise ValueError("Preflight missing block weight is nonzero.")
        if not torch.allclose(block_weight.sum(dim=1), torch.ones_like(block_weight[:, 0]), atol=1e-5):
            raise ValueError("Preflight available block weights do not sum to one.")
        router_parameters = [parameter for name, parameter in model.named_parameters() if "router" in name and parameter.requires_grad]
        router_gradient = _gradient_norm(router_parameters)
        if router_gradient <= 0:
            raise ValueError("Preflight router gradient is zero.")
        beam_value = diagnostics.get("loss/beam", diagnostics.get("loss_beam"))
        if beam_value is None:
            raise KeyError("Preflight diagnostics contain neither loss/beam nor loss_beam.")
        beam = float(beam_value)
        new_weighted = float(diagnostics.get("loss/pcer_route_weighted", 0.0))
        new_weighted += float(diagnostics.get("loss/pcer_lomo_weighted", 0.0))
        new_weighted += float(diagnostics.get("loss/pcer_unimodal_aux_weighted", 0.0))
        adjustment = _maybe_adjust_new_loss(cfg, beam, new_weighted)
        if adjustment["adjusted"]:
            _write_yaml(config_path, cfg)
        payload = {
            "experiment": experiment,
            "candidate": job["candidate"],
            "device": str(device),
            "sample_count": int(labels.shape[0]),
            "loss_total": float(result.total_loss.detach().cpu()),
            "loss_beam": beam,
            "loss_prototype": float(diagnostics.get("loss/prototype_total", 0.0)),
            "loss_consistency": float(diagnostics.get("loss/pcer_mask_consistency_weighted", 0.0)),
            "loss_new_weighted": new_weighted,
            "router_gradient_norm": router_gradient,
            "backbone_gradient_norm": _gradient_norm(list(model.encoders.parameters())),
            "prototype_gradient_norm": _gradient_norm(list(model.prototype_bank.parameters())),
            "missing_weight_max": float(block_weight.masked_select(~block_available).max()) if (~block_available).any() else 0.0,
            "available_weight_sum_error": float((block_weight.sum(dim=1) - 1).abs().max()),
            "diagnostics": diagnostics,
            "auto_adjustment": adjustment,
            "passed": True,
        }
        _write_json(Path(job["run_dir"]) / "preflight.json", payload)
        print(json.dumps(payload, indent=2))
        return 0
    finally:
        shutdown_all_dataloaders(dataloaders)


def _maybe_adjust_new_loss(cfg: dict[str, Any], beam: float, weighted: float) -> dict[str, Any]:
    pcer = cfg["loss"]["u_mask_beam_jepa"]["pcer"]
    if weighted <= 0 or 0.01 * beam <= weighted <= beam:
        return {"adjusted": False, "reason": "within_range_or_no_candidate_loss"}
    factor = 0.1 * beam / weighted
    original = {
        "lambda_route": pcer["lambda_route"],
        "lambda_lomo": pcer["evidence_learning"]["lambda_lomo"],
        "lambda_unimodal": pcer["evidence_learning"]["lambda_unimodal"],
    }
    if pcer["lambda_route"] > 0:
        pcer["lambda_route"] *= factor
    if pcer["evidence_learning"]["enabled"]:
        pcer["evidence_learning"]["lambda_lomo"] *= factor
        pcer["evidence_learning"]["lambda_unimodal"] *= factor
    return {
        "adjusted": True,
        "reason": "weighted_new_loss_outside_1_to_100_percent_of_beam",
        "factor": factor,
        "original": original,
        "adjusted_values": {
            "lambda_route": pcer["lambda_route"],
            "lambda_lomo": pcer["evidence_learning"]["lambda_lomo"],
            "lambda_unimodal": pcer["evidence_learning"]["lambda_unimodal"],
        },
    }


def launch(output: Path, *, min_free_mib: int, poll_seconds: float) -> int:
    manifest_path = output / "training_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("preflight_status") != "passed":
        raise RuntimeError("All direction preflights must pass before launch.")
    free = _gpu_free_memory()
    blocked = {int(job["gpu"]): free.get(int(job["gpu"]), 0) for job in manifest["jobs"] if not _completed(job) and free.get(int(job["gpu"]), 0) < int(min_free_mib)}
    if blocked:
        raise RuntimeError(f"Direction-search GPUs below free-memory threshold: {blocked}")
    baseline = _gpu_memory_used()
    running = []
    for job in manifest["jobs"]:
        if _completed(job):
            job["status"] = "done"
            continue
        handle = Path(job["train_log"]).open("a", encoding="utf-8")
        command = ["conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "kd-sensing-train", "--config", job["config_path"]]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(job["gpu"]), "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        job.update(status="running", pid=process.pid, start_time=_now(), baseline_gpu_memory_mib=baseline.get(int(job["gpu"]), 0), peak_gpu_memory_mib=0)
        running.append((process, handle, job))
        time.sleep(2)
    manifest.update(status="running" if running else "complete", launched_at=_now())
    _write_json(manifest_path, manifest)
    _write_pids(output, manifest)
    while running:
        used = _gpu_memory_used()
        for process, handle, job in list(running):
            gpu = int(job["gpu"])
            delta = max(0, used.get(gpu, 0) - int(job.get("baseline_gpu_memory_mib", 0)))
            job["peak_gpu_memory_mib"] = max(int(job.get("peak_gpu_memory_mib", 0)), delta)
            code = process.poll()
            if code is None:
                continue
            handle.close()
            job.update(status="done" if code == 0 and _completed(job) else "failed", return_code=int(code), end_time=_now())
            running.remove((process, handle, job))
        _write_json(manifest_path, manifest)
        _write_pids(output, manifest)
        if running:
            time.sleep(float(poll_seconds))
    manifest.update(status="complete" if all(job["status"] == "done" for job in manifest["jobs"]) else "failed", completed_at=_now())
    _write_json(manifest_path, manifest)
    return 0 if manifest["status"] == "complete" else 1


def _completed(job: dict[str, Any]) -> bool:
    run = Path(job["run_dir"])
    return (run / "checkpoints/best.pth").is_file() and (run / "run_status.json").is_file() and _read_json(run / "run_status.json").get("state") == "complete"


def _implementation_notes(request: dict[str, Any]) -> str:
    return f"""# PCER 八方向快速筛选实现记录

- 数据/预算：MMW 15-domain inner train/validation、historical development test、seed={request['seed']}、{request['epochs']} epoch、batch={request['batch_size']}。
- 初始化：历史 A0-A3 均为 seed1 scratch，无共同基础 checkpoint；B0-B7 同样从 seed1 scratch，且共同 encoder/prototype 在候选 router 前实例化。
- 共同机制：64-beam BPA/topology、block prototype evidence、pcer curriculum、full-to-masked consistency、validation-best checkpoint。
- 运行差异只在 `mmw_pcer_direction_search` 与 `model/loss pcer` 字段；canonical configs 未修改。
- 为降低八任务共享盘争用，B0-B7 统一使用 {NUM_WORKERS} workers/prefetch1；训练语义和 effective batch 不变。
- Preflight 只使用首个训练 batch；新增 loss 仅在加权量级不处于 beam loss 的 1%-100% 时按 10% 目标调整一次。
- 结果为单 seed、inner/development、claim-ineligible，不进入正式 claim。
"""


def _write_shell(output: Path) -> None:
    source = ROOT / "scripts/run_pcer_direction_search_gpu0_7.sh"
    if not source.is_file():
        raise FileNotFoundError(source)
    content = source.read_text(encoding="utf-8")
    target = output / "run_pcer_direction_search_gpu0_7.sh"
    target.write_text(content, encoding="utf-8")
    target.chmod(0o755)


def _write_pids(output: Path, manifest: dict[str, Any]) -> None:
    _write_json(output / "pids.json", {job["experiment"]: {key: job.get(key) for key in ("gpu", "pid", "status")} for job in manifest["jobs"]})


def _gradient_norm(parameters) -> float:
    values = [parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(values).sum().sqrt().cpu()) if values else 0.0


def _gpu_free_memory() -> dict[int, int]:
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    return {int(index): int(memory) for index, memory in (line.split(",", 1) for line in output.splitlines())}


def _gpu_memory_used() -> dict[int, int]:
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    return {int(index): int(memory) for index, memory in (line.split(",", 1) for line in output.splitlines())}


def _payload_sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
