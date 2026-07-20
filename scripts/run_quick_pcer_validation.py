#!/usr/bin/env python3
"""Prepare and launch the four-run MMW PCER quick validation."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import yaml

from kd_sensing.data.temporal_block_mask import (
    PCER_STABLE_PROBABILITIES,
    PCER_TRANSITION_PROBABILITIES,
    PCER_WARMUP_PROBABILITIES,
    TemporalBlockMaskGenerator,
)

import launch_mmw_all_weather_matrix as all_weather


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "mmw_quick_pcer_validation_v1"
DEFAULT_OUTPUT = ROOT / "outputs/quick_pcer_validation"
PROTOCOL_MANIFEST = ROOT / "outputs/cache/mmw_twc_outer_v1/protocol_manifest.json"
SEED = 1
EVAL_SEED = 20260720
EPOCHS = 16
BATCH_SIZE = 32
NUM_WORKERS = 12
PREFETCH_FACTOR = 1
MODALITIES = ("image", "radar", "gps", "lidar")
GPU_MAP = {
    "qv_a0_proto_static": 4,
    "qv_a1_proto_old_router": 5,
    "qv_a2_proto_consistency_static": 6,
    "qv_a3_pcer_full": 7,
}
EXPERIMENTS = (
    ("qv_a0_proto_static", "A0", "disabled", "uniform_mean", 0.0),
    ("qv_a1_proto_old_router", "A1", "disabled", "supervised_router", 0.1),
    ("qv_a2_proto_consistency_static", "A2", "evidence_static", "uniform_mean", 0.0),
    ("qv_a3_pcer_full", "A3", "counterfactual_router", "uniform_mean", 0.0),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=40000)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if not args.prepare and not args.launch:
        parser.error("select --prepare or --launch")
    output_root = _path(args.output_root)
    manifest_path = prepare(output_root) if args.prepare else output_root / "training_manifest.json"
    if args.launch:
        return launch(manifest_path, min_free_mib=int(args.min_free_mib), poll_seconds=float(args.poll_seconds))
    print(json.dumps({"status": "prepared", "manifest": str(manifest_path)}, indent=2))
    return 0


def prepare(output_root: Path) -> Path:
    protocol = _read_json(PROTOCOL_MANIFEST)
    domain_inventory = _quick_domains(protocol)
    audit = _replication_audit(domain_inventory)
    request = {
        "protocol": PROTOCOL_ID,
        "seed": SEED,
        "eval_seed": EVAL_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "modalities": list(MODALITIES),
        "history_window": 5,
        "num_blocks": 20,
        "gpus": GPU_MAP,
        "experiments": [list(item) for item in EXPERIMENTS],
        "source_protocol_manifest": str(PROTOCOL_MANIFEST),
        "source_protocol_sha256": _sha256(PROTOCOL_MANIFEST),
        "selection_split": "frozen_inner_validation",
        "test_split": "historical_h5p1_strict_v2_claim_ineligible",
        "claim_eligible": False,
    }
    request_sha256 = _payload_sha256(request)
    manifest_path = output_root / "training_manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("request_sha256") != request_sha256:
            raise ValueError(f"Existing PCER request differs from the frozen request: {manifest_path}")
        return manifest_path

    output_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    configs = {}
    for name, ablation, pcer_mode, fusion_type, oracle_weight in EXPERIMENTS:
        config = build_experiment_config(
            output_root,
            domain_inventory,
            name=name,
            ablation=ablation,
            pcer_mode=pcer_mode,
            fusion_type=fusion_type,
            oracle_weight=oracle_weight,
        )
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "resolved_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        configs[name] = config
        jobs.append(
            {
                "experiment": name,
                "ablation": ablation,
                "gpu": GPU_MAP[name],
                "config_path": str(config_path),
                "config_sha256": _sha256(config_path),
                "run_dir": str(run_dir),
                "log_path": str(run_dir / "train.log"),
                "status": "planned",
                "return_code": None,
                "claim_eligible": False,
            }
        )
    _write_yaml(
        output_root / "resolved_common_config.yaml",
        {
            **request,
            "training_mask_probabilities": {
                "warmup": PCER_WARMUP_PROBABILITIES,
                "transition": PCER_TRANSITION_PROBABILITIES,
                "stable": PCER_STABLE_PROBABILITIES,
            },
            "loss": {
                "lambda_proto": 0.2,
                "lambda_modality_proto": 0.1,
                "lambda_mask": 0.5,
                "lambda_route": 0.2,
                "distill_temperature": 2.0,
                "contribution_temperature": 0.5,
            },
        },
    )
    _write_json(output_root / "replicated_frame_audit.json", audit)
    _write_mask_examples(output_root)
    _write_implementation_notes(output_root, domain_inventory, audit)
    _write_shell_entry(output_root)
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "protocol": PROTOCOL_ID,
            "request": request,
            "request_sha256": request_sha256,
            "jobs": jobs,
            "status": "planned",
            "created_at": _now(),
        },
    )
    return manifest_path


def build_experiment_config(
    output_root: Path,
    domains: list[dict[str, str]],
    *,
    name: str,
    ablation: str,
    pcer_mode: str,
    fusion_type: str,
    oracle_weight: float,
) -> dict[str, Any]:
    config = all_weather.build_config(
        "T2",
        output_root,
        seed=SEED,
        smoke=True,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        umask_training_profile="umask_h4_v1",
        umask_router_architecture_profile="umask_router_nopattern_v1",
    )
    config["experiment"].update({"name": name, "ablation_id": ablation, "seed": SEED, "device": "auto"})
    dataset = config["data"]["dataset"]
    dataset.update(
        {
            "domains": deepcopy(domains),
            "portion": 1.0,
            "portion_strategy": "even",
            "portion_seed": 42,
            "include_router_utility_targets": False,
            "include_router_corruption_metadata": False,
        }
    )
    config["data"]["dataloader"].update(
        {
            "train_batch_size": BATCH_SIZE,
            "validation_batch_size": BATCH_SIZE,
            "test_batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "persistent_workers": True,
            "prefetch_factor": PREFETCH_FACTOR,
        }
    )
    primary = config["model"]["primary"]
    primary.update({"fusion_type": fusion_type, "router_variant": "current", "seq_length": 5})
    primary.pop("router_variant_config", None)
    if pcer_mode == "disabled":
        primary.pop("pcer", None)
    else:
        primary["pcer"] = {"mode": pcer_mode, "hidden_dim": 64, "embedding_dim": 8, "dropout": 0.0}

    loss = config["loss"]["u_mask_beam_jepa"]
    loss.update(
        {
            "enabled": True,
            "use_beam_prototype_alignment": True,
            "lambda_proto": 0.2,
            "lambda_modality_proto": 0.1,
            "router_oracle_weight": float(oracle_weight),
            "router_oracle_target_mode": "hard_first",
            "superset_consistency": {
                "enabled": False,
                "confidence_gated_kl": False,
                "kl_weight": 0.0,
                "temperature": 2.0,
            },
        }
    )
    for key in ("dynamic_router", "router_quality_pairing", "pcer"):
        loss.pop(key, None)
    if pcer_mode != "disabled":
        loss["pcer"] = {
            "lambda_mask": 0.5,
            "lambda_route": 0.2 if pcer_mode == "counterfactual_router" else 0.0,
            "distill_temperature": 2.0,
            "contribution_temperature": 0.5,
            "contribution_clip": 5.0,
        }
    config["temporal_missing"] = {
        "enabled": True,
        "history_window": 5,
        "prediction_window": 1,
        "mode": "pcer_curriculum",
        "seed": SEED,
        "preserve_unmasked_for_superset": True,
    }
    config["training"].update(
        {
            "epochs": EPOCHS,
            "max_epochs": EPOCHS,
            "resume": False,
            "checkpoint_selection": "best_validation_loss",
            "amp": {"enabled": True, "dtype": "bfloat16", "grad_scaler": False},
            "validation": {"interval_epochs": 1},
            "final_test": {"enabled": False, "reason": "fixed_mask_evaluator_after_training"},
            "allow_tf32": True,
            "cudnn_benchmark": True,
        }
    )
    config["evaluation"].update(
        {
            "k_values": [1, 3, 5],
            "dba_delta": 3,
            "beam_distance_circular": True,
            "dba_distance_mode": "circular",
        }
    )
    config["evaluation"]["missing_patterns"] = {"enabled": False, "patterns": [], "prediction_index": "last"}
    config["output"] = {
        "dir": str(output_root),
        "run_name": name,
        "group_by_scene": False,
        "overwrite": True,
        "progress": {"enabled": False},
        "tensorboard": {"enabled": False},
    }
    config["mmw_all_weather_protocol"].update(
        {
            "split_tag": PROTOCOL_ID,
            "screening_role": "quick_validation_claim_ineligible",
            "checkpoint_policy": "best_validation_loss",
            "seed": SEED,
        }
    )
    config["mmw_quick_pcer_protocol"] = {
        "protocol": PROTOCOL_ID,
        "ablation": ablation,
        "experiment": name,
        "pcer_mode": pcer_mode,
        "seed": SEED,
        "eval_seed": EVAL_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "selection_split": "frozen_inner_validation",
        "test_split": "historical_h5p1_strict_v2_claim_ineligible",
        "claim_eligible": False,
    }
    return config


def launch(manifest_path: Path, *, min_free_mib: int, poll_seconds: float) -> int:
    manifest = _read_json(manifest_path)
    free = _gpu_free_memory()
    blocked = {
        int(job["gpu"]): free.get(int(job["gpu"]), 0)
        for job in manifest["jobs"]
        if not _completed_run(job) and free.get(int(job["gpu"]), 0) < int(min_free_mib)
    }
    if blocked:
        raise RuntimeError(f"PCER GPUs do not meet the free-memory threshold: {blocked}")
    running: list[tuple[subprocess.Popen[Any], Any, dict[str, Any]]] = []
    for job in manifest["jobs"]:
        if _completed_run(job):
            job["status"] = "done"
            continue
        log_path = Path(job["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        command = [
            "conda",
            "run",
            "-n",
            "kd_mm_beam",
            "--no-capture-output",
            "kd-sensing-train",
            "--config",
            str(job["config_path"]),
        ]
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "OMP_NUM_THREADS": "4",
                "PYTHONUNBUFFERED": "1",
            }
        )
        handle.write(f"[{_now()}] GPU{job['gpu']}: {' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        job.update({"status": "running", "pid": process.pid, "start_time": _now()})
        running.append((process, handle, job))
    manifest.update(status="running" if running else "complete", launched_at=_now())
    _write_json(manifest_path, manifest)
    _write_json(
        manifest_path.parent / "pids.json",
        {job["experiment"]: {"gpu": job["gpu"], "pid": job.get("pid"), "status": job["status"]} for job in manifest["jobs"]},
    )
    while running:
        for process, handle, job in list(running):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            job.update(
                {
                    "status": "done" if code == 0 and _completed_run(job) else "failed",
                    "return_code": int(code),
                    "end_time": _now(),
                }
            )
            running.remove((process, handle, job))
            _write_json(manifest_path, manifest)
        if running:
            time.sleep(float(poll_seconds))
    manifest.update(
        status="complete" if all(job["status"] == "done" for job in manifest["jobs"]) else "failed",
        completed_at=_now(),
    )
    _write_json(manifest_path, manifest)
    return 0 if manifest["status"] == "complete" else 1


def _quick_domains(protocol: Mapping[str, Any]) -> list[dict[str, str]]:
    records = protocol.get("domains")
    if not isinstance(records, list) or len(records) != 15:
        raise ValueError("PCER requires the 15-domain mmw_twc_outer_v1 manifest.")
    historical = {item["id"]: item for item in all_weather.domains()}
    result = []
    for record in records:
        identity = str(record.get("id", ""))
        split = record.get("split", {})
        source = historical.get(identity)
        if source is None or not isinstance(split, Mapping):
            raise ValueError(f"Invalid PCER domain record {identity!r}.")
        inner_train = Path(str(split.get("inner_train", {}).get("csv", "")))
        inner_validation = Path(str(split.get("inner_validation", {}).get("csv", "")))
        test_path = ROOT / source["data_root"] / source["test_csv_name"]
        for path in (inner_train, inner_validation, test_path):
            if not path.is_file():
                raise FileNotFoundError(f"PCER split input is missing: {path}")
        result.append(
            {
                "id": identity,
                "condition": str(record["condition"]),
                "scene": str(record["scene"]),
                "data_root": str(record["data_root"]),
                "train_csv_name": str(inner_train),
                "val_csv_name": str(inner_validation),
                "test_csv_name": str(test_path.resolve()),
            }
        )
    return result


def _replication_audit(domains: list[dict[str, str]]) -> dict[str, Any]:
    prefixes = ("camera", "radar", "gps", "bs_gps", "lidar")
    duplicate_rows = []
    row_count = 0
    for domain in domains:
        for split_name in ("train_csv_name", "val_csv_name", "test_csv_name"):
            path = Path(domain[split_name])
            with path.open(newline="", encoding="utf-8") as handle:
                for row_index, row in enumerate(csv.DictReader(handle)):
                    row_count += 1
                    for prefix in prefixes:
                        values = [str(row.get(f"{prefix}{index}", "")).strip() for index in range(1, 6)]
                        duplicates = sorted({value for value in values if value and values.count(value) > 1})
                        if duplicates and len(duplicate_rows) < 50:
                            duplicate_rows.append(
                                {
                                    "domain": domain["id"],
                                    "split": split_name,
                                    "row": row_index,
                                    "modality": prefix,
                                    "duplicates": duplicates,
                                }
                            )
    return {
        "schema_version": 1,
        "audited_rows": row_count,
        "audited_domains": len(domains),
        "duplicate_within_window_count": len(duplicate_rows),
        "duplicate_examples": duplicate_rows,
        "grouped_masking_runtime_available": True,
        "source_frame_ids_in_current_batch": False,
        "status": "no_within_window_replicas_found" if not duplicate_rows else "replicas_found_without_batch_group_ids",
    }


def _write_mask_examples(output_root: Path) -> None:
    generator = TemporalBlockMaskGenerator(EVAL_SEED)
    examples = {}
    for kind in (
        "full",
        "sparse_easy",
        "single_modality_burst2",
        "single_modality_missing",
        "latest_sync_missing",
        "two_modality_recent_async",
    ):
        result = generator(
            batch_size=1,
            num_modalities=4,
            num_timesteps=5,
            sample_ids=("mmw:example",),
            mask_type=kind,
            severity=None,
            seed=EVAL_SEED,
            training=False,
            variant_ids=0,
        )
        examples[kind] = {
            "availability_mask_m_t": result["availability_mask"][0].tolist(),
            "metadata": result["mask_metadata"][0],
        }
    _write_json(output_root / "mask_examples.json", examples)


def _write_implementation_notes(output_root: Path, domains: list[dict[str, str]], audit: dict[str, Any]) -> None:
    notes = f"""# PCER 快速验证实现记录

## 现有实现定位

- 主训练入口：`kd-sensing-train --config <resolved_config.yaml>`；训练 owner 为 `src/kd_sensing/engine/trainer.py`。
- 主配置：`configs/mmw/t2.yaml` 与 `configs/mmw/_base.yaml`；MMW 15-domain 入口和 domain inventory 来自 `scripts/launch_mmw_all_weather_matrix.py`。
- 评测 owner：`src/kd_sensing/engine/runtime.py`、`src/kd_sensing/evaluation/metrics.py`；本轮固定 mask helper 为 `scripts/eval_quick_pcer_validation.py`。
- 模型：`src/kd_sensing/models/u_mask_beam_jepa.py`；融合前 `latent_sequence` 为 `[B,T,M,D]=[B,5,4,64]`，展平 block features 为 `[B,20,64]`。
- Prototype：`BeamPrototypeBank.prototypes` 为 `[K,D]=[64,64]`；BPA/topology loss 位于 `src/kd_sensing/losses/beam_prototype_alignment.py` 与 `u_mask_beam_jepa_prototype.py`。
- 旧 Router：`UMaskBeamJEPA.supervised_router` 在 masked temporal pooling 后消费 confidence、entropy、prototype margin、reliability 与 modality identity，输出 `[B,4]`。
- 现有静态融合：`uniform_mean` 对可用模态重新归一化；A2 的 PCER static 则在 20 个可用 block evidence 上等权归一化。
- MMW 模态顺序：image、radar、gps、lidar；M=4、T=5、N=20。组件实现不硬编码 M/T。

## 四组组件

- A0 `qv_a0_proto_static`：BPA + 统一 PCER mask curriculum + 现有 `uniform_mean`；关闭旧 Router、PCER consistency 和 route loss。
- A1 `qv_a1_proto_old_router`：BPA + 同一 mask curriculum + current confidence/prototype-center Router/oracle loss；关闭 PCER。
- A2 `qv_a2_proto_consistency_static`：BPA + `[B,20,64]` block evidence + 等权 evidence fusion + full-to-masked KL；关闭新旧 Router loss。
- A3 `qv_a3_pcer_full`：BPA + block evidence + full-to-masked KL + counterfactual block Router KL；旧 Router 不参与融合或 loss。

## 数据与运行协议

- 数据集：MMW，3 种天气 × 5 个场景 = {len(domains)} domains。
- 训练/验证：`mmw_twc_outer_v1` 的 frozen group-safe inner train/validation；seed={SEED}。
- 测试：历史 `h5p1_strict_v2` test，仅用于 claim-ineligible quick validation；不读取冻结 outer evidence，不用测试集选择 checkpoint。
- 预算：{EPOCHS} epoch（正式 40 的 40%）、batch size {BATCH_SIZE}、{NUM_WORKERS} workers、prefetch {PREFETCH_FACTOR}、逐 epoch validation、最低 validation loss 的 `best.pth`。
- optimizer/scheduler：current H4 AdamW + cosine warm restart；四组相同。AMP 使用 bf16，不启用 fp16 GradScaler。

## 时间对齐与复制帧

- MMW prepared CSV 显式提供 camera/radar/gps/bs_gps/lidar 的 1..5 路径；loader 未执行 forward fill 或 backward fill。`prepare_fusion_inputs` 仅在输入序列短于配置长度时左侧重复首帧，但本轮 CSV 均提供 5 帧，不触发该 fallback。
- 路径审计覆盖 {audit['audited_rows']} 个 domain/split rows，窗口内重复路径计数为 {audit['duplicate_within_window_count']}，状态：`{audit['status']}`。
- `TemporalBlockMaskGenerator` 支持传入 `source_frame_ids[B,M,T]` 后同组 mask；当前 MMW batch 不提供该字段，因此若未来数据引入复制对齐，必须先扩展正式 dataset contract。当前结果标记为 replicated-frame grouped masking 未从 batch identity 激活。

## 修改文件

- `src/kd_sensing/data/temporal_block_mask.py`、`temporal_missing.py`、`temporal_missing_contract.py`
- `src/kd_sensing/models/pcer_temporal_fusion.py`、`u_mask_beam_jepa.py`
- `src/kd_sensing/losses/pcer_temporal_fusion.py`、`u_mask_beam_jepa.py`、`u_mask_beam_jepa_config.py`
- `src/kd_sensing/engine/checkpointing.py`、`trainer_runtime_helpers.py`
- `scripts/run_quick_pcer_validation.py`、`scripts/eval_quick_pcer_validation.py`、`scripts/run_quick_pcer_gpu4_7.sh`
- `tests/test_pcer_*.py` 与 OpenSpec change `add-pcer-temporal-block-fusion`
"""
    (output_root / "implementation_notes.md").write_text(notes, encoding="utf-8")


def _write_shell_entry(output_root: Path) -> None:
    content = """#!/usr/bin/env bash
set -u
ROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/../..\" && pwd)\"
cd \"$ROOT\"
conda run -n kd_mm_beam python scripts/run_quick_pcer_validation.py --prepare
conda run -n kd_mm_beam python scripts/run_quick_pcer_validation.py --launch
train_status=$?
if [ \"$train_status\" -ne 0 ]; then
  exit \"$train_status\"
fi
conda run -n kd_mm_beam python scripts/eval_quick_pcer_validation.py --all
"""
    path = output_root / "run_quick_pcer_gpu4_7.sh"
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _completed_run(job: Mapping[str, Any]) -> bool:
    run_dir = Path(str(job["run_dir"]))
    status = run_dir / "run_status.json"
    best = run_dir / "checkpoints/best.pth"
    return status.is_file() and best.is_file() and _read_json(status).get("state") == "complete"


def _gpu_free_memory() -> dict[int, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True
    )
    return {
        int(index.strip()): int(memory.strip())
        for index, memory in (line.split(",", 1) for line in output.splitlines())
    }


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


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


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
