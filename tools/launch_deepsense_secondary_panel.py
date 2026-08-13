#!/usr/bin/env python3
"""Generate and launch the fixed DeepSense6G three-method secondary panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from kd_sensing.config import load_config
from kd_sensing.config.io import dump_config


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/cache/deepsense6g_twc_secondary_v1/protocol_manifest.json"
OUTPUT = ROOT / "outputs/deepsense6g_secondary_transfer_v3"
METHODS = ("prototype_only", "amber_full", "rmbp_mm")
SEEDS = (1, 2, 3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    if not gpus:
        raise ValueError("At least one GPU is required.")
    jobs = generate_configs()
    if args.dry_run:
        print(json.dumps({"jobs": jobs, "gpus": gpus}, indent=2))
        return 0
    return launch(jobs, gpus)


def generate_configs() -> list[dict[str, Any]]:
    manifest = _validated_manifest()
    config_dir = OUTPUT / "generated_configs"
    log_dir = OUTPUT / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for method in METHODS:
        template = ROOT / f"tools/configs/deepsense6g/{method}.yaml"
        for seed in SEEDS:
            cfg = load_config(template)
            cfg["experiment"]["seed"] = seed
            cfg["data"]["dataset"]["portion_seed"] = seed
            cfg["temporal_missing"]["seed"] = seed
            cfg["data"]["dataset"].pop("scene", None)
            cfg["data"]["dataset"].pop("data_root", None)
            cfg["data"]["dataset"].pop("train_csv_name", None)
            cfg["data"]["dataset"].pop("test_csv_name", None)
            cfg["data"]["dataset"]["domains"] = [
                {
                    "id": f"scenario{item['scene']}",
                    "scene": int(item["scene"]),
                    "data_root": str(Path(item["data_root"]).resolve()),
                    "train_csv_name": str(Path(item["train"]["path"]).resolve()),
                    "test_csv_name": str(Path(item["test"]["path"]).resolve()),
                }
                for item in manifest["scenes"]
            ]
            cfg["data"]["dataloader"].update(
                train_batch_size=32,
                test_batch_size=64,
                validation_batch_size=64,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                train_drop_last=False,
                test_drop_last=False,
            )
            cfg["training"].update(epochs=40, max_epochs=40, resume=False, checkpoint_selection="last")
            cfg["training"]["final_test"] = {"enabled": True, "reason": "fixed_secondary_one_shot_test"}
            cfg["training"]["final_test_missing_matrix"] = True
            cfg["evaluation"]["k_values"] = [1, 3, 5]
            cfg["output"].update(dir=str(OUTPUT / method), run_name=f"seed{seed}", overwrite=False)
            cfg["deepsense6g_secondary_evidence"] = _evidence_binding(manifest, seed)
            path = config_dir / f"{method}_seed{seed}.yaml"
            if (OUTPUT / method / f"seed{seed}").exists():
                raise ValueError(f"Refusing to overwrite existing panel run: {OUTPUT / method / f'seed{seed}'}")
            dump_config(cfg, path)
            load_config(path)
            jobs.append(
                {
                    "method": method,
                    "seed": seed,
                    "config": str(path),
                    "log": str(log_dir / f"{method}_seed{seed}.log"),
                    "run_dir": str(OUTPUT / method / f"seed{seed}"),
                }
            )
    (OUTPUT / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    return jobs


def launch(jobs: list[dict[str, Any]], gpus: tuple[str, ...]) -> int:
    pending = list(jobs)
    running: dict[str, tuple[subprocess.Popen[Any], dict[str, Any], Any, threading.Thread]] = {}
    failures: list[dict[str, Any]] = []
    while pending or running:
        for gpu in gpus:
            if gpu in running or not pending:
                continue
            job = pending.pop(0)
            handle = open(job["log"], "w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                ["conda", "run", "-n", "kd_mm_beam", "kd-sensing-train", "--config", job["config"]],
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            thread = threading.Thread(target=_copy_output, args=(process, handle), daemon=True)
            thread.start()
            job.update(gpu=gpu, pid=process.pid, started_at=time.time())
            running[gpu] = (process, job, handle, thread)
            print(json.dumps({"event": "started", **job}), flush=True)
        time.sleep(15)
        for gpu, (process, job, handle, thread) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            thread.join(timeout=30)
            handle.close()
            job.update(returncode=code, finished_at=time.time())
            if code:
                failures.append(dict(job))
            print(json.dumps({"event": "finished", **job}), flush=True)
            del running[gpu]
    summary = {"jobs": jobs, "failures": failures}
    (OUTPUT / "launch_result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 1 if failures else 0


def _copy_output(process: subprocess.Popen[Any], handle: Any) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        handle.write(line)
        handle.flush()


def _validated_manifest() -> dict[str, Any]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    declared = payload.get("manifest_sha256")
    body = dict(payload)
    body.pop("manifest_sha256", None)
    actual = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if declared != actual or payload.get("protocol_id") != "deepsense6g_twc_secondary_v1":
        raise ValueError("DeepSense6G filtered protocol manifest identity is invalid.")
    pooled = payload.get("pooled_dataset", {})
    if pooled.get("train_row_count") != 13240 or pooled.get("test_row_count") != 4090:
        raise ValueError("DeepSense6G filtered protocol counts drifted.")
    for scene in payload.get("scenes", []):
        for split in ("train", "test"):
            path = Path(scene[split]["path"])
            if not path.is_file() or _sha256(path) != scene[split]["sha256"]:
                raise ValueError(f"DeepSense6G {split} CSV identity drifted: {path}")
    return payload


def _evidence_binding(manifest: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "protocol_id": manifest["protocol_id"],
        "protocol_manifest": str(MANIFEST.resolve()),
        "protocol_manifest_payload_sha256": manifest["manifest_sha256"],
        "protocol_manifest_file_sha256": _sha256(MANIFEST),
        "filter_rule": "future_beam1_has_exactly_64_finite_nonnegative_values",
        "pooled_dataset": manifest["pooled_dataset"],
        "scene_split_sha256": {
            f"scenario{scene['scene']}": {split: scene[split]["sha256"] for split in ("train", "test")}
            for scene in manifest["scenes"]
        },
        "training_mask_seed": int(seed),
        "checkpoint_selection": "fixed_epoch_last",
        "test_policy": "one_shot_after_fixed_40_epochs",
        "prototype_topology_scope": "linear_label_index_not_physical_ula",
        "claim_ineligible": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
