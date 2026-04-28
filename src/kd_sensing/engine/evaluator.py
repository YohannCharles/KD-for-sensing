from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kd_sensing.config.io import dump_config
from kd_sensing.engine.builders import (
    build_dataset,
    build_device,
    build_model,
    build_task_criterion,
    prepare_lidar_normalizer,
)
from kd_sensing.engine.validator import validate
from kd_sensing.utils.checkpoint import checkpoint_load_summary, load_model_state
from kd_sensing.utils.paths import output_dir as resolve_output_dir, resolve_path
from kd_sensing.utils.seed import set_seed


def evaluate(cfg: dict, weights: str | None = None, output_dir: str | None = None) -> dict:
    set_seed(cfg.get("experiment", {}).get("seed", 0))
    device = build_device(cfg)
    run_dir = resolve_output_dir(output_dir or cfg.get("output", {}).get("dir", "outputs")) / "evaluation"
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "final_config.yaml")
    dataset_kwargs = {}
    if _evaluation_uses_gps(cfg):
        train_dataset = build_dataset(cfg, "train")
        prepare_lidar_normalizer(cfg, train_dataset)
        dataset_kwargs["gps_scaler"] = getattr(train_dataset, "gps_scaler", None)
        if getattr(train_dataset, "use_lidar", False):
            dataset_kwargs["lidar_normalizer"] = getattr(train_dataset, "lidar_normalizer", None)
    elif _evaluation_uses_lidar(cfg):
        train_dataset = build_dataset(cfg, "train")
        prepare_lidar_normalizer(cfg, train_dataset)
        dataset_kwargs["lidar_normalizer"] = getattr(train_dataset, "lidar_normalizer", None)
    dataset = build_dataset(cfg, "test", **dataset_kwargs)
    loader_cfg = cfg["data"]["dataloader"]
    dataloader = DataLoader(
        dataset,
        batch_size=loader_cfg.get("test_batch_size", 3),
        shuffle=False,
        num_workers=loader_cfg.get("num_workers", 0),
    )
    model = build_model(cfg["model"]["student"]).to(device)
    weight_path = weights or cfg.get("evaluation", {}).get("weights")
    checkpoint_load = None
    if weight_path:
        resolved = resolve_path(weight_path)
        load_result = load_model_state(
            resolved,
            model,
            role="evaluation",
            map_location=device,
            strict=bool(cfg.get("checkpoint", {}).get("strict_load", True)),
        )
        checkpoint_load = checkpoint_load_summary(load_result)
    criterion = build_task_criterion(cfg)
    metrics = validate(model, dataloader, cfg, criterion, device, output_dir=run_dir)
    with (run_dir / "test_report.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return {"run_dir": str(run_dir), "metrics": metrics, "checkpoint_load": checkpoint_load}


def _evaluation_uses_gps(cfg: dict) -> bool:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if dataset_cfg.get("use_gps", False):
        return True
    task = cfg.get("experiment", {}).get("task", "image")
    if task == "gps":
        return True
    if task != "fusion":
        return False
    for role in ("student", "teacher"):
        modalities = cfg.get("model", {}).get(role, {}).get("modalities")
        if modalities and "gps" in modalities:
            return True
    return False


def _evaluation_uses_lidar(cfg: dict) -> bool:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if dataset_cfg.get("use_lidar", False):
        return True
    task = cfg.get("experiment", {}).get("task", "image")
    if task == "lidar":
        return True
    if task != "fusion":
        return False
    for role in ("student", "teacher"):
        modalities = cfg.get("model", {}).get(role, {}).get("modalities")
        if modalities and "lidar" in modalities:
            return True
    return False
