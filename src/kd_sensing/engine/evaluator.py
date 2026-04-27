from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kd_sensing.config.io import dump_config
from kd_sensing.engine.builders import build_dataset, build_device, build_model, build_task_criterion
from kd_sensing.engine.validator import validate
from kd_sensing.utils.paths import output_dir as resolve_output_dir, resolve_path
from kd_sensing.utils.seed import set_seed


def evaluate(cfg: dict, weights: str | None = None, output_dir: str | None = None) -> dict:
    set_seed(cfg.get("experiment", {}).get("seed", 0))
    device = build_device(cfg)
    run_dir = resolve_output_dir(output_dir or cfg.get("output", {}).get("dir", "outputs")) / "evaluation"
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "final_config.yaml")
    dataset = build_dataset(cfg, "test")
    loader_cfg = cfg["data"]["dataloader"]
    dataloader = DataLoader(
        dataset,
        batch_size=loader_cfg.get("test_batch_size", 3),
        shuffle=False,
        num_workers=loader_cfg.get("num_workers", 0),
    )
    model = build_model(cfg["model"]["student"]).to(device)
    weight_path = weights or cfg.get("evaluation", {}).get("weights")
    if weight_path:
        resolved = resolve_path(weight_path)
        state_dict = torch.load(resolved, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    criterion = build_task_criterion(cfg)
    metrics = validate(model, dataloader, cfg, criterion, device, output_dir=run_dir)
    with (run_dir / "test_report.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return {"run_dir": str(run_dir), "metrics": metrics}

