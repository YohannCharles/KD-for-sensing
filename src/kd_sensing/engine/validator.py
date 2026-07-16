import json
from pathlib import Path

import torch

from kd_sensing.engine.evaluation_pass import run_evaluation_pass
from kd_sensing.engine.run_metadata import dataset_run_metadata, prediction_setup_metadata


def validate(model, dataloader, cfg: dict, criterion, device: torch.device, output_dir: str | Path | None = None) -> dict:
    metrics = dict(run_evaluation_pass(model, dataloader, cfg, criterion, device).metrics)
    dataset = getattr(dataloader, "dataset", None)
    split_metadata = {getattr(dataset, "split", "test"): dataset_run_metadata(dataset)} if dataset is not None else None
    metrics["prediction_setup"] = prediction_setup_metadata(cfg, split_metadata=split_metadata)
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
