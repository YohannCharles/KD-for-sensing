import json
from pathlib import Path

import torch

from kd_sensing.data.temporal_missing import (
    apply_training_temporal_missing,
    fixed_single_modality_from_config,
    fixed_single_modality_mask_from_config,
)
from kd_sensing.engine.evaluation_pass import run_evaluation_pass
from kd_sensing.engine.run_metadata import dataset_run_metadata, prediction_setup_metadata
from kd_sensing.engine.runtime import prepare_task_batch


def validate(model, dataloader, cfg: dict, criterion, device: torch.device, output_dir: str | Path | None = None) -> dict:
    fixed_modality = fixed_single_modality_from_config(cfg)

    def fixed_single_modality_batch_transform(raw_batch):
        return apply_training_temporal_missing(prepare_task_batch(raw_batch), cfg, epoch=0, step=0)

    metrics = dict(
        run_evaluation_pass(
            model,
            dataloader,
            cfg,
            criterion,
            device,
            force_modality_mask=fixed_single_modality_mask_from_config(cfg),
            batch_transform=fixed_single_modality_batch_transform if fixed_modality is not None else None,
        ).metrics
    )
    if fixed_modality is not None:
        metrics["fixed_modality"] = fixed_modality
    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None:
        metadata = dataset_run_metadata(dataset)
        split = metadata.get("split") or getattr(dataset, "split", None) or "test"
        split_metadata = {split: metadata}
    else:
        split_metadata = None
    metrics["prediction_setup"] = prediction_setup_metadata(cfg, split_metadata=split_metadata)
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
