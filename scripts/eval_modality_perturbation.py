#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.engine.batch import normalize_batch, prepare_fusion_inputs, prepare_labels, forward_model  # noqa: E402
from kd_sensing.engine.data_factory import build_dataloaders  # noqa: E402
from kd_sensing.engine.model_output import adapt_model_output, select_prediction_slots  # noqa: E402
from kd_sensing.engine.optim import build_device, build_model, build_task_criterion  # noqa: E402
from kd_sensing.engine.validator import _metrics_from_outputs  # noqa: E402
from kd_sensing.utils.checkpoint import load_model_state  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Evaluate clean and modality-perturbed fusion metrics.")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--ckpt", "--weights", dest="ckpt", required=True)
    parser.add_argument(
        "--perturbations",
        nargs="*",
        help="Perturbations like shuffle_gps or zero_mmwave. Defaults to shuffle/zero for every enabled modality.",
    )
    parser.add_argument("--override", "-o", action="append", default=[])
    args, unknown = parser.parse_known_args(argv)
    cfg = load_config(args.config, [*args.override, *(item for item in unknown if "=" in item)])
    modalities = [str(name) for name in cfg["model"]["student"].get("modalities", ["image", "radar"])]
    perturbations = args.perturbations or [f"{kind}_{modality}" for modality in modalities for kind in ("shuffle", "zero")]
    device = build_device(cfg)
    dataloader = build_dataloaders(cfg)["test"]
    model = build_model(cfg["model"]["student"]).to(device)
    load_model_state(args.ckpt, model, role="eval_modality_perturbation", map_location=device, strict=bool(cfg.get("checkpoint", {}).get("strict_load", True)))
    criterion = build_task_criterion(cfg)
    result = {"clean": _evaluate(model, dataloader, cfg, criterion, device, perturbation=None), "perturbations": {}}
    for perturbation in perturbations:
        result["perturbations"][perturbation] = _evaluate(model, dataloader, cfg, criterion, device, perturbation=perturbation)
    print(json.dumps(result, indent=2))
    return result


def _evaluate(model, dataloader, cfg: dict, criterion, device: torch.device, *, perturbation: str | None) -> dict:
    model.eval()
    model_cfg = cfg["model"]
    num_pred = model_cfg.get("num_pred", 3)
    seq_length = model_cfg.get("seq_length_student", 8)
    downsample_ratio = model_cfg.get("downsample_ratio", 1)
    num_classes = model_cfg.get("num_classes", 64)
    val_loss = 0.0
    outputs = []
    labels_list = []
    with torch.no_grad():
        for raw_batch in dataloader:
            batch = normalize_batch(raw_batch)
            batch = _perturb_batch(batch, perturbation) if perturbation else batch
            labels = prepare_labels(batch, num_pred=num_pred, downsample_ratio=downsample_ratio, device=device)
            fusion_inputs = prepare_fusion_inputs(
                batch,
                seq_length=seq_length,
                num_pred=num_pred,
                device=device,
                modalities=cfg["model"]["student"].get("modalities"),
            )
            model_output = adapt_model_output(forward_model(model, "fusion", **fusion_inputs))
            logits = select_prediction_slots(model_output.logits, num_pred)
            val_loss += criterion(logits.reshape(-1, num_classes), labels.flatten()).item()
            outputs.append(logits.detach().cpu())
            labels_list.append(labels.detach().cpu())
    return _metrics_from_outputs(
        val_loss / max(len(outputs), 1),
        torch.cat(outputs, dim=0),
        torch.cat(labels_list, dim=0),
        cfg,
    )


def _perturb_batch(batch: dict[str, torch.Tensor], perturbation: str) -> dict[str, torch.Tensor]:
    if "_" not in perturbation:
        raise ValueError(f"Perturbation must be shuffle_<modality> or zero_<modality>, got '{perturbation}'.")
    kind, modality = perturbation.split("_", 1)
    keys_by_modality = {
        "image": ("image",),
        "radar": ("radar_ra", "radar_da"),
        "gps": ("gps",),
        "lidar": ("lidar",),
        "mmwave": ("mmwave",),
    }
    keys = keys_by_modality.get(modality)
    if kind not in {"shuffle", "zero"} or keys is None:
        raise ValueError(f"Unsupported perturbation '{perturbation}'.")
    perturbed = dict(batch)
    first = next((batch[key] for key in keys if key in batch and torch.is_tensor(batch[key])), None)
    if first is None:
        return perturbed
    order = torch.randperm(first.shape[0], device=first.device) if first.shape[0] > 1 else None
    for key in keys:
        value = batch.get(key)
        if not torch.is_tensor(value):
            continue
        if kind == "zero":
            perturbed[key] = torch.zeros_like(value)
        elif order is not None and value.shape[0] == first.shape[0]:
            perturbed[key] = value.index_select(0, order)
    return perturbed


if __name__ == "__main__":
    main()
