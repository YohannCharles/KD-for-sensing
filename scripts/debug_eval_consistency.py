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
from kd_sensing.evaluation.metrics import calculate_topk_accuracy  # noqa: E402
from kd_sensing.utils.checkpoint import load_model_state  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Compare official validation with the all-modality subset path.")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--ckpt", "--weights", dest="ckpt", required=True)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--override", "-o", action="append", default=[])
    args, unknown = parser.parse_known_args(argv)
    cfg = load_config(args.config, [*args.override, *(item for item in unknown if "=" in item)])
    device = build_device(cfg)
    dataloader = build_dataloaders(cfg)["test"]
    model = build_model(cfg["model"]["primary"]).to(device)
    load_model_state(args.ckpt, model, role="debug_eval_consistency", map_location=device, strict=bool(cfg.get("checkpoint", {}).get("strict_load", True)))
    model.eval()
    official = _collect_outputs(model, dataloader, cfg, device, mask=None)
    all_mask = torch.ones(len(cfg["model"]["primary"].get("modalities", ["image", "radar"])), dtype=torch.bool, device=device)
    subset_all = _collect_outputs(model, dataloader, cfg, device, mask=all_mask)
    official_top1 = _top1(official["logits"], official["labels"], cfg)
    subset_top1 = _top1(subset_all["logits"], subset_all["labels"], cfg)
    diff = abs(official_top1 - subset_top1)
    result = {
        "official_top1": official_top1,
        "subset_all_top1": subset_top1,
        "top1_diff": diff,
        "threshold": float(args.threshold),
        "samples": int(official["labels"].shape[0]),
        "batches": int(official["batches"]),
        "official_logits_shape": list(official["logits"].shape),
        "subset_logits_shape": list(subset_all["logits"].shape),
        "labels_shape": list(official["labels"].shape),
        "first_batch_predictions_equal": bool(official["first_predictions"] == subset_all["first_predictions"]),
    }
    print(json.dumps(result, indent=2))
    if diff > float(args.threshold):
        raise SystemExit(1)
    return result


def _collect_outputs(model, dataloader, cfg: dict, device: torch.device, *, mask: torch.Tensor | None) -> dict:
    model_cfg = cfg["model"]
    num_pred = model_cfg.get("num_pred", 3)
    seq_length = model_cfg.get("seq_length", 8)
    downsample_ratio = model_cfg.get("downsample_ratio", 1)
    logits = []
    labels_list = []
    first_predictions = None
    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(dataloader):
            batch = normalize_batch(raw_batch)
            labels = prepare_labels(batch, num_pred=num_pred, downsample_ratio=downsample_ratio, device=device)
            fusion_inputs = prepare_fusion_inputs(
                batch,
                seq_length=seq_length,
                num_pred=num_pred,
                device=device,
                modalities=cfg["model"]["primary"].get("modalities"),
            )
            output = adapt_model_output(forward_model(model, "fusion", **fusion_inputs, force_modality_mask=mask))
            selected = select_prediction_slots(output.logits, num_pred)
            logits.append(selected.detach().cpu())
            labels_list.append(labels.detach().cpu())
            if batch_idx == 0:
                first_predictions = selected.argmax(dim=-1).detach().cpu().tolist()
    return {
        "logits": torch.cat(logits, dim=0),
        "labels": torch.cat(labels_list, dim=0),
        "batches": len(logits),
        "first_predictions": first_predictions,
    }


def _top1(logits: torch.Tensor, labels: torch.Tensor, cfg: dict) -> float:
    topk, _ = calculate_topk_accuracy(logits, labels, cfg.get("evaluation", {}).get("k_values", [1, 2, 3, 5, 10]))
    values = topk.get(1)
    if values is None:
        return 0.0
    totals = labels.ne(-100).sum(dim=0).numpy()
    valid = totals > 0
    if not valid.any():
        return 0.0
    return float(values[int(valid.argmax())])


if __name__ == "__main__":
    main()
