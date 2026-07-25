#!/usr/bin/env python3
"""Independently analyze the single-seed mask-bias novelty triage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from kd_sensing.baselines.prototype_decision_adapter import MASKS, write_json
from kd_sensing.utils.checkpoint import load_torch_payload
from analyze_full_pool_adba_surrogate import _recompute, _summary


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mask_keys(raw_masks: list[list[int]]) -> list[str]:
    by_mask = {tuple(mask): key for key, _, mask in MASKS}
    return [by_mask[tuple(mask)] for mask in raw_masks]


def _macro(result: dict[str, Any], keys: list[str]) -> dict[str, float]:
    return {
        metric: float(np.mean([result["masks"][key][metric] for key in keys]))
        for metric in ("top1", "top3", "top5", "within3", "mae", "adba")
    }


def _mlp_biases(checkpoint: Path, masks: list[tuple[int, ...]]) -> np.ndarray:
    state = load_torch_payload(checkpoint, map_location="cpu")["state_dict"]
    value = torch.tensor(masks, dtype=torch.float32)
    value = F.gelu(F.linear(value, state["mask_projector.0.weight"], state["mask_projector.0.bias"]))
    value = F.gelu(F.linear(value, state["condition_head.0.weight"], state["condition_head.0.bias"]))
    value = F.linear(value, state["condition_head.2.weight"], state["condition_head.2.bias"])
    return value.numpy()


def _weight_probe(root: Path, fold: dict[str, Any]) -> dict[str, Any]:
    all_masks = [tuple(mask) for _, _, mask in MASKS if tuple(mask) != (1, 1, 1, 1)]
    held_out = [tuple(mask) for mask in fold["held_out_masks"]]
    allowed = [mask for mask in all_masks if mask not in held_out]
    biases = _mlp_biases(root / "adba_surrogate/b1/checkpoints/last.pth", all_masks)
    by_mask = {mask: biases[index] for index, mask in enumerate(all_masks)}
    design = lambda masks: np.asarray([[1.0, *(1 - bit for bit in mask)] for mask in masks], dtype=np.float64)
    coefficients = np.linalg.lstsq(design(allowed), np.stack([by_mask[mask] for mask in allowed]), rcond=None)[0]
    predicted = design(held_out) @ coefficients
    actual = np.stack([by_mask[mask] for mask in held_out])
    rows = []
    for mask, left, right in zip(held_out, actual, predicted):
        denominator = max(float(np.linalg.norm(left)), 1e-12)
        rows.append(
            {
                "mask": list(mask),
                "rmse": float(np.sqrt(np.mean((left - right) ** 2))),
                "relative_l2_error": float(np.linalg.norm(left - right) / denominator),
                "cosine_similarity": float(np.dot(left, right) / max(denominator * np.linalg.norm(right), 1e-12)),
            }
        )
    return {
        "label_accessed": False,
        "fit_masks": [list(mask) for mask in allowed],
        "held_out_masks": [list(mask) for mask in held_out],
        "per_mask": rows,
        "mean_relative_l2_error": float(np.mean([row["relative_l2_error"] for row in rows])),
        "mean_cosine_similarity": float(np.mean([row["cosine_similarity"] for row in rows])),
    }


def analyze(root: Path) -> dict[str, Any]:
    run_root = root / "mask_bias_ablation"
    fold = _read(run_root / "protocol/unseen_fold0.json")
    reference_ids: dict[str, np.ndarray] = {}
    directories = {
        "a0": root / "stage2/a0",
        "mask_mlp": root / "adba_surrogate/b1",
        "global_bias": run_root / "all_seen/global_bias",
        "mask_lookup": run_root / "all_seen/mask_lookup",
    }
    recomputed = {}
    maximum = 0.0
    for key, directory in directories.items():
        recomputed[key], difference = _recompute(directory, reference_ids)
        maximum = max(maximum, difference)
    summaries = {key: _summary(value) for key, value in recomputed.items()}
    gate = summaries["mask_mlp"]["adba"] > summaries["mask_lookup"]["adba"]
    decision = {
        "metric": "all14_macro.adba",
        "mask_mlp": summaries["mask_mlp"]["adba"],
        "mask_lookup": summaries["mask_lookup"]["adba"],
        "difference": summaries["mask_mlp"]["adba"] - summaries["mask_lookup"]["adba"],
        "unseen_pilot_authorized": gate,
    }
    write_json(run_root / "stage_decision.json", decision)
    probe = _weight_probe(root, fold)
    write_json(run_root / "weight_space_probe.json", probe)

    unseen = {}
    held_out = _mask_keys(fold["held_out_masks"])
    unseen_directories = {
        "unseen_mask_mlp": run_root / "unseen_fold0/mask_mlp",
        "unseen_factorized": run_root / "unseen_fold0/factorized_bias",
    }
    if all((directory / "metrics.json").is_file() for directory in unseen_directories.values()):
        for key, directory in unseen_directories.items():
            result, difference = _recompute(directory, reference_ids)
            maximum = max(maximum, difference)
            unseen[key] = {"all14": _summary(result), "held_out": _macro(result, held_out), "result": result}
        unseen["a0"] = {"held_out": _macro(recomputed["a0"], held_out)}
        unseen["all_seen_mask_mlp"] = {"held_out": _macro(recomputed["mask_mlp"], held_out)}

    if maximum > 1e-7:
        raise ValueError(f"Independent metric difference exceeds 1e-7: {maximum}")
    payload = {
        "schema_version": 1,
        "outer_test_accessed": False,
        "independent_metric_max_abs_difference": maximum,
        "fold": fold,
        "all_seen": summaries,
        "stage_decision": decision,
        "weight_space_probe": probe,
        "unseen": unseen,
    }
    write_json(run_root / "independent_recompute.json", payload)
    with (run_root / "all_seen_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("method", "top1", "top3", "top5", "within3", "mae", "adba")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in ("a0", "global_bias", "mask_lookup", "mask_mlp"):
            writer.writerow({"method": key, **{field: summaries[key][field] for field in fields[1:]}})
    print(json.dumps({"status": "completed", "stage_decision": decision, "unseen_completed": bool(unseen)}, indent=2))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    analyze(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
