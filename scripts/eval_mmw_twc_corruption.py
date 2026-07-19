#!/usr/bin/env python3
"""Run one fixed inference-only MMW sensor corruption shard."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from kd_sensing.evaluation.corruptions import CorruptionSpec, apply_inference_corruption


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "mmw_twc_corruption_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one MMW checkpoint under a deterministic sensor corruption.")
    parser.add_argument("--root", default="outputs/mmw_twc_fair_pattern_v1")
    parser.add_argument("--protocol-manifest", default="outputs/cache/mmw_twc_outer_v1/protocol_manifest.json")
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--corruption", required=True, choices=("gps_noise", "image_occlusion", "image_blur", "radar_noise", "lidar_sparsify"))
    parser.add_argument("--severity", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    strict = _strict_module()
    root = _path(args.root)
    protocol_path = _path(args.protocol_manifest)
    protocol = strict.load_protocol(protocol_path)
    cache = strict._load_immutable_cache(protocol)
    indices = strict._mechanism_trace_indices(list(cache["conditions"]))
    spec = CorruptionSpec(args.corruption, args.severity)
    corruption_seed = 20260718 + args.seed * 101 + args.severity
    gps_scaler = None
    if args.corruption == "gps_noise":
        scaler_path = root / args.method / f"seed{args.seed}" / "artifacts/gps_scaler.npz"
        with np.load(scaler_path) as payload:
            gps_scaler = (np.asarray(payload["mean"]), np.asarray(payload["scale"]))

    def transform(batch, batch_index):
        return apply_inference_corruption(
            batch,
            spec,
            seed=corruption_seed,
            batch_index=batch_index,
            gps_scaler_mean=gps_scaler[0] if gps_scaler else None,
            gps_scaler_scale=gps_scaler[1] if gps_scaler else None,
        )

    output = root / "eval_corruption" / args.method / f"seed{args.seed}" / f"{args.corruption}_s{args.severity}"
    result = strict.evaluate_run(
        root=root,
        method=args.method,
        seed=args.seed,
        protocol_path=protocol_path,
        output_dir=output,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        batch_transform=transform,
        condition_indices=indices,
        evaluation_extension={
            "id": PROTOCOL_ID,
            "corruption": args.corruption,
            "severity": args.severity,
            "seed": corruption_seed,
            "training_recipe_changed": False,
            "checkpoint_changed": False,
            "conditions": "clean_and_canonical_block80",
        },
    )
    print(json.dumps(result, indent=2))
    return 0


def _strict_module():
    path = ROOT / "scripts/eval_mmw_twc_evidence.py"
    spec = importlib.util.spec_from_file_location("_mmw_twc_corruption_strict", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
