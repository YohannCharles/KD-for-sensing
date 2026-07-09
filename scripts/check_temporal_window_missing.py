#!/usr/bin/env python
import argparse
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from kd_sensing.config.defaults import DEFAULT_CONFIG
from kd_sensing.config.io import deep_merge
from kd_sensing.config.normalization import normalize_temporal_window_missing_config
from kd_sensing.data.difficulty.pipeline import apply_configured_difficulty
from kd_sensing.data.difficulty.schema import DifficultyContext, normalize_config_difficulty
from kd_sensing.data.datasets.synthetic import SyntheticSequenceDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect temporal window and temporal missing batch masks.")
    parser.add_argument("--history_window", "--history-window", type=int, default=5)
    parser.add_argument("--prediction_window", "--prediction-window", type=int, default=1)
    parser.add_argument(
        "--temporal_missing_mode",
        "--temporal-missing-mode",
        choices=("none", "frame_bernoulli", "modality_frame_bernoulli", "block"),
        default="modality_frame_bernoulli",
    )
    parser.add_argument("--temporal_missing_prob", "--temporal-missing-prob", type=float, default=0.2)
    parser.add_argument("--temporal_missing_block_len", "--temporal-missing-block-len", type=int, default=1)
    parser.add_argument("--num_samples", "--num-samples", type=int, default=16)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = deep_merge(
        DEFAULT_CONFIG,
        {
            "experiment": {"task": "fusion", "seed": int(args.seed)},
            "model": {"modalities": ["image", "radar", "lidar", "gps"], "primary": {"modalities": ["image", "radar", "lidar", "gps"]}},
            "temporal_missing": {
                "enabled": args.temporal_missing_mode != "none" or float(args.temporal_missing_prob) > 0.0,
                "history_window": int(args.history_window),
                "prediction_window": int(args.prediction_window),
                "mode": args.temporal_missing_mode,
                "prob": float(args.temporal_missing_prob),
                "block_len": int(args.temporal_missing_block_len),
                "apply": "train",
                "seed": int(args.seed),
            },
        },
    )
    normalize_temporal_window_missing_config(cfg)
    normalize_config_difficulty(cfg)
    dataset = SyntheticSequenceDataset(
        length=int(args.num_samples),
        seq_len=int(args.history_window),
        num_pred=int(args.prediction_window),
        use_gps=True,
        use_lidar=True,
        seed=int(args.seed),
    )
    raw_batch = next(iter(DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)))
    result = apply_configured_difficulty(
        raw_batch,
        cfg,
        DifficultyContext(stage="train", split="train", seed=int(args.seed), step=0),
    )
    batch = result.batch
    modalities = ["image", "radar", "lidar", "gps"]
    print(f"history_window={args.history_window}")
    print(f"prediction_window={args.prediction_window}")
    for key in ("image", "radar_ra", "radar_da", "lidar", "gps"):
        if torch.is_tensor(batch.get(key)):
            print(f"{key}.shape={tuple(batch[key].shape)}")
    temporal_mask = batch.get("temporal_mask")
    modality_mask = batch.get("modality_temporal_mask")
    available = batch.get("available_modalities")
    print(f"temporal_mask.shape={tuple(temporal_mask.shape) if torch.is_tensor(temporal_mask) else None}")
    print(f"temporal_mask.sample={temporal_mask[:2].int().tolist() if torch.is_tensor(temporal_mask) else []}")
    print(f"modality_temporal_mask.shape={tuple(modality_mask.shape) if torch.is_tensor(modality_mask) else None}")
    print(f"modality_temporal_mask.sample={modality_mask[:1].int().tolist() if torch.is_tensor(modality_mask) else []}")
    if torch.is_tensor(available):
        modality_rates = OrderedDict(
            (name, float(available[:, idx].float().mean().item())) for idx, name in enumerate(modalities)
        )
        print(f"available_modalities.shape={tuple(available.shape)}")
        print(f"modality_available_rate={dict(modality_rates)}")
    if torch.is_tensor(temporal_mask):
        time_rates = temporal_mask.float().mean(dim=0).tolist()
        print(f"temporal_available_rate_by_step={[round(float(value), 4) for value in time_rates]}")
        print(f"has_all_missing_sample={bool((~modality_mask.any(dim=(1, 2))).any().item()) if torch.is_tensor(modality_mask) else False}")
    print(f"history_indices.example={batch.get('history_indices')[:2].tolist() if torch.is_tensor(batch.get('history_indices')) else []}")
    print(f"target_index.example={batch.get('target_index')[:4].tolist() if torch.is_tensor(batch.get('target_index')) else []}")
    print(f"temporal_missing_metadata={batch.get('temporal_missing_metadata', {})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
