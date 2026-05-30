#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.data.mmw.preparation import build_sequence_splits_from_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MMW sequence split CSVs from an existing frame manifest.")
    parser.add_argument("--data-root", default="dataset/MMW/sunny", help="MMW condition root.")
    parser.add_argument("--scene", action="append", required=True, help="Scene name. Repeat for multiple scenes.")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--pred-len", type=int, required=True)
    parser.add_argument("--split-tag", required=True, help="Subdirectory under Prepared/<scene>/splits.")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--split-strategy",
        choices=("group_safe_time_block",),
        default="group_safe_time_block",
        help="Sequence split strategy. Only strict group-safe time blocks are supported.",
    )
    parser.add_argument("--block-size-frames", type=int, help="Frames per group-safe time block.")
    parser.add_argument("--guard-band-frames", type=int, help="Minimum guard band between train/test blocks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = Path(args.data_root)
    reports = []
    for scene in args.scene:
        reports.append(
            build_scene_splits(
                data_root=data_root,
                scene=str(scene),
                seq_len=int(args.seq_len),
                pred_len=int(args.pred_len),
                split_tag=str(args.split_tag),
                split_seed=int(args.split_seed),
                train_ratio=float(args.train_ratio),
                split_strategy=str(args.split_strategy),
                block_size_frames=args.block_size_frames,
                guard_band_frames=args.guard_band_frames,
            )
        )
    print(json.dumps({"scenes": reports}, indent=2, sort_keys=True))
    return 0


def build_scene_splits(
    *,
    data_root: Path,
    scene: str,
    seq_len: int,
    pred_len: int,
    split_tag: str,
    split_seed: int,
    train_ratio: float,
    split_strategy: str,
    block_size_frames: int | None,
    guard_band_frames: int | None,
) -> dict:
    return build_sequence_splits_from_manifest(
        data_root=data_root,
        scene=scene,
        seq_len=seq_len,
        pred_len=pred_len,
        split_tag=split_tag,
        split_seed=split_seed,
        train_ratio=train_ratio,
        split_strategy=split_strategy,
        block_size_frames=block_size_frames,
        guard_band_frames=guard_band_frames,
    )


if __name__ == "__main__":
    raise SystemExit(main())
