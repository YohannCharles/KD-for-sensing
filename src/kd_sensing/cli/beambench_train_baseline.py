from __future__ import annotations

import argparse
import json
from pathlib import Path

from kd_sensing.baselines.beambench.mock import create_mock_dataset
from kd_sensing.baselines.beambench.pipeline import MockTrainingConfig, train_mock_baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate a BeamBench baseline wrapper.")
    parser.add_argument("--mock", action="store_true", help="Create and run the explicit MOCK smoke dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("outputs/beambench_baseline/mock_dataset"))
    parser.add_argument("--csv", type=Path, default=Path("ml_challenge_mock_multi_modal.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/beambench_baseline/mock_smoke"))
    parser.add_argument("--num-beams", type=int, default=64)
    parser.add_argument("--beam-shift", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)
    if args.mock:
        create_mock_dataset(args.data_root, rows=12, num_beams=args.num_beams)
    config = MockTrainingConfig(
        data_root=str(args.data_root),
        csv=str(args.csv),
        output_dir=str(args.output_dir),
        num_beams=args.num_beams,
        beam_shift=args.beam_shift,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    report = train_mock_baseline(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
