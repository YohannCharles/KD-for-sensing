from __future__ import annotations

import argparse
import json
from pathlib import Path

from kd_sensing.baselines.beambench.mock import create_mock_dataset
from kd_sensing.baselines.beambench.official import plan_official_evaluation
from kd_sensing.baselines.beambench.pipeline import evaluate_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a BeamBench official or MOCK baseline wrapper.")
    parser.add_argument("--mock", action="store_true", help="Evaluate a mock checkpoint instead of official challenge.py.")
    parser.add_argument("--official-root", type=Path, default=Path("/tmp/beambench-official"))
    parser.add_argument("--data-root", type=Path, default=Path("dataset/DeepSense6G/raw_data/test"))
    parser.add_argument("--csv", type=Path, default=Path("ml_challenge_test_multi_modal.csv"))
    parser.add_argument("--type-list", type=str, default="radar_dense_camera_ae_gps")
    parser.add_argument("--adapt", type=str, default="adapt_")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/beambench_baseline/eval"))
    parser.add_argument("--execute", action="store_true", help="Actually run official challenge.py when all inputs exist.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-beams", type=int, default=64)
    args = parser.parse_args(argv)
    if args.mock:
        if args.checkpoint is None:
            raise SystemExit("--mock evaluation requires --checkpoint.")
        if not (args.data_root / args.csv).exists():
            create_mock_dataset(args.data_root, rows=12, num_beams=args.num_beams)
        report = evaluate_checkpoint(
            args.checkpoint,
            data_root=args.data_root,
            csv=args.csv,
            output_dir=args.output_dir,
            device=args.device,
        )
        return_code = 0
    else:
        report = plan_official_evaluation(
            official_root=args.official_root,
            data_folder=args.data_root,
            csv=str(args.csv),
            type_list=args.type_list,
            seed=args.seed,
            adapt=args.adapt,
            gpu_id=args.gpu_id,
            output_dir=args.output_dir,
            execute=args.execute,
        )
        return_code = int(report.get("returncode", 2 if report.get("blocked") else 0))
        if not args.execute and report.get("blocked"):
            return_code = 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
