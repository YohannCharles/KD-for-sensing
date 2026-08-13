#!/usr/bin/env python3
"""Evaluate an AMBER-Full-local or RMBP-MM-local sensing-only baseline."""

from __future__ import annotations

import argparse
import json

from kd_sensing.config import load_config
from kd_sensing.eval.sensing_baseline import run_sensing_baseline_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a sensing-only AMBER-Full/RMBP-MM validation-best checkpoint "
            "and replay the shared TBCP-3 diagnostic."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    evaluate = subparsers.add_parser("evaluate", help="Collect the 15-mask posterior and run TBCP-3.")
    evaluate.add_argument("--baseline-config", required=True, help="Resolved sensing-only baseline config.")
    evaluate.add_argument("--checkpoint", required=True, help="Published validation_best baseline checkpoint.")
    evaluate.add_argument(
        "--topology-config",
        required=True,
        help="Resolved topology-predictor config used only for ULA/protocol/power binding.",
    )
    evaluate.add_argument("--topology-likelihood", required=True, help="Train-only topology likelihood artifact.")
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device")
    evaluate.add_argument("--max-batches", type=int)
    evaluate.add_argument("--max-samples-per-pattern", type=int)
    evaluate.add_argument("--batch-size", type=int, default=256)
    evaluate.add_argument("--include-diagonal-covariance-ablation", action="store_true")
    evaluate.add_argument("--include-defense-experiments", action="store_true")
    evaluate.add_argument("--include-batch-feedback-experiments", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action != "evaluate":  # pragma: no cover - argparse enforces this.
        raise ValueError(f"Unsupported sensing baseline action {args.action!r}.")
    baseline_cfg = load_config(args.baseline_config)
    topology_cfg = load_config(args.topology_config)
    result = run_sensing_baseline_evaluation(
        baseline_cfg=baseline_cfg,
        checkpoint=args.checkpoint,
        topology_cfg=topology_cfg,
        topology_likelihood=args.topology_likelihood,
        output_dir=args.output_dir,
        device=args.device,
        max_batches=args.max_batches,
        max_samples_per_pattern=args.max_samples_per_pattern,
        batch_size=args.batch_size,
        include_diagonal_covariance_ablation=args.include_diagonal_covariance_ablation,
        include_defense_experiments=args.include_defense_experiments,
        include_batch_feedback_experiments=args.include_batch_feedback_experiments,
    )
    print(
        json.dumps(
            {
                "report": result.get("report"),
                "baseline_matrix_report": result.get("baseline_matrix_report"),
                "baseline_sample_evidence": result.get("baseline_sample_evidence"),
                "output_dir": result.get("output_dir"),
                "baseline_family": result.get("baseline_family"),
                "claim_ineligible": result.get("claim_ineligible"),
                "outer_test_accessed": result.get("outer_test_accessed"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

