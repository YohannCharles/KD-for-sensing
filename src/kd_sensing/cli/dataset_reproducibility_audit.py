import argparse
import json
from pathlib import Path

from kd_sensing.diagnostics.dataset_reproducibility_audit import (
    DEFAULT_OUTPUT_DIR,
    run_dataset_audit,
    write_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a read-only dataset and reproducibility audit.")
    parser.add_argument("--dataset-family", required=True, choices=("deepsense6g", "beambench", "mmw"))
    parser.add_argument("--data-root", type=Path, help="Dataset root override.")
    parser.add_argument("--csv", type=Path, help="CSV path or path relative to --data-root.")
    parser.add_argument("--scene", help="DeepSense6G/BeamBench scene id. Defaults to 31.")
    parser.add_argument("--condition", default="sunny", help="MMW condition. Defaults to sunny.")
    parser.add_argument("--num-beams", type=int, default=64)
    parser.add_argument("--beam-shift", type=int, default=0)
    parser.add_argument("--split-metadata", type=Path, help="Optional split_metadata.json path.")
    parser.add_argument("--official-data", type=Path)
    parser.add_argument("--official-weights", type=Path)
    parser.add_argument("--official-source", type=Path)
    parser.add_argument("--official-environment", type=Path)
    parser.add_argument("--local-config", type=Path)
    parser.add_argument("--local-checkpoint-provenance")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", action="store_true", help="Print the full audit report JSON.")
    return parser


def run(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    report = run_dataset_audit(
        dataset_family=args.dataset_family,
        data_root=args.data_root,
        csv_path=args.csv,
        scene=args.scene,
        condition=args.condition,
        num_beams=args.num_beams,
        beam_shift=args.beam_shift,
        split_metadata=args.split_metadata,
        official_artifacts={
            "official_data": args.official_data,
            "official_weights": args.official_weights,
            "official_source": args.official_source,
            "official_environment": args.official_environment,
        },
        local_config=args.local_config,
        local_checkpoint_provenance=args.local_checkpoint_provenance,
    )
    outputs = write_audit_report(report, args.output_dir)
    report["outputs"] = outputs
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "dataset audit complete\n"
            f"json: {outputs['json']}\n"
            f"markdown: {outputs['markdown']}\n"
            f"official: {report['official_reproduction']['status']}\n"
            f"local substitute: {report['local_substitute']['status']}"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
