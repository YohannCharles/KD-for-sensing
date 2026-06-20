import argparse
import json
import sys

from kd_sensing.diagnostics.jepa_gps_shortcut_benchmark import run_jepa_gps_shortcut_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the JEPA vs GPS shortcut robustness benchmark.")
    parser.add_argument("--manifest", required=True, help="Path to the benchmark YAML/JSON manifest.")
    parser.add_argument("--output-dir", required=False, help="Directory for benchmark outputs.")
    parser.add_argument("--force", action="store_true", help="Allow writing into a non-empty benchmark output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Parse the manifest and write planned outputs without model execution.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = ["kd-sensing-jepa-gps-shortcut-benchmark", *(argv if argv is not None else sys.argv[1:])]
    result = run_jepa_gps_shortcut_benchmark(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        command=command,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
