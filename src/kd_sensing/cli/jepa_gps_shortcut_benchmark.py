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
    parser.add_argument(
        "--predictive-explanatory-figures",
        action="store_true",
        help="Also write Predictive GPS-query explanatory figures from the benchmark bundle.",
    )
    parser.add_argument(
        "--predictive-explanatory-output-dir",
        help="Directory for explanatory tables/figures; defaults under the benchmark output directory.",
    )
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
        predictive_explanatory_visualizations=bool(args.predictive_explanatory_figures),
        predictive_explanatory_output_dir=args.predictive_explanatory_output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
