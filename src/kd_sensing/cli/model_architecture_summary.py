import argparse
import json
import sys
from pathlib import Path
from typing import Any

from kd_sensing.models.architecture_summary import (
    render_architecture_summary,
    summarize_model_config,
    summarize_startup_summary_artifact,
    summarize_sweep_candidate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize model architecture and parameter counts without training.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="Training/evaluation config path to summarize.")
    source.add_argument("--model-config-json", help="Inline JSON object for model.primary config.")
    source.add_argument("--sweep-manifest", type=Path, help="CNN/hybrid JEPA visual-prior sweep manifest path.")
    source.add_argument("--startup-summary", type=Path, help="Existing startup_summary.json artifact to read.")
    parser.add_argument("-o", "--override", action="append", default=[], help="Config override using load_config dotted syntax.")
    parser.add_argument("--variant-id", default="", help="Sweep variant id to render, or 'all'.")
    parser.add_argument(
        "--format",
        choices=("json", "markdown", "csv"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument("--output", type=Path, help="Optional output file. Defaults to stdout.")
    parser.add_argument("--allow-download", action="store_true", help="Allow checkpoint downloads during config summary build.")
    parser.add_argument("--no-build", action="store_true", help="Only run config preflight; do not build the model.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summaries = _summaries_from_args(args)
    rendered = render_architecture_summary(summaries if len(summaries) != 1 else summaries[0], format=args.format)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered if rendered.endswith("\n") else rendered + "\n", encoding="utf-8")
    else:
        sys.stdout.write(rendered if rendered.endswith("\n") else rendered + "\n")
    return 0


def _summaries_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.config is not None:
        return [
            summarize_model_config(
                args.config,
                overrides=args.override,
                build=not args.no_build,
                allow_download=bool(args.allow_download),
            )
        ]
    if args.model_config_json:
        payload = json.loads(args.model_config_json)
        if not isinstance(payload, dict):
            raise SystemExit("--model-config-json must decode to a JSON object.")
        return [
            summarize_model_config(
                payload,
                overrides=args.override,
                build=not args.no_build,
                allow_download=bool(args.allow_download),
            )
        ]
    if args.sweep_manifest is not None:
        return _summaries_from_sweep_manifest(args.sweep_manifest, variant_id=str(args.variant_id or "all"))
    if args.startup_summary is not None:
        return [summarize_startup_summary_artifact(args.startup_summary)]
    raise SystemExit("One input source is required.")


def _summaries_from_sweep_manifest(path: Path, *, variant_id: str) -> list[dict[str, Any]]:
    from kd_sensing.diagnostics.cnn_hybrid_jepa_visual_prior_sweep import load_full_sweep_manifest

    manifest = load_full_sweep_manifest(path)
    candidates = list(manifest["base_candidates"])
    if variant_id and variant_id != "all":
        candidates = [candidate for candidate in candidates if str(candidate.get("variant_id")) == variant_id]
        if not candidates:
            raise SystemExit(f"Sweep manifest {path} has no variant_id {variant_id!r}.")
    return [summarize_sweep_candidate(candidate) for candidate in candidates]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
