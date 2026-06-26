import argparse
import json
import sys
from pathlib import Path
from typing import Any

from kd_sensing.baselines.rmbp_mm import DEFAULT_OUTPUT_ROOT, run_source_audit_dry_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a WCL 2025 missing-modality source-audit dry-run manifest.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Ignored output root for dry-run artifacts.")
    parser.add_argument("--manifest", type=Path, help="Manifest path. Defaults to <output-root>/source_audit_manifest.json.")
    parser.add_argument("--official-code-url", help="Optional official code URL if found locally by the maintainer.")
    parser.add_argument("--source-commit", help="Optional official source commit.")
    parser.add_argument("--checkpoint-uri", help="Optional official checkpoint URI/path.")
    parser.add_argument("--json", action="store_true", help="Print the full manifest JSON.")
    return parser


def run(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = ["kd-sensing-wcl2025-missing-modality-audit", *(argv if argv is not None else sys.argv[1:])]
    manifest = run_source_audit_dry_run(
        output_root=args.output_root,
        manifest_path=args.manifest,
        command_args=command,
        official_code_url=args.official_code_url,
        source_commit=args.source_commit,
        checkpoint_uri=args.checkpoint_uri,
    )
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        official = manifest["branches"]["official_code"]
        local = manifest["branches"]["local_substitute"]
        print(
            "WCL 2025 missing-modality source audit\n"
            f"manifest: {manifest['metadata']['manifest_path']}\n"
            f"official: {official['status']}\n"
            f"local substitute: {local['claim_status']}"
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
