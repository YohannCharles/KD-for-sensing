#!/usr/bin/env python3
"""Freeze local MMW post-selection confirmation split and fixed-mask evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kd_sensing.data.mmw.twc_evidence import DEFAULT_MASK_SEED, DEFAULT_SPLIT_SEED, prepare_protocol


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare immutable MMW TWC confirmation-fold protocol artifacts.")
    parser.add_argument("--output-root", default="outputs/cache/mmw_twc_outer_v1")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--mask-seed", type=int, default=DEFAULT_MASK_SEED)
    parser.add_argument("--exclude-csv", action="append", default=[])
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Project-relative glob of previously observed development-validation CSVs.",
    )
    args = parser.parse_args()
    exclusions = [Path(item) if Path(item).is_absolute() else ROOT / item for item in args.exclude_csv]
    for pattern in args.exclude_glob:
        exclusions.extend(sorted(ROOT.glob(pattern)))
    unique = sorted({path.resolve() for path in exclusions})
    manifest = prepare_protocol(
        ROOT / args.output_root,
        project_root=ROOT,
        split_seed=int(args.split_seed),
        mask_seed=int(args.mask_seed),
        excluded_csvs=unique,
    )
    print(
        json.dumps(
            {
                "protocol_id": manifest["protocol_id"],
                "protocol_kind": manifest["protocol_kind"],
                "manifest_sha256": manifest["manifest_sha256"],
                "domain_count": len(manifest["domains"]),
                "mask_condition_count": manifest["fixed_mask_cache"]["condition_count"],
                "manifest": str((ROOT / args.output_root / "protocol_manifest.json").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
