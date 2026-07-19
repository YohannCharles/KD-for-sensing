#!/usr/bin/env python3
"""Prepare the immutable exact-cardinality MMW temporal token-stress extension."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kd_sensing.data.mmw.twc_temporal_token_stress import STRESS_MASK_SEED, prepare_temporal_token_stress_protocol


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = "outputs/cache/mmw_twc_temporal_token_stress_v3"
DEFAULT_PARENT_PROTOCOL = "outputs/cache/mmw_twc_outer_v1/protocol_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--parent-protocol-manifest", default=DEFAULT_PARENT_PROTOCOL)
    parser.add_argument("--mask-seed", type=int, default=STRESS_MASK_SEED)
    args = parser.parse_args()
    try:
        manifest = prepare_temporal_token_stress_protocol(
            _repo_path(args.output_root),
            parent_protocol_manifest=_repo_path(args.parent_protocol_manifest),
            mask_seed=int(args.mask_seed),
        )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "refused", "type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
