#!/usr/bin/env python3
"""Audit the local ULA-DFT topology that produced MMW beam labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kd_sensing.data.mmw.codebook_topology import audit_mmw_codebook_topology


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a read-only MMW ULA-DFT codebook topology audit.")
    parser.add_argument("--output-root", default="outputs/cache/mmw_codebook_topology/v1")
    parser.add_argument("--replay-samples-per-domain", type=int, default=4)
    parser.add_argument("--endpoint-samples-per-domain", type=int, default=8)
    args = parser.parse_args()
    if args.replay_samples_per_domain <= 0 or args.endpoint_samples_per_domain <= 0:
        parser.error("sample counts must be positive")
    manifest = audit_mmw_codebook_topology(
        ROOT,
        ROOT / args.output_root,
        replay_samples_per_domain=int(args.replay_samples_per_domain),
        endpoint_samples_per_domain=int(args.endpoint_samples_per_domain),
    )
    print(
        json.dumps(
            {
                "topology_id": manifest["descriptor"]["topology_id"],
                "descriptor_sha256": manifest["descriptor_sha256"],
                "power_replay_count": manifest["power_replay_count"],
                "metadata_consistent": manifest["metadata_consistent"],
                "claim_boundary": manifest["descriptor"]["claim_boundary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
