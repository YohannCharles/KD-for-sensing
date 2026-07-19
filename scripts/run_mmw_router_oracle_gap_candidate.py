#!/usr/bin/env python3
"""Run all fixed Oracle-Gap conditions for one Router-screen candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from eval_mmw_router_oracle_gap import CONDITIONS, PROTOCOL_ID
from kd_sensing.evaluation.corruptions import CORRUPTION_PARAMETERS
from summarize_mmw_router_oracle_gap import summarize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    config = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    jobs = []
    for condition in CONDITIONS:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("eval_mmw_router_oracle_gap.py")),
                "--condition",
                condition,
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--output-root",
                str(root),
                "--batch-size",
                str(args.batch_size),
            ],
            check=True,
        )
        jobs.append({"condition": condition, "status": "complete", "output": str(root / condition)})
    request = {
        "protocol": PROTOCOL_ID,
        "candidate": args.candidate,
        "config": str(config),
        "config_sha256": _sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "conditions": list(CONDITIONS),
        "corruption_parameters": CORRUPTION_PARAMETERS,
        "corruption_seed": 20260718,
        "batch_size": int(args.batch_size),
        "split": "frozen_inner_validation_only",
        "claim_eligible": False,
    }
    manifest = {
        "schema_version": 1,
        "request": request,
        "request_sha256": hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "jobs": jobs,
    }
    _write_json(root / "evaluation_manifest.json", manifest)
    summarize(root)
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
