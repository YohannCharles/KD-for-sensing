import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable

from kd_sensing.diagnostics.research_claim_harvester_base import (
    DEFAULT_LEDGER_DIR,
    LedgerRecord,
    _ensure_utc,
    _format_dt,
    _utc_now,
)


def ledger_records_from_candidates(candidates: Iterable[dict[str, Any]], *, generated_at: str | None = None) -> list[dict[str, Any]]:
    generated = generated_at or _format_dt(_utc_now())
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        warnings = candidate.get("warnings", [])
        caveat = "; ".join(warning.get("message") or warning.get("kind", "") for warning in warnings if warning) or "draft candidate"
        records.append(
            LedgerRecord(
                run_id=str(candidate.get("run_id") or candidate.get("candidate_id")),
                run_name=candidate.get("run_name"),
                config_path=candidate.get("config_path"),
                config_digest=candidate.get("config_digest"),
                seed=candidate.get("seed"),
                scene_scope=candidate.get("scene_scope"),
                artifact_paths=dict(candidate.get("artifact_paths", {})),
                metric_summary=dict(candidate.get("metrics", {})),
                claim_status=str(candidate.get("claim_status") or "draft"),
                comparability_status=str(candidate.get("comparability_status") or "needs_review"),
                caveat=caveat,
                generated_at=generated,
                candidate_id=candidate.get("candidate_id"),
            ).to_dict()
        )
    return records

def write_jsonl_ledger(
    records: Iterable[dict[str, Any]],
    *,
    ledger_dir: str | Path = DEFAULT_LEDGER_DIR,
    now: dt.datetime | None = None,
) -> Path:
    generated = _format_dt(_utc_now() if now is None else _ensure_utc(now))
    stamp = generated.replace("-", "").replace(":", "").replace("+00:00", "Z").replace(".", "_")
    target_dir = Path(ledger_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"research_ledger_{stamp}.jsonl"
    with target.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    return target

def write_ledger_csv(records: Iterable[dict[str, Any]], *, output_path: str | Path) -> Path:
    rows = list(records)
    target = Path(output_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "run_name",
        "config_path",
        "config_digest",
        "seed",
        "scene_scope",
        "metric_summary",
        "claim_status",
        "comparability_status",
        "caveat",
        "generated_at",
        "candidate_id",
    ]
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            copy = dict(row)
            copy["metric_summary"] = json.dumps(copy.get("metric_summary", {}), sort_keys=True)
            writer.writerow({key: copy.get(key) for key in fieldnames})
    return target
