from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Mapping

from kd_sensing.engine.hist_beam_loso_artifacts import _write_json
from kd_sensing.engine.hist_beam_loso_config import _source_variant_for



def _base_run_record(run: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    identity = _run_identity(run)
    return {
        "run_id": _run_id(run),
        "index": index,
        "status": "running",
        **identity,
        "source_variant": _source_variant_for(run),
        "stages": [],
        "artifacts": {},
        "metrics": {},
        "checkpoint_reuse": {},
        "failure_reason": None,
        "started_at": _utc_now(),
        "ended_at": None,
    }


def _missing_run_record(run: Mapping[str, Any], *, index: int, reason: str) -> dict[str, Any]:
    record = _base_run_record(run, index=index)
    record["status"] = "missing"
    record["failure_reason"] = reason
    record["ended_at"] = _utc_now()
    return record


def _run_identity(run: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "fold": run.get("fold"),
        "target_scene": run.get("target_scene"),
        "source_scenes": list(run.get("source_scenes", [])),
        "variant": run.get("variant"),
        "budget": run.get("budget"),
        "seed": run.get("seed"),
    }
    for key in (
        "dataset_family",
        "scene_family",
        "condition",
        "town",
        "protocol",
        "claim_scope",
        "cross_scene_claim_allowed",
        "profile",
        "modality_profile",
        "enabled_modalities",
        "excluded_sensitive_fields",
        "matrix_scope",
        "quick_validation",
    ):
        if run.get(key) is not None:
            identity[key] = run.get(key)
    return identity


def _run_id(run: Mapping[str, Any]) -> str:
    sources = "-".join(str(item) for item in run.get("source_scenes", []))
    return (
        f"{run.get('fold', 'fold')}"
        f"__src{sources}"
        f"__{run.get('variant')}"
        f"__budget{run.get('budget')}"
        f"__seed{run.get('seed')}"
    ).replace("/", "_")


def _run_dir(output_dir: Path, run_id: str) -> Path:
    return output_dir / "runs" / run_id


def _stage_started(stage: str) -> dict[str, Any]:
    return {
        "name": stage,
        "status": "running",
        "started_at": _utc_now(),
        "ended_at": None,
        "duration_seconds": None,
        "artifacts": {},
        "metrics": {},
        "checkpoint_reuse": {},
        "failure_reason": None,
    }


def _finish_stage_record(stage_record: dict[str, Any], result: Mapping[str, Any]) -> None:
    stage_record["ended_at"] = _utc_now()
    stage_record["duration_seconds"] = _duration_seconds(stage_record["started_at"], stage_record["ended_at"])
    stage_record["status"] = str(result.get("status", "completed"))
    stage_record["artifacts"] = dict(result.get("artifacts", {}))
    stage_record["metrics"] = dict(result.get("metrics", {}))
    stage_record["checkpoint_reuse"] = dict(result.get("checkpoint_reuse", {}))
    stage_record["failure_reason"] = result.get("failure_reason")
    if result.get("message"):
        stage_record["message"] = result.get("message")


def _merge_stage_artifacts(run_record: dict[str, Any], stage: str, result: Mapping[str, Any]) -> None:
    for key, value in dict(result.get("artifacts", {})).items():
        if value is not None:
            run_record["artifacts"][key] = value
            run_record["artifacts"][f"{stage}.{key}"] = value
    if result.get("metrics"):
        run_record["metrics"][stage] = dict(result["metrics"])
    if result.get("checkpoint_reuse"):
        run_record["checkpoint_reuse"][stage] = dict(result["checkpoint_reuse"])


def _write_run_metadata_stage(run_record: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    path = _write_run_metadata(run_record, run_dir)
    return {"status": "completed", "artifacts": {"run_metadata_path": str(path)}, "metrics": {}}


def _write_run_metadata(run_record: dict[str, Any], run_dir: Path) -> Path:
    run_record["ended_at"] = _utc_now()
    path = run_dir / "metadata.json"
    _write_json(path, run_record)
    return path


def _execution_status(records: list[dict[str, Any]], *, interrupted: bool = False) -> str:
    if not records:
        return "failed"
    if interrupted:
        return "partial_failed"
    incomplete = [record for record in records if record.get("status") in {"failed", "missing"}]
    if not incomplete:
        return "completed"
    if len(incomplete) == len(records):
        return "failed"
    return "partial_failed"


def _run_event_payload(run_record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_record.get("run_id"),
        "index": run_record.get("index"),
        "fold": run_record.get("fold"),
        "target_scene": run_record.get("target_scene"),
        "source_scenes": run_record.get("source_scenes"),
        "variant": run_record.get("variant"),
        "budget": run_record.get("budget"),
        "seed": run_record.get("seed"),
    }


def _duration_seconds(start: str, end: str) -> float:
    start_dt = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_dt = dt.datetime.fromisoformat(end.replace("Z", "+00:00"))
    return float((end_dt - start_dt).total_seconds())


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")