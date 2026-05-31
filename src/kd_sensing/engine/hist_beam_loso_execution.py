from __future__ import annotations

from pathlib import Path
from typing import Any

from kd_sensing.engine.hist_beam_loso_artifacts import _append_execution_event, _write_json, write_loso_execute_summary
from kd_sensing.engine.hist_beam_loso_config import (
    ADAPTATION_VARIANTS,
    DEFAULT_QUICK_BUDGETS,
    DEFAULT_QUICK_SEEDS,
    DEFAULT_QUICK_TARGET_SCENES,
    DEFAULT_QUICK_VARIANTS,
    EXECUTION_PROGRESS_FILENAME,
    EXECUTION_STATUSES,
    SENSOR_ASSISTED_QUICK_BUDGETS,
    SENSOR_ASSISTED_QUICK_SEEDS,
    SENSOR_ASSISTED_QUICK_VARIANTS,
    SOURCE_ONLY_VARIANTS,
    SUPPORTED_VARIANTS,
    _prototype_decision,
    _stage_cfg,
)
from kd_sensing.engine.hist_beam_loso_preflight import run_loso_execute_preflight
from kd_sensing.engine.hist_beam_loso_records import (
    _base_run_record,
    _execution_status,
    _finish_stage_record,
    _merge_stage_artifacts,
    _missing_run_record,
    _run_dir,
    _run_event_payload,
    _stage_started,
    _write_run_metadata,
    _write_run_metadata_stage,
)
from kd_sensing.engine.hist_beam_loso_stages import (
    DefaultHistBeamLosoStageExecutor,
    StageExecutionContext,
    StageExecutor,
    StageRunCallbacks,
    _few_shot_adaptation_loaders,
    execute_loso_stage_runs,
)
from kd_sensing.engine.hist_beam_loso_summary import write_quick_validation_conclusion

def execute_loso_run_plan(
    plan: dict[str, Any],
    cfg: dict[str, Any],
    *,
    output_dir: str | Path,
    overwrite: bool = False,
    resume: bool = False,
    stage_executor: StageExecutor | None = None,
    plan_path: str | Path | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    executor: StageExecutor = stage_executor or DefaultHistBeamLosoStageExecutor()
    preflight = run_loso_execute_preflight(plan, cfg, out_dir)
    if preflight["status"] != "passed":
        error_path = out_dir / "preflight_errors.json"
        _write_json(error_path, preflight)
        return {
            "status": "failed",
            "message": "LOSO execute preflight failed before any training stage started.",
            "preflight": {**preflight, "errors_path": str(error_path)},
            "runs": [],
            "summary_paths": {},
            "plan_path": str(plan_path) if plan_path is not None else None,
        }

    preflight_path = out_dir / "preflight_metadata.json"
    _write_json(preflight_path, preflight)
    state: dict[str, Any] = {
        "source_checkpoints": {},
        "source_prototypes": {},
        "source_normalization": {},
        "source_eval": {},
        "adaptation_checkpoints": {},
    }
    runs = list(plan.get("runs", []))
    def context_factory(run_dir: Path, stage_dir: Path, run_record: dict[str, Any]) -> StageExecutionContext:
        return StageExecutionContext(
            cfg=cfg,
            output_dir=out_dir,
            run_dir=run_dir,
            stage_dir=stage_dir,
            overwrite=overwrite,
            resume=resume,
            preflight=preflight,
            state=state,
        )

    stage_result = execute_loso_stage_runs(
        runs=runs,
        output_dir=out_dir,
        stage_executor=executor,
        callbacks=StageRunCallbacks(
            base_run_record=lambda run, index: _base_run_record(run, index=index),
            missing_run_record=lambda run, index, reason: _missing_run_record(run, index=index, reason=reason),
            run_dir=_run_dir,
            write_run_metadata=_write_run_metadata,
            append_execution_event=_append_execution_event,
            run_event_payload=_run_event_payload,
            stage_started=_stage_started,
            finish_stage_record=_finish_stage_record,
            merge_stage_artifacts=_merge_stage_artifacts,
            write_run_metadata_stage=_write_run_metadata_stage,
            context_factory=context_factory,
        ),
    )
    run_records = list(stage_result["runs"])
    interrupted = bool(stage_result["interrupted"])
    interrupted_reason = stage_result["interrupted_reason"]

    status = _execution_status(run_records, interrupted=interrupted)
    summary_paths = write_loso_execute_summary(out_dir, run_records, status=status)
    conclusion_path = write_quick_validation_conclusion(out_dir, run_records, summary_paths["json"])
    summary_paths["quick_validation_conclusion"] = str(conclusion_path)
    _append_execution_event(out_dir, "execution_finished", {"status": status, "interrupted": interrupted, "summary_paths": summary_paths})
    return {
        "status": status,
        "interrupted": interrupted,
        "interrupted_reason": interrupted_reason,
        "preflight": {**preflight, "metadata_path": str(preflight_path)},
        "runs": run_records,
        "summary_paths": summary_paths,
        "plan_path": str(plan_path) if plan_path is not None else None,
    }

__all__ = [
    "ADAPTATION_VARIANTS",
    "DEFAULT_QUICK_BUDGETS",
    "DEFAULT_QUICK_SEEDS",
    "DEFAULT_QUICK_TARGET_SCENES",
    "DEFAULT_QUICK_VARIANTS",
    "SENSOR_ASSISTED_QUICK_BUDGETS",
    "SENSOR_ASSISTED_QUICK_SEEDS",
    "SENSOR_ASSISTED_QUICK_VARIANTS",
    "EXECUTION_STATUSES",
    "SOURCE_ONLY_VARIANTS",
    "SUPPORTED_VARIANTS",
    "DefaultHistBeamLosoStageExecutor",
    "StageExecutionContext",
    "execute_loso_run_plan",
    "run_loso_execute_preflight",
    "write_loso_execute_summary",
    "write_quick_validation_conclusion",
]
