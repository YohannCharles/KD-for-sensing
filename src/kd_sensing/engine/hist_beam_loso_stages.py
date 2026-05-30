from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class StageRunCallbacks:
    base_run_record: Callable[[Mapping[str, Any], int], dict[str, Any]]
    missing_run_record: Callable[[Mapping[str, Any], int, str], dict[str, Any]]
    run_dir: Callable[[Path, str], Path]
    write_run_metadata: Callable[[dict[str, Any], Path], Path]
    append_execution_event: Callable[[Path, str, Mapping[str, Any]], Path]
    run_event_payload: Callable[[Mapping[str, Any]], dict[str, Any]]
    stage_started: Callable[[str], dict[str, Any]]
    finish_stage_record: Callable[[dict[str, Any], Mapping[str, Any]], None]
    merge_stage_artifacts: Callable[[dict[str, Any], str, Mapping[str, Any]], None]
    write_run_metadata_stage: Callable[[dict[str, Any], Path], dict[str, Any]]
    context_factory: Callable[[Path, Path, dict[str, Any]], Any]


def execute_loso_stage_runs(
    *,
    runs: list[Mapping[str, Any]],
    output_dir: Path,
    stage_executor: Any,
    callbacks: StageRunCallbacks,
) -> dict[str, Any]:
    run_records: list[dict[str, Any]] = []
    interrupted = False
    interrupted_reason: str | None = None
    for index, run in enumerate(runs, start=1):
        run_record = callbacks.base_run_record(run, index)
        run_dir = callbacks.run_dir(output_dir, str(run_record["run_id"]))
        run_dir.mkdir(parents=True, exist_ok=True)
        callbacks.write_run_metadata(run_record, run_dir)
        callbacks.append_execution_event(output_dir, "run_started", callbacks.run_event_payload(run_record))
        stage_failed = False
        for stage in run.get("stages", []):
            stage_record = callbacks.stage_started(str(stage))
            stage_dir = run_dir / str(stage)
            stage_dir.mkdir(parents=True, exist_ok=True)
            run_record["stages"].append(stage_record)
            callbacks.write_run_metadata(run_record, run_dir)
            callbacks.append_execution_event(
                output_dir,
                "stage_started",
                callbacks.run_event_payload(run_record) | {"stage": stage, "stage_dir": str(stage_dir)},
            )
            try:
                if stage == "summary":
                    result = callbacks.write_run_metadata_stage(run_record, run_dir)
                else:
                    result = stage_executor.execute(stage, run, callbacks.context_factory(run_dir, stage_dir, run_record))
                callbacks.finish_stage_record(stage_record, result)
                callbacks.merge_stage_artifacts(run_record, str(stage), result)
                if stage_record["status"] == "failed":
                    stage_failed = True
                    run_record["failure_reason"] = stage_record.get("failure_reason")
                    break
            except KeyboardInterrupt as exc:
                interrupted = True
                interrupted_reason = f"{type(exc).__name__}: interrupted by user"
                callbacks.finish_stage_record(stage_record, {"status": "failed", "failure_reason": interrupted_reason})
                run_record["failure_reason"] = interrupted_reason
                stage_failed = True
                callbacks.append_execution_event(
                    output_dir,
                    "stage_interrupted",
                    callbacks.run_event_payload(run_record) | {"stage": stage, "failure_reason": interrupted_reason},
                )
                break
            except Exception as exc:  # noqa: BLE001 - stage metadata must preserve failure details.
                callbacks.finish_stage_record(
                    stage_record,
                    {
                        "status": "failed",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    },
                )
                run_record["failure_reason"] = stage_record["failure_reason"]
                stage_failed = True
                break
            finally:
                callbacks.write_run_metadata(run_record, run_dir)
                callbacks.append_execution_event(
                    output_dir,
                    "stage_finished",
                    callbacks.run_event_payload(run_record)
                    | {"stage": stage, "stage_status": stage_record["status"], "failure_reason": stage_record.get("failure_reason")},
                )
        run_record["status"] = "failed" if stage_failed else "completed"
        callbacks.write_run_metadata(run_record, run_dir)
        callbacks.append_execution_event(output_dir, "run_finished", callbacks.run_event_payload(run_record) | {"run_status": run_record["status"]})
        run_records.append(run_record)
        if interrupted:
            for remaining_index, remaining_run in enumerate(runs[index:], start=index + 1):
                run_records.append(
                    callbacks.missing_run_record(
                        remaining_run,
                        remaining_index,
                        interrupted_reason or "execution_interrupted",
                    )
                )
            break
    return {
        "runs": run_records,
        "interrupted": interrupted,
        "interrupted_reason": interrupted_reason,
    }


__all__ = ["StageRunCallbacks", "execute_loso_stage_runs"]
