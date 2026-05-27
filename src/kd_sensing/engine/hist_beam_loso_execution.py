from __future__ import annotations

import csv
import datetime as dt
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Protocol

from kd_sensing.data.loso import SUPPORTED_LABEL_BUDGETS, sample_few_shot_records
from kd_sensing.data.scenes import normalize_deepsense_dataset_config, retarget_deepsense_dataset_config
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.utils.paths import resolve_path
from kd_sensing.utils.seed import set_seed


EXECUTION_STATUSES = ("completed", "failed", "partial_failed")
SOURCE_ONLY_VARIANTS = {"v0_flat", "v1_hierarchical", "v2_shared_private", "v3_decoupled"}
ADAPTATION_VARIANTS = {"v4_adapter", "v5_adapter_proto", "v6_full_finetune"}
SUPPORTED_VARIANTS = SOURCE_ONLY_VARIANTS | ADAPTATION_VARIANTS
DEFAULT_QUICK_VARIANTS = ["v0_flat", "v3_decoupled", "v4_adapter", "v5_adapter_proto", "v6_full_finetune"]
DEFAULT_QUICK_BUDGETS = [0, 10]
DEFAULT_QUICK_SEEDS = [0]
DEFAULT_QUICK_TARGET_SCENES = [34]
EXECUTION_PROGRESS_FILENAME = "execution_progress.jsonl"


@dataclass(frozen=True)
class StageExecutionContext:
    cfg: dict[str, Any]
    output_dir: Path
    run_dir: Path
    stage_dir: Path
    overwrite: bool
    resume: bool
    preflight: dict[str, Any]
    state: dict[str, Any]


class StageExecutor(Protocol):
    def execute(
        self,
        stage: str,
        run: Mapping[str, Any],
        context: StageExecutionContext,
    ) -> dict[str, Any]:
        ...


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
        "source_eval": {},
        "adaptation_checkpoints": {},
    }
    run_records: list[dict[str, Any]] = []
    runs = list(plan.get("runs", []))
    interrupted = False
    interrupted_reason: str | None = None
    for index, run in enumerate(runs, start=1):
        run_record = _base_run_record(run, index=index)
        run_dir = _run_dir(out_dir, run_record["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_run_metadata(run_record, run_dir)
        _append_execution_event(out_dir, "run_started", _run_event_payload(run_record))
        stage_failed = False
        for stage in run.get("stages", []):
            stage_record = _stage_started(stage)
            stage_dir = run_dir / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            run_record["stages"].append(stage_record)
            _write_run_metadata(run_record, run_dir)
            _append_execution_event(
                out_dir,
                "stage_started",
                _run_event_payload(run_record) | {"stage": stage, "stage_dir": str(stage_dir)},
            )
            try:
                if stage == "summary":
                    result = _write_run_metadata_stage(run_record, run_dir)
                else:
                    context = StageExecutionContext(
                        cfg=cfg,
                        output_dir=out_dir,
                        run_dir=run_dir,
                        stage_dir=stage_dir,
                        overwrite=overwrite,
                        resume=resume,
                        preflight=preflight,
                        state=state,
                    )
                    result = executor.execute(stage, run, context)
                _finish_stage_record(stage_record, result)
                _merge_stage_artifacts(run_record, stage, result)
                if stage_record["status"] == "failed":
                    stage_failed = True
                    run_record["failure_reason"] = stage_record.get("failure_reason")
                    break
            except KeyboardInterrupt as exc:
                interrupted = True
                interrupted_reason = f"{type(exc).__name__}: interrupted by user"
                _finish_stage_record(stage_record, {"status": "failed", "failure_reason": interrupted_reason})
                run_record["failure_reason"] = interrupted_reason
                stage_failed = True
                _append_execution_event(
                    out_dir,
                    "stage_interrupted",
                    _run_event_payload(run_record) | {"stage": stage, "failure_reason": interrupted_reason},
                )
                break
            except Exception as exc:  # noqa: BLE001 - metadata must preserve stage failure.
                _finish_stage_record(
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
                _write_run_metadata(run_record, run_dir)
                _append_execution_event(
                    out_dir,
                    "stage_finished",
                    _run_event_payload(run_record)
                    | {"stage": stage, "stage_status": stage_record["status"], "failure_reason": stage_record.get("failure_reason")},
                )
        run_record["status"] = "failed" if stage_failed else "completed"
        _write_run_metadata(run_record, run_dir)
        _append_execution_event(out_dir, "run_finished", _run_event_payload(run_record) | {"run_status": run_record["status"]})
        run_records.append(run_record)
        if interrupted:
            for remaining_index, remaining_run in enumerate(runs[index:], start=index + 1):
                run_records.append(_missing_run_record(remaining_run, index=remaining_index, reason=interrupted_reason or "execution_interrupted"))
            break

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


def run_loso_execute_preflight(plan: dict[str, Any], cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(output_dir)
    errors: list[dict[str, Any]] = []
    checked_paths: list[dict[str, Any]] = []
    matrix = _matrix_summary(plan)
    runs = list(plan.get("runs", []))
    if not runs:
        errors.append(_preflight_error("matrix", "runs", None, "LOSO execute matrix contains no runs.", None))
    for variant in matrix["variants"]:
        if variant not in SUPPORTED_VARIANTS:
            errors.append(_preflight_error("matrix", "variant", None, f"Unsupported variant '{variant}'.", None))
    for budget in matrix["budgets"]:
        if int(budget) not in SUPPORTED_LABEL_BUDGETS:
            errors.append(
                _preflight_error(
                    "matrix",
                    "budget",
                    None,
                    f"Unsupported label budget '{budget}'. Supported budgets: {list(SUPPORTED_LABEL_BUDGETS)}.",
                    None,
                )
            )
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".loso_preflight_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checked_paths.append({"resource_type": "output_dir", "path": str(out_dir), "status": "ok"})
    except Exception as exc:  # noqa: BLE001
        errors.append(
            _preflight_error(
                "output",
                "output_dir",
                str(out_dir),
                f"Output directory is not writable: {exc}",
                None,
            )
        )

    enabled_modalities = _enabled_modalities(plan, cfg)
    scene_ids = sorted(
        {
            int(scene)
            for run in runs
            for scene in [run.get("target_scene"), *list(run.get("source_scenes", []))]
            if scene is not None
        }
    )
    for scene in scene_ids:
        scene_cfg = _cfg_for_scene(cfg, scene)
        dataset_cfg = scene_cfg.get("data", {}).get("dataset", {})
        data_root = resolve_path(dataset_cfg.get("data_root", "."))
        if not data_root.exists() or not data_root.is_dir():
            errors.append(
                _preflight_error(
                    scene,
                    "data_root",
                    str(data_root),
                    f"Scene {scene} data root is missing.",
                    _runs_for_scene(runs, scene),
                )
            )
            continue
        checked_paths.append({"scene": scene, "resource_type": "data_root", "path": str(data_root), "status": "ok"})
        for split_key, csv_name in (
            ("train_csv", dataset_cfg.get("train_csv_name")),
            ("test_csv", dataset_cfg.get("test_csv_name")),
        ):
            csv_path = _resolve_csv_path(data_root, csv_name)
            if csv_path is None or not csv_path.exists():
                errors.append(
                    _preflight_error(
                        scene,
                        split_key,
                        str(csv_path) if csv_path is not None else None,
                        f"Scene {scene} required CSV is missing.",
                        _runs_for_scene(runs, scene),
                    )
                )
                continue
            checked_paths.append({"scene": scene, "resource_type": split_key, "path": str(csv_path), "status": "ok"})
            errors.extend(
                _preflight_csv_resources(
                    scene=scene,
                    csv_path=csv_path,
                    data_root=data_root,
                    enabled_modalities=enabled_modalities,
                    cfg=scene_cfg,
                    runs=_runs_for_scene(runs, scene),
                )
            )

    return {
        "status": "passed" if not errors else "failed",
        "checked_at": _utc_now(),
        "checked_scenes": scene_ids,
        "checked_paths": checked_paths,
        "enabled_modalities": list(enabled_modalities),
        "output_dir": str(out_dir),
        "matrix": matrix,
        "errors": errors,
    }


class DefaultHistBeamLosoStageExecutor:
    def execute(
        self,
        stage: str,
        run: Mapping[str, Any],
        context: StageExecutionContext,
    ) -> dict[str, Any]:
        if stage == "source_train":
            return self._source_train(run, context)
        if stage == "source_only_target_test_eval":
            return self._source_only_eval(run, context)
        if stage == "target_adaptation":
            return self._target_adaptation(run, context)
        if stage == "adapted_target_test_eval":
            return self._adapted_eval(run, context)
        raise ValueError(f"Unsupported LOSO execute stage '{stage}'.")

    def _source_train(self, run: Mapping[str, Any], context: StageExecutionContext) -> dict[str, Any]:
        variant = _source_variant_for(run)
        cache_key = _source_cache_key(run, variant)
        if cache_key in context.state["source_checkpoints"] and _reuse_source_checkpoint(context.cfg):
            cached = context.state["source_checkpoints"][cache_key]
            return {
                "status": "completed",
                "message": "Reused source checkpoint from an earlier run in this execution.",
                "checkpoint_reuse": {"enabled": True, "reused": True, "cache_key": cache_key},
                "artifacts": dict(cached.get("artifacts", {})),
                "metrics": cached.get("metrics", {}),
            }

        import torch

        from kd_sensing.engine.data_factory import shutdown_dataloader_workers
        from kd_sensing.engine.hist_beam_prototypes import generate_source_prototypes, prototype_coverage_from_counts
        from kd_sensing.engine.loso_data import build_loso_dataloaders
        from kd_sensing.engine.optim import build_device, build_model, build_optimizer, build_task_criterion
        from kd_sensing.engine.runtime import run_model_step, transfer_non_blocking

        cfg = _stage_cfg(context.cfg, run, variant=variant, stage_name="source_train", stage_dir=context.stage_dir)
        set_seed(cfg.get("experiment", {}).get("seed", 0))
        device = build_device(cfg)
        loaders = build_loso_dataloaders(cfg, dict(run), split_seed=int(run.get("seed", 0)))
        model = build_model(cfg["model"]["student"]).to(device)
        optimizer = build_optimizer(cfg, model)
        criterion = build_task_criterion(cfg)
        task = cfg.get("experiment", {}).get("task", "fusion")
        model_cfg = cfg["model"]
        student_cfg = model_cfg.get("student", model_cfg)
        num_classes = int(student_cfg.get("num_classes", model_cfg.get("num_classes", 64)))
        epochs = int(cfg.get("training", {}).get("epochs", 1))
        losses: list[float] = []
        non_blocking = transfer_non_blocking(cfg)
        progress_path = context.stage_dir / "progress.jsonl"
        try:
            for epoch_index in range(epochs):
                epoch_start = time.perf_counter()
                epoch_losses: list[float] = []
                model.train()
                for batch in loaders["source_train"]:
                    optimizer.zero_grad()
                    step = run_model_step(
                        model,
                        task,
                        batch,
                        model_cfg=student_cfg,
                        seq_length=model_cfg.get("seq_length_student", cfg.get("data", {}).get("dataset", {}).get("seq_len", 8)),
                        num_pred=model_cfg.get("num_pred", cfg.get("data", {}).get("dataset", {}).get("num_pred", 1)),
                        device=device,
                        downsample_ratio=model_cfg.get("downsample_ratio", 1),
                        non_blocking=non_blocking,
                    )
                    if step.labels is None:
                        raise RuntimeError("Source training labels were not prepared.")
                    loss = criterion(step.logits.reshape(-1, num_classes), step.labels.flatten())
                    loss.backward()
                    optimizer.step()
                    loss_value = float(loss.detach().cpu().item())
                    losses.append(loss_value)
                    epoch_losses.append(loss_value)
                _append_stage_progress(
                    context.stage_dir,
                    "source_train",
                    {
                        "epoch": epoch_index + 1,
                        "epochs": epochs,
                        "duration_seconds": float(time.perf_counter() - epoch_start),
                        "loss_last": epoch_losses[-1] if epoch_losses else None,
                        "loss_mean": float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else None,
                        "batches": len(epoch_losses),
                    },
                )
            checkpoint_path = context.stage_dir / "source_checkpoint.pth"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "metadata": _run_identity(run) | {"source_variant": variant, "stage": "source_train"},
                },
                checkpoint_path,
            )
            prototype_path = context.stage_dir / "source_prototypes.pt"
            prototype_artifact = generate_source_prototypes(
                model,
                loaders["source_train"],
                cfg,
                device,
                output_path=prototype_path,
                metadata=_run_identity(run) | {"source_variant": variant},
            )
            prototype_coverage = prototype_coverage_from_counts(prototype_artifact["counts"])
            metrics = {
                "train_loss_last": losses[-1] if losses else None,
                "train_loss_mean": sum(losses) / len(losses) if losses else None,
                "epochs": epochs,
                "source_variant": variant,
                "prototype_coverage": prototype_coverage,
            }
            metrics_path = context.stage_dir / "metrics.json"
            _write_json(metrics_path, metrics)
            artifacts = {
                "run_dir": str(context.stage_dir),
                "metrics_path": str(metrics_path),
                "source_checkpoint_path": str(checkpoint_path),
                "source_prototype_path": str(prototype_path),
                "progress_path": str(progress_path),
            }
            result = {
                "status": "completed",
                "artifacts": artifacts,
                "metrics": metrics,
                "checkpoint_reuse": {"enabled": _reuse_source_checkpoint(context.cfg), "reused": False, "cache_key": cache_key},
            }
            context.state["source_checkpoints"][cache_key] = result
            context.state["source_prototypes"][cache_key] = str(prototype_path)
            return result
        finally:
            for key in ("source_train", "target_adapt", "target_test"):
                loader = loaders.get(key)
                if loader is not None:
                    shutdown_dataloader_workers(loader)

    def _source_only_eval(self, run: Mapping[str, Any], context: StageExecutionContext) -> dict[str, Any]:
        variant = _source_variant_for(run)
        checkpoint = self._source_checkpoint_for(run, context, variant=variant)
        cfg = _stage_cfg(context.cfg, run, variant=variant, stage_name="source_only_target_test_eval", stage_dir=context.stage_dir)
        return _evaluate_target_test(
            cfg,
            run,
            context,
            checkpoint_path=checkpoint,
            variant=run.get("variant"),
            summary_type="source_only",
            stage_name="source_only_target_test_eval",
        )

    def _target_adaptation(self, run: Mapping[str, Any], context: StageExecutionContext) -> dict[str, Any]:
        variant = str(run.get("variant"))
        if variant in SOURCE_ONLY_VARIANTS:
            return {
                "status": "skipped",
                "message": "Source-only variant does not run target adaptation.",
                "artifacts": {},
                "metrics": {"prototype_coverage_unavailable_reason": "source_only_variant"},
            }

        import torch
        from torch.utils.data import DataLoader, Subset

        from kd_sensing.engine.data_factory import build_dataloader_kwargs, shutdown_dataloader_workers
        from kd_sensing.engine.hist_beam_adaptation import (
            adapt_hist_beam_target,
            apply_hist_beam_adaptation_strategy,
            trainable_parameter_summary,
        )
        from kd_sensing.engine.hist_beam_prototypes import load_source_prototypes, prototype_coverage_from_counts
        from kd_sensing.engine.loso_data import build_loso_dataloaders
        from kd_sensing.engine.optim import build_device, build_model, build_optimizer

        source_variant = _source_variant_for(run)
        source_checkpoint = self._source_checkpoint_for(run, context, variant=source_variant)
        cfg = _stage_cfg(context.cfg, run, variant=variant, stage_name="target_adaptation", stage_dir=context.stage_dir)
        set_seed(cfg.get("experiment", {}).get("seed", 0))
        device = build_device(cfg)
        loaders = build_loso_dataloaders(cfg, dict(run), split_seed=int(run.get("seed", 0)))
        try:
            model = build_model(cfg["model"]["student"]).to(device)
            _load_checkpoint_state(model, source_checkpoint, device=device, strict=False)
            strategy = "v6_full_finetune" if variant == "v6_full_finetune" else variant
            strategy_metadata = apply_hist_beam_adaptation_strategy(model, strategy)
            optimizer = build_optimizer(cfg, model)
            target_adapt_loader = loaders["target_adapt"]
            labeled_loader, unlabeled_loader, sampling = _few_shot_adaptation_loaders(
                target_adapt_loader.dataset,
                cfg,
                run,
                loader_kwargs=build_dataloader_kwargs(cfg["data"]["dataloader"], split="train"),
            )
            prototypes = None
            prototype_metadata: dict[str, Any]
            if variant == "v5_adapter_proto":
                proto_path = self._source_prototype_for(run, context, variant=source_variant)
                if proto_path is not None and Path(proto_path).exists():
                    prototypes = load_source_prototypes(proto_path, map_location=device)
                    prototype_metadata = {
                        "source_prototype_path": str(proto_path),
                        **prototype_coverage_from_counts(prototypes["counts"]),
                    }
                else:
                    prototype_metadata = {
                        "prototype_coverage_available": False,
                        "prototype_coverage_unavailable_reason": "source_prototype_missing",
                    }
            else:
                prototype_metadata = {
                    "prototype_coverage_available": False,
                    "prototype_coverage_unavailable_reason": "variant_without_prototype_alignment",
                }
            adaptation = adapt_hist_beam_target(
                model,
                labeled_loader,
                unlabeled_loader,
                cfg,
                device,
                optimizer,
                prototypes=prototypes,
                epochs=int(cfg.get("hist_beam", {}).get("adaptation", {}).get("epochs", cfg.get("training", {}).get("adaptation_epochs", 1))),
                confidence_threshold=float(cfg.get("hist_beam", {}).get("prototype", {}).get("confidence_threshold", 0.0)),
                progress_callback=lambda progress: _append_stage_progress(context.stage_dir, "target_adaptation", progress),
            )
            params = trainable_parameter_summary(model).to_dict()
            metrics = {
                **strategy_metadata,
                **params,
                **adaptation,
                "adaptation_strategy": strategy,
                "source_checkpoint_path": str(source_checkpoint),
                "sampling": sampling,
                **prototype_metadata,
            }
            checkpoint_path = context.stage_dir / "adaptation_checkpoint.pth"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "metadata": _run_identity(run) | {"stage": "target_adaptation", "adaptation_strategy": strategy},
                },
                checkpoint_path,
            )
            metrics_path = context.stage_dir / "metrics.json"
            _write_json(metrics_path, metrics)
            artifacts = {
                "run_dir": str(context.stage_dir),
                "metrics_path": str(metrics_path),
                "source_checkpoint_path": str(source_checkpoint),
                "adaptation_checkpoint_path": str(checkpoint_path),
                "source_prototype_path": prototype_metadata.get("source_prototype_path"),
                "progress_path": str(context.stage_dir / "progress.jsonl"),
            }
            key = _adaptation_cache_key(run)
            context.state["adaptation_checkpoints"][key] = {
                "checkpoint_path": str(checkpoint_path),
                "metrics": metrics,
                "artifacts": artifacts,
            }
            return {
                "status": "completed",
                "artifacts": artifacts,
                "metrics": metrics,
                "checkpoint_reuse": {
                    "source_checkpoint_path": str(source_checkpoint),
                    "source_variant": source_variant,
                },
            }
        finally:
            for loader in list(loaders.values()):
                if hasattr(loader, "_iterator"):
                    shutdown_dataloader_workers(loader)

    def _adapted_eval(self, run: Mapping[str, Any], context: StageExecutionContext) -> dict[str, Any]:
        variant = str(run.get("variant"))
        if variant in SOURCE_ONLY_VARIANTS:
            return {
                "status": "skipped",
                "message": "Source-only variant does not run adapted target_test evaluation.",
                "artifacts": {},
                "metrics": {},
            }
        cached = context.state["adaptation_checkpoints"].get(_adaptation_cache_key(run))
        if not cached:
            return {
                "status": "failed",
                "failure_reason": "target_adaptation_checkpoint_missing",
                "artifacts": {},
                "metrics": {},
            }
        cfg = _stage_cfg(context.cfg, run, variant=variant, stage_name="adapted_target_test_eval", stage_dir=context.stage_dir)
        return _evaluate_target_test(
            cfg,
            run,
            context,
            checkpoint_path=Path(cached["checkpoint_path"]),
            variant=variant,
            summary_type="adapted",
            stage_name="adapted_target_test_eval",
        )

    @staticmethod
    def _source_checkpoint_for(run: Mapping[str, Any], context: StageExecutionContext, *, variant: str) -> Path:
        cached = context.state["source_checkpoints"].get(_source_cache_key(run, variant))
        path = ((cached or {}).get("artifacts") or {}).get("source_checkpoint_path")
        if not path:
            raise RuntimeError(f"Source checkpoint is unavailable for variant {variant}.")
        return Path(path)

    @staticmethod
    def _source_prototype_for(run: Mapping[str, Any], context: StageExecutionContext, *, variant: str) -> str | None:
        return context.state["source_prototypes"].get(_source_cache_key(run, variant))


def write_loso_execute_summary(output_dir: str | Path, run_records: list[dict[str, Any]], *, status: str) -> dict[str, str]:
    out_dir = Path(output_dir)
    rows = [_summary_row(record) for record in run_records]
    payload = {
        "status": status,
        "generated_at": _utc_now(),
        "run_count": len(run_records),
        "runs": rows,
    }
    json_path = out_dir / "loso_summary.json"
    csv_path = out_dir / "loso_summary.csv"
    _write_json(json_path, payload)
    _write_summary_csv(csv_path, rows)
    return {"json": str(json_path), "csv": str(csv_path)}


def write_quick_validation_conclusion(
    output_dir: str | Path,
    run_records: list[dict[str, Any]],
    summary_path: str | Path,
) -> Path:
    out_dir = Path(output_dir)
    rows = [_summary_row(record) for record in run_records]
    by_key = {
        (row["target_scene"], row["budget"], row["seed"], row["variant"]): row
        for row in rows
    }
    comparisons: list[dict[str, Any]] = []
    groups = sorted({(row["target_scene"], row["budget"], row["seed"]) for row in rows})
    for target_scene, budget, seed in groups:
        baseline = by_key.get((target_scene, budget, seed, "v3_decoupled"))
        for variant in ("v4_adapter", "v5_adapter_proto"):
            candidate = by_key.get((target_scene, budget, seed, variant))
            comparisons.append(
                _compare_adapter_to_source(
                    target_scene=target_scene,
                    budget=budget,
                    seed=seed,
                    variant=variant,
                    baseline=baseline,
                    candidate=candidate,
                )
            )
        comparisons.append(
            _compare_proto_to_full(
                target_scene=target_scene,
                budget=budget,
                seed=seed,
                proto=by_key.get((target_scene, budget, seed, "v5_adapter_proto")),
                full=by_key.get((target_scene, budget, seed, "v6_full_finetune")),
            )
        )
    payload = {
        "generated_at": _utc_now(),
        "summary_path": str(summary_path),
        "status": "completed" if comparisons and all(item["status"] == "complete" for item in comparisons) else "inconclusive",
        "comparisons": comparisons,
    }
    path = out_dir / "quick_validation_conclusion.json"
    _write_json(path, payload)
    return path


def _evaluate_target_test(
    cfg: dict[str, Any],
    run: Mapping[str, Any],
    context: StageExecutionContext,
    *,
    checkpoint_path: Path,
    variant: Any,
    summary_type: str,
    stage_name: str,
) -> dict[str, Any]:
    from kd_sensing.engine.data_factory import shutdown_dataloader_workers
    from kd_sensing.engine.evaluation_pass import run_evaluation_pass
    from kd_sensing.engine.loso_data import build_loso_dataloaders
    from kd_sensing.engine.optim import build_device, build_model, build_task_criterion
    from kd_sensing.evaluation.hist_beam_outputs import write_hist_beam_predictions

    device = build_device(cfg)
    loaders = build_loso_dataloaders(cfg, dict(run), split_seed=int(run.get("seed", 0)))
    try:
        model = build_model(cfg["model"]["student"]).to(device)
        _load_checkpoint_state(model, checkpoint_path, device=device, strict=False)
        criterion = build_task_criterion(cfg)
        result = run_evaluation_pass(model, loaders["target_test"], cfg, criterion, device)
        metrics = dict(result.metrics)
        metrics.update(
            {
                "summary_type": summary_type,
                "stage": stage_name,
                "variant": str(variant),
                "split": "target_test",
                "source_checkpoint_path": str(checkpoint_path),
            }
        )
        metrics_path = context.stage_dir / "metrics.json"
        predictions_path = context.stage_dir / "predictions.csv"
        _write_json(metrics_path, metrics)
        setup = dict(metrics.get("prediction_setup", {})) if isinstance(metrics.get("prediction_setup"), dict) else {}
        setup.update(_run_identity(run))
        setup.update({"variant": str(variant), "split": "target_test", "summary_type": summary_type})
        write_hist_beam_predictions(
            predictions_path,
            result.outputs,
            result.labels,
            metadata=result.metadata,
            group_size=int(cfg.get("hist_beam", {}).get("group_size", cfg.get("model", {}).get("student", {}).get("group_size", 8))),
            top_k=max(int(value) for value in cfg.get("evaluation", {}).get("k_values", [1, 3, 5])),
            variant_metadata=setup,
        )
        return {
            "status": "completed",
            "artifacts": {
                "run_dir": str(context.stage_dir),
                "metrics_path": str(metrics_path),
                "predictions_path": str(predictions_path),
                "source_checkpoint_path": str(checkpoint_path),
            },
            "metrics": metrics,
        }
    finally:
        for loader in list(loaders.values()):
            if hasattr(loader, "_iterator"):
                shutdown_dataloader_workers(loader)


def _few_shot_adaptation_loaders(target_adapt_dataset: Any, cfg: dict[str, Any], run: Mapping[str, Any], *, loader_kwargs: dict[str, Any]):
    from torch.utils.data import DataLoader, Subset

    budget = int(run.get("budget", 0))
    base_dataset = getattr(target_adapt_dataset, "dataset", None)
    base_indices = list(getattr(target_adapt_dataset, "indices", range(len(target_adapt_dataset))))
    csv_indices = list(getattr(target_adapt_dataset, "csv_indices", base_indices))
    if budget <= 0:
        return None, DataLoader(target_adapt_dataset, **loader_kwargs), {
            "requested_budget": 0,
            "actual_labeled_count": 0,
            "unlabeled_count": len(target_adapt_dataset),
            "labeled_samples": [],
        }
    records = _csv_records(getattr(base_dataset, "root_csv", None))
    adapt_records = [records[index] if index < len(records) else {} for index in csv_indices]
    future_beam_key = _first_numbered_key(adapt_records, "future_beam") or "beam"
    sampling = sample_few_shot_records(
        adapt_records,
        budget=budget,
        seed=int(run.get("seed", 0)),
        group_size=int(cfg.get("hist_beam", {}).get("group_size", cfg.get("model", {}).get("student", {}).get("group_size", 8))),
        label_key=future_beam_key,
        data_root=getattr(base_dataset, "data_root", None),
        num_classes=int(cfg.get("hist_beam", {}).get("num_classes", cfg.get("model", {}).get("student", {}).get("num_classes", 64))),
    )
    labeled_local = list(sampling.labeled_indices)
    unlabeled_local = list(sampling.unlabeled_indices)
    labeled_loader = DataLoader(Subset(target_adapt_dataset, labeled_local), **loader_kwargs) if labeled_local else None
    unlabeled_loader = DataLoader(Subset(target_adapt_dataset, unlabeled_local), **loader_kwargs) if unlabeled_local else None
    return labeled_loader, unlabeled_loader, sampling.manifest


def _load_checkpoint_state(model: Any, checkpoint_path: str | Path, *, device: Any, strict: bool) -> None:
    import torch

    payload = torch.load(Path(checkpoint_path), map_location=device)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=strict)


def _preflight_csv_resources(
    *,
    scene: int,
    csv_path: Path,
    data_root: Path,
    enabled_modalities: tuple[str, ...],
    cfg: dict[str, Any],
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    rows = _csv_records(csv_path)
    header = list(rows[0].keys()) if rows else _csv_header(csv_path)
    required = _required_column_prefixes(enabled_modalities)
    seq_len = int(cfg.get("data", {}).get("dataset", {}).get("seq_len", 1))
    num_pred = int(cfg.get("data", {}).get("dataset", {}).get("num_pred", 1))
    minimum_by_prefix = {prefix: seq_len for prefix in required}
    minimum_by_prefix["future_beam"] = num_pred
    for prefix in required:
        cols = _numbered_columns(header, prefix)
        minimum = minimum_by_prefix[prefix]
        if len(cols) < minimum:
            errors.append(
                _preflight_error(
                    scene,
                    "csv_columns",
                    str(csv_path),
                    f"{csv_path} contains {len(cols)} {prefix} columns; expected at least {minimum}.",
                    runs,
                )
            )
            continue
        for row_index, row in enumerate(rows):
            for col in cols[:minimum]:
                value = str(row.get(col, "")).strip()
                if not value or value == "-99":
                    continue
                path = _resolve_resource_path(data_root, value)
                if path is not None and not path.exists():
                    errors.append(
                        _preflight_error(
                            scene,
                            f"{prefix}_resource",
                            str(path),
                            f"Scene {scene} enabled resource '{prefix}' referenced by {csv_path}:{row_index + 2} is missing.",
                            runs,
                        )
                    )
                    break
                if prefix == "radar":
                    doppler_path = _resolve_resource_path(data_root, str(value).replace("_RA", "_DA"))
                    if doppler_path is not None and not doppler_path.exists():
                        errors.append(
                            _preflight_error(
                                scene,
                                "radar_doppler_resource",
                                str(doppler_path),
                                f"Scene {scene} radar Doppler resource derived from {csv_path}:{row_index + 2} is missing.",
                                runs,
                            )
                        )
                        break
            if errors and errors[-1].get("resource_type") == f"{prefix}_resource":
                break
            if errors and errors[-1].get("resource_type") == "radar_doppler_resource":
                break
    return errors


def _required_column_prefixes(enabled_modalities: tuple[str, ...]) -> list[str]:
    prefixes = ["beam", "future_beam"]
    if "image" in enabled_modalities:
        prefixes.append("camera")
    if "radar" in enabled_modalities:
        prefixes.append("radar")
    if "gps" in enabled_modalities:
        prefixes.extend(["gps", "bs_gps"])
    if "lidar" in enabled_modalities:
        prefixes.append("lidar")
    if "mmwave" in enabled_modalities:
        prefixes.append("mmwave")
    if "csi" in enabled_modalities:
        prefixes.append("csi")
    return prefixes


def _cfg_for_scene(cfg: dict[str, Any], scene: int) -> dict[str, Any]:
    scene_cfg = deepcopy(cfg)
    dataset_cfg = scene_cfg.setdefault("data", {}).setdefault("dataset", {})
    normalize_deepsense_dataset_config(dataset_cfg)
    retarget_deepsense_dataset_config(dataset_cfg, scene)
    loso_cfg = scene_cfg.get("loso", {}) if isinstance(scene_cfg.get("loso"), dict) else {}
    roots = loso_cfg.get("scene_data_roots") if isinstance(loso_cfg.get("scene_data_roots"), dict) else {}
    root = roots.get(str(scene), roots.get(scene)) if isinstance(roots, dict) else None
    if root:
        dataset_cfg["data_root"] = str(root)
    csv_names = loso_cfg.get("scene_csv_names") if isinstance(loso_cfg.get("scene_csv_names"), dict) else {}
    scene_csv = csv_names.get(str(scene), csv_names.get(scene)) if isinstance(csv_names, dict) else None
    if isinstance(scene_csv, dict):
        for key in ("train_csv_name", "test_csv_name", "val_csv_name"):
            if scene_csv.get(key):
                dataset_cfg[key] = scene_csv[key]
    return scene_cfg


def _stage_cfg(
    cfg: dict[str, Any],
    run: Mapping[str, Any],
    *,
    variant: str,
    stage_name: str,
    stage_dir: Path,
) -> dict[str, Any]:
    stage_cfg = deepcopy(cfg)
    stage_cfg.setdefault("experiment", {})["seed"] = int(run.get("seed", 0))
    stage_cfg["experiment"]["name"] = f"{cfg.get('experiment', {}).get('name', 'hist_beam_loso')}_{stage_name}"
    model_cfg = stage_cfg.setdefault("model", {})
    model_cfg["modalities"] = list(_enabled_modalities({"enabled_modalities": model_cfg.get("modalities")}, stage_cfg))
    for key in ("student", "teacher"):
        role = model_cfg.get(key)
        if isinstance(role, dict):
            role["variant"] = variant
            role["modalities"] = list(model_cfg["modalities"])
    stage_cfg.setdefault("hist_beam", {})["variant"] = variant
    stage_cfg.setdefault("output", {})["dir"] = str(stage_dir)
    stage_cfg["output"]["run_name"] = stage_name
    stage_cfg["output"]["group_by_scene"] = False
    stage_cfg["output"].setdefault("progress", {})["enabled"] = False
    if variant == "v0_flat":
        weights = stage_cfg.setdefault("hist_beam", {}).setdefault("loss_weights", {})
        weights.update({"hierarchical": 0.0, "flat": 1.0, "orthogonality": 0.0, "scene_confusion": 0.0, "scene_private": 0.0})
    return stage_cfg


def _enabled_modalities(plan: dict[str, Any], cfg: dict[str, Any]) -> tuple[str, ...]:
    if plan.get("enabled_modalities"):
        return tuple(str(item) for item in plan["enabled_modalities"])
    return resolve_enabled_modalities(cfg)


def _reuse_source_checkpoint(cfg: dict[str, Any]) -> bool:
    loso_cfg = cfg.get("loso", {}) if isinstance(cfg.get("loso"), dict) else {}
    return bool(loso_cfg.get("reuse_source_checkpoint", True))


def _source_variant_for(run: Mapping[str, Any]) -> str:
    variant = str(run.get("variant"))
    if variant in ADAPTATION_VARIANTS:
        return "v3_decoupled"
    return variant


def _source_cache_key(run: Mapping[str, Any], variant: str) -> str:
    sources = "-".join(str(item) for item in run.get("source_scenes", []))
    return f"{run.get('fold')}|target={run.get('target_scene')}|sources={sources}|variant={variant}|seed={run.get('seed')}"


def _adaptation_cache_key(run: Mapping[str, Any]) -> str:
    return f"{run.get('fold')}|{run.get('variant')}|budget={run.get('budget')}|seed={run.get('seed')}"


def _matrix_summary(plan: dict[str, Any]) -> dict[str, Any]:
    runs = list(plan.get("runs", []))
    return {
        "target_scenes": sorted({int(run["target_scene"]) for run in runs if run.get("target_scene") is not None}),
        "variants": sorted({str(run["variant"]) for run in runs if run.get("variant") is not None}),
        "budgets": sorted({int(run["budget"]) for run in runs if run.get("budget") is not None}),
        "seeds": sorted({int(run["seed"]) for run in runs if run.get("seed") is not None}),
        "run_count": len(runs),
    }


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
    return {
        "fold": run.get("fold"),
        "target_scene": run.get("target_scene"),
        "source_scenes": list(run.get("source_scenes", [])),
        "variant": run.get("variant"),
        "budget": run.get("budget"),
        "seed": run.get("seed"),
    }


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


def _summary_row(record: dict[str, Any]) -> dict[str, Any]:
    source_metrics = record.get("metrics", {}).get("source_only_target_test_eval", {})
    adapted_metrics = record.get("metrics", {}).get("adapted_target_test_eval", {})
    adaptation_metrics = record.get("metrics", {}).get("target_adaptation", {})
    primary_metrics = adapted_metrics if adapted_metrics else source_metrics
    row = {
        "run_id": record.get("run_id"),
        "run_status": record.get("status"),
        "fold": record.get("fold"),
        "target_scene": record.get("target_scene"),
        "source_scenes": record.get("source_scenes"),
        "variant": record.get("variant"),
        "budget": record.get("budget"),
        "seed": record.get("seed"),
        "stage_status": {stage["name"]: stage["status"] for stage in record.get("stages", [])},
        "failure_reason": record.get("failure_reason"),
        "metrics_path": _artifact(record, "adapted_target_test_eval.metrics_path") or _artifact(record, "source_only_target_test_eval.metrics_path"),
        "predictions_path": _artifact(record, "adapted_target_test_eval.predictions_path") or _artifact(record, "source_only_target_test_eval.predictions_path"),
        "source_checkpoint_path": _artifact(record, "source_train.source_checkpoint_path") or _artifact(record, "source_checkpoint_path"),
        "adaptation_checkpoint_path": _artifact(record, "target_adaptation.adaptation_checkpoint_path"),
        "source_prototype_path": _artifact(record, "target_adaptation.source_prototype_path") or _artifact(record, "source_train.source_prototype_path"),
        "top1": _metric(primary_metrics, "top1"),
        "top3": _metric(primary_metrics, "top3"),
        "top5": _metric(primary_metrics, "top5"),
        "coarse_accuracy": primary_metrics.get("coarse_accuracy"),
        "fine_accuracy": primary_metrics.get("fine_offset_accuracy"),
        "trainable_params": adaptation_metrics.get("trainable_params"),
        "total_params": adaptation_metrics.get("total_params"),
        "trainable_ratio": adaptation_metrics.get("trainable_ratio"),
        "adaptation_time_seconds": adaptation_metrics.get("adaptation_time_seconds"),
        "adaptation_time_per_epoch": adaptation_metrics.get("adaptation_time_per_epoch"),
        "prototype_coverage": adaptation_metrics.get("prototype_coverage"),
        "prototype_coverage_unavailable_reason": adaptation_metrics.get("prototype_coverage_unavailable_reason"),
    }
    if record.get("status") == "failed":
        row["missing_reason"] = record.get("failure_reason")
    return row


def _artifact(record: dict[str, Any], key: str) -> Any:
    return record.get("artifacts", {}).get(key)


def _metric(metrics: Mapping[str, Any], name: str) -> float | None:
    if not metrics:
        return None
    if name in metrics and isinstance(metrics[name], (int, float)):
        return float(metrics[name])
    mapping = {"top1": "val_top1_avg", "top3": "val_top3_avg", "top5": "val_top5_avg"}
    mapped = mapping.get(name)
    if mapped and isinstance(metrics.get(mapped), (int, float)):
        return float(metrics[mapped])
    topk = metrics.get("topk")
    if isinstance(topk, dict):
        k = name.removeprefix("top")
        values = topk.get(k) or topk.get(int(k)) if k.isdigit() else None
        if isinstance(values, list) and values:
            numeric = [float(value) for value in values if isinstance(value, (int, float))]
            return sum(numeric) / len(numeric) if numeric else None
    return None


def _compare_adapter_to_source(
    *,
    target_scene: Any,
    budget: Any,
    seed: Any,
    variant: str,
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "comparison": "adapter_vs_source_only",
        "target_scene": target_scene,
        "budget": budget,
        "seed": seed,
        "baseline_variant": "v3_decoupled",
        "candidate_variant": variant,
    }
    missing = _missing_comparison_inputs({"v3_decoupled": baseline, variant: candidate})
    if missing:
        return {**base, "status": "inconclusive", "missing": missing}
    deltas = _accuracy_deltas(candidate, baseline)
    return {
        **base,
        "status": "complete",
        "accuracy_deltas": deltas,
        "efficiency": _efficiency_summary(candidate),
        "candidate_better_than_source_only": _is_better(deltas),
    }


def _compare_proto_to_full(
    *,
    target_scene: Any,
    budget: Any,
    seed: Any,
    proto: dict[str, Any] | None,
    full: dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "comparison": "adapter_proto_vs_full_finetune",
        "target_scene": target_scene,
        "budget": budget,
        "seed": seed,
        "candidate_variant": "v5_adapter_proto",
        "baseline_variant": "v6_full_finetune",
    }
    missing = _missing_comparison_inputs({"v5_adapter_proto": proto, "v6_full_finetune": full})
    if missing:
        return {**base, "status": "inconclusive", "missing": missing}
    deltas = _accuracy_deltas(proto, full)
    trainable_delta = _numeric_delta(proto, full, "trainable_ratio")
    time_delta = _numeric_delta(proto, full, "adaptation_time_seconds")
    return {
        **base,
        "status": "complete",
        "accuracy_deltas": deltas,
        "trainable_ratio_delta": trainable_delta,
        "adaptation_time_seconds_delta": time_delta,
        "adapter_proto_better_than_full_finetune": _is_better(deltas)
        and (trainable_delta is None or trainable_delta <= 0)
        and (time_delta is None or time_delta <= 0),
    }


def _missing_comparison_inputs(items: Mapping[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
    missing = []
    for variant, row in items.items():
        if row is None:
            missing.append({"variant": variant, "reason": "missing_run"})
            continue
        if row.get("run_status") != "completed":
            missing.append({"variant": variant, "reason": row.get("failure_reason") or "run_not_completed"})
            continue
        if all(row.get(metric) is None for metric in ("top1", "top3", "top5", "coarse_accuracy", "fine_accuracy")):
            missing.append({"variant": variant, "reason": "metrics_missing"})
    return missing


def _accuracy_deltas(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        metric: _numeric_delta(candidate, baseline, metric)
        for metric in ("top1", "top3", "top5", "coarse_accuracy", "fine_accuracy")
    }


def _numeric_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any], metric: str) -> float | None:
    lhs = candidate.get(metric)
    rhs = baseline.get(metric)
    if not isinstance(lhs, (int, float)) or not isinstance(rhs, (int, float)):
        return None
    return float(lhs) - float(rhs)


def _is_better(deltas: Mapping[str, float | None]) -> bool:
    available = [value for value in deltas.values() if value is not None]
    return bool(available) and all(value >= 0 for value in available) and any(value > 0 for value in available)


def _efficiency_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trainable_ratio": row.get("trainable_ratio"),
        "adaptation_time_seconds": row.get("adaptation_time_seconds"),
        "adaptation_time_per_epoch": row.get("adaptation_time_per_epoch"),
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


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


def _append_execution_event(output_dir: str | Path, event: str, payload: Mapping[str, Any]) -> Path:
    path = Path(output_dir) / EXECUTION_PROGRESS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "timestamp": _utc_now(), **dict(payload)}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def _append_stage_progress(stage_dir: str | Path, stage: str, payload: Mapping[str, Any]) -> Path:
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    record = {"stage": stage, "timestamp": _utc_now(), **dict(payload)}
    path = directory / "progress.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    _write_json(directory / "progress_latest.json", record)
    return path


def _preflight_error(scene: Any, resource_type: str, path: str | None, message: str, runs: list[dict[str, Any]] | None) -> dict[str, Any]:
    return {
        "scene": scene,
        "resource_type": resource_type,
        "path": path,
        "message": message,
        "runs": [_run_identity(run) for run in (runs or [])],
    }


def _runs_for_scene(runs: list[dict[str, Any]], scene: int) -> list[dict[str, Any]]:
    return [
        run
        for run in runs
        if int(run.get("target_scene", -1)) == int(scene)
        or int(scene) in {int(item) for item in run.get("source_scenes", [])}
    ]


def _resolve_csv_path(root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return root / path


def _resolve_resource_path(root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return joined_resource(root, str(value)).expanduser()


def _csv_records(path: Any) -> list[dict[str, Any]]:
    if path is None:
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        return next(reader, [])


def _numbered_columns(columns: list[str], prefix: str) -> list[str]:
    selected = []
    for col in columns:
        if not col.startswith(prefix):
            continue
        suffix = col[len(prefix) :]
        if suffix.isdigit():
            selected.append(col)
    return sorted(selected, key=lambda item: int(item[len(prefix) :]))


def _first_numbered_key(records: list[dict[str, Any]], prefix: str) -> str | None:
    if not records:
        return None
    columns = _numbered_columns(list(records[0].keys()), prefix)
    return columns[0] if columns else None


def _duration_seconds(start: str, end: str) -> float:
    start_dt = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_dt = dt.datetime.fromisoformat(end.replace("Z", "+00:00"))
    return float((end_dt - start_dt).total_seconds())


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


__all__ = [
    "ADAPTATION_VARIANTS",
    "DEFAULT_QUICK_BUDGETS",
    "DEFAULT_QUICK_SEEDS",
    "DEFAULT_QUICK_TARGET_SCENES",
    "DEFAULT_QUICK_VARIANTS",
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
