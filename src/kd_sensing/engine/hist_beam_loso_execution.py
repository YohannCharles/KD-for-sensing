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
from kd_sensing.engine.hist_beam_loso_matrix import matrix_summary
from kd_sensing.engine.hist_beam_loso_preflight import ensure_mmw_radar_csv_for_preflight, preflight_error
from kd_sensing.engine.hist_beam_loso_stages import StageRunCallbacks, execute_loso_stage_runs
from kd_sensing.engine.hist_beam_loso_summary import (
    conclusion_source_artifacts,
    excluded_run_summary,
    prototype_is_no_op,
    prototype_is_required,
    reason_histogram,
    row_eligibility,
    unique_reasons,
)
from kd_sensing.engine.modality_resolution import (
    SENSOR_ASSISTED_DISALLOWED_MODALITIES,
    SENSOR_ASSISTED_PROFILE,
    sensor_assisted_profile_enabled,
    resolve_enabled_modalities,
)
from kd_sensing.modalities import normalize_modalities
from kd_sensing.utils.paths import resolve_path
from kd_sensing.utils.seed import set_seed


EXECUTION_STATUSES = ("completed", "failed", "partial_failed")
SOURCE_ONLY_VARIANTS = {"v0_flat", "v1_hierarchical", "v2_shared_private", "v3_decoupled"}
ADAPTATION_VARIANTS = {
    "v4_adapter",
    "v5_adapter_proto",
    "v6_radio_proto",
    "adapter_radio_proto",
    "v8_path_proto",
    "adapter_path_proto",
    "v6_full_finetune",
}
SUPPORTED_VARIANTS = SOURCE_ONLY_VARIANTS | ADAPTATION_VARIANTS
DEFAULT_QUICK_VARIANTS = ["v0_flat", "v3_decoupled", "v4_adapter", "v5_adapter_proto", "v6_radio_proto", "v8_path_proto", "v6_full_finetune"]
SENSOR_ASSISTED_QUICK_VARIANTS = [
    "v3_decoupled",
    "v4_adapter",
    "v6_radio_proto",
    "v8_path_proto",
    "adapter_path_proto",
    "v6_full_finetune",
]
SENSOR_ASSISTED_QUICK_BUDGETS = [10]
SENSOR_ASSISTED_QUICK_SEEDS = [0, 1]
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


def run_loso_execute_preflight(plan: dict[str, Any], cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    from kd_sensing.engine.run_metadata import cache_run_metadata
    from kd_sensing.engine.runtime import configure_torch_runtime_threads

    out_dir = Path(output_dir)
    errors: list[dict[str, Any]] = []
    checked_paths: list[dict[str, Any]] = []
    matrix = _matrix_summary(plan)
    try:
        cpu_threads = {
            "configured": _cpu_thread_config(cfg),
            "applied": configure_torch_runtime_threads(cfg),
        }
    except Exception as exc:  # noqa: BLE001 - preflight should report thread config errors.
        cpu_threads = {"configured": _cpu_thread_config(cfg), "applied": {}, "error": f"{type(exc).__name__}: {exc}"}
        errors.append(_preflight_error("runtime", "cpu_threads", None, str(cpu_threads["error"]), None))
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
            scene
            for run in runs
            for scene in [run.get("target_scene"), *list(run.get("source_scenes", []))]
            if scene is not None
        },
        key=str,
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
            if str(dataset_cfg.get("type", "deepsense6g")).strip().lower() == "mmw" and "radar" in enabled_modalities:
                try:
                    csv_path = _ensure_mmw_radar_csv_for_preflight(data_root, csv_path, str(dataset_cfg.get("scene", scene)))
                except Exception as exc:  # noqa: BLE001 - report before training starts.
                    errors.append(
                        _preflight_error(
                            scene,
                            "radar_derived_csv",
                            str(csv_path),
                            f"Could not materialize MMW radar columns before training: {exc}",
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
        "modality_profile": _modality_profile_metadata(plan, cfg),
        "excluded_sensitive_fields": list(_excluded_sensitive_fields(cfg)),
        "cache": cache_run_metadata(cfg),
        "dataloader": _dataloader_preflight_metadata(cfg),
        "cpu_threads": cpu_threads,
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
            prototype_status = _prototype_decision(run, context.cfg, source_variant=variant)
            if prototype_status["generate"] and cache_key not in context.state["source_prototypes"]:
                generated = self._generate_missing_source_prototype(run, context, variant=variant, checkpoint=Path(cached["artifacts"]["source_checkpoint_path"]))
                cached_artifacts = dict(cached.get("artifacts", {}))
                cached_artifacts["source_prototype_path"] = generated.get("artifacts", {}).get("source_prototype_path")
                cached_metrics = dict(cached.get("metrics", {}))
                cached_metrics.update(generated.get("metrics", {}))
                cached = {**cached, "artifacts": cached_artifacts, "metrics": cached_metrics}
                context.state["source_checkpoints"][cache_key] = cached
            return {
                "status": "completed",
                "message": "Reused source checkpoint from an earlier run in this execution.",
                "checkpoint_reuse": {"enabled": True, "reused": True, "cache_key": cache_key},
                "artifacts": dict(cached.get("artifacts", {})),
                "metrics": cached.get("metrics", {}),
            }

        import torch

        from kd_sensing.engine.data_factory import shutdown_dataloader_workers
        from kd_sensing.engine.batch import prepare_radio_semantic_labels
        from kd_sensing.engine.hist_beam_losses import compute_hist_beam_loss, hist_beam_enabled
        from kd_sensing.engine.hist_beam_prototypes import generate_source_prototypes, prototype_coverage_from_counts
        from kd_sensing.engine.loso_data import build_loso_source_train_loader
        from kd_sensing.engine.optim import build_device, build_model, build_optimizer, build_task_criterion
        from kd_sensing.engine.runtime import (
            amp_runtime_metadata,
            autocast_context,
            make_grad_scaler,
            resolve_amp_settings,
            run_model_step,
            transfer_non_blocking,
        )

        cfg = _stage_cfg(context.cfg, run, variant=variant, stage_name="source_train", stage_dir=context.stage_dir)
        set_seed(cfg.get("experiment", {}).get("seed", 0))
        device = build_device(cfg)
        loaders = build_loso_source_train_loader(cfg, dict(run))
        model = build_model(cfg["model"]["student"]).to(device)
        optimizer = build_optimizer(cfg, model)
        criterion = build_task_criterion(cfg)
        task = cfg.get("experiment", {}).get("task", "fusion")
        amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
        grad_scaler = make_grad_scaler(cfg, amp_enabled)
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
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(amp_enabled, device, amp_dtype):
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
                        output = {"logits": step.logits, **step.model_output.diagnostics}
                        if hist_beam_enabled(cfg, output):
                            hist_loss = compute_hist_beam_loss(
                                output,
                                step.labels,
                                cfg=cfg,
                                radio_semantic_labels=prepare_radio_semantic_labels(
                                    step.batch,
                                    num_pred=step.labels.shape[1],
                                    device=device,
                                    non_blocking=non_blocking,
                                ),
                                num_classes=num_classes,
                            )
                            loss = hist_loss.total
                        else:
                            loss = criterion(step.logits.reshape(-1, num_classes), step.labels.flatten())
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
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
            prototype_status = _prototype_decision(run, context.cfg, source_variant=variant)
            prototype_path = context.stage_dir / "source_prototypes.pt"
            prototype_coverage: dict[str, Any] | None = None
            prototype_duration = 0.0
            if prototype_status["generate"]:
                prototype_start = time.perf_counter()
                prototype_artifact = generate_source_prototypes(
                    model,
                    loaders["source_train"],
                    cfg,
                    device,
                    output_path=prototype_path,
                    metadata=_run_identity(run) | {"source_variant": variant},
                    progress_callback=lambda progress: _append_stage_progress(context.stage_dir, "source_prototype", progress),
                )
                prototype_duration = time.perf_counter() - prototype_start
                prototype_coverage = prototype_coverage_from_counts(prototype_artifact["counts"])
                prototype_status.update({"status": "generated", "path": str(prototype_path)})
                context.state["source_prototypes"][cache_key] = str(prototype_path)
            else:
                prototype_status.setdefault("status", "skipped")
            metrics = {
                "train_loss_last": losses[-1] if losses else None,
                "train_loss_mean": sum(losses) / len(losses) if losses else None,
                "epochs": epochs,
                "source_variant": variant,
                "source_training_duration_seconds": sum(
                    float(item.get("duration_seconds", 0.0) or 0.0)
                    for item in _read_stage_progress(context.stage_dir, phase="source_train")
                ),
                "prototype_generation_duration_seconds": prototype_duration,
                "prototype_status": prototype_status["status"],
                "prototype_skipped_reason": prototype_status.get("reason"),
                "prototype_coverage": prototype_coverage,
                "amp": amp_runtime_metadata(cfg, device),
                "throughput_config": _throughput_config_summary(cfg, prototype_strategy=prototype_status.get("strategy")),
            }
            metrics_path = context.stage_dir / "metrics.json"
            _write_json(metrics_path, metrics)
            artifacts = {
                "run_dir": str(context.stage_dir),
                "metrics_path": str(metrics_path),
                "source_checkpoint_path": str(checkpoint_path),
                "source_prototype_path": str(prototype_path) if prototype_status.get("path") else None,
                "progress_path": str(progress_path),
            }
            result = {
                "status": "completed",
                "artifacts": artifacts,
                "metrics": metrics,
                "checkpoint_reuse": {"enabled": _reuse_source_checkpoint(context.cfg), "reused": False, "cache_key": cache_key},
            }
            context.state["source_checkpoints"][cache_key] = result
            context.state.setdefault("source_normalization", {})[cache_key] = dict(loaders.get("normalization_kwargs", {}))
            return result
        finally:
            for key in ("source_train",):
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
        from kd_sensing.engine.loso_data import build_loso_target_stage_loader
        from kd_sensing.engine.optim import build_device, build_model, build_optimizer

        source_variant = _source_variant_for(run)
        source_checkpoint = self._source_checkpoint_for(run, context, variant=source_variant)
        cfg = _stage_cfg(context.cfg, run, variant=variant, stage_name="target_adaptation", stage_dir=context.stage_dir)
        set_seed(cfg.get("experiment", {}).get("seed", 0))
        device = build_device(cfg)
        loaders = build_loso_target_stage_loader(
            cfg,
            dict(run),
            stage="target_adapt",
            split_seed=int(run.get("seed", 0)),
            dataset_kwargs=self._source_normalization_for(run, context, variant=source_variant),
        )
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
            if variant in {"v5_adapter_proto", "v6_radio_proto", "adapter_radio_proto", "v8_path_proto", "adapter_path_proto"}:
                proto_path = self._source_prototype_for(run, context, variant=source_variant)
                if proto_path is not None and Path(proto_path).exists():
                    prototypes = load_source_prototypes(proto_path, map_location=device)
                    counts_key = (
                        "count_path"
                        if variant in {"v8_path_proto", "adapter_path_proto"} and "count_path" in prototypes
                        else "count_radio"
                        if variant in {"v6_radio_proto", "adapter_radio_proto"} and "count_radio" in prototypes
                        else "counts"
                    )
                    prototype_metadata = {
                        "source_prototype_path": str(proto_path),
                        **prototype_coverage_from_counts(prototypes[counts_key]),
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
                label_budget=int(run.get("budget", 0)),
                progress_callback=lambda progress: _append_stage_progress(context.stage_dir, "target_adaptation", progress),
            )
            adaptation_diagnostics = adaptation.pop("diagnostics", {})
            flattened_diagnostics = _flatten_adaptation_diagnostics(adaptation_diagnostics)
            params = trainable_parameter_summary(model).to_dict()
            metrics = {
                **strategy_metadata,
                **params,
                **adaptation,
                **flattened_diagnostics,
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
            adapt_log_path = context.stage_dir / "adapt_log.json"
            _write_json(metrics_path, metrics)
            _write_json(
                adapt_log_path,
                {
                    "proto_type": metrics.get("proto_type"),
                    "label_budget": int(run.get("budget", 0)),
                    "target_labeled_subset_available": bool(metrics.get("target_labeled_subset_available", False)),
                    "target_unlabeled_subset_available": bool(metrics.get("target_unlabeled_subset_available", False)),
                    "sensitive_field_policy": metrics.get("sensitive_field_policy", {}),
                    "main_conclusion_eligible": bool(metrics.get("main_conclusion_eligible", True)),
                    "eligibility_reasons": list(metrics.get("eligibility_reasons", [])),
                    "used_target_beam_for_training": bool(metrics.get("used_target_beam_for_training", metrics.get("used_target_labels", False))),
                    "used_target_beam_power_for_training": bool(metrics.get("used_target_beam_power_for_training", False)),
                    "used_target_csi_for_training": bool(metrics.get("used_target_csi_for_training", False)),
                    "used_target_path_params_for_training": bool(metrics.get("used_target_path_params_for_training", False)),
                    "used_target_path_descriptor_for_training": bool(metrics.get("used_target_path_descriptor_for_training", False)),
                    "used_target_path_label_for_training": bool(metrics.get("used_target_path_label_for_training", False)),
                    "used_target_radio_label_for_training": bool(metrics.get("used_target_radio_label_for_training", False)),
                },
            )
            artifacts = {
                "run_dir": str(context.stage_dir),
                "metrics_path": str(metrics_path),
                "adapt_log_path": str(adapt_log_path),
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
        source_variant = _source_variant_for(run)
        proto_path = self._source_prototype_for(run, context, variant=source_variant)
        if proto_path:
            cfg.setdefault("hist_beam", {}).setdefault("prototype", {})["source_prototype_path"] = str(proto_path)
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

    @staticmethod
    def _source_normalization_for(run: Mapping[str, Any], context: StageExecutionContext, *, variant: str) -> dict[str, Any]:
        return dict(context.state.get("source_normalization", {}).get(_source_cache_key(run, variant), {}))

    def _generate_missing_source_prototype(
        self,
        run: Mapping[str, Any],
        context: StageExecutionContext,
        *,
        variant: str,
        checkpoint: Path,
    ) -> dict[str, Any]:
        import torch

        from kd_sensing.engine.data_factory import shutdown_dataloader_workers
        from kd_sensing.engine.hist_beam_prototypes import generate_source_prototypes, prototype_coverage_from_counts
        from kd_sensing.engine.loso_data import build_loso_source_train_loader
        from kd_sensing.engine.optim import build_device, build_model

        cfg = _stage_cfg(context.cfg, run, variant=variant, stage_name="source_prototype", stage_dir=context.stage_dir)
        device = build_device(cfg)
        loaders = build_loso_source_train_loader(cfg, dict(run))
        try:
            model = build_model(cfg["model"]["student"]).to(device)
            _load_checkpoint_state(model, checkpoint, device=device, strict=False)
            prototype_path = context.stage_dir / "source_prototypes.pt"
            start = time.perf_counter()
            artifact = generate_source_prototypes(
                model,
                loaders["source_train"],
                cfg,
                device,
                output_path=prototype_path,
                metadata=_run_identity(run) | {"source_variant": variant, "generated_after_checkpoint_reuse": True},
                progress_callback=lambda progress: _append_stage_progress(context.stage_dir, "source_prototype", progress),
            )
            duration = time.perf_counter() - start
            context.state["source_prototypes"][_source_cache_key(run, variant)] = str(prototype_path)
            context.state.setdefault("source_normalization", {})[_source_cache_key(run, variant)] = dict(
                loaders.get("normalization_kwargs", {})
            )
            metrics = {
                "prototype_status": "generated",
                "prototype_generation_duration_seconds": duration,
                "prototype_coverage": prototype_coverage_from_counts(artifact["counts"]),
            }
            return {
                "status": "completed",
                "artifacts": {"source_prototype_path": str(prototype_path)},
                "metrics": metrics,
            }
        finally:
            loader = loaders.get("source_train")
            if loader is not None:
                shutdown_dataloader_workers(loader)


def write_loso_execute_summary(output_dir: str | Path, run_records: list[dict[str, Any]], *, status: str) -> dict[str, str]:
    out_dir = Path(output_dir)
    rows = [_summary_row(record) for record in run_records]
    completed_count = sum(1 for record in run_records if record.get("status") == "completed")
    failed_count = sum(1 for record in run_records if record.get("status") == "failed")
    missing_count = sum(1 for record in run_records if record.get("status") == "missing")
    eligible_count = sum(1 for row in rows if bool(row.get("main_conclusion_eligible", True)))
    excluded_count = len(rows) - eligible_count
    payload = {
        "status": status,
        "generated_at": _utc_now(),
        "run_count": len(run_records),
        "completed_count": completed_count,
        "failed_count": failed_count,
        "missing_count": missing_count,
        "eligible_run_count": eligible_count,
        "excluded_run_count": excluded_count,
        "exclusion_reason_histogram": _reason_histogram(row.get("eligibility_reasons", []) for row in rows),
        "claim_scope": _claim_scope_from_rows(rows),
        "cross_scene_claim_allowed": all(bool(row.get("cross_scene_claim_allowed", True)) for row in rows) if rows else False,
        "runs": rows,
    }
    json_path = out_dir / "loso_summary.json"
    csv_path = out_dir / "loso_summary.csv"
    _write_json(json_path, payload)
    _write_summary_csv(csv_path, rows)
    return {"json": str(json_path), "csv": str(csv_path)}


def _claim_scope_from_rows(rows: list[dict[str, Any]]) -> str:
    scopes = sorted({str(row.get("claim_scope", "cross_scene")) for row in rows})
    if not scopes:
        return "unavailable"
    if len(scopes) == 1:
        return scopes[0]
    return "mixed"


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
        for variant in ("v4_adapter", "v5_adapter_proto", "v8_path_proto"):
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
        comparisons.append(
            _compare_coarse_to_radio(
                target_scene=target_scene,
                budget=budget,
                seed=seed,
                coarse=by_key.get((target_scene, budget, seed, "v5_adapter_proto")),
                radio=by_key.get((target_scene, budget, seed, "v6_radio_proto")),
            )
        )
        comparisons.append(
            _compare_radio_condition(
                target_scene=target_scene,
                budget=budget,
                seed=seed,
                off=by_key.get((target_scene, budget, seed, "adapter_radio_proto")),
                on=by_key.get((target_scene, budget, seed, "v6_radio_proto")),
            )
        )
        comparisons.append(
            _compare_radio_to_path(
                target_scene=target_scene,
                budget=budget,
                seed=seed,
                radio=by_key.get((target_scene, budget, seed, "v6_radio_proto")),
                path=by_key.get((target_scene, budget, seed, "v8_path_proto")),
            )
        )
        comparisons.append(
            _compare_path_to_full(
                target_scene=target_scene,
                budget=budget,
                seed=seed,
                path=by_key.get((target_scene, budget, seed, "v8_path_proto")),
                full=by_key.get((target_scene, budget, seed, "v6_full_finetune")),
            )
        )
        comparisons.append(
            _compare_path_condition(
                target_scene=target_scene,
                budget=budget,
                seed=seed,
                off=by_key.get((target_scene, budget, seed, "adapter_path_proto")),
                on=by_key.get((target_scene, budget, seed, "v8_path_proto")),
            )
        )
    excluded_runs = [_excluded_run_summary(row) for row in rows if not bool(row.get("main_conclusion_eligible", True))]
    inconclusive_count = sum(1 for item in comparisons if item.get("status") != "complete")
    payload = {
        "generated_at": _utc_now(),
        "summary_path": str(summary_path),
        "source_paths": {
            "summary_path": str(summary_path),
            "run_artifacts": _conclusion_source_artifacts(rows),
        },
        "eligible_run_count": len(rows) - len(excluded_runs),
        "excluded_run_count": len(excluded_runs),
        "inconclusive_comparison_count": inconclusive_count,
        "exclusion_reason_histogram": _reason_histogram(row.get("eligibility_reasons", []) for row in rows),
        "excluded_runs": excluded_runs,
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
    from kd_sensing.engine.loso_data import build_loso_target_stage_loader
    from kd_sensing.engine.optim import build_device, build_model, build_task_criterion
    from kd_sensing.evaluation.hist_beam_outputs import write_hist_beam_predictions

    device = build_device(cfg)
    executor = DefaultHistBeamLosoStageExecutor()
    loaders = build_loso_target_stage_loader(
        cfg,
        dict(run),
        stage="target_test",
        split_seed=int(run.get("seed", 0)),
        dataset_kwargs=executor._source_normalization_for(run, context, variant=_source_variant_for(run)),
    )
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
            radio_logits=result.radio_logits,
            radio_labels=result.radio_labels,
            path_logits=result.path_logits,
            path_labels=result.path_labels,
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
        radio_builder_config=_radio_semantic_config_for_sampling(cfg),
    )
    labeled_local = list(sampling.labeled_indices)
    unlabeled_local = list(sampling.unlabeled_indices)
    labeled_loader = DataLoader(Subset(target_adapt_dataset, labeled_local), **loader_kwargs) if labeled_local else None
    unlabeled_loader = DataLoader(Subset(target_adapt_dataset, unlabeled_local), **loader_kwargs) if unlabeled_local else None
    return labeled_loader, unlabeled_loader, sampling.manifest


def _radio_semantic_config_for_sampling(cfg: dict[str, Any]) -> dict[str, Any] | None:
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    radio_cfg = hist_cfg.get("radio_semantic") if isinstance(hist_cfg.get("radio_semantic"), dict) else None
    dataset_cfg = cfg.get("data", {}).get("dataset", {}) if isinstance(cfg.get("data"), dict) else {}
    dataset_radio = dataset_cfg.get("radio_semantic") if isinstance(dataset_cfg.get("radio_semantic"), dict) else None
    selected = radio_cfg or dataset_radio
    if not isinstance(selected, dict) or selected.get("enabled") is False:
        return None
    return selected


def _load_checkpoint_state(model: Any, checkpoint_path: str | Path, *, device: Any, strict: bool) -> None:
    import torch

    payload = torch.load(Path(checkpoint_path), map_location=device)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=strict)


def _preflight_csv_resources(
    *,
    scene: Any,
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
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if str(dataset_cfg.get("type", "deepsense6g")).strip().lower() == "mmw":
        required = [prefix for prefix in required if prefix != "bs_gps"]
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


def _ensure_mmw_radar_csv_for_preflight(data_root: Path, csv_path: Path, scene: str) -> Path:
    return ensure_mmw_radar_csv_for_preflight(data_root, csv_path, scene)


def _dataloader_preflight_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    from kd_sensing.engine.data_factory import resolve_dataloader_split_config

    loader_cfg = cfg.get("data", {}).get("dataloader", {}) if isinstance(cfg.get("data"), dict) else {}
    return {
        "train": resolve_dataloader_split_config(loader_cfg, split="train"),
        "test": resolve_dataloader_split_config(loader_cfg, split="test"),
    }


def _cpu_thread_config(cfg: dict[str, Any]) -> dict[str, Any]:
    thread_cfg = cfg.get("training", {}).get("cpu_threads", {}) if isinstance(cfg.get("training"), dict) else {}
    return dict(thread_cfg) if isinstance(thread_cfg, dict) else {}


def _excluded_sensitive_fields(cfg: dict[str, Any]) -> tuple[str, ...]:
    if sensor_assisted_profile_enabled(cfg):
        return SENSOR_ASSISTED_DISALLOWED_MODALITIES
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    configured = hist_cfg.get("excluded_sensitive_fields")
    if configured:
        return tuple(str(item) for item in configured)
    return ()


def _modality_profile_metadata(plan: Mapping[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    enabled = list(_enabled_modalities(dict(plan), cfg))
    profile = _matrix_profile(cfg)
    return {
        "profile": profile,
        "enabled_modalities": enabled,
        "excluded_sensitive_fields": list(_excluded_sensitive_fields(cfg)),
        "sensor_assisted": bool(sensor_assisted_profile_enabled(cfg)),
    }


def _cfg_for_scene(cfg: dict[str, Any], scene: Any) -> dict[str, Any]:
    scene_cfg = deepcopy(cfg)
    dataset_cfg = scene_cfg.setdefault("data", {}).setdefault("dataset", {})
    if str(dataset_cfg.get("type", "deepsense6g")).strip().lower() == "mmw":
        dataset_cfg["scene"] = str(scene)
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
    hist_cfg = stage_cfg.setdefault("hist_beam", {})
    student_cfg = model_cfg.get("student") if isinstance(model_cfg.get("student"), dict) else {}
    if variant in {"v6_radio_proto", "adapter_radio_proto"}:
        radio_cfg = hist_cfg.setdefault("radio_semantic", {})
        radio_cfg.setdefault("enabled", True)
        radio_cfg.setdefault("mode", "peak_spread")
        radio_cfg.setdefault("num_spread_bins", 3)
        radio_cfg.setdefault("entropy_thresholds", [0.35, 0.65])
        hist_cfg["proto_type"] = "radio_semantic"
        hist_cfg.setdefault("prototype", {})["proto_type"] = "radio_semantic"
        weights = hist_cfg.setdefault("loss_weights", {})
        weights.setdefault("radio_semantic", 1.0)
        dataset_cfg = stage_cfg.setdefault("data", {}).setdefault("dataset", {})
        dataset_cfg.setdefault("radio_semantic", dict(radio_cfg))
        if isinstance(student_cfg, dict):
            student_cfg.setdefault("radio_semantic", dict(radio_cfg))
            student_cfg.setdefault("use_radio_head", True)
            student_cfg.setdefault("num_radio_classes", int(radio_cfg.get("num_radio_classes", 24)))
            student_cfg.setdefault("proto_type", "radio_semantic")
            student_cfg.setdefault("radio_tau", float(hist_cfg.get("radio_tau", 1.0)))
            if variant == "adapter_radio_proto":
                student_cfg.setdefault("use_radio_condition_in_beam_head", False)
            else:
                student_cfg.setdefault(
                    "use_radio_condition_in_beam_head",
                    bool(radio_cfg.get("use_radio_condition_in_beam_head", True)),
                )
    elif variant in {"v8_path_proto", "adapter_path_proto"}:
        path_cfg = hist_cfg.setdefault("path_semantic", {})
        path_cfg.setdefault("enabled", True)
        path_cfg.setdefault("mode", "kmeans_path_descriptor")
        path_cfg.setdefault("num_path_classes", 24)
        path_cfg.setdefault("fit_on_source_only", True)
        path_cfg.setdefault("fallback_if_missing", "radio_power")
        path_cfg.setdefault("use_path_regression", True)
        hist_cfg["proto_type"] = "path"
        hist_cfg.setdefault("prototype", {})["proto_type"] = "path"
        adapt_cfg = hist_cfg.setdefault("adaptation", {})
        adapt_cfg.setdefault("proto_type", "path")
        adapt_cfg.setdefault("proto_tau", 0.1)
        adapt_cfg.setdefault("confidence_threshold", 0.75)
        adapt_cfg.setdefault("proto_warmup_epochs", 5)
        adapt_cfg.setdefault("target_proto_momentum", 0.9)
        adapt_cfg.setdefault("allow_labeled_target_path_supervision", False)
        weights = hist_cfg.setdefault("loss_weights", {})
        weights.setdefault("lambda_path", 0.3)
        weights.setdefault("lambda_path_reg", 0.05)
        dataset_cfg = stage_cfg.setdefault("data", {}).setdefault("dataset", {})
        dataset_cfg.setdefault("path_semantic", dict(path_cfg))
        if isinstance(student_cfg, dict):
            student_cfg.setdefault("path_semantic", dict(path_cfg))
            student_cfg.setdefault("use_path_head", True)
            student_cfg.setdefault("use_path_condition_in_beam_head", variant != "adapter_path_proto")
            student_cfg.setdefault("path_embed_dim", 32)
            student_cfg.setdefault("num_path_classes", int(path_cfg.get("num_path_classes", 24)))
            student_cfg.setdefault("proto_type", "path")
    elif variant in {"v5_adapter_proto", "adapter_proto"}:
        hist_cfg["proto_type"] = "coarse"
        hist_cfg.setdefault("prototype", {})["proto_type"] = "coarse"
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
    if not sensor_assisted_profile_enabled(cfg):
        model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
        for context, raw in (
            ("model.modalities", model_cfg.get("modalities")),
            ("model.student.modalities", model_cfg.get("student", {}).get("modalities") if isinstance(model_cfg.get("student"), dict) else None),
            ("model.teacher.modalities", model_cfg.get("teacher", {}).get("modalities") if isinstance(model_cfg.get("teacher"), dict) else None),
        ):
            if raw:
                return tuple(normalize_modalities(raw, context=context))
    return resolve_enabled_modalities(cfg)


def _reuse_source_checkpoint(cfg: dict[str, Any]) -> bool:
    loso_cfg = cfg.get("loso", {}) if isinstance(cfg.get("loso"), dict) else {}
    return bool(loso_cfg.get("reuse_source_checkpoint", True))


def _prototype_decision(run: Mapping[str, Any], cfg: dict[str, Any], *, source_variant: str) -> dict[str, Any]:
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    proto_cfg = hist_cfg.get("prototype", {}) if isinstance(hist_cfg.get("prototype"), dict) else {}
    strategy = str(proto_cfg.get("strategy", proto_cfg.get("generation", "auto"))).strip().lower()
    run_variant = str(run.get("variant"))
    requires = run_variant in {"v5_adapter_proto", "v6_radio_proto", "adapter_radio_proto", "v8_path_proto", "adapter_path_proto"}
    explicit_save = bool(proto_cfg.get("save_source_prototypes", False))
    if strategy in {"off", "skip", "none"}:
        return {"generate": False, "status": "skipped", "reason": "prototype_strategy_off", "strategy": strategy}
    if strategy in {"always", "force"} or explicit_save:
        return {"generate": True, "status": "pending", "reason": "prototype_strategy_forced", "strategy": strategy}
    if requires:
        return {"generate": True, "status": "pending", "reason": f"variant_requires_prototype:{run_variant}", "strategy": strategy}
    return {
        "generate": False,
        "status": "skipped",
        "reason": f"source_only_variant:{source_variant}",
        "strategy": strategy,
    }


def _throughput_config_summary(cfg: dict[str, Any], *, prototype_strategy: str | None) -> dict[str, Any]:
    loader_cfg = cfg.get("data", {}).get("dataloader", {}) if isinstance(cfg.get("data"), dict) else {}
    dataset_cfg = cfg.get("data", {}).get("dataset", {}) if isinstance(cfg.get("data"), dict) else {}
    cache_cfg = cfg.get("data", {}).get("cache", {}) if isinstance(cfg.get("data"), dict) else {}
    image_cache_cfg = cache_cfg.get("image", {}) if isinstance(cache_cfg.get("image", {}), dict) else {}
    lidar_cache_cfg = cache_cfg.get("lidar", {}) if isinstance(cache_cfg.get("lidar", {}), dict) else {}
    return {
        "batch_size": loader_cfg.get("batch_size", loader_cfg.get("train_batch_size")),
        "num_workers": loader_cfg.get("train_num_workers", loader_cfg.get("num_workers")),
        "persistent_workers": loader_cfg.get("train_persistent_workers", loader_cfg.get("persistent_workers")),
        "prefetch_factor": loader_cfg.get("train_prefetch_factor", loader_cfg.get("prefetch_factor")),
        "enabled_modalities": list(resolve_enabled_modalities(cfg)),
        "seq_len": dataset_cfg.get("seq_len"),
        "modality_profile": _matrix_profile(cfg),
        "image_cache_policy": image_cache_cfg.get("policy", cache_cfg.get("policy", "auto")),
        "lidar_cache_policy": lidar_cache_cfg.get("policy", cache_cfg.get("policy", "auto")),
        "lidar_cache_dir": dataset_cfg.get("lidar_cache_dir"),
        "cpu_threads": _cpu_thread_config(cfg),
        "prototype_strategy": prototype_strategy,
    }


def _source_variant_for(run: Mapping[str, Any]) -> str:
    variant = str(run.get("variant"))
    if variant in {"v6_radio_proto", "adapter_radio_proto", "v8_path_proto", "adapter_path_proto"}:
        return variant
    if variant in ADAPTATION_VARIANTS:
        return "v3_decoupled"
    return variant


def _source_cache_key(run: Mapping[str, Any], variant: str) -> str:
    sources = "-".join(str(item) for item in run.get("source_scenes", []))
    return f"{run.get('fold')}|target={run.get('target_scene')}|sources={sources}|variant={variant}|seed={run.get('seed')}"


def _adaptation_cache_key(run: Mapping[str, Any]) -> str:
    return f"{run.get('fold')}|{run.get('variant')}|budget={run.get('budget')}|seed={run.get('seed')}"


def _matrix_summary(plan: dict[str, Any]) -> dict[str, Any]:
    return matrix_summary(plan)


def _matrix_profile(cfg: dict[str, Any]) -> str | None:
    loso_cfg = cfg.get("loso", {}) if isinstance(cfg.get("loso"), dict) else {}
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    dataset_cfg = cfg.get("data", {}).get("dataset", {}) if isinstance(cfg.get("data"), dict) else {}
    for value in (
        loso_cfg.get("profile"),
        loso_cfg.get("matrix_profile"),
        hist_cfg.get("profile"),
        dataset_cfg.get("modality_profile"),
    ):
        if value not in (None, ""):
            return str(value)
    if sensor_assisted_profile_enabled(cfg):
        return SENSOR_ASSISTED_PROFILE
    return None


def _matrix_scene_value(scene: Any) -> Any:
    try:
        return int(scene)
    except (TypeError, ValueError):
        return str(scene)


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


def _summary_row(record: dict[str, Any]) -> dict[str, Any]:
    source_train_metrics = record.get("metrics", {}).get("source_train", {})
    source_metrics = record.get("metrics", {}).get("source_only_target_test_eval", {})
    adapted_metrics = record.get("metrics", {}).get("adapted_target_test_eval", {})
    adaptation_metrics = record.get("metrics", {}).get("target_adaptation", {})
    primary_metrics = adapted_metrics if adapted_metrics else source_metrics
    source_top1 = _metric(source_metrics, "top1")
    source_top3 = _metric(source_metrics, "top3")
    source_top5 = _metric(source_metrics, "top5")
    adapted_top1 = _metric(adapted_metrics, "top1")
    adapted_top3 = _metric(adapted_metrics, "top3")
    adapted_top5 = _metric(adapted_metrics, "top5")
    last_beam = _last_beam_summary(primary_metrics)
    cache_summary = _cache_summary(record, source_train_metrics)
    split_summary = _split_summary(record, source_metrics, adapted_metrics, primary_metrics)
    row = {
        "run_id": record.get("run_id"),
        "run_status": record.get("status"),
        "fold": record.get("fold"),
        "target_scene": record.get("target_scene"),
        "source_scenes": record.get("source_scenes"),
        "dataset_family": record.get("dataset_family") or record.get("scene_family") or "DeepSense6G",
        "condition": record.get("condition"),
        "town": record.get("town"),
        "profile": record.get("profile"),
        "modality_profile": record.get("modality_profile") or record.get("profile"),
        "matrix_scope": record.get("matrix_scope"),
        "quick_validation": record.get("quick_validation"),
        "enabled_modalities": list(record.get("enabled_modalities") or primary_metrics.get("enabled_modalities") or []),
        "excluded_sensitive_fields": record.get("excluded_sensitive_fields"),
        "cache_policy": cache_summary.get("cache_policy"),
        "lidar_cache_policy": cache_summary.get("lidar_cache_policy"),
        "lidar_cache_dir": cache_summary.get("lidar_cache_dir"),
        "num_workers": cache_summary.get("num_workers"),
        "cpu_threads": cache_summary.get("cpu_threads"),
        "claim_scope": record.get("claim_scope") or "cross_scene",
        "cross_scene_claim_allowed": True if record.get("cross_scene_claim_allowed") is None else record.get("cross_scene_claim_allowed"),
        "split_protocol": split_summary.get("split_protocol"),
        "split_strategy": split_summary.get("split_strategy"),
        "split_protocol_version": split_summary.get("split_protocol_version"),
        "split_metadata_path": split_summary.get("split_metadata_path"),
        "split_metadata_available": split_summary.get("split_metadata_available"),
        "strict_validation_eligible": split_summary.get("strict_validation_eligible"),
        "split_eligibility": split_summary.get("split_eligibility"),
        "split_eligibility_reasons": split_summary.get("eligibility_reasons"),
        "split_fix_hint": split_summary.get("fix_hint"),
        "split_seed": split_summary.get("split_seed"),
        "split_sequence_count": split_summary.get("split_sequence_count"),
        "split_num_samples": split_summary.get("split_num_samples"),
        "leakage_diagnostics": split_summary.get("leakage_diagnostics"),
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
        "source_top1": source_top1,
        "source_top3": source_top3,
        "source_top5": source_top5,
        "adapted_top1": adapted_top1,
        "adapted_top3": adapted_top3,
        "adapted_top5": adapted_top5,
        "adapted_source_top1_delta": _numeric_delta_from_values(adapted_top1, source_top1),
        "adapted_source_top3_delta": _numeric_delta_from_values(adapted_top3, source_top3),
        "adapted_source_top5_delta": _numeric_delta_from_values(adapted_top5, source_top5),
        "coarse_accuracy": primary_metrics.get("coarse_accuracy"),
        "fine_accuracy": primary_metrics.get("fine_offset_accuracy"),
        "radio_semantic_accuracy": primary_metrics.get("radio_semantic_accuracy"),
        "radio_semantic_coverage": primary_metrics.get("radio_semantic_coverage"),
        "radio_metrics_unavailable_reason": primary_metrics.get("radio_metrics_unavailable_reason"),
        "path_semantic_accuracy": primary_metrics.get("path_semantic_accuracy"),
        "path_semantic_coverage": primary_metrics.get("path_semantic_coverage"),
        "path_metrics_unavailable_reason": primary_metrics.get("path_metrics_unavailable_reason"),
        "path_descriptor_regression_mse": primary_metrics.get("path_descriptor_regression_mse"),
        "prototype_assignment_confidence": primary_metrics.get("prototype_assignment_confidence"),
        "prototype_coverage_per_class": primary_metrics.get("prototype_coverage_per_class"),
        "source_target_path_class_histogram": primary_metrics.get("source_target_path_class_histogram"),
        "normalized_received_power": primary_metrics.get("normalized_received_power"),
        "beam_power_loss_db": primary_metrics.get("beam_power_loss_db"),
        "source_normalized_received_power": source_metrics.get("normalized_received_power"),
        "adapted_normalized_received_power": adapted_metrics.get("normalized_received_power"),
        "adapted_source_normalized_received_power_delta": _numeric_delta_from_values(
            adapted_metrics.get("normalized_received_power"),
            source_metrics.get("normalized_received_power"),
        ),
        "source_beam_power_loss_db": source_metrics.get("beam_power_loss_db"),
        "adapted_beam_power_loss_db": adapted_metrics.get("beam_power_loss_db"),
        "adapted_source_beam_power_loss_db_delta": _numeric_delta_from_values(
            adapted_metrics.get("beam_power_loss_db"),
            source_metrics.get("beam_power_loss_db"),
        ),
        "negative_transfer": _negative_transfer(adapted_top1, source_top1),
        "negative_transfer_metric": "top1" if _negative_transfer(adapted_top1, source_top1) is not None else None,
        "last_beam_top1": last_beam.get("top1"),
        "last_beam_top3": last_beam.get("top3"),
        "last_beam_avg_top1": last_beam.get("avg_top1"),
        "last_beam_avg_top3": last_beam.get("avg_top3"),
        "last_beam_available": last_beam.get("available"),
        "last_beam_baseline_type": "diagnostic",
        "last_beam_comparable_baseline": bool(primary_metrics.get("last_beam_comparable_baseline", False)),
        "power_metrics_unavailable_reason": primary_metrics.get("power_metrics_unavailable_reason"),
        "trainable_params": adaptation_metrics.get("trainable_params"),
        "total_params": adaptation_metrics.get("total_params"),
        "trainable_ratio": adaptation_metrics.get("trainable_ratio"),
        "adaptation_time_seconds": adaptation_metrics.get("adaptation_time_seconds"),
        "adaptation_time_per_epoch": adaptation_metrics.get("adaptation_time_per_epoch"),
        "source_training_duration_seconds": source_train_metrics.get("source_training_duration_seconds"),
        "prototype_generation_duration_seconds": source_train_metrics.get("prototype_generation_duration_seconds"),
        "prototype_coverage": adaptation_metrics.get("prototype_coverage"),
        "prototype_coverage_unavailable_reason": adaptation_metrics.get("prototype_coverage_unavailable_reason"),
        "prototype_status": adaptation_metrics.get("prototype_status") or source_train_metrics.get("prototype_status"),
        "prototype_skipped_reason": source_train_metrics.get("prototype_skipped_reason"),
        "prototype_confidence_mean": adaptation_metrics.get("prototype_confidence_mean"),
        "prototype_used_sample_count": adaptation_metrics.get("prototype_used_sample_count"),
        "prototype_loss_mean": _first_present(adaptation_metrics, "prototype_loss_mean", "prototype_loss"),
        "proto_type": adaptation_metrics.get("proto_type"),
        "label_budget": adaptation_metrics.get("label_budget", record.get("budget")),
        "target_labeled_subset_available": _bool_or_false(adaptation_metrics.get("target_labeled_subset_available")),
        "target_unlabeled_subset_available": _bool_or_false(adaptation_metrics.get("target_unlabeled_subset_available")),
        "sensitive_field_policy": adaptation_metrics.get("sensitive_field_policy", {}),
        "used_target_labels": _bool_or_false(adaptation_metrics.get("used_target_labels")),
        "used_target_beam_for_training": _bool_or_false(adaptation_metrics.get("used_target_beam_for_training")),
        "used_target_beam_power_for_training": _bool_or_false(adaptation_metrics.get("used_target_beam_power_for_training")),
        "used_target_csi_for_training": _bool_or_false(adaptation_metrics.get("used_target_csi_for_training")),
        "used_target_path_params_for_training": _bool_or_false(adaptation_metrics.get("used_target_path_params_for_training")),
        "used_target_path_descriptor_for_training": _bool_or_false(adaptation_metrics.get("used_target_path_descriptor_for_training")),
        "used_target_path_label_for_training": _bool_or_false(adaptation_metrics.get("used_target_path_label_for_training")),
        "used_target_radio_label_for_training": _bool_or_false(adaptation_metrics.get("used_target_radio_label_for_training")),
        "radio_assignment_confidence_mean": adaptation_metrics.get("radio_assignment_confidence_mean"),
        "radio_assignment_used_sample_count": adaptation_metrics.get("radio_assignment_used_sample_count"),
        "path_assignment_confidence_mean": adaptation_metrics.get("path_assignment_confidence_mean"),
        "path_assignment_used_sample_count": adaptation_metrics.get("path_assignment_used_sample_count"),
        "target_private_initialized_count": adaptation_metrics.get("target_private_initialized_count"),
        "geometry_loss_coverage": primary_metrics.get("hist/geometry_consistency_coverage")
        or adaptation_metrics.get("geometry_consistency_coverage"),
    }
    row["method_family"] = _method_family(row)
    row["sensitive_field_usage"] = {
        key: row[key]
        for key in (
            "used_target_beam_for_training",
            "used_target_beam_power_for_training",
            "used_target_csi_for_training",
            "used_target_path_params_for_training",
            "used_target_path_descriptor_for_training",
            "used_target_path_label_for_training",
            "used_target_radio_label_for_training",
        )
    }
    eligibility = _row_eligibility(row, adaptation_metrics, primary_metrics)
    row["main_conclusion_eligible"] = eligibility["main_conclusion_eligible"]
    row["eligibility_reasons"] = eligibility["eligibility_reasons"]
    row["eligibility_source_artifacts"] = {
        "metrics_path": row.get("metrics_path"),
        "adapt_log_path": _artifact(record, "target_adaptation.adapt_log_path"),
        "run_metadata_path": _artifact(record, "run_metadata_path"),
    }
    if record.get("status") == "failed":
        row["missing_reason"] = record.get("failure_reason")
    return row


def _method_family(row: Mapping[str, Any]) -> str:
    variant = str(row.get("variant"))
    if variant in {"v6_full_finetune", "full_finetune"}:
        return "full_finetuning_baseline"
    if variant in {"v6_radio_proto", "adapter_radio_proto"} or row.get("proto_type") == "radio_semantic":
        return "radio_semantic_prototype"
    if variant in {"v8_path_proto", "adapter_path_proto"} or row.get("proto_type") == "path":
        return "path_physical_prototype"
    if variant in {"v5_adapter_proto", "adapter_proto"}:
        return "coarse_prototype_baseline"
    return "source_or_adapter_baseline"


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def _split_summary(
    record: Mapping[str, Any],
    source_metrics: Mapping[str, Any],
    adapted_metrics: Mapping[str, Any],
    primary_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = []
    for mapping in (primary_metrics, adapted_metrics, source_metrics):
        if isinstance(mapping, Mapping):
            setup = mapping.get("prediction_setup")
            if isinstance(setup, Mapping):
                candidates.append(setup)
            candidates.append(mapping)
    candidates.append(record)
    for key in ("prediction_setup", "split_metadata", "split_protocol"):
        value = record.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    merged: dict[str, Any] = {}
    for candidate in candidates:
        extracted = _extract_split_summary(candidate)
        for key, value in extracted.items():
            if key not in merged and value is not None:
                merged[key] = value
    sidecar = merged.get("split_metadata")
    if isinstance(sidecar, Mapping) and "available" in sidecar:
        merged.setdefault("split_metadata_available", bool(sidecar.get("available")))
        merged.setdefault("fix_hint", sidecar.get("fix_hint") or sidecar.get("warning"))
    if "split_metadata_available" not in merged and merged.get("split_metadata_path"):
        merged["split_metadata_available"] = True
    strict = merged.get("strict_validation_eligible")
    if strict is True:
        merged["split_eligibility"] = "strict"
    elif strict is False:
        merged["split_eligibility"] = "ineligible"
    else:
        merged["split_eligibility"] = "unknown"
    return merged


def _extract_split_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    split_payload = _preferred_split_payload(candidate)
    sidecar = candidate.get("split_metadata") if isinstance(candidate.get("split_metadata"), Mapping) else {}
    result = {
        "split_protocol": _first_non_none(candidate.get("split_protocol"), split_payload.get("split_protocol")),
        "split_strategy": _first_non_none(candidate.get("split_strategy"), split_payload.get("split_strategy")),
        "split_protocol_version": _first_non_none(
            candidate.get("split_protocol_version"),
            split_payload.get("split_protocol_version"),
        ),
        "split_metadata_path": _first_non_none(
            candidate.get("split_metadata_path"),
            split_payload.get("split_metadata_path"),
            sidecar.get("path"),
        ),
        "split_metadata_available": _first_non_none(
            candidate.get("split_metadata_available"),
            split_payload.get("split_metadata_available"),
            sidecar.get("available"),
        ),
        "strict_validation_eligible": _first_non_none(
            candidate.get("strict_validation_eligible"),
            split_payload.get("strict_validation_eligible"),
        ),
        "eligibility_reasons": _first_non_none(
            candidate.get("eligibility_reasons"),
            split_payload.get("eligibility_reasons"),
        ),
        "leakage_diagnostics": _first_non_none(
            candidate.get("leakage_diagnostics"),
            split_payload.get("leakage_diagnostics"),
        ),
        "split_seed": _first_non_none(candidate.get("split_seed"), split_payload.get("split_seed")),
        "split_sequence_count": _first_non_none(
            candidate.get("split_sequence_count"),
            split_payload.get("split_sequence_count"),
        ),
        "split_num_samples": _first_non_none(candidate.get("split_num_samples"), split_payload.get("split_num_samples")),
        "fix_hint": _first_non_none(candidate.get("fix_hint"), split_payload.get("fix_hint"), sidecar.get("warning")),
        "split_metadata": sidecar,
    }
    if result["eligibility_reasons"] is not None:
        result["eligibility_reasons"] = list(result["eligibility_reasons"] or [])
    return result


def _preferred_split_payload(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    splits = candidate.get("splits")
    if not isinstance(splits, Mapping):
        for split in ("target_test", "test", "validation", "val", "train"):
            payload = candidate.get(split)
            if isinstance(payload, Mapping):
                return payload
        return {}
    for split in ("target_test", "test", "validation", "val", "train"):
        payload = splits.get(split)
        if isinstance(payload, Mapping):
            return payload
    for payload in splits.values():
        if isinstance(payload, Mapping):
            return payload
    return {}


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _numeric_delta_from_values(lhs: Any, rhs: Any) -> float | None:
    if not isinstance(lhs, (int, float)) or not isinstance(rhs, (int, float)):
        return None
    return float(lhs) - float(rhs)


def _negative_transfer(adapted_top1: Any, source_top1: Any) -> bool | None:
    delta = _numeric_delta_from_values(adapted_top1, source_top1)
    return None if delta is None else bool(delta < 0.0)


def _bool_or_false(value: Any) -> bool:
    return bool(value) if value is not None else False


def _row_eligibility(
    row: Mapping[str, Any],
    adaptation_metrics: Mapping[str, Any],
    primary_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return row_eligibility(row, adaptation_metrics, primary_metrics)


def _prototype_is_required(row: Mapping[str, Any]) -> bool:
    return prototype_is_required(row)


def _prototype_is_no_op(row: Mapping[str, Any]) -> bool:
    return prototype_is_no_op(row)


def _unique_reasons(reasons: list[Any]) -> list[str]:
    return unique_reasons(reasons)


def _reason_histogram(reason_lists: Any) -> dict[str, int]:
    return reason_histogram(reason_lists)


def _excluded_run_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return excluded_run_summary(row)


def _conclusion_source_artifacts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return conclusion_source_artifacts(rows)


def _last_beam_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    baselines = metrics.get("degradation_baselines") if isinstance(metrics, Mapping) else None
    last = baselines.get("last_beam") if isinstance(baselines, Mapping) else None
    if not isinstance(last, Mapping):
        return {
            "available": False,
            "top1": None,
            "top3": None,
            "avg_top1": None,
            "avg_top3": None,
        }
    return {
        "available": bool(last.get("available", False)),
        "top1": last.get("top1"),
        "top3": last.get("top3"),
        "avg_top1": last.get("avg_top1"),
        "avg_top3": last.get("avg_top3"),
    }


def _cache_summary(record: Mapping[str, Any], source_train_metrics: Mapping[str, Any]) -> dict[str, Any]:
    throughput = source_train_metrics.get("throughput_config") if isinstance(source_train_metrics, Mapping) else None
    throughput = throughput if isinstance(throughput, Mapping) else {}
    return {
        "cache_policy": throughput.get("image_cache_policy") or throughput.get("cache_policy"),
        "lidar_cache_policy": throughput.get("lidar_cache_policy"),
        "lidar_cache_dir": throughput.get("lidar_cache_dir"),
        "num_workers": throughput.get("num_workers"),
        "cpu_threads": throughput.get("cpu_threads"),
    }


def _flatten_adaptation_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in diagnostics.items():
        short = str(key)
        if short.startswith("adaptation/"):
            short = short.split("/", 1)[1]
        if short == "prototype_loss":
            flat["prototype_loss_mean"] = value
        elif short == "prototype_used":
            flat["prototype_used_sample_count"] = value
        elif short == "prototype_status":
            flat["prototype_status"] = "effective" if float(value or 0.0) > 0 else "no_op"
        else:
            flat[short] = value
    if "prototype_status" not in flat and any(str(key).startswith("adaptation/prototype") for key in diagnostics):
        flat["prototype_status"] = "no_op"
    return flat


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


def _compare_coarse_to_radio(
    *,
    target_scene: Any,
    budget: Any,
    seed: Any,
    coarse: dict[str, Any] | None,
    radio: dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "comparison": "v5_coarse_vs_v6_radio",
        "target_scene": target_scene,
        "budget": budget,
        "seed": seed,
        "baseline_variant": "v5_adapter_proto",
        "candidate_variant": "v6_radio_proto",
    }
    missing = _missing_comparison_inputs({"v5_adapter_proto": coarse, "v6_radio_proto": radio})
    radio_missing = _missing_radio_metrics(radio)
    if missing or radio_missing:
        return {**base, "status": "inconclusive", "missing": missing + radio_missing}
    deltas = _accuracy_deltas(radio, coarse)
    return {
        **base,
        "status": "complete",
        "accuracy_deltas": deltas,
        "radio_accuracy": radio.get("radio_semantic_accuracy"),
        "power": {
            "normalized_received_power": radio.get("normalized_received_power"),
            "beam_power_loss_db": radio.get("beam_power_loss_db"),
        },
        "prototype": {
            "coverage": radio.get("prototype_coverage"),
            "confidence_mean": radio.get("prototype_confidence_mean") or radio.get("radio_assignment_confidence_mean"),
        },
        "efficiency": _efficiency_summary(radio),
        "radio_prototype_better_than_coarse": _is_better(deltas),
    }


def _compare_radio_condition(
    *,
    target_scene: Any,
    budget: Any,
    seed: Any,
    off: dict[str, Any] | None,
    on: dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "comparison": "radio_condition_off_vs_on",
        "target_scene": target_scene,
        "budget": budget,
        "seed": seed,
        "baseline_variant": "adapter_radio_proto",
        "candidate_variant": "v6_radio_proto",
    }
    missing = _missing_comparison_inputs({"adapter_radio_proto": off, "v6_radio_proto": on})
    if missing:
        return {**base, "status": "inconclusive", "missing": missing}
    deltas = _accuracy_deltas(on, off)
    prediction_delta = 0 if all(value == 0 for value in deltas.values() if value is not None) else None
    return {
        **base,
        "status": "complete",
        "accuracy_deltas": deltas,
        "radio_condition_prediction_delta": prediction_delta,
        "radio_assignment": {
            "off_confidence_mean": off.get("radio_assignment_confidence_mean"),
            "on_confidence_mean": on.get("radio_assignment_confidence_mean"),
        },
    }


def _compare_radio_to_path(
    *,
    target_scene: Any,
    budget: Any,
    seed: Any,
    radio: dict[str, Any] | None,
    path: dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "comparison": "v6_radio_vs_v8_path",
        "target_scene": target_scene,
        "budget": budget,
        "seed": seed,
        "baseline_variant": "v6_radio_proto",
        "candidate_variant": "v8_path_proto",
    }
    missing = _missing_comparison_inputs({"v6_radio_proto": radio, "v8_path_proto": path})
    path_missing = _missing_path_metrics(path)
    if missing or path_missing:
        return {**base, "status": "inconclusive", "missing": missing + path_missing}
    deltas = _accuracy_deltas(path, radio)
    return {
        **base,
        "status": "complete",
        "accuracy_deltas": deltas,
        "path_accuracy": path.get("path_semantic_accuracy"),
        "path_descriptor_mse": path.get("path_descriptor_regression_mse"),
        "prototype": {
            "coverage": path.get("prototype_coverage"),
            "confidence_mean": path.get("prototype_confidence_mean") or path.get("path_assignment_confidence_mean"),
        },
        "efficiency": _efficiency_summary(path),
        "path_prototype_better_than_radio": _is_better(deltas),
    }


def _compare_path_to_full(
    *,
    target_scene: Any,
    budget: Any,
    seed: Any,
    path: dict[str, Any] | None,
    full: dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "comparison": "v8_path_vs_full_finetune",
        "target_scene": target_scene,
        "budget": budget,
        "seed": seed,
        "baseline_variant": "v6_full_finetune",
        "candidate_variant": "v8_path_proto",
    }
    missing = _missing_comparison_inputs({"v8_path_proto": path, "v6_full_finetune": full})
    if missing:
        return {**base, "status": "inconclusive", "missing": missing}
    deltas = _accuracy_deltas(path, full)
    trainable_delta = _numeric_delta(path, full, "trainable_ratio")
    time_delta = _numeric_delta(path, full, "adaptation_time_seconds")
    return {
        **base,
        "status": "complete",
        "accuracy_deltas": deltas,
        "trainable_ratio_delta": trainable_delta,
        "adaptation_time_seconds_delta": time_delta,
        "path_prototype_better_than_full_finetune": _is_better(deltas)
        and (trainable_delta is None or trainable_delta <= 0)
        and (time_delta is None or time_delta <= 0),
    }


def _compare_path_condition(
    *,
    target_scene: Any,
    budget: Any,
    seed: Any,
    off: dict[str, Any] | None,
    on: dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "comparison": "path_condition_off_vs_on",
        "target_scene": target_scene,
        "budget": budget,
        "seed": seed,
        "baseline_variant": "adapter_path_proto",
        "candidate_variant": "v8_path_proto",
    }
    missing = _missing_comparison_inputs({"adapter_path_proto": off, "v8_path_proto": on})
    if missing:
        return {**base, "status": "inconclusive", "missing": missing}
    deltas = _accuracy_deltas(on, off)
    improves = _is_better(deltas)
    return {
        **base,
        "status": "complete",
        "accuracy_deltas": deltas,
        "path_condition_improved": improves,
        "diagnosis": None if improves else "path_prototype_may_be_more_effective_as_adaptation_anchor_than_beam_head_condition",
        "path_assignment": {
            "off_confidence_mean": off.get("path_assignment_confidence_mean"),
            "on_confidence_mean": on.get("path_assignment_confidence_mean"),
        },
    }


def _missing_radio_metrics(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if row is None:
        return []
    missing = []
    if row.get("radio_semantic_accuracy") is None:
        missing.append(
            {
                "variant": row.get("variant"),
                "reason": row.get("radio_metrics_unavailable_reason") or "radio_metrics_missing",
                "run_path": row.get("metrics_path"),
            }
        )
    if row.get("normalized_received_power") is None and row.get("beam_power_loss_db") is None:
        missing.append(
            {
                "variant": row.get("variant"),
                "reason": row.get("power_metrics_unavailable_reason") or "power_metrics_missing",
                "run_path": row.get("metrics_path"),
            }
        )
    return missing


def _missing_path_metrics(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if row is None:
        return []
    missing = []
    if row.get("path_semantic_accuracy") is None:
        missing.append(
            {
                "variant": row.get("variant"),
                "reason": row.get("path_metrics_unavailable_reason") or "path_metrics_missing",
                "run_path": row.get("metrics_path"),
            }
        )
    return missing


def _missing_comparison_inputs(items: Mapping[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
    missing = []
    for variant, row in items.items():
        if row is None:
            missing.append({"variant": variant, "reason": "missing_run"})
            continue
        if row.get("run_status") != "completed":
            missing.append({"variant": variant, "reason": row.get("failure_reason") or "run_not_completed"})
            continue
        if bool(row.get("main_conclusion_eligible", True)) is False:
            missing.append(
                {
                    "variant": variant,
                    "reason": "run_excluded_from_main_conclusion",
                    "eligibility_reasons": list(row.get("eligibility_reasons", []) or []),
                    "run_path": row.get("metrics_path"),
                }
            )
            continue
        if all(row.get(metric) is None for metric in ("top1", "top3", "top5", "coarse_accuracy", "fine_accuracy")):
            missing.append({"variant": variant, "reason": "metrics_missing"})
    return missing


def _accuracy_deltas(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        metric: _numeric_delta(candidate, baseline, metric)
        for metric in ("top1", "top3", "top5", "coarse_accuracy", "fine_accuracy", "radio_semantic_accuracy", "path_semantic_accuracy")
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


def _read_stage_progress(stage_dir: str | Path, *, phase: str | None = None) -> list[dict[str, Any]]:
    path = Path(stage_dir) / "progress.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if phase is None or payload.get("stage") == phase or payload.get("phase") == phase:
            rows.append(payload)
    return rows


def _preflight_error(scene: Any, resource_type: str, path: str | None, message: str, runs: list[dict[str, Any]] | None) -> dict[str, Any]:
    return preflight_error(scene, resource_type, path, message, runs)


def _runs_for_scene(runs: list[dict[str, Any]], scene: Any) -> list[dict[str, Any]]:
    return [
        run
        for run in runs
        if str(run.get("target_scene", "")) == str(scene)
        or str(scene) in {str(item) for item in run.get("source_scenes", [])}
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
