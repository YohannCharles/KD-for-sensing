from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from kd_sensing.data.loso import sample_few_shot_records
from kd_sensing.engine.hist_beam_history_anchor import history_anchor_run_metadata
from kd_sensing.engine.hist_beam_loso_artifacts import _append_stage_progress, _csv_records, _first_numbered_key, _read_stage_progress, _write_json
from kd_sensing.engine.hist_beam_loso_config import (
    SOURCE_ONLY_VARIANTS,
    _adaptation_cache_key,
    _prototype_decision,
    _reuse_source_checkpoint,
    _source_cache_key,
    _source_variant_for,
    _stage_cfg,
    _throughput_config_summary,
)
from kd_sensing.engine.hist_beam_loso_records import _run_identity
from kd_sensing.engine.hist_beam_loso_summary import _flatten_adaptation_diagnostics
from kd_sensing.engine.run_lineage import run_lineage_metadata
from kd_sensing.utils.seed import set_seed

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
        from kd_sensing.engine.batch import prepare_beamspace_power_targets, prepare_history_anchor_inputs, prepare_radio_semantic_labels
        from kd_sensing.engine.hist_beam_baselines import collect_source_beam_reference
        from kd_sensing.engine.hist_beam_losses import compute_hist_beam_loss, hist_beam_enabled
        from kd_sensing.engine.hist_beam_residuals import history_anchor_enabled, num_delta_classes_from_config
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
                        history_kwargs = prepare_history_anchor_inputs(
                            batch,
                            num_pred=model_cfg.get("num_pred", cfg.get("data", {}).get("dataset", {}).get("num_pred", 1)),
                            num_classes=num_delta_classes_from_config(cfg, default=num_classes),
                            downsample_ratio=model_cfg.get("downsample_ratio", 1),
                            device=device,
                            enabled=history_anchor_enabled(cfg),
                            non_blocking=non_blocking,
                        )
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
                            extra_model_kwargs={key: value for key, value in history_kwargs.items() if key != "residual_labels"},
                        )
                        if step.labels is None:
                            raise RuntimeError("Source training labels were not prepared.")
                        output = {"logits": step.logits, **step.model_output.diagnostics}
                        if hist_beam_enabled(cfg, output):
                            beamspace_targets = prepare_beamspace_power_targets(
                                step.batch,
                                num_pred=step.labels.shape[1],
                                device=device,
                                non_blocking=non_blocking,
                            )
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
                                beamspace_power_labels=beamspace_targets[0] if beamspace_targets is not None else None,
                                beamspace_power_mask=beamspace_targets[1] if beamspace_targets is not None else None,
                                current_epoch=epoch_index,
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
            source_reference = collect_source_beam_reference(loaders["source_train"], cfg, device, output_path=context.stage_dir / "source_beam_reference.pt")
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
                "source_sampling": dict(loaders.get("source_sampling", {})),
                **dict(source_reference.get("metadata", {})),
                **run_lineage_metadata(cfg, default_method_family="hist_beam_mainline"),
                **history_anchor_run_metadata(cfg),
            }
            metrics_path = context.stage_dir / "metrics.json"
            _write_json(metrics_path, metrics)
            artifacts = {
                "run_dir": str(context.stage_dir),
                "metrics_path": str(metrics_path),
                "source_checkpoint_path": str(checkpoint_path),
                "source_prototype_path": str(prototype_path) if prototype_status.get("path") else None,
                "source_beam_reference_path": source_reference.get("path"),
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
            strategy = "v6_full_finetune" if variant == "v6_full_finetune" else "v7_private_residual" if variant == "v7_shared_physical_private_residual" else variant
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
                **run_lineage_metadata(cfg, default_method_family="hist_beam_mainline"),
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
    from kd_sensing.engine.hist_beam_baselines import attach_source_beam_reference, source_prior_collapse_metrics
    from kd_sensing.engine.loso_data import build_loso_target_stage_loader
    from kd_sensing.engine.optim import build_device, build_model, build_task_criterion
    from kd_sensing.evaluation.hist_beam_outputs import write_hist_beam_predictions

    device = build_device(cfg)
    executor = DefaultHistBeamLosoStageExecutor()
    source_reference_path = ((context.state["source_checkpoints"].get(_source_cache_key(run, _source_variant_for(run))) or {}).get("artifacts") or {}).get("source_beam_reference_path")
    source_reference = attach_source_beam_reference(cfg, source_reference_path, map_location=device)
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
                **run_lineage_metadata(cfg, default_method_family="hist_beam_mainline"),
            }
        )
        metrics.update(source_prior_collapse_metrics(source_reference, metrics))
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
            shared_logits=result.shared_logits,
            last_beams=result.last_beams,
            residual_logits=result.residual_logits,
            residual_labels=result.residual_labels,
        )
        return {
            "status": "completed",
            "artifacts": {
                "run_dir": str(context.stage_dir),
                "metrics_path": str(metrics_path),
                "predictions_path": str(predictions_path),
                "source_checkpoint_path": str(checkpoint_path),
                "source_beam_reference_path": source_reference_path,
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
        stratification=_few_shot_stratification_for_sampling(cfg),
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


def _few_shot_stratification_for_sampling(cfg: dict[str, Any]) -> str | None:
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    adapt_cfg = hist_cfg.get("adaptation") if isinstance(hist_cfg.get("adaptation"), dict) else {}
    sampling_cfg = adapt_cfg.get("sampling") if isinstance(adapt_cfg.get("sampling"), dict) else {}
    value = adapt_cfg.get("few_shot_stratification", sampling_cfg.get("stratification"))
    return str(value) if value is not None else None


def _load_checkpoint_state(model: Any, checkpoint_path: str | Path, *, device: Any, strict: bool) -> None:
    import torch

    payload = torch.load(Path(checkpoint_path), map_location=device)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=strict)

__all__ = ["DefaultHistBeamLosoStageExecutor", "StageExecutionContext", "StageExecutor", "StageRunCallbacks", "execute_loso_stage_runs"]
