from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from kd_sensing.engine.training_metrics import checkpoint_task_metrics
from kd_sensing.engine.checkpoint_selection import (
    checkpoint_selection_score,
    model_selection_enabled,
    resolve_checkpoint_selection_metric,
)
from kd_sensing.engine.training_state import (
    TrainingState,
    early_stopping_improved,
    early_stopping_metric_value,
    initial_early_stopping_value,
)
from kd_sensing.engine.training_resume import (
    CHECKPOINT_SCHEMA_VERSION,
    build_resume_contract,
    capture_runtime_state as capture_training_runtime_state,
    preflight_resume,
    resolve_resume_path,
    restore_runtime_state,
)
from kd_sensing.utils.artifact_registry import (
    archive_best_checkpoint,
    gps_checkpoint_provenance,
    load_checkpoint_metadata,
)
from kd_sensing.utils.checkpoint import load_checkpoint, publish_checkpoint


@dataclass(frozen=True)
class CheckpointUpdate:
    early_stopping_value: float
    improved: bool
    top1_improved: bool
    selection_value: float | None = None


def checkpoint_strict(cfg: dict) -> bool:
    return bool(cfg.get("checkpoint", {}).get("strict_load", True))


def resolve_resume_checkpoint(cfg: dict, run_dir: Path) -> Path | None:
    return resolve_resume_path(cfg, run_dir)


class CheckpointManager:
    def __init__(
        self,
        *,
        cfg: dict,
        run_dir: Path,
        primary_model,
        optimizer,
        scheduler,
        split_metadata: dict | None,
        normalization_artifacts: dict | None,
        objective_metadata: dict[str, Any],
        early_stopping_metric: str,
        early_stopping_mode: str,
        dataloaders: dict[str, Any] | None = None,
        grad_scaler: Any | None = None,
        extensions: list[Any] | None = None,
        extension_states: list[Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.primary_model = primary_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.split_metadata = split_metadata
        self.normalization_artifacts = normalization_artifacts
        self.objective_metadata = objective_metadata
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_mode = early_stopping_mode
        self.selection_metric = resolve_checkpoint_selection_metric(cfg)
        self.dataloaders = dataloaders or {}
        self.grad_scaler = grad_scaler
        self.extensions = extensions or []
        self.extension_states = extension_states or []

    def restore_if_needed(self, state: TrainingState, *, objective: str, device: torch.device) -> tuple[str, str]:
        plan = preflight_resume(
            self.cfg,
            self.run_dir,
            scheduler_enabled=self.scheduler is not None,
            split_metadata=self.split_metadata,
            normalization_artifacts=self.normalization_artifacts,
        )
        if plan is None:
            return self.early_stopping_metric, self.early_stopping_mode
        checkpoint = load_checkpoint(
            plan.path,
            self.primary_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            strict=checkpoint_strict(self.cfg),
            role="resume",
            map_location=device,
        )
        if plan.schema == "legacy":
            checkpoint.update({"resume_migration": plan.payload["resume_migration"]})
            checkpoint.setdefault("best_val_loss", plan.payload.get("best_val_loss"))
        else:
            restore_runtime_state(
                checkpoint["runtime_state"],
                dataloaders=self.dataloaders,
                grad_scaler=self.grad_scaler,
                extensions=self.extensions,
                extension_states=self.extension_states,
                training_state=state,
            )
        _apply_sidecar_resume_metadata(checkpoint, plan.path)
        state.checkpoint_loads.append(checkpoint.get("_load_info"))
        self.early_stopping_metric, self.early_stopping_mode = state.apply_resume_checkpoint(
            checkpoint,
            early_stopping_metric=self.early_stopping_metric,
            early_stopping_mode=self.early_stopping_mode,
            objective=objective,
        )
        return self.early_stopping_metric, self.early_stopping_mode

    def update_best_checkpoints(
        self,
        *,
        state: TrainingState,
        epoch: int,
        epoch_log: dict,
        val_loss: float,
        val_acc: float,
        train_dataset,
        runtime_state: dict[str, Any] | None = None,
    ) -> CheckpointUpdate:
        early_stopping_value = early_stopping_metric_value(epoch_log, self.early_stopping_metric)
        selection_value = None
        if self.objective_metadata.get("name") != "gps_conditioned_jepa":
            selection_value = checkpoint_selection_score(epoch_log, self.selection_metric, val_acc=val_acc)
            epoch_log["checkpoint_selection_metric"] = self.selection_metric
            epoch_log["checkpoint_selection_value"] = float(selection_value)
        if float(val_loss) < state.best_val_loss:
            state.best_val_loss = float(val_loss)
        if state.best_early_stopping_epoch == 0:
            state.best_early_stopping_value = initial_early_stopping_value(self.early_stopping_mode)
        improved = early_stopping_improved(
            early_stopping_value,
            state.best_early_stopping_value,
            mode=self.early_stopping_mode,
            min_delta=float(self.cfg.get("training", {}).get("min_delta", 0.0)),
        )
        top1_improved = self.objective_metadata.get("name") != "gps_conditioned_jepa" and val_acc > state.best_val_top1
        selection_improved = (
            self.selection_metric != "val_acc"
            and selection_value is not None
            and selection_value > state.best_selection_value
        )
        if improved:
            state.best_early_stopping_value = early_stopping_value
            state.best_early_stopping_epoch = epoch + 1
            state.epochs_without_improvement = 0
        else:
            state.epochs_without_improvement += 1
        if top1_improved:
            state.best_val_top1 = val_acc
            state.best_top1_epoch = epoch + 1
        if selection_improved:
            state.best_selection_value = float(selection_value)
            state.best_selection_epoch = epoch + 1
        if improved:
            self._publish_checkpoint(
                state=state,
                epoch=epoch,
                val_loss=val_loss,
                filename="best.pth",
                checkpoint_role="objective_best",
                catalog_key="objective_best",
                checkpoint_source="objective-checkpoint",
                selection_metric=self.early_stopping_metric,
                selection_mode=self.early_stopping_mode,
                selected_epoch=state.best_early_stopping_epoch,
                selection_value=early_stopping_value,
                final_test_candidate=True,
                early_stopping_value=early_stopping_value,
                epoch_log=epoch_log,
                train_dataset=train_dataset,
                runtime_state=runtime_state,
            )
        if top1_improved:
            best_top1_path = self._publish_checkpoint(
                state=state,
                epoch=epoch,
                val_loss=val_loss,
                filename="best_top1.pth",
                checkpoint_role="top1_best",
                catalog_key="top1_best",
                checkpoint_source="top1-checkpoint",
                selection_metric="val_acc_top1",
                selection_mode="max",
                selected_epoch=state.best_top1_epoch,
                selection_value=val_acc,
                final_test_candidate=True,
                early_stopping_value=early_stopping_value,
                epoch_log=epoch_log,
                train_dataset=train_dataset,
                runtime_state=runtime_state,
            )
            state.registry_checkpoint = archive_best_checkpoint(
                self.cfg,
                source_checkpoint=best_top1_path,
                val_top1=state.best_val_top1,
                epoch=state.best_top1_epoch,
                run_dir=self.run_dir,
                split_metadata=self.split_metadata,
                normalization_artifacts=self.normalization_artifacts,
                objective_metric={
                    "name": self.early_stopping_metric,
                    "mode": self.early_stopping_mode,
                    "value": early_stopping_value,
                },
                task_metrics=checkpoint_task_metrics(epoch_log),
                selection_metric="val_acc_top1",
                selection_mode="top1-selection",
                checkpoint_source="top1-checkpoint",
            )
        if selection_improved:
            self._publish_checkpoint(
                state=state,
                epoch=epoch,
                val_loss=val_loss,
                filename=f"best_{self.selection_metric}.pth",
                checkpoint_role="selection_best",
                catalog_key=self.selection_metric,
                checkpoint_source="selection-checkpoint",
                selection_metric=self.selection_metric,
                selection_mode="max",
                selected_epoch=state.best_selection_epoch,
                selection_value=selection_value,
                final_test_candidate=True,
                early_stopping_value=early_stopping_value,
                epoch_log=epoch_log,
                train_dataset=train_dataset,
                runtime_state=runtime_state,
            )
        return CheckpointUpdate(
            early_stopping_value=float(early_stopping_value),
            improved=bool(improved),
            top1_improved=bool(top1_improved),
            selection_value=float(selection_value) if selection_value is not None else None,
        )

    def save_last_checkpoint(
        self,
        *,
        state: TrainingState,
        epoch: int,
        val_loss: float | None,
        runtime_state: dict[str, Any] | None = None,
    ) -> None:
        self._publish_checkpoint(
            state=state,
            epoch=epoch,
            val_loss=val_loss,
            filename="last.pth",
            checkpoint_role="last",
            catalog_key="last",
            checkpoint_source="last-checkpoint",
            selection_metric=None,
            selection_mode=None,
            selected_epoch=epoch + 1,
            selection_value=None,
            final_test_candidate=False,
            early_stopping_value=None,
            epoch_log={},
            train_dataset=None,
            runtime_state=runtime_state,
        )

    def capture_runtime_state(self, state: TrainingState) -> dict[str, Any]:
        """Freeze the state shared by every checkpoint published for one epoch."""
        return capture_training_runtime_state(
            dataloaders=self.dataloaders,
            grad_scaler=self.grad_scaler,
            extensions=self.extensions,
            extension_states=self.extension_states,
            training_state=state,
        )

    def _publish_checkpoint(
        self,
        *,
        state: TrainingState,
        epoch: int,
        val_loss: float | None,
        filename: str,
        checkpoint_role: str,
        catalog_key: str,
        checkpoint_source: str,
        selection_metric: str | None,
        selection_mode: str | None,
        selected_epoch: int,
        selection_value: float | None,
        final_test_candidate: bool,
        early_stopping_value: float | None,
        epoch_log: dict,
        train_dataset,
        runtime_state: dict[str, Any] | None,
    ) -> Path:
        selection = {
            "metric": selection_metric,
            "mode": selection_mode,
            "value": float(selection_value) if selection_value is not None else None,
            "selected_epoch": int(selected_epoch),
            "source_run": str(self.run_dir),
            "final_test_candidate": bool(final_test_candidate),
        }
        payload = self._checkpoint_payload(
            state=state,
            epoch=epoch,
            val_loss=val_loss,
            checkpoint_role=checkpoint_role,
            selection=selection,
            runtime_state=runtime_state,
        )
        path, metadata = publish_checkpoint(
            payload,
            self.run_dir / "checkpoints",
            filename,
            metadata=self._checkpoint_sidecar(
                path=self.run_dir / "checkpoints" / filename,
                checkpoint_source=checkpoint_source,
                selection=selection,
                early_stopping_value=early_stopping_value,
                epoch_log=epoch_log,
                train_dataset=train_dataset,
            ),
        )
        state.selection_catalog[catalog_key] = {
            "path": str(path),
            "checkpoint_role": checkpoint_role,
            "selection": selection,
            "checkpoint_sha256": metadata["checkpoint_sha256"],
        }
        return path

    def _checkpoint_payload(
        self,
        *,
        state: TrainingState,
        epoch: int,
        val_loss: float | None,
        checkpoint_role: str,
        selection: dict[str, Any],
        runtime_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_role": checkpoint_role,
            "epoch": epoch + 1,
            "state_dict": self.primary_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "validation_loss": val_loss,
            "best_val_loss": state.best_val_loss,
            "best_val_top1": state.best_val_top1,
            "best_top1_epoch": state.best_top1_epoch,
            "early_stopping_metric": self.early_stopping_metric,
            "early_stopping_mode": self.early_stopping_mode,
            "best_early_stopping_value": state.best_early_stopping_value,
            "best_early_stopping_epoch": state.best_early_stopping_epoch,
            "selection_metric": self.selection_metric,
            "best_selection_value": state.best_selection_value,
            "best_selection_epoch": state.best_selection_epoch,
            "epochs_without_improvement": state.epochs_without_improvement,
            "normalization_artifacts": self.normalization_artifacts,
            "checkpoint_registry": state.registry_checkpoint,
            "prediction_objective": self.objective_metadata,
            "model_selection_enabled": model_selection_enabled(self.cfg),
            "runtime_state": runtime_state if runtime_state is not None else self.capture_runtime_state(state),
            "resume_contract": build_resume_contract(
                self.cfg,
                self.split_metadata,
                self.normalization_artifacts,
            ),
            "selection": selection,
            "selection_catalog": dict(state.selection_catalog),
            **gps_checkpoint_provenance(self.cfg),
        }

    def _checkpoint_sidecar(
        self,
        path: Path,
        *,
        checkpoint_source: str,
        selection: dict[str, Any],
        early_stopping_value: float | None,
        epoch_log: dict,
        train_dataset,
    ) -> dict[str, Any]:
        return {
            "path": str(path),
            "source": "local",
            "checkpoint_source": checkpoint_source,
            "run_dir": str(self.run_dir),
            "selection_metric": selection["metric"],
            "selection_mode": selection["mode"],
            "selection_value": selection["value"],
            "selected_epoch": selection["selected_epoch"],
            "selection": selection,
            "objective_metric": {
                "name": self.early_stopping_metric,
                "mode": self.early_stopping_mode,
                "value": float(early_stopping_value) if early_stopping_value is not None else None,
            },
            "task_metrics": checkpoint_task_metrics(epoch_log),
            "normalization_artifacts": self.normalization_artifacts,
            "split_metadata": self.split_metadata,
            "task": self.cfg.get("experiment", {}).get("task"),
            "enabled_modalities": list(getattr(train_dataset, "enabled_modalities", [])) if train_dataset is not None else [],
            **gps_checkpoint_provenance(self.cfg),
        }


def _apply_sidecar_resume_metadata(checkpoint: dict[str, Any], resume_path: Path) -> None:
    if "epoch" in checkpoint:
        return
    metadata = load_checkpoint_metadata(resume_path)
    if not isinstance(metadata, dict):
        return
    if "selected_epoch" in metadata:
        checkpoint["epoch"] = int(metadata["selected_epoch"])
    objective_metric = metadata.get("objective_metric")
    if isinstance(objective_metric, dict):
        if "name" in objective_metric:
            checkpoint.setdefault("early_stopping_metric", objective_metric["name"])
        if "mode" in objective_metric:
            checkpoint.setdefault("early_stopping_mode", objective_metric["mode"])
        if "value" in objective_metric:
            checkpoint.setdefault("best_early_stopping_value", objective_metric["value"])
            checkpoint.setdefault("best_early_stopping_epoch", metadata.get("selected_epoch", 0))
    task_metrics = metadata.get("task_metrics")
    if isinstance(task_metrics, dict):
        checkpoint.setdefault("best_val_top1", task_metrics.get("val_acc", metadata.get("metric_value", float("-inf"))))
        checkpoint.setdefault("best_val_loss", task_metrics.get("val_loss", checkpoint.get("best_val_loss", float("inf"))))
