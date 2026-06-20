from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from kd_sensing.engine.training_metrics import checkpoint_task_metrics
from kd_sensing.engine.training_state import (
    TrainingState,
    early_stopping_improved,
    early_stopping_metric_value,
)
from kd_sensing.utils.artifact_registry import archive_best_checkpoint, write_sidecar
from kd_sensing.utils.checkpoint import load_checkpoint, save_checkpoint
from kd_sensing.utils.paths import resolve_path


@dataclass(frozen=True)
class CheckpointUpdate:
    early_stopping_value: float
    improved: bool
    top1_improved: bool


def checkpoint_strict(cfg: dict) -> bool:
    return bool(cfg.get("checkpoint", {}).get("strict_load", True))


def resolve_resume_checkpoint(cfg: dict, run_dir: Path) -> Path | None:
    resume = cfg.get("training", {}).get("resume", False)
    if not resume:
        return None
    if resume is True:
        if not cfg.get("output", {}).get("run_name"):
            raise ValueError("training.resume=true requires output.run_name so checkpoints/last.pth can be resolved.")
        checkpoint_path = run_dir / "checkpoints" / "last.pth"
    elif isinstance(resume, str):
        checkpoint_path = resolve_path(resume)
    else:
        raise ValueError("training.resume must be false, true, or a checkpoint path string.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    return checkpoint_path


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

    def restore_if_needed(self, state: TrainingState, *, objective: str, device: torch.device) -> tuple[str, str]:
        resume_path = resolve_resume_checkpoint(self.cfg, self.run_dir)
        if resume_path is None:
            return self.early_stopping_metric, self.early_stopping_mode
        checkpoint = load_checkpoint(
            resume_path,
            self.primary_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            strict=checkpoint_strict(self.cfg),
            role="resume",
            map_location=device,
        )
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
    ) -> CheckpointUpdate:
        early_stopping_value = early_stopping_metric_value(epoch_log, self.early_stopping_metric)
        if float(val_loss) < state.best_val_loss:
            state.best_val_loss = float(val_loss)
        improved = early_stopping_improved(
            early_stopping_value,
            state.best_early_stopping_value,
            mode=self.early_stopping_mode,
            min_delta=float(self.cfg.get("training", {}).get("min_delta", 0.0)),
        )
        if improved:
            state.best_early_stopping_value = early_stopping_value
            state.best_early_stopping_epoch = epoch + 1
            state.epochs_without_improvement = 0
            best_objective_path = self.run_dir / "checkpoints" / "best.pth"
            torch.save(self.primary_model.state_dict(), best_objective_path)
            write_sidecar(
                best_objective_path,
                self._checkpoint_sidecar(
                    best_objective_path,
                    checkpoint_source="objective-checkpoint",
                    selection_metric=self.early_stopping_metric,
                    selection_mode="early_stopping",
                    selected_epoch=state.best_early_stopping_epoch,
                    early_stopping_value=early_stopping_value,
                    epoch_log=epoch_log,
                    train_dataset=train_dataset,
                ),
            )
        else:
            state.epochs_without_improvement += 1

        top1_improved = self.objective_metadata.get("name") != "gps_conditioned_jepa" and val_acc > state.best_val_top1
        if top1_improved:
            state.best_val_top1 = val_acc
            state.best_top1_epoch = epoch + 1
            best_top1_path = self.run_dir / "checkpoints" / "best_top1.pth"
            torch.save(self.primary_model.state_dict(), best_top1_path)
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
            write_sidecar(
                best_top1_path,
                self._checkpoint_sidecar(
                    best_top1_path,
                    checkpoint_source="top1-checkpoint",
                    selection_metric="val_acc_top1",
                    selection_mode="top1-selection",
                    selected_epoch=state.best_top1_epoch,
                    early_stopping_value=early_stopping_value,
                    epoch_log=epoch_log,
                    train_dataset=train_dataset,
                ),
            )
        return CheckpointUpdate(
            early_stopping_value=float(early_stopping_value),
            improved=bool(improved),
            top1_improved=bool(top1_improved),
        )

    def save_last_checkpoint(self, *, state: TrainingState, epoch: int, val_loss: float) -> None:
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": self.primary_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "test_loss": val_loss,
                "best_val_loss": state.best_val_loss,
                "best_val_top1": state.best_val_top1,
                "best_top1_epoch": state.best_top1_epoch,
                "early_stopping_metric": self.early_stopping_metric,
                "early_stopping_mode": self.early_stopping_mode,
                "best_early_stopping_value": state.best_early_stopping_value,
                "best_early_stopping_epoch": state.best_early_stopping_epoch,
                "epochs_without_improvement": state.epochs_without_improvement,
                "normalization_artifacts": self.normalization_artifacts,
                "checkpoint_registry": state.registry_checkpoint,
                "prediction_objective": self.objective_metadata,
            },
            self.run_dir / "checkpoints",
            "last.pth",
        )

    def _checkpoint_sidecar(
        self,
        path: Path,
        *,
        checkpoint_source: str,
        selection_metric: str,
        selection_mode: str,
        selected_epoch: int,
        early_stopping_value: float,
        epoch_log: dict,
        train_dataset,
    ) -> dict[str, Any]:
        return {
            "path": str(path),
            "source": "local",
            "checkpoint_source": checkpoint_source,
            "run_dir": str(self.run_dir),
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "selected_epoch": int(selected_epoch),
            "objective_metric": {
                "name": self.early_stopping_metric,
                "mode": self.early_stopping_mode,
                "value": float(early_stopping_value),
            },
            "task_metrics": checkpoint_task_metrics(epoch_log),
            "normalization_artifacts": self.normalization_artifacts,
            "split_metadata": self.split_metadata,
            "task": self.cfg.get("experiment", {}).get("task"),
            "enabled_modalities": list(getattr(train_dataset, "enabled_modalities", [])),
        }
