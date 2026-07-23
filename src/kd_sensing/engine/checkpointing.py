from pathlib import Path
from typing import Any

import torch

from kd_sensing.engine.training_resume import (
    CHECKPOINT_SCHEMA_VERSION,
    build_resume_contract,
    capture_runtime_state,
    preflight_resume,
    resolve_resume_path,
    restore_runtime_state,
)
from kd_sensing.engine.training_state import TrainingState
from kd_sensing.utils.artifact_registry import gps_checkpoint_provenance
from kd_sensing.utils.checkpoint import load_checkpoint, publish_checkpoint


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
        dataloaders: dict[str, Any],
        grad_scaler: Any | None,
        extensions: list[Any],
        extension_states: list[Any],
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.primary_model = primary_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.split_metadata = split_metadata
        self.normalization_artifacts = normalization_artifacts
        self.objective_metadata = objective_metadata
        self.dataloaders = dataloaders
        self.grad_scaler = grad_scaler
        self.extensions = extensions
        self.extension_states = extension_states

    def restore_if_needed(self, state: TrainingState, *, device: torch.device) -> None:
        plan = preflight_resume(
            self.cfg,
            self.run_dir,
            scheduler_enabled=self.scheduler is not None,
            split_metadata=self.split_metadata,
            normalization_artifacts=self.normalization_artifacts,
        )
        if plan is None:
            return
        checkpoint = load_checkpoint(
            plan.path,
            self.primary_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            strict=checkpoint_strict(self.cfg),
            role="resume",
            map_location=device,
        )
        restore_runtime_state(
            checkpoint["runtime_state"],
            dataloaders=self.dataloaders,
            grad_scaler=self.grad_scaler,
            extensions=self.extensions,
            extension_states=self.extension_states,
            training_state=state,
        )
        state.start_epoch = int(checkpoint["epoch"])
        state.checkpoint_loads.append(checkpoint.get("_load_info"))

    def save_last_checkpoint(self, *, state: TrainingState, epoch: int, val_loss: float | None) -> Path:
        return self._save_checkpoint(
            state=state,
            epoch=epoch,
            val_loss=val_loss,
            filename="last.pth",
            role="last",
            checkpoint_source="fixed-epoch-last",
            checkpoint_policy="fixed_epoch_last_pth",
        )

    def save_best_checkpoint(self, *, state: TrainingState, epoch: int, val_loss: float) -> Path:
        return self._save_checkpoint(
            state=state,
            epoch=epoch,
            val_loss=float(val_loss),
            filename="best.pth",
            role="validation_best",
            checkpoint_source="validation-best",
            checkpoint_policy="best_validation_loss",
            selection={
                "metric": "validation_loss",
                "mode": "min",
                "value": float(val_loss),
                "epoch": int(epoch) + 1,
            },
        )

    def _save_checkpoint(
        self,
        *,
        state: TrainingState,
        epoch: int,
        val_loss: float | None,
        filename: str,
        role: str,
        checkpoint_source: str,
        checkpoint_policy: str,
        selection: dict[str, Any] | None = None,
    ) -> Path:
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_role": role,
            "epoch": int(epoch) + 1,
            "state_dict": self.primary_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "validation_loss": val_loss,
            "normalization_artifacts": self.normalization_artifacts,
            "prediction_objective": self.objective_metadata,
            "runtime_state": self.capture_runtime_state(state),
            "resume_contract": build_resume_contract(
                self.cfg,
                self.split_metadata,
                self.normalization_artifacts,
            ),
            **gps_checkpoint_provenance(self.cfg),
        }
        if selection is not None:
            payload["selection"] = dict(selection)
        path, _ = publish_checkpoint(
            payload,
            self.run_dir / "checkpoints",
            filename,
            metadata={
                "source": "local",
                "checkpoint_source": checkpoint_source,
                "checkpoint_policy": checkpoint_policy,
                "run_dir": str(self.run_dir),
                "normalization_artifacts": self.normalization_artifacts,
                "split_metadata": self.split_metadata,
                "task": self.cfg.get("experiment", {}).get("task"),
                "enabled_modalities": list(self.cfg.get("model", {}).get("primary", {}).get("modalities", ())),
                **gps_checkpoint_provenance(self.cfg),
            },
        )
        return path

    def capture_runtime_state(self, state: TrainingState) -> dict[str, Any]:
        return capture_runtime_state(
            dataloaders=self.dataloaders,
            grad_scaler=self.grad_scaler,
            extensions=self.extensions,
            extension_states=self.extension_states,
            training_state=state,
        )
