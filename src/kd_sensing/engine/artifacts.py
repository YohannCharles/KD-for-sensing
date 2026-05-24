from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.config.io import dump_config
from kd_sensing.config.lidar_normalization import canonicalize_lidar_normalization_config
from kd_sensing.data.scenes import scene_metadata_from_config
from kd_sensing.engine.debug_diagnostics import (
    evaluate_pilot_noise_validity,
    write_csi_debug_records,
    write_pilot_noise_validity_artifact,
)
from kd_sensing.engine.multimodal_nf_runtime import validate_multimodal_nf_runtime_contract
from kd_sensing.engine.objectives.metadata import objective_runtime_metadata
from kd_sensing.engine.run_metadata import prediction_setup_metadata
from kd_sensing.engine.training_metrics import training_outputs_payload
from kd_sensing.engine.training_state import early_stopping_state
from kd_sensing.utils.plotting import plot_training_curves
from kd_sensing.utils.teacher_registry import teacher_metrics_from_training


def final_config_with_runtime(
    cfg: dict,
    *,
    run_dir: Path,
    split_metadata: dict | None = None,
    normalization_artifacts: dict | None = None,
    checkpoint_registry: dict | None = None,
    throughput_metadata: dict | None = None,
    teacher_prior: dict | None = None,
    early_stopping: dict | None = None,
    pilot_noise_validity: dict | None = None,
) -> dict:
    validate_multimodal_nf_runtime_contract(cfg, split_metadata=split_metadata)
    final_cfg = deepcopy(cfg)
    canonicalize_lidar_normalization_config(final_cfg)
    runtime = final_cfg.setdefault("runtime", {})
    runtime["run_dir"] = str(run_dir)
    runtime["output_overwrite"] = bool(cfg.get("output", {}).get("overwrite", False))
    runtime["prediction_objective"] = objective_runtime_metadata(cfg)
    if split_metadata is not None:
        runtime["splits"] = split_metadata
    if normalization_artifacts is not None:
        runtime["normalization_artifacts"] = normalization_artifacts
    if checkpoint_registry is not None:
        runtime["checkpoint_registry"] = checkpoint_registry
    if throughput_metadata is not None:
        runtime["throughput"] = throughput_metadata
        if isinstance(throughput_metadata, dict) and "cache" in throughput_metadata:
            runtime["cache"] = throughput_metadata["cache"]
    if teacher_prior is not None:
        runtime["teacher_prior"] = teacher_prior
    if early_stopping is not None:
        runtime["early_stopping"] = early_stopping
    if pilot_noise_validity is not None:
        runtime["pilot_noise_validity"] = pilot_noise_validity
    scene_metadata = scene_metadata_from_config(cfg)
    if scene_metadata:
        runtime["scene"] = scene_metadata
    runtime["prediction_setup"] = prediction_setup_metadata(cfg, split_metadata=split_metadata)
    return final_cfg


class ArtifactWriter:
    def __init__(self, *, cfg: dict, run_dir: Path) -> None:
        self.cfg = cfg
        self.run_dir = run_dir

    def write_initial_configs(
        self,
        *,
        split_metadata: dict | None,
        normalization_artifacts: dict | None,
        throughput_metadata: dict | None,
    ) -> dict:
        resolved_cfg = final_config_with_runtime(
            self.cfg,
            run_dir=self.run_dir,
            split_metadata=split_metadata,
            normalization_artifacts=normalization_artifacts,
            throughput_metadata=throughput_metadata,
        )
        dump_config(resolved_cfg, self.run_dir / "resolved_config.yaml")
        dump_config(resolved_cfg, self.run_dir / "final_config.yaml")
        return resolved_cfg

    def write_final_artifacts(
        self,
        *,
        history: dict[str, list],
        epoch_logs: list[dict[str, Any]],
        objective_metadata: dict[str, Any],
        early_stopping_metric: str,
        early_stopping_mode: str,
        best_early_stopping_value: float,
        best_early_stopping_epoch: int,
        epochs_without_improvement: int,
        checkpoint_loads: list[dict[str, Any] | None],
        teacher_prior_info: dict[str, Any] | None,
        optimizer_groups: list[dict[str, Any]],
        normalization_artifacts: dict | None,
        checkpoint_registry: dict | None,
        throughput_metadata: dict | None,
        split_metadata: dict | None,
        startup_summary: dict[str, Any],
        config_diff: dict[str, Any] | None,
        csi_debug_records: list[dict[str, Any]],
        best_top1_epoch: int,
    ) -> dict[str, Any]:
        np.savez(
            self.run_dir / "training_outputs.npz",
            **training_outputs_payload(history, objective_metadata, early_stopping_metric, early_stopping_mode),
        )
        teacher_metrics = teacher_metrics_from_training(
            self.cfg,
            history,
            epoch_logs,
            best_selected_epoch=best_early_stopping_epoch,
            selection_metric=early_stopping_metric,
            selection_mode="early_stopping",
            checkpoint="checkpoints/best.pth",
            best_top1_epoch=best_top1_epoch,
        )
        if teacher_metrics is not None:
            with (self.run_dir / "teacher_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(teacher_metrics, f, indent=2)
        early_stopping_metadata = early_stopping_state(
            metric=early_stopping_metric,
            mode=early_stopping_mode,
            best_value=best_early_stopping_value,
            best_epoch=best_early_stopping_epoch,
            epochs_without_improvement=epochs_without_improvement,
        )
        write_csi_debug_records(self.run_dir, csi_debug_records)
        pilot_noise_validity = evaluate_pilot_noise_validity(self.cfg, csi_debug_records)
        write_pilot_noise_validity_artifact(self.run_dir, pilot_noise_validity)
        train_log = {
            **history,
            "epoch_logs": epoch_logs,
            "early_stopping": early_stopping_metadata,
            "startup_summary": startup_summary,
            "config_diff": config_diff,
            "csi_first_batch_diagnostics": csi_debug_records,
            "pilot_noise_validity": pilot_noise_validity,
            "teacher_metrics": teacher_metrics,
            "checkpoint_loads": checkpoint_loads,
            "teacher_prior": teacher_prior_info,
            "optimizer_param_groups": optimizer_groups,
            "normalization_artifacts": normalization_artifacts,
            "checkpoint_registry": checkpoint_registry,
            "throughput": throughput_metadata,
            "prediction_objective": objective_metadata,
            "prediction_setup": prediction_setup_metadata(self.cfg, split_metadata=split_metadata),
            "runtime": {
                "run_dir": str(self.run_dir),
                "output_overwrite": bool(self.cfg.get("output", {}).get("overwrite", False)),
                "splits": split_metadata,
                "normalization_artifacts": normalization_artifacts,
                "checkpoint_registry": checkpoint_registry,
                "throughput": throughput_metadata,
                "teacher_prior": teacher_prior_info,
                "early_stopping": early_stopping_metadata,
                "startup_summary": startup_summary,
                "config_diff": config_diff,
                "pilot_noise_validity": pilot_noise_validity,
                "prediction_objective": objective_metadata,
                "prediction_setup": prediction_setup_metadata(self.cfg, split_metadata=split_metadata),
            },
        }
        with (self.run_dir / "train_log.json").open("w", encoding="utf-8") as f:
            json.dump(train_log, f, indent=2)
        plot_training_curves(history, self.run_dir)
        dump_config(
            final_config_with_runtime(
                self.cfg,
                run_dir=self.run_dir,
                split_metadata=split_metadata,
                normalization_artifacts=normalization_artifacts,
                checkpoint_registry=checkpoint_registry,
                throughput_metadata=throughput_metadata,
                teacher_prior=teacher_prior_info,
                early_stopping=early_stopping_metadata,
                pilot_noise_validity=pilot_noise_validity,
            ),
            self.run_dir / "final_config.yaml",
        )
        return {
            "teacher_metrics": teacher_metrics,
            "early_stopping": early_stopping_metadata,
            "pilot_noise_validity": pilot_noise_validity,
        }
