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
from kd_sensing.engine.objectives.metadata import objective_runtime_metadata
from kd_sensing.engine.run_metadata import (
    jepa_downstream_metadata,
    prediction_setup_metadata,
    vision_position_baseline_metadata,
)
from kd_sensing.engine.run_lineage import run_lineage_metadata
from kd_sensing.engine.training_metrics import training_outputs_payload
from kd_sensing.engine.training_state import early_stopping_state
from kd_sensing.utils.runtime_output_layout import output_layout_summary, runtime_scope_metadata_from_config
from kd_sensing.utils.plotting import plot_training_curves


def final_config_with_runtime(
    cfg: dict,
    *,
    run_dir: Path,
    primary_model: Any | None = None,
    model: Any | None = None,
    optimizer_groups: list[dict[str, Any]] | None = None,
    split_metadata: dict | None = None,
    normalization_artifacts: dict | None = None,
    checkpoint_registry: dict | None = None,
    throughput_metadata: dict | None = None,
    early_stopping: dict | None = None,
    pilot_noise_validity: dict | None = None,
    final_test_metrics: dict | None = None,
) -> dict:
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
    if early_stopping is not None:
        runtime["early_stopping"] = early_stopping
    if pilot_noise_validity is not None:
        runtime["pilot_noise_validity"] = pilot_noise_validity
    if final_test_metrics is not None:
        runtime["final_test_metrics"] = final_test_metrics
    scene_metadata = scene_metadata_from_config(cfg)
    if scene_metadata:
        runtime["scene"] = scene_metadata
    scope_metadata = runtime_scope_metadata_from_config(cfg)
    if scope_metadata:
        runtime["scene_scope"] = scope_metadata
        runtime["output_scope"] = {
            **scope_metadata,
            "run_dir": str(run_dir),
            "layout": output_layout_summary(run_dir),
        }
    runtime["lineage"] = run_lineage_metadata(cfg)
    runtime["prediction_setup"] = prediction_setup_metadata(cfg, split_metadata=split_metadata)
    runtime_model = primary_model if primary_model is not None else model
    baseline_metadata = vision_position_baseline_metadata(cfg, model=runtime_model)
    if baseline_metadata:
        runtime["baseline"] = baseline_metadata
    jepa_metadata = jepa_downstream_metadata(cfg, model=runtime_model, optimizer_groups=optimizer_groups)
    if jepa_metadata:
        runtime["jepa_downstream"] = jepa_metadata
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
        optimizer_groups: list[dict[str, Any]],
        normalization_artifacts: dict | None,
        checkpoint_registry: dict | None,
        throughput_metadata: dict | None,
        split_metadata: dict | None,
        startup_summary: dict[str, Any],
        config_diff: dict[str, Any] | None,
        csi_debug_records: list[dict[str, Any]],
        best_top1_epoch: int,
        final_test_metrics: dict | None = None,
        primary_model: Any | None = None,
    ) -> dict[str, Any]:
        np.savez(
            self.run_dir / "training_outputs.npz",
            **training_outputs_payload(history, objective_metadata, early_stopping_metric, early_stopping_mode),
        )
        early_stopping_metadata = early_stopping_state(
            metric=early_stopping_metric,
            mode=early_stopping_mode,
            best_value=best_early_stopping_value,
            best_epoch=best_early_stopping_epoch,
            epochs_without_improvement=epochs_without_improvement,
        )
        lineage = run_lineage_metadata(self.cfg)
        baseline_metadata = vision_position_baseline_metadata(self.cfg, model=primary_model)
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
            "final_test_metrics": final_test_metrics,
            "lineage": lineage,
            "checkpoint_loads": checkpoint_loads,
            "optimizer_param_groups": optimizer_groups,
            "normalization_artifacts": normalization_artifacts,
            "checkpoint_registry": checkpoint_registry,
            "throughput": throughput_metadata,
            "prediction_objective": objective_metadata,
            "prediction_setup": prediction_setup_metadata(self.cfg, split_metadata=split_metadata),
            "baseline": baseline_metadata or None,
            "scene_scope": runtime_scope_metadata_from_config(self.cfg),
            "runtime": {
                "run_dir": str(self.run_dir),
                "output_overwrite": bool(self.cfg.get("output", {}).get("overwrite", False)),
                "scene_scope": runtime_scope_metadata_from_config(self.cfg),
                "output_scope": {
                    **runtime_scope_metadata_from_config(self.cfg),
                    "run_dir": str(self.run_dir),
                    "layout": output_layout_summary(self.run_dir),
                },
                "splits": split_metadata,
                "normalization_artifacts": normalization_artifacts,
                "checkpoint_registry": checkpoint_registry,
                "throughput": throughput_metadata,
                "early_stopping": early_stopping_metadata,
                "startup_summary": startup_summary,
                "config_diff": config_diff,
                "pilot_noise_validity": pilot_noise_validity,
                "final_test_metrics": final_test_metrics,
                "lineage": lineage,
                "prediction_objective": objective_metadata,
                "prediction_setup": prediction_setup_metadata(self.cfg, split_metadata=split_metadata),
                "baseline": baseline_metadata or None,
            },
        }
        with (self.run_dir / "train_log.json").open("w", encoding="utf-8") as f:
            json.dump(train_log, f, indent=2)
        plot_training_curves(history, self.run_dir)
        dump_config(
            final_config_with_runtime(
                self.cfg,
                run_dir=self.run_dir,
                primary_model=primary_model,
                optimizer_groups=optimizer_groups,
                split_metadata=split_metadata,
                normalization_artifacts=normalization_artifacts,
                checkpoint_registry=checkpoint_registry,
                throughput_metadata=throughput_metadata,
                early_stopping=early_stopping_metadata,
                pilot_noise_validity=pilot_noise_validity,
                final_test_metrics=final_test_metrics,
            ),
            self.run_dir / "final_config.yaml",
        )
        return {
            "early_stopping": early_stopping_metadata,
            "pilot_noise_validity": pilot_noise_validity,
        }
