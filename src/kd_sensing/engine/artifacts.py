from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.config.io import dump_config
from kd_sensing.config.lidar_normalization import canonicalize_lidar_normalization_config
from kd_sensing.data.difficulty import runtime_difficulty_metadata
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
    difficulty_metadata = runtime_difficulty_metadata(cfg)
    if difficulty_metadata is not None:
        runtime["difficulty"] = difficulty_metadata
    runtime_model = primary_model if primary_model is not None else model
    baseline_metadata = vision_position_baseline_metadata(cfg, model=runtime_model)
    if baseline_metadata:
        runtime["baseline"] = baseline_metadata
    jepa_metadata = jepa_downstream_metadata(cfg, model=runtime_model, optimizer_groups=optimizer_groups)
    if jepa_metadata:
        runtime["jepa_downstream"] = jepa_metadata
    physics_metadata = physics_informed_runtime_metadata(cfg, model=runtime_model)
    if physics_metadata:
        runtime["physics_informed"] = physics_metadata
    return final_cfg


def physics_informed_runtime_metadata(cfg: dict, model: Any | None = None) -> dict[str, Any]:
    primary_cfg = cfg.get("model", {}).get("primary", {}) if isinstance(cfg.get("model"), dict) else {}
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg.get("loss"), dict) else {}
    physics_cfg = loss_cfg.get("physics", {}) if isinstance(loss_cfg.get("physics"), dict) else {}
    if primary_cfg.get("type") != "pinn_multimodal_beam" and not physics_cfg:
        return {}
    model_metadata = {}
    if model is not None and hasattr(model, "training_strategy_metadata"):
        raw = model.training_strategy_metadata()
        model_metadata = raw if isinstance(raw, dict) else {}
    modalities = list(primary_cfg.get("modalities") or cfg.get("model", {}).get("modalities") or model_metadata.get("enabled_modalities", []))
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    dataset_cfg = data_cfg.get("dataset", {}) if isinstance(data_cfg.get("dataset"), dict) else {}
    csi_input_mode = str(data_cfg.get("csi_input_mode", dataset_cfg.get("csi_input_mode", "none")))
    used_csi = "csi" in modalities and csi_input_mode != "none"
    weights = {
        key: value.get("weight", 0.0)
        for key, value in physics_cfg.items()
        if isinstance(value, dict) and "weight" in value
    }
    used_path = bool(weights.get("path_consistency", 0.0))
    used_beam_power = bool(weights.get("beam_power_distribution", 0.0))
    frontend_cfg = primary_cfg.get("frontend", {}) if isinstance(primary_cfg.get("frontend"), dict) else {}
    channel_scope = model_metadata.get("channel_target_scope")
    if not channel_scope:
        channel_scope = "narrowband_array_channel" if int(primary_cfg.get("num_subcarriers", 0) or 0) == 1 else "array_channel"
    formal_eligible = bool(
        model_metadata.get("formal_experiment_eligible", frontend_cfg.get("formal_experiment_eligible", True))
    ) and csi_input_mode != "oracle_full"
    sensitive = {
        "used_csi_as_input": bool(used_csi),
        "used_current_full_csi_as_input": csi_input_mode == "oracle_full",
        "used_path_label_for_training": used_path,
        "used_beam_power_for_training": used_beam_power,
        "used_target_physical_oracle": bool(physics_cfg.get("used_target_physical_oracle", False)),
    }
    eligible = not any(sensitive.values())
    return {
        "enabled": bool(physics_cfg.get("enabled", primary_cfg.get("type") == "pinn_multimodal_beam")),
        "enabled_modalities": modalities,
        "csi_input_mode": csi_input_mode,
        "oracle_upper_bound": csi_input_mode == "oracle_full",
        "physics_loss_weights": weights,
        "array_type": primary_cfg.get("array_type", model_metadata.get("array_type", "ula")),
        "codebook_source": primary_cfg.get("codebook_source", model_metadata.get("codebook_source", "ula_dft_fallback")),
        "num_subcarriers": int(primary_cfg.get("num_subcarriers", 0) or 0),
        "num_antennas": int(primary_cfg.get("num_antennas", 0) or 0),
        "num_paths": int(primary_cfg.get("num_paths", 0) or 0),
        "frontend_type": model_metadata.get("frontend_type", frontend_cfg.get("type", "stats")),
        "tokenizers": model_metadata.get("tokenizers", {}),
        "shared_transformer_layers": model_metadata.get("shared_transformer_layers", 0),
        "hidden_dim": model_metadata.get("hidden_dim", primary_cfg.get("hidden_dim")),
        "formal_experiment_eligible": formal_eligible,
        "channel_target_scope": channel_scope,
        "main_conclusion_eligible": eligible and formal_eligible,
        "eligibility_reason": (
            ""
            if eligible and formal_eligible
            else "debug_or_physics_oracle_or_sensitive_supervision_enabled"
        ),
        **sensitive,
    }


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
            "difficulty": runtime_difficulty_metadata(self.cfg),
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
                "difficulty": runtime_difficulty_metadata(self.cfg),
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
