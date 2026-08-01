from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from kd_sensing.config.io import dump_config
from kd_sensing.engine.objectives.metadata import objective_runtime_metadata
from kd_sensing.engine.run_lineage import run_lineage_metadata
from kd_sensing.engine.run_metadata import prediction_setup_metadata
from kd_sensing.engine.training_metrics import training_outputs_payload
from kd_sensing.utils.runtime_output_layout import output_layout_summary, runtime_scope_metadata_from_config


def write_final_test_metrics(run_dir: str | Path, metrics: dict[str, Any]) -> Path:
    if metrics.get("evaluation_split") != "test":
        raise ValueError("final_test_metrics must declare evaluation_split='test' before publication.")
    selected = metrics.get("selected_checkpoint")
    if not isinstance(selected, dict) or not selected.get("path") or not selected.get("checkpoint_role"):
        raise ValueError("final_test_metrics requires selected checkpoint path and role provenance.")
    target = Path(run_dir) / "final_test_metrics.json"
    _write_json_atomic(target, metrics)
    return target


def final_config_with_runtime(
    cfg: dict,
    *,
    run_dir: Path,
    split_metadata: dict | None = None,
    normalization_artifacts: dict | None = None,
    evaluation_checkpoint: dict | None = None,
    throughput_metadata: dict | None = None,
    final_test_metrics: dict | None = None,
) -> dict:
    final_cfg = deepcopy(cfg)
    runtime = final_cfg.setdefault("runtime", {})
    runtime.update(
        {
            "run_dir": str(run_dir),
            "output_overwrite": bool(cfg.get("output", {}).get("overwrite", False)),
            "prediction_objective": objective_runtime_metadata(),
            "lineage": run_lineage_metadata(cfg),
            "prediction_setup": prediction_setup_metadata(cfg, split_metadata=split_metadata),
        }
    )
    if split_metadata is not None:
        runtime["splits"] = split_metadata
    if normalization_artifacts is not None:
        runtime["normalization_artifacts"] = normalization_artifacts
    if evaluation_checkpoint is not None:
        runtime["evaluation_checkpoint"] = evaluation_checkpoint
    if throughput_metadata is not None:
        runtime["throughput"] = throughput_metadata
    if final_test_metrics is not None:
        runtime["final_test_metrics"] = final_test_metrics
    scope_metadata = runtime_scope_metadata_from_config(cfg)
    if scope_metadata:
        runtime["scene_scope"] = scope_metadata
        runtime["output_scope"] = {**scope_metadata, "run_dir": str(run_dir), "layout": output_layout_summary(run_dir)}
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
        checkpoint_loads: list[dict[str, Any] | None],
        optimizer_groups: list[dict[str, Any]],
        normalization_artifacts: dict | None,
        throughput_metadata: dict | None,
        split_metadata: dict | None,
        startup_summary: dict[str, Any],
        final_test_metrics: dict | None = None,
    ) -> dict[str, Any]:
        np.savez(self.run_dir / "training_outputs.npz", **training_outputs_payload(history, objective_metadata))
        lineage = run_lineage_metadata(self.cfg)
        runtime_scope = runtime_scope_metadata_from_config(self.cfg)
        train_log = {
            **history,
            "data_protocol": deepcopy(self.cfg.get("data_protocol", {})),
            "epoch_logs": epoch_logs,
            "startup_summary": startup_summary,
            "final_test_metrics": final_test_metrics,
            "lineage": lineage,
            "checkpoint_loads": checkpoint_loads,
            "optimizer_param_groups": optimizer_groups,
            "normalization_artifacts": normalization_artifacts,
            "throughput": throughput_metadata,
            "prediction_objective": objective_metadata,
            "prediction_setup": prediction_setup_metadata(self.cfg, split_metadata=split_metadata),
            "scene_scope": runtime_scope,
            "runtime": {
                "run_dir": str(self.run_dir),
                "output_overwrite": bool(self.cfg.get("output", {}).get("overwrite", False)),
                "scene_scope": runtime_scope,
                "output_scope": {**runtime_scope, "run_dir": str(self.run_dir), "layout": output_layout_summary(self.run_dir)},
                "splits": split_metadata,
                "normalization_artifacts": normalization_artifacts,
                "throughput": throughput_metadata,
                "startup_summary": startup_summary,
                "final_test_metrics": final_test_metrics,
                "lineage": lineage,
                "prediction_objective": objective_metadata,
                "prediction_setup": prediction_setup_metadata(self.cfg, split_metadata=split_metadata),
            },
        }
        _write_json_atomic(self.run_dir / "train_log.json", train_log)
        dump_config(
            final_config_with_runtime(
                self.cfg,
                run_dir=self.run_dir,
                split_metadata=split_metadata,
                normalization_artifacts=normalization_artifacts,
                throughput_metadata=throughput_metadata,
                final_test_metrics=final_test_metrics,
            ),
            self.run_dir / "final_config.yaml",
        )
        return {"final_test_metrics": final_test_metrics}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
