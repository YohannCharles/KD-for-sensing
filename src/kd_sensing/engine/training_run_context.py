from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingRunContext:
    cfg: dict[str, Any]
    objective: str
    objective_metadata: dict[str, Any]
    training_cfg: dict[str, Any]
    early_stopping_metric: str
    early_stopping_mode: str
    run_dir: Path
    artifact_writer: Any
    dataloaders: dict[str, Any]
    split_metadata: dict[str, Any]
    normalization_artifacts: dict[str, Any]
    device: Any
    throughput_metadata: dict[str, Any]
    resolved_cfg: dict[str, Any]
    config_diff: dict[str, Any]
    non_blocking: bool
    amp_enabled: bool
    amp_dtype: Any
    task: str
    model_cfg: dict[str, Any]
    num_pred: int
    num_classes: int
    seq_length: int
    primary_model: Any = None
    state: Any = None
    task_criterion: Any = None
    optimizer: Any = None
    scheduler: Any = None
    optimizer_groups: Any = None
    startup_summary: dict[str, Any] | None = None
    health_tracker: Any = None
    csi_debug_records: list[dict[str, Any]] = field(default_factory=list)
    grad_scaler: Any = None
    extension_context: Any = None
    extensions: list[Any] = field(default_factory=list)
    extension_states: list[Any] = field(default_factory=list)
    recorder: Any = None
    checkpoint_manager: Any = None
    batch_runner: Any = None
    tensorboard_writer: Any = None
    progress_enabled: bool = True
    total_epochs: int = 0
    early_stopping_min_epoch: int = 0
    model_selection_enabled: bool = False
    validation_loader: Any = None
    validation_split_name: str = "none"
    final_test_metrics: dict[str, Any] | None = None
    final_test_checkpoint_load: dict[str, Any] | None = None
    final_artifacts: dict[str, Any] | None = None
