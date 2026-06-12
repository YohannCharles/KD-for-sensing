"""BeamBench baseline reproduction helpers."""

from __future__ import annotations

from kd_sensing.baselines.beambench.dataset_check import check_dataset, resolve_csv_fields
from kd_sensing.baselines.beambench.image_ae_gps import (
    BeamBenchImageAEGPSDataset,
    BeamBenchImageAEGPSDirectModel,
    BeamBenchImageAEGPSFeatureDataset,
    BeamBenchImageOnlyDataset,
    ImageAEGPSDirectTrainingConfig,
    run_image_ae_gps_paper_split_training,
    run_image_ae_gps_training,
)
from kd_sensing.baselines.beambench.metrics import (
    beambench_metric_summary_from_logits,
    official_dba_score,
    official_topk_accuracy,
)
from kd_sensing.baselines.beambench.mock import create_mock_dataset
from kd_sensing.baselines.beambench.official import (
    audit_official_repository,
    plan_official_classical_evaluation,
    plan_official_evaluation,
)
from kd_sensing.baselines.beambench.pipeline import evaluate_checkpoint, train_mock_baseline

__all__ = [
    "audit_official_repository",
    "beambench_metric_summary_from_logits",
    "BeamBenchImageAEGPSDataset",
    "BeamBenchImageAEGPSDirectModel",
    "BeamBenchImageAEGPSFeatureDataset",
    "BeamBenchImageOnlyDataset",
    "check_dataset",
    "create_mock_dataset",
    "evaluate_checkpoint",
    "ImageAEGPSDirectTrainingConfig",
    "official_dba_score",
    "official_topk_accuracy",
    "plan_official_evaluation",
    "plan_official_classical_evaluation",
    "resolve_csv_fields",
    "run_image_ae_gps_paper_split_training",
    "run_image_ae_gps_training",
    "train_mock_baseline",
]
