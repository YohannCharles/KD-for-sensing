from kd_sensing.data.difficulty.operators.gps import (
    GpsCleanOperator,
    GpsCumulativeDriftOperator,
    GpsDistractorOperator,
    GpsGaussianJitterOperator,
    GpsMissingOperator,
    GpsTemporalDelayOperator,
    ScenarioCAsyncPositionFeedbackOperator,
)
from kd_sensing.data.difficulty.operators.image import (
    ImageCleanOperator,
    ImageFogRainOperator,
    ImageMotionBlurOperator,
    ImageNightOperator,
    ImageObservabilityTransform,
    ImageOcclusionOperator,
    PredictiveJepaRobustnessOperator,
)
from kd_sensing.data.difficulty.operators.modality import ModalityMissingOperator, RandomModalityDropoutOperator
from kd_sensing.data.difficulty.operators.temporal import TemporalMissingOperator
from kd_sensing.registries import DIFFICULTY_OPERATORS


def _register_defaults() -> None:
    for name, component in {
        "gps_clean": GpsCleanOperator,
        "gps_gaussian_jitter": GpsGaussianJitterOperator,
        "gps_cumulative_drift": GpsCumulativeDriftOperator,
        "gps_missing": GpsMissingOperator,
        "gps_dropout": GpsMissingOperator,
        "gps_distractor": GpsDistractorOperator,
        "temporal_delay": GpsTemporalDelayOperator,
        "gps_temporal_delay": GpsTemporalDelayOperator,
        "sampling_rate_mismatch": GpsTemporalDelayOperator,
        "gps_low_rate_stride": GpsTemporalDelayOperator,
        "gps_timestamp_delay": ScenarioCAsyncPositionFeedbackOperator,
        "scenario_c": ScenarioCAsyncPositionFeedbackOperator,
        "scenario_c_async_position_feedback": ScenarioCAsyncPositionFeedbackOperator,
        "image_clean": ImageCleanOperator,
        "image_observability": ImageObservabilityTransform,
        "scenario_d_image_observability": ImageObservabilityTransform,
        "image_fog_rain": ImageFogRainOperator,
        "image_night": ImageNightOperator,
        "image_occlusion": ImageOcclusionOperator,
        "image_motion_blur": ImageMotionBlurOperator,
        "predictive_jepa_robustness": PredictiveJepaRobustnessOperator,
        "modality_missing": ModalityMissingOperator,
        "modality_dropout": ModalityMissingOperator,
        "modality_unavailable": ModalityMissingOperator,
        "amber_lite_modality_dropout": ModalityMissingOperator,
        "random_modality_dropout": RandomModalityDropoutOperator,
        "temporal_missing": TemporalMissingOperator,
    }.items():
        DIFFICULTY_OPERATORS.register(name, force=True)(component)


_register_defaults()


__all__ = [
    "GpsCleanOperator",
    "GpsCumulativeDriftOperator",
    "GpsDistractorOperator",
    "GpsGaussianJitterOperator",
    "GpsMissingOperator",
    "GpsTemporalDelayOperator",
    "ImageCleanOperator",
    "ImageFogRainOperator",
    "ImageMotionBlurOperator",
    "ImageNightOperator",
    "ImageObservabilityTransform",
    "ModalityMissingOperator",
    "RandomModalityDropoutOperator",
    "TemporalMissingOperator",
    "ImageOcclusionOperator",
    "PredictiveJepaRobustnessOperator",
    "ScenarioCAsyncPositionFeedbackOperator",
]
