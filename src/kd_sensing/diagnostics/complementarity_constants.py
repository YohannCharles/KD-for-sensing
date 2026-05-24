from __future__ import annotations

WEAK_MODALITIES = ("image", "radar", "lidar")
STRONG_MODALITIES = ("gps", "mmwave")
CASE_RESCUE = "strong_wrong_weak_correct_fusion_correct"
CASE_UNUSED_COMPLEMENTARY = "strong_wrong_weak_correct_fusion_wrong"
CASE_NEGATIVE_TRANSFER = "strong_correct_fusion_wrong"
CASE_STRONG_WRONG_FUSION_CORRECT = "strong_wrong_fusion_correct"
CASE_ALL_CORRECT = "all_correct"
CASE_ALL_WRONG = "all_wrong"
CASE_OTHER = "other"
DEFAULT_CASE_FILTERS = (
    "strong_wrong_weak_correct",
    CASE_RESCUE,
    CASE_UNUSED_COMPLEMENTARY,
    CASE_NEGATIVE_TRANSFER,
)

KEY_COLUMNS = ["sample_id", "dataset_index", "horizon_idx", "horizon_name"]
DELTA_COLUMNS = ["delta_ce", "delta_top1", "delta_top3", "delta_dba"]
PATH_COLUMNS = [
    "root_csv",
    "input_beam_path",
    "target_beam_path",
    "image_path",
    "radar_path",
    "gps_path",
    "lidar_path",
    "mmwave_path",
]
METADATA_COLUMNS = ["scene", "scene_id", "scene_slug", "split", *PATH_COLUMNS]
COMMUNICATION_BUCKET_FEATURES = [
    "mmwave_entropy",
    "mmwave_top1_prob",
    "mmwave_top1_top2_margin",
    "mmwave_peak_sharpness",
    "mmwave_total_power",
    "mmwave_peak_drift",
    "range_to_bs",
    "bearing",
    "delta_range",
    "delta_bearing",
    "angular_velocity",
    "gps_jump_magnitude",
    "beam_transition",
    "beam_delta",
]



__all__ = [
    "CASE_ALL_CORRECT",
    "CASE_ALL_WRONG",
    "CASE_NEGATIVE_TRANSFER",
    "CASE_OTHER",
    "CASE_RESCUE",
    "CASE_STRONG_WRONG_FUSION_CORRECT",
    "CASE_UNUSED_COMPLEMENTARY",
    "COMMUNICATION_BUCKET_FEATURES",
    "DEFAULT_CASE_FILTERS",
    "DELTA_COLUMNS",
    "KEY_COLUMNS",
    "METADATA_COLUMNS",
    "PATH_COLUMNS",
    "STRONG_MODALITIES",
    "WEAK_MODALITIES",
]
