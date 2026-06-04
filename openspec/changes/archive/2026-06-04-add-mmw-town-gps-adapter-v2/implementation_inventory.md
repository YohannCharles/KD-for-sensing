## MMW Town GPS Adapter v2 Implementation Inventory

Captured at implementation start for OpenSpec change `add-mmw-town-gps-adapter-v2`.

### Label Mapping And Calibration Artifacts

- `outputs/analysis/mmw_town_label_distribution/mapping_enabled/beam_label_mapping_midpoint_31_32.json`
- `outputs/analysis/mmw_town_label_distribution/source_other_three_calibration.json`
- `outputs/analysis/mmw_town_label_distribution/target_adapt_beambench_calibration.json`
- `outputs/analysis/mmw_town_label_distribution/target_adapt_beambench_dba_calibration.json`
- `outputs/analysis/mmw_town_label_distribution/within_scene_train_calibration.json`

### Previous Prediction And Metric Artifacts

- `outputs/gps_coarse_anchor/source_other_three_auto_mapping_dba/*/predictions.csv`
- `outputs/gps_coarse_anchor/source_other_three_auto_mapping_dba/*/metrics.json`
- `outputs/gps_coarse_anchor/source_other_three_auto_mapping_dba/summary.json`
- `outputs/gps_coarse_anchor/target_adapt_beambench_dba/*/predictions.csv`
- `outputs/gps_coarse_anchor/target_adapt_beambench_dba/*/metrics.json`
- `outputs/gps_coarse_anchor/target_adapt_beambench_dba/summary.json`
- `outputs/gps_coarse_anchor/within_scene_upper_bound_all_train/*/predictions.csv`
- `outputs/gps_coarse_anchor/within_scene_upper_bound_all_train/*/metrics.json`
- `outputs/gps_coarse_anchor/within_scene_upper_bound_all_train/summary.json`
- `outputs/analysis/mmw_town_label_distribution/<label_space>/label_distribution/*summary.json`
- `outputs/analysis/mmw_town_label_distribution/<label_space>/prediction_error_label_distribution/*summary.json`

### DBA Implementation

- `src/kd_sensing/evaluation/metrics.py:31` defines `calculate_dba_score()`.
- `src/kd_sensing/evaluation/metrics.py:67` defines `_circular_class_distance()`, which is the existing circular DBA distance basis.

### MMW Data CSV And Split Metadata

Preferred split tag: `l5p3_group_safe`.

- `dataset/MMW/sunny/Prepared/Town10_crossroad_seed24/manifests/frame_manifest.csv`
- `dataset/MMW/sunny/Prepared/Town10_crossroad_seed24/splits/l5p3_group_safe/train.csv`
- `dataset/MMW/sunny/Prepared/Town10_crossroad_seed24/splits/l5p3_group_safe/test.csv`
- `dataset/MMW/sunny/Prepared/Town10_crossroad_seed24/splits/l5p3_group_safe/all_sequences.csv`
- `dataset/MMW/sunny/Prepared/Town10_crossroad_seed24/splits/l5p3_group_safe/split_metadata.json`
- `dataset/MMW/sunny/Prepared/Town10_skybridge_seed24/manifests/frame_manifest.csv`
- `dataset/MMW/sunny/Prepared/Town10_skybridge_seed24/splits/l5p3_group_safe/train.csv`
- `dataset/MMW/sunny/Prepared/Town10_skybridge_seed24/splits/l5p3_group_safe/test.csv`
- `dataset/MMW/sunny/Prepared/Town10_skybridge_seed24/splits/l5p3_group_safe/all_sequences.csv`
- `dataset/MMW/sunny/Prepared/Town10_skybridge_seed24/splits/l5p3_group_safe/split_metadata.json`
- `dataset/MMW/sunny/Prepared/Town10_curvyroad_seed42/manifests/frame_manifest.csv`
- `dataset/MMW/sunny/Prepared/Town10_curvyroad_seed42/splits/l5p3_group_safe/train.csv`
- `dataset/MMW/sunny/Prepared/Town10_curvyroad_seed42/splits/l5p3_group_safe/test.csv`
- `dataset/MMW/sunny/Prepared/Town10_curvyroad_seed42/splits/l5p3_group_safe/all_sequences.csv`
- `dataset/MMW/sunny/Prepared/Town10_curvyroad_seed42/splits/l5p3_group_safe/split_metadata.json`
- `dataset/MMW/sunny/Prepared/Town10_Hroad_seed42/manifests/frame_manifest.csv`
- `dataset/MMW/sunny/Prepared/Town10_Hroad_seed42/splits/l5p3_group_safe/train.csv`
- `dataset/MMW/sunny/Prepared/Town10_Hroad_seed42/splits/l5p3_group_safe/test.csv`
- `dataset/MMW/sunny/Prepared/Town10_Hroad_seed42/splits/l5p3_group_safe/all_sequences.csv`
- `dataset/MMW/sunny/Prepared/Town10_Hroad_seed42/splits/l5p3_group_safe/split_metadata.json`

### Expected Source Changes

- `configs/mmw_town_gps_adapter_v2.yaml`
- `src/kd_sensing/evaluation/metrics.py`
- `src/kd_sensing/losses/circular.py`
- `src/kd_sensing/models/mmw_town_gps_v2.py`
- `src/kd_sensing/engine/mmw_town_gps_v2.py`
- `src/kd_sensing/cli/mmw_town_gps_v2.py`
- `src/kd_sensing/cli/plot_mmw_town_gps_v2.py`
- `src/kd_sensing/cli/compare_mmw_town_gps_v2.py`
- `pyproject.toml`
- `README.md`
- `tests/test_circular_metrics.py`
- `tests/test_mmw_town_gps_adapter_v2.py`
