## ADDED Requirements

### Requirement: MMW GPS v2 诊断入口收敛到 package CLI
MMW Town GPS-only v2 MUST 将当前运行、绘图和旧诊断对比入口收敛到包内 CLI。项目 MUST 不要求用户通过 `scripts/mmw/visualize_gps_angle_beam_correspondence.py`、`scripts/mmw/visualize_gps_prediction_trajectory.py` 或 `scripts/mmw/visualize_prediction_error_label_distribution.py` 生成 current v2 诊断图。

#### Scenario: package plotter 覆盖 current 图表需求
- **WHEN** 用户运行 `kd-sensing-plot-mmw-town-gps-v2 --results-dir <dir>`
- **THEN** 系统 MUST 生成当前 spec 要求的结构残差图或不可用说明
- **AND** 用户 MUST NOT need to run retired `scripts/mmw/visualize_gps_*` scripts for current documentation or validation

#### Scenario: 旁支脚本退役
- **WHEN** 开发者检查 MMW GPS v2 当前入口、README 和 maintainer index
- **THEN** 文档 MUST 指向 `kd-sensing-mmw-town-gps-v2`、`kd-sensing-plot-mmw-town-gps-v2` 和 `kd-sensing-compare-mmw-town-gps-v2`
- **AND** `scripts/mmw/visualize_gps_angle_beam_correspondence.py`、`scripts/mmw/visualize_gps_prediction_trajectory.py` 和 `scripts/mmw/visualize_prediction_error_label_distribution.py` MUST 不作为 current research diagnostic 入口保留

