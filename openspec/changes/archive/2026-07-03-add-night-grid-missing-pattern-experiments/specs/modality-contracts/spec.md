## ADDED Requirements

### Requirement: missing pattern 分类 helper
系统 MUST 提供统一 missing pattern API 覆盖标准 mask、pattern name 反查、标准 pattern 列表和 pattern 分类。标准四模态顺序 MUST 为 `["gps", "image", "radar", "lidar"]`。

#### Scenario: weak single modality 分类
- **WHEN** 调用 `is_weak_single_modality_pattern("radar_only")` 或 `is_weak_single_modality_pattern("lidar_only")`
- **THEN** 系统 MUST 返回 true
- **AND** 对 `gps_only`、`image_only`、`missing_gps` MUST 返回 false

#### Scenario: sensing only 分类
- **WHEN** 调用 `is_sensing_only_pattern` 判断 `image_only`、`radar_only`、`lidar_only`、`missing_gps` 或 `non_gps_only`
- **THEN** 系统 MUST 返回 true
- **AND** 对 `gps_only` MUST 返回 false

#### Scenario: 标准 pattern 列表包含聚合项
- **WHEN** 调用 `list_standard_missing_patterns(include_avg=True)`
- **THEN** 返回列表 MUST 包含 `avg_missing`
- **AND** `avg_missing` MUST 被识别为聚合项而不是可直接 forward 的 mask
