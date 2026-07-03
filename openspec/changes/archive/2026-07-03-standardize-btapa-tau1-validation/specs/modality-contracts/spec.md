## ADDED Requirements

### Requirement: 四模态 missing pattern 标准构造
系统 MUST 提供统一的四模态 missing pattern 构造 helper，用于训练评估、复评、BTAPA 分析和 summary。标准输出名称 MUST 使用 `gps`、`image`、`radar`、`lidar`，标准四模态顺序 MUST 为 `["gps", "image", "radar", "lidar"]`，并 MUST 支持常见大小写和旧字段 alias 映射到 canonical 名称。

#### Scenario: 查询标准 pattern mask
- **WHEN** 调用方以标准四模态顺序请求 `radar_only`
- **THEN** 系统 MUST 返回 `[0, 0, 1, 0]`
- **AND** `full`、`missing_gps`、`missing_image`、`missing_radar`、`missing_lidar`、`gps_only`、`image_only`、`radar_only` 和 `lidar_only` MUST 使用同一套构造逻辑

#### Scenario: 保留 missing_gps 与 non_gps_only 两个名称
- **WHEN** 当前四模态设置同时请求 `missing_gps` 和 `non_gps_only`
- **THEN** 两者 MAY 映射到相同 mask `[0, 1, 1, 1]`
- **AND** 输出表 MUST 保留两个 pattern 名称，避免旧结果读取和横向比较失败

#### Scenario: avg_missing 不是直接 mask
- **WHEN** 调用方请求 `avg_missing`
- **THEN** 系统 MUST 将其识别为聚合 pattern
- **AND** 系统 MUST 不把 `avg_missing` 当作可直接 forward 的单个 modality mask

#### Scenario: alias 标准化
- **WHEN** 调用方传入 `GPS`、`RGB`、`rad` 或大小写不同的模态名称
- **THEN** 系统 MUST 标准化为 `gps`、`image`、`radar` 或 `lidar`
- **AND** 未知或重复模态 MUST 抛出清晰错误
