## ADDED Requirements

### Requirement: LiDAR legacy model names are removed
LiDAR legacy whole-model 注册名和 feature extractor `MODELS` 注册名 MUST 被 removed guard 拒绝。Current LiDAR canonical 配置 MUST 继续使用 `modular_sequence + lidar_cnn`。

#### Scenario: 请求 LiDAR legacy 注册名
- **WHEN** 用户请求 `lidar_teacher`、`lidar_student`、`lidar_strong`、`lidar_lightweight` 或 `lidar_feature_extractor`
- **THEN** registry MUST 抛出 removed component 错误
- **AND** 错误信息 MUST 建议使用 `modular_sequence + lidar_cnn + single_gru`

#### Scenario: LiDAR canonical 配置仍使用 modular path
- **WHEN** 用户加载 `configs/lidar/strong.yaml`、`configs/lidar/lightweight.yaml` 或 `configs/lidar/supervised.yaml`
- **THEN** 最终配置的 `model.primary.type` MUST 为 `modular_sequence`
- **AND** `model.primary.encoders.lidar.type` MUST 为 `lidar_cnn`

## MODIFIED Requirements

### Requirement: LiDARFeatureExtractor 结构
系统 MUST 提供 `LidarFeatureExtractor`，用于从 LiDAR BEV 序列中提取每个时隙的固定长度特征。该 feature extractor MUST 接收形状为 `(batch, sequence, channels, height, width)` 的 LiDAR BEV 张量，并输出 `(batch, sequence, feature_size)`。该类 MAY 通过 `kd_sensing.models.lidar` 或 `kd_sensing.models` 窄导入暴露，但 MUST NOT 作为 current `MODELS` 注册名暴露。

#### Scenario: LiDAR feature extractor 前向输出
- **WHEN** `LidarFeatureExtractor` 接收形状为 `(B, T, C, H, W)` 的 LiDAR BEV 输入
- **THEN** 输出 MUST 为形状 `(B, T, feature_size)` 的特征张量
- **AND** 输出 feature 维 MUST 等于构造参数 `n_feature` 或 `feature_size`

#### Scenario: LiDAR feature extractor 不作为完整模型注册
- **WHEN** 开发者查看 current `MODELS.list()`
- **THEN** 输出 MUST NOT 包含 `lidar_feature_extractor`
- **AND** 需要配置构建 LiDAR encoder 时 MUST 使用 `ENCODERS` 中的 `lidar_cnn`

## REMOVED Requirements

### Requirement: LiDARTeacher 模型结构
**Reason**: LiDAR strong baseline 已迁移到 `modular_sequence + lidar_cnn`。
**Migration**: 使用 `configs/lidar/strong.yaml` 或 `configs/lidar/supervised.yaml`。

### Requirement: LiDARStudent 模型结构
**Reason**: LiDAR lightweight baseline 已迁移到 `modular_sequence + lidar_cnn`。
**Migration**: 使用 `configs/lidar/lightweight.yaml`。

### Requirement: LiDAR-only 基线配置
**Reason**: 该要求指定通过 `lidar_teacher` / `lidar_student` 构建主模型，已被现有 canonical modular LiDAR requirement 取代。
**Migration**: 使用 `LiDAR canonical 模型配置使用 modular BEV encoder` 要求。
