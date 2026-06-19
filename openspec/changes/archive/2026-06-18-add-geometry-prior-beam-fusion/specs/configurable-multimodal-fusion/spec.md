## ADDED Requirements

### Requirement: Geometry-prior fusion component configuration
Fusion 配置 MUST 支持 opt-in geometry-prior component baseline。该 baseline MUST 通过 `model.primary` 的 encoder/core/head 或明确窄组件字段选择，不得新增 root-level 训练脚本或复制训练循环。

#### Scenario: 配置启用 geometry prior
- **WHEN** 配置声明 `model.primary.geometry_prior.enabled=true` 或等价 opt-in 字段
- **THEN** 系统 MUST 构建 GPS geometry prior 分支和 logit fusion 组件
- **AND** final config MUST 记录 prior input mode、fusion mode、loss mode、teacher guidance 开关和 reliability metadata consumption

#### Scenario: 配置关闭时默认行为不变
- **WHEN** 配置未声明 geometry-prior fusion
- **THEN** 现有 fusion teacher/student、modular_sequence、Image ResNet+GPS 和 JEPA GPS-query baseline 行为 MUST 保持不变
- **AND** batch runtime MUST 不要求 geometry prior 字段存在

### Requirement: Geometry-prior fusion input fields
启用 geometry-prior fusion 的 image+GPS 配置 MUST 使用现有 GPS batch contract 或显式声明的几何特征转换。未启用 LiDAR、radar 或 mmWave 时，系统 MUST 不要求这些模态字段。

#### Scenario: image+GPS geometry prior 配置
- **WHEN** geometry-prior 配置的 modalities 为 `["image", "gps"]`
- **THEN** batch 准备 MUST 提供 image 输入和 GPS 输入
- **AND** batch 准备 MUST 不要求 radar、LiDAR、mmWave 或 CSI 输入

#### Scenario: GPS feature mode 可审计
- **WHEN** geometry-prior 分支消费 GPS-Rel-Polar、relative Cartesian 或 calibrated angle feature
- **THEN** run metadata MUST 记录 feature mode、scaler/normalization artifact、calibration source 和 history/source window
- **AND** 训练、验证和评估 MUST 使用相同的 feature contract 或在 comparability warnings 中标记 mismatch

### Requirement: Geometry-prior canonical configs
项目 MUST 提供 H5/G2/F1、scene32-34、future=1、seed=17 的 geometry-prior smoke 和 strict comparison 配置。配置 MUST 覆盖 prior-only、image-only control、logit fusion、DBA-aware loss 和 teacher-guided ablation。

#### Scenario: strict config 字段齐全
- **WHEN** 开发者加载 geometry-prior strict 配置或 manifest
- **THEN** 配置 MUST 声明 history_window、gps_input_source_window、image_history_window、prediction_horizon、scene_set、seed、distance_metric 和 beam_label_space
- **AND** strict comparison 聚合 MUST 在这些字段不一致时拒绝 claim upgrade

#### Scenario: ablation 配置不互相污染
- **WHEN** prior-only、fusion、DBA-aware loss 或 teacher-guided ablation 被分别运行
- **THEN** output run_name、experiment ablation、model_group 和 metadata MUST 能区分这些配置
- **AND** summary 表 MUST 不把 ablation 指标混成同一 model row
