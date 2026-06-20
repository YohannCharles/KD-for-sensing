## ADDED Requirements

### Requirement: Legacy model registry names are retired with migration guards
项目 MUST 将已退役的 legacy model、encoder、core 和 head 注册名登记为 removed guard，而不是继续作为 current 可构建组件暴露。removed guard MUST 区分未知名称和已退役名称，并 MUST 给出明确迁移目标。

#### Scenario: 旧整模型注册名被拒绝并给出迁移目标
- **WHEN** 用户通过 `MODELS.build()` 请求 `radar_strong`、`gps_lightweight`、`mmwave_strong`、`fusion_lightweight` 或其它本 change 退役的旧整模型注册名
- **THEN** 系统 MUST 抛出 removed component 错误
- **AND** 错误信息 MUST 包含请求名称、registry 名称和 `modular_sequence` 迁移目标

#### Scenario: 旧别名被拒绝并指向 canonical 名称
- **WHEN** 用户请求 `modular_sequence_model`、`gps_only_neural_baseline`、`jepa_token_transformer` 或 `safe_residual_reranker`
- **THEN** 系统 MUST 抛出 removed component 错误
- **AND** 错误信息 MUST 指向对应 canonical 名称或配置路径

#### Scenario: feature extractor 不作为完整模型列出
- **WHEN** 默认组件导入完成后开发者查看 `MODELS.list()`
- **THEN** 输出 MUST NOT 包含 `radar_feature_extractor`、`lidar_feature_extractor` 或 `mmwave_feature_extractor`
- **AND** 对应 feature extractor 类 MAY 继续通过窄模块导入或由 encoder 组件内部复用

#### Scenario: current registry discovery 只列当前入口
- **WHEN** 文档、架构摘要或架构边界测试检查 current registry surface
- **THEN** current model/encoder/core/head 清单 MUST 不把 removed guard 名称展示为可推荐入口
- **AND** removed 名称 MAY 出现在退役边界或 migration table 中

## MODIFIED Requirements

### Requirement: 可扩展模型和模态
新增普通 strong、lightweight、backbone、head、radar、GPS、LiDAR、mmWave、CSI 或 fusion baseline 时，开发者 MUST 优先通过 `modular_sequence` 配置、virtual recipe 或新增 `ENCODERS`、`PROJECTORS`、`REPRESENTATION_CORES`、`HEADS` 子组件扩展系统，而不需要复制训练脚本或修改训练循环主体。新增完整 `MODELS` 注册名 MUST 作为 whole-model exception 或 workflow/paper reproduction 在 OpenSpec artifact 中说明原因。

#### Scenario: 新增 image-only lightweight baseline
- **WHEN** 开发者实现一个新的 image-only lightweight baseline
- **THEN** 用户 MUST 能通过 `model.primary.type: modular_sequence` 和 `model.primary.encoders.image.type` 选择该 baseline
- **AND** 实现 MUST 复用现有 image-only 训练流程

#### Scenario: 新增多模态 fusion baseline
- **WHEN** 开发者实现一个新的 image+radar 或 radar+GPS fusion baseline
- **THEN** 用户 MUST 能通过 `modular_sequence` 的 encoders、projectors、representation core 和 heads 配置表达该 baseline
- **AND** 实现 MUST 不新增完整 `MODELS` 注册名，除非 active OpenSpec design 记录 whole-model exception 理由

#### Scenario: 新增 radar-only baseline
- **WHEN** 开发者实现 radar-only strong 或 lightweight baseline
- **THEN** 用户 MUST 能通过 `model.primary.encoders.radar.type: radar_cnn` 或新的 radar encoder 组件选择该行为
- **AND** 模型输出 MUST 继续兼容 `ModelOutput` 适配和 beam prediction loss/metric

## REMOVED Requirements

### Requirement: LiDAR 组件注册
**Reason**: 该要求仍以 `lidar_teacher`、`lidar_student` 和 feature extractor `MODELS` 注册为当前入口，已与当前 `modular_sequence + lidar_cnn` 主路径冲突。
**Migration**: 使用 `model.primary.type: modular_sequence`、`encoders.lidar.type: lidar_cnn`；`LidarFeatureExtractor` 只作为窄类或 encoder 内部实现保留。

### Requirement: mmWave 组件注册
**Reason**: 该要求仍以 `mmwave_teacher`、`mmwave_student` 和 `mmwave_feature_extractor` `MODELS` 注册为当前入口，已与当前 `modular_sequence + mmwave_mlp` 主路径冲突。
**Migration**: 使用 `model.primary.type: modular_sequence`、`encoders.mmwave.type: mmwave_mlp`；`MmWaveFeatureExtractor` 只作为窄类或 encoder 内部实现保留。
