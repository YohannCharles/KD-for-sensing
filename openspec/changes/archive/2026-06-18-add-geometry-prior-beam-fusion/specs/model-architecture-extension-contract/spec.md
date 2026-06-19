## ADDED Requirements

### Requirement: Geometry-prior route classification
Geometry-prior beam fusion MUST 默认归类为 component baseline。实现 MUST 优先通过现有 `modular_sequence`、encoder/projector/core/head 或窄 fusion component 表达。

#### Scenario: 使用 component baseline 路径
- **WHEN** 开发者实现 GPS geometry prior、logit fusion 或 DBA-aware head
- **THEN** 实现 MUST 落在可注册的窄组件、loss/objective helper 或 diagnostics helper 中
- **AND** 系统 MUST 不新增完整 `MODELS.register(...)` 例外，除非 design 另行记录不可组合原因

#### Scenario: whole-model exception 需要明确理由
- **WHEN** geometry-prior 实现需要新增完整模型注册名
- **THEN** OpenSpec design 或 spec MUST 说明为什么不能使用 component baseline
- **AND** tasks MUST 包含 registry build、synthetic forward、ModelOutput adaptation、metadata 和 architecture boundary tests

### Requirement: BEV-Fusion reproduction boundary
完整 BEV-Fusion 论文复现 MUST 作为 workflow/paper reproduction 处理，而不是混入当前 geometry-prior component baseline。

#### Scenario: BEV-lite component 允许
- **WHEN** 实现只加入 GPS prior map、angle prior 或轻量 spatial prior token
- **THEN** 系统 MAY 将其作为 component baseline 实现
- **AND** metadata MUST 标记为 geometry-prior 或 BEV-lite，而不是完整 BEV-Fusion reproduction

#### Scenario: 完整论文复现走 workflow 路径
- **WHEN** 实现包含 camera-to-BEV、LiDAR/radar/GPS BEV、多阶段 preprocessing、论文 Table 复现或专用 feature cache
- **THEN** 系统 MUST 将其归类为 workflow/paper reproduction
- **AND** 入口 MUST 位于包内 CLI 或 `src/kd_sensing/baselines/<family>/`，不得新增旧式根脚本

### Requirement: Geometry-prior training metadata
Geometry-prior baseline MUST 写出可审计训练策略 metadata，覆盖 geometry prior、fusion、loss、teacher guidance 和 curriculum。

#### Scenario: metadata 最小字段
- **WHEN** geometry-prior model 构建或训练完成
- **THEN** metadata MUST 包含 model_group、architecture category、enabled modalities、geometry prior mode、fusion mode、loss mode、teacher guidance mode、curriculum mode 和 reliability metadata consumption
- **AND** 缺少这些字段 MUST 被 focused tests 或 architecture boundary tests 捕获

#### Scenario: baseline comparability metadata
- **WHEN** geometry-prior candidate 与 Image ResNet+GPS 或 JEPA GPS-query baseline 聚合比较
- **THEN** metadata MUST 声明 split、sample_count、metric_profile、normalization artifact、difficulty digest、history window、GPS source window、prediction horizon、scene set、seed、distance metric 和 beam label space
- **AND** 任一 strict 字段 mismatch MUST 阻止 claim upgrade
