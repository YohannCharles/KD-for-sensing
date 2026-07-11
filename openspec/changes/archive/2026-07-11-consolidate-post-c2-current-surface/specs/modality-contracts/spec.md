## MODIFIED Requirements

### Requirement: Image observability metadata 字段
模态契约或等价中心化 helper MUST 定义通用 image observability difficulty metadata 字段语义。字段至少 MUST 覆盖 `image_valid_mask`、`image_observability_score`、`image_dropout_mask`、`image_burst_dropout_mask`、`image_degradation_metadata`、corruption type、severity、seed 和 frame range；字段 MUST 不依赖已退役 Scenario-D condition id。

#### Scenario: 查询 image observability metadata
- **WHEN** 开发者查询 image modality 的 difficulty metadata fields
- **THEN** 系统 MUST 返回 image valid mask、observability score、dropout/burst masks 和 degradation metadata 的字段说明
- **AND** 这些字段 MUST 被标记为输入 reliability metadata，而不是 target supervision 或辅助标签

#### Scenario: metadata 字段不创建伪模态
- **WHEN** 配置启用通用 image degradation 或 missing difficulty
- **THEN** affected modality MUST 仍标准化为 canonical `image`
- **AND** 系统 MUST 拒绝 `image_hard`、`missing_image_modality` 或其它伪模态名称

### Requirement: Reliability metadata 进入 batch 输入映射
训练和评估 batch 输入映射 MUST 能将通用 image/GPS reliability metadata 传递给显式支持的 current 模型，同时保持不支持该 metadata 的模型兼容。metadata 传递 MUST 不要求每个 difficulty condition 新增专用模型输入分支，也 MUST 不保留 Scenario-D/GPS-query benchmark 专属条件映射。

#### Scenario: observability-aware 模型接收 metadata
- **WHEN** current 模型配置声明需要 observability-aware fusion
- **THEN** batch 准备 MUST 向模型 forward 提供其声明的 image observability 和 GPS reliability metadata
- **AND** 缺少字段时 MUST 抛出清晰错误或记录配置声明的 fallback warning

#### Scenario: 普通 baseline 忽略 metadata
- **WHEN** standard Image ResNet+GPS 或其它 baseline 不声明 reliability metadata 输入
- **THEN** batch 准备 MUST 允许其忽略通用 reliability metadata
- **AND** run comparability metadata MUST 记录该模型是否消费 reliability metadata
