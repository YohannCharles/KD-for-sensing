## ADDED Requirements

### Requirement: Image observability metadata 字段
模态契约或等价中心化 helper MUST 定义 image observability 相关 difficulty metadata 字段语义。字段至少 MUST 覆盖 `image_valid_mask`、`image_observability_score`、`image_dropout_mask`、`image_burst_dropout_mask`、`image_degradation_metadata`、corruption type、severity、seed 和 frame range。

#### Scenario: 查询 image observability metadata
- **WHEN** 开发者查询 image modality 的 difficulty metadata fields
- **THEN** 系统 MUST 返回 image valid mask、observability score、dropout/burst masks 和 degradation metadata 的字段说明
- **AND** 这些字段 MUST 被标记为输入 reliability metadata，而不是 target supervision 或辅助标签

#### Scenario: metadata 字段不创建伪模态
- **WHEN** 配置启用 Scenario D image observability difficulty
- **THEN** affected modality MUST 仍标准化为 canonical `image`
- **AND** 系统 MUST 拒绝 `image_hard`、`missing_image_modality` 或其它伪模态名称

### Requirement: Reliability metadata 进入 batch 输入映射
训练、评估和 benchmark batch 输入映射 MUST 能将 image/GPS reliability metadata 传递给显式支持的模型，同时保持不支持该 metadata 的模型兼容。metadata 传递 MUST 不要求每个 difficulty condition 新增专用模型输入分支。

#### Scenario: observability-aware 模型接收 metadata
- **WHEN** 模型配置声明需要 observability-aware fusion
- **THEN** batch 准备 MUST 向模型 forward 提供 image observability 和 GPS reliability metadata
- **AND** 缺少字段时 MUST 抛出清晰错误或记录配置声明的 fallback warning

#### Scenario: legacy-compatible baseline 忽略 metadata
- **WHEN** standard CNN+GPS 或 Image-AE+GPS baseline 不声明 reliability metadata 输入
- **THEN** batch 准备 MUST 允许其忽略 Scenario D metadata
- **AND** benchmark comparability metadata MUST 记录该模型未消费 reliability metadata
