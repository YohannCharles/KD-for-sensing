## ADDED Requirements

### Requirement: 新增 baseline 默认使用模块化路径
新增普通 supervised/adaptation baseline MUST 默认通过 `modular_sequence` 及其子组件配置表达。若 baseline 的变化只涉及模态 encoder、投影、时序/融合 core 或 task head，系统 MUST 不要求新增完整模型注册名。

#### Scenario: 新增 baseline 只替换 encoder
- **WHEN** 开发者新增一个 ResNet、AE、JEPA、CSI 或其它模态 encoder 对照
- **THEN** 该实现 MUST 能作为 `model.primary.encoders.<modality>.type` 被 `modular_sequence` 构建
- **AND** 训练循环 MUST 不需要为该 encoder 新增专用分支

#### Scenario: 新增 baseline 只替换 core
- **WHEN** 开发者新增一个 fusion、snapshot、query 或 temporal core 对照
- **THEN** 该实现 MUST 能作为 `model.primary.representation_core.type` 或等价模块化子组件被选择
- **AND** core MUST 不直接读取 dataset 字段或执行模态特定预处理

### Requirement: 模块化模型暴露可审计 metadata
`ModularSequenceModel` MUST 汇总启用模态、encoder、projector、representation core、head、conditioned encoder 和 reliability metadata 消费信息。新增子组件若提供 `training_strategy_metadata()`，模块化模型 MUST 将其纳入聚合 metadata。

#### Scenario: 组件 metadata 被聚合
- **WHEN** `modular_sequence` 使用提供 `training_strategy_metadata()` 的 image encoder 或 representation core
- **THEN** 模型 metadata MUST 包含该组件声明的关键训练策略字段
- **AND** run metadata 或 startup summary MUST 能区分 checkpoint reuse、freeze policy 和 pooling/fusion 策略

#### Scenario: reliability metadata 消费被记录
- **WHEN** 模块化 image encoder、fusion core 或 adapter 声明消费 observability/reliability metadata
- **THEN** 模块化模型 metadata MUST 标记该消费行为
- **AND** batch runtime MUST 只在配置声明 opt-in 时传递对应 metadata

### Requirement: Adaptive fusion 优先作为模块化组件
observability-aware、reliability-aware 或 uncertainty-gated fusion 行为 MUST NOT 直接复制到多个整模型中。若该行为服务普通 supervised/adaptation baseline，系统 MUST 优先将其实现为 representation core、adapter helper 或等价可组合组件，并通过配置启用。

#### Scenario: observability-aware fusion 可配置复用
- **WHEN** Scenario D 或后续 robustness baseline 需要 image/GPS reliability weighting
- **THEN** 配置 MUST 能显式选择可组合 adaptive fusion 行为或记录使用显式 helper 的边界
- **AND** 普通 early-concat、CLS-token transformer 和 JEPA baseline MUST 不被静默替换语义
