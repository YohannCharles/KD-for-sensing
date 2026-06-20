## ADDED Requirements

### Requirement: 模型扩展可被架构摘要审计
新增 baseline、组件 baseline、whole-model exception 和 workflow/paper reproduction MUST 能被模型架构摘要能力审计。审计信息 MUST 覆盖模型注册名或候选 ID、架构类别、启用模态、组件组合、参数量、checkpoint/freeze 策略、reliability metadata 消费和比较口径来源。

#### Scenario: component baseline summary 兼容
- **WHEN** 开发者新增或替换 `ENCODERS`、`PROJECTORS`、`REPRESENTATION_CORES` 或 `HEADS` 子组件
- **THEN** 该组件 MUST 能通过统一模型架构摘要出现在对应 role 分组中
- **AND** 摘要 MUST 记录该组件的 registry type、class、total params 和 trainable params

#### Scenario: whole-model exception summary 兼容
- **WHEN** 开发者新增完整 `MODELS.register(...)` 的 whole-model exception
- **THEN** 该模型 MUST 提供 `training_strategy_metadata()` 或等价 metadata，使架构摘要能记录模型注册名、架构类别、启用模态、checkpoint/freeze 策略和 reliability metadata 消费
- **AND** 如果无法自动分组内部组件，摘要 MUST 至少保留正确 total/trainable 参数和 unknown component role

#### Scenario: workflow baseline summary 兼容
- **WHEN** workflow/paper reproduction 生成候选、run manifest 或 summary table
- **THEN** 其参数量和 compute proxy 字段 MUST 能映射到统一模型架构摘要 schema
- **AND** summary MUST 区分真实实例统计和声明候选 metadata

### Requirement: 参数比较口径可审计
模型架构扩展 MUST 明确参数比较口径。系统 MUST 区分 total params、trainable params、effective params、excluded params、image encoder params、visual/context encoder params 和 compute proxy。任何参数量声明 MUST 记录来源，避免把 manifest 估算误当作真实 module 统计。

#### Scenario: 真实模型参数来源
- **WHEN** 参数量来自已构建 `nn.Module`
- **THEN** summary MUST 标记来源为实际 module 统计
- **AND** 参数量 MUST 使用去重后的 `named_parameters()` 或等价机制计算

#### Scenario: manifest 候选参数来源
- **WHEN** 参数量来自 sweep manifest、candidate metadata 或设计期估算
- **THEN** summary MUST 标记来源为声明候选 metadata
- **AND** summary MUST 不把该参数量标记为实际 module 统计

#### Scenario: 语义排除参数可追踪
- **WHEN** 模型实例包含不参与 downstream forward 的参数组
- **THEN** summary MUST 保留 total params
- **AND** summary MUST 在 effective/excluded 字段中记录排除口径和原因

### Requirement: 新模型 focused tests 包含摘要覆盖
新增模型、encoder、representation core、whole-model exception 或 sweep 候选矩阵时，focused tests MUST 覆盖模型架构摘要的关键字段。测试 MUST 至少验证 registry/candidate ID、组件 role、参数量字段、metadata 合并和 warning 语义。

#### Scenario: 新 encoder 摘要测试
- **WHEN** change 新增一个 image encoder 或其它模态 encoder
- **THEN** tasks MUST 包含该 encoder 在 `modular_sequence` 中生成架构摘要的 focused test
- **AND** test MUST 验证该 encoder 的 registry type、组件路径和参数量字段

#### Scenario: 新 sweep 候选摘要测试
- **WHEN** change 新增 sweep 候选族或参数/compute controls
- **THEN** tasks MUST 包含候选摘要 fixture 或 summary table test
- **AND** test MUST 验证候选参数来源、total/trainable params、token count 和 compute proxy
