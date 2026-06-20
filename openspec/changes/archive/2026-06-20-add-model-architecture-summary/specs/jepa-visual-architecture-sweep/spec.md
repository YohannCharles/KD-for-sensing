## ADDED Requirements

### Requirement: Sweep 参数摘要使用统一 schema
JEPA visual architecture sweep MUST 将候选参数和 compute metadata 映射到统一模型架构摘要 schema。summary table MUST 保留 `variant_id`、family、stage plan、checkpoint policy、token metadata、total params、trainable params、image encoder params、visual/context encoder params、compute proxy 和参数来源字段。

#### Scenario: full results 行包含统一参数字段
- **WHEN** architecture sweep 生成 full results 或 expanded manifest summary
- **THEN** 每个候选行 MUST 包含 total params、trainable params、image encoder params、visual/context encoder params、token count 和 compute proxy
- **AND** 每个候选行 MUST 记录参数来源是声明 metadata、真实 module 统计还是混合来源

#### Scenario: missing metrics 不删除参数摘要
- **WHEN** 候选训练失败、被跳过、missing metrics 或 availability 为 unavailable
- **THEN** summary MUST 仍保留该候选的参数摘要字段
- **AND** summary MUST 不因缺失指标而从 full table 中静默移除该候选

### Requirement: 极小参数量 JEPA 候选作为基准口径
JEPA visual architecture sweep MUST 将 `patch14_stage1_gps_query` 作为参数摘要基准候选之一。summary fixture 或 focused test MUST 锁定其约 0.197M total params、约 0.117M image encoder params 和约 0.088M visual/context encoder params 的当前口径，允许使用明确容差或 source-managed fixture。

#### Scenario: patch14 极小模型参数口径
- **WHEN** summary 处理 `patch14_stage1_gps_query` 候选
- **THEN** 输出 MUST 包含约 0.197M total params
- **AND** 输出 MUST 包含约 0.117M image encoder params
- **AND** 输出 MUST 包含约 0.088M visual/context encoder params

#### Scenario: patch14 与 ResNet token 候选同表比较
- **WHEN** summary 同时包含 `patch14_stage1_gps_query`、`resnet18_layer4_tokens` 和 `resnet18_layer3_layer4_tokens`
- **THEN** 三个候选 MUST 使用同一参数字段名和同一参数来源标记
- **AND** summary MUST 能按 total params、image encoder params 或 visual/context encoder params 排序

### Requirement: ResNet token 候选参数口径保持可比
JEPA visual architecture sweep MUST 保留 ResNet token 候选的参数摘要口径。`resnet18_layer4_tokens` 和 `resnet18_layer3_layer4_tokens` MUST 在 summary 中报告 total params、image encoder params 和 visual/context encoder params，以支持与 patch/overlap/hybrid 候选比较。

#### Scenario: resnet18 layer4 token 参数口径
- **WHEN** summary 处理 `resnet18_layer4_tokens` 候选
- **THEN** 输出 MUST 包含约 11.32M total params
- **AND** 输出 MUST 包含约 11.24M image encoder params
- **AND** 输出 MUST 包含约 11.21M visual/context encoder params

#### Scenario: resnet18 layer3+layer4 token 参数口径
- **WHEN** summary 处理 `resnet18_layer3_layer4_tokens` 候选
- **THEN** 输出 MUST 包含约 14.13M total params
- **AND** 输出 MUST 包含约 14.05M image encoder params
- **AND** 输出 MUST 包含约 14.02M visual/context encoder params

### Requirement: Sweep Pareto 使用统一参数字段
JEPA visual architecture sweep 的 Pareto、family best 和 Markdown summary MUST 使用统一模型架构摘要字段。系统 MUST 支持按 DBA、Top-1、trainable params、total params、image encoder params、visual/context encoder params、token count 和 compute proxy 生成候选解释。

#### Scenario: Pareto 区分极小模型和大 CNN token 模型
- **WHEN** summary 生成 params/compute Pareto
- **THEN** `patch14_stage1_gps_query` 的极小参数量优势 MUST 能在 Pareto 表中体现
- **AND** ResNet token 候选的较大视觉参数规模 MUST 能在同一表中体现

#### Scenario: Markdown summary 解释规模收益
- **WHEN** summary 生成 Markdown 报告
- **THEN** 报告 MUST 包含参数规模对照段落或表格
- **AND** 报告 MUST 避免只按最终指标排名而隐藏参数量差异
