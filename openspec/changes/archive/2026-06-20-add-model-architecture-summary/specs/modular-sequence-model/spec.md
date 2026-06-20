## ADDED Requirements

### Requirement: 模块化模型架构摘要分组
`ModularSequenceModel` MUST 支持统一模型架构摘要能力识别其内部组件。摘要 MUST 按 `encoders.<modality>`、`projectors.<modality>`、`representation_core`、`heads.<name>`、可选 geometry prior、logit fusion 和 reranker 分组，并 MUST 保持现有 forward、batch runtime 和 `training_strategy_metadata()` 行为兼容。

#### Scenario: image-only modular summary
- **WHEN** 用户对 image-only `modular_sequence` 模型生成架构摘要
- **THEN** 摘要 MUST 包含 image encoder、image projector、representation core 和 beam head 组件
- **AND** 每个组件 MUST 包含 path、class、registry type 或 fallback class name、total params 和 trainable params

#### Scenario: image+GPS modular summary
- **WHEN** 用户对 image+GPS `modular_sequence` 模型生成架构摘要
- **THEN** 摘要 MUST 分别包含 image encoder 和 GPS encoder 参数量
- **AND** 摘要 MUST 包含多模态 representation core 参数量

#### Scenario: optional component summary
- **WHEN** `modular_sequence` 启用 geometry prior、logit fusion 或 safe residual reranker
- **THEN** 摘要 MUST 将这些 opt-in 组件作为独立组件条目记录
- **AND** 摘要 MUST 记录其是否消费 reliability metadata

### Requirement: 模块化组件 metadata 与参数摘要合并
`ModularSequenceModel` 的架构摘要 MUST 合并组件 `training_strategy_metadata()` 与实际参数统计。组件 metadata 中的 registry type、checkpoint、freeze policy、token metadata、reliability metadata 和 output dimension MUST 保留；参数统计 MUST 由实际 module 参数或声明候选 metadata 提供。

#### Scenario: TinyViT metadata 合并
- **WHEN** `modular_sequence` 使用 TinyViT image encoder
- **THEN** 摘要 MUST 记录 TinyViT registry type、variant、pretrained source、checkpoint source、freeze policy、trainable stages、backbone_dim 和 output_dim
- **AND** 摘要 MUST 记录 image encoder total params、trainable params 和 effective/excluded 参数口径

#### Scenario: JEPA context image metadata 合并
- **WHEN** `modular_sequence` 使用 JEPA context image encoder
- **THEN** 摘要 MUST 记录 visual tokenizer 或 context encoder 相关 metadata
- **AND** 摘要 MUST 能报告 image encoder params 和 visual/context encoder params

#### Scenario: 普通组件缺少 metadata
- **WHEN** 某个 projector、core 或 head 没有 `training_strategy_metadata()`
- **THEN** 摘要 MUST 仍记录该组件 class、path、total params 和 trainable params
- **AND** 摘要 MUST 不要求组件为了被统计而改变 forward 签名

### Requirement: 模块化摘要不改变运行契约
架构摘要能力 MUST 是只读观测能力。生成 `modular_sequence` 摘要 MUST 不改变模型参数、`requires_grad` 状态、forward 输出、batch runtime 输入或训练 optimizer 参数组。

#### Scenario: 摘要前后参数状态不变
- **WHEN** 用户对 `modular_sequence` 模型调用架构摘要 helper
- **THEN** 模型所有参数的 `requires_grad` 状态 MUST 保持不变
- **AND** 模型 forward 输出结构 MUST 不因摘要调用而改变

#### Scenario: 摘要不创建 optimizer
- **WHEN** 用户只生成 `modular_sequence` 架构摘要
- **THEN** 系统 MUST 不创建 optimizer 或 scheduler
- **AND** 系统 MUST 不执行训练 batch 或 validation batch
