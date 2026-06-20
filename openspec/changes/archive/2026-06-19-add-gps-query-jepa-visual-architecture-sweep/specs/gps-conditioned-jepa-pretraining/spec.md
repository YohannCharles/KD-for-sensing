# gps-conditioned-jepa-pretraining Specification Delta

## ADDED Requirements

### Requirement: JEPA visual token encoder variants
GPS-conditioned JEPA 主模型 MUST 支持可配置 visual token encoder variants。默认 variant MUST 保持现有 patch16 `VisualPatchTokenEncoder` 行为兼容；opt-in variants MAY 包含 overlap patch tokenizer、conv stem tokenizer、local token mixing、CvT-style convolutional projection、CNN feature-map tokens 或多尺度 tokens。所有 variants MUST 输出可供 JEPA predictor 和 mask sampler 消费的 `[B,T,N,D]` tokens。

#### Scenario: 默认 patch encoder 兼容
- **WHEN** 用户未设置 `visual_encoder.type` 或设置为现有 patch encoder 等价类型
- **THEN** 系统 MUST 保持现有 patch16 非重叠 tokenization、checkpoint loading 和 forward shape 行为兼容
- **AND** 现有 GPS-conditioned JEPA 配置 MUST 无需修改即可构建和训练

#### Scenario: opt-in tokenizer 输出统一 token 契约
- **WHEN** 用户配置 overlap patch、conv stem、local token mixing、CvT-style 或 CNN-token visual encoder variant
- **THEN** encoder MUST 输出 `[B,T,N,D]` tokens 和 token/grid metadata
- **AND** metadata MUST 记录 `visual_encoder.type`、image size、effective stride、token grid、token count、positional encoding 和 max token budget

#### Scenario: token budget 超限时报错
- **WHEN** visual encoder variant 产生的 token 数超过配置的 `max_tokens` 或模型预算
- **THEN** 系统 MUST 抛出包含实际 token count、max token count、image size 和 variant type 的清晰错误
- **AND** 系统 MUST 不静默截断 tokens

### Requirement: JEPA mask sampler 适配可变 token grid
JEPA mask sampler MUST 基于 visual token encoder 提供的 token/grid metadata 采样 context 和 target tokens。sampler MUST 不硬编码 patch16、14x14 或 196 tokens，并 MUST 在 GPS angle biased mode 下记录使用的 token grid。

#### Scenario: GPS angle biased mask 使用 token metadata
- **WHEN** visual encoder variant 的 token grid 不是 14x14
- **THEN** GPS angle biased mask sampler MUST 使用该 variant 的 token/grid metadata 构造采样权重
- **AND** diagnostics MUST 记录 mask mode、token grid、context ratio、target ratio 和有效 target token 数

#### Scenario: 多尺度 token 可审计
- **WHEN** visual encoder 输出多尺度 tokens 或合并后的 token sequence
- **THEN** mask sampler diagnostics MUST 记录每个 scale 的 token count 或合并策略
- **AND** predictor target shape MUST 与被采样 target tokens 对齐

### Requirement: JEPA visual encoder checkpoint policy
GPS-conditioned JEPA 预训练和下游复用 MUST 显式记录 visual encoder checkpoint policy。policy MUST 区分 `exact_reuse`、`partial_reuse`、`pos_interpolate`、`fresh_stage1_required` 和 `supervised_only_anchor` 或等价状态。

#### Scenario: 形状不匹配不能伪装为 exact reuse
- **WHEN** visual encoder variant 的 patch embedding、position embedding 或 backbone 参数形状与 checkpoint 不匹配
- **THEN** 系统 MUST 不允许将该 run 标记为 `exact_reuse`
- **AND** metadata MUST 记录 missing keys、unexpected keys、interpolated position grid 或 fresh initialization reason

#### Scenario: 新 tokenizer 需要新 Stage 1 checkpoint
- **WHEN** tokenizer/backbone 改变导致无法复用现有 GPS-biased JEPA context encoder
- **THEN** 配置或 metadata MUST 标记 `fresh_stage1_required`
- **AND** downstream strict comparison MUST 使用该 tokenizer 对应的 Stage 1 checkpoint，而不是旧 patch16 checkpoint
