## ADDED Requirements

### Requirement: Next-beam query Transformer representation core
模块化序列模型 MUST 支持 `next_beam_query_transformer` representation core，用于多模态历史输入的下一时刻单步预测。该 core MUST 接收多模态 projector 输出 `[B, K, T, D]`，注入 time embedding 与 modality embedding，追加 learned next-beam query token，并输出可被现有 heads 消费的 `[B, 1, D_out]` 表征。

#### Scenario: 构建 next-beam query core
- **WHEN** 用户配置 `model.primary.representation_core.type: next_beam_query_transformer`
- **THEN** 系统 MUST 构建注册到 `REPRESENTATION_CORES` 的 next-beam query Transformer core
- **AND** core 配置 MUST 支持 `d_model`、`modality_count`、`num_heads`、`num_layers`、`dropout`、`max_seq_len` 和 `output_dim`

#### Scenario: 多模态历史 token 输入
- **WHEN** `next_beam_query_transformer` core 收到 `[B, K, T, D]` 输入
- **THEN** core MUST 校验 `K` 与配置的 `modality_count` 一致
- **AND** core MUST 校验 `D` 与配置的 `d_model` 一致
- **AND** core MUST 输出 `[B, 1, D_out]`

#### Scenario: 拒绝单模态三维输入
- **WHEN** `next_beam_query_transformer` core 收到 `[B, T, D]` 输入
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出 next-beam query Transformer 需要多模态 `[B, K, T, D]` 输入

#### Scenario: 时间长度超过上限
- **WHEN** 输入时间维 `T` 大于配置的 `max_seq_len`
- **THEN** 系统 MUST 拒绝 forward
- **AND** 错误信息 MUST 包含实际 `T` 和 `max_seq_len`

### Requirement: Next-beam query Transformer token embedding
`next_beam_query_transformer` MUST 显式区分模态来源、时间位置和查询 token。历史 token MUST 加上 modality embedding 与 time embedding；learned next-beam query MUST 作为独立 token 参与 Transformer 编码，并且最终输出 MUST 来自该 query token。

#### Scenario: 注入模态和时间 embedding
- **WHEN** core 对 `[B, K, T, D]` 历史 token 执行 forward
- **THEN** 每个历史 token MUST 加上对应模态 embedding
- **AND** 每个历史 token MUST 加上对应时间位置 embedding

#### Scenario: 使用 query token 输出
- **WHEN** Transformer 编码完成
- **THEN** core MUST 取 learned next-beam query token 的编码结果作为输出
- **AND** core MUST NOT 对全部历史 token 简单 mean pooling 作为 next-query 主输出

### Requirement: Next-beam query Transformer 与模块化 head 兼容
`next_beam_query_transformer` MUST 保持模块化模型的 head 输出契约。对于 beam prediction，现有 `beam_head` MUST 能消费 core 输出 `[B, 1, D_out]` 并产生 `[B, 1, num_classes]` logits。

#### Scenario: beam head 消费 next-query 输出
- **WHEN** `ModularSequenceModel` 使用 `next_beam_query_transformer` core 和 `beam_head`
- **THEN** model forward MUST 返回 `logits` 字段
- **AND** `logits` 形状 MUST 为 `[B, 1, num_classes]`

#### Scenario: 保留中间诊断字段
- **WHEN** `ModularSequenceModel` 使用 `next_beam_query_transformer`
- **THEN** model forward MUST 继续返回 `input_features`、`output_features`、`modalities`、`modality_features` 和 `encoder_features`
- **AND** `output_features` 时间维 MUST 为 `1`
