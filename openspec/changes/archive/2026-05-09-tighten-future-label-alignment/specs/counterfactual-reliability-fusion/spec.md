## MODIFIED Requirements

### Requirement: 单模态辅助预测与 confidence
CRAF MUST 为每个启用模态提供轻量单模态辅助预测头，并 MUST 从辅助预测中计算可用于 reliability 估计的 confidence 特征。单模态辅助预测的 horizon MUST 与主 future-label horizon 一致，配置 `model.num_pred: N` 时输出 `N` 个 future prediction slot。

#### Scenario: 单模态 logits 输出
- **WHEN** CRAF forward 启用 `return_unimodal` 且配置中的 `model.num_pred` 为 `N`
- **THEN** 输出 MUST 包含单模态 logits
- **AND** 单模态 logits MUST 能按 `[B, K, N, C]` 或等价结构表示
- **AND** 第 0 个 prediction slot MUST 表示 `t+1`

#### Scenario: confidence 特征计算
- **WHEN** 单模态 logits 可用
- **THEN** 系统 MUST 计算 entropy-based confidence 和 top probability margin
- **AND** confidence 特征 MUST 与启用模态顺序一致

#### Scenario: 单模态辅助 loss 可关闭
- **WHEN** 配置将单模态辅助 loss 权重设为 0
- **THEN** 训练总 loss MUST 不包含单模态辅助 loss
- **AND** forward 输出无论是否包含 diagnostics 字段，训练流程 MUST 保持有效

#### Scenario: 单模态 horizon 不匹配时报错
- **WHEN** 单模态辅助 logits 的 prediction slot 数不等于 `num_pred`
- **THEN** 单模态辅助 loss 路径 MUST 报错
- **AND** 系统 MUST 不静默裁剪 `num_pred + 1` 旧辅助 head

### Requirement: Transformer fusion 与 horizon prediction
CRAF MUST 使用 token-level fusion 模块融合启用模态历史 token，并 MUST 输出与当前训练标签语义一致的预测 slot。配置 `model.num_pred: N` 时，CRAF 主 prediction head MUST 直接输出 `N` 个 future prediction slot。

#### Scenario: Transformer 忽略 padding token
- **WHEN** token padding mask 中某些位置为 True
- **THEN** Transformer fusion MUST 在 self-attention 中忽略这些 token

#### Scenario: 预测长度对齐现有标签
- **WHEN** 配置中的 `model.num_pred` 为 `N`
- **THEN** CRAF 默认 MUST 输出 `N` 个预测 slot
- **AND** 这些 slot MUST 能直接与 `prepare_labels()` 的 `[t+1, ..., t+N]` 输出对齐
- **AND** CRAF MUST 不再输出用于当前或历史最后一个 beam 的额外 prediction slot

#### Scenario: 输出类别数对齐
- **WHEN** 配置中的 `model.num_classes` 为 `C`
- **THEN** CRAF 输出 logits 的最后一维 MUST 为 `C`

#### Scenario: 旧 horizon 配置不被接受
- **WHEN** CRAF 主 logits 的 prediction slot 数为 `num_pred + 1`
- **THEN** CRAF 定向测试或训练 shape 检查 MUST 将其视为 horizon 契约错误
- **AND** 系统 MUST 不把第一个 slot 当作历史 beam 静默丢弃
