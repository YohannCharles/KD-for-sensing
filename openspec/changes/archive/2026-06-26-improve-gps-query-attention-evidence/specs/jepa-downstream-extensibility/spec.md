## ADDED Requirements

### Requirement: GPS-query attention aggregation metadata
JEPA downstream GPS-query 类 pooler MUST 在启用 attention diagnostics 时记录 attention 聚合 metadata。Metadata MUST 说明 attention 是否跨 head 平均、是否跨 query/time 聚合、原始 attention shape、诊断输出 shape、condition feature source 和 token grid 或 token count。

#### Scenario: 记录平均 attention metadata
- **WHEN** `GPSQueryPool` 或等价 GPS-query pooler 使用默认 averaged attention diagnostics
- **THEN** pooler diagnostics MUST 记录 `attention_head_aggregation=averaged`
- **AND** diagnostics MUST 记录原始可见 attention shape、输出 attention map shape、query count、token count 和 condition feature source

#### Scenario: 记录分支 attention metadata
- **WHEN** predictive 或 hybrid GPS-query pooler 同时产生 content attention 和 GPS attention
- **THEN** diagnostics MUST 分别记录 content branch 和 GPS branch 的 attention summary 或 unavailable reason
- **AND** GPS branch attention MUST 标明是否作为 `last_attention_map` 暴露给 visual analysis

### Requirement: Opt-in per-head attention diagnostics
JEPA downstream GPS-query 类 pooler MUST 支持 opt-in per-head attention diagnostics，且该模式 MUST 不改变训练主输出、loss 输入、checkpoint 加载语义或默认配置行为。未显式开启时，系统 MUST 保持现有 averaged attention 行为。

#### Scenario: 默认保持 averaged attention
- **WHEN** 用户未显式启用 per-head attention diagnostics
- **THEN** GPS-query 类 pooler MUST 保持现有 averaged attention map 输出语义
- **AND** 现有 `return_attention=True` shape 兼容测试 MUST 继续通过

#### Scenario: 开启 per-head diagnostics
- **WHEN** 分析或诊断配置显式启用 per-head attention diagnostics
- **THEN** pooler MUST 返回或缓存包含 head 维度的 attention diagnostics
- **AND** diagnostics MUST 记录 per-head shape、head count 和用于下游 summary 的 head aggregation method
- **AND** 训练 forward 的主 pooled feature shape MUST 不变

#### Scenario: per-head diagnostics 受采样限制
- **WHEN** per-head attention diagnostics 开启且样本数超过 attention case 限制
- **THEN** 系统 MUST 只为受 `max_attention_cases` 或等价配置限制的样本保留 per-head 明细
- **AND** manifest MUST 记录被截断的样本数或 skipped reason
