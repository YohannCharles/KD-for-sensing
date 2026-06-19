## ADDED Requirements

### Requirement: Safe residual rerank fusion configuration
Fusion 配置 MUST 支持 opt-in safe residual rerank component baseline。该 baseline MUST 通过 `model.primary` 的窄字段选择，不得新增根训练脚本、复制训练循环或注册 whole-model exception，除非 design 另行记录不可组合原因。

#### Scenario: 配置启用 reranker
- **WHEN** 配置声明 `model.primary.reranker.enabled=true`
- **THEN** final config MUST 记录 anchor source、candidate sources、candidate top-k、residual scale、fallback policy、loss mode 和 diagnostics mode
- **AND** 模型 MUST 仍能由现有 registry/config loader 构建

#### Scenario: 配置关闭时 baseline 不变
- **WHEN** 配置未启用 reranker
- **THEN** Image ResNet+GPS、JEPA GPS-query 和 geometry-prior fusion baseline 行为 MUST 保持不变
- **AND** batch runtime MUST 不要求 reranker diagnostics 或 candidate fields

### Requirement: Anchor source declaration
Safe residual rerank 配置 MUST 显式声明 anchor logits 来源。Anchor MAY 来自同一 `modular_sequence` image+GPS branch、frozen checkpoint logits cache 或显式 teacher/anchor provider。

#### Scenario: 内部 anchor
- **WHEN** reranker 使用内部 anchor branch
- **THEN** anchor branch MUST 输出 standalone `anchor_logits`
- **AND** final output MUST 同时记录 anchor 和 reranked logits 的 provenance

#### Scenario: 外部 anchor checkpoint
- **WHEN** reranker 使用外部 checkpoint 或 logits cache
- **THEN** 配置 MUST 记录 checkpoint/config path、provenance、temperature 和 allowed splits
- **AND** 系统 MUST 不通过旧 KD/distillation runtime 加载该 anchor
