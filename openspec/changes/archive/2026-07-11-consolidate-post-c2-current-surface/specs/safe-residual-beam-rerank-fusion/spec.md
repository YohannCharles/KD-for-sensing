## REMOVED Requirements

### Requirement: Anchor-safe residual reranker
**Reason**: Safe residual reranker 没有 current config、CLI 或 claim consumer。
**Migration**: 删除专属 reranker；current model 直接使用其保留 head/fusion logits。

#### Scenario: Reranker 不再构建
- **WHEN** current component registry 被加载
- **THEN** anchor-safe residual reranker MUST 不再注册或执行

### Requirement: Candidate beam set
**Reason**: Candidate builder 只服务已退役 safe reranker。
**Migration**: 删除 anchor/geometry/teacher candidate assembly；保留通用 Top-K metrics helper。

#### Scenario: Candidate builder 退出
- **WHEN** current model forward 运行
- **THEN** 系统 MUST 不要求 safe-rerank candidate beam set
- **AND** ordinary Top-K prediction MUST 保持可用

### Requirement: No-regret fallback gate
**Reason**: No-regret gate 只服务已退役 rerank attachment。
**Migration**: 删除专属 fallback；current fusion/model owners 保持自己的 fallback semantics。

#### Scenario: Rerank gate 不再执行
- **WHEN** current logits 被生成
- **THEN** runtime MUST 不应用 safe-rerank no-regret fallback gate

### Requirement: Reranker training objectives
**Reason**: Reranker 删除后，其 residual/ranking objectives 没有模型参数或 consumer。
**Migration**: 保留 current supervised/U-Mask losses；删除 reranker-only losses 与 metrics。

#### Scenario: Reranker objectives 退出
- **WHEN** current training loss 被构建
- **THEN** 系统 MUST 不要求 reranker-specific training objectives

### Requirement: Safe residual rerank fusion configuration
**Reason**: Retired component 不应继续保留可接受 config surface。
**Migration**: 删除 config keys/validation；旧配置以 unknown/removed route 拒绝。

#### Scenario: 旧 rerank config 被拒绝
- **WHEN** config 请求 safe residual rerank fusion
- **THEN** validation 或 component construction MUST fail clearly
- **AND** 系统 MUST 不静默映射到 current fusion

### Requirement: Anchor source declaration
**Reason**: Anchor source schema 只用于 safe reranker diagnostics/configuration。
**Migration**: 删除 reranker-only anchor metadata；普通 model provenance 继续由 current owner 记录。

#### Scenario: Anchor metadata 退出
- **WHEN** current run metadata 被写出
- **THEN** 系统 MUST 不要求 safe-reranker anchor source declaration

