## ADDED Requirements

### Requirement: No-regret reliability gate
Observability-aware fusion MUST 支持 no-regret reliability gate，用于 anchor-safe reranker。Gate MUST 使用连续 reliability/uncertainty 信号，而不是 benchmark condition id。

#### Scenario: gate 输入
- **WHEN** reranker gate 构建输入
- **THEN** gate MAY 使用 image observability、GPS valid mask、GPS delay、GPS counterfactual mask、anchor entropy、prior entropy 和 branch disagreement
- **AND** gate MUST NOT 使用 condition、suite、P/C/D id 或 claim label

#### Scenario: gate 输出
- **WHEN** gate forward 成功
- **THEN** diagnostics MUST 包含 gate confidence、fallback_to_anchor、fallback reason 和 residual scale
- **AND** clean/high-observability 条件下 fallback behavior MUST 可聚合统计

### Requirement: Anchor fallback branch diagnostics
Reranker-aware observability diagnostics MUST 区分 anchor branch、geometry prior branch 和 residual/rerank branch 的贡献。

#### Scenario: branch contribution
- **WHEN** reranker 改变最终 top prediction
- **THEN** diagnostics MUST 记录 changed_from_anchor=true、selected beam、anchor beam、prior beam、target rank delta 和 DBA delta
- **AND** aggregate MUST 能区分 beneficial、neutral 和 harmful changes

#### Scenario: wrong GPS 降低 prior trust
- **WHEN** gps_counterfactual_mask=true 或 prior-image disagreement 超过阈值
- **THEN** gate MUST 能降低 prior/rerank residual 权重或 fallback anchor
- **AND** diagnostics MUST 记录该 reliability signal
