## ADDED Requirements

### Requirement: Geometry-prior claim requires real perturbation forward
Geometry-prior robustness claim gate MUST distinguish real-forward evidence from delegated clean-only、synthetic metrics 或 deterministic degradation rows。只有真实 per-condition forward metrics MAY 升级 primary claim。

#### Scenario: real-forward evidence 升级 claim
- **WHEN** candidate clean gate 通过且 P-suite/advantage metrics 均来自 real-forward logits
- **THEN** geometry-prior claim gate MAY 返回 `pass`
- **AND** output MUST 记录 real-forward shard completeness、sample_count 和 cache fingerprint

#### Scenario: delegated evidence 保持 pending
- **WHEN** candidate 只有 clean delegated evaluation 或 synthetic/degradation perturbation rows
- **THEN** geometry-prior claim gate MUST 返回 `pending`
- **AND** reason MUST 包含 `delegated_clean_only_perturbations_not_real_forward` 或等价 machine-readable code

### Requirement: Geometry prior rerank diagnostics
Geometry-prior rerank evaluation MUST 输出 prior 是否进入 candidate set、是否改变最终 beam、是否改善 target rank/DBA 的 diagnostics。

#### Scenario: prior 参与候选
- **WHEN** geometry prior top-k beam 进入 rerank candidate set
- **THEN** diagnostics MUST 记录 prior candidate count、prior selected count、prior target rank 和 prior-anchor agreement
- **AND** aggregate tables MUST 可按 condition、split、seed 和 model group 分组

#### Scenario: prior 不可用
- **WHEN** GPS prior unavailable、high entropy 或 reliability metadata 声明 GPS 无效
- **THEN** reranker MUST 标记 prior branch unavailable 或 low trust
- **AND** diagnostics MUST 记录 fallback reason，不能静默声明 prior 帮助

### Requirement: Clean no-regret summary
Geometry-prior strict diagnostics MUST 输出 clean no-regret summary，量化 reranker 对 anchor clean prediction 的改变和损害。

#### Scenario: clean regression gate
- **WHEN** clean/P0 real-forward evaluation 完成
- **THEN** diagnostics MUST 输出 anchor DBA、rerank DBA、delta、changed-top1 rate 和 harmful-change rate
- **AND** clean regression 超阈值时 claim gate MUST 标记 failed 或 pending
