## ADDED Requirements

### Requirement: Harvested claim draft governance
主线实验文档 MUST 区分 harvested claim draft 和正式 claim registry。自动生成的 candidate、dashboard summary 或 ledger record MUST 不被描述为已审阅结论。

#### Scenario: candidate 不自动进入 claim registry
- **WHEN** harvester 输出 claim candidate
- **THEN** `docs/result_claims_registry.md` MUST 只有在人工审阅后才新增或更新对应 claim 行
- **AND** candidate 输出 MUST 保留 `draft`、`candidate_only` 或等价状态标记

#### Scenario: README 和实验矩阵引用 dashboard
- **WHEN** 文档新增 research dashboard 或 harvester 入口说明
- **THEN** README 或 `docs/experiment_matrix.md` MAY 指向该入口作为本地研究辅助工具
- **AND** 文档 MUST 说明它不生成正式论文结论、不移动产物、不替代 claim registry
