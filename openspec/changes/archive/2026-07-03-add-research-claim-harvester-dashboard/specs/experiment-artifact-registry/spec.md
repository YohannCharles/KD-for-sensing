## ADDED Requirements

### Requirement: Ledger provenance links
实验 artifact registry MUST 为 experiment ledger 提供可追溯的 checkpoint/run provenance。Ledger 记录 MAY 引用 registry sidecar，但 MUST 不复制真实 checkpoint。

#### Scenario: ledger 引用 checkpoint sidecar
- **WHEN** run 目录包含 checkpoint sidecar metadata
- **THEN** ledger 或 harvester summary MUST 记录 sidecar path、checkpoint path、selection metric、selected epoch、run_dir 和 config digest
- **AND** 缺少 sidecar 时 MUST 标记 provenance incomplete

#### Scenario: registry 与 claim candidate 关联
- **WHEN** claim candidate 引用某个 best checkpoint
- **THEN** candidate MUST 记录 checkpoint 来源是 explicit path、scene/scenegroup registry、run-local checkpoint 还是 unavailable
- **AND** 不明确来源 MUST 阻止 candidate 自动升级为 reviewed claim
