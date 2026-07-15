## ADDED Requirements

### Requirement: Paper main table 使用 reviewed allowlist 与必填 schema
paper exporter MUST 只允许明确 reviewed 的状态进入 main table，并 MUST 在纳入前验证 claim schema 必填字段。未知、空、pending、not_comparable、invalidated、mock/smoke、historical、upper-bound、blocked、unverified 或 candidate-only 行 MUST 默认进入 excluded report。

#### Scenario: 空或未知状态
- **WHEN** claim status 为空或不在 reviewed allowlist
- **THEN** row MUST NOT 进入 main table
- **AND** excluded report MUST 记录 `status_not_reviewed`

#### Scenario: Reviewed 状态但字段缺失
- **WHEN** status 在 allowlist 中但 method、dataset/split、metric/value、统计或 provenance 必填字段缺失
- **THEN** row MUST NOT 进入 main table
- **AND** excluded report MUST 列出 missing fields

#### Scenario: 完整 reviewed claim
- **WHEN** claim status 在 allowlist、candidate flag 为 false 且必填字段完整
- **THEN** exporter MUST 允许 row 进入 main table
- **AND** output MUST 保留 status、provenance 和 caveat
