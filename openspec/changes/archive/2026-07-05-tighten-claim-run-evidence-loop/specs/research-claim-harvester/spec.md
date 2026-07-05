## ADDED Requirements

### Requirement: Claim doctor 输出缺失证据
研究 claim harvester MUST 提供 claim doctor 能力，用于读取 claim registry、run index、candidate ledger 或 summary artifacts，并输出每个 pending/unverified/not_comparable claim 缺失的 seed、split、metric profile、label space、checkpoint provenance、difficulty digest、stress provenance、paired baseline 或统计证据。Claim doctor MUST 不自动升级 claim status。

#### Scenario: pending claim 缺失字段
- **WHEN** claim doctor 读取一个 `pending` 或 `unverified` claim
- **THEN** 输出 MUST 列出缺失 provenance 字段和 next action hints
- **AND** 输出 MUST 保持该 claim 为 candidate 或待人工审阅状态

#### Scenario: claim 可升级候选
- **WHEN** claim candidate 具备 strict comparability、真实 metrics、多 seed 或对应 capability 要求的完整 provenance
- **THEN** claim doctor MUST 将其列为 upgradable candidate
- **AND** 系统 MUST 不自动修改 `docs/result_claims_registry.md`

### Requirement: Run card artifact
研究 dashboard 或 run index MUST 能为本地 run 生成 run card。Run card MUST 至少记录 command、git commit 或 dirty status、config path、config digest、dataset/split metadata、checkpoint provenance、metrics path、claim candidate id 和 caveat。Run card 默认写入 ignored output root。

#### Scenario: 生成 run card
- **WHEN** 用户对某个 run 生成 run card
- **THEN** 系统 MUST 写出 JSON 或 Markdown artifact 到 ignored output root 或用户显式路径
- **AND** run card MUST 不包含真实 checkpoint 内容或敏感凭证
