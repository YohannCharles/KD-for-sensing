## ADDED Requirements

### Requirement: 主表导出 gate 硬排除不合格 claim
Paper artifact export MUST 默认排除 `pending`、`mock/smoke`、`historical ablation`、`upper-bound`、`blocked official reproduction`、`not_comparable`、`unverified` 和 `candidate_only=true` 的行进入正式主表。被排除行如被导出，MUST 进入 excluded report 或显式标注的 appendix draft，并 MUST 保留 status 和 caveat。

#### Scenario: pending rows 不进入主表
- **WHEN** 输入 claim registry 或 ledger 包含 pending、mock/smoke、upper-bound 或 not_comparable 行
- **THEN** paper export 主表 MUST 不包含这些行
- **AND** excluded report MUST 说明排除原因和 claim id

#### Scenario: 人工覆盖需要显式参数
- **WHEN** 用户显式要求导出非正式 appendix 或 diagnostics table
- **THEN** 系统 MUST 仅将不合格状态行导出到显式标注的非正式 appendix 或 diagnostics table
- **AND** 输出表 MUST 显示 claim status 和 caveat，不得伪装成正式主表
