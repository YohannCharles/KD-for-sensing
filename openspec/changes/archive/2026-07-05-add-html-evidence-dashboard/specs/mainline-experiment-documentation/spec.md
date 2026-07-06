## MODIFIED Requirements

### Requirement: Research dashboard 汇总 paper readiness
主线实验文档治理 MUST 支持只读 dashboard 或 readiness report，汇总 active OpenSpec change、run states、pending/unverified claim 计数、upgradable candidates、paper export exclusions 和下一步证据缺口。Dashboard MUST 能输出文本、JSON 或静态 HTML report。Dashboard 输出 MUST 写入 ignored output root 或用户显式路径，不得自动修改 current docs。

#### Scenario: dashboard 生成 readiness report
- **WHEN** 用户运行 research dashboard readiness 输出
- **THEN** report MUST 包含 pending claim 数、可升级候选、缺失字段类别和 paper export gate 状态
- **AND** report MUST 标记 candidate-only 内容，不得声明正式论文结论

#### Scenario: HTML report 不替代正式文档
- **WHEN** dashboard 生成 HTML paper readiness report
- **THEN** HTML MUST 显示 candidate-only、pending、unverified 或 not_comparable caveat
- **AND** HTML MUST 不自动更新 `docs/result_claims_registry.md`、`docs/experiment_matrix.md`、`docs/mainline_model_catalog.md` 或 README
