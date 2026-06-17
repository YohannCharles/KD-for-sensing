## ADDED Requirements

### Requirement: package CLI 索引双向同步
维护上下文索引 SHALL 将 package CLI 视为 pyproject console scripts 的机器可读分类，而不是单向备注。索引中的 package CLI 集合 MUST 与 `pyproject.toml` 的 `[project.scripts]` 保持双向一致。

#### Scenario: package CLI 完整登记
- **WHEN** 项目声明 package console script
- **THEN** 维护上下文索引 MUST 登记该 script 的 name、target 和 lifecycle
- **AND** lifecycle MUST 属于索引允许的 entrypoint lifecycle values

#### Scenario: 删除 CLI 同步索引
- **WHEN** 某 package console script 从 pyproject 删除
- **THEN** 维护上下文索引 MUST 同步删除或重新分类该入口
- **AND** 架构边界测试 MUST 不允许 stale package CLI 登记长期存在
