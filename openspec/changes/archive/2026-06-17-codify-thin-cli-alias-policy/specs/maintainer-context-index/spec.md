## ADDED Requirements

### Requirement: Entrypoint owner metadata
维护上下文索引 SHALL 为长期保留 entrypoint 记录 owner metadata。每个 package CLI、thin alias、research diagnostic、dataset preparation 和 shell orchestration entry MUST 记录 owner module 或 owner script、responsibility、output boundary 和 lifecycle。

#### Scenario: entrypoint metadata 完整
- **WHEN** entrypoint 出现在维护上下文索引
- **THEN** entry MUST 包含 lifecycle、owner module 或 owner script、responsibility 和 output boundary
- **AND** output boundary MUST 表明 read-only、ignored outputs/logs/cache、dataset preparation target 或显式用户路径

#### Scenario: retired route guard 可审计
- **WHEN** entrypoint 名称、owner module 或参数容易与退役路线混淆
- **THEN** 索引 MUST 记录 retired route guard 或 caveat
- **AND** inventory MUST 保留人类可读解释
