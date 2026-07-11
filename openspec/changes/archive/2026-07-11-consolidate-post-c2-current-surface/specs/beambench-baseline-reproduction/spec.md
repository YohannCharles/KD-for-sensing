## REMOVED Requirements

### Requirement: BeamBench reproduction surface is retired
**Reason**: 单独 tombstone spec 只重复已有集中退役路由契约。
**Migration**: BeamBench 名称的 negative guard 并入 `retired-route-summary`。

#### Scenario: BeamBench 退役状态仍受保护
- **WHEN** maintainers 扫描 current CLI、configs、docs 和 registry
- **THEN** 集中 retired-route guard MUST 保持 BeamBench reproduction 入口缺席
