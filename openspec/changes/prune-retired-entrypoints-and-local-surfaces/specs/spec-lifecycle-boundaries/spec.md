## ADDED Requirements

### Requirement: Retired tombstones are folded unless they provide active guard value
Retired-tombstone specs MUST 只在提供独立 guard 价值时保留。Guard 价值包括 registry removed-name guard、config path/override 拒绝、CLI retired error、current docs wording 防回流、外部迁移说明或 focused tests 防回归。只重复“已退役/不得回流”的 tombstone MUST 折叠到集中 retired-route summary 或归档。

#### Scenario: Tombstone without unique guard
- **WHEN** 一个 retired-tombstone spec 没有独立 registry/config/CLI/docs/tests guard 价值
- **THEN** 它 MUST 从 current specs 中移除或折叠
- **AND** 集中 retired-route summary MUST 保留旧路线不属于 current support surface 的事实

#### Scenario: Retired summary preserves rejection
- **WHEN** 多个 retired specs 被折叠
- **THEN** 项目 MUST 保留旧名称、拒绝点和迁移方向的集中记录
- **AND** 防回流测试 MUST 继续验证旧入口不会恢复
