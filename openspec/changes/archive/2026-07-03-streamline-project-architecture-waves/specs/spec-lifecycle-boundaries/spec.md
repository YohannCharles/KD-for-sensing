## ADDED Requirements

### Requirement: Retired tombstones require guard-value audit before retention
仍保留在 `openspec/specs/` 下的 retired-tombstone capability MUST 在本 change 中复核 guard 价值。Guard 价值包括 registry removed guard、config path/override 拒绝、CLI retired error、current docs wording 防回流、外部迁移说明或 focused tests 防回归。无 guard 价值的 tombstone MAY 归档或折叠到集中 retired summary。

#### Scenario: Tombstone 保留有理由
- **WHEN** retired-tombstone spec 继续保留在 current specs 下
- **THEN** inventory 或 spec 开头 MUST 能说明它提供的 guard 价值
- **AND** 文档 MUST 不把该能力描述为 current workflow、current config 或 current CLI

#### Scenario: Tombstone 可折叠
- **WHEN** retired spec 没有 current registry/config/CLI/docs/tests guard，也没有迁移说明价值
- **THEN** 本 change MAY 将其归档或折叠到集中 retired summary
- **AND** summary MUST 继续说明旧路线不属于当前支持面，且不得恢复旧入口

### Requirement: Completed active changes are resolved before surface cleanup
当 `openspec list --json` 显示 active change 的 artifacts/tasks 已完成时，本 change 的 Wave 0 MUST 先归档该 change，或记录明确 deferral。后续 docs、inventory 和 specs 不得把已完成但未归档的 change 误读为仍在实施的需求。

#### Scenario: Complete change deferred
- **WHEN** 已完成 active change 因用户工作树、审查或提交节奏暂不归档
- **THEN** Wave 0 MUST 记录 change name、暂缓原因、与本 change 的重叠范围和后续归档触发条件
- **AND** 后续 wave MUST 避免覆盖该 change 范围内未收口的用户工作

### Requirement: Lifecycle cleanup cannot weaken retired-route guards
折叠 OpenSpec tombstone、删除历史 wording 或收缩 migration guard 时，项目 MUST 保持 retired route 的实际拒绝边界。若删除某个专属 guard，必须证明普通 unknown-name 错误、集中 retired summary 或其它 guard 仍足以防止旧入口被误判为 current。

#### Scenario: 删除专属 guard 前验证
- **WHEN** 本 change 删除 registry/config/CLI 中某个 retired 名称的专属错误或文档段落
- **THEN** focused tests 或 architecture boundary tests MUST 验证该旧名称仍不会构建、不会被 virtual config 接管、不会出现在 current docs 推荐入口中
- **AND** 删除理由 MUST 说明迁移路径或 unknown-name fallback 是否可接受
