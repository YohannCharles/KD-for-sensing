## MODIFIED Requirements

### Requirement: OpenSpec capability lifecycle 分类
项目 MUST 维护 OpenSpec capability lifecycle 分类，用于区分当前能力、支撑能力和集中退役边界。`openspec/specs/` 下保留的 capability MUST 分类为 `current` 或 `supporting`；无独立 guard 价值的 retired capability MUST 从 current specs 删除并由 `retired-route-summary`、inventory historical row 或 archive 记录。`retired-tombstone` MAY 只用于确有独立运行时拒绝或外部迁移价值的例外。

#### Scenario: 每个保留 spec 被分类
- **WHEN** 架构边界测试枚举 `openspec/specs/*/spec.md`
- **THEN** 每个保留 capability MUST 在 lifecycle inventory 中有且仅有一个分类
- **AND** 未分类 capability MUST 被视为文档生命周期漂移

#### Scenario: 无独立 guard 的墓碑不占 current spec
- **WHEN** retired capability 只重复“已退役/不得回流”且无独立 config、registry、CLI、docs 或测试 guard
- **THEN** 该 spec MUST 折叠到集中 retired summary 或 archive
- **AND** lifecycle inventory MUST 不再要求该目录存在

#### Scenario: lifecycle 分类含义稳定
- **WHEN** 开发者阅读 lifecycle inventory
- **THEN** `current` MUST 表示当前需求契约或运行能力
- **AND** `supporting` MUST 表示被 current workflow 消费但不作为 standalone 推荐入口的能力
- **AND** 集中 retired summary MUST 表示旧路线只保留历史和拒绝边界

### Requirement: Retired tombstones are folded unless they provide active guard value
Retired capability MUST 只在提供独立 registry/config/CLI/docs/tests guard 或外部迁移价值时保留专属 current spec。只重复退役事实的 capability MUST 从 `openspec/specs/` 删除并由集中 summary 覆盖；本轮 27 个 inventory tombstone 中，24 个 MUST 折叠，三个存在 current consumer 的误分类 owner MUST 改为 supporting。

#### Scenario: Tombstone without unique guard
- **WHEN** retired spec 没有独立 guard 或迁移价值
- **THEN** 它 MUST 从 current specs 中移除
- **AND** 集中 retired summary MUST 保留旧名称和非 current 事实

#### Scenario: Retired summary preserves rejection
- **WHEN** 多个 retired specs 被折叠
- **THEN** 项目 MUST 保留代表性旧 CLI/config/module token 与普通 unknown-name 或集中 guard 语义
- **AND** 防回流测试 MUST 继续验证旧入口不会恢复

#### Scenario: Tombstone 分类被 consumer 证据纠正
- **WHEN** source audit 证明 capability 被 current MMW、training startup、U-Mask/AMR/AMBER 或 Scene31-34 workflow 消费
- **THEN** lifecycle inventory MUST 将该 capability 改为 `supporting` 而不是删除
- **AND** supporting spec MUST 只保留真实消费契约，不得借此恢复 standalone CLI 或历史 sweep

## REMOVED Requirements

### Requirement: 退役 tombstone 折叠必须保留 guard 价值
**Reason**: 与更新后的英文同名折叠 requirement 重复。
**Migration**: 统一使用 “Retired tombstones are folded unless they provide active guard value”。

#### Scenario: 重复折叠规则删除
- **WHEN** 维护者审计 tombstone
- **THEN** 只应用保留的集中折叠 requirement
- **AND** 不再维护重复版本

### Requirement: Retired tombstones require guard-value audit before retention
**Reason**: 与更新后的折叠 requirement 完全重叠。
**Migration**: Guard-value audit 由保留 requirement 和集中 summary 承接。

#### Scenario: Guard 审计规则归一
- **WHEN** retired spec 申请保留
- **THEN** 它 MUST 按保留 requirement 证明独立 guard 价值
- **AND** 不再引用本重复 requirement

### Requirement: Completed active changes are resolved before surface cleanup
**Reason**: 与“已完成 active change 必须收口”重复。
**Migration**: 使用中文 active-change 收口 requirement。

#### Scenario: Complete change 规则归一
- **WHEN** active change 显示完成
- **THEN** cleanup MUST 依据保留的收口 requirement 记录 archive 或 deferral
- **AND** 不再维护第二份英文要求
