## ADDED Requirements

### Requirement: 已完成 active change 必须收口
当 `openspec list --json` 显示 active change 已完成所有任务或 artifact 时，后续表面清理 change MUST 先归档该 change，或在 proposal/design/tasks 中明确说明暂不归档的原因和影响范围。维护者 MUST NOT 将已完成但未归档的 active change 误读为仍在实施的需求。

#### Scenario: 已完成 RBMA change 仍 active
- **WHEN** `add-rbma-prototype-kd-missing-workflow` 或其它 change 显示 status 为 complete
- **THEN** 本次支持面清理 MUST 先执行归档或记录明确 deferral
- **AND** 后续 inventory、current specs 和文档解释 MUST 以归档后的 current surface 或记录的 deferral 为准

#### Scenario: active change 未完成
- **WHEN** active change 仍缺 tasks、design、specs 或实现验证
- **THEN** 维护者 MUST 将其视为当前上下文
- **AND** 新的清理 change MUST 避免覆盖该 active change 范围内的用户工作

### Requirement: 退役 tombstone 折叠必须保留 guard 价值
retired-tombstone spec MAY 被归档或折叠到集中 retired summary，但只有在该 spec 不再提供 current registry/config/CLI/documentation migration guard、外部迁移说明或 wording 防回流价值时才允许。保留的 tombstone MUST 明确记录其 guard 价值；折叠后的 summary MUST 继续说明旧路线不属于当前支持面。

#### Scenario: tombstone 仍有 guard 价值
- **WHEN** retired spec 仍对应 registry removed guard、配置拒绝路径、CLI 退役错误或当前文档防回流说明
- **THEN** 该 spec MUST 保留在 current specs 或被等价 guard summary 覆盖
- **AND** 折叠不得导致旧入口被误判为 unknown current capability

#### Scenario: tombstone 只剩历史叙述
- **WHEN** retired spec 没有 current guard、没有当前文档引用、没有配置/registry/CLI 拒绝边界，也没有迁移说明价值
- **THEN** 项目 MAY 将其归档或集中到 retired summary
- **AND** archive 或 summary MUST 明确该路线不再作为 current workflow、配置、模型或 CLI 维护

### Requirement: 归档脚手架不得进入 current lifecycle
归档 change 生成或修改 current spec 后，维护者 MUST 清理 `TBD` Purpose、模板说明和不完整 lifecycle 分类。current lifecycle 只表示能力契约仍属于当前支持面，不表示 pending/mock/unverified 数值 claim 已经成立。

#### Scenario: 归档后 Purpose 未清理
- **WHEN** current spec 的 Purpose 仍是归档工具生成的占位文本
- **THEN** lifecycle 收口 MUST 被视为未完成
- **AND** 架构边界或 OpenSpec hygiene 检查 MUST 要求补全真实 Purpose

#### Scenario: pending 能力仍为 current
- **WHEN** capability 是 current 入口或契约但结果 claim 仍为 pending、mock/smoke 或 unverified
- **THEN** lifecycle inventory MAY 将其标记为 current
- **AND** mainline catalog、experiment protocols 和 result claims registry MUST 明确标注 claim caveat
