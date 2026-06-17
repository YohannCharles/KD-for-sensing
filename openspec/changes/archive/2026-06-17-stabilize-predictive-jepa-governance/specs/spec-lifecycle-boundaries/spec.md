## ADDED Requirements

### Requirement: 归档后状态收口
OpenSpec change 归档后，维护者 MUST 以 `openspec list --json`、current specs、lifecycle inventory 和 git status 的组合判断项目状态。归档目录存在或未跟踪 archive 文件 MUST 被解释为收口状态，不得单独视为 active change。

#### Scenario: 无 active change 但存在未跟踪 archive
- **WHEN** `openspec list --json` 返回空 change 列表，但 `git status --short` 显示 `openspec/changes/archive/<date>-<change>/` 未跟踪
- **THEN** agent MUST 报告这是归档后待提交或待清理状态
- **AND** agent MUST NOT 将该 archive 目录当作仍在实施的 active change

#### Scenario: current spec 来自归档 change
- **WHEN** 归档 change 新增或修改了 `openspec/specs/<capability>/spec.md`
- **THEN** 同一收口工作 MUST 确认该 capability 的 lifecycle inventory 已更新
- **AND** spec 的 Purpose MUST 使用真实当前能力说明，而不是保留归档 scaffold 占位文本

### Requirement: lifecycle 与文档 caveat 同步
新增 current capability 的 lifecycle 分类 MUST 与 README、主线模型目录、实验协议表和 claim 账本中的状态 caveat 保持一致。若 capability 仍处于 pending、mock/smoke、blocked 或 unverified 状态，文档 MUST 明确说明不可作为正式 claim。

#### Scenario: pending capability 被分类为 current
- **WHEN** lifecycle inventory 将新 capability 标记为 `current`，但该能力的实验结果仍为 `pending`、`mock/smoke`、`blocked` 或 `unverified`
- **THEN** current 文档 MUST 区分“能力入口/契约为 current”和“数值 claim 尚未 verified”
- **AND** 文档 MUST 不把 smoke 或 synthetic 指标写成真实结果
