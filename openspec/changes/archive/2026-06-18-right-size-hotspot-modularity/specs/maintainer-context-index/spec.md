## MODIFIED Requirements

### Requirement: Hotspot budget 行动元数据
维护上下文索引 SHALL 为 hotspot file budgets 和 symbol budgets 提供机器可读行动元数据。每个 budget entry MUST 记录 priority、status、rationale、validation commands 和至少一种后续动作线索；后续动作线索 MAY 是 split targets、consolidation targets、accepted-size rationale 或推荐 next change。索引 MUST 能表达硬预算、软预算、理由化例外、合并候选和右尺寸化接受状态。

#### Scenario: hotspot entry 包含行动字段
- **WHEN** `docs/maintainer_context_index.yaml` 登记 file 或 symbol budget
- **THEN** 每个 entry MUST 包含 `priority`、`status`、`rationale` 和 `validation_commands`
- **AND** entry MUST 包含 `split_targets`、`consolidation_targets`、`accepted_size_rationale` 或 `next_change` 中至少一种后续动作线索
- **AND** `priority`、`status` 和 enforcement 相关字段 MUST 使用索引声明的允许值

#### Scenario: Codex 可从索引定位下一步
- **WHEN** AI agent 读取 hotspot budget
- **THEN** 索引 MUST 提供足以判断下一步应拆分、合并、监控、接受当前尺寸或调整预算的机器可读字段
- **AND** 详细 caveat 可继续由 `docs/project_surface_inventory.md` 提供

#### Scenario: facade 与业务热点 enforcement 可区分
- **WHEN** 架构边界测试读取热点预算
- **THEN** 索引 MUST 能区分 `facade-budget` 或等价硬预算状态与 `monitor`、`defer-with-rationale`、`right-size-accepted`、`merge-candidate` 等非硬预算状态
- **AND** facade 硬预算超限 MUST 被判为失败，业务热点 headroom 内的理由化超限 MAY 被接受

#### Scenario: 合并候选有明确 owner
- **WHEN** hotspot entry 被标记为 `merge-candidate` 或包含 `consolidation_targets`
- **THEN** 索引 MUST 记录合并目标、owner module 或足够明确的 owner 说明
- **AND** validation commands MUST 覆盖合并后可能受影响的公开契约

#### Scenario: right-size accepted 不替代验证
- **WHEN** hotspot entry 被标记为 `right-size-accepted` 或登记 accepted-size rationale
- **THEN** 索引 MUST 记录为什么保持当前尺寸比继续拆分更可维护
- **AND** entry MUST 保留 focused validation commands，防止该状态被解释为永不重构的永久豁免

## ADDED Requirements

### Requirement: Hotspot remediation wave metadata
维护上下文索引 SHALL 能为高风险热点修复记录 remediation wave metadata。每个 wave entry MUST 指明 wave id、目标路径、owner module、planned action、public surface policy、validation commands 和 rollback note。planned action MUST 能表达 split、consolidate、keep-and-test、owner-facade、hard-budget 或 accepted-size。

#### Scenario: wave metadata 可定位实施范围
- **WHEN** AI agent 或架构测试读取维护上下文索引
- **THEN** 索引 MUST 能列出当前 change 涉及的热点 wave、目标源码路径和 owner module
- **AND** 每个 wave MUST 有 planned action 和 focused validation commands

#### Scenario: public surface policy 防止误删入口
- **WHEN** wave 触碰已登记 CLI、public import owner 或 baseline reproduction module
- **THEN** 索引 MUST 记录 public surface policy，例如 keep-public-import、thin-owner、no-public-surface 或 remove-internal-only
- **AND** 架构测试或 focused tests MUST 能验证公开入口没有被意外删除

#### Scenario: keep-and-test 是合法动作
- **WHEN** hotspot 审核发现某个模块规模较小、职责内聚且继续拆分会增加跳转成本
- **THEN** 索引 MAY 将 planned action 记录为 keep-and-test
- **AND** entry MUST 说明保留理由和对应 focused tests
