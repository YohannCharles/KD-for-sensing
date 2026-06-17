## ADDED Requirements

### Requirement: Hotspot budget 行动元数据
维护上下文索引 SHALL 为 hotspot file budgets 和 symbol budgets 提供机器可读行动元数据。每个 budget entry MUST 记录 priority、status、split targets、rationale 和 validation commands，且 MAY 记录推荐 next change。

#### Scenario: hotspot entry 包含行动字段
- **WHEN** `docs/maintainer_context_index.yaml` 登记 file 或 symbol budget
- **THEN** 每个 entry MUST 包含 `priority`、`status`、`split_targets`、`rationale` 和 `validation_commands`
- **AND** `priority` 和 `status` MUST 使用索引声明的允许值

#### Scenario: Codex 可从索引定位下一步
- **WHEN** AI agent 读取 hotspot budget
- **THEN** 索引 MUST 提供足以定位下一步拆分方向的 `split_targets` 或 `next_change`
- **AND** 详细 caveat 可继续由 `docs/project_surface_inventory.md` 提供

### Requirement: Hotspot metadata 不替代 inventory 解释
维护上下文索引中的 hotspot metadata SHALL 作为机器可读行动摘要。inventory MUST 继续保留热点原因、暂缓解释和审计上下文；二者看似冲突时 MUST 被视为治理漂移。

#### Scenario: inventory 提供长解释
- **WHEN** hotspot budget 在索引中登记
- **THEN** `docs/project_surface_inventory.md` MUST 继续包含该路径或 symbol 的解释性条目
- **AND** 索引 `rationale` MUST 是短摘要，不得替代 inventory 的审计说明
