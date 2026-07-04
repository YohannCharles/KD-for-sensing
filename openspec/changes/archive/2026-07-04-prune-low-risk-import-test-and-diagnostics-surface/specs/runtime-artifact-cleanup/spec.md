## ADDED Requirements

### Requirement: Runtime cleanup 拆分必须保留 dry-run 与确认删除边界
Runtime artifact cleanup 重构 MUST 拆分扫描规则、manifest 渲染、delete/apply 校验和 organize 计划，并保留默认 dry-run 与破坏性操作显式确认。

#### Scenario: 删除仍需 manifest 与确认
- **WHEN** user invokes cleanup delete mode
- **THEN** 命令 MUST 要求同时传入 `--delete`、`--manifest <path>` 和 `--confirm-delete`
- **AND** 实现 MUST 在删除任何候选前重新校验 tracked/protected/path 状态
