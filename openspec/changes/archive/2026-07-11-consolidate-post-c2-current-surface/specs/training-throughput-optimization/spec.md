## REMOVED Requirements

### Requirement: Throughput profiling is local-only
**Reason**: 该 tombstone 只记录已删除 CLI，不需继续作为独立 current capability。
**Migration**: 退役命令名由 `retired-route-summary` 防止恢复；本地 profiling 直接包裹 current training command。

#### Scenario: throughput CLI 仍保持缺席
- **WHEN** maintainers 检查 console scripts 和文档
- **THEN** 集中 retired-route guard MUST 保持 standalone throughput CLI 缺席
