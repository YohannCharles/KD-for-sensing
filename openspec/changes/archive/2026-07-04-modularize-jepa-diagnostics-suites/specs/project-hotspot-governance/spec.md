## ADDED Requirements

### Requirement: JEPA diagnostics 热点预算必须随拆分更新
Project hotspot governance MUST record the post-refactor JEPA visual analysis and benchmark owner modules, facade budgets, accepted temporary hotspots and focused validation commands.

#### Scenario: JEPA owner 清单更新
- **WHEN** JEPA diagnostics modules are split or renamed
- **THEN** project surface inventory MUST list the new owner modules and describe which public facade remains
- **AND** architecture tests MUST 继续拒绝 suite-specific helper logic 的 facade 回流
