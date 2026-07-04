## ADDED Requirements

### Requirement: JEPA downstream query/pooling helper 必须可独立演进
JEPA downstream model 重构 MUST 在不改变 current config 行为的前提下拆分 query construction、token pooling、GPS-query compatibility、head construction 和 diagnostics metadata。

#### Scenario: downstream config 兼容
- **WHEN** downstream helper modules are introduced
- **THEN** 既有 JEPA downstream configs MUST 加载并构建相同 model surface
- **AND** GPS-query/pooler diagnostics fields MUST remain available to visual analysis and benchmark workflows
