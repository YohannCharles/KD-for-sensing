## ADDED Requirements

### Requirement: Canonical config 解析必须拆分 recipe 与 migration guard
Canonical config 重构 MUST 将 virtual recipe generation、overlay resolution、path alias handling 和 retired-route migration guards 保持为独立职责，并保持 load error 兼容。

#### Scenario: retired route 不被 virtual config 接管
- **WHEN** user loads a retired config path or retired KD alias
- **THEN** config loading MUST 按既有 migration guard 语义 fast fail
- **AND** virtual config generation MUST NOT create replacement configs for retired research lines
