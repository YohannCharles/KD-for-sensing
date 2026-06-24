## MODIFIED Requirements

### Requirement: 已删除组件错误可诊断
当用户引用已删除的兼容组件名称或退役研究线组件名称时，注册表错误 MUST 至少包含请求名称、registry 名称或可用名称上下文。对于仍有当前迁移价值的名称，错误 MUST 区分“未知名称”和“已删除名称”并给出迁移方向；对于完全退役且不再承诺兼容的历史名称，系统 MUST 允许使用普通 unknown-name 错误、配置 migration guard 或集中退役说明替代长期 removed guard table。registry 实现 MUST 不为了历史说明长期维护没有 current migration value 的 removed-name 表项。

#### Scenario: 已删除 dataset type
- **WHEN** 用户请求构建 `scenario9` dataset 且项目仍保留该迁移说明
- **THEN** 系统 MUST 抛出包含 `scenario9` 的错误
- **AND** 错误信息 MUST 说明该名称已删除并给出 `deepsense6g + scene` 配置示例

#### Scenario: 已删除模型 alias
- **WHEN** 用户请求旧 fusion 类名 alias 或已删除 image encoder alias，且该名称仍在 current migration table 中
- **THEN** 系统 MUST 抛出包含请求名称的错误
- **AND** 错误信息 MUST 列出当前支持的 canonical 注册名

#### Scenario: 已退役研究线组件
- **WHEN** 用户请求 `craf_fusion`、`marf_fusion`、`g2d` distiller 或 `multimodal_nf` dataset
- **THEN** 系统 MUST 拒绝构建
- **AND** 系统 MUST 不通过 deprecated alias、overlay 或兼容 facade 重定向到其它实现

#### Scenario: 完全退役名称不要求 removed table
- **WHEN** 某个历史组件名称已经由 retired-tombstone spec、配置 migration guard 或文档生命周期边界覆盖，且没有当前迁移路径
- **THEN** registry MUST 允许不在 removed-name table 中保留该名称
- **AND** unknown-name 错误 MUST 仍列出 registry 名称、请求名称或可用 canonical 名称上下文
