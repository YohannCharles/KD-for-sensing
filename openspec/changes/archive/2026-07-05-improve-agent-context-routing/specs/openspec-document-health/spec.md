## ADDED Requirements

### Requirement: Agent context 文件纳入文档健康
项目文档健康检查 MUST 覆盖 agent context、atlas 或项目 skills 的引用一致性。新增 scoped context 或技能时，必须能从 AGENTS、agent navigation、inventory 或技能清单中定位其用途和适用范围。

#### Scenario: scoped context 引用失效
- **WHEN** scoped agent context 文件引用不存在的 spec、config、owner module 或验证命令
- **THEN** 文档健康或架构边界检查 MUST 失败
- **AND** 失败信息 MUST 指向失效引用

#### Scenario: 技能说明绕过 OpenSpec
- **WHEN** 项目级技能描述要求直接修改非平凡功能但不提 OpenSpec
- **THEN** 文档健康检查 MUST 要求补充 OpenSpec change 边界
- **AND** 技能 MUST 不把自己描述为需求契约权威
