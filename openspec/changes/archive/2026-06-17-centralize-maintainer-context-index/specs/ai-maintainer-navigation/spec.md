## ADDED Requirements

### Requirement: 导航优先读取维护上下文索引
AI 维护导航文档 SHALL 将中心化维护上下文索引纳入非平凡改动前的当前状态检查顺序。导航文档 MUST 要求 agent 先通过索引定位任务路由、治理表和验证命令，再按需读取 README、project surface inventory、OpenSpec specs 和源码。

#### Scenario: 非平凡改动前读取索引
- **WHEN** AI agent 计划修改模型、数据契约、配置、CLI、诊断 workflow、OpenSpec artifact 或文档生命周期
- **THEN** `docs/agent_navigation.md` MUST 指向维护上下文索引
- **AND** 导航文档 MUST 说明索引用于快速定位上下文，不替代 OpenSpec requirements 或 README quickstart

#### Scenario: 当前打开文件不覆盖索引路由
- **WHEN** IDE 当前打开文件是薄 CLI alias、generated metadata、测试 allowlist 或本地输出摘要
- **THEN** 导航文档 MUST 要求 agent 使用维护上下文索引确认该文件所属 lifecycle 和任务路由
- **AND** agent MUST 不把当前打开文件单独视为项目权威入口

### Requirement: 导航说明索引与 inventory 的职责边界
AI 维护导航文档 SHALL 说明维护上下文索引与 `docs/project_surface_inventory.md` 的职责边界。索引 MUST 被描述为机器可读的治理事实入口，inventory MUST 被描述为解释性审计和历史上下文来源。

#### Scenario: 读取 lifecycle 时知道事实来源
- **WHEN** AI agent 需要判断某 capability、entrypoint、config 或热点是否属于当前支持面
- **THEN** 导航文档 MUST 指向维护上下文索引中的结构化分类
- **AND** 导航文档 MUST 指向 inventory 或对应 OpenSpec spec 以理解分类原因和 caveat

#### Scenario: 文档表格与索引冲突时有处理方式
- **WHEN** inventory、README、导航文档和维护上下文索引之间出现看似冲突的分类或入口说明
- **THEN** 导航文档 MUST 要求把它视为治理漂移
- **AND** 后续变更 MUST 通过 OpenSpec change 同步索引、inventory 和对应 specs，而不是任选一处作为事实
