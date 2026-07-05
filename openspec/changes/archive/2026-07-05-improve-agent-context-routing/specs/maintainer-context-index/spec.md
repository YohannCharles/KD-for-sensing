## ADDED Requirements

### Requirement: 维护上下文索引可服务 agent routing
维护上下文索引 MUST 提供足以支持 agent routing 的最小结构化字段，例如 task route id、权威文档路径、owner module、focused validation 和 retired-route guard。索引 MUST NOT 扩展为完整源码目录镜像、完整 config 数据库或完整 OpenSpec requirement 复制。

#### Scenario: agent 查询任务路由
- **WHEN** agent 需要判断模型、配置、CLI、诊断、claim 或文档任务的先读内容
- **THEN** 维护上下文索引 MUST 提供对应 route 的最小事实
- **AND** 详细 rationale MUST 仍由 inventory、README 或 OpenSpec spec 提供

### Requirement: Atlas 输出引用权威来源
如果项目提供 spec/config/claim atlas，atlas MUST 引用权威路径、capability lifecycle、owner、验证命令和 caveat，并 MUST 标记生成时间或来源。Atlas MUST 不覆盖 current specs、inventory 或 claim registry。

#### Scenario: atlas 与 inventory 冲突
- **WHEN** atlas 和 inventory 对同一 capability lifecycle 给出冲突信息
- **THEN** agent MUST 将其视为治理漂移
- **AND** 后续变更 MUST 同步 atlas 生成源、inventory 和相关 specs
