## ADDED Requirements

### Requirement: Agent context 渐进加载
AI 维护导航 SHALL 支持按任务渐进加载上下文。根 `AGENTS.md` 和 `docs/agent_navigation.md` MUST 保持为高信号入口；超出根入口职责的模型、数据、配置、CLI、诊断、OpenSpec、文档和实验 claim 等细节 MUST 放入 scoped context 文件、atlas 或项目技能中。Scoped context MUST 指向权威 specs、inventory、owner modules 和 focused validation，而不得成为第二套完整事实源。

#### Scenario: 模型任务加载模型上下文
- **WHEN** AI agent 处理模型、forward、registry 或 baseline 配置任务
- **THEN** 导航 MUST 指向模型相关 scoped context 或等价 atlas 入口
- **AND** 该上下文 MUST 包含 owner modules、相关 specs、禁止恢复退役路线和 focused tests

#### Scenario: 文档任务不加载训练细节
- **WHEN** AI agent 只处理 README、OpenSpec 或 docs 生命周期改动
- **THEN** 导航 MUST 允许 agent 读取文档生命周期上下文
- **AND** agent 不需要读取无关模型训练流程全文

### Requirement: Completed change 状态处理
AI 维护导航 MUST 明确已完成但未归档的 OpenSpec change 的处理方式。Agent MUST 通过 `openspec list --json` 和 `openspec status --change <name>` 判断该 change 是否 complete；complete 但未 archive 的 change MUST 被标记为治理收口项，而不是当作仍在实施的新需求。

#### Scenario: complete active change 不被误当实施中
- **WHEN** `openspec list --json` 返回 status 为 `complete` 的 change
- **THEN** agent MUST 将其作为待归档或待提交状态处理
- **AND** agent MUST 不把该 change 当作当前新功能范围，除非用户明确要求继续该 change

### Requirement: 项目级 agent skills
项目 MUST 为高频维护流程提供项目级 agent skills 或等价命令说明。技能 MUST 使用渐进披露，只在任务匹配时加载完整说明，并 MUST 不绕过 OpenSpec、`kd_mm_beam` 环境和本地产物边界。

#### Scenario: 新增模型走模型技能
- **WHEN** 用户要求新增普通 baseline 或模型组件
- **THEN** agent MUST 使用模型扩展技能或等价流程
- **AND** 该流程 MUST 要求先判断 config-only、component baseline、whole-model exception 或 workflow reproduction

#### Scenario: 更新 claim 走 claim 技能
- **WHEN** 用户要求把本地结果写入 claim registry 或论文表格
- **THEN** agent MUST 使用 claim 更新技能或等价流程
- **AND** 该流程 MUST 检查 provenance、status、caveat 和 ignored output boundary
