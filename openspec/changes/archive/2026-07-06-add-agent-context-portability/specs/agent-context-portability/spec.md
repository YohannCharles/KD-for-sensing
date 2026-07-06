## ADDED Requirements

### Requirement: 工具中立 agent context 适配
项目 MUST 提供工具中立的 agent context 适配策略，使 Codex、Claude Code、Cursor、GitHub Copilot、Kiro 或其它 agent 工具能定位同一套项目规则。`AGENTS.md`、`docs/agent_navigation.md`、`docs/agent_context/README.md` 和当前 OpenSpec specs MUST 继续作为权威来源；工具专属适配文件 MUST 保持薄引用，不得复制完整规则体系。

#### Scenario: 工具适配文件引用权威入口
- **WHEN** 仓库包含 `CLAUDE.md`、`.github/copilot-instructions.md`、`.cursor/rules/*.mdc`、`.kiro/steering/*.md`、`replit.md` 或等价 agent 适配文件
- **THEN** 适配文件 MUST 指向 `AGENTS.md` 或当前 agent navigation
- **AND** 适配文件 MUST 说明非平凡改动仍需遵守 OpenSpec change、`kd_mm_beam` 命令环境和本地产物边界

#### Scenario: 适配文件不得复制完整治理事实
- **WHEN** 适配文件描述任务路由、退役路线、验证命令或文档生命周期
- **THEN** 它 MUST 只保留短摘要或链接
- **AND** 它 MUST NOT 维护完整源码目录清单、完整 retired token 清单、完整 OpenSpec requirement 或完整 claim 表

### Requirement: 当前研究简报
项目 MUST 提供或记录一份短研究简报，用于帮助 agent 快速理解当前论文/实验主线、冻结方法、不要追的路线、claim 升级条件和下一步证据缺口。该简报 MUST 不替代主线模型目录、实验协议表、claim registry、experiment matrix 或 OpenSpec specs。

#### Scenario: 研究简报覆盖当前主线
- **WHEN** AI agent 读取研究简报
- **THEN** 文档 MUST 标明当前主线、主要 baseline/control、当前冻结方法、pending evidence 和下一步高价值实验
- **AND** 文档 MUST 明确退役路线不得恢复为当前 CLI、config、registry 或 package facade

#### Scenario: 研究简报不冒充正式 claim
- **WHEN** 简报提到本地结果、dashboard candidate、mock/smoke 或 pending evidence
- **THEN** 文档 MUST 标明 claim status 或指向 `docs/result_claims_registry.md`
- **AND** 文档 MUST NOT 把未审阅、candidate-only、pending、mock/smoke 或 not-comparable 数值写成正式论文结论

### Requirement: Agent 复盘和记忆候选
项目 MUST 为重复 AI 失误提供可审查的复盘或记忆候选机制。该机制 MUST 记录错误模式、正确规则、建议沉淀位置和验证命令；正式长期文档更新仍 MUST 由人工确认或 OpenSpec change 完成。

#### Scenario: 重复错误进入候选清单
- **WHEN** AI agent 第二次犯同类项目规则错误，例如恢复退役入口、漏用 `kd_mm_beam`、把 ignored outputs 当源码或误读 active change
- **THEN** 维护者 MAY 将该错误记录为记忆候选
- **AND** 候选记录 MUST 指向应更新的 `AGENTS.md`、navigation、scoped context、skill、test 或 OpenSpec artifact

#### Scenario: 记忆候选不自动改权威文档
- **WHEN** 记忆候选被创建或更新
- **THEN** 系统 MUST NOT 自动重写 README、OpenSpec current specs、claim registry 或 `AGENTS.md`
- **AND** 正式沉淀 MUST 通过人工确认、focused documentation change 或 OpenSpec change 完成

### Requirement: 只读角色 agent 和 skills
项目 MAY 定义只读角色 agent 或 skills，用于 claim audit、experiment triage、surface doctor review、literature scouting 或其它高噪声分析任务。只读角色 MUST 不修改源码、OpenSpec、README、claim registry、配置、运行产物或 checkpoint。

#### Scenario: 只读角色返回建议
- **WHEN** 用户或主 agent 调用 claim auditor、experiment triage、surface doctor reviewer 或等价角色
- **THEN** 该角色 MUST 只读取允许的 tracked docs/source 或用户明确指定的本地产物
- **AND** 输出 MUST 是建议、风险、缺口或候选任务，不得直接修改文件

#### Scenario: 角色不得绕过项目边界
- **WHEN** 角色 agent 需要运行 Python 检查或引用项目命令
- **THEN** 命令 MUST 使用 `conda run -n kd_mm_beam ...`
- **AND** 角色 MUST 不启动真实训练、清理本地产物、提交 checkpoint、恢复退役入口或绕过 `src/kd_sensing` 包结构

### Requirement: Agent context portability 验证
项目健康检查或文档检查 MUST 能验证 agent context portability 的关键边界。检查 MUST 不读取真实 `dataset/`、不加载 checkpoint、不启动训练、不写入 ignored runtime artifacts。

#### Scenario: 适配文件漂移被发现
- **WHEN** 适配文件引用不存在的导航文档、推荐退役入口、遗漏 `kd_mm_beam` 或复制过期命令
- **THEN** focused validation MUST 失败或报告 warning
- **AND** 失败信息 MUST 指向适配文件和建议修复入口

#### Scenario: 角色说明越权被发现
- **WHEN** 角色 agent/skill 描述允许自动修改正式 claim 文档、自动 archive change、自动清理 outputs 或绕过 OpenSpec
- **THEN** 文档健康检查 MUST 失败
- **AND** 修复建议 MUST 要求改为只读建议或显式主 agent 实施流程
