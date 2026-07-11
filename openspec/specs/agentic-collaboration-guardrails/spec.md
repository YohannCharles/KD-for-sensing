# agentic-collaboration-guardrails Specification

## Purpose
定义多人和多 agent 协作中的 PR/Issue 模板、本地/手动验证、安全/产物扫描、脏工作树收口 preflight 与 AI review 边界；这些护栏只帮助审查和报告，不自动 merge、升级 claim、清理产物或覆盖用户改动。
## Requirements
### Requirement: Agentic PR/Issue 协作模板
项目 MUST 提供或记录 PR/Issue 协作模板，用于结构化说明 OpenSpec change、改动范围、验证命令、产物边界、claim 状态和回滚/停止条件。模板 MUST 帮助 agent 和人类 reviewer 判断改动是否可审查，而不是替代 OpenSpec 或 tests。

#### Scenario: PR 模板包含验证和边界
- **WHEN** 开发者或 agent 创建 PR 描述
- **THEN** 模板 SHOULD 要求填写 change id 或窄修复说明、触碰范围、运行的验证命令、未运行验证原因和本地产物边界
- **AND** 如改动影响 claim、paper table、dashboard 或实验结论，模板 MUST 要求标明 claim status 和 provenance 状态

#### Scenario: Issue 模板可启动 agent 任务
- **WHEN** 用户希望将 issue 委派给 coding agent
- **THEN** issue SHOULD 包含目标、非目标、相关 OpenSpec/docs、允许读取的本地产物、预期验证和禁止触碰路径
- **AND** issue MUST 不要求 agent 提交真实 dataset、outputs、logs、cache 或 checkpoint

### Requirement: 本地/手动验证和调度友好检查
项目 MUST 提供本地/手动验证策略。默认 quick verify MUST 保持无真实数据、无训练、无 checkpoint；OpenSpec、architecture、CLI/config、compile 和小型安全扫描 MAY 由人工或任务系统组合运行。项目 MUST 不要求 surface doctor 或 research preview 作为验证编排层。

#### Scenario: Quick verify 无训练
- **WHEN** 开发者或 agent 运行 quick verify
- **THEN** 检查 MUST 运行 OpenSpec strict 和架构边界或等价 quick checks
- **AND** MUST 不启动真实训练、不读取 dataset、不加载 checkpoint

#### Scenario: Focused checks 直接报告漂移
- **WHEN** 人工运行 security、CLI/config 或 compile check
- **THEN** 原生命令 MUST 报告路径和失败原因
- **AND** 输出 MUST 不修改源码、OpenSpec、本地产物或用户改动

### Requirement: 安全、依赖和产物扫描
项目 MUST 提供或记录安全与产物扫描，用于发现 secrets、系统启动/认证配置污染、tracked runtime artifacts、危险 shell runner 和依赖风险。扫描 MUST 只读取 tracked source/docs/config/spec/tests/scripts 和必要 git metadata。

#### Scenario: secrets 和系统配置风险
- **WHEN** 扫描发现 token、private key、密码、`/root/.container_env` 修改建议、系统 profile 修改或将训练命令写入凭证字段的内容
- **THEN** 检查 MUST 失败或至少报告 high-severity warning
- **AND** 修复建议 MUST 指向 AGENTS 的系统配置与启动项安全边界

#### Scenario: runtime artifacts 被跟踪
- **WHEN** git tracked files 包含真实 dataset 内容、outputs、logs、cache、TensorBoard event 或新 checkpoint
- **THEN** 检查 MUST 失败
- **AND** 失败信息 MUST 说明这些产物应留在 ignored 本地路径

#### Scenario: shell runner 危险命令
- **WHEN** tracked shell 或 Python runner 包含无确认删除、裸 `rm -rf`、后台长跑污染系统启动配置或绕过 `kd_mm_beam` 的项目 Python 命令
- **THEN** 检查 SHOULD 报告 warning 或失败
- **AND** 检查 MUST 允许现有 manifest-backed cleanup 的显式确认流程

### Requirement: 脏工作树和 OpenSpec 收口 preflight
协作流程 MUST 通过 `git status --short`、`openspec list --json` 和必要 status/validate 命令报告 active change、complete 未归档 change、dirty tracked files 和 runtime artifact 边界。Preflight MUST 不依赖 project surface doctor/research preview，不自动 archive、不删除、不 reset、不覆盖用户改动。

#### Scenario: Active/complete change 状态清晰
- **WHEN** preflight 发现 complete 未归档或重叠 active change
- **THEN** implementation MUST 记录 archive、abandon、scope correction 或 deferral 决策
- **AND** MUST 不把其过时 artifact 当作 current implementation 事实

#### Scenario: Dirty worktree 不被自动清理
- **WHEN** git status 显示用户改动
- **THEN** implementation MUST 记录并避开该 diff
- **AND** MUST NOT reset、checkout、删除或覆盖文件

### Requirement: AI review 作为附加信号
项目 MAY 集成 Codex、GitHub Copilot 或其它 AI review 信号，用于发现 regression、missing tests、安全问题、claim caveat 缺失和 OpenSpec drift。AI review MUST 不替代 human review、OpenSpec validate 和 focused tests。

#### Scenario: AI review 不自动 merge 或升级 claim
- **WHEN** AI review 给出通过或修复建议
- **THEN** PR 仍 MUST 通过项目要求的验证命令和人工审查
- **AND** AI review MUST NOT 自动将 candidate/pending claim 升级为 reviewed claim

#### Scenario: AI review 可请求修复
- **WHEN** reviewer 要求 agent 修复某个明确问题
- **THEN** agent MUST 在同一 change/PR 范围内修改
- **AND** agent MUST 汇报运行或未运行的 focused validation
