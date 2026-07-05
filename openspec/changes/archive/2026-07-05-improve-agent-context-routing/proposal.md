## Why

仓库已经有强规则和长导航，但 agent 每次仍需要读取大量文档才能定位任务上下文。当前热门 agent/vibe-coding 工具的共同经验是：规则应按路径和任务渐进加载，项目知识应有高信号索引，重复流程应沉淀为技能或命令。

## What Changes

- 增加按目录或任务分层的 agent context 路由，避免根 `AGENTS.md` 承担所有细节。
- 将 `docs/agent_navigation.md` 收敛为入口索引，并为模型、配置、CLI、诊断、OpenSpec、文档等任务提供可按需读取的片段。
- 增加 spec/config/claim 的 agent-friendly atlas 或生成式索引，输出 capability、owner、focused tests、caveat 和 lifecycle。
- 增加项目级 Codex skills 或等价工作流说明，用于常见任务：新增模型、添加配置、更新 claim、诊断 run、归档 change。
- 明确 completed-but-unarchived change 的处理方式，减少 active/历史状态误读。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `ai-maintainer-navigation`: 增加分层 agent context、atlas、技能化工作流和 completed change 状态处理要求。
- `maintainer-context-index`: 增加机器可读索引服务 agent routing 的边界要求。
- `openspec-document-health`: 增加 agent context 文件和索引一致性的文档健康要求。

## Impact

- 可能新增 `docs/agent_context/`、`.codex/skills/` 项目技能、生成式 atlas 或 context 索引。
- 影响 AGENTS、agent navigation、inventory、maintainer context index 和架构边界测试。
- 不改变训练、评估、配置解析或模型 runtime。
