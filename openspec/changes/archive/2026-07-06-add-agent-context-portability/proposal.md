## Why

项目已经有 `AGENTS.md`、OpenSpec、scoped context 和项目级 skills，但这些规则主要服务当前 Codex 工作流；换到 Claude Code、Cursor、GitHub Copilot、Kiro、Replit/Lovable/Bolt 风格工具时，agent 仍可能漏读关键边界、重复探索或恢复退役路线。现在需要把“同一套项目事实”做成薄适配、可验证、可复盘的 agent context 层，降低不同 vibe-coding 工具之间的行为漂移。

## What Changes

- 新增 agent context portability 能力，定义工具中立的 instruction adapter 策略：`AGENTS.md` 和当前导航文档继续是源头，工具专属文件只做薄引用或极短补充。
- 新增当前研究简报契约，用一页短文档记录当前主线、冻结方法、不要追的路线、claim 升级条件和下一步高价值实验。
- 新增 agent 复盘/记忆候选契约，把重复 AI 错误、人工纠正和应沉淀位置记录为可审查清单，而不是让 agent 自动改写长期文档。
- 新增只读角色 agent / skill 契约，例如 claim auditor、experiment triage、surface doctor reviewer、literature scout；它们默认只读，不绕过 OpenSpec、`kd_mm_beam` 和本地产物边界。
- 新增 portability validation 要求，检查适配文件、研究简报、角色说明和记忆候选不会复制过期规则、不会引入退役入口、不会要求提交本地产物。

## Capabilities

### New Capabilities
- `agent-context-portability`: 定义跨 Codex/Claude/Cursor/Copilot/Kiro/Replit 等 agent 工具共享项目上下文、研究简报、记忆候选和只读角色的契约。

### Modified Capabilities

无。现有 `ai-maintainer-navigation`、`openspec-document-health` 和 `project-health-guardrails` 可在实现阶段引用该新能力，但本 change 先用独立能力承载新行为，避免扩大既有导航规格。

## Impact

- 可能新增或修改工具适配文档，例如 `CLAUDE.md`、`.github/copilot-instructions.md`、`.cursor/rules/*.mdc`、`.kiro/steering/*.md`、`docs/current_research_brief.md` 或等价路径。
- 可能新增 agent memory/retrospective ledger 文档或诊断检查。
- 可能新增只读 agent/skill 描述和 focused validation，仍不得自动修改 README、OpenSpec、claim registry 或正式文档。
- 不改变训练、评估、预处理、配置解析、模型 forward、dataset split、checkpoint schema 或运行产物边界。
