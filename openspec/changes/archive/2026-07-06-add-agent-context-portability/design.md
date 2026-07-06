## Context

当前仓库的 agent 规则很强，但规则分层主要面向 Codex：`AGENTS.md` 是硬入口，`docs/agent_navigation.md` 是薄导航，`docs/agent_context/` 和 `.codex/skills/` 做按需加载。热门 vibe-coding 工具普遍也有类似概念，但文件名和加载时机不同：Claude 偏 `CLAUDE.md` / skills / subagents，Cursor 偏 rules，GitHub Copilot 偏 `.github/copilot-instructions.md` 和 `AGENTS.md`，Kiro 偏 steering/spec/hooks，Replit/Lovable 偏 project knowledge。若每个工具各写一份完整规则，漂移风险会迅速上升。

## Goals / Non-Goals

**Goals:**

- 让多种 agent 工具都能读到同一套项目边界。
- 保持 `AGENTS.md`、OpenSpec 和 scoped context 的权威地位。
- 用短研究简报减少 agent 对主线、claim gate 和退役路线的误读。
- 为重复 AI 失误提供人工审核后的沉淀路径。
- 允许只读专业角色协助分析，但不扩大写权限。

**Non-Goals:**

- 不把所有工具专属配置都做成必需项。
- 不维护完整源码目录镜像或长篇 prompt 大全。
- 不让 hooks 或 agent 自动重写 README、OpenSpec、claim registry。
- 不引入外部 SaaS 依赖作为项目工作流前提。

## Design

### 1. Thin adapters

每个工具适配文件只允许包含三类内容：

- 指向 `AGENTS.md`、`docs/agent_navigation.md`、`docs/agent_context/README.md` 的引用。
- 该工具必须知道的加载差异或命令差异。
- 极少量工具专属禁止事项，例如不要把生成产物纳入提交。

适配文件不得复制完整任务路由表、完整 OpenSpec requirement、完整 README quickstart 或完整 retired token 清单。长内容继续留在已有权威文档中。

### 2. Current research brief

新增一份短研究简报，建议职责如下：

- 当前论文/实验主线。
- 当前冻结方法和主要对照。
- 不再追的路线和退役入口。
- claim 升级条件。
- 下一步最值得跑的实验或诊断。
- 最新收口状态和仍未完成的证据缺口。

该简报是 agent 快速理解研究方向的入口，不替代 result claim registry、experiment protocols、mainline catalog 或 OpenSpec。

### 3. Memory and retrospective ledger

不启用“agent 自动改长期文档”的默认行为。重复出现的 AI 错误先进入可审查 ledger，字段至少包括：

- 错误模式。
- 触发场景。
- 正确规则。
- 建议沉淀位置。
- 需要的验证命令。
- 是否已人工确认。

只有人工确认或后续 OpenSpec change 才能把 ledger 条目转为 `AGENTS.md`、navigation、context、skill 或 tests 的变更。

### 4. Read-only role agents

角色 agent / skills 默认只读，用于把高噪声调查从主上下文隔离出去：

- `claim-auditor`: 审核 claim provenance、状态和升级条件。
- `experiment-triage`: 汇总 run state、缺失 seed、fresh eval 和 budget 风险。
- `surface-doctor-reviewer`: 阅读 doctor 输出并提出 inventory/guardrail 收口建议。
- `literature-scout`: 整理外部资料与 `docs/literature_matrix.md` 的差距。

这些角色只能返回建议。真正修改源码、OpenSpec 或正式 claim 文档仍由主 agent 在当前 change 范围内执行。

### 5. Validation

验证重点是防漂移而非跑训练：

- 适配文件存在时必须引用当前权威入口。
- 适配文件不得推荐退役 CLI/config/registry 名称。
- 研究简报不得写入未经 review 的正式数值 claim。
- 角色 agent/skill 不得声明可绕过 OpenSpec 或写入 ignored 运行产物。
- Python 验证命令仍使用 `conda run -n kd_mm_beam ...`。

## Risks / Trade-offs

- [Risk] 适配文件太多导致维护成本上升。  
  Mitigation: 只保留薄引用和短补充；完整内容仍集中在已有文档。
- [Risk] 研究简报变成第二个 claim registry。  
  Mitigation: 简报只写状态和指针，不写完整 provenance 表。
- [Risk] 角色 agent 被误用为自动实现者。  
  Mitigation: 角色契约明确只读，写操作仍走主 agent 和 OpenSpec change。

## Migration Plan

1. 创建最小工具适配文件，先覆盖最常用 agent surface。
2. 新增当前研究简报并在 navigation/inventory 中登记职责。
3. 新增记忆候选 ledger 或等价文档。
4. 定义只读角色说明，必要时转成 skills 或工具专属 agent 文件。
5. 增加文档/架构 focused checks，验证引用和退役边界。

## Open Questions

- 是否需要把 Cursor/Kiro/Replit 适配文件纳入仓库，还是先只提供模板文档。
- 研究简报应该放在 `docs/current_research_brief.md` 还是合并到现有 `docs/mainline_experiment_history.md` 的短节。
