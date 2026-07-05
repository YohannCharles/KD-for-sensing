## Context

当前 `AGENTS.md` 负责操作规则，`docs/agent_navigation.md` 负责导航，`docs/project_surface_inventory.md` 负责表面积审计，`docs/maintainer_context_index.yaml` 负责最小结构化事实。它们已经清晰分工，但对 agent 来说仍偏“大块读取”。随着 specs、configs、scripts 数量增长，需要把上下文组织成可按任务加载的结构。

## Goals / Non-Goals

**Goals:**

- 让 agent 能从一屏入口快速选择任务路由。
- 把长导航拆成稳定片段或 atlas，减少无关上下文。
- 为重复维护流程提供项目级技能或标准命令。
- 让 completed-but-unarchived change 有明确处理策略。

**Non-Goals:**

- 不删除现有 AGENTS 或 OpenSpec 工作流。
- 不把 machine-readable index 变成完整源码目录镜像。
- 不要求所有外部工具都采用同一规则文件格式。

## Decisions

1. 根 `AGENTS.md` 保持短规则，细节转向 scoped docs/skills。
   - 理由：根规则必须高信号，避免每次任务加载过多。
   - 备选：继续扩写 AGENTS；会加重上下文负担。

2. Atlas 由权威来源生成或人工维护最小字段。
   - 理由：103 个 specs 和大量 configs 需要可扫视入口。
   - 备选：让 agent 每次读 inventory 全文；成本高且容易漏。

3. 项目 skills 只覆盖高频流程，不替代 OpenSpec。
   - 理由：skills 适合流程化任务，需求契约仍属于 specs。
   - 备选：将所有流程写进 AGENTS；不利于渐进加载。

## Risks / Trade-offs

- [Risk] 新索引变成第二套事实源。→ Mitigation: 明确 atlas/index 只引用权威路径和 lifecycle，不复制完整 requirement。
- [Risk] 分层文件过多导致维护成本增加。→ Mitigation: 只对高频任务创建片段，新增片段需架构边界或文档健康检查登记。
- [Risk] 多工具规则文件互相冲突。→ Mitigation: 以 AGENTS/OpenSpec 为权威，其它规则只做派生提示。

## Migration Plan

- 先定义 agent context 文件布局和索引字段。
- 再生成或手工创建第一版 atlas。
- 最后将高频流程沉淀为项目 skills，并更新导航引用。

## Open Questions

- 是否需要为 Cursor/Cline/Claude 生成派生规则文件，还是只维护 Codex/AGENTS 原生上下文。
