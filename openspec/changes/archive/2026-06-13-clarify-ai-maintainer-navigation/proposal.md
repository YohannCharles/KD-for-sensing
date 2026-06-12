## Why

当前仓库已经有 README、AGENTS、OpenSpec specs、项目表面积 inventory 和复现实验说明，但这些信息分散在多个层级。AI 或协作者在修改代码时容易抓错权威来源，例如把 `src/kd_sensing.egg-info/` 当作源码清单、把 `outputs/`/`logs/` 当作维护对象、把 archive/历史路线当作当前入口，或忽略 active change 与当前工作树状态。

本 change 目标是增加一层薄的 AI/维护者导航契约，让修改前的阅读顺序、任务路由、常见误读和验证选择更明确，从而降低后续代码变更误解项目结构和说明文档的风险。

## What Changes

- 新增一份面向 AI agent 和维护者的导航文档，暂定路径为 `docs/agent_navigation.md`。
- 在 `AGENTS.md` 中增加最小入口提示，要求非平凡改动前先阅读该导航文档，但不把 AGENTS 扩写成完整结构清单。
- 导航文档明确权威来源优先级：用户请求、AGENTS、active OpenSpec change、当前 specs、README/docs、源码测试、历史 archive/本地产物。
- 导航文档提供任务路由表，说明修改模型、数据/batch contract、配置、CLI、输出产物、诊断/viewer、OpenSpec artifact 时应先读什么、主要改哪里、跑哪些检查。
- 导航文档列出常见误读清单，包括 generated metadata、ignored runtime artifacts、virtual configs、retired research lines、active change 状态和 OpenSpec archive 边界。
- 更新项目表面积 inventory，将新导航文档纳入文档生命周期分类，并说明其与 README、AGENTS、OpenSpec specs 的边界。
- 增加或扩展架构边界测试，确保导航文档存在并覆盖关键标记，防止后续文档漂移。
- 不改变训练、评估、预处理、模型 forward、数据 split、配置解析运行语义或本地产物清理行为。

## Capabilities

### New Capabilities

- `ai-maintainer-navigation`: 定义面向 AI agent 和维护者的修改前导航、权威来源优先级、任务路由、误读防护和验证选择契约。

### Modified Capabilities

无。

## Impact

- 文档：新增 `docs/agent_navigation.md`，小幅更新 `AGENTS.md` 和 `docs/project_surface_inventory.md`。
- 测试：扩展 `tests/test_architecture_boundaries.py`，检查导航文档与 inventory/AGENTS 的关键约束一致。
- OpenSpec：新增 `ai-maintainer-navigation` capability spec，并保持当前 `project-architecture`、`project-health-guardrails` 和 `project-surface-cleanup` 语义不变。
- 运行行为：无 runtime 行为变更；不新增长期 CLI，不修改训练默认输出，不删除或移动 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`.egg-info` 或其它 ignored 本地产物。
