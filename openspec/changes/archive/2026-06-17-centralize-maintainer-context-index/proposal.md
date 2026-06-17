## Why

当前项目已经用 README、OpenSpec specs、`docs/project_surface_inventory.md`、`docs/agent_navigation.md` 和架构边界测试共同约束入口、生命周期、退役路线与验证命令，但这些事实分散在多处，AI agent 和维护者在新增模型、CLI、配置或实验 workflow 时需要反复拼接上下文。这个 change 通过引入中心化、机器可读的 maintainer context index，降低误读概率，并减少文档表格与测试 allowlist 的重复维护。

## What Changes

- 新增一份中心化 maintainer context index，记录当前任务路由、capability lifecycle、entrypoint/config/model governance、热点预算、推荐验证命令和退役路线边界。
- 更新 AI 维护导航，使其把中心化 index 作为非平凡改动前的结构化入口，同时继续保留 README、AGENTS、OpenSpec specs 和 inventory 的权威边界。
- 更新项目健康护栏，使架构边界测试读取中心化 index 中的 governance 表，而不是在测试文件中维护第二套长期 allowlist。
- 保留 `docs/project_surface_inventory.md` 的审计和解释职责，但把可被测试消费的重复清单迁移到中心化 index。
- 不改变训练、评估、预处理、模型 forward、数据 split、配置解析、checkpoint schema 或本地产物清理语义。

## Capabilities

### New Capabilities

- `maintainer-context-index`: 定义中心化、机器可读的维护上下文索引，覆盖 AI/maintainer 路由、治理表、生命周期入口和无运行副作用边界。

### Modified Capabilities

- `ai-maintainer-navigation`: 导航文档需要指向并使用 maintainer context index，避免把长期清单继续散落在 Markdown 叙述和测试常量中。
- `project-health-guardrails`: 健康护栏需要验证 maintainer context index 存在、schema 合法、与 inventory/AGENTS/OpenSpec lifecycle 对齐，并从该 index 读取入口和治理 allowlist。

## Impact

- 主要影响文档、OpenSpec artifact、测试治理数据和架构边界测试。
- 可能新增 `docs/maintainer_context_index.yaml` 或等价稳定路径，以及一个轻量 schema/reader helper 供测试读取。
- `tests/test_architecture_boundaries.py` 中的入口、root fusion config、模型注册、batch/runtime 分支、热点预算等治理常量会逐步迁移为读取中心化 index。
- README 和 `docs/project_surface_inventory.md` 只保留面向人的摘要、解释和审计上下文，不再作为所有机器可读 allowlist 的唯一来源。
- 不新增公开训练/评估 CLI，不恢复 retired 入口，不读取真实 `dataset/`，不写入 `outputs/`、`logs/`、cache 或 checkpoint。
