# 文档任务上下文

用于 README、AGENTS、agent navigation、inventory、maintainer context index、主线实验文档、文档生命周期和文档健康检查。

## 先读

- `AGENTS.md`
- README 的文档索引和数据/产物边界
- `openspec/specs/ai-maintainer-navigation/spec.md`
- `openspec/specs/maintainer-context-index/spec.md`
- `openspec/specs/openspec-document-health/spec.md`
- `docs/project_surface_inventory.md` 的文档生命周期分类

## 分工

| 文件 | 职责 |
| --- | --- |
| `AGENTS.md` | 操作规则、命令环境、本地产物边界和短入口 |
| `docs/agent_navigation.md` | 修改前导航、权威来源、任务路由、误读边界和验证选择 |
| `docs/maintainer_context_index.yaml` | 最小结构化事实和任务路由字段 |
| `docs/project_surface_inventory.md` | lifecycle、入口分类、热点 rationale、文档生命周期和历史 caveat |
| `docs/agent_context/` | 可按任务加载的 scoped context 和 atlas |

## 边界

- README 保留安装、quickstart、主要入口、数据/产物边界和文档索引。
- OpenSpec 记录需求、架构约束、设计决策和变更历史。
- Inventory 可以解释 lifecycle 和 caveat；maintainer context index 不复制完整 inventory。
- 新增 context 或 skill 时，应在 AGENTS、agent navigation、inventory 或技能清单中可定位用途和适用范围。

## 验证

- `openspec validate --all --strict`
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
