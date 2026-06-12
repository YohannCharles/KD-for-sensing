## 1. 导航文档

- [x] 1.1 新增 `docs/agent_navigation.md`，包含当前状态检查顺序、权威来源优先级、任务路由表、常见误读清单、修改前检查清单和验证命令选择表。
- [x] 1.2 在导航文档中明确 generated metadata、ignored runtime artifacts、本地数据、OpenSpec archive、retired research lines、virtual configs 和 active change 状态的阅读边界。
- [x] 1.3 在导航文档中覆盖模型/forward、数据与 batch contract、配置和 virtual config、CLI/脚本入口、输出产物/cache、诊断/viewer、OpenSpec artifact 和文档生命周期改动的任务路由。

## 2. 文档入口与生命周期

- [x] 2.1 更新 `AGENTS.md`，用简短指针要求非平凡改动前阅读 `docs/agent_navigation.md`，并保持 AGENTS 不维护完整目录清单。
- [x] 2.2 更新 `docs/project_surface_inventory.md`，将 `docs/agent_navigation.md` 分类为当前 agent/maintainer navigation，并说明它不替代 README、AGENTS 或 OpenSpec specs。
- [x] 2.3 视需要在 README 项目文档索引中补充导航文档链接；若不补充，确认 AGENTS 和 inventory 已提供足够入口。

## 3. 健康检查

- [x] 3.1 扩展 `tests/test_architecture_boundaries.py`，验证 `docs/agent_navigation.md` 存在并包含权威来源、任务路由、generated metadata、ignored runtime artifacts、virtual config、retired research line 和 `kd_mm_beam` 等关键标记。
- [x] 3.2 扩展同一测试，验证 `AGENTS.md` 指向 `docs/agent_navigation.md`，且 `docs/project_surface_inventory.md` 对该文档有生命周期分类。
- [x] 3.3 确认新增测试不读取真实 `dataset/`、不加载 checkpoint、不启动训练、不写入 `outputs/` 或 `logs/`。

## 4. 验证与收尾

- [x] 4.1 运行 `openspec validate clarify-ai-maintainer-navigation --strict` 并修复所有 OpenSpec 问题。
- [x] 4.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，确认导航文档护栏通过。
- [x] 4.3 运行 `git status --short`，确认实现未纳入 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`.egg-info` 或其它本地产物变更。
