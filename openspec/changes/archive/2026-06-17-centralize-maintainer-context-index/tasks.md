## 1. 索引结构与初始数据

- [x] 1.1 新增 `docs/maintainer_context_index.yaml`，声明用途、权威边界、schema version 和无运行时副作用约束。
- [x] 1.2 将任务路由表写入索引，覆盖模型/forward、数据与 batch、配置/virtual config、CLI/scripts、诊断/viewer、输出产物/cache、OpenSpec artifact 和文档生命周期改动。
- [x] 1.3 将首批治理表写入索引，包括 Python 脚本入口、shell orchestration、root fusion config、模型注册、batch/runtime 分支、热点 symbol/file budget、健康检查命令和退役路线 token。
- [x] 1.4 确认索引只引用源码、配置、文档、OpenSpec artifact、pyproject 和测试文件，不引用真实 `dataset/`、`outputs/`、`logs/`、cache 或 checkpoint。

## 2. 健康护栏接入

- [x] 2.1 新增测试侧轻量 YAML reader/schema validator，校验必填 section、唯一性、合法 lifecycle、路径存在性和 `kd_mm_beam` 命令约束。
- [x] 2.2 更新 `tests/test_architecture_boundaries.py`，让脚本入口 allowlist 从维护上下文索引读取。
- [x] 2.3 更新 `tests/test_architecture_boundaries.py`，让 root fusion config、模型注册 allowlist、batch/runtime allowlist 和 hotspot budget 从维护上下文索引读取。
- [x] 2.4 保持现有架构边界断言语义不放宽，确保未登记入口、未说明整模型例外、retired route 回流和热点扩张仍会失败。

## 3. 文档与导航同步

- [x] 3.1 更新 `docs/agent_navigation.md`，将维护上下文索引加入非平凡改动前的检查顺序和任务路由说明。
- [x] 3.2 更新 `docs/project_surface_inventory.md`，说明机器可读治理表的来源迁移到维护上下文索引，inventory 保留审计解释和 caveat。
- [x] 3.3 更新 `AGENTS.md` 或相关导航段落，确保 agent 操作规则能稳定指向维护上下文索引或通过 `docs/agent_navigation.md` 指向它。
- [x] 3.4 检查 README 不被扩写成治理数据库，只在必要位置保留简短文档索引或不变。

## 4. 验证

- [x] 4.1 运行 `openspec validate centralize-maintainer-context-index --strict`。
- [x] 4.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 4.3 如修改了 CLI/help 或配置加载相关断言，追加运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`。
- [x] 4.4 确认 `git status --short` 中没有新增 `dataset/` 内容、`outputs/`、`logs/`、cache、checkpoint、`.egg-info` 或其它本地产物。
