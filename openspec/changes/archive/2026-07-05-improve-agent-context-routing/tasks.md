## 1. Context 路由设计

- [x] 1.1 设计 agent context 布局，例如 `docs/agent_context/` 或等价位置。
- [x] 1.2 将模型、数据、配置、CLI、诊断、OpenSpec、文档、claim 更新拆成 scoped context 初版。
- [x] 1.3 更新 `AGENTS.md` 和 `docs/agent_navigation.md`，让根规则保持短入口并指向 scoped context。

## 2. Atlas / 索引

- [x] 2.1 增加 spec/config/claim atlas 的字段定义和生成或维护方式。
- [x] 2.2 确保 atlas 只引用权威路径、lifecycle、owner、focused tests 和 caveat，不复制完整 requirements。
- [x] 2.3 更新 `docs/maintainer_context_index.yaml` 或相关生成源，支持任务路由查询。

## 3. 项目技能

- [x] 3.1 增加或规划 `kd-add-model`、`kd-add-config`、`kd-update-claim`、`kd-diagnose-run`、`kd-archive-change` 等项目技能。
- [x] 3.2 每个技能说明必须包含 OpenSpec 边界、`kd_mm_beam` 命令环境和本地产物边界。
- [x] 3.3 更新 inventory 文档生命周期，登记新增 context/skill 文件。

## 4. 验证

- [x] 4.1 运行 `openspec validate improve-agent-context-routing --strict`。
- [x] 4.2 运行 `openspec validate --all --strict`。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
