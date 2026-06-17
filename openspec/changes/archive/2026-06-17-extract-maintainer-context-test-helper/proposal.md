## Why

`tests/test_architecture_boundaries.py` 已经从 `docs/maintainer_context_index.yaml` 读取治理事实，但 YAML reader 和 schema validator 仍集中在该测试文件顶部。随着索引字段增加，架构测试会继续变胖，Codex 在修改具体架构断言时也更容易误动索引解析逻辑。

## What Changes

- 将 maintainer context index 的 YAML 读取、schema validation 和常用投影函数迁移到测试私有 helper。
- 架构边界测试继续使用 helper 提供的数据，不维护重复治理常量。
- 增加 pyproject console scripts 与 index `package_cli` 的双向一致性检查。
- helper 只供测试使用，不成为 runtime API，不被 `src/kd_sensing` 导入。
- 不改变 maintainer context index schema 语义，只整理测试结构和一致性检查。

## Capabilities

### New Capabilities

### Modified Capabilities

- `project-health-guardrails`: 测试基础设施需要支持维护上下文索引私有 helper 和双向一致性检查。
- `maintainer-context-index`: 索引验证需要明确 package CLI 与 pyproject 的双向同步要求。

## Impact

- 主要影响 `tests/test_architecture_boundaries.py` 和新增 `tests/helpers/maintainer_context.py` 或等价测试私有模块。
- 可能影响 `tests/conftest.py` 或测试 helper import 路径，但不得引入 runtime eager import。
- 验证重点是 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
