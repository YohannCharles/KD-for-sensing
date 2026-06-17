## Context

维护上下文索引已经解决了治理事实分散的问题，但测试层的读取和校验逻辑仍写在架构边界测试主体中。这个文件还包含大量 retired wording、config reference、hotspot budget、facade、轻量导入等断言。继续把 YAML schema 逻辑放在同一文件，会让测试自身成为新的维护热点。

## Goals / Non-Goals

**Goals:**

- 新增测试私有 helper 负责读取、验证和投影 maintainer context index。
- 架构边界测试只表达断言意图，避免顶部大段 schema plumbing。
- 增加 pyproject `[project.scripts]` 与 index `package_cli` 的双向一致性。
- 保持测试无运行副作用，不读取真实数据或 outputs。

**Non-Goals:**

- 不把 helper 放到 `src/kd_sensing`。
- 不改变 index YAML 字段含义。
- 不重写全部架构边界测试。

## Decisions

### Decision 1: helper 放在 tests 私有命名空间

推荐路径为 `tests/helpers/maintainer_context.py`。它可以被 `tests/test_architecture_boundaries.py` 导入，但不得被 runtime 模块引用。这样避免把治理索引变成运行时 API。

### Decision 2: helper 输出 typed-ish projection

helper 返回普通 dict/set/tuple，不引入 pydantic 或外部 schema 依赖。可以提供函数：

- `load_maintainer_context_index(root)`
- `entrypoint_allowlists(index)`
- `hotspot_budgets(index)`
- `assert_pyproject_scripts_match_index(root, index)`

### Decision 3: 增加双向 pyproject 检查

当前校验主要确认 index 中声明的 console script 存在于 pyproject。下一步必须反向确认 pyproject 中每个 script 都在 index 中登记，防止新增 CLI 忘记治理分类。

## Risks / Trade-offs

- [Risk] helper 抽出后失败信息变远。  
  → Mitigation: helper error message 继续带 `docs/maintainer_context_index.yaml` 和具体字段路径。

- [Risk] 测试 helper 变成通用库。  
  → Mitigation: 放在 `tests/helpers`，不导出到 package，不被 README 推荐。

- [Risk] 双向检查发现现有未登记脚本。  
  → Mitigation: 将缺失项登记到 index 或明确删除/退役，不放宽检查。
