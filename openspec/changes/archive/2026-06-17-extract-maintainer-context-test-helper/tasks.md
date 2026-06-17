## 1. Helper 抽取

- [x] 1.1 新增 `tests/helpers/maintainer_context.py` 或等价测试私有 helper。
- [x] 1.2 将 YAML 读取、schema validation、路径检查、命令检查和投影函数迁入 helper。
- [x] 1.3 更新 `tests/test_architecture_boundaries.py` 使用 helper 返回的 allowlist、budget 和 token。

## 2. 双向一致性

- [x] 2.1 实现 pyproject `[project.scripts]` 解析或轻量读取逻辑。
- [x] 2.2 校验 index `package_cli` 中每个 entry 存在于 pyproject。
- [x] 2.3 校验 pyproject 中每个 package script 存在于 index。

## 3. 防运行时污染

- [x] 3.1 确认 `src/kd_sensing` 不导入测试 helper。
- [x] 3.2 确认 helper 不读取真实 `dataset/`、`outputs/`、`logs/`、checkpoint 或 cache。
- [x] 3.3 保持失败信息包含 index 字段路径和修复提示。

## 4. 验证

- [x] 4.1 运行 `openspec validate extract-maintainer-context-test-helper --strict`。
- [x] 4.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
