## Why

维护上下文索引已经集中保存 hotspot budgets，但当前预算主要是 `path/symbol/max_lines`，只能告诉 Codex “哪里超长”，不能告诉它“为什么暂缓、下一步该拆哪里、优先级是什么”。这会让后续 agent 面对热点时仍要回到 inventory 长文档里手动找拆分方向。

## What Changes

- 扩展 `docs/maintainer_context_index.yaml` 的 hotspot metadata，为每个 file/symbol budget 增加优先级、状态、推荐拆分目标、下一步 change、暂缓原因摘要和验证命令。
- 保持 `docs/project_surface_inventory.md` 作为解释性审计文本；index 保存可机器读取的下一步行动线索。
- 更新 schema validator 和架构边界测试，检查 hotspot metadata 必填字段、合法优先级、合法状态和路径存在性。
- 不改变 runtime 行为，不拆任何热点代码。

## Capabilities

### New Capabilities

### Modified Capabilities

- `maintainer-context-index`: hotspot budget 需要支持优先级、状态、拆分目标和下一步行动 metadata。
- `project-health-guardrails`: 健康护栏需要验证增强后的 hotspot metadata 并继续执行预算检查。

## Impact

- 影响 `docs/maintainer_context_index.yaml`、`tests/test_architecture_boundaries.py` 或测试 helper、`docs/project_surface_inventory.md`。
- 为后续 `split-jepa-shortcut-benchmark-runner` 和 `extract-deepsense6g-dataset-contract-helpers` 提供结构化路由。
- 不影响训练、评估、模型 forward、dataset 读取或输出产物。
